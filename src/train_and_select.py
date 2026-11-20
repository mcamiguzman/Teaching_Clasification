"""
Flujo offline de selección de modelo, evaluación en test y serialización con Joblib.

Implementa explícitamente:
- [Parte 1.1 — Seleccionar el mejor modelo]:
    * División estratificada 80/20 (`random_state=42`) con verificación de ausencia
      de fuga de información (duplicados y similitud coseno TF-IDF test → train).
    * Evaluación de línea base con validación cruzada estratificada de 5 pliegues
      sobre 5 clasificadores × 2 vectorizadores TF-IDF (palabras 1–2 y caracteres 2–5).
    * Ajuste de hiperparámetros con `GridSearchCV` (102 configuraciones idénticas al
      cuaderno `Parcial_SI_corregido.ipynb`) usando **Macro-F1** como criterio principal.
    * Evaluación única sobre el conjunto de prueba reservado (20%) comparando frente
      a `DummyClassifier`, incluyendo reporte por clase, matriz de confusión y desglose
      por cláusulas contrastantes.
    * Registro completo de métricas, hiperparámetros y versión del dataset.
- [Parte 1.2 — Guardar el modelo con Joblib] y [Parte 2.1 — Pipeline unificado]:
    * Empaqueta `TextNormalizer()` + `TfidfVectorizer(...)` + `CalibratedClassifierCV(clf)`
      en un ÚNICO `sklearn.pipeline.Pipeline` serializado en `.joblib`.
    * Registra versiones exactas de Python y dependencias en `metadata.json`.
    * Guarda también predicciones de referencia en test para validación de coherencia
      en entorno limpio [Parte 1.3] y estadísticas base para monitoreo de drift [Parte 1.6].
"""

from __future__ import annotations

import argparse
import json
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    CV_FOLDS,
    MODELS_DIR,
    PARCIAL_REFERENCE_METRICS,
    POLARITY_CLASSES,
    RANDOM_SEED,
    TEST_SIZE,
)
from src.dataset_builder import build_and_version_dataset
from src.model_registry import get_runtime_versions, register_model_version
from src.monitoring import save_training_baseline
from src.preprocessing import TextNormalizer

warnings.filterwarnings("ignore", category=FutureWarning)


def _compute_metrics_dict(y_true: pd.Series, y_hat: np.ndarray) -> Dict[str, float]:
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_hat, average="macro", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_hat)), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "macro_f1": round(float(macro_f1), 4),
    }


def _check_data_leakage(X_train_raw: pd.Series, X_test_raw: pd.Series) -> Dict[str, Any]:
    """
    Replica el Paso 3b de `Parcial_SI_corregido.ipynb`:
    Mide similitud coseno máxima TF-IDF entre cada comentario de test y su vecino
    más cercano en train (ajustando el vectorizador únicamente en train).
    """
    normalizer = TextNormalizer()
    X_train_clean = normalizer.transform(X_train_raw)
    X_test_clean = normalizer.transform(X_test_raw)

    overlap_exact = len(set(X_train_clean).intersection(set(X_test_clean)))
    vec = TfidfVectorizer(sublinear_tf=True).fit(X_train_clean)
    sim_max = cosine_similarity(vec.transform(X_test_clean), vec.transform(X_train_clean)).max(axis=1)

    return {
        "exact_overlap_train_test": int(overlap_exact),
        "mean_max_cosine_sim": round(float(sim_max.mean()), 4),
        "max_cosine_sim": round(float(sim_max.max()), 4),
        "high_similarity_gt_0_90": int((sim_max > 0.90).sum()),
    }


def _run_baseline_cv(
    X_train: pd.Series,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> List[Dict[str, Any]]:
    """
    Replica el Paso 5 de `Parcial_SI_corregido.ipynb`:
    5 modelos × 2 vectorizadores TF-IDF evaluados con StratifiedKFold(5).
    """
    vectorizers = {
        "TF-IDF palabras (1-2)": lambda: TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), min_df=2
        ),
        "TF-IDF caracteres (2-5)": lambda: TfidfVectorizer(
            sublinear_tf=True, analyzer="char_wb", ngram_range=(2, 5), min_df=2
        ),
    }
    models = {
        "Dummy (mayoritaria)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=RANDOM_SEED),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED
        ),
    }

    rows: List[Dict[str, Any]] = []
    for vec_name, vec_factory in vectorizers.items():
        for model_name, clf in models.items():
            # [Parte 2.1] El Pipeline incluye TextNormalizer desde la validación cruzada
            pipe = Pipeline(
                [
                    ("normalizer", TextNormalizer()),
                    ("vec", vec_factory()),
                    ("clf", clf),
                ]
            )
            scores = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv,
                scoring=["accuracy", "f1_macro"],
                n_jobs=-1,
            )
            rows.append(
                {
                    "vectorizador": vec_name,
                    "modelo": model_name,
                    "accuracy_cv": round(float(scores["test_accuracy"].mean()), 4),
                    "macro_f1_cv": round(float(scores["test_f1_macro"].mean()), 4),
                    "std_macro_f1": round(float(scores["test_f1_macro"].std()), 4),
                }
            )

    rows.sort(key=lambda item: item["macro_f1_cv"], reverse=True)
    return rows


def _build_grid_search(cv: StratifiedKFold, fast_mode: bool = False) -> GridSearchCV:
    """
    Construye el `GridSearchCV` del Paso 6 de `Parcial_SI_corregido.ipynb` con
    Pipeline unificado (`TextNormalizer` + `TfidfVectorizer` + clasificador).
    """
    base_pipe = Pipeline(
        [
            ("normalizer", TextNormalizer()),
            ("vec", TfidfVectorizer(sublinear_tf=True)),
            ("clf", LogisticRegression()),
        ]
    )

    vec_palabras = {
        "vec__analyzer": ["word"],
        "vec__ngram_range": [(1, 1), (1, 2)],
        "vec__min_df": [1, 2],
    }
    vec_chars = {
        "vec__analyzer": ["char_wb"],
        "vec__ngram_range": [(2, 5)] if fast_mode else [(2, 5), (3, 6)],
        "vec__min_df": [2],
    }

    param_grid: List[Dict[str, Any]] = []
    for vg in (vec_palabras, vec_chars):
        param_grid.extend(
            [
                {
                    **vg,
                    "clf": [LogisticRegression(max_iter=3000, class_weight="balanced")],
                    "clf__C": [0.1, 1, 10] if fast_mode else [0.1, 1, 10, 100],
                },
                {
                    **vg,
                    "clf": [LinearSVC(class_weight="balanced", random_state=RANDOM_SEED)],
                    "clf__C": [0.05, 0.1, 0.5, 1] if fast_mode else [0.05, 0.1, 0.5, 1, 5],
                },
                {
                    **vg,
                    "clf": [
                        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED)
                    ],
                    "clf__n_estimators": [200] if fast_mode else [200, 400],
                    "clf__max_depth": [None, 30],
                    "clf__min_samples_leaf": [1, 2],
                },
            ]
        )

    return GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )


def train_select_and_save(
    model_version: str = "v1.0.0",
    dataset_version: str = "1.0.0",
    force_synthetic: bool = False,
    fast_mode: bool = False,
    activate: bool = True,
    notes: str = "Entrenamiento inicial basado en Parcial_SI_corregido.ipynb",
    override_clf_family: str | None = None,
) -> Dict[str, Any]:
    """
    Ejecuta el flujo completo de Parte 1.1, Parte 1.2 y Parte 2.1:
      1. Construye/versiona el dataset limpio.
      2. Divide estratificadamente 80/20 y audita fuga de información.
      3. Ejecuta línea base CV y `GridSearchCV` seleccionando el mejor modelo.
      4. Evalúa una única vez en test contra `DummyClassifier`.
      5. Calibra probabilidades si el clasificador ganador no expone `predict_proba`
         (ej. `LinearSVC` -> `CalibratedClassifierCV(cv=5)`).
      6. Serializa el `Pipeline` unificado (`TextNormalizer` + `TfidfVectorizer` + `clf`)
         en `.joblib`, guarda metadatos, checksum SHA-256, registro de versiones,
         referencias de prueba [Parte 1.3] y línea base de drift [Parte 1.6].
    """
    print("=" * 72)
    print(f"[Parte 1.1 & 1.2] Entrenamiento y selección — Versión modelo: {model_version}")
    print("=" * 72)

    df, dataset_manifest = build_and_version_dataset(
        dataset_version=dataset_version,
        force_synthetic=force_synthetic,
    )
    print(
        f"Dataset cargado ({dataset_manifest['source_mode']}): "
        f"n={len(df)} | versión={dataset_version} | sha256={dataset_manifest['dataset_sha256'][:12]}..."
    )

    # Usamos el texto CRUDO (`feedback`) como entrada para que TextNormalizer()
    # actúe dentro del Pipeline unificado [Parte 2.1] tanto en train como en inferencia.
    X = df["feedback"]
    y = df["polaridad_final"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    leakage_report = _check_data_leakage(X_train, X_test)
    print(f"Auditoría data leakage: {json.dumps(leakage_report, ensure_ascii=False)}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # 1. Línea base CV (Paso 5)
    baseline_cv_results = _run_baseline_cv(X_train, y_train, cv)
    print("\nTop 3 línea base (CV 5-fold, Macro-F1):")
    for row in baseline_cv_results[:3]:
        print(f"  - {row['modelo']} + {row['vectorizador']}: Macro-F1={row['macro_f1_cv']:.4f}")

    # 2. Selección con GridSearchCV (Paso 6) o familia específica para pruebas de versionado
    if override_clf_family == "logistic_regression":
        best_pipe_uncalibrated = Pipeline(
            [
                ("normalizer", TextNormalizer()),
                (
                    "vec",
                    TfidfVectorizer(
                        sublinear_tf=True, analyzer="word", ngram_range=(1, 2), min_df=2
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"),
                ),
            ]
        )
        cv_scores = cross_validate(
            best_pipe_uncalibrated, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
        )
        best_cv_macro_f1 = float(cv_scores["test_score"].mean())
        best_pipe_uncalibrated.fit(X_train, y_train)
        best_params = {
            "clf": "LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000)",
            "vec__analyzer": "word",
            "vec__min_df": "2",
            "vec__ngram_range": "(1, 2)",
        }
        grid_summary = [
            {
                "modelo": "LogisticRegression",
                "vectorizador": "palabras",
                "mejor_macro_f1_cv": round(best_cv_macro_f1, 4),
                "n_configs": 1,
            }
        ]
    else:
        gs = _build_grid_search(cv=cv, fast_mode=fast_mode)
        gs.fit(X_train, y_train)
        best_pipe_uncalibrated = gs.best_estimator_
        best_cv_macro_f1 = float(gs.best_score_)
        best_params = {k: str(v) for k, v in gs.best_params_.items()}

        cvr = pd.DataFrame(gs.cv_results_)
        cvr["modelo"] = cvr["param_clf"].apply(lambda m: type(m).__name__)
        cvr["vectorizador"] = cvr["param_vec__analyzer"].map(
            {"word": "palabras", "char_wb": "caracteres"}
        )
        grouped = (
            cvr.groupby(["modelo", "vectorizador"])
            .agg(mejor_macro_f1_cv=("mean_test_score", "max"), n_configs=("mean_test_score", "size"))
            .sort_values("mejor_macro_f1_cv", ascending=False)
            .reset_index()
        )
        grid_summary = [
            {
                "modelo": str(r["modelo"]),
                "vectorizador": str(r["vectorizador"]),
                "mejor_macro_f1_cv": round(float(r["mejor_macro_f1_cv"]), 4),
                "n_configs": int(r["n_configs"]),
            }
            for _, r in grouped.iterrows()
        ]

    winner_clf_name = type(best_pipe_uncalibrated.named_steps["clf"]).__name__
    print(f"\nMejor modelo seleccionado en CV: {winner_clf_name} (Macro-F1 CV={best_cv_macro_f1:.4f})")

    # 3. Evaluación única sobre test (20%) vs DummyClassifier (Paso 7)
    dummy_pipe = Pipeline(
        [
            ("normalizer", TextNormalizer()),
            ("vec", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]
    ).fit(X_train, y_train)

    y_pred_uncalibrated = best_pipe_uncalibrated.predict(X_test)
    y_pred_dummy = dummy_pipe.predict(X_test)

    metrics_dummy_test = _compute_metrics_dict(y_test, y_pred_dummy)
    metrics_best_test = _compute_metrics_dict(y_test, y_pred_uncalibrated)

    # 4. Calibración de probabilidades para servicio API [Parte 1.2 & 1.4]
    # Si el ganador (ej. LinearSVC) no implementa `predict_proba`, lo envolvemos
    # con `CalibratedClassifierCV(cv=5)` manteniendo intacto el Pipeline unificado.
    modelo_final = clone(best_pipe_uncalibrated)
    calibrated = False
    if not hasattr(modelo_final.named_steps["clf"], "predict_proba"):
        modelo_final.set_params(
            clf=CalibratedClassifierCV(modelo_final.named_steps["clf"], cv=CV_FOLDS)
        )
        calibrated = True

    # Reentrenar únicamente sobre X_train (el conjunto test NUNCA participa en fit)
    modelo_final.fit(X_train, y_train)
    y_pred_final = modelo_final.predict(X_test)
    y_proba_final = modelo_final.predict_proba(X_test)
    metrics_calibrated_test = _compute_metrics_dict(y_test, y_pred_final)

    cm = confusion_matrix(y_test, y_pred_final, labels=POLARITY_CLASSES).tolist()
    per_class_report = classification_report(
        y_test, y_pred_final, labels=POLARITY_CLASSES, output_dict=True, zero_division=0
    )

    # Desglose de exactitud en comentarios contrastantes vs no contrastantes
    test_meta_df = df.loc[X_test.index, ["es_contrastante"]].copy()
    test_meta_df["correcto"] = (y_test.values == y_pred_final)
    contrastive_breakdown = {
        str(bool(k)): {
            "accuracy": round(float(g["correcto"].mean()), 4),
            "n": int(len(g)),
        }
        for k, g in test_meta_df.groupby("es_contrastante")
    }

    print(
        f"Evaluación en test (n={len(X_test)}): "
        f"Macro-F1={metrics_calibrated_test['macro_f1']:.4f} | "
        f"Accuracy={metrics_calibrated_test['accuracy']:.4f} "
        f"(Dummy Macro-F1={metrics_dummy_test['macro_f1']:.4f})"
    )

    # 5. Guardar metadatos completos + artefacto Joblib en registro de versiones
    metadata: Dict[str, Any] = {
        "modelo": winner_clf_name,
        "calibrado_con_CalibratedClassifierCV": calibrated,
        "pipeline_steps": list(modelo_final.named_steps.keys()),
        "parametros": best_params,
        "clases": [str(c) for c in modelo_final.classes_],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metricas": {
            "macro_f1_cv_train": round(best_cv_macro_f1, 4),
            "macro_f1_test_uncalibrated": metrics_best_test["macro_f1"],
            "accuracy_test_uncalibrated": metrics_best_test["accuracy"],
            "macro_f1_test": metrics_calibrated_test["macro_f1"],
            "accuracy_test": metrics_calibrated_test["accuracy"],
            "precision_macro_test": metrics_calibrated_test["precision_macro"],
            "recall_macro_test": metrics_calibrated_test["recall_macro"],
            "dummy_test": metrics_dummy_test,
            "confusion_matrix_labels": POLARITY_CLASSES,
            "confusion_matrix": cm,
            "per_class_report": per_class_report,
            "contrastive_breakdown": contrastive_breakdown,
        },
        "comparacion_linea_base_cv": baseline_cv_results,
        "resumen_gridsearch_cv": grid_summary,
        "auditoria_data_leakage": leakage_report,
        "dataset": dataset_manifest,
        "referencia_parcial_pdf": PARCIAL_REFERENCE_METRICS,
        # [Parte 1.2 & Parte 2.4] Versiones exactas de Python y dependencias
        "versiones": get_runtime_versions(),
    }

    registered_meta = register_model_version(
        version=model_version,
        pipeline=modelo_final,
        metadata=metadata,
        activate=activate,
        notes=notes,
    )

    # Guardar predicciones de referencia en test para validar 100% de coherencia
    # en entorno limpio [Parte 1.3]
    ref_path = MODELS_DIR / model_version / "test_reference.json"
    ref_payload = {
        "model_version": model_version,
        "n_test": int(len(X_test)),
        "X_test": X_test.tolist(),
        "y_test": y_test.tolist(),
        "y_pred_expected": [str(x) for x in y_pred_final.tolist()],
        "y_proba_expected": [[round(float(p), 6) for p in row] for row in y_proba_final.tolist()],
    }
    ref_path.write_text(json.dumps(ref_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Guardar línea base estadística de entrenamiento para monitoreo de drift [Parte 1.6]
    if activate:
        save_training_baseline(
            texts=X_train,
            labels=y_train,
            contrastive_flags=df.loc[X_train.index, "es_contrastante"],
            model_version=model_version,
            dataset_version=dataset_version,
        )

    print(f"\nModelo guardado y registrado en: models/{model_version}/modelo_polaridad.joblib")
    print(f"SHA-256: {registered_meta['artifact_sha256']}")
    return registered_meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena, selecciona, evalúa y serializa el Pipeline unificado en Joblib."
    )
    parser.add_argument("--version", default="v1.0.0", help="Versión semántica del modelo (ej. v1.0.0)")
    parser.add_argument("--dataset-version", default="1.0.0", help="Versión del dataset")
    parser.add_argument("--force-synthetic", action="store_true", help="Forzar réplica sintética del parcial")
    parser.add_argument("--fast", action="store_true", help="Usar grilla reducida para iteración rápida")
    parser.add_argument("--no-activate", action="store_true", help="Registrar sin activar inmediatamente")
    parser.add_argument("--notes", default="Entrenamiento reproducible basado en el trabajo parcial")
    parser.add_argument(
        "--clf-family",
        choices=["auto", "logistic_regression"],
        default="auto",
        help="Selección automática por GridSearchCV ('auto') o familia alternativa para versionado",
    )
    args = parser.parse_args()

    train_select_and_save(
        model_version=args.version,
        dataset_version=args.dataset_version,
        force_synthetic=args.force_synthetic,
        fast_mode=args.fast,
        activate=not args.no_activate,
        notes=args.notes,
        override_clf_family=None if args.clf_family == "auto" else args.clf_family,
    )


if __name__ == "__main__":
    main()
