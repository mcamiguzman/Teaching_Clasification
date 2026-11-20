"""
Validación de carga y predicciones en un entorno limpio (subproceso aislado).

Implementa explícitamente:
- [Parte 1.3 — Validar la carga y las predicciones]:
    * Lanza un subproceso Python completamente limpio (`subprocess.run`) sin estado
      en memoria del entrenamiento para verificar que el archivo `.joblib` es
      100% autocontenido y reproducible.
    * Comprueba coherencia exacta (100% de coincidencia en etiquetas y probabilidades)
      entre las predicciones recargadas y las predicciones de referencia en el conjunto
      de prueba (`test_reference.json`).
    * Evalúa de forma explícita casos borde y de calidad:
        - Comentario claramente positivo, negativo y neutral.
        - Comentarios ambiguos y con cláusulas contrastantes (`pero`, `sin embargo`, `aunque`).
        - Textos vacíos (`""`), solo espacios (`"   \t\n"`), `None` y tipos inválidos.
        - Solo emojis (`"😀😀😀"`) y solo puntuación (`"!!!???"`).
        - Palabras alargadas e intensificadores (`"suuuuper bueeeeno"`).
        - Comentario neutro descriptivo del parcial (`"El horario es en las mañanas y hay tres parciales."`).
        - Salida estructurada de los 9 aspectos [Parte 2.5].
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.aspects import predecir_nueve_aspectos
from src.config import MODELS_DIR, NINE_ASPECTS, PROJECT_ROOT
from src.model_registry import load_active_model
from src.preprocessing import detectar_contraste, es_solo_emojis_o_simbolos, normalizar_texto


def predecir_seguro(comentario: Any, pipeline: Any, incluir_aspectos: bool = True) -> Dict[str, Any]:
    """
    Función de inferencia controlada que valida entradas vacías/erróneas antes de
    llamar al Pipeline unificado (que además incluye `TextNormalizer` internamente).
    """
    if comentario is None or not isinstance(comentario, str):
        return {
            "valido": False,
            "error": "Entrada inválida: se esperaba una cadena de texto no nula.",
            "clase": None,
            "probabilidades": {},
            "advertencias": [],
        }

    texto_limpio = normalizar_texto(comentario)
    if not texto_limpio:
        return {
            "valido": False,
            "error": "Texto vacío o compuesto únicamente por espacios.",
            "clase": None,
            "probabilidades": {},
            "advertencias": [],
        }

    advertencias: List[str] = []
    if es_solo_emojis_o_simbolos(comentario):
        advertencias.append(
            "Entrada compuesta únicamente por emojis o símbolos sin texto alfanumérico; "
            "la predicción puede tener baja confiabilidad (hallazgo documentado en el informe)."
        )
    if detectar_contraste(texto_limpio):
        advertencias.append(
            "Comentario con cláusula contrastante (pero/sin embargo/aunque): revisar el desglose por aspectos."
        )

    # [Parte 2.1] Pasamos el comentario original directamente al Pipeline unificado
    proba = pipeline.predict_proba([comentario])[0]
    clases = [str(c) for c in pipeline.classes_]
    clase_pred = clases[int(proba.argmax())]
    prob_dict = {c: round(float(p), 4) for c, p in zip(clases, proba)}

    resultado: Dict[str, Any] = {
        "valido": True,
        "error": None,
        "texto_normalizado": texto_limpio,
        "clase": clase_pred,
        "confianza": round(float(proba.max()), 4),
        "probabilidades": prob_dict,
        "es_contrastante": detectar_contraste(texto_limpio),
        "advertencias": advertencias,
    }
    if incluir_aspectos:
        resultado["aspectos"] = predecir_nueve_aspectos(comentario, pipeline)
    return resultado


def run_validation_in_current_process(version: str | None = None) -> Dict[str, Any]:
    """Ejecuta toda la batería de validación de carga, coherencia y casos borde."""
    bundle = load_active_model(version=version)
    pipeline = bundle.pipeline

    # 1. Coherencia 100% contra las predicciones de referencia en test
    ref_path = MODELS_DIR / bundle.version / "test_reference.json"
    coherence_summary: Dict[str, Any] = {"reference_file_found": ref_path.exists()}
    if ref_path.exists():
        ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
        X_test = ref_data["X_test"]
        expected_preds = ref_data["y_pred_expected"]
        expected_probas = ref_data["y_proba_expected"]

        reloaded_preds = [str(x) for x in pipeline.predict(X_test).tolist()]
        reloaded_probas = pipeline.predict_proba(X_test).tolist()

        exact_label_match = reloaded_preds == expected_preds
        max_proba_diff = max(
            abs(float(a) - float(b))
            for row_a, row_b in zip(reloaded_probas, expected_probas)
            for a, b in zip(row_a, row_b)
        )
        assert exact_label_match, "FALLO DE COHERENCIA: las etiquetas recargadas difieren del entrenamiento."
        assert max_proba_diff < 1e-5, f"FALLO DE COHERENCIA: diferencia de probabilidad {max_proba_diff}"

        coherence_summary.update(
            {
                "n_test_checked": len(X_test),
                "exact_label_match": exact_label_match,
                "max_proba_abs_diff": round(max_proba_diff, 8),
            }
        )

    # 2. Batería de casos borde (inspirada en Celda 33 de Parcial_SI_corregido.ipynb)
    casos_prueba = [
        ("positivo_claro", "Excelente curso, el profesor explica muy bien y los ejemplos son claros."),
        ("negativo_claro", "Muy mal organizado, las evaluaciones no tienen sentido y nadie responde dudas."),
        ("ambiguo_contrastante", "El contenido es interesante, pero el ritmo es demasiado rápido."),
        ("descriptivo_neutro", "El horario es en las mañanas y hay tres parciales."),
        ("alargamientos_y_emoji", "Suuuuper bueeeeno el material de clase y el docente 😊"),
        ("texto_vacio", ""),
        ("solo_espacios", "    \t\n  "),
        ("solo_emojis", "😀😀😀"),
        ("solo_simbolos", "!!! ??? ---"),
        ("valor_nulo", None),
        ("tipo_invalido", 12345),
    ]

    resultados_casos = []
    for nombre, entrada in casos_prueba:
        res = predecir_seguro(entrada, pipeline, incluir_aspectos=True)
        if res["valido"]:
            assert len(res["aspectos"]) == len(NINE_ASPECTS), "Debe retornar exactamente 9 aspectos."
            suma_probs = sum(res["probabilidades"].values())
            assert abs(suma_probs - 1.0) < 0.01, f"Las probabilidades deben sumar ~1 (dio {suma_probs})."
        resultados_casos.append(
            {
                "caso": nombre,
                "entrada_preview": repr(entrada)[:70],
                "valido": res["valido"],
                "clase": res["clase"],
                "probabilidades": res["probabilidades"],
                "error": res["error"],
                "advertencias": res["advertencias"],
                "aspectos_mencionados": [
                    k for k, v in res.get("aspectos", {}).items() if v["mencionado"]
                ],
            }
        )

    return {
        "status": "OK",
        "model_version": bundle.version,
        "sha256_verified": bundle.sha256_verified,
        "pipeline_steps": list(pipeline.named_steps.keys()),
        "version_warnings": bundle.version_warnings,
        "coherence": coherence_summary,
        "edge_cases": resultados_casos,
    }


def validate_in_clean_subprocess(version: str | None = None) -> Dict[str, Any]:
    """
    [Parte 1.3] Ejecuta la validación en un proceso Python totalmente aislado
    (entorno limpio sin variables de entrenamiento en memoria).
    """
    code = (
        "import json; "
        "from src.validate_model import run_validation_in_current_process; "
        f"res = run_validation_in_current_process(version={version!r}); "
        "print(json.dumps(res, ensure_ascii=False))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valida la carga segura del modelo .joblib y sus predicciones en entorno limpio."
    )
    parser.add_argument("--version", default=None, help="Versión específica a validar (por defecto la activa)")
    args = parser.parse_args()

    report = validate_in_clean_subprocess(version=args.version)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
