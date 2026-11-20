"""
Pruebas de control, actualización y reversión (rollback) de versiones [Parte 2.6].
"""

from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.model_registry import (
    get_runtime_versions,
    load_active_model,
    load_registry,
    register_model_version,
    rollback_model_version,
    switch_active_version,
)
from src.preprocessing import TextNormalizer


def _dummy_pipeline(c_val: float) -> Pipeline:
    X = [
        "Excelente curso y profesor",
        "Muy buen contenido y material",
        "Pésimo curso y parciales malos",
        "Mala metodología y desorganizado",
        "Curso regular normal",
        "Horario estándar promedio",
    ]
    y = ["positivo", "positivo", "negativo", "negativo", "neutral", "neutral"]
    pipe = Pipeline(
        [
            ("normalizer", TextNormalizer()),
            ("vec", TfidfVectorizer()),
            ("clf", LogisticRegression(C=c_val, max_iter=500)),
        ]
    )
    pipe.fit(X, y)
    return pipe


def test_registro_actualizacion_y_rollback_completo(tmp_path: Path, monkeypatch):
    """Verifica el ciclo completo: v1.0.0 -> v1.1.0 -> rollback a v1.0.0 -> switch a v1.1.0."""
    import src.model_registry as mr

    # Redirigir PROJECT_ROOT en el módulo para que las rutas relativas resuelvan dentro de tmp_path
    monkeypatch.setattr(mr, "PROJECT_ROOT", tmp_path)
    models_dir = tmp_path / "models"
    registry_path = models_dir / "registry.json"

    base_meta = {
        "modelo": "LogisticRegression",
        "metricas": {"macro_f1_cv_train": 0.70, "macro_f1_test": 0.72, "accuracy_test": 0.74},
        "dataset": {"dataset_version": "1.0.0"},
        "versiones": get_runtime_versions(),
    }

    register_model_version(
        version="v1.0.0",
        pipeline=_dummy_pipeline(1.0),
        metadata=base_meta,
        activate=True,
        notes="Versión inicial estable",
        models_dir=models_dir,
        registry_path=registry_path,
    )
    reg = load_registry(registry_path)
    assert reg["active_version"] == "v1.0.0"

    register_model_version(
        version="v1.1.0",
        pipeline=_dummy_pipeline(2.0),
        metadata={**base_meta, "dataset": {"dataset_version": "1.1.0"}},
        activate=True,
        notes="Nueva versión candidata",
        models_dir=models_dir,
        registry_path=registry_path,
    )
    reg = load_registry(registry_path)
    assert reg["active_version"] == "v1.1.0"
    assert reg["previous_active_version"] == "v1.0.0"

    # Rollback automático a la versión anterior estable (v1.0.0)
    rb = rollback_model_version(
        reason="Regresión en producción",
        registry_path=registry_path,
        models_dir=models_dir,
    )
    assert rb["rolled_back_from"] == "v1.1.0"
    assert rb["active_version"] == "v1.0.0"

    bundle = load_active_model(registry_path=registry_path, models_dir=models_dir)
    assert bundle.version == "v1.0.0"
    assert bundle.sha256_verified is True

    # Promoción controlada de vuelta a v1.1.0
    sw = switch_active_version(
        "v1.1.0",
        reason="Validación corregida",
        registry_path=registry_path,
        models_dir=models_dir,
    )
    assert sw["active_version"] == "v1.1.0"
    assert len(load_registry(registry_path)["history"]) == 4
