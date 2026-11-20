"""
Pruebas de validación de carga en entorno limpio, seguridad de deserialización
y compatibilidad de versiones [Parte 1.3, Parte 2.3 y Parte 2.4].
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.model_registry import (
    UntrustedModelError,
    VersionCompatibilityError,
    check_version_compatibility,
    load_active_model,
    verify_sha256_integrity,
    verify_trusted_path,
)
from src.validate_model import predecir_seguro, validate_in_clean_subprocess


def test_validacion_en_subproceso_limpio_y_coherencia_100():
    """[Parte 1.3] Recarga en subproceso aislado y valida 100% de coherencia + casos borde."""
    report = validate_in_clean_subprocess()
    assert report["status"] == "OK"
    assert report["sha256_verified"] is True
    assert report["pipeline_steps"] == ["normalizer", "vec", "clf"]
    assert report["coherence"]["exact_label_match"] is True
    assert report["coherence"]["max_proba_abs_diff"] < 1e-5

    casos_por_nombre = {c["caso"]: c for c in report["edge_cases"]}
    assert casos_por_nombre["texto_vacio"]["valido"] is False
    assert casos_por_nombre["solo_espacios"]["valido"] is False
    assert casos_por_nombre["valor_nulo"]["valido"] is False
    assert casos_por_nombre["tipo_invalido"]["valido"] is False
    assert casos_por_nombre["solo_emojis"]["valido"] is True
    assert len(casos_por_nombre["solo_emojis"]["advertencias"]) >= 1
    assert casos_por_nombre["ambiguo_contrastante"]["valido"] is True


def test_seguridad_rechaza_joblib_externo(tmp_path: Path):
    """[Parte 2.3] Bloquea cualquier intento de cargar un .joblib fuera de models/."""
    external_file = tmp_path / "malicioso.joblib"
    external_file.write_bytes(b"fake pickle payload")
    with pytest.raises(UntrustedModelError):
        verify_trusted_path(external_file)


def test_seguridad_rechaza_hash_alterado(tmp_path: Path):
    """[Parte 2.3] Detecta manipulación del archivo .joblib antes de deserializar."""
    artifact = tmp_path / "modelo.joblib"
    artifact.write_bytes(b"contenido_original")
    with pytest.raises(UntrustedModelError):
        verify_sha256_integrity(artifact, expected_sha256="0" * 64)


def test_compatibilidad_versiones_detecta_desajuste():
    """[Parte 2.4] Advierte o bloquea cuando la versión de scikit-learn difiere."""
    fake_trained = {"python": "3.9.0", "scikit-learn": "0.24.2", "joblib": "1.0.0"}
    warns = check_version_compatibility(fake_trained, strict=False)
    assert len(warns) >= 2

    with pytest.raises(VersionCompatibilityError):
        check_version_compatibility(fake_trained, strict=True)


def test_prediccion_controlada_casos_borde():
    bundle = load_active_model()
    res_empty = predecir_seguro("   ", bundle.pipeline)
    assert res_empty["valido"] is False
    assert res_empty["clase"] is None

    res_ambiguous = predecir_seguro(
        "El contenido fue muy bueno, pero los parciales fueron confusos.",
        bundle.pipeline,
        incluir_aspectos=True,
    )
    assert res_ambiguous["valido"] is True
    assert res_ambiguous["es_contrastante"] is True
    assert len(res_ambiguous["aspectos"]) == 9
    assert res_ambiguous["aspectos"]["contenidos"]["mencionado"] is True
    assert res_ambiguous["aspectos"]["evaluaciones_parciales"]["mencionado"] is True
