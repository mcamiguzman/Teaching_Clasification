"""
Pruebas unitarias para preprocesamiento y Pipeline unificado [Parte 2.1].
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    TextNormalizer,
    corregir_polaridad,
    detectar_contraste,
    es_solo_emojis_o_simbolos,
    normalizar_texto,
)


def test_normalizar_texto_emojis_elongaciones_e_intensificadores():
    raw = "Suuuuper bueeeeno el curso, muuuuy recomendado 😀  \n "
    clean = normalizar_texto(raw)
    assert "cara" in clean.lower() or ":" in clean  # demojize en español
    assert "suuuuper" not in clean.lower()
    assert "súper" in clean.lower()
    assert "muy" in clean.lower()
    assert "\n" not in clean


def test_normalizar_texto_respeta_vacio_y_nulo():
    assert normalizar_texto("") == ""
    assert normalizar_texto("    \t\n ") == ""
    assert normalizar_texto(None) == ""
    assert es_solo_emojis_o_simbolos("😀😀😀") is True
    assert es_solo_emojis_o_simbolos("Buen curso 😀") is False


def test_corregir_polaridad_nueve_variantes_parcial():
    assert corregir_polaridad("['negativo']") == "negativo"
    assert corregir_polaridad("[\"['positivo']\"]") == "positivo"
    assert corregir_polaridad("positiva") == "positivo"
    assert corregir_polaridad("['posicion']") == "positivo"
    assert corregir_polaridad("['neutral']") == "neutral"


def test_detectar_contraste():
    assert detectar_contraste("El contenido fue bueno, pero los parciales confusos") is True
    assert detectar_contraste("Excelente curso y profesor claro") is False


def test_pipeline_unificado_incluye_normalizacion_interna():
    """[Parte 2.1] Verifica que TextNormalizer funciona dentro de sklearn.pipeline.Pipeline."""
    X = [
        "Excelente curso muy claro 😊",
        "Muy buen profesor y excelentes ejemplos",
        "Pésimo curso muy mal organizado 😠",
        "Horrible experiencia y parciales injustos",
        "El curso estuvo regular y el horario normal",
        "Contenido promedio sin mayores novedades",
    ]
    y = ["positivo", "positivo", "negativo", "negativo", "neutral", "neutral"]

    pipe = Pipeline(
        [
            ("normalizer", TextNormalizer()),
            ("vec", TfidfVectorizer(sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipe.fit(X, y)
    preds = pipe.predict(["Suuuuper excelente curso 😀"])
    probas = pipe.predict_proba(["Peeesimo curso 😠"])
    assert len(preds) == 1
    assert abs(float(probas[0].sum()) - 1.0) < 1e-6
