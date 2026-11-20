"""
Módulo de preprocesamiento léxico y transformador scikit-learn unificado.

Implementa:
- [Parte 2.1 — Pipeline unificado]: encapsula `normalizar_texto` dentro del
  transformador `TextNormalizer(BaseEstimator, TransformerMixin)` para que toda
  transformación de texto quede serializada junto con el vectorizador TF-IDF y
  el clasificador dentro del mismo archivo `.joblib`.
- Reglas fieles al Paso 1 y Paso 2 de `Parcial_SI_corregido.ipynb` e
  `Informe_Parcial_SisInt.pdf`:
    1. Conversión de emojis a descriptores textuales en español (`emoji.demojize`).
    2. Reducción de caracteres repetidos consecutivos (`r'(.)\\1{2,}' -> r'\\1\\1'`),
       preservando dígrafos del español como "ll" y "rr".
    3. Estandarización de intensificadores coloquiales alargados.
    4. Limpieza de espacios redundantes y manejo seguro de nulos/vacíos.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Iterable, List, Union

import emoji
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Diccionario de intensificadores del Paso 2 de Parcial_SI_corregido.ipynb
# (corregido para coincidir tras la reducción de caracteres repetidos a máx. 2)
INTENSIFICADORES = {
    r"\bsu+per+\b": "súper",
    r"\bree+\b": "re",
    r"\bmu+y+\b": "muy",
    r"\bpe+si+mo+\b": "pésimo",
    r"\bexcele+nte+\b": "excelente",
    r"\bbueni+simo+\b": "buenísimo",
}

# Conectores adversativos / concesivos auditados en el Paso 2 del parcial (76.38% del corpus)
PATRON_CONTRASTE = re.compile(
    r"\b(?:pero|sin embargo|aunque|no obstante|por el contrario|a pesar de)\b",
    flags=re.IGNORECASE,
)

# Patrón para detectar texto que quedó sin ningún carácter alfanumérico útil
PATRON_ALFANUMERICO = re.compile(r"[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ0-9]")


def normalizar_texto(texto: Any) -> str:
    """
    Normaliza un comentario estudiantil conservando carga emocional y léxica.

    Pasos aplicados (fieles al Paso 2 de `Parcial_SI_corregido.ipynb`):
      a. Control de nulos, tipos no textuales y cadenas vacías.
      b. Conversión de emojis a palabras en español (ej. 😀 -> :cara_sonriendo:).
      c. Recorte de caracteres repetidos (>=3 repeticiones -> 2, respetando 'll', 'rr').
      d. Estandarización de intensificadores alargados ('suuuper' -> 'súper', etc.).
      e. Normalización de espacios y saltos de línea.

    Parameters
    ----------
    texto : Any
        Comentario de entrada en texto crudo.

    Returns
    -------
    str
        Texto normalizado (o cadena vacía `""` si la entrada era nula o sin contenido).
    """
    if texto is None:
        return ""
    if isinstance(texto, float) and np.isnan(texto):
        return ""
    try:
        if pd.isna(texto):
            return ""
    except (TypeError, ValueError):
        pass

    texto_str = str(texto)
    if not texto_str.strip():
        return ""

    # a. Convertir emojis a descripciones textuales en español (:cara_sonriendo:)
    texto_str = emoji.demojize(texto_str, language="es")

    # b. Reducir caracteres repetidos consecutivos (ej. "bueeeno" -> "bueeno", respeta "ll"/"rr")
    texto_str = re.sub(r"(.)\1{2,}", r"\1\1", texto_str)

    # c. Estandarizar variantes de intensificadores alargados
    for patron, reemplazo in INTENSIFICADORES.items():
        texto_str = re.sub(patron, reemplazo, texto_str, flags=re.IGNORECASE)

    # d. Limpiar espacios redundantes y saltos de línea
    texto_str = re.sub(r"\s+", " ", texto_str).strip()
    return texto_str


def detectar_contraste(texto: str) -> bool:
    """Indica si el comentario contiene conectores adversativos o concesivos."""
    if not texto:
        return False
    return bool(PATRON_CONTRASTE.search(texto))


def es_solo_emojis_o_simbolos(texto_original: str) -> bool:
    """
    Detecta comentarios que no contenían texto alfanumérico antes de `demojize`
    (ej. '😀😀😀' o '???!!!'). Útil para advertencias de baja confiabilidad en
    validación y producción [Parte 1.3], replicando el hallazgo del informe.
    """
    if not texto_original or not str(texto_original).strip():
        return False
    return not bool(PATRON_ALFANUMERICO.search(str(texto_original)))


def corregir_polaridad(val: Any) -> str:
    """
    Normaliza las 9 variantes inconsistentes de `polaridad_basica` detectadas en el Paso 1
    del parcial (`['negativo']`, `positiva`, `["['positivo']"]`, `['posicion']`, etc.)
    a la taxonomía canónica: `positivo`, `negativo`, `neutral`.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "neutral"
    v = re.sub(r"[^a-záéíóúñ]", "", str(val).lower().strip())
    mapeo = {
        "positivo": "positivo",
        "positiva": "positivo",
        "posicion": "positivo",
        "negativo": "negativo",
        "negativa": "negativo",
        "neutral": "neutral",
        "neutro": "neutral",
    }
    return mapeo.get(v, "neutral")


def parsear_aspectos(val: Any) -> List[dict]:
    """
    Convierte de forma segura la columna `aspectos` (JSON o lista Python serializada
    como string) en una lista nativa de diccionarios.
    """
    if val is None:
        return []
    if isinstance(val, list):
        return val
    val_str = str(val).strip()
    if val_str in {"", "[]", "nan", "None"}:
        return []
    try:
        resultado = json.loads(val_str)
        if isinstance(resultado, list):
            return resultado
    except Exception:
        pass
    try:
        resultado = ast.literal_eval(val_str)
        if isinstance(resultado, list):
            return resultado
    except Exception:
        pass
    return []


class TextNormalizer(BaseEstimator, TransformerMixin):
    """
    [Parte 2.1 — Pipeline unificado]
    Transformador scikit-learn sin estado (stateless) que aplica `normalizar_texto`
    a colecciones de comentarios dentro del `Pipeline` serializado en `.joblib`.

    Ventaja clave frente al cuaderno parcial original:
    En `Parcial_SI_corregido.ipynb`, `normalizar_texto` debía invocarse manualmente
    antes de llamar a `modelo.predict(...)`. Al integrarlo como primer paso del
    `Pipeline` (`('normalizer', TextNormalizer())`), el archivo `.joblib` es
    100% autocontenido y garantiza que entrenamiento, validación y producción
    ejecuten exactamente la misma transformación textual sin sesgos ni olvidos.
    """

    def __init__(self, lowercase: bool = False) -> None:
        # `TfidfVectorizer` ya aplica `lowercase=True` internamente, por lo que
        # se deja opcional para conservar compatibilidad exacta con el parcial.
        self.lowercase = lowercase

    def fit(self, X: Iterable[Any], y: Any = None) -> "TextNormalizer":
        """No requiere ajuste sobre datos (evita cualquier fuga de información)."""
        return self

    def transform(self, X: Union[Iterable[Any], pd.Series, np.ndarray]) -> List[str]:
        """Normaliza cada texto de entrada conservando el orden e índice."""
        if isinstance(X, str):
            X_iter: Iterable[Any] = [X]
        else:
            X_iter = X

        resultados: List[str] = []
        for item in X_iter:
            limpio = normalizar_texto(item)
            if self.lowercase:
                limpio = limpio.lower()
            resultados.append(limpio)
        return resultados
