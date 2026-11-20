"""
Analizador estructurado de los nueve aspectos de evaluación docente.

Implementa:
- [Parte 2.5 — Salida estructurada]: cuando se evalúan múltiples aspectos
  (nueve dimensiones del curso universitario), garantiza un formato de salida
  estricto y homogéneo con exactamente UNA entrada estructurada por aspecto:
    1. `contenidos`
    2. `metodologia_docente`
    3. `evaluaciones_parciales`
    4. `material_clase`
    5. `retroalimentacion`
    6. `organizacion_horario`
    7. `disponibilidad_docente`
    8. `recursos_aula_digital`
    9. `dificultad_ritmo`

Estrategia técnica (coherente con el hallazgo de cláusulas contrastantes del parcial):
- Un comentario suele elogiar un aspecto y criticar otro separados por conectores
  adversativos ("El contenido fue amplio y útil, pero la disponibilidad de recursos
  en el aula digital era limitada").
- Este módulo segmenta el comentario en cláusulas pragmáticas (por signos de
  puntuación y conectores contrastantes), asigna cada cláusula a los aspectos
  mencionados y evalúa la polaridad de la cláusula correspondiente usando el
  MISMO `Pipeline` unificado de polaridad calibrado en `.joblib`.
- Para los aspectos no mencionados en el comentario, devuelve explícitamente
  `mencionado=False`, `polaridad="no_mencionado"` y probabilidades en `0.0`,
  garantizando siempre las 9 llaves en la respuesta estructurada.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from src.config import ASPECT_KEYWORDS, ASPECT_LABELS_ES, NINE_ASPECTS, POLARITY_CLASSES
from src.preprocessing import normalizar_texto

# Separador de cláusulas para aislar opiniones encontradas antes/después de conectores
_CLAUSE_SPLIT_PATTERN = re.compile(
    r"(?:[.;!?]+|\b(?:pero|sin embargo|aunque|no obstante|por el contrario|a pesar de(?: ello| esto| que)?)\b)",
    flags=re.IGNORECASE,
)

# Compilación previa de patrones léxicos por cada uno de los 9 aspectos
_COMPILED_ASPECT_PATTERNS: Dict[str, List[re.Pattern[str]]] = {
    aspect: [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
    for aspect, patterns in ASPECT_KEYWORDS.items()
}


def segmentar_clausulas(texto: str) -> List[str]:
    """Divide un comentario en cláusulas pragmáticas preservando contexto suficiente."""
    texto_limpio = normalizar_texto(texto)
    if not texto_limpio:
        return []
    partes = [p.strip(" ,:-") for p in _CLAUSE_SPLIT_PATTERN.split(texto_limpio)]
    clausulas = [p for p in partes if len(p) >= 3]
    return clausulas if clausulas else [texto_limpio]


def _detectar_clausulas_por_aspecto(clausulas: List[str]) -> Dict[str, List[str]]:
    """Asocia cada uno de los 9 aspectos con las cláusulas donde es mencionado."""
    matches: Dict[str, List[str]] = {aspect: [] for aspect in NINE_ASPECTS}
    for clausula in clausulas:
        for aspect, regex_list in _COMPILED_ASPECT_PATTERNS.items():
            if any(rx.search(clausula) for rx in regex_list):
                matches[aspect].append(clausula)
    return matches


def predecir_nueve_aspectos(
    comentario: str,
    pipeline_polaridad: Any,
) -> Dict[str, Dict[str, Any]]:
    """
    [Parte 2.5 — Salida estructurada]
    Genera exactamente una predicción estructurada para cada uno de los 9 aspectos.

    Parameters
    ----------
    comentario : str
        Comentario crudo del estudiante.
    pipeline_polaridad : sklearn.pipeline.Pipeline
        Pipeline unificado cargado desde `.joblib` con método `predict_proba`.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Diccionario indexado por los 9 aspectos oficiales (`NINE_ASPECTS`).
        Cada valor contiene:
          - `aspecto`: identificador canónico
          - `nombre_legible`: etiqueta en español para UI/reportes
          - `mencionado`: bool
          - `polaridad`: `'positivo' | 'negativo' | 'neutral' | 'no_mencionado'`
          - `confianza`: float entre 0 y 1
          - `probabilidades`: dict con claves `negativo`, `neutral`, `positivo`
          - `evidencia`: fragmento textual (cláusula) usada para la inferencia o `None`
    """
    clausulas = segmentar_clausulas(comentario)
    clausulas_por_aspecto = _detectar_clausulas_por_aspecto(clausulas)

    # Preparar inferencia en lote únicamente para los aspectos mencionados (eficiencia)
    aspectos_mencionados = [a for a in NINE_ASPECTS if clausulas_por_aspecto[a]]
    textos_aspectos = [" . ".join(clausulas_por_aspecto[a]) for a in aspectos_mencionados]

    probas_por_aspecto: Dict[str, Dict[str, float]] = {}
    clases_modelo = list(pipeline_polaridad.classes_)

    if textos_aspectos:
        matriz_proba = pipeline_polaridad.predict_proba(textos_aspectos)
        for aspecto, fila_proba in zip(aspectos_mencionados, matriz_proba):
            probas_por_aspecto[aspecto] = {
                str(clase): round(float(prob), 4)
                for clase, prob in zip(clases_modelo, fila_proba)
            }

    salida_estructurada: Dict[str, Dict[str, Any]] = {}
    for aspecto in NINE_ASPECTS:
        if aspecto in probas_por_aspecto:
            probs = probas_por_aspecto[aspecto]
            clase_ganadora = max(probs, key=probs.get)
            confianza = round(float(probs[clase_ganadora]), 4)
            evidencia = " | ".join(clausulas_por_aspecto[aspecto])
            salida_estructurada[aspecto] = {
                "aspecto": aspecto,
                "nombre_legible": ASPECT_LABELS_ES[aspecto],
                "mencionado": True,
                "polaridad": clase_ganadora,
                "confianza": confianza,
                "probabilidades": probs,
                "evidencia": evidencia,
            }
        else:
            salida_estructurada[aspecto] = {
                "aspecto": aspecto,
                "nombre_legible": ASPECT_LABELS_ES[aspecto],
                "mencionado": False,
                "polaridad": "no_mencionado",
                "confianza": 0.0,
                "probabilidades": {c: 0.0 for c in POLARITY_CLASSES},
                "evidencia": None,
            }

    return salida_estructurada
