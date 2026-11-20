"""
Esquemas Pydantic de entrada y salida para el servicio REST [Parte 1.4 & Parte 2.5].

Destaca especialmente la definición estricta de **salida estructurada para 9 aspectos**
(`NineAspectsPrediction`), garantizando que cada comentario clasificado retorne
exactamente una predicción estructurada por cada una de las nueve dimensiones docentes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.config import MAX_BATCH_SIZE, MAX_COMMENT_LENGTH

PolarityLabel = Literal["negativo", "neutral", "positivo"]
AspectPolarityLabel = Literal["negativo", "neutral", "positivo", "no_mencionado"]


class AspectPrediction(BaseModel):
    """[Parte 2.5] Predicción estructurada para un único aspecto docente."""

    aspecto: str = Field(..., description="Identificador canónico del aspecto.")
    nombre_legible: str = Field(..., description="Nombre descriptivo en español.")
    mencionado: bool = Field(..., description="Indica si el aspecto fue detectado en el comentario.")
    polaridad: AspectPolarityLabel = Field(
        ...,
        description="Polaridad predicha del aspecto o 'no_mencionado' si no aplica.",
    )
    confianza: float = Field(..., ge=0.0, le=1.0, description="Probabilidad máxima del aspecto.")
    probabilidades: Dict[str, float] = Field(
        ...,
        description="Probabilidades calibradas por clase (negativo, neutral, positivo).",
    )
    evidencia: Optional[str] = Field(
        default=None,
        description="Cláusula textual del comentario asociada al aspecto.",
    )


class NineAspectsOutput(BaseModel):
    """
    [Parte 2.5 — Salida estructurada]
    Contrato fijo con exactamente nueve campos tipados (uno por aspecto docente).
    """

    contenidos: AspectPrediction
    metodologia_docente: AspectPrediction
    evaluaciones_parciales: AspectPrediction
    material_clase: AspectPrediction
    retroalimentacion: AspectPrediction
    organizacion_horario: AspectPrediction
    disponibilidad_docente: AspectPrediction
    recursos_aula_digital: AspectPrediction
    dificultad_ritmo: AspectPrediction


class PredictRequest(BaseModel):
    """Solicitud de clasificación para un comentario individual [Parte 1.4]."""

    comentario: str = Field(
        ...,
        max_length=MAX_COMMENT_LENGTH,
        description="Texto abierto del estudiante a clasificar.",
        examples=[
            "El contenido fue amplio y útil, pero la disponibilidad de recursos en el aula digital era limitada."
        ],
    )
    curso: Optional[str] = Field(default=None, max_length=120, description="Asignatura opcional.")
    semestre: Optional[str] = Field(default=None, max_length=32, description="Periodo académico opcional.")
    incluir_aspectos: bool = Field(
        default=True,
        description="Si es True, incluye la salida estructurada de los 9 aspectos [Parte 2.5].",
    )


class PredictResponse(BaseModel):
    """Respuesta estructurada de inferencia individual [Parte 1.4 & Parte 2.5]."""

    valido: bool
    error: Optional[str] = None
    clase: Optional[PolarityLabel] = None
    confianza: float = Field(default=0.0, ge=0.0, le=1.0)
    probabilidades: Dict[str, float] = Field(default_factory=dict)
    texto_normalizado: str = ""
    es_contrastante: bool = False
    advertencias: List[str] = Field(default_factory=list)
    aspectos: Optional[NineAspectsOutput] = None
    model_version: str
    latency_ms: float


class BatchPredictRequest(BaseModel):
    """Solicitud de clasificación en lote."""

    comentarios: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"Lista de comentarios a clasificar (máximo {MAX_BATCH_SIZE}).",
    )
    incluir_aspectos: bool = Field(default=True)


class BatchPredictResponse(BaseModel):
    """Respuesta consolidada para clasificación en lote."""

    total: int
    validos: int
    con_error: int
    distribucion_clases: Dict[str, int]
    model_version: str
    total_latency_ms: float
    resultados: List[PredictResponse]


class VersionSwitchRequest(BaseModel):
    """Solicitud administrativa para cambiar o revertir versión del modelo [Parte 2.6]."""

    target_version: Optional[str] = Field(
        default=None,
        description="Versión destino (ej. 'v1.0.0'). En rollback puede omitirse para volver a la anterior.",
    )
    reason: str = Field(
        default="Operación de control de versiones vía API",
        max_length=300,
    )


class HealthResponse(BaseModel):
    """Estado de salud del servicio de predicción [Parte 1.6]."""

    status: str
    model_loaded: bool
    active_model_version: Optional[str]
    sha256_verified: bool
    version_warnings: List[str]
    runtime_versions: Dict[str, str]
    uptime_seconds: float
    extra: Dict[str, Any] = Field(default_factory=dict)
