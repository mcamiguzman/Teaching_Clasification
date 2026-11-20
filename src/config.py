"""
Configuración centralizada del proyecto de clasificación de retroalimentación estudiantil.

Define rutas del repositorio, constantes de dominio heredadas del trabajo parcial
(`Informe_Parcial_SisInt.pdf` / `Parcial_SI_corregido.ipynb`), la taxonomía oficial
de los nueve aspectos docentes [Parte 2.5] y parámetros de seguridad/monitoreo [Parte 1.6].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

# -----------------------------------------------------------------------------
# Rutas base del proyecto
# -----------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REGISTRY_PATH: Path = MODELS_DIR / "registry.json"
MONITORING_DIR: Path = PROJECT_ROOT / "monitoring"
PREDICTIONS_LOG_PATH: Path = MONITORING_DIR / "predictions_log.jsonl"
BASELINE_STATS_PATH: Path = MONITORING_DIR / "baseline_stats.json"

# Asegurar que existan los directorios operativos
for _directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, MONITORING_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Constantes del dominio y del trabajo parcial previo
# -----------------------------------------------------------------------------
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

POLARITY_CLASSES: List[str] = ["negativo", "neutral", "positivo"]

RAW_CSV_FILENAMES: List[str] = [
    "feedback_curso_universitario.csv",
    "feedback_curso_universitario2.csv",
    "feedback_curso_universitario3.csv",
]

COURSES_CANONICAL: List[str] = [
    "Sistemas operativos",
    "Arquitectura del computador",
    "Programacion paralela",
    "Sistemas distribuidos",
    "Sistemas Inteligentes",
    "Fundamentos de programación",
    "Bases de datos",
]

SEMESTERS_CANONICAL: List[str] = ["2023-1", "2023-2", "2024-1", "2024-2"]

# Métricas de referencia reportadas en Informe_Parcial_SisInt.pdf (Paso 6 y Paso 7)
PARCIAL_REFERENCE_METRICS: Dict[str, object] = {
    "source": "Informe_Parcial_SisInt.pdf & Parcial_SI_corregido.ipynb",
    "best_model": "LinearSVC (class_weight='balanced', C=0.05) + CalibratedClassifierCV(cv=5)",
    "best_vectorizer": "TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=2, sublinear_tf=True)",
    "n_total": 618,
    "n_train": 494,
    "n_test": 124,
    "macro_f1_cv_train": 0.6163,
    "macro_f1_test": 0.6411,
    "accuracy_test": 0.6452,
    "dummy_macro_f1_test": 0.197,
    "dummy_accuracy_test": 0.419,
}

# -----------------------------------------------------------------------------
# [Parte 2.5 — Salida estructurada] Taxonomía fija de 9 aspectos docentes
# Derivada de la columna `aspectos` auditada en el parcial y dimensiones SET
# (Student Evaluation of Teaching) en cursos de Ingeniería.
# -----------------------------------------------------------------------------
NINE_ASPECTS: List[str] = [
    "contenidos",
    "metodologia_docente",
    "evaluaciones_parciales",
    "material_clase",
    "retroalimentacion",
    "organizacion_horario",
    "disponibilidad_docente",
    "recursos_aula_digital",
    "dificultad_ritmo",
]

ASPECT_LABELS_ES: Dict[str, str] = {
    "contenidos": "Contenidos y pertinencia",
    "metodologia_docente": "Metodología y pedagogía docente",
    "evaluaciones_parciales": "Evaluaciones y parciales",
    "material_clase": "Material de clase y diapositivas",
    "retroalimentacion": "Retroalimentación y calificación",
    "organizacion_horario": "Organización y horario del curso",
    "disponibilidad_docente": "Disponibilidad y trato docente",
    "recursos_aula_digital": "Recursos y aula digital",
    "dificultad_ritmo": "Dificultad y ritmo de avance",
}

# Patrones léxicos para detectar menciones de cada uno de los 9 aspectos en español
ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "contenidos": [
        r"\bcontenid\w*",
        r"\btema\w*",
        r"\btem[aá]tic\w*",
        r"\bpertinenc\w*",
        r"\bconcepto\w*",
        r"\bprograma\b",
        r"\bsyllabus\b",
        r"\bs[ií]labo\b",
    ],
    "metodologia_docente": [
        r"\bmetodolog[ií]\w*",
        r"\bpedagog[ií]\w*",
        r"\bexplic\w*",
        r"\benseñ\w*",
        r"\bclases?\b",
        r"\bdin[aá]mic\w*",
        r"\bpr[aá]ctic\w*",
        r"\bprofesor\w*",
        r"\bdocente\w*",
    ],
    "evaluaciones_parciales": [
        r"\bparcial\w*",
        r"\bevaluaci[oó]n\w*",
        r"\bex[aá]men\w*",
        r"\bexamen\w*",
        r"\bquiz\w*",
        r"\bquices\b",
        r"\bpreguntas?\b",
        r"\bcalificaci[oó]n\w*",
        r"\bnotas?\b",
        r"\btaller\w*",
        r"\bproyecto\w*",
    ],
    "material_clase": [
        r"\bmaterial\w*",
        r"\bdiapositiv\w*",
        r"\bpresentaci[oó]n\w*",
        r"\bgu[ií]as?\b",
        r"\bejemplo\w*",
        r"\bejercicio\w*",
        r"\blectura\w*",
        r"\bapunte\w*",
        r"\bdocumentaci[oó]n\b",
    ],
    "retroalimentacion": [
        r"\bretroalimentaci[oó]n\w*",
        r"\bfeedback\b",
        r"\bdevoluci[oó]n\w*",
        r"\bcomentarios?\s+sobre\b",
        r"\bcorrecci[oó]n\w*",
        r"\brevisi[oó]n\w*",
    ],
    "organizacion_horario": [
        r"\borganizaci[oó]n\w*",
        r"\borganizad\w*",
        r"\bdesorganizad\w*",
        r"\bhorario\w*",
        r"\bcronograma\w*",
        r"\bplaneaci[oó]n\w*",
        r"\bplanificaci[oó]n\w*",
        r"\bestructura\w*",
        r"\btiempo\s+libre\b",
        r"\bpuntual\w*",
    ],
    "disponibilidad_docente": [
        r"\bdisponibilid\w*",
        r"\bcorreo\w*",
        r"\bdudas?\b",
        r"\brespond\w*",
        r"\brespuesta\w*",
        r"\baten(?:ci[oó]n|t[oa])\w*",
        r"\bamabl\w*",
        r"\btrato\b",
        r"\bacompañamiento\b",
        r"\btutor[ií]a\w*",
        r"\bcomunicaci[oó]n\b",
    ],
    "recursos_aula_digital": [
        r"\baula\s+digital\b",
        r"\bplataforma\w*",
        r"\brecurso\w*",
        r"\blaboratorio\w*",
        r"\bvirtual\w*",
        r"\bmoodle\b",
        r"\bcampus\b",
        r"\bherramienta\w*",
        r"\bsoftware\b",
        r"\bequipo\w*",
    ],
    "dificultad_ritmo": [
        r"\bdificult\w*",
        r"\bdif[ií]cil\w*",
        r"\bf[aá]cil\w*",
        r"\britmo\b",
        r"\br[aá]pid\w*",
        r"\blent\w*",
        r"\bpesad\w*",
        r"\bdesafiant\w*",
        r"\bexigent\w*",
        r"\bcarga\s+acad[eé]mic\w*",
        r"\bcomplej\w*",
    ],
}

# -----------------------------------------------------------------------------
# [Parte 1.6 & Parte 2.4] Configuración de seguridad, límites y drift
# -----------------------------------------------------------------------------
ADMIN_API_KEY: str = os.getenv("ADMIN_API_KEY", "")
STRICT_VERSION_CHECK: bool = os.getenv("STRICT_VERSION_CHECK", "false").lower() == "true"
MAX_COMMENT_LENGTH: int = int(os.getenv("MAX_COMMENT_LENGTH", "3000"))
MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "500"))
MAX_CSV_ROWS: int = int(os.getenv("MAX_CSV_ROWS", "5000"))
DRIFT_PSI_WARNING: float = float(os.getenv("DRIFT_PSI_WARNING", "0.10"))
DRIFT_PSI_CRITICAL: float = float(os.getenv("DRIFT_PSI_CRITICAL", "0.25"))
