"""
Módulo de monitoreo en producción: latencia, auditoría de inferencias y data/prediction drift.

Implementa:
- [Parte 1.6 — Probar, desplegar y monitorear]:
    * Registra cada inferencia en `monitoring/predictions_log.jsonl` (sin exponer PII:
      almacena hash corto del comentario, longitud en caracteres/palabras, presencia de
      cláusula contrastante, clase predicha, confianza, latencia en ms y versión de modelo).
    * Guarda estadísticas base del conjunto de entrenamiento en `monitoring/baseline_stats.json`.
    * Calcula el índice **PSI (Population Stability Index)** entre la distribución de clases
      en producción y la línea base de entrenamiento, además del desplazamiento en longitud
      media de textos, tasa de cláusulas contrastantes y confianza media.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    BASELINE_STATS_PATH,
    DRIFT_PSI_CRITICAL,
    DRIFT_PSI_WARNING,
    POLARITY_CLASSES,
    PREDICTIONS_LOG_PATH,
)


def save_training_baseline(
    texts: pd.Series,
    labels: pd.Series,
    contrastive_flags: pd.Series,
    model_version: str,
    dataset_version: str,
    baseline_path: Path = BASELINE_STATS_PATH,
) -> Dict[str, Any]:
    """
    Calcula y persiste las estadísticas de referencia del conjunto de entrenamiento
    para detectar cambios en los datos (data drift) y en las predicciones (prediction drift).
    """
    lengths_chars = texts.astype(str).str.len()
    lengths_words = texts.astype(str).str.split().str.len()
    class_dist = labels.value_counts(normalize=True).to_dict()

    stats: Dict[str, Any] = {
        "model_version": model_version,
        "dataset_version": dataset_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_reference": int(len(texts)),
        "class_proportions": {
            c: round(float(class_dist.get(c, 0.0)), 4) for c in POLARITY_CLASSES
        },
        "char_length_mean": round(float(lengths_chars.mean()), 2),
        "char_length_std": round(float(lengths_chars.std(ddof=0)), 2),
        "word_count_mean": round(float(lengths_words.mean()), 2),
        "word_count_std": round(float(lengths_words.std(ddof=0)), 2),
        "contrastive_ratio": round(float(contrastive_flags.mean()), 4),
    }

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def log_prediction_event(
    *,
    comment_text: str,
    predicted_class: Optional[str],
    probabilities: Dict[str, float],
    confidence: float,
    latency_ms: float,
    model_version: str,
    is_contrastive: bool,
    source: str = "api",
    course: Optional[str] = None,
    error: Optional[str] = None,
    log_path: Path = PREDICTIONS_LOG_PATH,
) -> None:
    """Registra una línea JSONL por cada predicción (diseño append-only de bajo costo)."""
    text_str = str(comment_text or "")
    text_hash = hashlib.sha256(text_str.encode("utf-8")).hexdigest()[:12]

    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "model_version": model_version,
        "text_hash": text_hash,
        "char_length": len(text_str),
        "word_count": len(text_str.split()),
        "is_contrastive": bool(is_contrastive),
        "course": course,
        "predicted_class": predicted_class,
        "confidence": round(float(confidence), 4),
        "probabilities": probabilities,
        "latency_ms": round(float(latency_ms), 3),
        "error": error,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def compute_psi(expected: Dict[str, float], actual: Dict[str, float], eps: float = 1e-4) -> float:
    """
    Calcula el Population Stability Index (PSI) para detectar prediction drift:
      - PSI < 0.10: estable (sin deriva relevante)
      - 0.10 <= PSI < 0.25: advertencia (cambio moderado en la distribución)
      - PSI >= 0.25: crítico (deriva significativa; evaluar reentrenamiento)
    """
    psi_total = 0.0
    for cls in POLARITY_CLASSES:
        e = max(float(expected.get(cls, 0.0)), eps)
        a = max(float(actual.get(cls, 0.0)), eps)
        psi_total += (a - e) * math.log(a / e)
    return round(float(psi_total), 4)


def build_monitoring_report(
    log_path: Path = PREDICTIONS_LOG_PATH,
    baseline_path: Path = BASELINE_STATS_PATH,
    last_n: int = 1000,
) -> Dict[str, Any]:
    """
    Genera un reporte consolidado de latencia, calidad y drift comparando las
    inferencias registradas en producción contra la línea base de entrenamiento.
    """
    baseline = (
        json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_path.exists()
        else None
    )

    events: List[Dict[str, Any]] = []
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").splitlines()
        for line in lines[-last_n:]:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not events:
        return {
            "total_predictions_logged": 0,
            "status": "sin_datos",
            "baseline_available": baseline is not None,
            "baseline": baseline,
            "latency_ms": {},
            "drift": {
                "psi_class_distribution": 0.0,
                "drift_alert": "sin_datos",
            },
            "recent_events": [],
        }

    valid_events = [e for e in events if e.get("predicted_class") in POLARITY_CLASSES]
    error_events = [e for e in events if e.get("error")]

    latencies = np.array([float(e.get("latency_ms", 0.0)) for e in events], dtype=float)
    char_lengths = np.array([float(e.get("char_length", 0.0)) for e in valid_events], dtype=float) if valid_events else np.array([0.0])
    confidences = np.array([float(e.get("confidence", 0.0)) for e in valid_events], dtype=float) if valid_events else np.array([0.0])
    contrast_rate = (
        float(np.mean([1.0 if e.get("is_contrastive") else 0.0 for e in valid_events]))
        if valid_events
        else 0.0
    )

    prod_counts = {c: 0 for c in POLARITY_CLASSES}
    for e in valid_events:
        prod_counts[e["predicted_class"]] += 1
    total_valid = max(len(valid_events), 1)
    prod_proportions = {c: round(prod_counts[c] / total_valid, 4) for c in POLARITY_CLASSES}

    psi_score = 0.0
    length_z_shift = 0.0
    drift_alert = "estable"

    if baseline and valid_events:
        psi_score = compute_psi(baseline["class_proportions"], prod_proportions)
        ref_std = max(float(baseline.get("char_length_std", 1.0)), 1.0)
        length_z_shift = round(
            (float(char_lengths.mean()) - float(baseline.get("char_length_mean", 0.0))) / ref_std,
            3,
        )
        if len(valid_events) < 10:
            drift_alert = "muestra_insuficiente"
        elif psi_score >= DRIFT_PSI_CRITICAL or abs(length_z_shift) >= 2.0:
            drift_alert = "critico"
        elif psi_score >= DRIFT_PSI_WARNING or abs(length_z_shift) >= 1.2:
            drift_alert = "advertencia"
        else:
            drift_alert = "estable"

    return {
        "total_predictions_logged": len(events),
        "valid_predictions": len(valid_events),
        "error_count": len(error_events),
        "error_rate": round(len(error_events) / max(len(events), 1), 4),
        "latency_ms": {
            "mean": round(float(latencies.mean()), 3),
            "p50": round(float(np.percentile(latencies, 50)), 3),
            "p95": round(float(np.percentile(latencies, 95)), 3),
            "p99": round(float(np.percentile(latencies, 99)), 3),
            "max": round(float(latencies.max()), 3),
        },
        "production_stats": {
            "class_counts": prod_counts,
            "class_proportions": prod_proportions,
            "mean_confidence": round(float(confidences.mean()), 4),
            "mean_char_length": round(float(char_lengths.mean()), 2),
            "contrastive_ratio": round(contrast_rate, 4),
        },
        "baseline": baseline,
        "drift": {
            "psi_class_distribution": psi_score,
            "char_length_z_shift": length_z_shift,
            "psi_thresholds": {
                "warning": DRIFT_PSI_WARNING,
                "critical": DRIFT_PSI_CRITICAL,
            },
            "drift_alert": drift_alert,
        },
        "recent_events": events[-20:],
    }
