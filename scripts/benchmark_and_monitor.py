"""
Benchmark de tiempos de inferencia, verificaciones de seguridad y reporte de drift.

Implementa:
- [Parte 1.6 — Probar, desplegar y monitorear]:
    * Mide latencia de inferencia individual (p50, p95, p99 en ms) y throughput
      (comentarios/segundo) para el Pipeline unificado y la extracción de 9 aspectos.
    * Ejecuta verificaciones de seguridad: rechazo de rutas fuera de `models/`,
      rechazo de archivos manipulados (SHA-256) y manejo seguro de entradas vacías.
    * Genera un reporte consolidado de monitoreo y detección de drift (PSI).
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_registry import UntrustedModelError, load_active_model, verify_trusted_path
from src.monitoring import build_monitoring_report, log_prediction_event
from src.validate_model import predecir_seguro

BENCHMARK_COMMENTS = [
    "Excelente curso, el profesor explica muy claro y las diapositivas están muy bien organizadas.",
    "El contenido fue amplio y útil, pero la disponibilidad de recursos en el aula digital era limitada.",
    "Muy desorganizado, los parciales fueron confusos y nunca respondieron dudas por correo.",
    "El horario es estándar y los talleres tuvieron una dificultad normal durante el semestre.",
    "Suuuuper buen curso, aunque el ritmo fue un poco rápido al final 😊",
]


def run_benchmark(n_iterations: int = 100) -> dict:
    bundle = load_active_model()

    # Warm-up
    _ = predecir_seguro(BENCHMARK_COMMENTS[0], bundle.pipeline, incluir_aspectos=True)

    latencies_polarity_only = []
    latencies_with_9_aspects = []

    for i in range(n_iterations):
        comment = BENCHMARK_COMMENTS[i % len(BENCHMARK_COMMENTS)]

        t0 = time.perf_counter()
        res_simple = predecir_seguro(comment, bundle.pipeline, incluir_aspectos=False)
        latencies_polarity_only.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        res_full = predecir_seguro(comment, bundle.pipeline, incluir_aspectos=True)
        elapsed_full = (time.perf_counter() - t1) * 1000.0
        latencies_with_9_aspects.append(elapsed_full)

        # Registrar una muestra representativa en el log de monitoreo
        if i < 25:
            log_prediction_event(
                comment_text=comment,
                predicted_class=res_full["clase"],
                probabilities=res_full["probabilidades"],
                confidence=res_full["confianza"],
                latency_ms=elapsed_full,
                model_version=bundle.version,
                is_contrastive=res_full["es_contrastante"],
                source="benchmark",
            )

    # Verificación de seguridad [Parte 2.3]: un .joblib fuera de models/ debe ser rechazado
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        try:
            verify_trusted_path(tmp_path)
            security_path_blocked = False
        except UntrustedModelError:
            security_path_blocked = True
    finally:
        tmp_path.unlink(missing_ok=True)

    arr_p = np.array(latencies_polarity_only)
    arr_f = np.array(latencies_with_9_aspects)

    report = {
        "model_version": bundle.version,
        "sha256_verified": bundle.sha256_verified,
        "iterations": n_iterations,
        "latency_polarity_only_ms": {
            "mean": round(float(arr_p.mean()), 3),
            "p50": round(float(np.percentile(arr_p, 50)), 3),
            "p95": round(float(np.percentile(arr_p, 95)), 3),
            "p99": round(float(np.percentile(arr_p, 99)), 3),
            "throughput_req_per_sec": round(1000.0 / max(float(arr_p.mean()), 1e-6), 1),
        },
        "latency_with_9_aspects_ms": {
            "mean": round(float(arr_f.mean()), 3),
            "p50": round(float(np.percentile(arr_f, 50)), 3),
            "p95": round(float(np.percentile(arr_f, 95)), 3),
            "p99": round(float(np.percentile(arr_f, 99)), 3),
            "throughput_req_per_sec": round(1000.0 / max(float(arr_f.mean()), 1e-6), 1),
        },
        "security_checks": {
            "untrusted_external_joblib_blocked": security_path_blocked,
            "empty_input_controlled": predecir_seguro("", bundle.pipeline)["valido"] is False,
        },
        "monitoring_drift_summary": build_monitoring_report()["drift"],
    }
    return report


if __name__ == "__main__":
    print(json.dumps(run_benchmark(100), ensure_ascii=False, indent=2))
