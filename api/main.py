"""
Servicio REST de predicción de retroalimentación estudiantil con FastAPI.

Implementa explícitamente:
- [Parte 1.4 — Crear un servicio de predicción]:
    * Expone `POST /predict`, `POST /predict/batch` y `POST /predict/csv` retornando
      la clase predicha y probabilidades calibradas por clase.
- [Parte 2.2 — Eficiencia de entrenamiento]:
    * El modelo `.joblib` se carga UNA SOLA VEZ al arrancar el servicio (`lifespan`)
      y se mantiene en memoria (`app.state.model_bundle`). Nunca se reentrena al iniciar.
- [Parte 2.3 & 2.4 — Seguridad en deserialización y compatibilidad de versiones]:
    * Delega la carga a `src.model_registry.load_active_model()`, que verifica ruta
      confiable, huella SHA-256 y compatibilidad de versiones antes de `joblib.load()`.
- [Parte 2.5 — Salida estructurada]:
    * Cada predicción incluye un bloque tipado `NineAspectsOutput` con exactamente
      una predicción estructurada por cada uno de los 9 aspectos docentes.
- [Parte 2.6 — Control y reversión de versiones]:
    * Endpoints `GET /model/versions`, `POST /model/switch` y `POST /model/rollback`
      con recarga en caliente del artefacto en memoria (protegidos por `X-API-Key` si
      `ADMIN_API_KEY` está configurada).
- [Parte 1.6 — Probar, desplegar y monitorear]:
    * Middleware de latencia (`X-Inference-Time-Ms`), validación de tamaños máximos,
      registro de métricas de inferencia y endpoint `GET /monitoring/summary` con PSI drift.
"""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    NineAspectsOutput,
    PredictRequest,
    PredictResponse,
    VersionSwitchRequest,
)
from src.config import ADMIN_API_KEY, MAX_COMMENT_LENGTH, MAX_CSV_ROWS, POLARITY_CLASSES
from src.model_registry import (
    LoadedModelBundle,
    get_runtime_versions,
    list_registered_versions,
    load_active_model,
    load_registry,
    rollback_model_version,
    switch_active_version,
)
from src.monitoring import build_monitoring_report, log_prediction_event
from src.validate_model import predecir_seguro


def _verify_admin_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    [Parte 1.6 — Seguridad]
    Protege operaciones administrativas (cambio/reversión de versión del modelo)
    cuando `ADMIN_API_KEY` está definida en el entorno.
    """
    if ADMIN_API_KEY and x_api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="No autorizado: cabecera X-API-Key inválida o ausente.",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    [Parte 2.2 — Eficiencia]
    Carga el Pipeline serializado en `.joblib` una única vez durante el arranque.
    """
    app.state.started_at = time.perf_counter()
    try:
        app.state.model_bundle = load_active_model()
    except Exception as exc:
        app.state.model_bundle = None
        app.state.load_error = str(exc)
    else:
        app.state.load_error = None
    yield


app = FastAPI(
    title="API de Clasificación de Retroalimentación Estudiantil",
    description=(
        "Servicio productivo basado en el proyecto parcial de Sistemas Inteligentes "
        "(PUJ Cali). Expone clasificación de polaridad calibrada + salida estructurada "
        "de 9 aspectos docentes, gestión/rollback de versiones y monitoreo de drift."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def latency_header_middleware(request: Request, call_next):
    """Mide el tiempo de atención de cada petición y lo expone en cabecera HTTP."""
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.3f}"
    return response


def _get_bundle(request: Request) -> LoadedModelBundle:
    bundle: Optional[LoadedModelBundle] = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        detail = getattr(request.app.state, "load_error", None) or "Modelo no cargado."
        raise HTTPException(status_code=503, detail=f"Servicio no disponible: {detail}")
    return bundle


def _predict_single_internal(
    comentario: str,
    bundle: LoadedModelBundle,
    incluir_aspectos: bool = True,
    curso: Optional[str] = None,
    source: str = "api",
) -> PredictResponse:
    """Ejecuta inferencia segura, mide latencia y registra evento de monitoreo."""
    t0 = time.perf_counter()
    raw_result = predecir_seguro(
        comentario=comentario,
        pipeline=bundle.pipeline,
        incluir_aspectos=incluir_aspectos,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    aspectos_obj = (
        NineAspectsOutput(**raw_result["aspectos"])
        if raw_result["valido"] and incluir_aspectos and raw_result.get("aspectos")
        else None
    )

    log_prediction_event(
        comment_text=comentario,
        predicted_class=raw_result.get("clase"),
        probabilities=raw_result.get("probabilidades", {}),
        confidence=float(raw_result.get("confianza", 0.0)),
        latency_ms=latency_ms,
        model_version=bundle.version,
        is_contrastive=bool(raw_result.get("es_contrastante", False)),
        source=source,
        course=curso,
        error=raw_result.get("error"),
    )

    return PredictResponse(
        valido=raw_result["valido"],
        error=raw_result.get("error"),
        clase=raw_result.get("clase"),
        confianza=float(raw_result.get("confianza", 0.0)),
        probabilidades=raw_result.get("probabilidades", {}),
        texto_normalizado=raw_result.get("texto_normalizado", ""),
        es_contrastante=bool(raw_result.get("es_contrastante", False)),
        advertencias=raw_result.get("advertencias", []),
        aspectos=aspectos_obj,
        model_version=bundle.version,
        latency_ms=latency_ms,
    )


@app.get("/health", response_model=HealthResponse, tags=["Operación"])
def health(request: Request) -> HealthResponse:
    """[Parte 1.6] Verifica disponibilidad del servicio, modelo activo e integridad."""
    bundle: Optional[LoadedModelBundle] = getattr(request.app.state, "model_bundle", None)
    started_at = getattr(request.app.state, "started_at", time.perf_counter())
    return HealthResponse(
        status="ok" if bundle is not None else "degraded",
        model_loaded=bundle is not None,
        active_model_version=bundle.version if bundle else None,
        sha256_verified=bundle.sha256_verified if bundle else False,
        version_warnings=bundle.version_warnings if bundle else [],
        runtime_versions=get_runtime_versions(),
        uptime_seconds=round(time.perf_counter() - started_at, 2),
        extra={"load_error": getattr(request.app.state, "load_error", None)},
    )


@app.post("/predict", response_model=PredictResponse, tags=["Predicción"])
def predict_comment(payload: PredictRequest, request: Request) -> PredictResponse:
    """
    [Parte 1.4 & Parte 2.5]
    Clasifica un comentario individual y devuelve clase predicha, probabilidades
    por clase y análisis estructurado de los 9 aspectos docentes.
    """
    bundle = _get_bundle(request)
    return _predict_single_internal(
        comentario=payload.comentario,
        bundle=bundle,
        incluir_aspectos=payload.incluir_aspectos,
        curso=payload.curso,
        source="api_single",
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Predicción"])
def predict_batch(payload: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    """[Parte 1.4] Clasifica un lote de comentarios en una única llamada HTTP."""
    bundle = _get_bundle(request)
    t0 = time.perf_counter()

    resultados: List[PredictResponse] = []
    counts = {c: 0 for c in POLARITY_CLASSES}
    validos = 0
    con_error = 0

    for text in payload.comentarios:
        if len(text) > MAX_COMMENT_LENGTH:
            text = text[:MAX_COMMENT_LENGTH]
        res = _predict_single_internal(
            comentario=text,
            bundle=bundle,
            incluir_aspectos=payload.incluir_aspectos,
            source="api_batch",
        )
        resultados.append(res)
        if res.valido and res.clase:
            validos += 1
            counts[res.clase] = counts.get(res.clase, 0) + 1
        else:
            con_error += 1

    total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return BatchPredictResponse(
        total=len(resultados),
        validos=validos,
        con_error=con_error,
        distribucion_clases=counts,
        model_version=bundle.version,
        total_latency_ms=total_ms,
        resultados=resultados,
    )


@app.post("/predict/csv", tags=["Predicción"])
async def predict_csv(
    request: Request,
    file: UploadFile = File(..., description="Archivo CSV con columna de comentarios."),
    text_column: str = Query(default="feedback", description="Nombre de la columna con el comentario."),
    incluir_aspectos: bool = Query(default=False, description="Incluir los 9 aspectos por cada fila."),
) -> JSONResponse:
    """[Parte 1.4 & Parte 1.5] Procesa un archivo CSV cargado vía HTTP multipart."""
    bundle = _get_bundle(request)
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="El archivo debe tener extensión .csv")

    raw_bytes = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el CSV: {exc}") from exc

    if text_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"La columna '{text_column}' no existe en el CSV. Columnas disponibles: {list(df.columns)}",
        )
    if len(df) > MAX_CSV_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"El CSV supera el límite máximo permitido de {MAX_CSV_ROWS} filas.",
        )

    t0 = time.perf_counter()
    rows_out: List[Dict[str, Any]] = []
    counts = {c: 0 for c in POLARITY_CLASSES}

    for idx, row in df.iterrows():
        comment = "" if pd.isna(row[text_column]) else str(row[text_column])[:MAX_COMMENT_LENGTH]
        curso = str(row["curso"]) if "curso" in df.columns and not pd.isna(row["curso"]) else None
        pred = _predict_single_internal(
            comentario=comment,
            bundle=bundle,
            incluir_aspectos=incluir_aspectos,
            curso=curso,
            source="api_csv",
        )
        if pred.valido and pred.clase:
            counts[pred.clase] += 1
        item = {
            "fila": int(idx),
            "curso": curso,
            "comentario": comment,
            "valido": pred.valido,
            "clase": pred.clase,
            "confianza": pred.confianza,
            "probabilidades": pred.probabilidades,
            "error": pred.error,
        }
        if incluir_aspectos and pred.aspectos:
            item["aspectos"] = pred.aspectos.model_dump()
        rows_out.append(item)

    return JSONResponse(
        content={
            "filename": file.filename,
            "total_filas": len(rows_out),
            "distribucion_clases": counts,
            "model_version": bundle.version,
            "total_latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "resultados": rows_out,
        }
    )


@app.get("/model/info", tags=["Versionado y Metadatos"])
def model_info(request: Request) -> Dict[str, Any]:
    """[Parte 1.1 & Parte 1.2] Expone métricas, hiperparámetros, dataset y versiones del modelo activo."""
    bundle = _get_bundle(request)
    return {
        "active_version": bundle.version,
        "sha256_verified": bundle.sha256_verified,
        "loaded_at_utc": bundle.loaded_at_utc,
        "version_warnings": bundle.version_warnings,
        "metadata": bundle.metadata,
    }


@app.get("/model/versions", tags=["Versionado y Metadatos"])
def list_versions() -> Dict[str, Any]:
    """[Parte 2.6] Lista todas las versiones registradas en `models/registry.json` y su historial."""
    registry = load_registry()
    active, versions = list_registered_versions()
    return {
        "active_version": active,
        "previous_active_version": registry.get("previous_active_version"),
        "versions": versions,
        "history": registry.get("history", [])[-20:],
    }


@app.post(
    "/model/switch",
    tags=["Versionado y Metadatos"],
    dependencies=[Depends(_verify_admin_key)],
)
def switch_version(payload: VersionSwitchRequest, request: Request) -> Dict[str, Any]:
    """[Parte 2.6] Promueve una versión registrada y recarga el modelo en caliente."""
    if not payload.target_version:
        raise HTTPException(status_code=400, detail="Debe especificar `target_version`.")
    try:
        change = switch_active_version(payload.target_version, reason=payload.reason)
        request.app.state.model_bundle = load_active_model(version=payload.target_version)
        return {"status": "switched", **change}
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/model/rollback",
    tags=["Versionado y Metadatos"],
    dependencies=[Depends(_verify_admin_key)],
)
def rollback_version(payload: VersionSwitchRequest, request: Request) -> Dict[str, Any]:
    """[Parte 2.6] Revierte a la versión estable anterior y recarga el modelo en caliente."""
    try:
        change = rollback_model_version(
            target_version=payload.target_version,
            reason=payload.reason,
        )
        request.app.state.model_bundle = load_active_model(version=change["active_version"])
        return {"status": "rolled_back", **change}
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/monitoring/summary", tags=["Monitoreo"])
def monitoring_summary(last_n: int = Query(default=1000, ge=10, le=10000)) -> Dict[str, Any]:
    """[Parte 1.6] Retorna estadísticas de latencia, calidad y detección de data/prediction drift."""
    return build_monitoring_report(last_n=last_n)
