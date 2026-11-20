"""
Registro versionado de modelos con carga segura, verificación de versiones y rollback.

Implementa explícitamente:
- [Parte 2.2 — Eficiencia de entrenamiento]: carga el artefacto `.joblib` ya entrenado
  con caché en memoria; nunca reentrena el modelo al iniciar la API ni la app web.
- [Parte 2.3 — Seguridad en deserialización]:
    * Verifica que la ruta del archivo `.joblib` resida estrictamente dentro del
      directorio confiable `models/` (previene path traversal y carga de archivos externos).
    * Calcula y compara la huella criptográfica SHA-256 contra `checksum.sha256` y
      `metadata.json` ANTES de invocar `joblib.load()`.
    * Documenta y mitiga el riesgo de ejecución de código arbitrario vía pickle/joblib.
- [Parte 2.4 — Compatibilidad de versiones]:
    * Registra versiones exactas de Python, scikit-learn, joblib, pandas, numpy y scipy.
    * Compara las versiones del entorno de ejecución actual contra las de entrenamiento
      y reporta advertencias (o bloquea en modo estricto `STRICT_VERSION_CHECK=true`).
- [Parte 2.6 — Control y reversión de versiones]:
    * Mantiene `models/registry.json` con todas las versiones publicadas, la versión
      activa, la versión anterior (`previous_active_version`) y el historial completo
      de registros, promociones y reversiones (`rollback`).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from src.config import MODELS_DIR, PROJECT_ROOT, REGISTRY_PATH, STRICT_VERSION_CHECK

logger = logging.getLogger(__name__)


class UntrustedModelError(RuntimeError):
    """Error lanzado cuando un archivo `.joblib` no supera las verificaciones de confianza/integridad."""


class VersionCompatibilityError(RuntimeError):
    """Error lanzado en modo estricto cuando las versiones de entrenamiento y producción difieren."""


@dataclass
class LoadedModelBundle:
    """Contenedor inmutable en memoria con el Pipeline cargado, metadatos y estado de verificación."""

    version: str
    pipeline: Any
    metadata: Dict[str, Any]
    model_path: Path
    sha256_verified: bool
    version_warnings: List[str]
    loaded_at_utc: str


def get_runtime_versions() -> Dict[str, str]:
    """
    [Parte 1.2 & Parte 2.4]
    Captura las versiones exactas de Python y librerías críticas del entorno actual
    mediante `importlib.metadata` (sin cargar módulos pesados innecesariamente).
    """
    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    for optional_pkg in ("emoji", "fastapi", "pydantic", "streamlit"):
        try:
            versions[optional_pkg] = importlib.metadata.version(optional_pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[optional_pkg] = "not_installed"
    return versions


def compute_sha256(file_path: Path) -> str:
    """Calcula la huella criptográfica SHA-256 de un archivo binario."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_trusted_path(model_path: Path, trusted_root: Path = MODELS_DIR) -> Path:
    """
    [Parte 2.3 — Seguridad en deserialización]
    Garantiza que el archivo `.joblib` pertenece exclusivamente al directorio confiable
    del registro de modelos y tiene extensión `.joblib`. Evita path traversal (`../`)
    o carga accidental de artefactos descargados de fuentes no verificadas.
    """
    resolved_root = trusted_root.resolve()
    resolved_target = model_path.resolve()

    if not resolved_target.exists() or not resolved_target.is_file():
        raise FileNotFoundError(f"No se encontró el artefacto del modelo en: {resolved_target}")

    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise UntrustedModelError(
            f"BLOQUEO DE SEGURIDAD [Parte 2.3]: intento de cargar un archivo Joblib "
            f"fuera del directorio confiable '{resolved_root}': {resolved_target}"
        ) from exc

    if resolved_target.suffix.lower() != ".joblib":
        raise UntrustedModelError(
            f"BLOQUEO DE SEGURIDAD [Parte 2.3]: extensión no permitida '{resolved_target.suffix}'. "
            f"Solo se permiten archivos '.joblib' registrados."
        )

    return resolved_target


def verify_sha256_integrity(
    model_path: Path,
    expected_sha256: Optional[str] = None,
    checksum_file: Optional[Path] = None,
) -> str:
    """
    [Parte 2.3 — Seguridad en deserialización]
    Valida la integridad criptográfica del archivo `.joblib` antes de deserializarlo.
    Si el archivo fue modificado o corrompido respecto al hash registrado durante
    el entrenamiento, aborta la carga para impedir deserialización manipulada.
    """
    actual_sha256 = compute_sha256(model_path)

    expected = expected_sha256
    if expected is None and checksum_file is not None and checksum_file.exists():
        raw_line = checksum_file.read_text(encoding="utf-8").strip()
        expected = raw_line.split()[0] if raw_line else None

    if not expected:
        raise UntrustedModelError(
            f"BLOQUEO DE SEGURIDAD [Parte 2.3]: no existe hash SHA-256 de referencia "
            f"para validar '{model_path.name}'. Se rechaza la deserialización."
        )

    if actual_sha256.lower() != expected.lower():
        raise UntrustedModelError(
            f"BLOQUEO DE SEGURIDAD [Parte 2.3]: el hash SHA-256 de '{model_path.name}' "
            f"no coincide con el registro de entrenamiento.\n"
            f"  Esperado: {expected}\n"
            f"  Actual:   {actual_sha256}\n"
            f"Posible manipulación o corrupción del archivo `.joblib`."
        )

    return actual_sha256


def check_version_compatibility(
    trained_versions: Dict[str, str],
    strict: bool = STRICT_VERSION_CHECK,
) -> List[str]:
    """
    [Parte 2.4 — Compatibilidad de versiones]
    Compara las versiones de Python y dependencias entre entrenamiento y producción.
    Especial atención a `scikit-learn` y `joblib`, cuya incompatibilidad puede
    romper la deserialización o alterar silenciosamente el vectorizador/clasificador.
    """
    runtime = get_runtime_versions()
    incompatibilities: List[str] = []

    critical_keys = ("python", "scikit-learn", "joblib", "numpy", "scipy")
    for key in critical_keys:
        v_train = trained_versions.get(key)
        v_run = runtime.get(key)
        if not v_train or not v_run:
            continue

        # Para Python comparamos major.minor; para scikit-learn comparamos exacto
        if key == "python":
            train_mm = ".".join(v_train.split(".")[:2])
            run_mm = ".".join(v_run.split(".")[:2])
            if train_mm != run_mm:
                incompatibilities.append(
                    f"Python entrenamiento={v_train} vs producción={v_run} (difiere major.minor)."
                )
        elif key == "scikit-learn":
            if v_train != v_run:
                incompatibilities.append(
                    f"scikit-learn entrenamiento={v_train} vs producción={v_run}. "
                    f"Se recomienda mantener exactamente la misma versión."
                )
        else:
            train_major = v_train.split(".")[0]
            run_major = v_run.split(".")[0]
            if train_major != run_major:
                incompatibilities.append(
                    f"{key} entrenamiento={v_train} vs producción={v_run} (difiere versión mayor)."
                )

    if incompatibilities:
        msg = "Advertencia de compatibilidad [Parte 2.4]: " + " | ".join(incompatibilities)
        if strict:
            raise VersionCompatibilityError(msg)
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    return incompatibilities


def _default_registry() -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "active_version": None,
        "previous_active_version": None,
        "versions": {},
        "history": [],
    }


def load_registry(registry_path: Path = REGISTRY_PATH) -> Dict[str, Any]:
    """Lee el registro de versiones `models/registry.json`."""
    if not registry_path.exists():
        return _default_registry()
    return json.loads(registry_path.read_text(encoding="utf-8"))


def save_registry(registry: Dict[str, Any], registry_path: Path = REGISTRY_PATH) -> None:
    """Persiste `models/registry.json` de forma atómica."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = registry_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(registry_path)


def register_model_version(
    version: str,
    pipeline: Any,
    metadata: Dict[str, Any],
    activate: bool = True,
    notes: str = "",
    models_dir: Path = MODELS_DIR,
    registry_path: Path = REGISTRY_PATH,
) -> Dict[str, Any]:
    """
    [Parte 1.2 & Parte 2.6]
    Guarda un Pipeline completo en `models/<version>/modelo_polaridad.joblib`,
    genera su checksum SHA-256, escribe `metadata.json` y actualiza `registry.json`.
    """
    version_dir = models_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / "modelo_polaridad.joblib"
    checksum_path = version_dir / "checksum.sha256"
    meta_path = version_dir / "metadata.json"

    # [Parte 1.2] Serializar Pipeline unificado con compresión moderada
    joblib.dump(pipeline, model_path, compress=3)
    sha256 = compute_sha256(model_path)
    checksum_path.write_text(f"{sha256}  {model_path.name}\n", encoding="utf-8")

    now_utc = datetime.now(timezone.utc).isoformat()
    enriched_meta = dict(metadata)
    enriched_meta.update(
        {
            "model_version": version,
            "registered_at_utc": now_utc,
            "artifact_filename": model_path.name,
            "artifact_sha256": sha256,
            "artifact_size_bytes": model_path.stat().st_size,
            "notes": notes,
        }
    )
    meta_path.write_text(
        json.dumps(enriched_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    registry = load_registry(registry_path)
    previous_active = registry.get("active_version")

    registry["versions"][version] = {
        "version": version,
        "artifact_path": str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "metadata_path": str(meta_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "checksum_path": str(checksum_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "sha256": sha256,
        "registered_at_utc": now_utc,
        "status": "active" if activate else "staged",
        "model_family": enriched_meta.get("modelo"),
        "dataset_version": enriched_meta.get("dataset", {}).get("dataset_version"),
        "macro_f1_cv_train": enriched_meta.get("metricas", {}).get("macro_f1_cv_train"),
        "macro_f1_test": enriched_meta.get("metricas", {}).get("macro_f1_test"),
        "accuracy_test": enriched_meta.get("metricas", {}).get("accuracy_test"),
        "notes": notes,
    }

    if activate:
        if previous_active and previous_active != version and previous_active in registry["versions"]:
            registry["versions"][previous_active]["status"] = "archived"
            registry["previous_active_version"] = previous_active
        registry["active_version"] = version

    registry["history"].append(
        {
            "action": "register_and_activate" if activate else "register_staged",
            "version": version,
            "previous_version": previous_active,
            "timestamp_utc": now_utc,
            "notes": notes,
        }
    )
    save_registry(registry, registry_path)
    return enriched_meta


def switch_active_version(
    target_version: str,
    reason: str = "Actualización controlada de versión",
    registry_path: Path = REGISTRY_PATH,
    models_dir: Path = MODELS_DIR,
) -> Dict[str, Any]:
    """
    [Parte 2.6 — Control de versiones]
    Promueve o cambia la versión activa del modelo verificando previamente que
    el artefacto exista y pase la validación de integridad SHA-256.
    """
    registry = load_registry(registry_path)
    if target_version not in registry["versions"]:
        raise KeyError(
            f"La versión '{target_version}' no existe en el registro. "
            f"Disponibles: {list(registry['versions'].keys())}"
        )

    entry = registry["versions"][target_version]
    model_path = (PROJECT_ROOT / entry["artifact_path"]).resolve()
    verify_trusted_path(model_path, trusted_root=models_dir)
    verify_sha256_integrity(model_path, expected_sha256=entry["sha256"])

    current_active = registry.get("active_version")
    if current_active and current_active != target_version and current_active in registry["versions"]:
        registry["versions"][current_active]["status"] = "archived"
        registry["previous_active_version"] = current_active

    registry["versions"][target_version]["status"] = "active"
    registry["active_version"] = target_version

    now_utc = datetime.now(timezone.utc).isoformat()
    registry["history"].append(
        {
            "action": "switch_version",
            "from_version": current_active,
            "to_version": target_version,
            "timestamp_utc": now_utc,
            "reason": reason,
        }
    )
    save_registry(registry, registry_path)
    return {
        "previous_version": current_active,
        "active_version": target_version,
        "timestamp_utc": now_utc,
        "reason": reason,
    }


def rollback_model_version(
    target_version: Optional[str] = None,
    reason: str = "Reversión de emergencia a versión estable anterior",
    registry_path: Path = REGISTRY_PATH,
    models_dir: Path = MODELS_DIR,
) -> Dict[str, Any]:
    """
    [Parte 2.6 — Control y reversión de versiones]
    Revierte el servicio a la versión activa anterior (`previous_active_version`)
    o a una `target_version` específica validando integridad antes de activar.
    """
    registry = load_registry(registry_path)
    current_active = registry.get("active_version")
    destination = target_version or registry.get("previous_active_version")

    if not destination:
        raise ValueError(
            "No existe una versión anterior registrada para realizar rollback. "
            "Registra al menos dos versiones o indica explícitamente `target_version`."
        )
    if destination not in registry["versions"]:
        raise KeyError(f"La versión destino '{destination}' no existe en el registro.")

    entry = registry["versions"][destination]
    model_path = (PROJECT_ROOT / entry["artifact_path"]).resolve()
    verify_trusted_path(model_path, trusted_root=models_dir)
    verify_sha256_integrity(model_path, expected_sha256=entry["sha256"])

    if current_active and current_active != destination and current_active in registry["versions"]:
        registry["versions"][current_active]["status"] = "rolled_back"
        registry["previous_active_version"] = current_active

    registry["versions"][destination]["status"] = "active"
    registry["active_version"] = destination

    now_utc = datetime.now(timezone.utc).isoformat()
    registry["history"].append(
        {
            "action": "rollback",
            "from_version": current_active,
            "to_version": destination,
            "timestamp_utc": now_utc,
            "reason": reason,
        }
    )
    save_registry(registry, registry_path)
    return {
        "rolled_back_from": current_active,
        "active_version": destination,
        "timestamp_utc": now_utc,
        "reason": reason,
    }


def load_active_model(
    version: Optional[str] = None,
    strict_version_check: bool = STRICT_VERSION_CHECK,
    registry_path: Path = REGISTRY_PATH,
    models_dir: Path = MODELS_DIR,
) -> LoadedModelBundle:
    """
    [Parte 2.2, 2.3, 2.4]
    Carga de forma segura la versión activa (o una versión específica) desde `.joblib`:
      1. Resuelve la ruta dentro del directorio confiable `models/`.
      2. Verifica el hash SHA-256 ANTES de llamar a `joblib.load()`.
      3. Verifica compatibilidad de versiones de Python/scikit-learn/joblib.
      4. Deserializa el Pipeline unificado y retorna un `LoadedModelBundle`.
    """
    registry = load_registry(registry_path)
    chosen_version = version or registry.get("active_version")
    if not chosen_version:
        raise FileNotFoundError(
            "No hay ningún modelo registrado/activo en `models/registry.json`. "
            "Ejecuta primero: python -m src.train_and_select"
        )
    if chosen_version not in registry["versions"]:
        raise KeyError(f"Versión '{chosen_version}' no encontrada en `models/registry.json`.")

    entry = registry["versions"][chosen_version]
    model_path = (PROJECT_ROOT / entry["artifact_path"]).resolve()
    meta_path = (PROJECT_ROOT / entry["metadata_path"]).resolve()
    checksum_path = (PROJECT_ROOT / entry["checksum_path"]).resolve()

    # [Parte 2.3] Paso 1: Verificar ruta confiable y extensión .joblib
    trusted_path = verify_trusted_path(model_path, trusted_root=models_dir)

    # [Parte 2.3] Paso 2: Verificar integridad criptográfica SHA-256 antes de deserializar
    verify_sha256_integrity(
        trusted_path,
        expected_sha256=entry.get("sha256"),
        checksum_file=checksum_path,
    )

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    # [Parte 2.4] Paso 3: Comprobar compatibilidad de versiones entrenamiento vs producción
    version_warnings = check_version_compatibility(
        metadata.get("versiones", {}),
        strict=strict_version_check,
    )

    # [Parte 2.3] Paso 4: Solo tras superar todas las verificaciones se invoca joblib.load()
    pipeline = joblib.load(trusted_path)

    return LoadedModelBundle(
        version=chosen_version,
        pipeline=pipeline,
        metadata=metadata,
        model_path=trusted_path,
        sha256_verified=True,
        version_warnings=version_warnings,
        loaded_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def list_registered_versions(registry_path: Path = REGISTRY_PATH) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Retorna `(active_version, lista_de_versiones)` para la API, CLI y UI."""
    registry = load_registry(registry_path)
    return registry.get("active_version"), list(registry.get("versions", {}).values())
