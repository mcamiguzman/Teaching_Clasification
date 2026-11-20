"""
CLI de gestión, actualización y reversión (rollback) de versiones del modelo.

Implementa:
- [Parte 2.6 — Control y reversión de versiones]:
    Permite auditar todas las versiones registradas, inspeccionar sus métricas y
    versiones de librerías, verificar integridad SHA-256, promover una nueva versión
    o revertir inmediatamente a la versión estable anterior.

Ejemplos de uso:
    python scripts/manage_versions.py list
    python scripts/manage_versions.py verify --version v1.0.0
    python scripts/manage_versions.py switch v1.1.0 --reason "Mejora en Macro-F1"
    python scripts/manage_versions.py rollback --reason "Regresión detectada en producción"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_registry import (
    list_registered_versions,
    load_active_model,
    load_registry,
    rollback_model_version,
    switch_active_version,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI de control y reversión de versiones del modelo [Parte 2.6]."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Lista todas las versiones registradas y la versión activa.")

    p_verify = sub.add_parser("verify", help="Verifica integridad SHA-256 y carga segura.")
    p_verify.add_argument("--version", default=None, help="Versión a verificar (por defecto activa).")

    p_switch = sub.add_parser("switch", help="Activa una versión específica del registro.")
    p_switch.add_argument("target_version", help="Versión destino (ej. v1.1.0).")
    p_switch.add_argument("--reason", default="Promoción controlada desde CLI.")

    p_rb = sub.add_parser("rollback", help="Revierte a la versión estable anterior o indicada.")
    p_rb.add_argument("--version", default=None, help="Versión destino opcional.")
    p_rb.add_argument("--reason", default="Reversión controlada desde CLI.")

    sub.add_parser("history", help="Muestra el historial completo de cambios y reversiones.")

    args = parser.parse_args()

    if args.command == "list":
        active, versions = list_registered_versions()
        print(json.dumps({"active_version": active, "versions": versions}, ensure_ascii=False, indent=2))

    elif args.command == "verify":
        bundle = load_active_model(version=args.version)
        print(
            json.dumps(
                {
                    "status": "VERIFIED",
                    "version": bundle.version,
                    "sha256_verified": bundle.sha256_verified,
                    "pipeline_steps": list(bundle.pipeline.named_steps.keys()),
                    "version_warnings": bundle.version_warnings,
                    "metrics": bundle.metadata.get("metricas", {}),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    elif args.command == "switch":
        result = switch_active_version(args.target_version, reason=args.reason)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "rollback":
        result = rollback_model_version(target_version=args.version, reason=args.reason)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "history":
        reg = load_registry()
        print(json.dumps(reg.get("history", []), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
