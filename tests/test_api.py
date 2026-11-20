"""
Pruebas end-to-end del servicio REST FastAPI [Parte 1.4, Parte 1.6, Parte 2.5].
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from src.config import NINE_ASPECTS, POLARITY_CLASSES


def test_health_endpoint():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["sha256_verified"] is True
        assert "X-Process-Time-Ms" in resp.headers


def test_predict_individual_y_salida_estructurada_nueve_aspectos():
    """[Parte 1.4 & Parte 2.5] Verifica clase, probabilidades y los 9 aspectos tipados."""
    with TestClient(app) as client:
        resp = client.post(
            "/predict",
            json={
                "comentario": (
                    "El contenido fue amplio y útil, pero la disponibilidad de recursos "
                    "en el aula digital era limitada y los parciales fueron confusos."
                ),
                "curso": "Sistemas distribuidos",
                "incluir_aspectos": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valido"] is True
        assert data["clase"] in POLARITY_CLASSES
        assert abs(sum(data["probabilidades"].values()) - 1.0) < 0.01
        assert data["es_contrastante"] is True

        # Contrato estricto de 9 aspectos [Parte 2.5]
        aspectos = data["aspectos"]
        assert set(aspectos.keys()) == set(NINE_ASPECTS)
        assert aspectos["contenidos"]["mencionado"] is True
        assert aspectos["recursos_aula_digital"]["mencionado"] is True
        assert aspectos["evaluaciones_parciales"]["mencionado"] is True
        assert aspectos["organizacion_horario"]["polaridad"] == "no_mencionado"


def test_predict_texto_vacio_controlado():
    with TestClient(app) as client:
        resp = client.post("/predict", json={"comentario": "    "})
        assert resp.status_code == 200
        data = resp.json()
        assert data["valido"] is False
        assert data["clase"] is None
        assert data["error"] is not None


def test_predict_batch_y_csv():
    with TestClient(app) as client:
        resp_batch = client.post(
            "/predict/batch",
            json={
                "comentarios": [
                    "Excelente curso y metodología muy clara.",
                    "Pésima organización y evaluaciones confusas.",
                    "",
                ],
                "incluir_aspectos": False,
            },
        )
        assert resp_batch.status_code == 200
        b = resp_batch.json()
        assert b["total"] == 3
        assert b["validos"] == 2
        assert b["con_error"] == 1

        csv_content = (
            "id,curso,feedback\n"
            "1,Sistemas operativos,Excelente profesor y muy buenos ejemplos.\n"
            "2,Bases de datos,El aula digital falló y los parciales fueron muy confusos.\n"
        )
        resp_csv = client.post(
            "/predict/csv",
            files={"file": ("muestra.csv", csv_content.encode("utf-8"), "text/csv")},
        )
        assert resp_csv.status_code == 200
        c = resp_csv.json()
        assert c["total_filas"] == 2
        assert len(c["resultados"]) == 2


def test_model_info_versions_y_monitoring_summary():
    with TestClient(app) as client:
        r_info = client.get("/model/info")
        assert r_info.status_code == 200
        assert "metadata" in r_info.json()

        r_ver = client.get("/model/versions")
        assert r_ver.status_code == 200
        assert r_ver.json()["active_version"] is not None

        r_mon = client.get("/monitoring/summary")
        assert r_mon.status_code == 200
        assert "drift" in r_mon.json()
