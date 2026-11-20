"""
Aplicación web institucional con Streamlit para clasificación de retroalimentación estudiantil.

Implementa explícitamente:
- [Parte 1.5 — Integrar con la aplicación]:
    * Interfaz institucional con ingreso de comentarios individuales Y carga masiva
      de archivos CSV con descarga de resultados enriquecidos.
    * Soporta dos modos de conexión configurables en la barra lateral:
        1. **API REST FastAPI** (`http://127.0.0.1:8000`) — arquitectura desacoplada recomendada.
        2. **Carga local directa** (`.joblib` cacheado con `@st.cache_resource`) — útil
           para demostración rápida sin levantar dos terminales.
- [Parte 2.2 — Eficiencia de entrenamiento]:
    * Usa `@st.cache_resource` o la API ya inicializada; JAMÁS reentrena el modelo.
- [Parte 2.5 — Salida estructurada]:
    * Muestra de forma visual y tabular la predicción estructurada para cada uno de
      los 9 aspectos docentes (`contenidos`, `metodologia_docente`, `evaluaciones_parciales`,
      `material_clase`, `retroalimentacion`, `organizacion_horario`, `disponibilidad_docente`,
      `recursos_aula_digital`, `dificultad_ritmo`).
- [Parte 1.6 & Parte 2.6]:
    * Pestañas dedicadas para auditoría de métricas/versiones (incluyendo reversión /
      rollback) y monitoreo de latencia y data/prediction drift (PSI).
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

# Asegurar que la raíz del proyecto esté en sys.path al lanzar `streamlit run app/streamlit_app.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ASPECT_LABELS_ES, NINE_ASPECTS, POLARITY_CLASSES
from src.model_registry import (
    list_registered_versions,
    load_active_model,
    load_registry,
    rollback_model_version,
    switch_active_version,
)
from src.monitoring import build_monitoring_report, log_prediction_event
from src.validate_model import predecir_seguro

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

EJEMPLOS_PARCIAL: Dict[str, str] = {
    "Positivo claro": (
        "Excelente curso, el profesor explica muy bien, las diapositivas son claras "
        "y los ejemplos prácticos ayudaron mucho."
    ),
    "Negativo claro": (
        "Muy mal organizado, las evaluaciones no tienen sentido, la retroalimentación "
        "fue confusa y nadie responde dudas por correo."
    ),
    "Contrastante / Ambiguo (ID 9 del Parcial)": (
        "El contenido fue amplio y útil, pero la disponibilidad de recursos en el aula "
        "digital era limitada. Me siento sorprendido por la falta de soporte."
    ),
    "Concesiva positiva (ID 3 del Parcial)": (
        "El contenido fue bastante interesante, aunque algunas partes pudieron haber sido "
        "más detalladas para facilitar comprensión."
    ),
    "Alargamientos + emojis": (
        "Suuuuper bueeeeno el curso, la metodología fue exceleenteee y el docente muy amable 😊"
    ),
    "Caso borde: solo emojis": "😀😀😀",
    "Caso borde: texto vacío": "   ",
}

COLOR_MAP = {
    "positivo": "#1f8f4e",
    "neutral": "#c78b1b",
    "negativo": "#c93b3b",
    "no_mencionado": "#7a869a",
}


@st.cache_resource(show_spinner="Cargando Pipeline unificado (.joblib) con verificación SHA-256...")
def _get_local_bundle(version_key: str):
    """[Parte 2.2 & 2.3] Cachea el modelo ya entrenado sin reentrenar."""
    target = None if version_key == "__active__" else version_key
    return load_active_model(version=target)


def _check_api_health(api_url: str) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(f"{api_url.rstrip('/')}/health", timeout=2.5)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        return None
    return None


def _predict_single(
    comentario: str,
    curso: Optional[str],
    incluir_aspectos: bool,
    mode: str,
    api_url: str,
    active_version_key: str,
) -> Dict[str, Any]:
    if mode == "API FastAPI":
        resp = requests.post(
            f"{api_url.rstrip('/')}/predict",
            json={
                "comentario": comentario,
                "curso": curso or None,
                "incluir_aspectos": incluir_aspectos,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    bundle = _get_local_bundle(active_version_key)
    t0 = time.perf_counter()
    res = predecir_seguro(comentario, bundle.pipeline, incluir_aspectos=incluir_aspectos)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    log_prediction_event(
        comment_text=comentario,
        predicted_class=res.get("clase"),
        probabilities=res.get("probabilidades", {}),
        confidence=float(res.get("confianza", 0.0)),
        latency_ms=latency_ms,
        model_version=bundle.version,
        is_contrastive=bool(res.get("es_contrastante", False)),
        source="streamlit_local",
        course=curso or None,
        error=res.get("error"),
    )
    res["model_version"] = bundle.version
    res["latency_ms"] = latency_ms
    return res


def _render_polarity_badge(clase: str, confianza: float) -> None:
    color = COLOR_MAP.get(clase, "#4a5568")
    st.markdown(
        f"""
        <div style="padding:14px 18px;border-radius:10px;background:{color}18;
                    border-left:6px solid {color};margin-bottom:12px;">
            <span style="font-size:0.9rem;color:#555;">Polaridad global predicha</span><br/>
            <span style="font-size:1.6rem;font-weight:700;color:{color};text-transform:uppercase;">
                {clase}
            </span>
            <span style="font-size:1.0rem;margin-left:12px;color:#333;">
                Confianza: <b>{confianza * 100:.1f}%</b>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Evaluación Docente — Clasificador de Retroalimentación",
        page_icon="🎓",
        layout="wide",
    )

    st.title("🎓 Análisis de Retroalimentación Estudiantil (SET)")
    st.caption(
        "Sistema productivo basado en el proyecto parcial de Sistemas Inteligentes "
        "(Pipeline unificado Joblib · FastAPI · Salida estructurada de 9 aspectos · Versionado y Drift)"
    )

    # -------------------------------------------------------------------------
    # Barra lateral: selección de modo (API FastAPI vs Local) y estado del modelo
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("Configuración del Servicio")
        api_url = st.text_input("URL de la API FastAPI", value=DEFAULT_API_URL)
        api_health = _check_api_health(api_url)

        default_mode_idx = 0 if api_health else 1
        mode = st.radio(
            "Modo de inferencia",
            options=["API FastAPI", "Local (.joblib cacheado)"],
            index=default_mode_idx,
            help="Usa la API FastAPI si está levantada, o carga el .joblib local con verificación SHA-256.",
        )

        if api_health:
            st.success(f"API conectada · Modelo `{api_health.get('active_model_version')}`")
        else:
            st.info("API no detectada en esa URL; puedes usar el modo Local o iniciar `uvicorn api.main:app`.")

        active_ver, all_versions = list_registered_versions()
        st.markdown("---")
        st.metric("Versión activa registrada", active_ver or "Sin registrar")
        st.metric("Versiones en registro", len(all_versions))

        admin_api_key = st.text_input(
            "X-API-Key (admin opcional)",
            type="password",
            help="Solo necesaria al cambiar o revertir versiones mediante la API si ADMIN_API_KEY está activa.",
        )

    active_version_key = active_ver or "__active__"

    tab_single, tab_csv, tab_versions, tab_monitor = st.tabs(
        [
            "💬 Comentario Individual",
            "📄 Carga Masiva (CSV)",
            "📦 Modelo y Versiones (Rollback)",
            "📈 Monitoreo y Drift",
        ]
    )

    # =========================================================================
    # TAB 1: Comentario Individual + Salida estructurada de 9 aspectos
    # =========================================================================
    with tab_single:
        st.subheader("Clasificación de comentario individual")
        col_ex, col_course = st.columns([2, 1])
        with col_ex:
            ejemplo = st.selectbox(
                "Cargar un ejemplo rápido del parcial / caso borde",
                options=["(Escribir manualmente)"] + list(EJEMPLOS_PARCIAL.keys()),
            )
        with col_course:
            curso_input = st.selectbox(
                "Asignatura (opcional)",
                options=[
                    "",
                    "Sistemas operativos",
                    "Arquitectura del computador",
                    "Programacion paralela",
                    "Sistemas distribuidos",
                    "Sistemas Inteligentes",
                    "Fundamentos de programación",
                    "Bases de datos",
                ],
            )

        texto_inicial = "" if ejemplo == "(Escribir manualmente)" else EJEMPLOS_PARCIAL[ejemplo]
        comentario = st.text_area(
            "Ingresa el comentario del estudiante:",
            value=texto_inicial,
            height=120,
            placeholder="Ej. El contenido fue excelente y actualizado, pero los parciales fueron muy confusos...",
        )
        incluir_aspectos = st.checkbox(
            "Incluir salida estructurada de los 9 aspectos docentes [Parte 2.5]",
            value=True,
        )

        if st.button("Analizar comentario", type="primary", use_container_width=False):
            try:
                res = _predict_single(
                    comentario=comentario,
                    curso=curso_input,
                    incluir_aspectos=incluir_aspectos,
                    mode=mode,
                    api_url=api_url,
                    active_version_key=active_version_key,
                )
            except Exception as exc:
                st.error(f"Error al ejecutar la predicción: {exc}")
            else:
                if not res.get("valido"):
                    st.warning(f"Entrada controlada sin clasificación: {res.get('error')}")
                else:
                    col_left, col_right = st.columns([1.2, 1])
                    with col_left:
                        _render_polarity_badge(res["clase"], float(res["confianza"]))
                        st.caption(
                            f"Modelo `{res['model_version']}` · Latencia: `{res['latency_ms']} ms` · "
                            f"Cláusula contrastante: `{'Sí' if res['es_contrastante'] else 'No'}`"
                        )
                        st.markdown(f"**Texto normalizado (dentro del Pipeline):** `{res['texto_normalizado']}`")
                        for adv in res.get("advertencias", []):
                            st.info(adv)

                    with col_right:
                        probs_df = pd.DataFrame(
                            [
                                {"Clase": c, "Probabilidad": float(res["probabilidades"].get(c, 0.0))}
                                for c in POLARITY_CLASSES
                            ]
                        ).set_index("Clase")
                        st.markdown("**Probabilidades calibradas por clase**")
                        st.bar_chart(probs_df)

                    # [Parte 2.5] Vista estructurada de los 9 aspectos
                    if incluir_aspectos and res.get("aspectos"):
                        st.markdown("---")
                        st.subheader("Salida estructurada por los 9 aspectos docentes [Parte 2.5]")
                        aspect_rows = []
                        for asp_key in NINE_ASPECTS:
                            asp_data = res["aspectos"][asp_key]
                            aspect_rows.append(
                                {
                                    "Aspecto (ID)": asp_key,
                                    "Dimensión Docente": asp_data["nombre_legible"],
                                    "Mencionado": "Sí" if asp_data["mencionado"] else "No",
                                    "Polaridad": asp_data["polaridad"],
                                    "Confianza": asp_data["confianza"],
                                    "Evidencia detectada": asp_data.get("evidencia") or "—",
                                }
                            )
                        st.dataframe(pd.DataFrame(aspect_rows), use_container_width=True, hide_index=True)

                        with st.expander("Ver JSON estructurado completo (contrato API)"):
                            st.json(res)

    # =========================================================================
    # TAB 2: Carga Masiva de CSV [Parte 1.5]
    # =========================================================================
    with tab_csv:
        st.subheader("Clasificación masiva mediante archivo CSV")
        st.write(
            "Carga un archivo CSV con comentarios estudiantiles (por ejemplo, encuestas "
            "de evaluación docente al cierre del semestre) para clasificar cada registro."
        )

        # Ofrecer un CSV de ejemplo descargable directamente desde la UI
        sample_df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "curso": "Sistemas operativos",
                    "feedback": "Excelente profesor, explica muy claro y los laboratorios ayudan bastante.",
                },
                {
                    "id": 2,
                    "curso": "Bases de datos",
                    "feedback": "El contenido fue amplio y útil, pero los parciales fueron confusos y sin retroalimentación.",
                },
                {
                    "id": 3,
                    "curso": "Sistemas Inteligentes",
                    "feedback": "La metodología fue regular y el horario se mantuvo dentro de lo esperado.",
                },
                {
                    "id": 4,
                    "curso": "Programacion paralela",
                    "feedback": "Muy desorganizado, el aula digital falló varias veces y nadie respondió los correos.",
                },
            ]
        )
        st.download_button(
            "Descargar CSV de plantilla de ejemplo",
            data=sample_df.to_csv(index=False).encode("utf-8"),
            file_name="ejemplo_feedback_estudiantil.csv",
            mime="text/csv",
        )

        uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df_in = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f"Error al leer el archivo CSV: {exc}")
            else:
                st.write(f"Filas detectadas: **{len(df_in)}** · Columnas: `{list(df_in.columns)}`")
                default_col = "feedback" if "feedback" in df_in.columns else df_in.columns[0]
                text_col = st.selectbox(
                    "Selecciona la columna que contiene los comentarios:",
                    options=list(df_in.columns),
                    index=list(df_in.columns).index(default_col),
                )
                incluir_asp_csv = st.checkbox(
                    "Extraer los 9 aspectos por cada fila del CSV",
                    value=False,
                    key="csv_aspects",
                )

                if st.button("Clasificar archivo CSV", type="primary"):
                    resultados_csv: List[Dict[str, Any]] = []
                    progress = st.progress(0)
                    total_rows = len(df_in)
                    for i, row in df_in.iterrows():
                        texto = "" if pd.isna(row[text_col]) else str(row[text_col])
                        curso_val = (
                            str(row["curso"])
                            if "curso" in df_in.columns and not pd.isna(row["curso"])
                            else None
                        )
                        pred = _predict_single(
                            comentario=texto,
                            curso=curso_val,
                            incluir_aspectos=incluir_asp_csv,
                            mode=mode,
                            api_url=api_url,
                            active_version_key=active_version_key,
                        )
                        fila_out = {
                            "clase_predicha": pred.get("clase") or "ERROR",
                            "confianza": pred.get("confianza", 0.0),
                            "prob_negativo": pred.get("probabilidades", {}).get("negativo", 0.0),
                            "prob_neutral": pred.get("probabilidades", {}).get("neutral", 0.0),
                            "prob_positivo": pred.get("probabilidades", {}).get("positivo", 0.0),
                            "es_contrastante": pred.get("es_contrastante", False),
                            "error": pred.get("error") or "",
                        }
                        if incluir_asp_csv and pred.get("aspectos"):
                            for asp_key in NINE_ASPECTS:
                                fila_out[f"aspecto_{asp_key}"] = pred["aspectos"][asp_key]["polaridad"]
                        resultados_csv.append(fila_out)
                        progress.progress((i + 1) / max(total_rows, 1))

                    df_out = pd.concat([df_in.reset_index(drop=True), pd.DataFrame(resultados_csv)], axis=1)
                    st.success(f"Clasificación completada sobre {len(df_out)} registros.")

                    c1, c2 = st.columns([1, 1.5])
                    with c1:
                        st.markdown("**Distribución de polaridad predicha**")
                        dist = df_out["clase_predicha"].value_counts()
                        st.bar_chart(dist)
                    with c2:
                        st.markdown("**Vista previa de resultados**")
                        st.dataframe(df_out.head(25), use_container_width=True)

                    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Descargar resultados clasificados (.csv)",
                        data=csv_bytes,
                        file_name="feedback_clasificado_resultados.csv",
                        mime="text/csv",
                    )

    # =========================================================================
    # TAB 3: Métricas del modelo, compatibilidad y Control/Reversión [Parte 1.1, 2.4, 2.6]
    # =========================================================================
    with tab_versions:
        st.subheader("Trazabilidad del modelo, compatibilidad de versiones y rollback")
        registry = load_registry()
        versions_list = list(registry.get("versions", {}).values())

        if not versions_list:
            st.warning("Aún no hay versiones registradas. Ejecuta `python -m src.train_and_select`.")
        else:
            bundle = _get_local_bundle(active_version_key)
            meta = bundle.metadata
            m = meta.get("metricas", {})

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Modelo ganador", meta.get("modelo", "N/A"))
            k2.metric("Macro-F1 CV (train)", f"{m.get('macro_f1_cv_train', 0):.4f}")
            k3.metric("Macro-F1 Test", f"{m.get('macro_f1_test', 0):.4f}")
            k4.metric("Accuracy Test", f"{m.get('accuracy_test', 0):.4f}")

            st.markdown(
                f"**Versión dataset:** `{meta.get('dataset', {}).get('dataset_version')}` "
                f"(`{meta.get('dataset', {}).get('source_mode')}`) · "
                f"**SHA-256 Joblib verificado:** `{'Sí' if bundle.sha256_verified else 'No'}` (`{meta.get('artifact_sha256', '')[:16]}...`)"
            )

            col_p, col_v = st.columns(2)
            with col_p:
                st.markdown("**Hiperparámetros y pasos del Pipeline unificado [Parte 2.1]**")
                st.json(
                    {
                        "pipeline_steps": meta.get("pipeline_steps"),
                        "calibrado": meta.get("calibrado_con_CalibratedClassifierCV"),
                        "parametros": meta.get("parametros"),
                    }
                )
            with col_v:
                st.markdown("**Versiones de Python y dependencias [Parte 1.2 & 2.4]**")
                st.json(meta.get("versiones", {}))

            st.markdown("---")
            st.markdown("**Registro de versiones disponibles [Parte 2.6]**")
            st.dataframe(pd.DataFrame(versions_list), use_container_width=True, hide_index=True)

            col_sw, col_rb = st.columns(2)
            with col_sw:
                target_ver = st.selectbox(
                    "Seleccionar versión para activar",
                    options=[v["version"] for v in versions_list],
                )
                reason_sw = st.text_input("Motivo del cambio", value="Cambio controlado desde interfaz")
                if st.button("Activar versión seleccionada"):
                    try:
                        if mode == "API FastAPI" and api_health:
                            headers = {"X-API-Key": admin_api_key} if admin_api_key else {}
                            r = requests.post(
                                f"{api_url.rstrip('/')}/model/switch",
                                json={"target_version": target_ver, "reason": reason_sw},
                                headers=headers,
                                timeout=10,
                            )
                            r.raise_for_status()
                            st.success(f"Versión cambiada vía API: {r.json()}")
                        else:
                            res_sw = switch_active_version(target_ver, reason=reason_sw)
                            _get_local_bundle.clear()
                            st.success(f"Versión activada localmente: {res_sw}")
                    except Exception as exc:
                        st.error(f"No se pudo cambiar la versión: {exc}")

            with col_rb:
                prev_ver = registry.get("previous_active_version")
                st.write(f"Versión estable anterior disponible para reversión: **{prev_ver or 'Ninguna'}**")
                reason_rb = st.text_input("Motivo de reversión (rollback)", value="Reversión preventiva")
                if st.button("Ejecutar Rollback a versión anterior", disabled=prev_ver is None):
                    try:
                        if mode == "API FastAPI" and api_health:
                            headers = {"X-API-Key": admin_api_key} if admin_api_key else {}
                            r = requests.post(
                                f"{api_url.rstrip('/')}/model/rollback",
                                json={"reason": reason_rb},
                                headers=headers,
                                timeout=10,
                            )
                            r.raise_for_status()
                            st.success(f"Rollback completado vía API: {r.json()}")
                        else:
                            res_rb = rollback_model_version(reason=reason_rb)
                            _get_local_bundle.clear()
                            st.success(f"Rollback completado localmente: {res_rb}")
                    except Exception as exc:
                        st.error(f"No se pudo ejecutar el rollback: {exc}")

    # =========================================================================
    # TAB 4: Monitoreo de latencia y Drift [Parte 1.6]
    # =========================================================================
    with tab_monitor:
        st.subheader("Monitoreo de inferencia, latencia y cambios en los datos (Data/Prediction Drift)")
        if st.button("Actualizar reporte de monitoreo"):
            st.rerun()

        report = build_monitoring_report()
        if report["total_predictions_logged"] == 0:
            st.info("Todavía no hay inferencias registradas. Clasifica algún comentario para ver métricas.")
        else:
            lat = report["latency_ms"]
            drift = report["drift"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Inferencias registradas", report["total_predictions_logged"])
            c2.metric("Latencia p50 / p95 (ms)", f"{lat.get('p50', 0):.2f} / {lat.get('p95', 0):.2f}")
            c3.metric("PSI (Deriva de clases)", f"{drift.get('psi_class_distribution', 0):.4f}")
            c4.metric("Estado de Drift", str(drift.get("drift_alert", "N/A")).upper())

            baseline = report.get("baseline")
            prod = report.get("production_stats", {})
            if baseline:
                comp_df = pd.DataFrame(
                    {
                        "Entrenamiento (Baseline)": baseline.get("class_proportions", {}),
                        "Producción (Actual)": prod.get("class_proportions", {}),
                    }
                )
                st.markdown("**Comparación de distribución de polaridad: Entrenamiento vs Producción**")
                st.bar_chart(comp_df)

            st.markdown("**Últimos eventos de inferencia auditados (anonimizados con hash)**")
            st.dataframe(pd.DataFrame(report.get("recent_events", [])), use_container_width=True)


if __name__ == "__main__":
    main()
