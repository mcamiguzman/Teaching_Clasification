# Clasificación de Retroalimentación Estudiantil Universitaria — Proyecto Productivo (MLOps)

Proyecto de ingeniería de aprendizaje automático desarrollado a partir de los recursos del trabajo parcial previo de **Sistemas Inteligentes (Pontificia Universidad Javeriana, Cali)**:
- `Informe_Parcial_SisInt.pdf`
- `Parcial_SI_corregido.ipynb`

El proyecto implementa y documenta en el código los **12 lineamientos solicitados** (Parte 1: pasos 1–6 y Parte 2: principios 1–6), desde la selección y serialización del mejor modelo hasta su despliegue con API REST (FastAPI), aplicación web institucional (Streamlit), salida estructurada para 9 aspectos docentes, control/reversión de versiones y monitoreo de drift.

---

## Mapa de trazabilidad (Parte 1 y Parte 2 → Código)

### Parte 1 — Ciclo de vida de despliegue
| Paso | Requisito | Implementación documentada |
|---|---|---|
| **1. Seleccionar el mejor modelo** | Elegir según validación CV y evaluación única en test; registrar métricas, parámetros y versión del dataset. | `src/train_and_select.py`, `src/dataset_builder.py`, `models/<version>/metadata.json`, `data/processed/dataset_manifest.json` |
| **2. Guardar el modelo con Joblib** | Serializar en `.joblib` un `Pipeline` completo (preprocesamiento + vectorizador + clasificador) registrando Python y dependencias. | `src/train_and_select.py`, `src/model_registry.py`, `models/<version>/modelo_polaridad.joblib`, `requirements-lock.txt` |
| **3. Validar la carga y las predicciones** | Cargar en entorno limpio (subproceso aislado), verificar 100% de coherencia en test y probar vacíos, errores, solo emojis y ambiguos. | `src/validate_model.py`, `tests/test_model_loading_security.py` |
| **4. Crear un servicio de predicción** | API REST con FastAPI que carga el modelo una sola vez y retorna clase predicha y probabilidades por clase. | `api/main.py`, `api/schemas.py` (`POST /predict`, `/predict/batch`, `/predict/csv`) |
| **5. Integrar con la aplicación** | Interfaz institucional con Streamlit para comentarios individuales y carga masiva de archivos CSV con descarga de resultados. | `app/streamlit_app.py` |
| **6. Probar, desplegar y monitorear** | Benchmarks de latencia (p50/p95/p99), pruebas de seguridad/calidad, Docker/Compose y monitoreo de drift (PSI). | `scripts/benchmark_and_monitor.py`, `src/monitoring.py`, `Dockerfile`, `docker-compose.yml`, `tests/` |

### Parte 2 — Principios de ingeniería y producción
| Principio | Requisito | Implementación documentada |
|---|---|---|
| **1. Pipeline unificado** | Vectorizador, clasificador y transformación de texto (`TextNormalizer`) juntos en un único `Pipeline` serializado. | `src/preprocessing.py` (`TextNormalizer`), `src/train_and_select.py` |
| **2. Eficiencia de entrenamiento** | No reentrenar el modelo al iniciar la API ni la app; carga única en memoria (`lifespan` / `@st.cache_resource`). | `api/main.py` (`lifespan`), `app/streamlit_app.py` (`@st.cache_resource`) |
| **3. Seguridad en deserialización** | Cargar solo archivos `.joblib` confiables dentro de `models/` verificando integridad criptográfica SHA-256 antes de `joblib.load()`. | `src/model_registry.py` (`verify_trusted_path`, `verify_sha256_integrity`) |
| **4. Compatibilidad de versiones** | Mantener versiones compatibles de Python, `scikit-learn` y dependencias entre entrenamiento y producción, con verificación automática al cargar. | `requirements.txt`, `requirements-lock.txt`, `src/model_registry.py` (`check_version_compatibility`) |
| **5. Salida estructurada (9 aspectos)** | Contrato tipado `NineAspectsOutput` con exactamente una predicción estructurada por cada uno de los 9 aspectos docentes. | `src/aspects.py`, `api/schemas.py` (`AspectPrediction`, `NineAspectsOutput`) |
| **6. Control y reversión de versiones** | Registro versionado (`models/registry.json`) con historial auditable y procedimiento de promoción y reversión (`rollback`). | `src/model_registry.py`, `scripts/manage_versions.py`, `POST /model/switch`, `POST /model/rollback` |

---

## Los 9 aspectos docentes estructurados (`Parte 2.5`)

Cada llamada a `POST /predict` (con `incluir_aspectos=true`) retorna un bloque estructurado con **exactamente estas 9 llaves**:
1. `contenidos` — Contenidos y pertinencia
2. `metodologia_docente` — Metodología y pedagogía docente
3. `evaluaciones_parciales` — Evaluaciones y parciales
4. `material_clase` — Material de clase y diapositivas
5. `retroalimentacion` — Retroalimentación y calificación
6. `organizacion_horario` — Organización y horario del curso
7. `disponibilidad_docente` — Disponibilidad y trato docente
8. `recursos_aula_digital` — Recursos y aula digital
9. `dificultad_ritmo` — Dificultad y ritmo de avance

---

## Estructura del proyecto

```text
Teaching_Clasification/
├── Informe_Parcial_SisInt.pdf        # Informe previo del parcial
├── Parcial_SI_corregido.ipynb        # Cuaderno experimental previo
├── requirements.txt                  # Dependencias compatibles [Parte 2.4]
├── requirements-lock.txt             # Versiones exactas congeladas del entorno
├── Dockerfile / docker-compose.yml   # Despliegue contenerizado [Parte 1.6]
├── data/
│   ├── raw/                          # Colocar aquí los 3 CSV originales si se desean reentrenar
│   └── processed/                    # Dataset limpio + dataset_manifest.json (versión y SHA-256)
├── models/
│   ├── registry.json                 # Registro de versiones, versión activa e historial de rollback
│   └── v1.0.0/                       # Artefacto .joblib, metadata.json, checksum.sha256, test_reference.json
├── monitoring/
│   ├── baseline_stats.json           # Estadísticas base de entrenamiento para detectar drift
│   └── predictions_log.jsonl         # Auditoría anonimizada de inferencias y latencia
├── src/
│   ├── config.py                     # Constantes, 9 aspectos, límites de seguridad y drift
│   ├── preprocessing.py              # TextNormalizer (Pipeline unificado) y limpieza léxica
│   ├── dataset_builder.py            # Carga CSV reales o réplica fiel del parcial + versionado
│   ├── aspects.py                    # Segmentación de cláusulas y predicción de 9 aspectos
│   ├── train_and_select.py           # CV, GridSearchCV, evaluación test, calibración y Joblib
│   ├── model_registry.py             # Carga segura SHA-256, validación versiones, switch y rollback
│   ├── validate_model.py             # Validación en subproceso limpio + casos borde
│   └── monitoring.py                 # Logging de latencia y cálculo de drift (PSI)
├── api/
│   ├── schemas.py                    # Contratos Pydantic (incl. NineAspectsOutput)
│   └── main.py                       # Servicio REST FastAPI
├── app/
│   └── streamlit_app.py              # Interfaz institucional Streamlit (individual + CSV + MLOps)
├── scripts/
│   ├── manage_versions.py            # CLI para listar, verificar, promover y revertir versiones
│   └── benchmark_and_monitor.py      # Pruebas de latencia p50/p95/p99, seguridad y drift
└── tests/                            # Suite automatizada pytest
```

---

## Uso rápido

> **Nota sobre los datos**: Si colocas los tres CSV originales (`feedback_curso_universitario.csv`, `feedback_curso_universitario2.csv`, `feedback_curso_universitario3.csv`) dentro de `data/raw/`, `src/train_and_select.py` los usará automáticamente aplicando exactamente las limpiezas del parcial. Si no están en disco, utiliza una réplica determinista de 618 registros que reproduce la estructura, asignaturas, distribución de clases y cláusulas contrastantes del informe.

### 1. Entrenar, seleccionar y guardar el modelo (`Parte 1.1 & 1.2`)
```powershell
.\.venv\Scripts\python.exe -m src.train_and_select --version v1.0.0 --dataset-version 1.0.0
```

### 2. Validar carga en entorno limpio y casos borde (`Parte 1.3`)
```powershell
.\.venv\Scripts\python.exe -m src.validate_model
```

### 3. Levantar el servicio de predicción FastAPI (`Parte 1.4`)
```powershell
.\.venv\Scripts\uvicorn.exe api.main:app --host 0.0.0.0 --port 8000
```
Documentación interactiva Swagger disponible en: `http://127.0.0.1:8000/docs`

### 4. Levantar la aplicación institucional Streamlit (`Parte 1.5`)
```powershell
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```
Disponible en: `http://127.0.0.1:8501`

### 5. Control y reversión de versiones (`Parte 2.6`)
```powershell
# Registrar una segunda versión candidata (ej. LogisticRegression)
.\.venv\Scripts\python.exe -m src.train_and_select --version v1.1.0 --clf-family logistic_regression --notes "Candidata alternativa"

# Listar versiones
.\.venv\Scripts\python.exe scripts/manage_versions.py list

# Revertir a la versión estable anterior (rollback)
.\.venv\Scripts\python.exe scripts/manage_versions.py rollback --reason "Volver a v1.0.0 estable"
```

### 6. Pruebas automatizadas, benchmark y despliegue Docker (`Parte 1.6`)
```powershell
# Ejecutar suite completa de pruebas
.\.venv\Scripts\pytest.exe

# Medir latencias p50/p95/p99, seguridad y drift
.\.venv\Scripts\python.exe scripts/benchmark_and_monitor.py

# Despliegue contenedorizado (API + Streamlit)
docker compose up --build
```
