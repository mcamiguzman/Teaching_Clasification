# Tutorial para el Equipo — Instalación y Ejecución desde Cero



## Requisitos previos

Asegúrate de tener instalado:
- **Python 3.11 o 3.12** (`python --version` o `python3 --version`)
  > **Importante (`Parte 2.4` — Compatibilidad de versiones):** procura usar la misma versión menor de Python (idealmente **3.11.x**) y las versiones de `requirements.txt` para evitar advertencias o incompatibilidades de `scikit-learn` al cargar el archivo `.joblib`.
- *(Opcional)* **Docker Desktop** si prefieres levantar todo con un solo comando.

---


### Opción A — Windows (PowerShell)
```powershell
# 1. Crear entorno virtual aislado
python -m venv .venv

# 2. Activar entorno virtual
# (Si PowerShell bloquea scripts, ejecuta antes: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
.\.venv\Scripts\Activate.ps1

# 3. Actualizar pip e instalar dependencias del proyecto
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Opción B — macOS / Linux (Bash / Zsh)
```bash
# 1. Crear entorno virtual aislado
python3 -m venv .venv

# 2. Activar entorno virtual
source .venv/bin/activate

# 3. Actualizar pip e instalar dependencias del proyecto
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Variables de entorno (opcional)
Si vas a configurar clave administrativa para endpoints de cambio/reversión de versiones:
```bash
# Windows PowerShell:
Copy-Item .env.example .env

# macOS / Linux:
cp .env.example .env
```

---

## Datos de entrenamiento (`data/raw/`)

Tienes dos modos de trabajo sin cambiar una sola línea de código:

1. **Con los 3 CSV originales del parcial (recomendado para producción / entrega final):**
   Copia los tres archivos dentro de `data/raw/`:
   - `data/raw/feedback_curso_universitario.csv`
   - `data/raw/feedback_curso_universitario2.csv`
   - `data/raw/feedback_curso_universitario3.csv`
   El script `src/dataset_builder.py` los detectará automáticamente y aplicará todas las limpiezas documentadas en `Informe_Parcial_SisInt.pdf`.

2. **Sin copiar los CSV (modo demostrativo inmediato):**
   Si `data/raw/` no contiene los CSV, el proyecto genera automáticamente una réplica determinista de 618 registros en español con la misma estructura, asignaturas y distribución del parcial para que puedas probar todo el flujo al instante.

---

## Flujo paso a paso de ejecución

> **Nota:** todos los comandos siguientes asumen que tienes el entorno virtual `.venv` **activado** y estás en la raíz del repositorio (`Teaching_Clasification/`).

### Paso 1 — Entrenar, seleccionar el mejor modelo y guardar en `.joblib` *(Parte 1.1, 1.2 y Parte 2.1)*
```bash
python -m src.train_and_select --version v1.0.0 --dataset-version 1.0.0
```
*(Si quieres una ejecución más rápida para pruebas locales, agrega `--fast` al final.)*

¿Qué produce?
- `models/v1.0.0/modelo_polaridad.joblib`: **Pipeline unificado** (`TextNormalizer` + `TfidfVectorizer` + `CalibratedClassifierCV(LinearSVC)`).
- `models/v1.0.0/metadata.json`: métricas de CV y test, hiperparámetros, versión/hash del dataset y versiones exactas de Python y librerías.
- `models/v1.0.0/checksum.sha256`: huella criptográfica para verificar integridad antes de deserializar.
- `models/registry.json`: registro de versiones actualizado con `v1.0.0` activa.
- `monitoring/baseline_stats.json`: estadísticas de referencia para detectar *drift*.

---

### Paso 2 — Validar la carga en un entorno limpio y casos borde *(Parte 1.3, 2.3 y 2.4)*
```bash
python -m src.validate_model
```
Este comando abre un **subproceso Python completamente limpio** (sin objetos del entrenamiento en memoria), valida el hash SHA-256, verifica 100% de coherencia en test y prueba casos borde (texto vacío, espacios, `None`, tipos inválidos, solo emojis y comentarios ambiguos/contrastantes).

---

### Paso 3 — Levantar el servicio API REST con FastAPI *(Parte 1.4, 2.2 y 2.5)*
Abre la **Terminal 1** (con `.venv` activado):
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
- **Swagger UI interactivo:** abre en tu navegador `http://127.0.0.1:8000/docs`
- **Health check:** `http://127.0.0.1:8000/health`
- **Importante (`Parte 2.2`):** la API carga el `.joblib` una sola vez al arrancar y **nunca reentrena**.

#### Probar una predicción desde terminal

**En Windows (PowerShell):**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"comentario":"El contenido fue amplio y útil, pero los recursos en el aula digital fueron limitados.","incluir_aspectos":true}' `
  | ConvertTo-Json -Depth 6
```

**En macOS / Linux (curl):**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"comentario":"El contenido fue amplio y útil, pero los recursos en el aula digital fueron limitados.","incluir_aspectos":true}'
```

La respuesta incluye:
- `clase` (`negativo` / `neutral` / `positivo`) y `probabilidades` calibradas.
- `aspectos`: bloque estructurado con **exactamente los 9 aspectos docentes** (`contenidos`, `metodologia_docente`, `evaluaciones_parciales`, `material_clase`, `retroalimentacion`, `organizacion_horario`, `disponibilidad_docente`, `recursos_aula_digital`, `dificultad_ritmo`).

---

### Paso 4 — Levantar la aplicación web institucional con Streamlit *(Parte 1.5)*
Abre una **Terminal 2** (con `.venv` activado):
```bash
streamlit run app/streamlit_app.py
```
Se abrirá en tu navegador en `http://localhost:8501`.

Dentro de la app podrás:
1. **Comentario Individual:** probar comentarios propios o ejemplos del parcial y ver la tabla estructurada de los 9 aspectos.
2. **Carga Masiva (CSV):** descargar una plantilla de ejemplo, subir un CSV con comentarios y descargar el CSV clasificado.
3. **Modelo y Versiones:** revisar métricas, SHA-256, compatibilidad de librerías y hacer cambios o *rollback* de versión.
4. **Monitoreo y Drift:** consultar latencia (`p50`/`p95`), distribución actual vs entrenamiento e índice de deriva (`PSI`).

> **Tip:** si no levantaste la API FastAPI, puedes cambiar el selector de la barra lateral de Streamlit a **«Local (.joblib cacheado)»** y la app funcionará igualmente sin reentrenar.

---

### Paso 5 — Control y reversión de versiones (*Rollback*) *(Parte 2.6)*
Para entrenar una segunda versión candidata, alternar entre versiones o revertir ante un problema:

```bash
# 1. Entrenar una nueva versión candidata sin activarla todavía
python -m src.train_and_select --version v1.1.0 --dataset-version 1.0.0 --clf-family logistic_regression --no-activate --notes "Alternativa LogisticRegression"

# 2. Ver todas las versiones registradas y cuál está activa
python scripts/manage_versions.py list

# 3. Activar la nueva versión
python scripts/manage_versions.py switch v1.1.0 --reason "Pruebas de rendimiento en producción"

# 4. Revertir inmediatamente a la versión estable anterior (v1.0.0)
python scripts/manage_versions.py rollback --reason "Regresión detectada; volver a v1.0.0"

# 5. Ver historial completo de auditoría
python scripts/manage_versions.py history
```

---

### Paso 6 — Ejecutar pruebas automatizadas y benchmark de latencia/monitoreo *(Parte 1.6)*
```bash
# Ejecutar toda la suite de pruebas (preprocesamiento, seguridad SHA-256, rollback y API)
pytest

# Medir latencia (p50/p95/p99), throughput, seguridad y drift
python scripts/benchmark_and_monitor.py
```

---

## 5. Ejecución rápida con Docker (opcional, ideal para despliegue)

Si tienes **Docker Desktop** instalado y ya existe al menos una versión en `models/` (por ejemplo `v1.0.0`):

```bash
docker compose up --build
```
Esto levanta automáticamente:
- **API FastAPI:** `http://localhost:8000/docs`
- **App Streamlit:** `http://localhost:8501`

Para detener los contenedores:
```bash
docker compose down
```

---

## 6. Buenas prácticas para el equipo al trabajar con Git

1. **Nunca subir `.venv/` ni `.env`** (ya están excluidos en `.gitignore`).
2. **Al agregar una dependencia nueva:** actualizar `requirements.txt` y avisar al equipo para que corra `pip install -r requirements.txt`.
3. **Al entrenar un nuevo modelo compartido:**
   - Usar una versión semántica nueva (`--version v1.2.0`, nunca sobrescribir silenciosamente una versión estable en producción).
   - Verificar que `python -m src.validate_model` y `pytest` pasen en verde antes de hacer commit/push.
4. **Seguridad (`Parte 2.3`):** nunca cargues archivos `.joblib` recibidos por fuera del repositorio; el registro bloquea archivos externos o con SHA-256 alterado.

---

## 7. Solución rápida de problemas comunes

| Problema | Causa probable | Solución |
|---|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` (Windows) | Política de ejecución de PowerShell | Ejecutar una vez: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `No hay ningún modelo registrado/activo en models/registry.json` | Aún no se ha entrenado ninguna versión localmente | Ejecutar `python -m src.train_and_select --version v1.0.0 --dataset-version 1.0.0` |
| `Address already in use` en puerto `8000` o `8501` | Otro proceso está usando el puerto | Cambiar el puerto: `uvicorn api.main:app --port 8001` o `streamlit run app/streamlit_app.py --server.port 8502` |
| `Advertencia de compatibilidad [Parte 2.4]: scikit-learn ...` | La versión local de `scikit-learn` difiere de la usada al entrenar `.joblib` | Ejecutar `pip install -r requirements.txt` o reentrenar en el entorno actual |
| `BLOQUEO DE SEGURIDAD [Parte 2.3]: el hash SHA-256 no coincide` | El archivo `.joblib` fue editado o corrompido manualmente | Re-descargar el artefacto original desde Git o reentrenar una nueva versión |
| `ModuleNotFoundError: No module named 'src'` | Comando ejecutado fuera de la raíz del proyecto | Asegurarse de estar en `Teaching_Clasification/` y usar `python -m src.<modulo>` |
