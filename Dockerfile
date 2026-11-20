# =============================================================================
# [Parte 1.6 & Parte 2.4] Contenedor reproducible para despliegue en servidor/cloud
# Usa Python 3.11-slim y fija las dependencias compatibles de requirements.txt.
# El modelo `.joblib` se monta o empaqueta ya entrenado (NUNCA se reentrena al iniciar).
# =============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Instalar dependencias exactas del proyecto
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar código fuente, API, aplicación Streamlit y modelos registrados
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY monitoring/ ./monitoring/

# Usuario no-root por buenas prácticas de seguridad [Parte 1.6]
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

# Por defecto levanta el servicio FastAPI; docker-compose sobreescribe para Streamlit
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
