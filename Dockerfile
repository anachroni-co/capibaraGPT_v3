# Dockerfile para Capibara6 - Sistema AI Avanzado
FROM python:3.10-slim

# Metadatos
LABEL maintainer="Capibara6 Team"
LABEL description="Advanced AI Agent System with Intelligent Routing, ACE, E2B, and Scalability"
LABEL version="1.0.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONPATH=/app/backend:/app/backend/core

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root
RUN groupadd -r capibara6 && useradd -r -g capibara6 capibara6

# Crear directorios de trabajo
WORKDIR /app
RUN mkdir -p /app/backend/data /app/backend/logs /app/backend/models
RUN chown -R capibara6:capibara6 /app

# Copiar requirements y instalar dependencias Python
COPY backend/requirements.txt /app/backend/
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copiar código de la aplicación
COPY backend/ /app/backend/
COPY web/ /app/web/

# Cambiar permisos
RUN chown -R capibara6:capibara6 /app

# Cambiar a usuario no-root
USER capibara6

# Exponer puertos
EXPOSE 8000 8001 8002

# Health check (usando wget ya que curl puede no estar disponible para no-root user)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

# Comando por defecto
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
