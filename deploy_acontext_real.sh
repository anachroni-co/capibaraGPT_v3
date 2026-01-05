#!/bin/bash

# Script para desplegar la infraestructura real de Acontext
# Este script levantará Acontext con Docker usando la configuración simple (PostgreSQL + Redis)

set -e  # Salir si algún comando falla

echo "🚀 Desplegando infraestructura real de Acontext..."

# Verificar que Docker esté instalado
if ! [ -x "$(command -v docker)" ]; then
  echo "❌ Docker no está instalado. Por favor instale Docker antes de continuar."
  exit 1
fi

# Verificar que docker-compose esté instalado
if ! [ -x "$(command -v docker-compose)" ]; then
  # Intentar usar docker compose (subcomando de Docker)
  if ! [ -x "$(command -v docker compose)" ]; then
    echo "❌ Docker Compose no está instalado. Por favor instale Docker Compose antes de continuar."
    exit 1
  fi
fi

# Directorio base de Acontext
ACONTEXT_DIR="/home/elect/capibara6/Acontext/src/server"

# Verificar que el directorio exista
if [ ! -d "$ACONTEXT_DIR" ]; then
  echo "❌ Directorio $ACONTEXT_DIR no encontrado."
  exit 1
fi

echo "📁 Cambiando al directorio: $ACONTEXT_DIR"
cd "$ACONTEXT_DIR"

# Asegurar que tengamos el archivo .env con la API key
ENV_FILE="$ACONTEXT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "❌ Archivo .env no encontrado. Copiando .env.example..."
  cp .env.example .env
  echo "⚠️  Asegúrese de actualizar su API key en el archivo .env antes de continuar"
  exit 1
else
  # Verificar si la API key está configurada
  API_KEY=$(grep -E '^LLM_API_KEY=' .env | cut -d'=' -f2 | sed 's/"//g')
  if [[ "$API_KEY" == *"your-openai-api-key-here"* ]]; then
    echo "⚠️  La API key no ha sido configurada en .env. Asegúrese de actualizarla antes de continuar."
    echo "   Editar archivo: $ENV_FILE"
    exit 1
  fi
fi

echo "🔧 Iniciando servicios de Acontext..."

# Usar docker-compose para levantar los servicios
if [ -x "$(command -v docker-compose)" ]; then
  docker-compose -f docker-compose-simple.yaml up -d
else
  # Usar el subcomando de Docker
  docker compose -f docker-compose-simple.yaml up -d
fi

echo "⏳ Esperando que los servicios estén listos..."

# Esperar a que el servicio API esté disponible
MAX_ATTEMPTS=60
ATTEMPT=1

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
  if curl -f -s http://localhost:8029/health > /dev/null; then
    echo "✅ Acontext API está disponible!"
    break
  else
    echo "⏳ Esperando Acontext API... ($ATTEMPT/$MAX_ATTEMPTS)"
    sleep 10
    ATTEMPT=$((ATTEMPT + 1))
  fi
done

if [ $ATTEMPT -gt $MAX_ATTEMPTS ]; then
  echo "❌ Acontext API no está disponible después de $MAX_ATTEMPTS intentos"
  if [ -x "$(command -v docker-compose)" ]; then
    docker-compose -f docker-compose-simple.yaml logs
  else
    docker compose -f docker-compose-simple.yaml logs
  fi
  exit 1
fi

echo ""
echo "🎉 Infraestructura real de Acontext desplegada exitosamente!"
echo ""
echo "🔧 Servicios disponibles:"
echo "   - API: http://localhost:8029/api/v1"
echo "   - Health check: http://localhost:8029/health"
echo "   - PostgreSQL: localhost:15432 (en Docker)"
echo "   - Redis: localhost:16379 (en Docker)"
echo ""
echo "🔑 Token de API: sk-ac-your-root-api-bearer-token (configurable en .env)"
echo ""
echo "ℹ️  Para detener los servicios:"
echo "   cd $ACONTEXT_DIR && [docker-compose | docker compose] -f docker-compose-simple.yaml down"
echo ""