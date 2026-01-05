#!/bin/bash

# Script de despliegue completo para Capibara6 con Acontext real
# Este script levanta todos los servicios necesarios para el sistema completo

set -e

echo "🚀 Desplegando Capibara6 con Acontext real..."

echo ""
echo "📦 Paso 1: Verificando dependencias..."
if ! [ -x "$(command -v docker)" ]; then
  echo "❌ Docker no está instalado"
  exit 1
fi

if ! [ -x "$(command -v docker compose)" ] && ! [ -x "$(command -v docker-compose)" ]; then
  echo "❌ Docker Compose no está instalado"
  exit 1
fi

echo "✅ Dependencias verificadas"

echo ""
echo "📦 Paso 2: Desplegando Acontext real..."
cd /home/elect/capibara6/Acontext/src/server

# Verificar que la API key esté configurada
API_KEY=$(grep -E '^LLM_API_KEY=' .env | cut -d'=' -f2 | sed 's/"//g')
if [[ "$API_KEY" == *"your-openai-api-key-here"* ]]; then
  echo "⚠️  ADVERTENCIA: La API key de OpenAI en .env aún es el valor por defecto."
  echo "   Por favor actualice el archivo Acontext/src/server/.env con su API key real"
  read -p "¿Desea continuar de todos modos? (s/n): " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    echo "Cancelando despliegue."
    exit 1
  fi
fi

if [ -x "$(command -v docker-compose)" ]; then
  docker-compose -f docker-compose-simple.yaml up -d
else
  docker compose -f docker-compose-simple.yaml up -d
fi

echo "⏳ Esperando que Acontext esté listo..."
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
  exit 1
fi

echo ""
echo "📦 Paso 3: Iniciando Gateway Server..."
cd /home/elect/capibara6/backend

# Asegurarse de que no haya procesos previos corriendo
if pgrep -f "gateway_server.py" > /dev/null; then
  echo "🛑 Deteniendo gateway server anterior..."
  pkill -f "gateway_server.py" || true
  sleep 3
fi

# Iniciar el gateway server en segundo plano
python3 gateway_server.py &
GATEWAY_PID=$!

echo "⏳ Esperando que Gateway Server esté listo..."
MAX_ATTEMPTS=30
ATTEMPT=1
while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
  if curl -f -s http://localhost:8080/api/health > /dev/null; then
    echo "✅ Gateway Server está disponible!"
    break
  else
    echo "⏳ Esperando Gateway Server... ($ATTEMPT/$MAX_ATTEMPTS)"
    sleep 5
    ATTEMPT=$((ATTEMPT + 1))
  fi
done

if [ $ATTEMPT -gt $MAX_ATTEMPTS ]; then
  echo "❌ Gateway Server no está disponible después de $MAX_ATTEMPTS intentos"
  exit 1
fi

echo ""
echo "📦 Paso 4: Iniciando servidor de mock (para servicios adicionales)..."
cd /home/elect/capibara6

# Detener mock anterior si existe
if pgrep -f "acontext_mock_server.py" > /dev/null; then
  pkill -f "acontext_mock_server.py" || true
  sleep 2
fi

# No iniciar el mock server ya que Acontext real está corriendo
echo "ℹ️  Mock server no iniciado - Acontext real está en uso"

echo ""
echo "✅ Despliegue completo exitoso!"
echo ""
echo "🔧 Servicios desplegados:"
echo "   - Acontext Server: http://localhost:8029 (API en /api/v1)"
echo "   - Gateway Server: http://localhost:8080"
echo "   - API Health: http://localhost:8080/api/health"
echo "   - Acontext Status: http://localhost:8080/api/acontext/status"
echo ""
echo "🧪 Pruebas rápidas:"
echo "   1. Estado del sistema: curl -s http://localhost:8080/api/health | jq"
echo "   2. Estado de Acontext: curl -s http://localhost:8080/api/acontext/status | jq"
echo "   3. Crear sesión: curl -s -X POST 'http://localhost:8080/api/acontext/session/create' -H 'Content-Type: application/json' -d '{}'"
echo ""
echo "📋 Para detener los servicios:"
echo "   - Gateway: pkill -f 'gateway_server.py'"
echo "   - Acontext: cd Acontext/src/server && [docker-compose|docker compose] -f docker-compose-simple.yaml down"
echo ""