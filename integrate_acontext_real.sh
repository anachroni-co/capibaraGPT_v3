#!/bin/bash

# Script para integrar Acontext real con el sistema de Capibara6
# Este script configura el gateway server para usar Acontext real en lugar del mock

set -e  # Salir si algún comando falla

echo "🔗 Configurando integración de Acontext real con Capibara6..."

# Directorio del backend de Capibara6
GATEWAY_DIR="/home/elect/capibara6/backend"

# Verificar que el directorio exista
if [ ! -d "$GATEWAY_DIR" ]; then
  echo "❌ Directorio $GATEWAY_DIR no encontrado."
  exit 1
fi

echo "📁 Cambiando al directorio: $GATEWAY_DIR"
cd "$GATEWAY_DIR"

# Actualizar archivo de entorno para usar Acontext real
ENV_FILE="$GATEWAY_DIR/.env.production"

# Crear o actualizar el archivo de entorno
if [ -f "$ENV_FILE" ]; then
  echo "📝 Actualizando archivo de entorno existente..."
  # Configurar para usar Acontext real
  sed -i 's/ACONTEXT_BASE_URL=.*/ACONTEXT_BASE_URL=http:\/\/localhost:8029\/api\/v1/' "$ENV_FILE" 2>/dev/null || echo "ACONTEXT_BASE_URL=http://localhost:8029/api/v1" >> "$ENV_FILE"
  sed -i 's/ACONTEXT_ENABLED=.*/ACONTEXT_ENABLED=true/' "$ENV_FILE" 2>/dev/null || echo "ACONTEXT_ENABLED=true" >> "$ENV_FILE"
  sed -i 's/ACONTEXT_API_KEY=.*/ACONTEXT_API_KEY=sk-ac-your-root-api-bearer-token/' "$ENV_FILE" 2>/dev/null || echo "ACONTEXT_API_KEY=sk-ac-your-root-api-bearer-token" >> "$ENV_FILE"
else
  echo "📝 Creando archivo de entorno con configuración de Acontext..."
  cat > "$ENV_FILE" << EOF
# Configuración de entorno para Capibara6 con Acontext real

# URLs de servicios
VLLM_URL=http://10.204.0.9:8080
OLLAMA_URL=http://10.204.0.9:11434
BRIDGE_API_URL=http://10.204.0.10:8000

# API Keys entre VMs
INTER_VM_API_KEY=your-inter-vm-api-key-here

# Configuración de Acontext - Ahora usando servidor real
ACONTEXT_ENABLED=true
ACONTEXT_BASE_URL=http://localhost:8029/api/v1
ACONTEXT_API_KEY=sk-ac-your-root-api-bearer-token
ACONTEXT_PROJECT_ID=capibara6-project
ACONTEXT_SPACE_ID=

# Rate limiting
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW=60

# Circuit breaker
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=60
EOF
fi

echo "✅ Archivo de entorno actualizado con configuración de Acontext real"

# Actualizar la configuración del módulo de integración si es necesario
echo "🔄 Reiniciando servicios para aplicar cambios (si están corriendo)..."

# Detener el gateway server si está corriendo
if pgrep -f "gateway_server.py" > /dev/null; then
  echo "🛑 Deteniendo gateway server anterior..."
  pkill -f "gateway_server.py" || true
  sleep 3
fi

# Levantar el gateway server con la nueva configuración
echo "🚀 Iniciando gateway server con integración de Acontext real..."
cd /home/elect/capibara6/backend && python3 gateway_server.py &

echo ""
echo "🎉 Integración de Acontext real configurada exitosamente!"
echo ""
echo "🔧 El gateway server ahora usará Acontext real en lugar del mock"
echo "   - Acontext API: http://localhost:8029/api/v1"
echo "   - Acontext está habilitado: true"
echo "   - Token de autenticación configurado"
echo ""
echo "🧪 Prueba la integración:"
echo "   curl -s http://localhost:8080/api/acontext/status | jq"
echo ""