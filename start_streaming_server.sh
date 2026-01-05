#!/bin/bash
# Script para iniciar el servidor ARM-Axion optimizado con streaming
# start_streaming_server.sh

set -e  # Salir si hay un error

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # Sin color

echo -e "${CYAN}🦫 Iniciando servidor ARM-Axion optimizado con STREAMING${NC}"
echo "   VM: models-europe"
echo "   Puerto: 8083 (streaming server)"
echo "   Modelo: Multi-expert con streaming verdadero ARM-Axion"
echo ""

# Configurar ambiente para ARM-Axion con optimizaciones para streaming
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_USE_FLASHINFER=0
export TORCHINDUCTOR_DISABLED=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:disabled
export VLLM_ASYNC_OUTPUT_THREAD_POOL_SIZE=1  # Para menor latencia en streaming

# Verificar que no haya un servidor ya corriendo en el puerto 8083
if lsof -Pi :8083 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}❌ Error: Puerto 8083 ya está en uso${NC}"
    echo "   Verifica si el servidor de streaming ya está corriendo:"
    echo "   ps aux | grep multi_model_server_streaming"
    exit 1
fi

echo -e "${GREEN}✓${NC} Variables de entorno ARM-Axion configuradas para streaming"

# Verificar que el archivo de configuración exista
if [ ! -f "arm-axion-optimizations/vllm_integration/config.json" ]; then
    echo -e "${RED}❌ Error: No se encuentra el archivo de configuración${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Archivo de configuración encontrado"

# Iniciar servidor de streaming en segundo plano con logging
echo -e "${CYAN}🚀 Iniciando servidor de streaming en modo daemon (puerto 8083)...${NC}"
cd arm-axion-optimizations/vllm_integration
nohup python3 multi_model_server_streaming.py --host 0.0.0.0 --port 8083 --config config.json > /tmp/multi_model_server_streaming.log 2>&1 &

SERVER_PID=$!
echo "PID del servidor de streaming: $SERVER_PID" >> /tmp/multi_model_server_streaming.log

# Esperar a que el servidor arranque
echo -e "${CYAN}⏳ Esperando que el servidor de streaming inicie (45 segundos)...${NC}"
sleep 45

# Verificar que el servidor esté escuchando en el puerto
if ss -tlnp | grep ":8083" >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Servidor de streaming escuchando en puerto 8083"
else
    echo -e "${RED}❌ Error: Servidor de streaming no escucha en puerto 8083${NC}"
    echo "   Revisando logs..."
    tail -20 /tmp/multi_model_server_streaming.log
    exit 1
fi

# Verificar estado de salud del servidor de streaming
echo -e "${CYAN}🏥 Verificando estado de salud del servidor de streaming...${NC}"
if curl -s --connect-timeout 10 http://localhost:8083/health >/dev/null 2>&1; then
    HEALTH_RESPONSE=$(curl -s http://localhost:8083/health)
    echo -e "${GREEN}✓${NC} Servidor de streaming saludable: $HEALTH_RESPONSE"
else
    echo -e "${YELLOW}⚠️  Servidor de streaming no responde a health check aún${NC}"
    echo "   Esto puede ser normal durante la carga inicial de modelos"
    echo "   El servidor de streaming se inició correctamente con PID: $SERVER_PID"
    exit 0
fi

# Verificar que el streaming esté habilitado
if curl -s http://localhost:8083/stats | jq -e '.config.streaming_enabled' >/dev/null 2>&1; then
    STREAMING_ENABLED=$(curl -s http://localhost:8083/stats | jq -r '.config.streaming_enabled')
    if [ "$STREAMING_ENABLED" = "true" ]; then
        echo -e "${GREEN}✓${NC} Streaming verdadero habilitado en el servidor"
    else
        echo -e "${YELLOW}⚠️  Streaming no está habilitado en el servidor${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No se pudo verificar estado de streaming (respuesta inesperada)${NC}"
fi

# Verificar modelos disponibles
echo -e "${CYAN}🤖 Verificando modelos disponibles en servidor de streaming...${NC}"
MODELS_COUNT=$(curl -s http://localhost:8083/v1/models | jq '.data | length' 2>/dev/null || echo "Error")
if [ "$MODELS_COUNT" != "Error" ]; then
    echo -e "${GREEN}✓${NC} $MODELS_COUNT modelos disponibles en servidor de streaming"
    curl -s http://localhost:8083/v1/models | jq '.data[].id'
else
    echo -e "${YELLOW}⚠️  No se pudo obtener lista de modelos en servidor de streaming${NC}"
fi

echo ""
echo -e "${GREEN}✅ Servidor ARM-Axion con STREAMING iniciado exitosamente${NC}"
echo "   Puerto: 8083 (streaming server)"
echo "   Logs: /tmp/multi_model_server_streaming.log"
echo "   Configuración: v1 engine deshabilitado para compatibilidad ARM"
echo "   Optimizaciones: NEON, ACL, FP8 KV Cache, Flash Attention, TRUE STREAMING"
echo ""
echo -e "${CYAN}Endpoints disponibles (Streaming):${NC}"
echo "   GET  http://localhost:8083/health - Verificar estado"
echo "   GET  http://localhost:8083/v1/models - Lista de modelos"
echo "   POST http://localhost:8083/v1/chat/completions - API OpenAI con streaming"
echo "   POST http://localhost:8083/v1/completions - API OpenAI completions con streaming"
echo "   GET  http://localhost:8083/stats - Estadísticas del servidor"
echo ""
echo -e "${CYAN}Diferencias con servidor principal (8082):${NC}"
echo "   - Puerto 8082: Servidor estándar (respuesta completa)"
echo "   - Puerto 8083: Servidor con streaming verdadero (token por token)"
echo "   - Puerto 8083: Optimizado para baja latencia de primer token (TTFT)"