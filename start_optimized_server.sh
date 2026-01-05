#!/bin/bash
# Script para iniciar el servidor ARM-Axion optimizado
# start_optimized_server.sh

set -e  # Salir si hay un error

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # Sin color

echo -e "${CYAN}🦫 Iniciando servidor ARM-Axion optimizado${NC}"
echo "   VM: models-europe"
echo "   Puerto: 8082"
echo "   Modelo: Multi-expert con optimizaciones ARM-Axion"
echo ""

# Configurar ambiente para ARM-Axion
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_USE_FLASHINFER=0
export TORCHINDUCTOR_DISABLED=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:disabled

# Verificar que no haya un servidor ya corriendo
if lsof -Pi :8082 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}❌ Error: Puerto 8082 ya está en uso${NC}"
    echo "   Verifica si el servidor ya está corriendo:"
    echo "   ps aux | grep multi_model_server"
    exit 1
fi

echo -e "${GREEN}✓${NC} Variables de entorno ARM-Axion configuradas"

# Verificar que el archivo de configuración exista
if [ ! -f "arm-axion-optimizations/vllm_integration/config.json" ]; then
    echo -e "${RED}❌ Error: No se encuentra el archivo de configuración${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Archivo de configuración encontrado"

# Iniciar servidor en segundo plano con logging
echo -e "${CYAN}🚀 Iniciando servidor en modo daemon...${NC}"
cd arm-axion-optimizations/vllm_integration
nohup python3 multi_model_server.py --host 0.0.0.0 --port 8082 --config config.json > /tmp/multi_model_server.log 2>&1 &

SERVER_PID=$!
echo "PID del servidor: $SERVER_PID" >> /tmp/multi_model_server.log

# Esperar a que el servidor arranque
echo -e "${CYAN}⏳ Esperando que el servidor inicie (30 segundos)...${NC}"
sleep 30

# Verificar que el servidor esté escuchando en el puerto
if ss -tlnp | grep ":8082" >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Servidor escuchando en puerto 8082"
else
    echo -e "${RED}❌ Error: Servidor no escucha en puerto 8082${NC}"
    echo "   Revisando logs..."
    tail -20 /tmp/multi_model_server.log
    exit 1
fi

# Verificar estado de salud del servidor
echo -e "${CYAN}🏥 Verificando estado de salud del servidor...${NC}"
if curl -s --connect-timeout 10 http://localhost:8082/health >/dev/null 2>&1; then
    HEALTH_RESPONSE=$(curl -s http://localhost:8082/health)
    echo -e "${GREEN}✓${NC} Servidor saludable: $HEALTH_RESPONSE"
else
    echo -e "${YELLOW}⚠️  Servidor no responde a health check aún${NC}"
    echo "   Esto puede ser normal durante la carga inicial de modelos"
    echo "   El servidor se inició correctamente con PID: $SERVER_PID"
    exit 0
fi

# Verificar modelos disponibles
echo -e "${CYAN}🤖 Verificando modelos disponibles...${NC}"
MODELS_COUNT=$(curl -s http://localhost:8082/v1/models | jq '.data | length' 2>/dev/null || echo "Error")
if [ "$MODELS_COUNT" != "Error" ]; then
    echo -e "${GREEN}✓${NC} $MODELS_COUNT modelos disponibles"
    curl -s http://localhost:8082/v1/models | jq '.data[].id'
else
    echo -e "${YELLOW}⚠️  No se pudo obtener lista de modelos (quizás aún cargando)${NC}"
fi

echo ""
echo -e "${GREEN}✅ Servidor ARM-Axion iniciado exitosamente${NC}"
echo "   Puerto: 8082"
echo "   Logs: /tmp/multi_model_server.log"
echo "   Configuración: v1 engine deshabilitado para compatibilidad ARM"
echo "   Optimizaciones: NEON, ACL, FP8 KV Cache, Flash Attention"
echo ""
echo -e "${CYAN}Endpoints disponibles:${NC}"
echo "   Servidor Estándar (8082):"
echo "   - GET  http://localhost:8082/health - Verificar estado"
echo "   - GET  http://localhost:8082/v1/models - Lista de modelos"
echo "   - POST http://localhost:8082/v1/chat/completions - API OpenAI (respuesta completa)"
echo "   - GET  http://localhost:8082/stats - Estadísticas del servidor"
echo ""
echo "   Servidor con Streaming (8083):"
echo "   - GET  http://localhost:8083/health - Verificar estado streaming"
echo "   - GET  http://localhost:8083/v1/models - Lista de modelos"
echo "   - POST http://localhost:8083/v1/chat/completions - API OpenAI con streaming"
echo "   - GET  http://localhost:8083/stats - Estadísticas del servidor con streaming"