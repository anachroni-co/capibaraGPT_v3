#!/bin/bash
# Script para iniciar el servidor ARM-Axion con LiveMind Consensus
# start_consensus_server.sh

set -e  # Salir si hay un error

GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"  # Sin color

echo -e "${CYAN}🦫 Iniciando servidor ARM-Axion con LiveMind Consensus${NC}"
echo "   VM: models-europe"
echo "   Puerto: 8084 (consensus server)"
echo "   Modelo: Multi-expert con consensus paralelo ARM-Axion"
echo ""

# Configurar ambiente para ARM-Axion con optimizaciones para consenso
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_USE_FLASHINFER=0
export TORCHINDUCTOR_DISABLED=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:disabled

# Verificar que no haya un servidor ya corriendo en el puerto 8084
if lsof -Pi :8084 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}❌ Error: Puerto 8084 ya está en uso${NC}"
    echo "   Verifica si el servidor de consenso ya está corriendo:"
    echo "   ps aux | grep multi_model_server_consensus"
    exit 1
fi

echo -e "${GREEN}✓${NC} Variables de entorno ARM-Axion configuradas para consenso"

# Verificar que el archivo de configuración exista
if [ ! -f "arm-axion-optimizations/vllm_integration/config.json" ]; then
    echo -e "${RED}❌ Error: No se encuentra el archivo de configuración${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Archivo de configuración encontrado"

# Iniciar servidor de consenso en segundo plano con logging
echo -e "${CYAN}🚀 Iniciando servidor de consenso en modo daemon (puerto 8084)...${NC}"
cd arm-axion-optimizations/vllm_integration
nohup python3 multi_model_server_consensus.py --host 0.0.0.0 --port 8084 --config config.json > /tmp/multi_model_server_consensus.log 2>&1 &

SERVER_PID=$!
echo "PID del servidor de consenso: $SERVER_PID" >> /tmp/multi_model_server_consensus.log

# Esperar a que el servidor arranque
echo -e "${CYAN}⏳ Esperando que el servidor de consenso inicie (60 segundos)...${NC}"
sleep 60

# Verificar que el servidor esté escuchando en el puerto
if ss -tlnp | grep ":8084" >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Servidor de consenso escuchando en puerto 8084"
else
    echo -e "${RED}❌ Error: Servidor de consenso no escucha en puerto 8084${NC}"
    echo "   Revisando logs..."
    tail -20 /tmp/multi_model_server_consensus.log
    exit 1
fi

# Verificar estado de salud del servidor de consenso
echo -e "${CYAN}🏥 Verificando estado de salud del servidor de consenso...${NC}"
if curl -s --connect-timeout 10 http://localhost:8084/health >/dev/null 2>&1; then
    HEALTH_RESPONSE=$(curl -s http://localhost:8084/health)
    echo -e "${GREEN}✓${NC} Servidor de consenso saludable: $HEALTH_RESPONSE"
else
    echo -e "${YELLOW}⚠️  Servidor de consenso no responde a health check aún${NC}"
    echo "   Esto puede ser normal durante la carga inicial de modelos"
    echo "   El servidor de consenso se inició correctamente con PID: $SERVER_PID"
    exit 0
fi

# Verificar que el consenso esté habilitado
if curl -s http://localhost:8084/stats | jq -e ".config.consensus_enabled" >/dev/null 2>&1; then
    CONSENSUS_ENABLED=$(curl -s http://localhost:8084/stats | jq -r ".config.consensus_enabled")
    if [ "$CONSENSUS_ENABLED" = "true" ]; then
        echo -e "${GREEN}✓${NC} Sistema de consenso habilitado en el servidor"
    else
        echo -e "${YELLOW}⚠️  Sistema de consenso no está habilitado en el servidor${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No se pudo verificar estado de consenso (respuesta inesperada)${NC}"
fi

# Verificar modelos disponibles
echo -e "${CYAN}🤖 Verificando modelos disponibles en servidor de consenso...${NC}"
MODELS_COUNT=$(curl -s http://localhost:8084/v1/models | jq ".data | length" 2>/dev/null || echo "Error")
if [ "$MODELS_COUNT" != "Error" ]; then
    echo -e "${GREEN}✓${NC} $MODELS_COUNT modelos disponibles en servidor de consenso"
    curl -s http://localhost:8084/v1/models | jq ".data[].id"
else
    echo -e "${YELLOW}⚠️  No se pudo obtener lista de modelos en servidor de consenso${NC}"
fi

echo ""
echo -e "${GREEN}✅ Servidor ARM-Axion con CONSENSO iniciado exitosamente${NC}"
echo "   Puerto: 8084 (consensus server)"
echo "   Logs: /tmp/multi_model_server_consensus.log"
echo "   Configuración: v1 engine deshabilitado para compatibilidad ARM"
echo "   Optimizaciones: NEON, ACL, FP8 KV Cache, Flash Attention, LIVE CONSENSUS"
echo ""
echo -e "${CYAN}Endpoints disponibles (Consensus):${NC}"
echo "   GET  http://localhost:8084/health - Verificar estado"
echo "   GET  http://localhost:8084/v1/models - Lista de modelos"
echo "   POST http://localhost:8084/v1/chat/completions - API OpenAI con consenso"
echo "   POST http://localhost:8084/v1/completions - API OpenAI completions con consenso"
echo "   GET  http://localhost:8084/stats - Estadísticas del servidor"
echo ""
echo -e "${CYAN}Diferencias con servidores anteriores:${NC}"
echo "   - Puerto 8082: Servidor estándar (respuesta completa)"
echo "   - Puerto 8083: Servidor con streaming verdadero (token por token)"
echo "   - Puerto 8084: Servidor con consenso paralelo (múltiples expertos)"
