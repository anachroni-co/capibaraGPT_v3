#!/bin/bash
# Script para iniciar el servidor ARM-Axion con LiveMind Consensus Seguro
# start_consensus_server_safe.sh

set -e  # Salir si hay un error

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # Sin color

echo -e "${CYAN}🦫 Iniciando servidor ARM-Axion con LiveMind Consensus SEGURO${NC}"
echo "   VM: models-europe"
echo "   Puerto: 8085 (consensus server seguro)"
echo "   Modelo: Multi-expert con consensus paralelo y lazy loading"
echo ""

# Configurar ambiente para ARM-Axion con optimizaciones para consenso seguro
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_USE_FLASHINFER=0
export TORCHINDUCTOR_DISABLED=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:disabled

# Verificar que no haya un servidor ya corriendo en el puerto 8085
if lsof -Pi :8085 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}❌ Error: Puerto 8085 ya está en uso${NC}"
    echo "   Verifica si el servidor de consenso seguro ya está corriendo:"
    echo "   ps aux | grep multi_model_server_consensus_safe"
    exit 1
fi

echo -e "${GREEN}✓${NC} Variables de entorno ARM-Axion configuradas para consenso seguro"

# Verificar que el archivo de configuración exista
if [ ! -f "arm-axion-optimizations/vllm_integration/config.json" ]; then
    echo -e "${RED}❌ Error: No se encuentra el archivo de configuración${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Archivo de configuración encontrado"

# Iniciar servidor de consenso seguro en segundo plano con logging
echo -e "${CYAN}🚀 Iniciando servidor de consenso seguro en modo daemon (puerto 8085)...${NC}"
cd arm-axion-optimizations/vllm_integration
nohup python3 multi_model_server_consensus_safe.py --host 0.0.0.0 --port 8085 --config config.json > /tmp/multi_model_server_consensus_safe.log 2>&1 &

SERVER_PID=$!
echo "PID del servidor de consenso seguro: $SERVER_PID" >> /tmp/multi_model_server_consensus_safe.log

# Esperar a que el servidor arranque (más tiempo para permitir lazy loading)
echo -e "${CYAN}⏳ Esperando que el servidor de consenso seguro inicie (90 segundos)...${NC}"
sleep 90

# Verificar que el servidor esté escuchando en el puerto
if ss -tlnp | grep ":8085" >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Servidor de consenso seguro escuchando en puerto 8085"
else
    echo -e "${RED}❌ Error: Servidor de consenso seguro no escucha en puerto 8085${NC}"
    echo "   Revisando logs..."
    tail -20 /tmp/multi_model_server_consensus_safe.log
    exit 1
fi

# Verificar estado de salud del servidor de consenso
echo -e "${CYAN}🏥 Verificando estado de salud del servidor de consenso seguro...${NC}"
if curl -s --connect-timeout 10 http://localhost:8085/health >/dev/null 2>&1; then
    HEALTH_RESPONSE=$(curl -s http://localhost:8085/health)
    echo -e "${GREEN}✓${NC} Servidor de consenso seguro saludable: $HEALTH_RESPONSE"
    
    # Verificar uso de RAM
    RAM_USAGE=$(echo $HEALTH_RESPONSE | jq -r '.ram_usage_percent')
    if (( $(echo "$RAM_USAGE > 80.0" | bc -l) )); then
        echo -e "${YELLOW}⚠️  RAM uso: ${RAM_USAGE}% - Superior al 80%${NC}"
    else
        echo -e "${GREEN}✓${NC} RAM uso: ${RAM_USAGE}% - Dentro de límites seguros${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Servidor de consenso seguro no responde a health check aún${NC}"
    echo "   Esto puede ser normal durante la carga inicial de modelos"
    echo "   El servidor de consenso seguro se inició correctamente con PID: $SERVER_PID"
    exit 0
fi

# Verificar que el consenso esté habilitado
if curl -s http://localhost:8085/stats | jq -e ".config.consensus_enabled" >/dev/null 2>&1; then
    CONSENSUS_ENABLED=$(curl -s http://localhost:8085/stats | jq -r ".config.consensus_enabled")
    if [ "$CONSENSUS_ENABLED" = "true" ]; then
        echo -e "${GREEN}✓${NC} Sistema de consenso habilitado en el servidor seguro"
    else
        echo -e "${YELLOW}⚠️  Sistema de consenso no está habilitado en el servidor seguro${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No se pudo verificar estado de consenso (respuesta inesperada)${NC}"
fi

# Verificar lazy loading
if curl -s http://localhost:8085/stats | jq -e ".config.lazy_loading.enabled" >/dev/null 2>&1; then
    LAZY_ENABLED=$(curl -s http://localhost:8085/stats | jq -r ".config.lazy_loading.enabled")
    if [ "$LAZY_ENABLED" = "true" ]; then
        echo -e "${GREEN}✓${NC} Lazy loading habilitado - Uso eficiente de RAM"
    else
        echo -e "${YELLOW}⚠️  Lazy loading no está habilitado${NC}"
    fi
fi

# Verificar modelos disponibles
echo -e "${CYAN}🤖 Verificando modelos disponibles en servidor de consenso seguro...${NC}"
MODELS_COUNT=$(curl -s http://localhost:8085/v1/models | jq ".data | length" 2>/dev/null || echo "Error")
if [ "$MODELS_COUNT" != "Error" ]; then
    echo -e "${GREEN}✓${NC} $MODELS_COUNT modelos disponibles en servidor de consenso seguro"
    curl -s http://localhost:8085/v1/models | jq ".data[].id"
else
    echo -e "${YELLOW}⚠️  No se pudo obtener lista de modelos en servidor de consenso seguro${NC}"
fi

echo ""
echo -e "${GREEN}✅ Servidor ARM-Axion con CONSENSO SEGURO iniciado exitosamente${NC}"
echo "   Puerto: 8085 (safe consensus server)"
echo "   Logs: /tmp/multi_model_server_consensus_safe.log"
echo "   Configuración: Lazy loading + RAM optimizations"
echo "   Optimizaciones: NEON, ACL, FP8 KV Cache, Flash Attention, SAFE CONSENSUS"
echo ""
echo -e "${CYAN}Endpoints disponibles (Safe Consensus):${NC}"
echo "   GET  http://localhost:8085/health - Verificar estado y uso de RAM"
echo "   GET  http://localhost:8085/v1/models - Lista de modelos"
echo "   POST http://localhost:8085/v1/chat/completions - API OpenAI con consenso seguro"
echo "   POST http://localhost:8085/v1/completions - API OpenAI completions con consenso seguro"
echo "   GET  http://localhost:8085/stats - Estadísticas del servidor con uso de RAM"
echo ""
echo -e "${CYAN}Diferencias con otros servidores:${NC}"
echo "   - Puerto 8082: Servidor estándar (respuesta completa)"
echo "   - Puerto 8083: Servidor con streaming verdadero (token por token)"
echo "   - Puerto 8084: Servidor con consenso (posible uso elevado de RAM)"
echo "   - Puerto 8085: Servidor con consenso seguro (lazy loading + control de RAM)"