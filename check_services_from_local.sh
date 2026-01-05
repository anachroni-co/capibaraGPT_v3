#!/bin/bash
# Script para verificar servicios desde el portátil local
# Usa las IPs conocidas de las VMs


# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# IPs conocidas (actualizar si es necesario)
MODELS_EUROPE_IP="34.175.48.2"    # models-europe VM
RAG_EUROPE_IP="34.175.110.120"    # rag-europe VM
SERVICES_IP="34.175.255.139"      # services VM

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🔍 Verificación de Servicios desde Local${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Función para verificar puerto
check_port() {
    local ip=$1
    local port=$2
    local service=$3
    
    if timeout 3 bash -c "echo > /dev/tcp/$ip/$port" 2>/dev/null; then
        echo -e "${GREEN}✅${NC} $service ($ip:$port) - ${GREEN}ACTIVO${NC}"
        return 0
    else
        echo -e "${RED}❌${NC} $service ($ip:$port) - ${RED}INACTIVO${NC}"
        return 1
    fi
}

# Función para verificar HTTP
check_http() {
    local ip=$1
    local port=$2
    local path=$3
    local service=$4
    
    if curl -s --connect-timeout 3 "http://$ip:$port$path" > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} $service HTTP ($ip:$port$path) - ${GREEN}RESPONDE${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️${NC} $service HTTP ($ip:$port$path) - ${YELLOW}No responde${NC}"
        return 1
    fi
}

echo -e "${CYAN}📡 Verificando VM models-europe ($MODELS_EUROPE_IP)...${NC}"
check_port "$MODELS_EUROPE_IP" 11434 "Ollama"
check_port "$MODELS_EUROPE_IP" 5001 "Backend Capibara6"
check_port "$MODELS_EUROPE_IP" 5000 "Backend alternativo"
check_http "$MODELS_EUROPE_IP" 11434 "/api/tags" "Ollama API"
check_http "$MODELS_EUROPE_IP" 5001 "/api/health" "Backend Health"

echo ""
echo -e "${CYAN}📡 Verificando VM services ($SERVICES_IP)...${NC}"
check_port "$SERVICES_IP" 5000 "Bridge/Main Server"
check_port "$SERVICES_IP" 5002 "TTS Server"
check_port "$SERVICES_IP" 5003 "MCP Server"
check_port "$SERVICES_IP" 5010 "MCP Server (alternativo)"
check_port "$SERVICES_IP" 5678 "N8n"
check_port "$SERVICES_IP" 8080 "Modelo"
check_http "$SERVICES_IP" 5000 "/api/health" "Bridge Health"
check_http "$SERVICES_IP" 5003 "/api/mcp/status" "MCP Status"
check_http "$SERVICES_IP" 5678 "/healthz" "N8n Health"

if [ -n "$RAG_EUROPE_IP" ]; then
    echo ""
    echo -e "${CYAN}📡 Verificando VM rag-europe ($RAG_EUROPE_IP)...${NC}"
    check_port "$RAG_EUROPE_IP" 8000 "RAG API"
    check_http "$RAG_EUROPE_IP" 8000 "/health" "RAG Health"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}📝 Resumen${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${YELLOW}IPs configuradas:${NC}"
echo "  • models-europe: $MODELS_EUROPE_IP"
echo "  • rag-europe: $RAG_EUROPE_IP"
echo "  • services: $SERVICES_IP"

echo ""
echo -e "${YELLOW}Para obtener IPs actualizadas, ejecuta:${NC}"
echo "  gcloud compute instances list --project=mamba-001"

