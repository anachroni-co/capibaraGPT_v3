#!/bin/bash
# Script completo para verificar conexiones entre VMs y frontend
# Verifica servicios internos y externos

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🔍 Verificación Completa de Conexiones${NC}"
echo -e "${BLUE}========================================${NC}\n"

# IPs conocidas
BOUNTY2_IP="34.12.166.76"
RAG3_IP=""  # Por determinar
GPT_OSS_IP="34.175.136.104"

# Función para verificar conexión
check_connection() {
    local ip=$1
    local port=$2
    local service=$3
    local path=${4:-"/health"}
    
    if timeout 3 bash -c "echo > /dev/tcp/$ip/$port" 2>/dev/null; then
        echo -e "${GREEN}✅${NC} Puerto $port ($service) - ${GREEN}ABIERTO${NC}"
        
        # Probar HTTP si hay path
        if [ -n "$path" ]; then
            if curl -s --connect-timeout 3 "http://$ip:$port$path" > /dev/null 2>&1; then
                echo -e "   ${GREEN}   HTTP responde en $path${NC}"
                return 0
            else
                echo -e "   ${YELLOW}   Puerto abierto pero HTTP no responde${NC}"
                return 1
            fi
        fi
        return 0
    else
        echo -e "${RED}❌${NC} Puerto $port ($service) - ${RED}CERRADO${NC}"
        return 1
    fi
}

echo -e "${CYAN}📡 Verificando VM bounty2 ($BOUNTY2_IP)...${NC}"
check_connection "$BOUNTY2_IP" 11434 "Ollama" "/api/tags"
check_connection "$BOUNTY2_IP" 5001 "Backend Capibara6" "/api/health"
check_connection "$BOUNTY2_IP" 5000 "Backend alternativo" "/api/health"

echo ""
echo -e "${CYAN}📡 Verificando VM gpt-oss-20b ($GPT_OSS_IP)...${NC}"
check_connection "$GPT_OSS_IP" 5000 "Bridge/Main Server" "/api/health"
check_connection "$GPT_OSS_IP" 5002 "TTS Server" "/api/tts/voices"
check_connection "$GPT_OSS_IP" 5003 "MCP Server" "/api/mcp/status"
check_connection "$GPT_OSS_IP" 5010 "MCP Server Alt" "/health"
check_connection "$GPT_OSS_IP" 5678 "N8n" "/healthz"

if [ -n "$RAG3_IP" ]; then
    echo ""
    echo -e "${CYAN}📡 Verificando VM rag3 ($RAG3_IP)...${NC}"
    check_connection "$RAG3_IP" 8000 "RAG API" "/health"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🌐 Verificación de Conexiones Internas${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${CYAN}Para verificar conexiones internas entre VMs, ejecuta desde cada VM:${NC}"
echo ""
echo -e "${YELLOW}Desde bounty2:${NC}"
echo "  ping [IP_INTERNA_RAG3]"
echo "  ping [IP_INTERNA_GPT_OSS_20B]"
echo "  curl http://[IP_INTERNA_RAG3]:8000/health"
echo ""
echo -e "${YELLOW}Desde rag3:${NC}"
echo "  ping [IP_INTERNA_BOUNTY2]"
echo "  ping [IP_INTERNA_GPT_OSS_20B]"
echo ""
echo -e "${YELLOW}Desde gpt-oss-20b:${NC}"
echo "  ping [IP_INTERNA_BOUNTY2]"
echo "  ping [IP_INTERNA_RAG3]"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}📝 Resumen${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${CYAN}IPs configuradas:${NC}"
echo "  • bounty2: $BOUNTY2_IP"
echo "  • rag3: ${RAG3_IP:-No configurada}"
echo "  • gpt-oss-20b: $GPT_OSS_IP"

echo ""
echo -e "${CYAN}Para obtener IPs internas:${NC}"
echo "  gcloud compute instances describe <VM_NAME> --zone=<ZONE> --project=mamba-001 --format='value(networkInterfaces[0].networkIP)'"

echo ""
echo -e "${GREEN}✅ Verificación completada${NC}"

