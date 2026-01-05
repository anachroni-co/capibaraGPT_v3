#!/bin/bash
# Script para verificar todas las conexiones del frontend con los servicios

set -e

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# IPs desde config.js (actualizar si cambian)
BOUNTY2_IP="34.12.166.76"
GPTOSS_IP="34.175.136.104"
RAG3_IP=""  # Actualizar cuando se tenga

echo "================================================================================"
echo "VERIFICACION DE CONEXIONES FRONTEND - CAPIBARA6"
echo "================================================================================"
echo ""
echo "IPs configuradas:"
echo "  Bounty2: $BOUNTY2_IP"
echo "  gpt-oss-20b: $GPTOSS_IP"
echo "  rag3: ${RAG3_IP:-NO CONFIGURADA}"
echo ""
echo "Probando servicios..."
echo "--------------------------------------------------------------------------------"

# Función para probar un endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -n "  $name... "
    
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$url" 2>/dev/null || echo "000")
    
    if [ "$http_code" = "$expected_status" ] || [ "$http_code" = "200" ] || [ "$http_code" = "404" ]; then
        if [ "$http_code" = "000" ]; then
            echo -e "${RED}ERROR - No responde${NC}"
            return 1
        elif [ "$http_code" = "404" ]; then
            echo -e "${YELLOW}WARNING - Servicio responde pero endpoint no encontrado (404)${NC}"
            return 2
        else
            echo -e "${GREEN}OK (HTTP $http_code)${NC}"
            return 0
        fi
    else
        echo -e "${RED}ERROR (HTTP $http_code)${NC}"
        return 1
    fi
}

# Contadores
total=0
success=0
failed=0
warning=0

echo ""
echo "BOUNTY2 (Ollama + Backend):"
echo "--------------------------------------------------------------------------------"

# Ollama
total=$((total + 1))
if test_endpoint "Ollama API" "http://$BOUNTY2_IP:11434/api/tags"; then
    success=$((success + 1))
else
    failed=$((failed + 1))
fi

# Backend Flask
total=$((total + 1))
if test_endpoint "Backend Flask" "http://$BOUNTY2_IP:5001/api/health"; then
    success=$((success + 1))
else
    failed=$((failed + 1))
fi

echo ""
echo "GPT-OSS-20B (TTS, MCP, N8n, Bridge):"
echo "--------------------------------------------------------------------------------"

# TTS
total=$((total + 1))
result=$(test_endpoint "TTS" "http://$GPTOSS_IP:5002/api/tts/voices"; echo $?)
if [ "$result" = "0" ]; then
    success=$((success + 1))
elif [ "$result" = "2" ]; then
    warning=$((warning + 1))
else
    failed=$((failed + 1))
fi

# MCP
total=$((total + 1))
if test_endpoint "MCP Server" "http://$GPTOSS_IP:5003/api/mcp/status"; then
    success=$((success + 1))
else
    failed=$((failed + 1))
fi

# MCP Alt
total=$((total + 1))
if test_endpoint "MCP Server Alt" "http://$GPTOSS_IP:5010/api/mcp/status"; then
    success=$((success + 1))
else
    failed=$((failed + 1))
fi

# N8n
total=$((total + 1))
if test_endpoint "N8n" "http://$GPTOSS_IP:5678/healthz"; then
    success=$((success + 1))
else
    failed=$((failed + 1))
fi

# Bridge
total=$((total + 1))
if test_endpoint "Bridge" "http://$GPTOSS_IP:5000/api/health"; then
    success=$((success + 1))
else
    failed=$((failed + 1))
fi

# RAG3 (si está configurado)
if [ -n "$RAG3_IP" ]; then
    echo ""
    echo "RAG3 (RAG API):"
    echo "--------------------------------------------------------------------------------"
    total=$((total + 1))
    if test_endpoint "RAG API" "http://$RAG3_IP:8000/health"; then
        success=$((success + 1))
    else
        failed=$((failed + 1))
    fi
fi

# Resumen
echo ""
echo "================================================================================"
echo "RESUMEN"
echo "================================================================================"
echo -e "Total de servicios probados: ${BLUE}$total${NC}"
echo -e "${GREEN}Exitosos: $success${NC}"
if [ $warning -gt 0 ]; then
    echo -e "${YELLOW}Con advertencias: $warning${NC}"
fi
echo -e "${RED}Fallidos: $failed${NC}"
echo ""

if [ $failed -eq 0 ] && [ $warning -eq 0 ]; then
    echo -e "${GREEN}TODOS LOS SERVICIOS ESTAN FUNCIONANDO CORRECTAMENTE${NC}"
    exit 0
elif [ $failed -eq 0 ]; then
    echo -e "${YELLOW}Algunos servicios tienen advertencias pero estan accesibles${NC}"
    exit 0
else
    echo -e "${RED}ALGUNOS SERVICIOS NO ESTAN ACCESIBLES${NC}"
    echo ""
    echo "Recomendaciones:"
    echo "1. Verifica que los servicios estan corriendo en cada VM:"
    echo "   bash scripts/check_services_on_vm.sh"
    echo ""
    echo "2. Verifica las reglas de firewall:"
    echo "   gcloud compute firewall-rules list --project=mamba-001"
    echo ""
    echo "3. Verifica que los puertos estan abiertos y los servicios escuchan en 0.0.0.0"
    exit 1
fi

