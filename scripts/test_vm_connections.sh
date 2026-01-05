#!/bin/bash
# Script para probar conexiones entre VMs
# Ejecutar desde cualquier VM o desde tu portátil

set -e

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# IPs (usar IPs internas para comunicación entre VMs, externas para acceso desde fuera)
BOUNTY2_EXTERNAL="34.12.166.76"
BOUNTY2_INTERNAL=""  # Se obtendrá automáticamente
RAG3_EXTERNAL=""
RAG3_INTERNAL=""  # Se obtendrá automáticamente
GPTOSS_EXTERNAL="34.175.136.104"
GPTOSS_INTERNAL=""  # Se obtendrá automáticamente

# Determinar si estamos en una VM o fuera
IS_VM=false
if [ -f /sys/class/dmi/id/product_name ] || [ -n "$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/name 2>/dev/null)" ]; then
    IS_VM=true
    echo -e "${BLUE}Ejecutando desde dentro de una VM${NC}"
    echo "Usando IPs internas para comunicación entre VMs"
else
    echo -e "${BLUE}Ejecutando desde fuera de GCloud${NC}"
    echo "Usando IPs externas"
fi

# Obtener IPs internas si estamos en una VM
if [ "$IS_VM" = true ]; then
    BOUNTY2_INTERNAL=$(gcloud compute instances describe bounty2 --zone=europe-west4-a --project=mamba-001 --format="value(networkInterfaces[0].networkIP)" 2>/dev/null || echo "")
    RAG3_INTERNAL=$(gcloud compute instances describe rag3 --zone=europe-west2-c --project=mamba-001 --format="value(networkInterfaces[0].networkIP)" 2>/dev/null || echo "")
    GPTOSS_INTERNAL=$(gcloud compute instances describe gpt-oss-20b --zone=europe-southwest1-b --project=mamba-001 --format="value(networkInterfaces[0].networkIP)" 2>/dev/null || echo "")
fi

echo "================================================================================"
echo "PRUEBAS DE CONEXION ENTRE VMs - CAPIBARA6"
echo "================================================================================"
echo ""

# Función para probar conexión
test_connection() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -n "  $name... "
    
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$url" 2>/dev/null || echo "000")
    
    if [ "$http_code" = "$expected_status" ] || [ "$http_code" = "200" ]; then
        echo -e "${GREEN}OK (HTTP $http_code)${NC}"
        return 0
    elif [ "$http_code" = "404" ]; then
        echo -e "${YELLOW}WARNING - Servicio responde pero endpoint no encontrado (404)${NC}"
        return 2
    elif [ "$http_code" = "000" ]; then
        echo -e "${RED}ERROR - No responde${NC}"
        return 1
    else
        echo -e "${RED}ERROR (HTTP $http_code)${NC}"
        return 1
    fi
}

# Determinar qué IPs usar
if [ "$IS_VM" = true ]; then
    BOUNTY2_IP=${BOUNTY2_INTERNAL:-$BOUNTY2_EXTERNAL}
    RAG3_IP=${RAG3_INTERNAL:-$RAG3_EXTERNAL}
    GPTOSS_IP=${GPTOSS_INTERNAL:-$GPTOSS_EXTERNAL}
    echo "Usando IPs internas para mejor rendimiento"
else
    BOUNTY2_IP=$BOUNTY2_EXTERNAL
    RAG3_IP=$RAG3_EXTERNAL
    GPTOSS_IP=$GPTOSS_EXTERNAL
    echo "Usando IPs externas"
fi

echo ""
echo "IPs configuradas:"
echo "  Bounty2: $BOUNTY2_IP"
echo "  rag3: ${RAG3_IP:-NO CONFIGURADA}"
echo "  gpt-oss-20b: $GPTOSS_IP"
echo ""

echo "================================================================================"
echo "PRUEBAS DESDE BOUNTY2"
echo "================================================================================"
echo ""

# Pruebas desde Bounty2 (si estamos en Bounty2 o desde fuera)
if [ "$IS_VM" = false ] || [ "$(hostname)" = "bounty2" ]; then
    echo "1. Servicios locales en Bounty2:"
    test_connection "Ollama API" "http://localhost:11434/api/tags"
    test_connection "Backend Flask" "http://localhost:5001/api/health"
    
    echo ""
    echo "2. Conexión a rag3:"
    if [ -n "$RAG3_IP" ]; then
        test_connection "RAG API" "http://$RAG3_IP:8000/health"
    else
        echo -e "${YELLOW}  rag3 no configurado${NC}"
    fi
    
    echo ""
    echo "3. Conexión a gpt-oss-20b:"
    test_connection "TTS" "http://$GPTOSS_IP:5002/api/tts/voices"
    test_connection "MCP Server" "http://$GPTOSS_IP:5003/api/mcp/status"
    test_connection "N8n" "http://$GPTOSS_IP:5678/healthz"
    test_connection "Bridge" "http://$GPTOSS_IP:5000/api/health"
fi

echo ""
echo "================================================================================"
echo "PRUEBAS DESDE RAG3"
echo "================================================================================"
echo ""

# Pruebas desde rag3 (si estamos en rag3 o desde fuera)
if [ "$IS_VM" = false ] || [ "$(hostname)" = "rag3" ]; then
    echo "1. Servicios locales en rag3:"
    test_connection "RAG API" "http://localhost:8000/health"
    
    echo ""
    echo "2. Conexión a Bounty2:"
    test_connection "Ollama API" "http://$BOUNTY2_IP:11434/api/tags"
    test_connection "Backend Flask" "http://$BOUNTY2_IP:5001/api/health"
    
    echo ""
    echo "3. Conexión a gpt-oss-20b:"
    test_connection "TTS" "http://$GPTOSS_IP:5002/api/tts/voices"
    test_connection "MCP Server" "http://$GPTOSS_IP:5003/api/mcp/status"
fi

echo ""
echo "================================================================================"
echo "PRUEBAS DESDE GPT-OSS-20B"
echo "================================================================================"
echo ""

# Pruebas desde gpt-oss-20b (si estamos en gpt-oss-20b o desde fuera)
if [ "$IS_VM" = false ] || [ "$(hostname | grep -i gpt)" ]; then
    echo "1. Servicios locales en gpt-oss-20b:"
    test_connection "TTS" "http://localhost:5002/api/tts/voices"
    test_connection "MCP Server" "http://localhost:5003/api/mcp/status"
    test_connection "N8n" "http://localhost:5678/healthz"
    test_connection "Bridge" "http://localhost:5000/api/health"
    
    echo ""
    echo "2. Conexión a Bounty2:"
    test_connection "Ollama API" "http://$BOUNTY2_IP:11434/api/tags"
    test_connection "Backend Flask" "http://$BOUNTY2_IP:5001/api/health"
    
    echo ""
    echo "3. Conexión a rag3:"
    if [ -n "$RAG3_IP" ]; then
        test_connection "RAG API" "http://$RAG3_IP:8000/health"
    else
        echo -e "${YELLOW}  rag3 no configurado${NC}"
    fi
fi

echo ""
echo "================================================================================"
echo "RESUMEN"
echo "================================================================================"
echo ""
echo "Pruebas completadas. Revisa los resultados arriba."
echo ""
echo "Si hay errores de conexión:"
echo "  1. Verifica que los servicios están corriendo en cada VM"
echo "  2. Verifica las reglas de firewall"
echo "  3. Verifica que los servicios escuchan en 0.0.0.0"
echo ""

