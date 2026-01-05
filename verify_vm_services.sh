#!/bin/bash
# Script para verificar servicios en las VMs de Capibara6

echo "🔍 Verificando servicios en las VMs..."
echo ""

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# IPs (actualizar con IPs reales)
BOUNTY2_IP="34.12.166.76"
RAG3_IP="[OBTENER_IP]"
GPT_OSS_20B_IP="34.175.136.104"

# Función para verificar servicio
check_service() {
    local name=$1
    local url=$2
    local description=$3
    
    echo -n "Verificando $name ($description)... "
    if curl -s --connect-timeout 5 "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Activo${NC}"
        return 0
    else
        echo -e "${RED}❌ No responde${NC}"
        return 1
    fi
}

echo "=========================================="
echo "  VERIFICACIÓN DE SERVICIOS"
echo "=========================================="
echo ""

# Verificar servicios en bounty2
echo "📡 VM bounty2 ($BOUNTY2_IP):"
check_service "Backend Capibara6" "http://$BOUNTY2_IP:5001/health" "Puerto 5001"
check_service "Ollama" "http://$BOUNTY2_IP:11434/api/tags" "Puerto 11434"
echo ""

# Verificar servicios en gpt-oss-20b
echo "📡 VM gpt-oss-20b ($GPT_OSS_20B_IP):"
check_service "Servidor Principal" "http://$GPT_OSS_20B_IP:5000/health" "Puerto 5000"
check_service "MCP Server" "http://$GPT_OSS_20B_IP:5003/health" "Puerto 5003"
check_service "MCP Alt" "http://$GPT_OSS_20B_IP:5010/health" "Puerto 5010"
check_service "Servidor Llama" "http://$GPT_OSS_20B_IP:8080/health" "Puerto 8080"
echo ""

# Verificar servicios en rag3
if [ "$RAG3_IP" != "[OBTENER_IP]" ]; then
    echo "📡 VM rag3 ($RAG3_IP):"
    echo "  (Puertos a verificar)"
    echo ""
fi

echo "=========================================="
echo "  VERIFICACIÓN COMPLETA"
echo "=========================================="
echo ""
echo "💡 Para verificar servicios desde dentro de las VMs, ejecuta:"
echo "   gcloud compute ssh --zone \"europe-west4-a\" \"bounty2\" --project \"mamba-001\""
echo "   # Luego dentro de la VM:"
echo "   curl http://localhost:5001/health"
echo "   curl http://localhost:11434/api/tags"

