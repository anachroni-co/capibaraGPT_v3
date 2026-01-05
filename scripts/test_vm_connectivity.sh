#!/bin/bash
# Script simple para probar conectividad a las VMs de Capibara6

echo "🔍 Probando conectividad a las VMs de Capibara6..."
echo ""

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Función para probar conexión
test_connection() {
    local url=$1
    local name=$2
    
    echo -n "Probando $name... "
    if curl -s --connect-timeout 5 "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Conectado${NC}"
        return 0
    else
        echo -e "${RED}❌ No conectado${NC}"
        return 1
    fi
}

echo "=== VM: gpt-oss-20b (34.175.136.104) ==="
test_connection "http://34.175.136.104:5000/api/health" "Servidor Principal (5000)"
test_connection "http://34.175.136.104:5003/api/mcp/status" "MCP Server (5003)"
test_connection "http://34.175.136.104:5010/api/mcp/analyze" "MCP Alternativo (5010)"
test_connection "http://34.175.136.104:8080/health" "Llama Server (8080)"
echo ""

echo "=== VM: bounty2 (34.12.166.76) ==="
test_connection "http://34.12.166.76:11434/api/tags" "Ollama (11434)"
test_connection "http://34.12.166.76:5001/api/health" "Backend Capibara6 (5001)"
echo ""

echo "=== Resumen ==="
echo "Para desarrollo local, el frontend está configurado para conectarse a:"
echo "  - Servidor Principal: http://34.175.136.104:5000"
echo "  - MCP Server: http://34.175.136.104:5003"
echo "  - Modelo: http://34.175.136.104:8080"
echo ""
echo "Para probar el frontend localmente:"
echo "  cd web && python3 -m http.server 8000"
echo "  Luego abrir: http://localhost:8000"

