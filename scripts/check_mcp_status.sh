#!/bin/bash

# Script para verificar el estado del servidor MCP en gpt-oss-20b
# El MCP Server DEBE estar en gpt-oss-20b, NO en bounty2

echo "========================================="
echo "🔍 Verificación del Servidor MCP"
echo "========================================="
echo ""

MCP_VM="gpt-oss-20b"
MCP_ZONE="europe-southwest1-b"
MCP_IP="34.175.136.104"
MCP_PORT="5010"
MCP_URL="http://${MCP_IP}:${MCP_PORT}"

echo "📍 Ubicación Correcta del MCP Server:"
echo "   VM: ${MCP_VM}"
echo "   IP: ${MCP_IP}"
echo "   Puerto: ${MCP_PORT}"
echo "   ⚠️  NOTA: El MCP debe estar en gpt-oss-20b, NO en bounty2"
echo ""

# Test 1: Conectividad básica
echo "1️⃣ Test de conectividad básica..."
if curl -s --connect-timeout 5 "${MCP_URL}/health" > /dev/null 2>&1; then
    echo "   ✅ MCP Server responde en ${MCP_URL}"
else
    echo "   ❌ MCP Server NO responde en ${MCP_URL}"
    echo "   ℹ️  El servidor MCP no está corriendo o no es accesible"
fi
echo ""

# Test 2: Health endpoint
echo "2️⃣ Test de health endpoint..."
HEALTH_RESPONSE=$(curl -s --connect-timeout 5 "${MCP_URL}/health" 2>&1)
if [ $? -eq 0 ]; then
    echo "   ✅ Health endpoint responde:"
    echo "   ${HEALTH_RESPONSE}" | jq '.' 2>/dev/null || echo "   ${HEALTH_RESPONSE}"
else
    echo "   ❌ Health endpoint no responde"
    echo "   ℹ️  Posibles causas:"
    echo "      - El servidor MCP no está corriendo en gpt-oss-20b"
    echo "      - El firewall bloquea el puerto 5010"
    echo "      - El servicio está en un puerto diferente"
fi
echo ""

# Test 3: Analyze endpoint
echo "3️⃣ Test de analyze endpoint..."
ANALYZE_RESPONSE=$(curl -s --connect-timeout 5 -X POST "${MCP_URL}/api/mcp/analyze" \
    -H "Content-Type: application/json" \
    -d '{"query": "test"}' 2>&1)
if [ $? -eq 0 ]; then
    echo "   ✅ Analyze endpoint responde:"
    echo "   ${ANALYZE_RESPONSE}" | jq '.' 2>/dev/null || echo "   ${ANALYZE_RESPONSE}"
else
    echo "   ❌ Analyze endpoint no responde"
fi
echo ""

# Test 4: Verificar puerto abierto
echo "4️⃣ Verificando puerto ${MCP_PORT}..."
if command -v nc &> /dev/null; then
    if nc -z -v -w5 ${MCP_IP} ${MCP_PORT} 2>&1 | grep -q "succeeded"; then
        echo "   ✅ Puerto ${MCP_PORT} está abierto"
    else
        echo "   ❌ Puerto ${MCP_PORT} está cerrado o no accesible"
        echo "   ℹ️  Verifica el firewall de GCloud"
    fi
else
    echo "   ⚠️  netcat (nc) no está instalado, saltando prueba de puerto"
fi
echo ""

# Test 5: Verificar firewall de GCloud
echo "5️⃣ Verificando firewall de GCloud..."
echo "   Ejecutando: gcloud compute firewall-rules list | grep 5010"
FIREWALL_RULES=$(gcloud compute firewall-rules list --project=mamba-001 2>/dev/null | grep 5010)
if [ -n "$FIREWALL_RULES" ]; then
    echo "   ✅ Regla de firewall para puerto 5010 encontrada:"
    echo "   $FIREWALL_RULES"
else
    echo "   ❌ NO se encontró regla de firewall para puerto 5010"
    echo "   ℹ️  Crear regla con:"
    echo "      gcloud compute firewall-rules create allow-smart-mcp-5010 \\"
    echo "        --project=mamba-001 \\"
    echo "        --direction=INGRESS \\"
    echo "        --priority=1000 \\"
    echo "        --network=default \\"
    echo "        --action=ALLOW \\"
    echo "        --rules=tcp:5010 \\"
    echo "        --source-ranges=0.0.0.0/0 \\"
    echo "        --description='Smart MCP Server en gpt-oss-20b'"
fi
echo ""

# Resumen
echo "========================================="
echo "📊 Resumen de Verificación"
echo "========================================="
echo "VM: ${MCP_VM} (${MCP_IP})"
echo "Puerto: ${MCP_PORT}"
echo "URL: ${MCP_URL}"
echo ""

# Determinar estado general
if curl -s --connect-timeout 3 "${MCP_URL}/health" > /dev/null 2>&1; then
    echo "✅ ESTADO: MCP Server está corriendo y accesible"
    echo ""
    echo "🎯 Próximos pasos:"
    echo "   1. Verificar que el frontend use la URL correcta"
    echo "   2. Probar desde el navegador: http://localhost:8000/chat.html"
else
    echo "❌ ESTADO: MCP Server NO está accesible"
    echo ""
    echo "🔧 Para solucionar:"
    echo ""
    echo "1. Conectarse a la VM gpt-oss-20b:"
    echo "   gcloud compute ssh --zone 'europe-southwest1-b' 'gpt-oss-20b' --project 'mamba-001'"
    echo ""
    echo "2. Verificar si el servidor está corriendo:"
    echo "   ps aux | grep smart_mcp_server"
    echo "   sudo netstat -tulpn | grep :5010"
    echo ""
    echo "3. Si NO está corriendo, iniciarlo:"
    echo "   cd /path/to/capibara6/backend"
    echo "   screen -S smart-mcp"
    echo "   python3 smart_mcp_server.py --port 5010"
    echo "   # Presionar Ctrl+A, D para desconectar"
    echo ""
    echo "4. Verificar firewall (si es necesario):"
    echo "   gcloud compute firewall-rules list --project=mamba-001 | grep 5010"
    echo ""
    echo "5. Probar nuevamente:"
    echo "   curl http://34.175.136.104:5010/health"
fi
echo ""
echo "========================================="

