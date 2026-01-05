#!/bin/bash

# Script para verificar la conexión con el servicio RAG en rag3
# IP de rag3: 34.105.131.8

echo "========================================="
echo "🧪 Test de Conexión con RAG3"
echo "========================================="
echo ""

RAG3_IP="34.105.131.8"
RAG3_PORT="8000"
RAG3_URL="http://${RAG3_IP}:${RAG3_PORT}"

echo "📡 Probando conexión con RAG3..."
echo "   URL: ${RAG3_URL}"
echo ""

# Test 1: Ping básico
echo "1️⃣ Test básico de conectividad..."
if curl -s --connect-timeout 5 "${RAG3_URL}" > /dev/null 2>&1; then
    echo "   ✅ RAG3 responde"
else
    echo "   ❌ RAG3 no responde"
    echo "   ℹ️  Verifica que el servicio esté corriendo en la VM rag3"
fi
echo ""

# Test 2: Health endpoint
echo "2️⃣ Test de health endpoint..."
HEALTH_RESPONSE=$(curl -s --connect-timeout 5 "${RAG3_URL}/health" 2>&1)
if [ $? -eq 0 ]; then
    echo "   ✅ Health endpoint responde:"
    echo "   ${HEALTH_RESPONSE}" | jq '.' 2>/dev/null || echo "   ${HEALTH_RESPONSE}"
else
    echo "   ❌ Health endpoint no responde"
fi
echo ""

# Test 3: API endpoint
echo "3️⃣ Test de API endpoint..."
API_RESPONSE=$(curl -s --connect-timeout 5 "${RAG3_URL}/api" 2>&1)
if [ $? -eq 0 ]; then
    echo "   ✅ API endpoint responde:"
    echo "   ${API_RESPONSE}" | jq '.' 2>/dev/null || echo "   ${API_RESPONSE}"
else
    echo "   ❌ API endpoint no responde"
fi
echo ""

# Test 4: Verificar puertos abiertos
echo "4️⃣ Verificando puertos abiertos en RAG3..."
if command -v nc &> /dev/null; then
    if nc -z -v -w5 ${RAG3_IP} ${RAG3_PORT} 2>&1 | grep -q "succeeded"; then
        echo "   ✅ Puerto ${RAG3_PORT} está abierto"
    else
        echo "   ❌ Puerto ${RAG3_PORT} está cerrado o no accesible"
    fi
else
    echo "   ⚠️  netcat (nc) no está instalado, saltando prueba de puerto"
fi
echo ""

# Test 5: Test de mensaje (simulación)
echo "5️⃣ Test de guardar mensaje (simulación)..."
TEST_MESSAGE=$(cat <<EOF
{
  "session_id": "test_session_$(date +%s)",
  "content": "Test message from connection script",
  "message_role": "user",
  "metadata": {
    "user_id": "test_user",
    "timestamp": "$(date -Iseconds)"
  }
}
EOF
)

MESSAGE_RESPONSE=$(curl -s --connect-timeout 5 -X POST "${RAG3_URL}/api/messages" \
    -H "Content-Type: application/json" \
    -d "${TEST_MESSAGE}" 2>&1)

if [ $? -eq 0 ]; then
    echo "   ✅ Endpoint de mensajes responde:"
    echo "   ${MESSAGE_RESPONSE}" | jq '.' 2>/dev/null || echo "   ${MESSAGE_RESPONSE}"
else
    echo "   ❌ Endpoint de mensajes no responde"
fi
echo ""

# Resumen
echo "========================================="
echo "📊 Resumen de Conexión con RAG3"
echo "========================================="
echo "IP: ${RAG3_IP}"
echo "Puerto: ${RAG3_PORT}"
echo "URL: ${RAG3_URL}"
echo ""
echo "Para más información, ejecuta en la VM rag3:"
echo "  gcloud compute ssh --zone 'europe-west2-c' 'rag3' --project 'mamba-001'"
echo ""

