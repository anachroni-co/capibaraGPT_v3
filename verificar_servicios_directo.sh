#!/bin/bash
# Script que ejecuta comandos directos sin esperar mucho

PROJECT="mamba-001"

echo "🔍 Verificación Directa de Servicios"
echo "======================================"
echo ""

# Verificar si podemos listar VMs primero
echo "1️⃣ Verificando acceso a proyecto..."
if timeout 10 gcloud compute instances list --project="$PROJECT" --limit=1 > /dev/null 2>&1; then
    echo "   ✅ Acceso OK"
else
    echo "   ❌ No se puede acceder al proyecto"
    exit 1
fi

echo ""
echo "2️⃣ Verificando bounty2 (timeout 20s)..."
timeout 20 gcloud compute ssh bounty2 \
    --zone=europe-west4-a \
    --project="$PROJECT" \
    --command="echo 'Conectado' && ps aux | grep -E '(python|ollama)' | grep -v grep | head -2" \
    2>&1 | grep -E "(Conectado|python|ollama|ERROR|error)" | head -5

echo ""
echo "3️⃣ Verificando rag3 (timeout 20s)..."
timeout 20 gcloud compute ssh rag3 \
    --zone=europe-west2-c \
    --project="$PROJECT" \
    --command="echo 'Conectado' && ps aux | grep python | grep -v grep | head -2" \
    2>&1 | grep -E "(Conectado|python|ERROR|error)" | head -5

echo ""
echo "4️⃣ Verificando gpt-oss-20b (timeout 20s)..."
timeout 20 gcloud compute ssh gpt-oss-20b \
    --zone=europe-southwest1-b \
    --project="$PROJECT" \
    --command="echo 'Conectado' && ps aux | grep python | grep -v grep | head -2" \
    2>&1 | grep -E "(Conectado|python|ERROR|error)" | head -5

echo ""
echo "✅ Verificación completada"

