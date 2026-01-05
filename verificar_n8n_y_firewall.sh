#!/bin/bash
# Script para verificar N8N y firewall en gpt-oss-20b

VM_NAME="gpt-oss-20b"
ZONE="europe-southwest1-b"
PROJECT="mamba-001"
VM_IP="34.175.136.104"
N8N_PORT="5678"

echo "🔍 Verificando N8N y firewall en $VM_NAME..."
echo ""

# 1. Verificar si N8N está corriendo en la VM
echo "1️⃣ Verificando si N8N está corriendo..."
gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT --command="
    echo '📊 Procesos de N8N:'
    ps aux | grep -i n8n | grep -v grep || echo '❌ N8N no está corriendo'
    echo ''
    echo '📡 Puertos abiertos:'
    sudo netstat -tlnp | grep $N8N_PORT || sudo ss -tlnp | grep $N8N_PORT || echo '⚠️ Puerto $N8N_PORT no está escuchando'
    echo ''
    echo '🔧 Estado del servicio systemd:'
    sudo systemctl status n8n.service --no-pager -l || echo '⚠️ Servicio n8n.service no encontrado'
" 2>&1 | head -30

echo ""
echo "2️⃣ Verificando firewall de GCP..."
echo "   Reglas de firewall para puerto $N8N_PORT:"
gcloud compute firewall-rules list --project=$PROJECT --filter="allowed.ports:$N8N_PORT" --format="table(name,allowed.ports,sourceRanges,targetTags)" 2>&1 | head -10

echo ""
echo "3️⃣ Verificando tags de red de la VM..."
gcloud compute instances describe $VM_NAME --zone=$ZONE --project=$PROJECT --format="get(tags.items)" 2>&1

echo ""
echo "4️⃣ Probando conexión desde local a N8N..."
timeout 5 curl -v http://$VM_IP:$N8N_PORT/healthz 2>&1 | head -10 || echo "❌ Timeout o conexión rechazada"

echo ""
echo "5️⃣ Verificando si N8N responde localmente en la VM..."
gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT --command="
    curl -s http://localhost:$N8N_PORT/healthz || echo '❌ N8N no responde localmente'
" 2>&1

echo ""
echo "✅ Verificación completada"

