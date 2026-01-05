#!/bin/bash
# Script para configurar firewall de GCP para N8N en gpt-oss-20b

PROJECT="mamba-001"
VM_NAME="gpt-oss-20b"
ZONE="europe-southwest1-b"
N8N_PORT="5678"
FIREWALL_RULE_NAME="allow-n8n-${N8N_PORT}"

echo "🔥 Configurando firewall para N8N (puerto $N8N_PORT)..."
echo ""

# Verificar si la regla ya existe
if gcloud compute firewall-rules describe $FIREWALL_RULE_NAME --project=$PROJECT &>/dev/null; then
    echo "✅ Regla de firewall '$FIREWALL_RULE_NAME' ya existe"
    echo "📋 Detalles de la regla:"
    gcloud compute firewall-rules describe $FIREWALL_RULE_NAME --project=$PROJECT --format="yaml(allowed,sourceRanges,targetTags)" 2>&1
else
    echo "➕ Creando regla de firewall '$FIREWALL_RULE_NAME'..."
    gcloud compute firewall-rules create $FIREWALL_RULE_NAME \
        --project=$PROJECT \
        --allow tcp:$N8N_PORT \
        --source-ranges 0.0.0.0/0 \
        --description "Allow N8N access on port $N8N_PORT" \
        --target-tags n8n-server 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ Regla de firewall creada exitosamente"
    else
        echo "❌ Error creando regla de firewall"
        exit 1
    fi
fi

echo ""
echo "🏷️  Añadiendo tag 'n8n-server' a la VM $VM_NAME..."
gcloud compute instances add-tags $VM_NAME \
    --zone=$ZONE \
    --project=$PROJECT \
    --tags n8n-server 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Tag añadido exitosamente"
else
    echo "⚠️  Error añadiendo tag (puede que ya exista)"
fi

echo ""
echo "🔍 Verificando tags actuales de la VM:"
gcloud compute instances describe $VM_NAME --zone=$ZONE --project=$PROJECT --format="get(tags.items)" 2>&1

echo ""
echo "✅ Configuración completada"
echo ""
echo "📝 Próximos pasos:"
echo "   1. Verificar que N8N está corriendo en la VM:"
echo "      gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT --command='sudo systemctl status n8n.service'"
echo ""
echo "   2. Si N8N no está corriendo, iniciarlo:"
echo "      gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT --command='sudo systemctl start n8n.service'"
echo ""
echo "   3. Probar conexión desde local:"
echo "      curl http://34.175.136.104:$N8N_PORT/healthz"

