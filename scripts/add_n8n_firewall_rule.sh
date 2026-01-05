#!/bin/bash
# Script para añadir regla de firewall para N8n en la VM gpt-oss-20b

echo "🔥 Añadiendo regla de firewall para N8n (puerto 5678) en gpt-oss-20b..."

# Crear regla de firewall para N8n
gcloud compute firewall-rules create allow-n8n \
    --project=mamba-001 \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --action=ALLOW \
    --rules=tcp:5678 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=gpt-oss-20b \
    --description="Allow N8n workflow automation on port 5678"

if [ $? -eq 0 ]; then
    echo "✅ Regla de firewall creada exitosamente"
    echo ""
    echo "📋 Detalles de la regla:"
    gcloud compute firewall-rules describe allow-n8n --project=mamba-001
    echo ""
    echo "🔗 N8n ahora debería ser accesible en: http://34.175.136.104:5678"
else
    echo "❌ Error al crear la regla de firewall"
    exit 1
fi

