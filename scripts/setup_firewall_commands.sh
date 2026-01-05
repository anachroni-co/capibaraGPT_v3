#!/bin/bash
# Script que genera los comandos exactos para configurar el firewall
# Ejecuta este script y copia los comandos generados

PROJECT="mamba-001"

echo "================================================================================"
echo "COMANDOS PARA CONFIGURAR FIREWALL - CAPIBARA6"
echo "================================================================================"
echo ""

# Obtener IP pública
echo "Obteniendo tu IP pública..."
USER_IP=$(curl -s https://api.ipify.org 2>/dev/null || echo "")

# Obtener red VPC
NETWORK=$(gcloud compute networks list --project=$PROJECT --format="value(name)" 2>/dev/null | head -1 || echo "default")

echo "Tu IP pública: ${USER_IP:-NO DETECTADA}"
echo "Red VPC: $NETWORK"
echo ""

echo "================================================================================"
echo "COMANDO 1: Regla para comunicación INTERNA entre VMs"
echo "================================================================================"
echo ""
echo "gcloud compute firewall-rules create allow-capibara6-internal \\"
echo "  --project=$PROJECT \\"
echo "  --network=$NETWORK \\"
echo "  --allow=tcp:11434,tcp:5000,tcp:5001,tcp:5002,tcp:5003,tcp:5010,tcp:5678,tcp:8000 \\"
echo "  --source-ranges=10.0.0.0/8 \\"
echo "  --description=\"Permitir comunicación interna entre VMs de Capibara6\" \\"
echo "  --direction=INGRESS \\"
echo "  --priority=1000"
echo ""

if [ -n "$USER_IP" ]; then
    echo "================================================================================"
    echo "COMANDO 2: Regla para acceso EXTERNO desde tu IP"
    echo "================================================================================"
    echo ""
    echo "gcloud compute firewall-rules create allow-capibara6-external-dev \\"
    echo "  --project=$PROJECT \\"
    echo "  --network=$NETWORK \\"
    echo "  --allow=tcp:11434,tcp:5000,tcp:5001,tcp:5002,tcp:5003,tcp:5010,tcp:5678,tcp:8000 \\"
    echo "  --source-ranges=$USER_IP/32 \\"
    echo "  --description=\"Permitir acceso a servicios Capibara6 desde desarrollo local\" \\"
    echo "  --direction=INGRESS \\"
    echo "  --priority=1000"
    echo ""
else
    echo "================================================================================"
    echo "COMANDO 2: Regla para acceso EXTERNO (reemplaza TU_IP con tu IP pública)"
    echo "================================================================================"
    echo ""
    echo "gcloud compute firewall-rules create allow-capibara6-external-dev \\"
    echo "  --project=$PROJECT \\"
    echo "  --network=$NETWORK \\"
    echo "  --allow=tcp:11434,tcp:5000,tcp:5001,tcp:5002,tcp:5003,tcp:5010,tcp:5678,tcp:8000 \\"
    echo "  --source-ranges=TU_IP/32 \\"
    echo "  --description=\"Permitir acceso a servicios Capibara6 desde desarrollo local\" \\"
    echo "  --direction=INGRESS \\"
    echo "  --priority=1000"
    echo ""
    echo "Para obtener tu IP pública: curl https://api.ipify.org"
    echo ""
fi

echo "================================================================================"
echo "COMANDO 3: Verificar reglas creadas"
echo "================================================================================"
echo ""
echo "gcloud compute firewall-rules list \\"
echo "  --project=$PROJECT \\"
echo "  --filter=\"name~allow-capibara6\" \\"
echo "  --format=\"table(name,network,direction,sourceRanges.list():label=SOURCE_RANGES,allowed[].map().firewall_rule().list():label=ALLOW)\""
echo ""

