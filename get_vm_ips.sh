#!/bin/bash
# Script rápido para obtener IPs de las VMs

echo "🔍 Obteniendo IPs de las VMs..."
echo ""

# bounty2
echo "📡 bounty2 (europe-west4-a):"
gcloud compute instances describe bounty2 --zone=europe-west4-a --project=mamba-001 --format="get(networkInterfaces[0].accessConfigs[0].natIP,networkInterfaces[0].networkIP)" 2>/dev/null || echo "  Error obteniendo IP"

echo ""
echo "📡 rag3 (europe-west2-c):"
gcloud compute instances describe rag3 --zone=europe-west2-c --project=mamba-001 --format="get(networkInterfaces[0].accessConfigs[0].natIP,networkInterfaces[0].networkIP)" 2>/dev/null || echo "  Error obteniendo IP"

echo ""
echo "📡 gpt-oss-20b (europe-southwest1-b):"
gcloud compute instances describe gpt-oss-20b --zone=europe-southwest1-b --project=mamba-001 --format="get(networkInterfaces[0].accessConfigs[0].natIP,networkInterfaces[0].networkIP)" 2>/dev/null || echo "  Error obteniendo IP"

