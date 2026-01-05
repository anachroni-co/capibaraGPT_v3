#!/bin/bash
# Script simple para probar conexión gcloud

PROJECT="mamba-001"

echo "🧪 Probando conexión básica con gcloud..."
echo ""

echo "1️⃣ Probando listar VMs (timeout 15s)..."
timeout 15 gcloud compute instances list --project="$PROJECT" --format="table(name,zone,status)" 2>&1 | head -10
echo ""

echo "2️⃣ Probando conexión SSH a bounty2 (timeout 20s)..."
timeout 20 gcloud compute ssh bounty2 --zone=europe-west4-a --project="$PROJECT" --command="echo '✅ Conexión OK' && hostname" 2>&1 | head -5
echo ""

echo "3️⃣ Probando conexión SSH a gpt-oss-20b (timeout 20s)..."
timeout 20 gcloud compute ssh gpt-oss-20b --zone=europe-southwest1-b --project="$PROJECT" --command="echo '✅ Conexión OK' && hostname" 2>&1 | head -5
echo ""

echo "✅ Pruebas completadas"

