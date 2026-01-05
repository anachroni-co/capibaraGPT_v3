#!/bin/bash
# Script para conectarse a la VM de SERVICIOS (gpt-oss-20b)

echo "🔧 Conectando a VM de SERVICIOS (gpt-oss-20b)..."
echo "  - Zona: europe-southwest1-b"
echo "  - IP: 34.175.136.104"
echo "  - Servicios: TTS (puerto 5002), MCP (puerto 5003), N8N (puerto 5678)"
echo ""

gcloud compute ssh --zone "europe-southwest1-b" "gpt-oss-20b" --project "mamba-001"
