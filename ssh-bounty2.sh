#!/bin/bash
# Script para conectarse a la VM de MODELOS (bounty2)

echo "🦫 Conectando a VM de MODELOS (bounty2)..."
echo "  - Zona: europe-west4-a"
echo "  - IP: 34.12.166.76"
echo "  - Servicios: GPT-OSS-20B (puerto 8080), Backend API (puerto 5001)"
echo ""

gcloud compute ssh --zone "europe-west4-a" "bounty2" --project "mamba-001"
