#!/bin/bash
# Script para conectarse a la VM de MODELOS (models-europe)

echo "🤖 Conectando a VM de MODELOS (models-europe)..."
echo "  - Zona: europe-southwest1-b"
echo "  - IP Interna: 10.204.0.9"
echo "  - IP Externa: 34.175.48.2"
echo "  - Servicios: Ollama (puerto 11434)"
echo "  - Modelos: gpt-oss:20b, mistral:latest, phi3:mini"
echo ""

gcloud compute ssh --zone "europe-southwest1-b" "models-europe" --project "mamba-001"
