#!/bin/bash
# Script para conectarse a la VM de RAG (rag-europe)

echo "🗄️  Conectando a VM de RAG (rag-europe)..."
echo "  - Zona: europe-southwest1-b"
echo "  - IP Interna: 10.204.0.10"
echo "  - IP Externa: 34.175.110.120"
echo "  - Servicios:"
echo "    * Bridge API (puerto 8000)"
echo "    * Nebula Graph (puerto 9669)"
echo "    * Nebula Studio (puerto 7001)"
echo "    * Milvus (puerto 19530)"
echo "    * PostgreSQL (puerto 5432)"
echo "    * Redis (puerto 6379)"
echo ""

gcloud compute ssh --zone "europe-southwest1-b" "rag-europe" --project "mamba-001"
