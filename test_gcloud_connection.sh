#!/bin/bash
# Script para probar conexiones gcloud y verificar servicios

set -e

PROJECT="mamba-001"

echo "🧪 Probando conexiones gcloud..."
echo ""

# Probar conexión básica a cada VM
test_vm_connection() {
    local vm_name=$1
    local zone=$2
    
    echo "📡 Probando conexión a $vm_name ($zone)..."
    
    if gcloud compute ssh "$vm_name" \
        --zone="$zone" \
        --project="$PROJECT" \
        --command="echo '✅ Conexión exitosa' && hostname" 2>&1 | grep -q "Conexión exitosa"; then
        echo "  ✅ Conexión OK"
        return 0
    else
        echo "  ❌ Error de conexión"
        return 1
    fi
}

# Probar cada VM
test_vm_connection "bounty2" "europe-west4-a"
test_vm_connection "rag3" "europe-west2-c"
test_vm_connection "gpt-oss-20b" "europe-southwest1-b"

echo ""
echo "✅ Pruebas completadas"

