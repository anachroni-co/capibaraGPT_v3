#!/bin/bash
# Script para obtener información completa de las VMs

set -e

PROJECT="mamba-001"

echo "🔍 Obteniendo información de las VMs..."
echo ""

# Función para obtener info de una VM
get_vm_info() {
    local vm_name=$1
    local zone=$2
    
    echo "📡 $vm_name ($zone):"
    
    # IP pública
    echo -n "  IP Pública: "
    PUBLIC_IP=$(gcloud compute instances describe "$vm_name" \
        --zone="$zone" \
        --project="$PROJECT" \
        --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null)
    if [ -n "$PUBLIC_IP" ]; then
        echo "$PUBLIC_IP"
    else
        echo "No tiene IP pública"
    fi
    
    # IP interna
    echo -n "  IP Interna: "
    INTERNAL_IP=$(gcloud compute instances describe "$vm_name" \
        --zone="$zone" \
        --project="$PROJECT" \
        --format="value(networkInterfaces[0].networkIP)" 2>/dev/null)
    if [ -n "$INTERNAL_IP" ]; then
        echo "$INTERNAL_IP"
    else
        echo "No disponible"
    fi
    
    # Estado
    echo -n "  Estado: "
    STATUS=$(gcloud compute instances describe "$vm_name" \
        --zone="$zone" \
        --project="$PROJECT" \
        --format="value(status)" 2>/dev/null)
    echo "$STATUS"
    
    echo ""
}

# Obtener info de cada VM
get_vm_info "bounty2" "europe-west4-a"
get_vm_info "rag3" "europe-west2-c"
get_vm_info "gpt-oss-20b" "europe-southwest1-b"

echo "✅ Información obtenida"

