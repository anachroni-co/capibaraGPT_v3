#!/bin/bash
# Script con timeouts cortos y mejor feedback

PROJECT="mamba-001"

echo "🔍 Verificación de Servicios (con timeouts)"
echo "============================================"
echo ""

# Función para verificar con timeout corto
check_vm_quick() {
    local vm=$1
    local zone=$2
    
    echo "📡 Verificando $vm..."
    echo "   (timeout: 25 segundos)"
    
    # Usar timeout más corto y mostrar progreso
    timeout 25 gcloud compute ssh "$vm" \
        --zone="$zone" \
        --project="$PROJECT" \
        --command="
            echo '✅ Conectado'
            echo 'Procesos Python:' 
            ps aux | grep python | grep -v grep | wc -l
            echo 'Puertos abiertos:'
            sudo ss -tuln 2>/dev/null | grep -E ':(500[0-9]|5010|5678|8000|11434)' | wc -l || echo 0
        " 2>&1 &
    
    local pid=$!
    local elapsed=0
    while kill -0 $pid 2>/dev/null && [ $elapsed -lt 25 ]; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ $((elapsed % 5)) -eq 0 ]; then
            echo "   ... esperando ($elapsed s)"
        fi
    done
    
    if kill -0 $pid 2>/dev/null; then
        echo "   ⏱️  Timeout - matando proceso"
        kill -9 $pid 2>/dev/null
        echo "   ❌ No se pudo completar"
    else
        wait $pid
        echo "   ✅ Completado"
    fi
    echo ""
}

check_vm_quick "bounty2" "europe-west4-a"
check_vm_quick "rag3" "europe-west2-c"  
check_vm_quick "gpt-oss-20b" "europe-southwest1-b"

echo "✅ Verificación completada"

