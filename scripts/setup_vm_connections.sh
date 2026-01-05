#!/bin/bash
# Script para configurar conexiones entre VMs de Capibara6
# Este script ayuda a configurar firewall rules y verificar conectividad

set -e

PROJECT="mamba-001"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🚀 Configuración de Conexiones entre VMs - Capibara6"
echo "====================================================="
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para imprimir mensajes
info() {
    echo -e "${GREEN}✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Verificar que gcloud está instalado
if ! command -v gcloud &> /dev/null; then
    error "gcloud CLI no está instalado. Por favor instala Google Cloud SDK"
    exit 1
fi

info "gcloud CLI encontrado"

# Obtener información de VMs
echo ""
echo "📊 Obteniendo información de VMs..."
if [ -f "$PROJECT_ROOT/vm_config.json" ]; then
    info "Usando configuración existente: vm_config.json"
    source <(python3 -c "
import json
with open('$PROJECT_ROOT/vm_config.json') as f:
    config = json.load(f)
    vms = config.get('vms', {})
    for vm_name, vm_data in vms.items():
        print(f'export {vm_name.upper().replace(\"-\", \"_\")}_IP_INTERNAL=\"{vm_data.get(\"ip_internal\", \"\")}\"')
        print(f'export {vm_name.upper().replace(\"-\", \"_\")}_IP_EXTERNAL=\"{vm_data.get(\"ip_external\", \"\")}\"')
        print(f'export {vm_name.upper().replace(\"-\", \"_\")}_NETWORK=\"{vm_data.get(\"network\", \"\")}\"')
")
else
    warn "vm_config.json no encontrado. Ejecutando get_vm_info.py..."
    cd "$PROJECT_ROOT"
    python3 scripts/get_vm_info.py || {
        error "No se pudo obtener información de VMs"
        exit 1
    }
fi

# Verificar red VPC
echo ""
echo "🌐 Verificando configuración de red..."
NETWORKS=$(python3 -c "
import json
with open('$PROJECT_ROOT/vm_config.json') as f:
    config = json.load(f)
    networks = set()
    for vm_name, vm_data in config.get('vms', {}).items():
        network = vm_data.get('network')
        if network:
            networks.add(network)
    print(' '.join(networks))
" 2>/dev/null || echo "")

if [ $(echo "$NETWORKS" | wc -w) -eq 1 ]; then
    info "Todas las VMs están en la misma VPC: $NETWORKS"
else
    warn "Las VMs están en diferentes VPCs: $NETWORKS"
    echo "   Se recomienda moverlas a la misma VPC para mejor rendimiento"
fi

# Verificar firewall rules
echo ""
echo "🔒 Verificando reglas de firewall..."
FIREWALL_RULES=$(gcloud compute firewall-rules list --project=$PROJECT --format="value(name)" 2>/dev/null || echo "")

if echo "$FIREWALL_RULES" | grep -q "allow-internal-vm-communication"; then
    info "Regla de firewall 'allow-internal-vm-communication' existe"
else
    warn "Regla de firewall 'allow-internal-vm-communication' no encontrada"
    echo ""
    read -p "¿Deseas crear la regla de firewall para comunicación interna? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        NETWORK_NAME=$(python3 -c "
import json
with open('$PROJECT_ROOT/vm_config.json') as f:
    config = json.load(f)
    networks = [vm_data.get('network') for vm_data in config.get('vms', {}).values() if vm_data.get('network')]
    print(networks[0] if networks else 'default')
" 2>/dev/null || echo "default")
        
        echo "Creando regla de firewall en red: $NETWORK_NAME"
        gcloud compute firewall-rules create allow-internal-vm-communication \
            --project=$PROJECT \
            --network=$NETWORK_NAME \
            --allow tcp:11434,tcp:5000,tcp:5001,tcp:5002,tcp:5003,tcp:5010,tcp:5678,tcp:8000 \
            --source-ranges=10.0.0.0/8 \
            --description="Permitir comunicación entre VMs de Capibara6" \
            --direction=INGRESS || {
            error "Error creando regla de firewall"
        }
    fi
fi

# Generar archivo .env de ejemplo
echo ""
echo "📝 Generando archivo .env.example..."
cat > "$PROJECT_ROOT/backend/.env.vm.example" << 'EOF'
# Configuración de endpoints de VMs para Capibara6
# Actualizar con las IPs reales obtenidas de vm_config.json

# Bounty2 (Ollama + Backend)
BOUNTY2_IP_INTERNAL=10.x.x.x
BOUNTY2_IP_EXTERNAL=34.x.x.x
OLLAMA_ENDPOINT=http://BOUNTY2_IP_INTERNAL:11434
BOUNTY2_BACKEND_ENDPOINT=http://BOUNTY2_IP_INTERNAL:5001

# rag3 (RAG Database)
RAG3_IP_INTERNAL=10.x.x.x
RAG3_IP_EXTERNAL=34.x.x.x
RAG_API_ENDPOINT=http://RAG3_IP_INTERNAL:8000

# gpt-oss-20b (TTS, MCP, N8n, Bridge)
GPTOSS_IP_INTERNAL=10.x.x.x
GPTOSS_IP_EXTERNAL=34.x.x.x
TTS_ENDPOINT=http://GPTOSS_IP_INTERNAL:5002
MCP_ENDPOINT=http://GPTOSS_IP_INTERNAL:5003
MCP_ALT_ENDPOINT=http://GPTOSS_IP_INTERNAL:5010
N8N_ENDPOINT=http://GPTOSS_IP_INTERNAL:5678
BRIDGE_ENDPOINT=http://GPTOSS_IP_INTERNAL:5000
EOF

info "Archivo .env.vm.example creado en backend/.env.vm.example"

# Actualizar config.js con IPs reales si están disponibles
echo ""
echo "🔧 Actualizando configuración del frontend..."
if [ -f "$PROJECT_ROOT/vm_config.json" ]; then
    BOUNTY2_IP=$(python3 -c "
import json
with open('$PROJECT_ROOT/vm_config.json') as f:
    config = json.load(f)
    print(config.get('vms', {}).get('bounty2', {}).get('ip_external', ''))
" 2>/dev/null || echo "")
    
    GPTOSS_IP=$(python3 -c "
import json
with open('$PROJECT_ROOT/vm_config.json') as f:
    config = json.load(f)
    print(config.get('vms', {}).get('gpt-oss-20b', {}).get('ip_external', ''))
" 2>/dev/null || echo "")
    
    RAG3_IP=$(python3 -c "
import json
with open('$PROJECT_ROOT/vm_config.json') as f:
    config = json.load(f)
    print(config.get('vms', {}).get('rag3', {}).get('ip_external', ''))
" 2>/dev/null || echo "")
    
    if [ -n "$BOUNTY2_IP" ] && [ -n "$GPTOSS_IP" ]; then
        warn "IPs encontradas. Por favor actualiza manualmente web/config.js con:"
        echo "   BOUNTY2_EXTERNAL: $BOUNTY2_IP"
        echo "   GPTOSS_EXTERNAL: $GPTOSS_IP"
        [ -n "$RAG3_IP" ] && echo "   RAG3_EXTERNAL: $RAG3_IP"
    fi
fi

echo ""
echo "✅ Configuración completada"
echo ""
echo "🎯 Próximos pasos:"
echo "=================="
echo "1. Revisa vm_config.json para verificar las IPs"
echo "2. Actualiza web/config.js con las IPs externas para desarrollo local"
echo "3. Copia backend/.env.vm.example a backend/.env y actualiza con IPs reales"
echo "4. Verifica servicios en cada VM ejecutando: bash scripts/check_services_on_vm.sh"
echo "5. Prueba conexiones desde tu portátil usando los endpoints en VM_SETUP_GUIDE.md"
echo ""

