#!/bin/bash
# Script combinado para verificar todos los servicios en las 3 VMs

PROJECT="mamba-001"

echo "=========================================="
echo "🔍 VERIFICACIÓN COMPLETA DE SERVICIOS"
echo "=========================================="
echo ""

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Función para verificar una VM
check_vm() {
    local vm=$1
    local zone=$2
    local description=$3
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}📡 Verificando $vm ($zone)${NC}"
    echo -e "${CYAN}$description${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    gcloud compute ssh "$vm" \
        --zone="$zone" \
        --project="$PROJECT" \
        --command="
            echo '📋 Procesos activos:'
            echo '  Python:'
            ps aux | grep python | grep -v grep | head -5 || echo '    Ninguno'
            echo '  Ollama:'
            ps aux | grep ollama | grep -v grep | head -3 || echo '    Ninguno'
            echo '  Node:'
            ps aux | grep node | grep -v grep | head -3 || echo '    Ninguno'
            echo ''
            echo '🔌 Puertos escuchando:'
            if command -v ss &> /dev/null; then
                sudo ss -tuln 2>/dev/null | grep -E ':(500[0-9]|5010|5678|8000|11434|8080|5432|6379)' || echo '    No se encontraron puertos relevantes'
            elif command -v netstat &> /dev/null; then
                sudo netstat -tuln 2>/dev/null | grep -E ':(500[0-9]|5010|5678|8000|11434|8080|5432|6379)' || echo '    No se encontraron puertos relevantes'
            else
                echo '    Herramientas de red no disponibles'
            fi
            echo ''
            echo '🧪 Probando servicios HTTP:'
            for port in 5000 5001 5002 5003 5010 5678 8000 11434; do
                if curl -s --connect-timeout 2 http://localhost:\$port/health > /dev/null 2>&1 || \
                   curl -s --connect-timeout 2 http://localhost:\$port/api/health > /dev/null 2>&1 || \
                   curl -s --connect-timeout 2 http://localhost:\$port/api/tags > /dev/null 2>&1; then
                    echo \"  ✅ Puerto \$port - ACTIVO\"
                fi
            done
            echo ''
            echo '🌐 IPs:'
            hostname -I 2>/dev/null | head -1 || echo '    No disponible'
        " 2>&1
    
    echo ""
    echo ""
}

# Verificar cada VM
check_vm "bounty2" "europe-west4-a" "Ollama con modelos (gpt-oss-20B, mixtral, phi-mini3)"
check_vm "rag3" "europe-west2-c" "Sistema de base de datos RAG"
check_vm "gpt-oss-20b" "europe-southwest1-b" "Servicios TTS, MCP, N8n y Bridge"

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}✅ Verificación completada${NC}"
echo -e "${GREEN}==========================================${NC}"

