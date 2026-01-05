#!/bin/bash
# Script para verificar servicios en las VMs usando gcloud
# Ejecuta comandos remotos en cada VM para verificar servicios

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

PROJECT="mamba-001"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🔍 Verificación Remota de Servicios${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Función para ejecutar comando remoto y verificar servicios
check_vm_services() {
    local vm_name=$1
    local zone=$2
    local description=$3
    
    echo -e "${CYAN}📡 Verificando $vm_name ($zone) - $description${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Comando para verificar servicios
    gcloud compute ssh "$vm_name" \
        --zone="$zone" \
        --project="$PROJECT" \
        --command="
            echo '📋 Procesos Python:'
            ps aux | grep python | grep -v grep | head -5 || echo '  No hay procesos Python'
            echo ''
            echo '📋 Procesos Ollama:'
            ps aux | grep ollama | grep -v grep | head -5 || echo '  No hay procesos Ollama'
            echo ''
            echo '📋 Procesos Node:'
            ps aux | grep node | grep -v grep | head -5 || echo '  No hay procesos Node'
            echo ''
            echo '🔌 Puertos escuchando:'
            if command -v ss &> /dev/null; then
                sudo ss -tuln | grep -E '(5000|5001|5002|5003|5010|5678|8000|11434|8080)' || echo '  No se encontraron puertos relevantes'
            elif command -v netstat &> /dev/null; then
                sudo netstat -tuln | grep -E '(5000|5001|5002|5003|5010|5678|8000|11434|8080)' || echo '  No se encontraron puertos relevantes'
            else
                echo '  ss y netstat no disponibles'
            fi
            echo ''
            echo '🧪 Probando servicios localmente:'
            echo -n '  Puerto 5000: '
            curl -s --connect-timeout 2 http://localhost:5000/health > /dev/null 2>&1 && echo '✅ ACTIVO' || echo '❌ INACTIVO'
            echo -n '  Puerto 5001: '
            curl -s --connect-timeout 2 http://localhost:5001/api/health > /dev/null 2>&1 && echo '✅ ACTIVO' || echo '❌ INACTIVO'
            echo -n '  Puerto 5002: '
            curl -s --connect-timeout 2 http://localhost:5002/health > /dev/null 2>&1 && echo '✅ ACTIVO' || echo '❌ INACTIVO'
            echo -n '  Puerto 5003: '
            curl -s --connect-timeout 2 http://localhost:5003/health > /dev/null 2>&1 && echo '✅ ACTIVO' || echo '❌ INACTIVO'
            echo -n '  Puerto 5010: '
            curl -s --connect-timeout 2 http://localhost:5010/health > /dev/null 2>&1 && echo '✅ ACTIVO' || echo '❌ INACTIVO'
            echo -n '  Puerto 5678: '
            curl -s --connect-timeout 2 http://localhost:5678/healthz > /dev/null 2>&1 && echo '✅ ACTIVO' || echo '❌ INACTIVO'
            echo -n '  Puerto 8000: '
            curl -s --connect-timeout 2 http://localhost:8000/health > /dev/null 2>&1 && echo '✅ ACTIVO' || echo '❌ INACTIVO'
            echo -n '  Puerto 11434 (Ollama): '
            curl -s --connect-timeout 2 http://localhost:11434/api/tags > /dev/null 2>&1 && echo '✅ ACTIVO' || echo '❌ INACTIVO'
            echo -n '  Puerto 8080: '
            curl -s --connect-timeout 2 http://localhost:8080/health > /dev/null 2>&1 && echo '✅ ACTIVO' || echo '❌ INACTIVO'
            echo ''
            echo '🌐 IPs de la VM:'
            hostname -I 2>/dev/null || ip addr show | grep 'inet ' | grep -v '127.0.0.1' | head -3
        " 2>&1
    
    echo ""
}

# Verificar cada VM
echo -e "${BLUE}Verificando VM bounty2...${NC}\n"
check_vm_services "bounty2" "europe-west4-a" "Ollama con modelos (gpt-oss-20B, mixtral, phi-mini3)"

echo -e "${BLUE}Verificando VM rag3...${NC}\n"
check_vm_services "rag3" "europe-west2-c" "Sistema de base de datos RAG"

echo -e "${BLUE}Verificando VM gpt-oss-20b...${NC}\n"
check_vm_services "gpt-oss-20b" "europe-southwest1-b" "Servicios TTS, MCP, N8n y Bridge"

echo -e "${GREEN}✅ Verificación remota completada${NC}\n"

