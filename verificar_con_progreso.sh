#!/bin/bash
# Script con indicadores de progreso para verificar servicios

PROJECT="mamba-001"

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🔍 Verificación de Servicios con Progreso${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Función para verificar una VM con progreso
check_vm_with_progress() {
    local vm=$1
    local zone=$2
    local description=$3
    
    echo -e "${CYAN}📡 Verificando $vm...${NC}"
    echo -e "${YELLOW}   Conectando (esto puede tardar 10-20 segundos)...${NC}"
    
    # Ejecutar con timeout y mostrar progreso
    (
        timeout 30 gcloud compute ssh "$vm" \
            --zone="$zone" \
            --project="$PROJECT" \
            --command="
                echo '✅ Conectado a $vm'
                echo ''
                echo '📋 Procesos Python:'
                ps aux | grep python | grep -v grep | head -3 || echo '  Ninguno'
                echo ''
                echo '🔌 Puertos escuchando:'
                if command -v ss &> /dev/null; then
                    sudo ss -tuln 2>/dev/null | grep -E ':(500[0-9]|5010|5678|8000|11434)' | head -5 || echo '  No se encontraron'
                else
                    sudo netstat -tuln 2>/dev/null | grep -E ':(500[0-9]|5010|5678|8000|11434)' | head -5 || echo '  No se encontraron'
                fi
            " 2>&1
    ) &
    
    local pid=$!
    local count=0
    while kill -0 $pid 2>/dev/null; do
        count=$((count + 1))
        if [ $((count % 3)) -eq 0 ]; then
            echo -n "."
        fi
        sleep 1
        if [ $count -gt 30 ]; then
            echo -e "\n${RED}⏱️  Timeout esperando respuesta${NC}"
            kill $pid 2>/dev/null
            break
        fi
    done
    wait $pid
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "\n${GREEN}✅ $vm verificado${NC}\n"
    else
        echo -e "\n${RED}❌ Error verificando $vm (código: $exit_code)${NC}\n"
    fi
}

# Verificar cada VM
check_vm_with_progress "bounty2" "europe-west4-a" "Ollama con modelos"
check_vm_with_progress "rag3" "europe-west2-c" "RAG Database"
check_vm_with_progress "gpt-oss-20b" "europe-southwest1-b" "TTS, MCP, N8n"

echo -e "${GREEN}✅ Verificación completada${NC}"

