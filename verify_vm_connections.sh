#!/bin/bash
# Script de verificación de conexiones entre VMs de Capibara6
# Verifica servicios en bounty2, rag3 y gpt-oss-20b

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuración de VMs
PROJECT="mamba-001"
VMS=(
    "bounty2:europe-west4-a"
    "rag3:europe-west2-c"
    "gpt-oss-20b:europe-southwest1-b"
)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🔍 Verificación de Conexiones VMs${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Función para obtener IP de una VM
get_vm_ip() {
    local vm_name=$1
    local zone=$2
    gcloud compute instances describe "$vm_name" \
        --zone="$zone" \
        --project="$PROJECT" \
        --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || \
    gcloud compute instances describe "$vm_name" \
        --zone="$zone" \
        --project="$PROJECT" \
        --format="value(networkInterfaces[0].networkIP)" 2>/dev/null || \
    echo "N/A"
}

# Función para verificar servicio en una VM
check_service() {
    local ip=$1
    local port=$2
    local service_name=$3
    
    if timeout 3 bash -c "echo > /dev/tcp/$ip/$port" 2>/dev/null; then
        echo -e "${GREEN}✅${NC} $service_name ($ip:$port) - ${GREEN}ACTIVO${NC}"
        return 0
    else
        echo -e "${RED}❌${NC} $service_name ($ip:$port) - ${RED}INACTIVO${NC}"
        return 1
    fi
}

# Función para verificar servicios en una VM mediante SSH
check_vm_services() {
    local vm_name=$1
    local zone=$2
    local ip=$3
    
    echo -e "\n${YELLOW}📡 Verificando servicios en $vm_name ($ip)...${NC}"
    
    # Servicios esperados según la VM
    case $vm_name in
        "bounty2")
            echo -e "${BLUE}Servicios esperados:${NC}"
            echo "  • Ollama (puerto 11434)"
            echo "  • Backend Capibara6 (puerto 5001)"
            echo "  • Modelos: gpt-oss-20B, mixtral, phi-mini3"
            
            # Verificar puertos comunes
            check_service "$ip" 11434 "Ollama"
            check_service "$ip" 5001 "Backend Capibara6"
            check_service "$ip" 5000 "Backend alternativo"
            ;;
            
        "rag3")
            echo -e "${BLUE}Servicios esperados:${NC}"
            echo "  • Base de datos RAG (puerto 8000)"
            echo "  • API RAG (puerto 8000)"
            
            check_service "$ip" 8000 "RAG API"
            check_service "$ip" 5432 "PostgreSQL (si aplica)"
            ;;
            
        "gpt-oss-20b")
            echo -e "${BLUE}Servicios esperados:${NC}"
            echo "  • TTS Server (puerto 5002)"
            echo "  • MCP Server (puerto 5003)"
            echo "  • N8n (puerto 5678)"
            echo "  • Bridge (puerto 5000)"
            
            check_service "$ip" 5000 "Bridge/Main Server"
            check_service "$ip" 5002 "TTS Server"
            check_service "$ip" 5003 "MCP Server"
            check_service "$ip" 5010 "MCP Server (alternativo)"
            check_service "$ip" 5678 "N8n"
            check_service "$ip" 8080 "Modelo (si aplica)"
            ;;
    esac
}

# Obtener IPs de todas las VMs
declare -A VM_IPS
echo -e "${BLUE}📋 Obteniendo IPs de las VMs...${NC}\n"

for vm_info in "${VMS[@]}"; do
    IFS=':' read -r vm_name zone <<< "$vm_info"
    ip=$(get_vm_ip "$vm_name" "$zone")
    VM_IPS["$vm_name"]=$ip
    echo -e "${GREEN}✓${NC} $vm_name ($zone): ${YELLOW}$ip${NC}"
done

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}🔌 Verificando Conectividad${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Verificar servicios en cada VM
for vm_info in "${VMS[@]}"; do
    IFS=':' read -r vm_name zone <<< "$vm_info"
    ip=${VM_IPS[$vm_name]}
    
    if [ "$ip" != "N/A" ] && [ -n "$ip" ]; then
        check_vm_services "$vm_name" "$zone" "$ip"
    else
        echo -e "${RED}❌${NC} No se pudo obtener IP para $vm_name"
    fi
done

# Verificar conectividad entre VMs
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}🌐 Verificando Conectividad entre VMs${NC}"
echo -e "${BLUE}========================================${NC}\n"

bounty2_ip=${VM_IPS["bounty2"]}
rag3_ip=${VM_IPS["rag3"]}
gpt_oss_ip=${VM_IPS["gpt-oss-20b"]}

if [ "$bounty2_ip" != "N/A" ] && [ "$rag3_ip" != "N/A" ]; then
    echo -e "${YELLOW}Verificando: bounty2 → rag3${NC}"
    if ping -c 1 -W 2 "$rag3_ip" >/dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} Conectividad OK"
    else
        echo -e "${RED}❌${NC} Sin conectividad"
    fi
fi

if [ "$bounty2_ip" != "N/A" ] && [ "$gpt_oss_ip" != "N/A" ]; then
    echo -e "${YELLOW}Verificando: bounty2 → gpt-oss-20b${NC}"
    if ping -c 1 -W 2 "$gpt_oss_ip" >/dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} Conectividad OK"
    else
        echo -e "${RED}❌${NC} Sin conectividad"
    fi
fi

if [ "$rag3_ip" != "N/A" ] && [ "$gpt_oss_ip" != "N/A" ]; then
    echo -e "${YELLOW}Verificando: rag3 → gpt-oss-20b${NC}"
    if ping -c 1 -W 2 "$gpt_oss_ip" >/dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} Conectividad OK"
    else
        echo -e "${RED}❌${NC} Sin conectividad"
    fi
fi

# Generar resumen de configuración
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}📝 Resumen de Configuración${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${YELLOW}IPs de las VMs:${NC}"
echo "  • bounty2: $bounty2_ip"
echo "  • rag3: $rag3_ip"
echo "  • gpt-oss-20b: $gpt_oss_ip"

echo -e "\n${YELLOW}Configuración recomendada para frontend local:${NC}"
echo "  • Backend (Ollama): http://$bounty2_ip:11434"
echo "  • Backend API: http://$bounty2_ip:5001"
echo "  • RAG API: http://$rag3_ip:8000"
echo "  • TTS: http://$gpt_oss_ip:5002"
echo "  • MCP: http://$gpt_oss_ip:5003"
echo "  • N8n: http://$gpt_oss_ip:5678"

echo -e "\n${GREEN}✅ Verificación completada${NC}\n"

