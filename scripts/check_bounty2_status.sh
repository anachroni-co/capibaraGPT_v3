#!/bin/bash
# Script para verificar el estado de los servicios en bounty2

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

VM_ZONE="europe-west4-a"
VM_NAME="bounty2"
PROJECT="mamba-001"

echo -e "${BLUE}🔍 Verificando estado de servicios en ${VM_NAME}...${NC}"
echo ""

gcloud compute ssh --zone "$VM_ZONE" "$VM_NAME" --project "$PROJECT" --command "
echo '========================================'
echo '  📊 Estado de Servicios'
echo '========================================'
echo ''

# Verificar Ollama
echo '1️⃣ Ollama (puerto 11434):'
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e '   ${GREEN}✅ ACTIVO${NC}'
    echo '   Modelos disponibles:'
    curl -s http://localhost:11434/api/tags | python3 -c \"
import sys, json
data = json.load(sys.stdin)
for model in data.get('models', []):
    print(f\"      - {model['name']}\")
\" 2>/dev/null || echo '      (no se pudieron listar modelos)'
else
    echo -e '   ${RED}❌ NO ACTIVO${NC}'
fi
echo ''

# Verificar puerto 5001
echo '2️⃣ Backend Capibara6 (puerto 5001):'
PID_5001=\$(lsof -ti:5001 2>/dev/null || echo '')
if [ ! -z \"\$PID_5001\" ]; then
    echo -e \"   ${GREEN}✅ ACTIVO (PID: \$PID_5001)${NC}\"
    PROCESS=\$(ps -p \$PID_5001 -o cmd= 2>/dev/null | head -c 80)
    echo \"   Proceso: \$PROCESS\"
    
    # Probar health endpoint
    if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
        echo -e '   ${GREEN}✅ Health check OK${NC}'
    else
        echo -e '   ${YELLOW}⚠️  Health check no responde${NC}'
    fi
else
    echo -e '   ${RED}❌ NO ACTIVO${NC}'
fi
echo ''

# Verificar screens
echo '3️⃣ Screens activos:'
SCREENS=\$(screen -ls 2>/dev/null | grep -E '(capibara6|backend|server)' | wc -l)
if [ \$SCREENS -gt 0 ]; then
    echo -e \"   ${GREEN}✅ \$SCREENS screen(s) encontrado(s):${NC}\"
    screen -ls | grep -E '(capibara6|backend|server)' | sed 's/^/   /'
else
    echo -e '   ${YELLOW}⚠️  No hay screens activos${NC}'
fi
echo ''

# Verificar puertos abiertos
echo '4️⃣ Puertos en uso:'
for port in 11434 5001 5000 8080; do
    PID=\$(lsof -ti:\$port 2>/dev/null || echo '')
    if [ ! -z \"\$PID\" ]; then
        echo -e \"   ${GREEN}✅ Puerto \$port: PID \$PID${NC}\"
    else
        echo -e \"   ${RED}❌ Puerto \$port: LIBRE${NC}\"
    fi
done
echo ''

# Verificar IP externa
echo '5️⃣ IP Externa:'
EXTERNAL_IP=\$(curl -s ifconfig.me 2>/dev/null || echo 'No disponible')
echo \"   IP: \$EXTERNAL_IP\"
echo ''
echo '========================================'
"

echo ""
echo -e "${BLUE}📝 Para iniciar servicios:${NC}"
echo "  ./scripts/start_bounty2_services.sh"
echo ""
echo -e "${BLUE}📝 Para ver logs:${NC}"
echo "  gcloud compute ssh --zone \"$VM_ZONE\" \"$VM_NAME\" --project \"$PROJECT\""
echo "  screen -r capibara6-backend"

