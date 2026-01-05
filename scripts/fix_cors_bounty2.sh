#!/bin/bash
# Script para arreglar CORS en bounty2

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

VM_ZONE="europe-west4-a"
VM_NAME="bounty2"
PROJECT="mamba-001"

echo -e "${BLUE}🔧 Arreglando CORS en ${VM_NAME}...${NC}"
echo ""

# 1. Verificar firewall
echo -e "${YELLOW}1️⃣  Verificando firewall...${NC}"
FIREWALL_EXISTS=$(gcloud compute firewall-rules list --project="$PROJECT" --filter="name~'5001' AND name~'bounty2'" --format="value(name)" 2>/dev/null | head -1)

if [ -z "$FIREWALL_EXISTS" ]; then
    echo -e "${YELLOW}   Creando regla de firewall...${NC}"
    gcloud compute firewall-rules create allow-bounty2-backend-5001 \
      --allow tcp:5001 \
      --source-ranges 0.0.0.0/0 \
      --target-tags bounty2 \
      --description "Backend Capibara6 en puerto 5001" \
      --project="$PROJECT" 2>/dev/null || echo -e "${YELLOW}   ⚠️  No se pudo crear regla (puede que ya exista)${NC}"
else
    echo -e "${GREEN}   ✅ Regla de firewall existe${NC}"
fi

echo ""

# 2. Conectarse y reiniciar servidor
echo -e "${YELLOW}2️⃣  Reiniciando servidor con CORS corregido...${NC}"

gcloud compute ssh --zone "$VM_ZONE" "$VM_NAME" --project "$PROJECT" --command "
cd ~/capibara6/backend

# Detener servidor actual
if lsof -ti:5001 > /dev/null 2>&1; then
    echo '🛑 Deteniendo servidor actual...'
    screen -S capibara6-backend -X quit 2>/dev/null || kill \$(lsof -ti:5001) 2>/dev/null
    sleep 2
fi

# Actualizar código
echo '📥 Actualizando código...'
git pull 2>/dev/null || echo '⚠️  No se pudo actualizar desde git (puede ser normal)'

# Verificar que flask-cors esté instalado
source venv/bin/activate 2>/dev/null || (python3 -m venv venv && source venv/bin/activate && pip install -q -r requirements.txt)
pip show flask-cors > /dev/null 2>&1 || pip install flask-cors

# Buscar servidor
SERVER_FILE=''
[ -f 'capibara6_integrated_server.py' ] && SERVER_FILE='capibara6_integrated_server.py'
[ -f 'server_gptoss.py' ] && SERVER_FILE='server_gptoss.py'

if [ -z \"\$SERVER_FILE\" ]; then
    echo '❌ No se encontró servidor adecuado'
    exit 1
fi

echo \"📦 Usando: \$SERVER_FILE\"

# Verificar que el archivo tenga CORS configurado
if grep -q 'CORS(app' \"\$SERVER_FILE\"; then
    echo '✅ CORS configurado en el servidor'
else
    echo '⚠️  CORS no encontrado en el servidor'
fi

# Iniciar servidor
echo '🚀 Iniciando servidor...'
screen -dmS capibara6-backend bash -c \"
    cd ~/capibara6/backend
    source venv/bin/activate
    export PORT=5001
    export OLLAMA_BASE_URL=http://localhost:11434
    python3 \$SERVER_FILE
\"

sleep 3

# Verificar
echo '🔍 Verificando servidor...'
if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
    echo '✅ Servidor responde en /api/health'
    
    # Probar preflight
    echo '🔍 Probando preflight request...'
    OPTIONS_RESPONSE=\$(curl -s -X OPTIONS http://localhost:5001/api/health \
      -H 'Origin: http://localhost:8000' \
      -H 'Access-Control-Request-Method: GET' \
      -i 2>&1 | grep -i 'access-control' | head -3)
    
    if [ ! -z \"\$OPTIONS_RESPONSE\" ]; then
        echo '✅ Preflight responde correctamente'
        echo \"   \$OPTIONS_RESPONSE\"
    else
        echo '⚠️  Preflight no responde con headers CORS'
    fi
else
    echo '⚠️  Servidor no responde. Verifica logs:'
    echo '   screen -r capibara6-backend'
fi

# Verificar que escucha en 0.0.0.0
echo ''
echo '🔍 Verificando que escucha en 0.0.0.0...'
LISTENING=\$(sudo ss -tulnp | grep ':5001' | grep '0.0.0.0' || echo '')
if [ ! -z \"\$LISTENING\" ]; then
    echo '✅ Servidor escucha en 0.0.0.0:5001'
else
    echo '⚠️  Servidor NO escucha en 0.0.0.0 (puede ser problema)'
    sudo ss -tulnp | grep ':5001' || echo 'Puerto 5001 no encontrado'
fi
"

echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}✅ Verificación Final${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

sleep 2

# Probar desde local
echo -n "Probando /api/health desde local: "
if curl -s --connect-timeout 5 http://34.12.166.76:5001/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ OK${NC}"
else
    echo -e "${RED}❌ No responde${NC}"
fi

echo -n "Probando preflight (OPTIONS): "
OPTIONS_TEST=$(curl -s -X OPTIONS http://34.12.166.76:5001/api/health \
  -H "Origin: http://localhost:8000" \
  -H "Access-Control-Request-Method: GET" \
  -i 2>&1 | grep -i "access-control-allow" | head -1)

if [ ! -z "$OPTIONS_TEST" ]; then
    echo -e "${GREEN}✅ OK${NC}"
    echo "   Headers CORS: $OPTIONS_TEST"
else
    echo -e "${RED}❌ No responde con headers CORS${NC}"
fi

echo ""
echo -e "${GREEN}✅ Script completado${NC}"
echo ""
echo "Si aún hay problemas:"
echo "  1. Verifica logs: screen -r capibara6-backend"
echo "  2. Verifica firewall: gcloud compute firewall-rules list --project=$PROJECT"
echo "  3. Verifica que escucha en 0.0.0.0: sudo ss -tulnp | grep 5001"

