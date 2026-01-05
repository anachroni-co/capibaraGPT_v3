#!/bin/bash
# Script para iniciar todos los servicios necesarios en las VMs

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Iniciando todos los servicios de Capibara6...${NC}"
echo ""

# ============================================
# 1. BOUNTY2 - Backend en puerto 5001
# ============================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}1️⃣  Iniciando Backend en bounty2 (puerto 5001)...${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

gcloud compute ssh --zone "europe-west4-a" "bounty2" --project "mamba-001" --command "
cd ~/capibara6/backend

# Verificar si ya está corriendo
if lsof -ti:5001 > /dev/null 2>&1; then
    echo '✅ Backend ya está corriendo en puerto 5001'
    exit 0
fi

# Buscar servidor adecuado
if [ -f 'capibara6_integrated_server_ollama.py' ]; then
    SERVER_FILE='capibara6_integrated_server_ollama.py'
elif [ -f 'capibara6_integrated_server.py' ]; then
    SERVER_FILE='capibara6_integrated_server.py'
elif [ -f 'server_gptoss.py' ]; then
    SERVER_FILE='server_gptoss.py'
else
    echo '❌ No se encontró servidor adecuado'
    exit 1
fi

echo \"📦 Usando: \$SERVER_FILE\"

# Activar entorno virtual
if [ -d 'venv' ]; then
    source venv/bin/activate
else
    echo '📦 Creando entorno virtual...'
    python3 -m venv venv
    source venv/bin/activate
    pip install -q -r requirements.txt
fi

# Verificar Ollama
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo '✅ Ollama está corriendo'
else
    echo '⚠️  Ollama no responde'
fi

# Iniciar servidor
echo '🚀 Iniciando servidor en screen...'
screen -dmS capibara6-backend bash -c \"
    cd ~/capibara6/backend
    source venv/bin/activate
    export PORT=5001
    export OLLAMA_BASE_URL=http://localhost:11434
    python3 \$SERVER_FILE
\"

sleep 3

# Verificar
if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
    echo '✅ Backend iniciado correctamente'
else
    echo '⚠️  Backend no responde aún. Verifica: screen -r capibara6-backend'
fi
" || echo -e "${RED}❌ Error iniciando backend en bounty2${NC}"

echo ""

# ============================================
# 2. GPT-OSS-20B - Smart MCP (puerto 5010)
# ============================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}2️⃣  Iniciando Smart MCP en gpt-oss-20b (puerto 5010)...${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

gcloud compute ssh --zone "europe-southwest1-b" "gpt-oss-20b" --project "mamba-001" --command "
cd ~/capibara6/backend

# Verificar si ya está corriendo
if lsof -ti:5010 > /dev/null 2>&1; then
    echo '✅ Smart MCP ya está corriendo en puerto 5010'
    exit 0
fi

# Verificar que existe el archivo
if [ ! -f 'smart_mcp_server.py' ]; then
    echo '❌ smart_mcp_server.py no encontrado'
    exit 1
fi

# Activar entorno virtual
if [ -d 'venv' ]; then
    source venv/bin/activate
else
    echo '📦 Creando entorno virtual...'
    python3 -m venv venv
    source venv/bin/activate
    pip install -q -r requirements.txt
fi

# Iniciar Smart MCP
echo '🚀 Iniciando Smart MCP en screen...'
screen -dmS smart-mcp bash -c \"
    cd ~/capibara6/backend
    source venv/bin/activate
    export PORT=5010
    python3 smart_mcp_server.py
\"

sleep 3

# Verificar
if curl -s http://localhost:5010/health > /dev/null 2>&1; then
    echo '✅ Smart MCP iniciado correctamente'
else
    echo '⚠️  Smart MCP no responde aún. Verifica: screen -r smart-mcp'
fi
" || echo -e "${RED}❌ Error iniciando Smart MCP en gpt-oss-20b${NC}"

echo ""

# ============================================
# 3. GPT-OSS-20B - N8n (puerto 5678)
# ============================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}3️⃣  Verificando N8n en gpt-oss-20b (puerto 5678)...${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

gcloud compute ssh --zone "europe-southwest1-b" "gpt-oss-20b" --project "mamba-001" --command "
# Verificar si N8n está corriendo
if curl -s http://localhost:5678/healthz > /dev/null 2>&1; then
    echo '✅ N8n ya está corriendo'
    exit 0
fi

# Intentar iniciar N8n
if command -v n8n > /dev/null 2>&1; then
    echo '🚀 Iniciando N8n...'
    screen -dmS n8n bash -c 'n8n start'
    sleep 5
    if curl -s http://localhost:5678/healthz > /dev/null 2>&1; then
        echo '✅ N8n iniciado correctamente'
    else
        echo '⚠️  N8n no responde'
    fi
elif docker ps | grep -q n8n; then
    echo '✅ N8n está corriendo en Docker'
elif [ -f 'docker-compose.yml' ]; then
    echo '🚀 Iniciando N8n con Docker Compose...'
    docker-compose up -d n8n
    sleep 5
    if curl -s http://localhost:5678/healthz > /dev/null 2>&1; then
        echo '✅ N8n iniciado correctamente'
    else
        echo '⚠️  N8n no responde'
    fi
else
    echo '⚠️  N8n no está instalado o configurado'
fi
" || echo -e "${YELLOW}⚠️  N8n no disponible o no configurado${NC}"

echo ""

# ============================================
# Resumen Final
# ============================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}📊 Verificación Final${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "Probando conexiones desde local..."
echo ""

# Probar Backend en bounty2
echo -n "Backend (bounty2:5001): "
if curl -s --connect-timeout 5 http://34.12.166.76:5001/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Conectado${NC}"
else
    echo -e "${RED}❌ No conectado${NC}"
fi

# Probar Smart MCP en gpt-oss-20b
echo -n "Smart MCP (gpt-oss-20b:5010): "
if curl -s --connect-timeout 5 http://34.175.136.104:5010/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Conectado${NC}"
else
    echo -e "${RED}❌ No conectado${NC}"
fi

# Probar N8n en gpt-oss-20b
echo -n "N8n (gpt-oss-20b:5678): "
if curl -s --connect-timeout 5 http://34.175.136.104:5678/healthz > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Conectado${NC}"
else
    echo -e "${YELLOW}⚠️  No conectado (puede ser normal si no está configurado)${NC}"
fi

echo ""
echo -e "${GREEN}✅ Script completado${NC}"
echo ""
echo "Para ver logs de los servicios:"
echo "  bounty2: gcloud compute ssh --zone 'europe-west4-a' 'bounty2' --project 'mamba-001' && screen -r capibara6-backend"
echo "  gpt-oss-20b: gcloud compute ssh --zone 'europe-southwest1-b' 'gpt-oss-20b' --project 'mamba-001' && screen -r smart-mcp"

