#!/bin/bash
# Script rápido para iniciar servicios faltantes

echo "🚀 Iniciando servicios faltantes..."

# 1. Backend en bounty2 (puerto 5001)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  Iniciando Backend en bounty2..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

gcloud compute ssh --zone "europe-west4-a" "bounty2" --project "mamba-001" --command "
cd ~/capibara6/backend

# Verificar si ya está corriendo
if lsof -ti:5001 > /dev/null 2>&1; then
    echo '✅ Backend ya está corriendo'
    exit 0
fi

# Buscar servidor
SERVER_FILE=''
[ -f 'capibara6_integrated_server_ollama.py' ] && SERVER_FILE='capibara6_integrated_server_ollama.py'
[ -f 'capibara6_integrated_server.py' ] && SERVER_FILE='capibara6_integrated_server.py'
[ -f 'server_gptoss.py' ] && SERVER_FILE='server_gptoss.py'

if [ -z \"\$SERVER_FILE\" ]; then
    echo '❌ No se encontró servidor'
    exit 1
fi

echo \"📦 Usando: \$SERVER_FILE\"

# Activar venv
[ -d 'venv' ] && source venv/bin/activate || (python3 -m venv venv && source venv/bin/activate && pip install -q -r requirements.txt)

# Iniciar en screen
screen -dmS capibara6-backend bash -c \"
    cd ~/capibara6/backend
    source venv/bin/activate
    export PORT=5001
    export OLLAMA_BASE_URL=http://localhost:11434
    python3 \$SERVER_FILE
\"

sleep 3
curl -s http://localhost:5001/api/health > /dev/null && echo '✅ Backend iniciado' || echo '⚠️  Verifica: screen -r capibara6-backend'
"

# 2. Smart MCP en gpt-oss-20b (puerto 5010)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  Iniciando Smart MCP en gpt-oss-20b..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

gcloud compute ssh --zone "europe-southwest1-b" "gpt-oss-20b" --project "mamba-001" --command "
cd ~/capibara6/backend

# Verificar si ya está corriendo
if lsof -ti:5010 > /dev/null 2>&1; then
    echo '✅ Smart MCP ya está corriendo'
    exit 0
fi

if [ ! -f 'smart_mcp_server.py' ]; then
    echo '❌ smart_mcp_server.py no encontrado'
    exit 1
fi

# Activar venv
[ -d 'venv' ] && source venv/bin/activate || (python3 -m venv venv && source venv/bin/activate && pip install -q -r requirements.txt)

# Iniciar en screen
screen -dmS smart-mcp bash -c \"
    cd ~/capibara6/backend
    source venv/bin/activate
    export PORT=5010
    python3 smart_mcp_server.py
\"

sleep 3
curl -s http://localhost:5010/health > /dev/null && echo '✅ Smart MCP iniciado' || echo '⚠️  Verifica: screen -r smart-mcp'
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Verificación final..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

sleep 2

echo -n "Backend (34.12.166.76:5001): "
curl -s --connect-timeout 3 http://34.12.166.76:5001/api/health > /dev/null 2>&1 && echo "✅ OK" || echo "❌ No responde"

echo -n "Smart MCP (34.175.136.104:5010): "
curl -s --connect-timeout 3 http://34.175.136.104:5010/health > /dev/null 2>&1 && echo "✅ OK" || echo "❌ No responde"

echo ""
echo "✅ Completado"

