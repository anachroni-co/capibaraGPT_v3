#!/bin/bash
# Script para iniciar servicios en la VM bounty2

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Iniciando servicios en bounty2...${NC}"
echo ""

# Conectarse a la VM y ejecutar comandos
VM_ZONE="europe-west4-a"
VM_NAME="bounty2"
PROJECT="mamba-001"

echo -e "${YELLOW}📡 Conectando a ${VM_NAME}...${NC}"

# Verificar si el puerto 5001 está en uso
echo -e "${YELLOW}🔍 Verificando puerto 5001...${NC}"
PORT_CHECK=$(gcloud compute ssh --zone "$VM_ZONE" "$VM_NAME" --project "$PROJECT" \
  --command "lsof -ti:5001 2>/dev/null || echo 'FREE'" 2>/dev/null || echo "ERROR")

if [ "$PORT_CHECK" != "FREE" ] && [ "$PORT_CHECK" != "ERROR" ]; then
    echo -e "${GREEN}✅ Puerto 5001 ya está en uso (PID: $PORT_CHECK)${NC}"
    echo -e "${YELLOW}⚠️  El servicio ya está corriendo${NC}"
    exit 0
fi

# Verificar qué servidor debe iniciarse
echo -e "${YELLOW}🔍 Buscando servidor integrado...${NC}"

# Intentar iniciar el servidor integrado con Ollama
echo -e "${YELLOW}🚀 Iniciando servidor integrado en puerto 5001...${NC}"

gcloud compute ssh --zone "$VM_ZONE" "$VM_NAME" --project "$PROJECT" --command "
cd ~/capibara6/backend

# Verificar si existe el servidor integrado con Ollama
if [ -f 'capibara6_integrated_server_ollama.py' ]; then
    echo '📦 Usando servidor integrado con Ollama'
    SERVER_FILE='capibara6_integrated_server_ollama.py'
elif [ -f 'capibara6_integrated_server.py' ]; then
    echo '📦 Usando servidor integrado estándar'
    SERVER_FILE='capibara6_integrated_server.py'
elif [ -f 'server_gptoss.py' ]; then
    echo '📦 Usando server_gptoss.py'
    SERVER_FILE='server_gptoss.py'
else
    echo '❌ No se encontró servidor adecuado'
    exit 1
fi

# Verificar entorno virtual
if [ ! -d 'venv' ]; then
    echo '📦 Creando entorno virtual...'
    python3 -m venv venv
fi

# Activar entorno virtual
source venv/bin/activate

# Instalar dependencias si es necesario
if [ ! -f 'venv/.deps_installed' ]; then
    echo '📥 Instalando dependencias...'
    pip install -q -r requirements.txt
    touch venv/.deps_installed
fi

# Verificar que Ollama esté corriendo
echo '🔍 Verificando Ollama...'
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo '✅ Ollama está corriendo'
else
    echo '⚠️  Ollama no responde en localhost:11434'
    echo '   Asegúrate de que Ollama esté corriendo'
fi

# Iniciar servidor en background usando screen
echo '🖥️  Iniciando servidor en screen...'
screen -dmS capibara6-backend bash -c \"
    cd ~/capibara6/backend
    source venv/bin/activate
    export PORT=5001
    python3 \$SERVER_FILE
\"

# Esperar un momento
sleep 3

# Verificar que el servidor esté corriendo
if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
    echo '✅ Servidor iniciado correctamente en puerto 5001'
    echo '📊 Estado:'
    curl -s http://localhost:5001/api/health | python3 -m json.tool 2>/dev/null || echo 'Servidor respondiendo'
else
    echo '⚠️  El servidor no responde aún. Verifica los logs:'
    echo '   screen -r capibara6-backend'
fi

echo ''
echo '📝 Comandos útiles:'
echo '   Ver logs: screen -r capibara6-backend'
echo '   Detener: screen -S capibara6-backend -X quit'
echo '   Ver screens: screen -ls'
"

echo ""
echo -e "${GREEN}✅ Script completado${NC}"
echo ""
echo "Para verificar el servicio:"
echo "  curl http://34.12.166.76:5001/api/health"

