#!/bin/bash
# Script para reiniciar el backend con la configuración CORS actualizada

echo "🔄 Reiniciando backend en bounty2 con configuración CORS..."

gcloud compute ssh --zone "europe-west4-a" "bounty2" --project "mamba-001" --command "
cd ~/capibara6/backend

# Detener servidor actual si está corriendo
if lsof -ti:5001 > /dev/null 2>&1; then
    echo '🛑 Deteniendo servidor actual...'
    screen -S capibara6-backend -X quit 2>/dev/null || kill \$(lsof -ti:5001) 2>/dev/null
    sleep 2
fi

# Asegurarse de que el código está actualizado
echo '📥 Actualizando código desde repositorio...'
git pull 2>/dev/null || echo '⚠️  No se pudo actualizar desde git (puede ser normal)'

# Activar entorno virtual
[ -d 'venv' ] && source venv/bin/activate || (python3 -m venv venv && source venv/bin/activate && pip install -q -r requirements.txt)

# Verificar que flask-cors esté instalado
pip show flask-cors > /dev/null 2>&1 || pip install flask-cors

# Buscar servidor adecuado
SERVER_FILE=''
[ -f 'capibara6_integrated_server.py' ] && SERVER_FILE='capibara6_integrated_server.py'
[ -f 'server_gptoss.py' ] && SERVER_FILE='server_gptoss.py'

if [ -z \"\$SERVER_FILE\" ]; then
    echo '❌ No se encontró servidor adecuado'
    exit 1
fi

echo \"📦 Usando: \$SERVER_FILE\"

# Iniciar servidor
echo '🚀 Iniciando servidor con CORS configurado...'
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
    echo '✅ Servidor reiniciado correctamente'
    echo '📊 Probando CORS...'
    curl -s -X OPTIONS http://localhost:5001/api/ai/classify \
      -H 'Origin: http://localhost:8000' \
      -H 'Access-Control-Request-Method: POST' \
      -v 2>&1 | grep -i 'access-control' || echo '⚠️  Verifica headers CORS manualmente'
else
    echo '⚠️  Servidor no responde. Verifica logs:'
    echo '   screen -r capibara6-backend'
fi
"

echo ""
echo "✅ Script completado"
echo ""
echo "Para verificar desde local:"
echo "  curl -X OPTIONS http://34.12.166.76:5001/api/ai/classify \\"
echo "    -H 'Origin: http://localhost:8000' \\"
echo "    -H 'Access-Control-Request-Method: POST' -v"

