#!/bin/bash
# Script para reiniciar el servidor integrado de Capibara6

echo "🔄 Reiniciando Servidor Integrado Capibara6"
echo "=========================================="

# Encontrar el directorio del backend
if [ -d "~/capibara6/backend" ]; then
    BACKEND_DIR="~/capibara6/backend"
elif [ -d "/home/$(whoami)/capibara6/backend" ]; then
    BACKEND_DIR="/home/$(whoami)/capibara6/backend"
elif [ -d "backend" ]; then
    BACKEND_DIR="backend"
else
    echo "⚠️ Buscando directorio del backend..."
    BACKEND_DIR=$(find ~ -name "capibara6_integrated_server.py" 2>/dev/null | head -1 | xargs dirname)
    if [ -z "$BACKEND_DIR" ]; then
        echo "❌ No se encontró el directorio del backend"
        exit 1
    fi
fi

echo "📁 Directorio del backend: $BACKEND_DIR"
cd "$BACKEND_DIR"

echo ""
echo "1️⃣ Deteniendo servidor existente..."
pkill -f "capibara6_integrated_server" || pkill -f "python.*5001" || echo "No hay procesos para detener"
sleep 2

echo ""
echo "2️⃣ Verificando que el puerto 5001 está libre..."
if lsof -i :5001 > /dev/null 2>&1; then
    echo "⚠️ Puerto 5001 aún en uso, forzando..."
    fuser -k 5001/tcp || sudo kill -9 $(lsof -t -i:5001) || echo "No se pudo liberar el puerto"
    sleep 2
else
    echo "✅ Puerto 5001 está libre"
fi

echo ""
echo "3️⃣ Verificando conexión con modelo GPT-OSS-20B..."
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Modelo GPT-OSS-20B está respondiendo"
else
    echo "❌ Modelo GPT-OSS-20B no responde"
    echo "⚠️ El servidor puede no funcionar correctamente sin el modelo"
fi

echo ""
echo "4️⃣ Verificando dependencias Python..."
python3 -c "import flask; import requests; print('✅ Dependencias OK')" || {
    echo "❌ Faltan dependencias"
    echo "Instalando dependencias..."
    pip3 install flask flask-cors requests || exit 1
}

echo ""
echo "5️⃣ Iniciando servidor integrado..."
nohup python3 capibara6_integrated_server.py > ../logs/server_$(date +%Y%m%d_%H%M%S).log 2>&1 &
SERVER_PID=$!

sleep 3

echo ""
echo "6️⃣ Verificando que el servidor está corriendo..."
if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "✅ Servidor iniciado (PID: $SERVER_PID)"
else
    echo "❌ El servidor no se inició correctamente"
    echo "📋 Últimas líneas del log:"
    tail -20 ../logs/server_*.log 2>/dev/null || echo "No hay logs disponibles"
    exit 1
fi

echo ""
echo "7️⃣ Verificando respuesta del servidor..."
sleep 2
if curl -s http://localhost:5001/health > /dev/null 2>&1; then
    echo "✅ Servidor responde correctamente"
    curl -s http://localhost:5001/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:5001/health
else
    echo "❌ Servidor no responde"
    echo "📋 Últimas líneas del log:"
    tail -30 ../logs/server_*.log 2>/dev/null || echo "No hay logs disponibles"
    exit 1
fi

echo ""
echo "✅ Servidor reiniciado exitosamente"
echo "📊 Proceso corriendo:"
ps aux | grep capibara6_integrated_server | grep -v grep

echo ""
echo "📝 Logs disponibles en: ../logs/server_*.log"
echo "🧪 Probar con: curl http://localhost:5001/health"
