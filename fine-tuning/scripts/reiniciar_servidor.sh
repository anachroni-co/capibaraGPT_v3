#!/bin/bash
# Script para reiniciar el servidor integrado de Capibara6
# Ejecutar directamente en la VM

echo "🔄 Reiniciando Servidor Integrado Capibara6"
echo "=========================================="

# Detener servidor existente
echo "1️⃣ Deteniendo servidor existente..."
pkill -f capibara6_integrated_server || echo "No hay procesos para detener"
sleep 2

# Verificar puerto 5001
echo ""
echo "2️⃣ Verificando puerto 5001..."
if lsof -i :5001 > /dev/null 2>&1; then
    echo "⚠️ Puerto 5001 aún en uso, liberando..."
    fuser -k 5001/tcp 2>/dev/null || sudo kill -9 $(lsof -t -i:5001) 2>/dev/null
    sleep 2
fi

# Verificar modelo GPT-OSS-20B
echo ""
echo "3️⃣ Verificando modelo GPT-OSS-20B..."
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Modelo GPT-OSS-20B está respondiendo"
else
    echo "❌ Modelo GPT-OSS-20B no responde"
    echo "⚠️ El servidor puede no funcionar correctamente sin el modelo"
fi

# Ir al directorio del backend
echo ""
echo "4️⃣ Buscando directorio del backend..."
if [ -d ~/capibara6/backend ]; then
    BACKEND_DIR=~/capibara6/backend
elif [ -d backend ]; then
    BACKEND_DIR=backend
else
    BACKEND_DIR=$(find ~ -name "capibara6_integrated_server.py" 2>/dev/null | head -1 | xargs dirname)
    if [ -z "$BACKEND_DIR" ]; then
        echo "❌ No se encontró el directorio del backend"
        exit 1
    fi
fi

echo "📁 Directorio: $BACKEND_DIR"
cd "$BACKEND_DIR"

# Verificar dependencias
echo ""
echo "5️⃣ Verificando dependencias..."
python3 -c "import flask; import requests; print('✅ Dependencias OK')" || {
    echo "❌ Faltan dependencias, instalando..."
    pip3 install flask flask-cors requests
}

# Iniciar servidor
echo ""
echo "6️⃣ Iniciando servidor integrado..."
nohup python3 capibara6_integrated_server.py > ../logs/server_$(date +%Y%m%d_%H%M%S).log 2>&1 &
SERVER_PID=$!

sleep 3

# Verificar que está corriendo
echo ""
echo "7️⃣ Verificando que el servidor está corriendo..."
if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "✅ Servidor iniciado (PID: $SERVER_PID)"
else
    echo "❌ El servidor no se inició correctamente"
    echo "📋 Últimas líneas del log:"
    tail -20 ../logs/server_*.log 2>/dev/null || echo "No hay logs disponibles"
    exit 1
fi

# Verificar respuesta
echo ""
echo "8️⃣ Verificando respuesta del servidor..."
if curl -s http://localhost:5001/health > /dev/null 2>&1; then
    echo "✅ Servidor responde correctamente"
    echo ""
    curl -s http://localhost:5001/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:5001/health
else
    echo "❌ Servidor no responde aún"
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
