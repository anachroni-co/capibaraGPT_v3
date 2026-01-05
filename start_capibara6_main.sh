#!/bin/bash
# Script para iniciar el servidor principal de Capibara6

echo "🚀 Iniciando servidor principal de Capibara6..."

cd /home/elect/capibara6/backend

# Verificar que los archivos necesarios existan
if [ ! -f "capibara6_integrated_server.py" ]; then
    echo "❌ Archivo capibara6_integrated_server.py no encontrado"
    exit 1
fi

if [ ! -f "models_config_updated.py" ]; then
    echo "❌ Archivo models_config_updated.py no encontrado"
    exit 1
fi

echo "📦 Iniciando servidor en puerto 5001..."
echo "   Servidor ARM-Axion ya corriendo en puerto 8080"
echo ""

# Iniciar el servidor
nohup python3 capibara6_integrated_server.py > capibara6_server.log 2>&1 &

# Obtener el PID del proceso
SERVER_PID=$!
echo "✅ Servidor iniciado con PID: $SERVER_PID"

# Esperar un momento para que el servidor inicie
sleep 5

# Verificar si el servidor está corriendo en el puerto 5001
if netstat -tuln | grep 5001 > /dev/null; then
    echo "✅ Servidor principal activo en puerto 5001"
else
    echo "❌ Servidor principal NO está activo en puerto 5001"
    echo "📋 Verificando logs..."
    tail -20 capibara6_server.log
fi

echo ""
echo "🔧 Servidores actuales:"
echo "   - ARM-Axion vLLM: puerto 8080 (ya activo)"
echo "   - Backend Capibara6: puerto 5001 (ahora activo)"
echo ""
echo "🧪 Prueba: curl http://localhost:5001/health"