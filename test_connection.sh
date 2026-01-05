#!/bin/bash
# test_connection.sh - Script para verificar la conexión con el backend de Capibara6

echo "🔍 Verificando conexión con backend remoto..."
echo ""

# Verificar si el backend remoto está accesible
echo "📡 Probando conexión con el backend remoto (34.12.166.76:5001)..."
if curl -s --connect-timeout 5 http://34.12.166.76:5001/api/health > /dev/null; then
    echo "✅ Backend remoto accesible"
    echo "📋 Información del backend:"
    curl -s http://34.12.166.76:5001/api/health | python3 -m json.tool
else
    echo "❌ Backend remoto no accesible (esto es normal si está detrás de firewall)"
fi

echo ""
echo "🔌 Iniciando proxy CORS local..."
cd backend

# Verificar si el puerto 8001 está ocupado
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null; then
    echo "⚠️  El puerto 8001 ya está en uso. Deténgalo antes de continuar."
    echo "PID en uso:"
    lsof -t -i:8001
    exit 1
fi

# Iniciar el proxy CORS local
python3 cors_proxy_local.py > cors_proxy.log 2>&1 &
PROXY_PID=$!
echo "🔗 Proxy CORS iniciado con PID: $PROXY_PID"

# Esperar a que el proxy inicie
sleep 3

# Verificar que el proxy esté corriendo
if curl -s http://localhost:8001/ > /dev/null; then
    echo "✅ Proxy CORS local corriendo en puerto 8001"
    echo ""
    echo "📋 Información del proxy:"
    curl -s http://localhost:8001/ | python3 -m json.tool
    echo ""
    echo "🧪 Probando conexión proxy -> backend remoto..."
    if curl -s http://localhost:8001/api/health > /dev/null; then
        echo "✅ Conexión proxy -> backend remoto exitosa"
    else
        echo "⚠️  Conexión proxy -> backend remoto fallida (puede ser por firewall)"
    fi
else
    echo "❌ Proxy CORS no está respondiendo"
    echo "📋 Logs del proxy:"
    tail -n 10 cors_proxy.log
    exit 1
fi

echo ""
echo "🌐 Servicios corriendo:"
echo "   Proxy CORS: http://localhost:8001"
echo "   Backend remoto: http://34.12.166.76:5001"
echo "   API Chat: http://localhost:8001/api/chat (a través del proxy)"
echo ""
echo "🚀 Capibara6 - Sistema listo para desarrollo"
echo ""
echo "📝 Para iniciar el frontend:"
echo "   1. cd web"
echo "   2. python3 -m http.server 8000"
echo "   3. Abrir http://localhost:8000/chat.html"
echo ""
echo "📌 Importante: El frontend está configurado para usar http://localhost:8001/api como proxy"
echo ""
echo "PID del proxy CORS: $PROXY_PID (guarde este número para detenerlo)"
echo ""
echo "📦 Para detener el proxy CORS: kill $PROXY_PID"