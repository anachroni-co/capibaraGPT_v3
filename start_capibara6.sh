#!/bin/bash
# start_capibara6.sh - Script para iniciar todos los servicios de Capibara6

echo "🚀 Iniciando sistema Capibara6..."

# Iniciar el proxy CORS local en el puerto 8001
echo "🔌 Iniciando proxy CORS local en puerto 8001..."
cd backend
python3 cors_proxy_local.py > cors_proxy.log 2>&1 &
PROXY_PID=$!
echo "🔗 Proxy CORS iniciado con PID: $PROXY_PID"

# Esperar un momento para que el proxy inicie
sleep 3

# Verificar que el proxy esté corriendo
if curl -s http://localhost:8001/ > /dev/null; then
    echo "✅ Proxy CORS está corriendo en el puerto 8001"
else
    echo "❌ Proxy CORS no está respondiendo. Revisando logs..."
    tail -n 20 cors_proxy.log
    exit 1
fi

echo ""
echo "🎉 Capibara6 está listo!"
echo ""
echo "🔌 Servicios:"
echo "   Proxy CORS local: http://localhost:8001"
echo "   Backend remoto: http://34.12.166.76:5001"
echo ""
echo "🌐 Para iniciar el frontend:"
echo "   cd web && python3 -m http.server 8000"
echo "   Luego abrir: http://localhost:8000/chat.html"
echo ""
echo "📝 Documentación: FIX_CONNECTION_ISSUE.md"
echo ""
echo "PID del proxy CORS: $PROXY_PID (guardar para detenerlo después)"