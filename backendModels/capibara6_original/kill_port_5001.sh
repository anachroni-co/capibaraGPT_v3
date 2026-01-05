#!/bin/bash
# Script rápido para liberar el puerto 5001

echo "🔍 Buscando procesos en puerto 5001..."

PORT_PID=$(lsof -ti:5001 2>/dev/null)

if [ -z "$PORT_PID" ]; then
    echo "✅ Puerto 5001 ya está libre"
    exit 0
fi

echo "⚠️  Proceso encontrado: PID $PORT_PID"
echo "📋 Información del proceso:"
ps -p $PORT_PID -o pid,cmd

echo ""
echo "🛑 Terminando proceso..."
kill -9 $PORT_PID

sleep 1

# Verificar que se mató
if lsof -ti:5001 &>/dev/null; then
    echo "❌ El proceso aún está corriendo"
    exit 1
else
    echo "✅ Puerto 5001 liberado exitosamente"
fi

