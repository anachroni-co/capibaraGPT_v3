#!/bin/bash
# Script para iniciar el backend en bounty2
# Ejecutar desde bounty2 después de conectarse via SSH

set -e

echo "🚀 Iniciando Backend en bounty2..."
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -d "backend" ]; then
    echo "⚠️  Directorio 'backend' no encontrado"
    echo "Buscando proyecto..."
    cd ~/capibara6 2>/dev/null || cd /home/*/capibara6 2>/dev/null || {
        echo "❌ No se encontró el proyecto. Por favor, navega al directorio del proyecto primero."
        exit 1
    }
fi

cd backend

echo "📁 Directorio actual: $(pwd)"
echo ""

# Verificar si Python 3 está disponible
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no está instalado"
    exit 1
fi

# Verificar si hay un entorno virtual
if [ -d "venv" ]; then
    echo "✅ Activando entorno virtual..."
    source venv/bin/activate
fi

# Verificar dependencias
if [ ! -f "requirements.txt" ]; then
    echo "⚠️  requirements.txt no encontrado"
else
    echo "📦 Verificando dependencias..."
    pip install -q -r requirements.txt 2>/dev/null || echo "⚠️  Algunas dependencias pueden faltar"
fi

echo ""
echo "🔍 Buscando archivo del servidor..."

# Buscar archivo del servidor
SERVER_FILE=""
if [ -f "capibara6_integrated_server.py" ]; then
    SERVER_FILE="capibara6_integrated_server.py"
elif [ -f "server.py" ]; then
    SERVER_FILE="server.py"
elif [ -f "server_gptoss.py" ]; then
    SERVER_FILE="server_gptoss.py"
else
    echo "❌ No se encontró archivo del servidor"
    echo "Archivos disponibles:"
    ls -la *.py 2>/dev/null | head -10
    exit 1
fi

echo "✅ Usando: $SERVER_FILE"
echo ""

# Verificar si el puerto 5001 está en uso
if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null 2>&1 || netstat -tuln 2>/dev/null | grep -q ":5001 " || ss -tuln 2>/dev/null | grep -q ":5001 "; then
    echo "⚠️  El puerto 5001 ya está en uso"
    echo "¿Deseas detener el proceso existente? (s/n)"
    read -r response
    if [ "$response" = "s" ] || [ "$response" = "S" ]; then
        echo "Deteniendo proceso en puerto 5001..."
        lsof -ti:5001 | xargs kill -9 2>/dev/null || pkill -f "$SERVER_FILE" 2>/dev/null || true
        sleep 2
    else
        echo "Usando puerto alternativo 5002..."
        PORT=5002
    fi
else
    PORT=5001
fi

echo ""
echo "🌐 Iniciando servidor en puerto $PORT..."
echo "📡 El servidor escuchará en: 0.0.0.0:$PORT"
echo "⚠️  Presiona Ctrl+C para detener"
echo ""

# Iniciar servidor
if [ "$SERVER_FILE" = "capibara6_integrated_server.py" ]; then
    python3 "$SERVER_FILE" --host 0.0.0.0 --port $PORT
else
    # Modificar temporalmente para escuchar en 0.0.0.0
    python3 -c "
import sys
sys.path.insert(0, '.')
from $SERVER_FILE import app
app.run(host='0.0.0.0', port=$PORT, debug=False)
"
fi

