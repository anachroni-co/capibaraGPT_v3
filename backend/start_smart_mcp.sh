#!/bin/bash
# Script para iniciar Smart MCP Server con virtualenv
# Uso: ./start_smart_mcp.sh

echo "========================================="
echo "  Iniciando Smart MCP Server"
echo "========================================="

cd ~/capibara6/backend

# Verificar que el archivo existe
if [ ! -f "smart_mcp_server.py" ]; then
    echo "❌ Error: smart_mcp_server.py no encontrado"
    echo "💡 Ejecutar primero: deploy_services_to_vm.sh"
    exit 1
fi

# Crear virtualenv si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando virtualenv..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Error creando virtualenv"
        echo "💡 Instalar: sudo apt install python3-venv"
        exit 1
    fi
    echo "✅ Virtualenv creado"
fi

# Activar virtualenv
source venv/bin/activate

# Verificar e instalar dependencias
echo "📦 Verificando dependencias..."

python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚙️  Instalando Flask..."
    pip install flask flask-cors
fi

echo ""
echo "✅ Dependencias listas"
echo "🚀 Iniciando servidor en puerto 5010..."
echo ""

# Ejecutar servidor
python smart_mcp_server.py

