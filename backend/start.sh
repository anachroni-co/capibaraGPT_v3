#!/bin/bash
# Script de inicio rápido para el backend de capibara6

echo "🦫 Iniciando backend de capibara6..."

# Verificar si existe el entorno virtual
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
echo "🔌 Activando entorno virtual..."
source venv/bin/activate

# Instalar/actualizar dependencias
echo "📥 Instalando dependencias..."
pip install -q -r requirements.txt

# Verificar si existe .env
if [ ! -f ".env" ]; then
    echo "⚠️  Archivo .env no encontrado!"
    echo "📝 Copia env.example a .env y configura tus credenciales SMTP:"
    echo "   cp env.example .env"
    echo "   nano .env"
    exit 1
fi

# Crear directorio de datos
mkdir -p user_data

# Iniciar servidor
echo "🚀 Iniciando servidor en http://localhost:5000"
python server.py

