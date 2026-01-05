#!/bin/bash
# Script para iniciar Coqui TTS Server con virtualenv
# Uso: ./start_coqui_tts.sh

echo "========================================="
echo "  Iniciando Coqui TTS Server"
echo "========================================="

cd ~/capibara6/backend

# Verificar que el archivo existe
if [ ! -f "coqui_tts_server.py" ]; then
    echo "❌ Error: coqui_tts_server.py no encontrado"
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

python -c "import TTS" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚙️  Instalando Coqui TTS (puede tardar 5-10 minutos)..."
    echo "📥 Descargando modelos y dependencias..."
    pip install TTS
fi

echo ""
echo "✅ Dependencias listas"
echo ""
echo "🎙️  Coqui TTS - Alta calidad en español"
echo "📦 Modelo: tts_models/es/css10/vits"
echo "🔊 Calidad: Excelente (VITS neural)"
echo ""
echo "🚀 Iniciando servidor en puerto 5002..."
echo "⏳ Primera ejecución: ~30-60 seg (descarga modelo)"
echo ""

# Ejecutar servidor Coqui
python coqui_tts_server.py

