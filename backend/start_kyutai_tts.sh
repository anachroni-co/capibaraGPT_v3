#!/bin/bash
# Script para iniciar Kyutai TTS Server con virtualenv
# Uso: ./start_kyutai_tts.sh

echo "========================================="
echo "  Iniciando Kyutai TTS Server"
echo "========================================="

cd ~/capibara6/backend

# Verificar que el archivo existe
if [ ! -f "kyutai_tts_server.py" ]; then
    echo "❌ Error: kyutai_tts_server.py no encontrado"
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

# Verificar e instalar moshi
PYTHON_CMD=$(python -c "import sys; print(sys.version_info.major)" 2>/dev/null)
if [ "$PYTHON_CMD" = "3" ]; then
    python -c "import moshi" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "⚙️  Instalando Moshi y dependencias (esto puede tardar varios minutos)..."
        pip install moshi>=0.2.11 safetensors>=0.4.0 sphn>=0.1.0 torch>=2.0.0 torchaudio>=2.0.0
        pip install sounddevice soundfile numpy transformers huggingface-hub
    fi
else
    echo "❌ Error: Se requiere Python 3"
    exit 1
fi

echo ""
echo "✅ Dependencias listas"
echo ""
echo "🚀 Iniciando servidor Kyutai TTS completo..."
echo ""

# Ejecutar servidor KYUTAI completo (no el fallback)
python kyutai_tts_server.py

echo ""
echo "========================================="
echo "  Kyutai TTS Server detenido"
echo "========================================="