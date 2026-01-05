#!/bin/bash
# Script para iniciar Coqui TTS con Python 3.11
# Requiere: Python 3.11 instalado (ver INSTALAR_PYTHON_311.md)

echo "========================================="
echo "  Iniciando Coqui TTS Server (Python 3.11)"
echo "========================================="

cd ~/capibara6/backend

# Verificar que Python 3.11 está instalado
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Error: Python 3.11 no encontrado"
    echo ""
    echo "Instalar con:"
    echo "  sudo add-apt-repository ppa:deadsnakes/ppa -y"
    echo "  sudo apt update"
    echo "  sudo apt install -y python3.11 python3.11-venv python3.11-dev"
    echo ""
    echo "Ver guía completa: INSTALAR_PYTHON_311.md"
    exit 1
fi

PYTHON_VERSION=$(python3.11 --version)
echo "🐍 Usando Python: $PYTHON_VERSION"

# Matar proceso en puerto 5002 si existe
PORT_PID=$(lsof -ti:5002 2>/dev/null)
if [ ! -z "$PORT_PID" ]; then
    echo "⚠️  Proceso encontrado en puerto 5002 (PID: $PORT_PID)"
    echo "🛑 Terminando proceso anterior..."
    kill -9 $PORT_PID 2>/dev/null
    sleep 1
    echo "✅ Puerto 5002 liberado"
fi

# Verificar que el archivo existe
if [ ! -f "coqui_tts_server.py" ]; then
    echo "❌ Error: coqui_tts_server.py no encontrado"
    exit 1
fi

# Crear virtualenv con Python 3.11 si no existe
if [ ! -d "venv_coqui" ]; then
    echo "📦 Creando virtualenv con Python 3.11..."
    python3.11 -m venv venv_coqui
    if [ $? -ne 0 ]; then
        echo "❌ Error creando virtualenv"
        echo "💡 Instalar: sudo apt install python3.11-venv"
        exit 1
    fi
    echo "✅ Virtualenv creado (venv_coqui/)"
fi

# Activar virtualenv
source venv_coqui/bin/activate

# Verificar que estamos usando Python 3.11
VENV_PYTHON_VERSION=$(python --version)
echo "✓ Python en venv: $VENV_PYTHON_VERSION"

# Verificar e instalar dependencias
echo "📦 Verificando dependencias..."

python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚙️  Instalando Flask..."
    pip install --upgrade pip
    pip install flask flask-cors
fi

python -c "import TTS" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "⚙️  Instalando Coqui TTS..."
    echo "📥 Esto puede tardar 5-10 minutos (descarga ~500 MB)"
    echo ""
    
    # Instalar con versión específica compatible
    pip install TTS==0.22.0
    
    if [ $? -ne 0 ]; then
        echo "❌ Error instalando Coqui TTS"
        exit 1
    fi
    
    echo "✅ Coqui TTS instalado exitosamente"
fi

echo ""
echo "✅ Todas las dependencias instaladas"
echo ""
echo "🎙️  Coqui TTS XTTS v2 - Máxima Calidad"
echo "📦 Modelo: xtts_v2 (multilingüe + clonación)"
echo "🔊 Calidad: ⭐⭐⭐⭐⭐ (La mejor disponible)"
echo "🌍 Idiomas: Español, Inglés, Francés, Alemán, y más"
echo "🐍 Python: 3.11 (compatible)"
echo ""
echo "🚀 Iniciando servidor en puerto 5002..."
echo "⏳ Primera ejecución: descargará ~2 GB (10-15 min)"
echo "💡 Descargas posteriores: ~30 seg (modelo en caché)"
echo ""

# Ejecutar servidor
python coqui_tts_server.py

