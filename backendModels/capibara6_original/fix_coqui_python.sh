#!/bin/bash
# Script para arreglar problema de Python 3.13 con Coqui TTS
# Coqui TTS requiere Python <3.12

echo "========================================="
echo "  Fix Coqui TTS - Python Compatibility"
echo "========================================="
echo ""

cd ~/capibara6/backend

# 1. Verificar versión de Python
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "🐍 Python actual: $PYTHON_VERSION"

# 2. Verificar si Python es >= 3.12
if [ "$(echo "$PYTHON_VERSION >= 3.12" | bc)" -eq 1 ]; then
    echo "⚠️  Python $PYTHON_VERSION NO es compatible con Coqui TTS (requiere <3.12)"
    echo ""
    echo "Soluciones:"
    echo ""
    echo "Opción 1: Instalar Python 3.11 con deadsnakes PPA"
    echo "-------------------------------------------------"
    echo "sudo apt update"
    echo "sudo apt install -y software-properties-common"
    echo "sudo add-apt-repository ppa:deadsnakes/ppa -y"
    echo "sudo apt update"
    echo "sudo apt install -y python3.11 python3.11-venv python3.11-dev"
    echo ""
    echo "Luego editar start_coqui_tts.sh y cambiar:"
    echo "  python3 -m venv venv"
    echo "  por:"
    echo "  python3.11 -m venv venv"
    echo ""
    echo "Opción 2: Usar servidor TTS alternativo (más simple)"
    echo "--------------------------------------------------"
    echo "Usar: ./start_kyutai_tts.sh (servidor fallback simple)"
    echo ""
    echo "Opción 3: Docker con Python 3.11 (avanzado)"
    echo "-------------------------------------------"
    echo "Ver COQUI_TTS_DOCKER.md"
    echo ""
    exit 1
else
    echo "✅ Python $PYTHON_VERSION es compatible con Coqui TTS"
fi

# 3. Verificar si el puerto 5001 está en uso
echo ""
echo "Verificando puerto 5001..."
PORT_IN_USE=$(lsof -ti:5001 2>/dev/null)

if [ ! -z "$PORT_IN_USE" ]; then
    echo "⚠️  Puerto 5001 está en uso por proceso: $PORT_IN_USE"
    echo ""
    echo "¿Quieres matar el proceso? (s/n)"
    read -r KILL_PROCESS
    
    if [ "$KILL_PROCESS" = "s" ] || [ "$KILL_PROCESS" = "S" ]; then
        kill -9 $PORT_IN_USE
        echo "✅ Proceso $PORT_IN_USE terminado"
    else
        echo "❌ No se puede iniciar servidor en puerto 5001"
        echo "💡 Mata el proceso manualmente: kill -9 $PORT_IN_USE"
        exit 1
    fi
else
    echo "✅ Puerto 5001 disponible"
fi

echo ""
echo "========================================="
echo "  Todo listo para instalar Coqui TTS"
echo "========================================="

