#!/bin/bash
#
# Script de inicio rápido para el servidor vLLM ARM-Axion con 5 modelos
# versión modificada para pruebas con puerto fijo 8080

set -e  # Exit on error

echo "🚀 Iniciando servidor vLLM ARM-Axion con 5 modelos para pruebas..."
echo ""

# Configuración
VLLM_PORT="${1:-8080}"
HOST="${2:-0.0.0.0}"
CONFIG_FILE="${3:-config.five_models.optimized.json}"

# Si no se proporciona un puerto, usamos 8080 como predeterminado
if [ $# -eq 0 ]; then
    VLLM_PORT=8080
fi

echo " Puerto: $VLLM_PORT"
echo " Host: $HOST"
echo " Configuración: $CONFIG_FILE"
echo ""

# Verificar arquitectura
ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
    echo "⚠️  Advertencia: Este script está optimizado para arquitectura ARM64"
    read -p "¿Continuar de todos modos? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Directorios
CAPIBARA6_ROOT="/home/elect/capibara6"
ARM_AXION_DIR="$CAPIBARA6_ROOT/arm-axion-optimizations"
VLLM_INTEGRATION_DIR="$ARM_AXION_DIR/vllm_integration"
VLLM_MODIFIED_DIR="$CAPIBARA6_ROOT/vllm-source-modified"

# Verificar existencia de directorios
if [ ! -d "$VLLM_INTEGRATION_DIR" ]; then
    echo "❌ Directorio vllm_integration no encontrado: $VLLM_INTEGRATION_DIR"
    exit 1
fi

if [ ! -d "$VLLM_MODIFIED_DIR" ]; then
    echo "❌ Directorio vllm-source-modified no encontrado: $VLLM_MODIFIED_DIR"
    exit 1
fi

# Configurar PYTHONPATH
export PYTHONPATH="$VLLM_MODIFIED_DIR:$ARM_AXION_DIR:$PYTHONPATH"

# FORZAR uso del engine clásico (V0) para compatibilidad con CPU
export VLLM_USE_V1=0
export VLLM_ENABLE_V1_ENGINE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Deshabilitar características problemáticas de Triton/PyTorch
export VLLM_USE_TRITON_FLASH_ATTN=0
export TORCH_COMPILE_BACKEND=eager
export TORCHDYNAMO_DISABLED=1
export TORCHINDUCTOR_DISABLED=1

echo "✅ PYTHONPATH configurado:"
echo "   - $VLLM_MODIFIED_DIR (vLLM modificado con detección ARM-Axion)"
echo "   - $ARM_AXION_DIR (módulos vllm_integration)"
echo ""

# Verificar archivo de configuración
if [ ! -f "$VLLM_INTEGRATION_DIR/$CONFIG_FILE" ]; then
    echo "❌ Archivo de configuración no encontrado: $VLLM_INTEGRATION_DIR/$CONFIG_FILE"
    echo "✅ Archivos de configuración disponibles:"
    ls -la "$VLLM_INTEGRATION_DIR" | grep "config." | awk '{print "  - " $9}'
    exit 1
fi

echo "✅ Archivo de configuración encontrado: $CONFIG_FILE"
echo ""

# Verificar que la plataforma ARM-Axion sea detectada
echo "🔍 Verificando detección de plataforma ARM-Axion..."
python3 -c "
import sys
sys.path.insert(0, '$VLLM_MODIFIED_DIR')
from vllm.platforms import current_platform
if current_platform.is_cpu() and current_platform.device_type == 'cpu':
    print('✅ Plataforma ARM-Axion detectada correctamente: ' + current_platform.device_type)
else:
    print('❌ Plataforma incorrecta: ' + str(current_platform.device_type))
    sys.exit(1)
" || exit 1

echo ""

# Calentar los modelos uno por uno para probar la carga
echo "🔥 Calentando modelos para pruebas de carga..."
cd "$VLLM_INTEGRATION_DIR"

# Iniciar servidor
echo "🚀 Iniciando servidor..."
echo "   Endpoint: http://$HOST:$VLLM_PORT"
echo "   Configuración: $CONFIG_FILE"
echo "   Presiona Ctrl+C para detener"
echo ""

# Comando para iniciar el servidor
SERVER_CMD="python3 inference_server.py --host $HOST --port $VLLM_PORT"

# Ejecutar el servidor con las variables de entorno
exec $SERVER_CMD