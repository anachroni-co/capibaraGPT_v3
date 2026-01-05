#!/bin/bash
# Script para iniciar el servidor ARM-Axion con los 5 modelos

set -e  # Exit on error

echo "🚀 INICIANDO SERVIDOR ARM-Axion MULTI-MODELO"
echo "   Con los 5 modelos ARM-Axion optimizados"
echo ""

# Configurar entorno ARM-Axion
export VLLM_USE_V1=0
export VLLM_ENABLE_V1_ENGINE=0
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_NO_DEPRECATION_WARNING=1
export VLLM_USE_FLASHINFER=0
export VLLM_USE_TRITON_FLASH_ATTN=0
export TORCHINDUCTOR_DISABLED=1
export TORCH_COMPILE_BACKEND=eager
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:disabled

echo "🔧 Variables de entorno ARM-Axion configuradas:"
echo "   - VLLM_USE_V1: $VLLM_USE_V1"
echo "   - VLLM_USE_FLASHINFER: $VLLM_USE_FLASHINFER"
echo "   - TORCHINDUCTOR_DISABLED: $TORCHINDUCTOR_DISABLED"

echo ""

# Directorios
CAPIBARA6_ROOT="/home/elect/capibara6"
ARM_AXION_DIR="$CAPIBARA6_ROOT/arm-axion-optimizations"
VLLM_INTEGRATION_DIR="$ARM_AXION_DIR/vllm_integration"
VLLM_MODIFIED_DIR="$CAPIBARA6_ROOT/vllm-source-modified"

# Configurar PYTHONPATH
export PYTHONPATH="$VLLM_MODIFIED_DIR:$ARM_AXION_DIR:$PYTHONPATH"
echo "✅ PYTHONPATH configurado:"
echo "   - $VLLM_MODIFIED_DIR (vLLM con detección ARM-Axion)"
echo "   - $ARM_AXION_DIR (módulos de integración)"
echo ""

# Cambiar al directorio correcto
cd "$VLLM_INTEGRATION_DIR"

# Verificar que la configuración exista
if [ ! -f "config.five_models.optimized.json" ]; then
    echo "❌ No se encontró config.five_models.optimized.json"
    ls -la config*.json
    exit 1
fi

# Crear enlace simbólico a la configuración de 5 modelos
ln -sf config.five_models.optimized.json config.json
echo "✅ Configuración de 5 modelos enlazada"

echo ""
echo "🔍 Verificando detección de plataforma ARM-Axion..."
python3 -c "
import sys
sys.path.insert(0, '$VLLM_MODIFIED_DIR')
from vllm.platforms import current_platform
if current_platform.is_cpu() and current_platform.device_type == 'cpu':
    print('✅ Plataforma ARM-Axion detectada correctamente:', current_platform.device_type)
else:
    print('❌ Plataforma ARM-Axion incorrecta:', current_platform.device_type)
    exit(1)
" || exit 1

echo ""
echo "🌐 Iniciando servidor ARM-Axion multi-modelo en puerto 8082..."
echo "   Presiona Ctrl+C para detener"
echo ""

# Iniciar servidor
PYTHONPATH="$PYTHONPATH" python3 multi_model_server.py --host 0.0.0.0 --port 8082