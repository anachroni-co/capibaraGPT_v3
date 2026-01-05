#!/bin/bash
# Script para iniciar todos los modelos en el servidor ARM-Axion

echo "🚀 INICIANDO SERVIDOR MULTI-MODELO ARM-AXION"
echo "   Con los 5 modelos: Qwen2.5, Phi4-mini, Mistral7B, Gemma3-27B, GPT-OSS-20B"
echo ""

# Configurar ambiente ARM-Axion
export VLLM_USE_V1=0
export VLLM_ENABLE_V1_ENGINE=0
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_USE_FLASHINFER=0
export VLLM_NO_DEPRECATION_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:disabled
export VLLM_USE_TRITON_FLASH_ATTN=0
export TORCHINDUCTOR_DISABLED=1
export TORCH_COMPILE_BACKEND=eager

echo "🔧 Variables de entorno ARM-Axion configuradas:"
echo "   - VLLM_USE_V1: $VLLM_USE_V1"
echo "   - VLLM_USE_TRITON_FLASH_ATTN: $VLLM_USE_TRITON_FLASH_ATTN" 
echo "   - TORCHINDUCTOR_DISABLED: $TORCHINDUCTOR_DISABLED"
echo "   - TORCH_COMPILE_BACKEND: $TORCH_COMPILE_BACKEND"

echo ""

# Directorios
CAPIBARA_ROOT="/home/elect/capibara6"
VLLM_MODIFIED_DIR="$CAPIBARA_ROOT/vllm-source-modified"
ARM_AXION_DIR="$CAPIBARA_ROOT/arm-axion-optimizations"

echo "📍 Directorios:"
echo "   - vLLM modificado: $VLLM_MODIFIED_DIR"
echo "   - ARM-Axion: $ARM_AXION_DIR"
echo ""

# Agregar al path
export PYTHONPATH="$VLLM_MODIFIED_DIR:$ARM_AXION_DIR:$PYTHONPATH"

# Verificar que la plataforma ARM-Axion esté correctamente detectada
echo "🔍 Verificando detección de plataforma ARM-Axion..."
python3 -c "
import sys
sys.path.insert(0, '$VLLM_MODIFIED_DIR')
from vllm.platforms import current_platform
print(f'Plataforma detectada: {current_platform.device_type}')
print(f'¿Es CPU?: {current_platform.is_cpu()}')
if current_platform.is_cpu() and current_platform.device_type == 'cpu':
    print('✅ ARM-Axion plataforma CPU detectada correctamente')
else:
    print('❌ Plataforma incorrecta')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Error en detección de plataforma ARM-Axion"
    exit 1
fi

echo ""

# Iniciar servidor en puerto 8082 para evitar conflicto con el que ya corre
PORT=8082
CONFIG_FILE="config.five_models.optimized.json"

echo "🌐 Iniciando servidor en puerto $PORT..."
echo "   Configuración: $CONFIG_FILE"
echo ""

cd "$ARM_AXION_DIR/vllm_integration"

# Usar el archivo de configuración correcto
if [ -f "$CONFIG_FILE" ]; then
    ln -sf "$CONFIG_FILE" config.json
elif [ -f "config.production.json" ]; then
    ln -sf "config.production.json" config.json
elif [ -f "config.five_models.optimized.json" ]; then
    ln -sf "config.five_models.optimized.json" config.json
else
    echo "❌ No se encontró archivo de configuración"
    ls -la config.*
    exit 1
fi

echo "✅ Archivo de configuración enlazado"

echo ""
echo "🚀 Iniciando servidor vLLM ARM-Axion multi-modelo..."
echo "   Accede a: http://localhost:$PORT"
echo "   Para detener: Ctrl+C"
echo ""

# Iniciar el servidor con el entorno ARM-Axion corregido
exec python3 multi_model_server.py --host 0.0.0.0 --port $PORT