#!/bin/bash
# Script para iniciar el servidor con los 5 modelos, usando AWQ como estaban originalmente

echo "🚀 INICIANDO SERVIDOR MULTI-MODELO CAPIBARA6 (5 MODELOS - CONFIGURACIÓN ORIGINAL)"
echo "   Modelos: Phi4 (AWQ), Mistral (AWQ), Qwen (AWQ), Gemma3 (BF16), Aya Expanse (BF16)"
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

# Directorios
CAPIBARA_ROOT="/home/elect/capibara6"
VLLM_MODIFIED_DIR="$CAPIBARA_ROOT/vllm-source-modified"
ARM_AXION_DIR="$CAPIBARA_ROOT/arm-axion-optimizations"

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

# Puerto y configuración
PORT=8090
CONFIG_FILE="config.five_models_original_awq.json"

echo "🌐 Iniciando servidor en puerto $PORT..."
echo "   Configuración: $CONFIG_FILE"
echo "   Usando configuración AWQ original que funcionaba previamente"
echo ""

cd "$ARM_AXION_DIR/vllm_integration"

# Verificar que el archivo de configuración existe
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Archivo de configuración no encontrado: $CONFIG_FILE"
    exit 1
fi

echo "✅ Archivo de configuración encontrado"

echo ""
echo "🚀 Iniciando servidor vLLM multi-modelo (5 modelos con configuración original)..."
echo "   Accede a: http://localhost:$PORT"
echo "   Para detener: Ctrl+C"
echo "   ⚠️  Este proceso puede tardar 10-15 minutos en cargar todos los modelos"
echo ""

# Iniciar el servidor con la configuración original
exec python3 multi_model_server.py --host 0.0.0.0 --port $PORT --config $CONFIG_FILE