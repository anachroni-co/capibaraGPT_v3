#!/bin/bash
# Script para iniciar el servidor multi-modelo Capibara6 con 5 modelos en consenso
# Incluyendo el modelo CohereLabs/aya-expanse-8b

echo "🚀 INICIANDO SERVIDOR MULTI-MODELO CAPIBARA6 CON 5 MODELOS EN CONSENSO"
echo "   Modelos: Phi4-mini, Mistral7B, Qwen2.5-coder, Gemma3-27b, Aya-expanse-8b"
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

# Verificar que el modelo Aya Expanse esté disponible
AYA_MODEL_PATH="/home/elect/models/aya-expanse-8b"
if [ ! -d "$AYA_MODEL_PATH" ] || [ ! -f "$AYA_MODEL_PATH/config.json" ]; then
    echo "⚠️  El modelo Aya Expanse 8B no está completamente descargado"
    echo "   Ruta: $AYA_MODEL_PATH"
    echo ""
    echo "💡 Para descargarlo:"
    echo "   1. Asegúrate de tener acceso a https://huggingface.co/CohereLabs/aya-expanse-8b"
    echo "   2. Ejecuta: bash $CAPIBARA_ROOT/download_aya_expanse.sh"
    echo ""
    echo "🔧 Continuando con los 4 modelos disponibles..."
    CONFIG_FILE="config.four_models_no_gptoss.json"  # Usar configuración de 4 modelos
    PORT=8081
else
    echo "✅ Modelo Aya Expanse 8B encontrado"
    CONFIG_FILE="config.five_models_with_aya.json"  # Usar configuración de 5 modelos
    PORT=8080
fi

echo "🌐 Iniciando servidor en puerto $PORT..."
echo "   Configuración: $CONFIG_FILE"
echo ""

cd "$ARM_AXION_DIR/vllm_integration"

# Verificar que el archivo de configuración existe
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Archivo de configuración no encontrado: $CONFIG_FILE"
    ls -la | grep config
    exit 1
fi

echo "✅ Archivo de configuración encontrado"

echo ""
echo "🚀 Iniciando servidor vLLM multi-modelo con $CONFIG_FILE..."
echo "   Accede a: http://localhost:$PORT"
echo "   Para detener: Ctrl+C"
echo ""

# Iniciar el servidor con el entorno ARM-Axion corregido y archivo de configuración específico
exec python3 multi_model_server.py --host 0.0.0.0 --port $PORT --config $CONFIG_FILE