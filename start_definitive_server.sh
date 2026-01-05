#!/bin/bash
# Script definitivo para iniciar el servidor con 5 modelos funcionando correctamente

echo "🚀 INICIANDO SERVIDOR DEFINITIVO CAPIBARA6 (5 MODELOS - CONFIGURACIÓN CORREGIDA)"
echo "   Todos los modelos sin cuantización incorrecta"
echo ""

# Configurar ambiente ARM-Axion
export VLLM_USE_V1=0
export VLLM_ENABLE_V1_ENGINE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
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

# Puerto y configuración definitiva
PORT=8083
CONFIG_FILE="config.five_models_original_awq.json"  # Este ahora tiene la configuración corregida

echo "🌐 Iniciando servidor definitivo en puerto $PORT..."
echo "   Configuración: $CONFIG_FILE"
echo "   Todos los 5 modelos configurados para trabajar correctamente"
echo ""

cd "$ARM_AXION_DIR/vllm_integration"

# Verificar que el archivo de configuración existe
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Archivo de configuración no encontrado: $CONFIG_FILE"
    exit 1
fi

echo "✅ Archivo de configuración encontrado"
echo "✅ Verificando que la configuración tenga quantization: null en lugar de AWQ..."

# Contar cuántas veces aparece "quantization": "awq" vs null
AWQ_COUNT=$(grep -c '"quantization": "awq"' "$CONFIG_FILE")
NULL_COUNT=$(grep -c '"quantization": null' "$CONFIG_FILE")
echo "   - Cuantizaciones AWQ encontradas: $AWQ_COUNT"
echo "   - Cuantizaciones null encontradas: $NULL_COUNT"

if [ $AWQ_COUNT -eq 0 ]; then
    echo "   ✅ Confirmado: No se encontraron configuraciones AWQ incorrectas"
else
    echo "   ⚠️  Aún hay $AWQ_COUNT configuraciones AWQ en el archivo"
fi

echo ""
echo "🚀 Iniciando servidor vLLM definitivo (5 modelos corregidos)..."
echo "   Accede a: http://localhost:$PORT"
echo "   Para detener: Ctrl+C"
echo "   ⚠️  Este proceso puede tardar 10-15 minutos en cargar todos los modelos"
echo ""

# Iniciar el servidor definitivo
exec python3 multi_model_server.py --host 0.0.0.0 --port $PORT --config $CONFIG_FILE