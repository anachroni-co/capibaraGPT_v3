#!/bin/bash
#
# Script para iniciar la interfaz interactiva ARM-Axion
# con el entorno adecuadamente configurado

set -e  # Exit on error

echo "🚀 Iniciando Interfaz Interactiva ARM-Axion..."
echo ""

# Directorios
CAPIBARA6_ROOT="/home/elect/capibara6"
VLLM_MODIFIED_DIR="$CAPIBARA6_ROOT/vllm-source-modified"
ARM_AXION_DIR="$CAPIBARA6_ROOT/arm-axion-optimizations"

# Configurar PYTHONPATH
export PYTHONPATH="$VLLM_MODIFIED_DIR:$ARM_AXION_DIR:$PYTHONPATH"

echo "✅ PYTHONPATH configurado:"
echo "   - $VLLM_MODIFIED_DIR (vLLM con detección ARM-Axion)"
echo "   - $ARM_AXION_DIR (módulos ARM-Axion)"
echo ""

# Verificar que la plataforma ARM-Axion sea detectada
echo "🔍 Verificando detección de plataforma ARM-Axion..."
python3 -c "
import sys
sys.path.insert(0, '$VLLM_MODIFIED_DIR')
from vllm.platforms import current_platform
if current_platform.is_cpu() and current_platform.device_type == 'cpu':
    print('✅ Plataforma ARM-Axion detectada correctamente')
else:
    print('❌ Plataforma incorrecta: ' + str(current_platform.device_type))
    sys.exit(1)
" || exit 1

echo ""

# Iniciar interfaz interactiva
echo "🎮 Iniciando interfaz interactiva..."
echo "Sigue las instrucciones en pantalla para probar los modelos"
echo ""

cd "$CAPIBARA6_ROOT"
exec python3 interactive_arm_axion_test.py