#!/bin/bash

# Script de Deployment para Gemma 3 27B Multimodal
# VM: models-europe (europe-southwest1-b)

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     Gemma 3 27B Multimodal - Deployment Script                    ║"
echo "║     VM: models-europe (ARM Axion)                                  ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Configuración
MODEL_DIR="/home/elect/models"
GEMMA3_MODEL="gemma-3-27b-it-awq"
CAPIBARA_DIR="$HOME/capibara6"
VLLM_DIR="$CAPIBARA_DIR/arm-axion-optimizations/vllm-integration"

# ============================================================================
# FASE 1: Verificación Pre-Deployment
# ============================================================================

echo "[STEP] Fase 1: Verificación Pre-Deployment"
echo ""

# Verificar arquitectura
echo "[STEP] 1.1 Verificando arquitectura ARM..."
ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
    echo "[ERROR] Arquitectura no es ARM: $ARCH"
    exit 1
fi
echo "  ✓ Arquitectura: $ARCH"

# Verificar RAM disponible
echo "[STEP] 1.2 Verificando RAM disponible..."
TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
AVAILABLE_RAM=$(free -g | awk '/^Mem:/{print $7}')
echo "  ✓ RAM total: ${TOTAL_RAM}GB"
echo "  ✓ RAM disponible: ${AVAILABLE_RAM}GB"

if [ "$AVAILABLE_RAM" -lt 20 ]; then
    echo "[WARNING] RAM disponible baja (<20GB). Considere liberar memoria."
fi

# Verificar disco disponible
echo "[STEP] 1.3 Verificando espacio en disco..."
AVAILABLE_DISK=$(df -h $MODEL_DIR | awk 'NR==2 {print $4}')
echo "  ✓ Espacio disponible en $MODEL_DIR: $AVAILABLE_DISK"

# Verificar vLLM
echo "[STEP] 1.4 Verificando vLLM..."
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "[WARNING] vLLM no está instalado. Instalando..."
    pip3 install vllm
fi
echo "  ✓ vLLM instalado"

# Verificar huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "[WARNING] huggingface-cli no está instalado. Instalando..."
    pip3 install -U "huggingface_hub[cli]"
fi
echo "  ✓ huggingface-cli instalado"

echo ""
echo "[STEP] ✓ Verificación Pre-Deployment completada"
echo ""

# ============================================================================
# FASE 2: Descarga del Modelo
# ============================================================================

echo "[STEP] Fase 2: Descarga del Modelo Gemma 3 27B AWQ"
echo ""

cd "$MODEL_DIR"

if [ -d "$GEMMA3_MODEL" ]; then
    echo "[WARNING] Modelo ya existe en $MODEL_DIR/$GEMMA3_MODEL"
    read -p "¿Desea re-descargar? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$GEMMA3_MODEL"
    else
        echo "[STEP] Saltando descarga del modelo"
    fi
fi

if [ ! -d "$GEMMA3_MODEL" ]; then
    echo "[STEP] Descargando Gemma 3 27B AWQ (~14-16GB)..."
    echo "[WARNING] Esto puede tomar 2-4 horas según la conexión"
    
    # Intentar con AWQ primero
    if huggingface-cli download casperhansen/gemma-3-27b-it-awq \
        --local-dir "$GEMMA3_MODEL" \
        --local-dir-use-symlinks False; then
        echo "  ✓ Modelo AWQ descargado"
    else
        echo "[ERROR] No se pudo descargar el modelo AWQ"
        echo "Intenta manualmente: huggingface-cli download google/gemma-3-27b-it --local-dir gemma-3-27b-it"
        exit 1
    fi
fi

# Verificar modelo descargado
echo "[STEP] Verificando modelo descargado..."
if [ ! -f "$GEMMA3_MODEL/config.json" ]; then
    echo "[ERROR] Modelo no descargado correctamente (falta config.json)"
    exit 1
fi

MODEL_SIZE=$(du -sh "$GEMMA3_MODEL" | cut -f1)
echo "  ✓ Modelo descargado: $MODEL_SIZE"

echo ""
echo "[STEP] ✓ Descarga del Modelo completada"
echo ""

# ============================================================================
# FASE 3: Configuración
# ============================================================================

echo "[STEP] Fase 3: Configuración de vLLM"
echo ""

cd "$VLLM_DIR"

# Backup de configuración actual
echo "[STEP] 3.1 Creando backup de configuración actual..."
if [ -f "config.production.json" ]; then
    cp config.production.json "config.production.json.backup.$(date +%Y%m%d_%H%M%S)"
    echo "  ✓ Backup creado"
fi

# Copiar nueva configuración
echo "[STEP] 3.2 Actualizando configuración..."
if [ -f "config.gemma3.json" ]; then
    cp config.gemma3.json config.production.json
    echo "  ✓ Configuración actualizada"
else
    echo "[ERROR] config.gemma3.json no encontrado"
    exit 1
fi

echo ""
echo "[STEP] ✓ Configuración completada"
echo ""

# ============================================================================
# FASE 4: Testing
# ============================================================================

echo "[STEP] Fase 4: Testing del Modelo"
echo ""

echo "[STEP] 4.1 Test de carga del modelo..."
python3 << 'EOFPYTHON'
import sys
from vllm import LLM

try:
    print("Cargando Gemma 3 27B AWQ...")
    llm = LLM(
        model="/home/elect/models/gemma-3-27b-it-awq",
        quantization="awq",
        max_model_len=8192,
        tensor_parallel_size=1,
        trust_remote_code=True
    )
    print("✓ Modelo cargado exitosamente")
    
    # Test de generación
    print("\nTest de generación...")
    outputs = llm.generate(
        ["Hola, ¿cómo estás?"],
        sampling_params={"temperature": 0.7, "max_tokens": 50}
    )
    print(f"Respuesta: {outputs[0].outputs[0].text}")
    print("\n✓ Test de generación exitoso")
    
except Exception as e:
    print(f"✗ Error: {e}", file=sys.stderr)
    sys.exit(1)
EOFPYTHON

if [ $? -ne 0 ]; then
    echo "[ERROR] Test de carga falló"
    exit 1
fi

echo ""
echo "[STEP] ✓ Testing completado"
echo ""

# ============================================================================
# FASE 5: Deployment
# ============================================================================

echo "[STEP] Fase 5: Deployment"
echo ""

echo "[WARNING] El servidor vLLM se reiniciará con la nueva configuración"
read -p "¿Continuar con el deployment? (y/N): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "[WARNING] Deployment cancelado por el usuario"
    exit 0
fi

# Reiniciar servicio
echo "[STEP] 5.1 Reiniciando servicio vLLM..."
if systemctl is-active --quiet vllm-capibara6; then
    sudo systemctl restart vllm-capibara6
    echo "  ✓ Servicio reiniciado"
else
    echo "[WARNING] Servicio vllm-capibara6 no está activo. Iniciando..."
    sudo systemctl start vllm-capibara6
fi

# Verificar estado del servicio
echo "[STEP] 5.2 Verificando estado del servicio..."
sleep 5

if systemctl is-active --quiet vllm-capibara6; then
    echo "  ✓ Servicio activo"
else
    echo "[ERROR] Servicio no se inició correctamente"
    echo "Ver logs: sudo journalctl -u vllm-capibara6 -n 50"
    exit 1
fi

# Health check
echo "[STEP] 5.3 Verificando health check..."
sleep 10

if curl -s http://localhost:8080/health | grep -q "healthy"; then
    echo "  ✓ Health check OK"
else
    echo "[WARNING] Health check falló. Verificar logs."
fi

echo ""
echo "[STEP] ✓ Deployment completado"
echo ""

# ============================================================================
# Resumen Final
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                    DEPLOYMENT EXITOSO                              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Modelo: Gemma 3 27B AWQ"
echo "Ubicación: $MODEL_DIR/$GEMMA3_MODEL"
echo "Tamaño: $MODEL_SIZE"
echo "Configuración: $VLLM_DIR/config.production.json"
echo ""
echo "Próximos pasos:"
echo "  1. Monitorear logs: sudo journalctl -u vllm-capibara6 -f"
echo "  2. Verificar métricas: curl http://localhost:8080/stats | jq"
echo "  3. Test de generación: curl http://localhost:8080/v1/completions -d '{...}'"
echo ""
echo "[STEP] ¡Deployment completado exitosamente!"
