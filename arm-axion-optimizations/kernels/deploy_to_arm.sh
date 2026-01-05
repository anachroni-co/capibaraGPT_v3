#!/bin/bash

# Script para desplegar y probar kernels optimizados en VM ARM Axion
# Uso: ./deploy_to_arm.sh [VM_NAME]

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "════════════════════════════════════════════════════════════════"
echo "  Deployment de Kernels NEON Optimizados a ARM Axion"
echo "════════════════════════════════════════════════════════════════"
echo -e "${NC}"

# Verificar parámetros
if [ -z "$1" ]; then
    echo -e "${YELLOW}Uso: $0 <VM_NAME>${NC}"
    echo ""
    echo "Ejemplo:"
    echo "  $0 vllm-arm-axion-1"
    echo ""
    echo "O sigue estos pasos manualmente:"
    echo ""
    echo -e "${GREEN}Paso 1: Subir archivos a VM ARM${NC}"
    echo "  gcloud compute scp --recurse . YOUR_VM:~/capibara6/arm-axion-optimizations/kernels/"
    echo ""
    echo -e "${GREEN}Paso 2: SSH a la VM${NC}"
    echo "  gcloud compute ssh YOUR_VM"
    echo ""
    echo -e "${GREEN}Paso 3: Compilar y ejecutar en la VM${NC}"
    echo "  cd ~/capibara6/arm-axion-optimizations/kernels"
    echo "  make check-arch  # Verificar ARM"
    echo "  make info        # Ver flags de compilación"
    echo "  make             # Compilar"
    echo "  make run         # Ejecutar benchmarks"
    echo ""
    exit 1
fi

VM_NAME=$1
ZONE=${2:-us-central1-a}  # Default zone

echo "VM Target: $VM_NAME"
echo "Zone: $ZONE"
echo ""

# Verificar si gcloud está instalado
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}✗ gcloud CLI no está instalado${NC}"
    echo "Instala desde: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Verificar si la VM existe
echo -e "${BLUE}[1/5]${NC} Verificando VM..."
if ! gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &> /dev/null; then
    echo -e "${RED}✗ VM '$VM_NAME' no encontrada en zone '$ZONE'${NC}"
    echo ""
    echo "VMs disponibles:"
    gcloud compute instances list
    exit 1
fi
echo -e "${GREEN}✓ VM encontrada${NC}"
echo ""

# Subir archivos
echo -e "${BLUE}[2/5]${NC} Subiendo archivos optimizados..."
gcloud compute scp --recurse \
    --zone="$ZONE" \
    ./* "$VM_NAME":~/capibara6/arm-axion-optimizations/kernels/

echo -e "${GREEN}✓ Archivos subidos${NC}"
echo ""

# Verificar arquitectura
echo -e "${BLUE}[3/5]${NC} Verificando arquitectura ARM..."
ARCH=$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="uname -m")
if [ "$ARCH" != "aarch64" ]; then
    echo -e "${RED}✗ VM no es ARM (arquitectura: $ARCH)${NC}"
    echo "Este código requiere una VM ARM Axion (C4A instance)"
    exit 1
fi
echo -e "${GREEN}✓ Arquitectura: $ARCH${NC}"
echo ""

# Compilar
echo -e "${BLUE}[4/5]${NC} Compilando kernels optimizados..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
    cd ~/capibara6/arm-axion-optimizations/kernels &&
    make clean &&
    make
"
echo -e "${GREEN}✓ Compilación exitosa${NC}"
echo ""

# Ejecutar benchmarks
echo -e "${BLUE}[5/5]${NC} Ejecutando benchmarks..."
echo ""
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
    cd ~/capibara6/arm-axion-optimizations/kernels &&
    ./benchmark_optimized
"

echo ""
echo -e "${GREEN}"
echo "════════════════════════════════════════════════════════════════"
echo "  ✓ Deployment y benchmarking completado"
echo "════════════════════════════════════════════════════════════════"
echo -e "${NC}"
echo ""
echo "Próximos pasos:"
echo ""
echo "1. Revisar resultados de benchmarks arriba"
echo "2. Comparar con baseline (ver README_OPTIMIZATIONS.md)"
echo "3. Integrar kernels optimizados en vLLM"
echo ""
echo "Para SSH a la VM:"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
