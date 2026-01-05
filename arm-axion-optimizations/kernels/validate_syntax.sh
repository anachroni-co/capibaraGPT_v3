#!/bin/bash

# Script para validar sintaxis de kernels NEON (sin compilar)
# Útil cuando estás desarrollando en x86 pero el target es ARM

echo "════════════════════════════════════════════════════════════════"
echo "  Validación de Sintaxis - Kernels NEON ARM Axion"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar arquitectura actual
ARCH=$(uname -m)
echo "Arquitectura actual: $ARCH"

if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo -e "${GREEN}✓ Corriendo en ARM - puedes compilar y ejecutar${NC}"
    echo ""
    echo "Para compilar y ejecutar:"
    echo "  make"
    echo "  make run"
    exit 0
else
    echo -e "${YELLOW}⚠ Corriendo en $ARCH - solo validación de sintaxis${NC}"
    echo ""
fi

# Verificar si g++ está disponible
if ! command -v g++ &> /dev/null; then
    echo -e "${RED}✗ g++ no encontrado${NC}"
    echo "Instala con: sudo apt install g++"
    exit 1
fi

echo "Verificando sintaxis de archivos C++..."
echo ""

# Lista de archivos a verificar
FILES=(
    "neon_matmul.cpp"
    "benchmark_optimized.cpp"
)

ERRORS=0

for file in "${FILES[@]}"; do
    echo -n "  Verificando $file... "

    # Verificar solo sintaxis (sin compilar)
    # Usamos -fsyntax-only y definimos __ARM_NEON para que reconozca los intrinsics
    if g++ -std=c++17 -fsyntax-only -D__ARM_NEON -D__aarch64__ \
           -Wno-attributes "$file" 2>/tmp/syntax_error_$$.txt; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ ERROR${NC}"
        cat /tmp/syntax_error_$$.txt
        ERRORS=$((ERRORS + 1))
    fi

    rm -f /tmp/syntax_error_$$.txt
done

echo ""
echo "════════════════════════════════════════════════════════════════"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ Todos los archivos tienen sintaxis válida${NC}"
    echo ""
    echo "Próximos pasos:"
    echo ""
    echo "1. Sube estos archivos a una VM ARM Axion:"
    echo "   gcloud compute scp --recurse kernels/ your-vm-name:~/capibara6/arm-axion-optimizations/"
    echo ""
    echo "2. SSH a la VM ARM:"
    echo "   gcloud compute ssh your-vm-name"
    echo ""
    echo "3. Compila y ejecuta:"
    echo "   cd ~/capibara6/arm-axion-optimizations/kernels"
    echo "   make check-arch  # Verificar que es ARM"
    echo "   make             # Compilar"
    echo "   make run         # Ejecutar benchmarks"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Se encontraron $ERRORS errores de sintaxis${NC}"
    echo "Por favor corrige los errores antes de continuar."
    exit 1
fi
