#!/bin/bash

# Script de prueba para verificar los scripts de Capibara6

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🧪 Verificando Scripts de Capibara6${NC}"
echo ""

# Test 1: Permisos
echo -e "${YELLOW}[1/6]${NC} Verificando permisos..."
if [ -x "start-capibara6.sh" ] && [ -x "quick-start.sh" ] && [ -x "stop-capibara6.sh" ]; then
    echo -e "${GREEN}  ✓ Permisos correctos${NC}"
else
    echo -e "${RED}  ✗ Faltan permisos de ejecución${NC}"
    exit 1
fi

# Test 2: Sintaxis
echo -e "${YELLOW}[2/6]${NC} Verificando sintaxis de scripts..."
errors=0
for script in start-capibara6.sh quick-start.sh stop-capibara6.sh; do
    if bash -n "$script" 2>/dev/null; then
        echo -e "${GREEN}  ✓ $script - sintaxis correcta${NC}"
    else
        echo -e "${RED}  ✗ $script - error de sintaxis${NC}"
        errors=$((errors + 1))
    fi
done

if [ $errors -gt 0 ]; then
    exit 1
fi

# Test 3: Docker Compose
echo -e "${YELLOW}[3/6]${NC} Verificando docker-compose.yml..."
if [ -f "docker-compose.yml" ]; then
    services=$(grep -c "container_name:" docker-compose.yml || echo 0)
    echo -e "${GREEN}  ✓ docker-compose.yml existe ($services servicios)${NC}"
else
    echo -e "${RED}  ✗ docker-compose.yml no encontrado${NC}"
    exit 1
fi

# Test 4: Directorios
echo -e "${YELLOW}[4/6]${NC} Verificando estructura de directorios..."
dirs=("backend" "web" "docs" "backend/logs")
missing=0
for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}  ✓ $dir/${NC}"
    else
        echo -e "${YELLOW}  ⚠ $dir/ no existe (se creará automáticamente)${NC}"
        mkdir -p "$dir"
    fi
done

# Test 5: Archivos backend
echo -e "${YELLOW}[5/6]${NC} Verificando archivos backend..."
if [ -f "backend/server.py" ]; then
    echo -e "${GREEN}  ✓ backend/server.py existe${NC}"
else
    echo -e "${RED}  ✗ backend/server.py no encontrado${NC}"
fi

# Test 6: Documentación
echo -e "${YELLOW}[6/6]${NC} Verificando documentación..."
docs=("INICIO-RAPIDO.md" "SERVICIOS.md")
for doc in "${docs[@]}"; do
    if [ -f "$doc" ]; then
        size=$(du -h "$doc" | cut -f1)
        echo -e "${GREEN}  ✓ $doc ($size)${NC}"
    else
        echo -e "${RED}  ✗ $doc no encontrado${NC}"
    fi
done

echo ""
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Verificación completada exitosamente${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo ""

echo -e "${CYAN}Resumen de Scripts:${NC}"
echo ""
echo -e "  ${GREEN}✓${NC} start-capibara6.sh  - Script maestro interactivo"
echo -e "  ${GREEN}✓${NC} quick-start.sh      - Inicio rápido automático"
echo -e "  ${GREEN}✓${NC} stop-capibara6.sh   - Detener todos los servicios"
echo ""

echo -e "${CYAN}Estado del Sistema:${NC}"
echo ""

# Verificar Docker
if command -v docker &> /dev/null; then
    docker_version=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
    echo -e "  ${GREEN}✓${NC} Docker $docker_version"
else
    echo -e "  ${RED}✗${NC} Docker no instalado"
fi

# Verificar Docker Compose
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Docker Compose instalado"
else
    echo -e "  ${RED}✗${NC} Docker Compose no instalado"
fi

# Verificar Python
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version | cut -d' ' -f2)
    echo -e "  ${GREEN}✓${NC} Python $python_version"
else
    echo -e "  ${RED}✗${NC} Python 3 no instalado"
fi

echo ""
echo -e "${CYAN}Próximo Paso:${NC}"
echo ""
echo -e "Para iniciar Capibara6, ejecuta:"
echo -e "  ${GREEN}./start-capibara6.sh${NC}"
echo ""
echo -e "Para inicio rápido:"
echo -e "  ${GREEN}./quick-start.sh${NC}"
echo ""
