#!/bin/bash

# Script para detener todos los servicios de Capibara6
# Uso: ./stop-capibara6.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🦫 Deteniendo Capibara6...${NC}"
echo ""

# Ir al directorio del script
cd "$(dirname "${BASH_SOURCE[0]}")"

# Detener backend Python
echo -e "${YELLOW}[1/3]${NC} Deteniendo backend Python..."
if [ -f "backend/logs/backend.pid" ]; then
    kill $(cat backend/logs/backend.pid) 2>/dev/null || true
    rm -f backend/logs/backend.pid
    echo -e "${GREEN}  ✓ Backend detenido${NC}"
else
    echo -e "  ℹ Backend no estaba corriendo"
fi

# Detener frontend
echo -e "${YELLOW}[2/3]${NC} Deteniendo frontend..."
if [ -f "backend/logs/frontend.pid" ]; then
    kill $(cat backend/logs/frontend.pid) 2>/dev/null || true
    rm -f backend/logs/frontend.pid
    echo -e "${GREEN}  ✓ Frontend detenido${NC}"
else
    echo -e "  ℹ Frontend no estaba corriendo"
fi

# Detener servicios Docker
echo -e "${YELLOW}[3/3]${NC} Deteniendo servicios Docker..."
docker-compose down
echo -e "${GREEN}  ✓ Servicios Docker detenidos${NC}"

echo ""
echo -e "${GREEN}✓ Todos los servicios de Capibara6 han sido detenidos${NC}"
echo ""
