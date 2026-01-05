#!/bin/bash

# Quick Start Script - Inicio rápido de Capibara6
# Uso: ./quick-start.sh

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🦫 Iniciando Capibara6...${NC}"
echo ""

# Ir al directorio del script
cd "$(dirname "${BASH_SOURCE[0]}")"

# Iniciar Docker Compose
echo -e "${GREEN}[1/3]${NC} Iniciando servicios Docker..."
docker-compose up -d

# Esperar a que los servicios estén listos
echo -e "${GREEN}[2/3]${NC} Esperando a que los servicios estén listos..."
sleep 5

# Iniciar backend
echo -e "${GREEN}[3/3]${NC} Iniciando servidor backend..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

nohup python3 server.py > logs/backend.log 2>&1 &
echo $! > logs/backend.pid

cd ..

echo ""
echo -e "${GREEN}✓ Capibara6 iniciado correctamente${NC}"
echo ""
echo "Accede a:"
echo -e "  • Frontend:  ${CYAN}http://localhost:8080${NC}"
echo -e "  • Backend:   ${CYAN}http://localhost:5000${NC}"
echo -e "  • n8n:       ${CYAN}http://localhost:5678${NC}"
echo -e "  • Grafana:   ${CYAN}http://localhost:3000${NC}"
echo ""
