#!/bin/bash
# Script para iniciar el servidor web del frontend

set -e

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 Iniciando Servidor Web Frontend${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Verificar que estamos en el directorio correcto
if [ ! -d "web" ]; then
    echo -e "${YELLOW}⚠️  Directorio 'web' no encontrado${NC}"
    echo -e "${CYAN}Cambiando al directorio del proyecto...${NC}"
    cd "$(dirname "$0")"
fi

# Verificar si el puerto 8000 está en uso
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 || netstat -tuln 2>/dev/null | grep -q ":8000 " || ss -tuln 2>/dev/null | grep -q ":8000 "; then
    echo -e "${YELLOW}⚠️  El puerto 8000 ya está en uso${NC}"
    echo -e "${CYAN}Intentando usar puerto 8001...${NC}"
    PORT=8001
else
    PORT=8000
fi

cd web

echo -e "${GREEN}✅ Directorio: $(pwd)${NC}"
echo -e "${GREEN}✅ Puerto: $PORT${NC}"
echo ""
echo -e "${CYAN}📡 Servidor iniciado en:${NC}"
echo -e "${BLUE}   http://localhost:$PORT${NC}"
echo ""
echo -e "${CYAN}📄 Archivos disponibles:${NC}"
echo -e "   • http://localhost:$PORT/chat.html"
echo -e "   • http://localhost:$PORT/index.html"
echo ""
echo -e "${YELLOW}⚠️  El servidor se ejecutará en primer plano${NC}"
echo -e "${YELLOW}⚠️  Presiona Ctrl+C para detenerlo${NC}"
echo ""
echo -e "${BLUE}========================================${NC}\n"

# Iniciar servidor
python3 -m http.server $PORT

