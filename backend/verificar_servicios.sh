#!/bin/bash

echo "========================================"
echo "  🔍 Verificador de Servicios Capibara6"
echo "========================================"
echo ""

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para verificar un servicio
check_service() {
    local name=$1
    local port=$2
    local endpoint=$3
    
    echo -n "Verificando $name (puerto $port)... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port$endpoint --connect-timeout 3)
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓ ACTIVO${NC}"
        return 0
    else
        echo -e "${RED}✗ NO RESPONDE (HTTP $response)${NC}"
        return 1
    fi
}

echo "1️⃣ Servicios principales:"
echo "----------------------------------------"
check_service "Gemma 3-12B" "8080" "/health"
check_service "Smart MCP" "5010" "/health"
check_service "Coqui TTS" "5002" "/health"

echo ""
echo "2️⃣ Screens activos:"
echo "----------------------------------------"
screens=$(screen -ls 2>/dev/null | grep -E "(gemma|mcp|tts|coqui)" | wc -l)
if [ $screens -gt 0 ]; then
    echo -e "${GREEN}✓ $screens screens encontrados:${NC}"
    screen -ls | grep -E "(gemma|mcp|tts|coqui)" | sed 's/^/  /'
else
    echo -e "${YELLOW}⚠️  No hay screens activos${NC}"
    echo "  Tip: Inicia servicios con screen -S nombre"
fi

echo ""
echo "3️⃣ Puertos en uso:"
echo "----------------------------------------"
for port in 8080 5010 5002; do
    if lsof -i :$port -t >/dev/null 2>&1; then
        pid=$(lsof -i :$port -t)
        process=$(ps -p $pid -o comm= 2>/dev/null)
        echo -e "${GREEN}✓ Puerto $port: PID $pid ($process)${NC}"
    else
        echo -e "${RED}✗ Puerto $port: NO ACTIVO${NC}"
    fi
done

echo ""
echo "4️⃣ Recursos del sistema:"
echo "----------------------------------------"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"
echo "RAM: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
echo "Disco: $(df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 " usado)"}')"

echo ""
echo "5️⃣ Conexión desde exterior:"
echo "----------------------------------------"
external_ip=$(curl -s ifconfig.me)
if [ ! -z "$external_ip" ]; then
    echo -e "${GREEN}✓ IP externa: $external_ip${NC}"
    echo ""
    echo "  URLs de verificación:"
    echo "  - http://$external_ip:8080/health  (Gemma)"
    echo "  - http://$external_ip:5010/health  (MCP)"
    echo "  - http://$external_ip:5002/health  (TTS)"
else
    echo -e "${RED}✗ No se pudo obtener IP externa${NC}"
fi

echo ""
echo "6️⃣ Variables de entorno Vercel:"
echo "----------------------------------------"
echo "  SMART_MCP_URL: http://$external_ip:5010/analyze"
echo "  KYUTAI_TTS_URL: http://$external_ip:5002/tts"

echo ""
echo "========================================"
echo "  Verificación completada"
echo "========================================"

