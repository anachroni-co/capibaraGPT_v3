#!/bin/bash
# Script para verificar todos los servicios de Capibara6

echo "============================================"
echo "  📊 Estado de Servicios - Capibara6"
echo "============================================"
echo ""

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Función para verificar un servicio
check_service() {
    local name=$1
    local port=$2
    local endpoint=$3
    
    echo -n "🔍 $name (puerto $port): "
    
    if curl -s -m 2 http://localhost:$port$endpoint > /dev/null 2>&1; then
        echo -e "${GREEN}✅ ACTIVO${NC}"
        
        # Mostrar información adicional si es posible
        response=$(curl -s -m 2 http://localhost:$port$endpoint 2>/dev/null)
        if [ ! -z "$response" ]; then
            echo "   📋 $(echo $response | jq -r '.service // .status // .' 2>/dev/null || echo $response | head -c 100)"
        fi
    else
        echo -e "${RED}❌ INACTIVO${NC}"
    fi
}

# Verificar cada servicio
echo "Verificando servicios en localhost..."
echo ""

check_service "Gemma Model    " 8080 "/health"
check_service "Smart MCP      " 5003 "/health"
check_service "Coqui TTS      " 5002 "/health"
check_service "TTS Fallback   " 5001 "/health"

echo ""
echo "============================================"
echo "  🖥️  Sesiones de Screen Activas"
echo "============================================"
echo ""

if command -v screen &> /dev/null; then
    screen -ls || echo "Sin sesiones de screen activas"
else
    echo "⚠️  Screen no instalado"
fi

echo ""
echo "============================================"
echo "  🔌 Puertos en Uso"
echo "============================================"
echo ""

# Mostrar puertos 5001-5003 y 8080
for port in 5001 5002 5003 8080; do
    PID=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$PID" ]; then
        echo -e "Puerto $port: ${GREEN}EN USO${NC} (PID: $PID)"
        ps -p $PID -o cmd= | head -c 80
        echo ""
    else
        echo -e "Puerto $port: ${YELLOW}LIBRE${NC}"
    fi
done

echo ""
echo "============================================"
echo "  📝 Comandos Útiles"
echo "============================================"
echo ""
echo "Ver logs de un servicio:"
echo "  screen -r coqui-tts"
echo "  screen -r smart-mcp"
echo "  screen -r gemma"
echo ""
echo "Iniciar servicios:"
echo "  screen -S coqui-tts; ./start_coqui_tts_py311.sh"
echo "  screen -S smart-mcp; ./start_smart_mcp.sh"
echo ""
echo "Salir de screen sin cerrar: Ctrl+A, luego D"
echo ""

