#!/bin/bash
# Script completo para verificar y levantar servicios en gpt-oss-20b
# Ejecutar desde gpt-oss-20b después de conectarse via SSH

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🔍 Verificación y Inicio de Servicios${NC}"
echo -e "${BLUE}VM: gpt-oss-20b${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Función para verificar si un puerto está en uso
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 || \
       netstat -tuln 2>/dev/null | grep -q ":$port " || \
       ss -tuln 2>/dev/null | grep -q ":$port "; then
        return 0
    else
        return 1
    fi
}

# Función para verificar servicio HTTP
check_http_service() {
    local port=$1
    local path=$2
    if curl -s --connect-timeout 2 "http://localhost:$port$path" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Función para iniciar servicio en screen
start_service_screen() {
    local name=$1
    local command=$2
    local port=$3
    
    if check_port $port; then
        echo -e "${YELLOW}⚠️  Puerto $port ya está en uso${NC}"
        return 1
    fi
    
    echo -e "${CYAN}🚀 Iniciando $name en puerto $port...${NC}"
    screen -dmS $name bash -c "$command"
    sleep 3
    
    if check_port $port; then
        echo -e "${GREEN}✅ $name iniciado correctamente${NC}"
        return 0
    else
        echo -e "${RED}❌ Error iniciando $name${NC}"
        return 1
    fi
}

# Navegar al directorio del proyecto
if [ ! -d "backend" ]; then
    echo -e "${YELLOW}⚠️  Buscando directorio del proyecto...${NC}"
    cd ~/capibara6 2>/dev/null || cd /home/*/capibara6 2>/dev/null || {
        echo -e "${RED}❌ No se encontró el proyecto. Por favor, navega al directorio del proyecto primero.${NC}"
        exit 1
    }
fi

cd backend
echo -e "${GREEN}📁 Directorio: $(pwd)${NC}\n"

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 no está instalado${NC}"
    exit 1
fi

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    echo -e "${CYAN}✅ Activando entorno virtual...${NC}"
    source venv/bin/activate
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}📊 Estado Actual de Servicios${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Verificar servicios existentes
SERVICES=(
    "5000:Bridge/Main Server:server.py o server_gptoss.py"
    "5002:TTS Server:kyutai_tts_server.py o coqui_tts_server.py"
    "5003:MCP Server:smart_mcp_server.py o mcp_server.py"
    "5010:MCP Server Alt:smart_mcp_server.py"
    "5678:N8n:n8n (Docker o servicio)"
)

for service_info in "${SERVICES[@]}"; do
    IFS=':' read -r port name file <<< "$service_info"
    if check_port $port; then
        echo -e "${GREEN}✅ Puerto $port ($name) - ACTIVO${NC}"
        if check_http_service $port "/health"; then
            echo -e "   ${GREEN}   HTTP responde correctamente${NC}"
        elif check_http_service $port "/api/health"; then
            echo -e "   ${GREEN}   HTTP responde en /api/health${NC}"
        else
            echo -e "   ${YELLOW}   Puerto abierto pero HTTP no responde${NC}"
        fi
    else
        echo -e "${RED}❌ Puerto $port ($name) - INACTIVO${NC}"
    fi
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 Iniciando Servicios Faltantes${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 1. Bridge/Main Server (puerto 5000)
if ! check_port 5000; then
    if [ -f "server_gptoss.py" ]; then
        start_service_screen "bridge" "python3 server_gptoss.py" 5000
    elif [ -f "server.py" ]; then
        start_service_screen "bridge" "python3 server.py" 5000
    else
        echo -e "${YELLOW}⚠️  No se encontró server.py o server_gptoss.py${NC}"
    fi
fi

# 2. TTS Server (puerto 5002)
if ! check_port 5002; then
    if [ -f "kyutai_tts_server.py" ]; then
        start_service_screen "tts" "python3 kyutai_tts_server.py" 5002
    elif [ -f "coqui_tts_server.py" ]; then
        start_service_screen "tts" "python3 coqui_tts_server.py" 5002
    else
        echo -e "${YELLOW}⚠️  No se encontró servidor TTS${NC}"
    fi
fi

# 3. MCP Server (puerto 5003)
if ! check_port 5003; then
    if [ -f "smart_mcp_server.py" ]; then
        start_service_screen "mcp" "python3 smart_mcp_server.py" 5003
    elif [ -f "mcp_server.py" ]; then
        start_service_screen "mcp" "python3 mcp_server.py" 5003
    else
        echo -e "${YELLOW}⚠️  No se encontró servidor MCP${NC}"
    fi
fi

# 4. MCP Server Alternativo (puerto 5010)
if ! check_port 5010; then
    if [ -f "smart_mcp_server.py" ]; then
        # Modificar para usar puerto 5010
        start_service_screen "mcp-alt" "python3 -c \"
import sys
sys.path.insert(0, '.')
from smart_mcp_server import app
app.run(host='0.0.0.0', port=5010, debug=False)
\"" 5010
    fi
fi

# 5. N8n (puerto 5678)
if ! check_port 5678; then
    echo -e "${CYAN}🚀 Iniciando N8n...${NC}"
    # Verificar si está en Docker
    if command -v docker &> /dev/null; then
        if docker ps -a | grep -q n8n; then
            docker start n8n 2>/dev/null || docker-compose up -d n8n 2>/dev/null
            echo -e "${GREEN}✅ N8n iniciado desde Docker${NC}"
        else
            echo -e "${YELLOW}⚠️  Contenedor N8n no encontrado${NC}"
        fi
    # Verificar si está como servicio systemd
    elif systemctl list-units --type=service | grep -q n8n; then
        sudo systemctl start n8n
        echo -e "${GREEN}✅ N8n iniciado como servicio${NC}"
    else
        echo -e "${YELLOW}⚠️  N8n no encontrado (Docker o systemd)${NC}"
    fi
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}📊 Estado Final${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Verificar estado final
for service_info in "${SERVICES[@]}"; do
    IFS=':' read -r port name file <<< "$service_info"
    if check_port $port; then
        echo -e "${GREEN}✅ Puerto $port ($name) - ACTIVO${NC}"
    else
        echo -e "${RED}❌ Puerto $port ($name) - INACTIVO${NC}"
    fi
done

echo ""
echo -e "${CYAN}📋 Screens activos:${NC}"
screen -ls

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🧪 Pruebas de Conexión${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Probar servicios
test_services() {
    local port=$1
    local path=$2
    local name=$3
    
    if curl -s --connect-timeout 2 "http://localhost:$port$path" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name ($port$path) - OK${NC}"
        return 0
    else
        echo -e "${RED}❌ $name ($port$path) - FALLO${NC}"
        return 1
    fi
}

test_services 5000 "/api/health" "Bridge Health"
test_services 5000 "/health" "Bridge Health (alt)"
test_services 5002 "/api/tts/voices" "TTS Voices"
test_services 5002 "/health" "TTS Health"
test_services 5003 "/api/mcp/status" "MCP Status"
test_services 5003 "/health" "MCP Health"
test_services 5010 "/health" "MCP Alt Health"
test_services 5678 "/healthz" "N8n Health"

echo ""
echo -e "${GREEN}✅ Verificación completada${NC}"
echo ""
echo -e "${CYAN}💡 Comandos útiles:${NC}"
echo -e "   Ver screens: ${YELLOW}screen -ls${NC}"
echo -e "   Entrar a screen: ${YELLOW}screen -r <nombre>${NC}"
echo -e "   Detener screen: ${YELLOW}screen -X -S <nombre> quit${NC}"
echo -e "   Ver logs: ${YELLOW}screen -r <nombre>${NC}"

