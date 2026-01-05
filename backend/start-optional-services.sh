#!/bin/bash
# Capibara6 - Script de inicio de servicios opcionales
# Este script inicia servicios adicionales que NO son requeridos para funcionamiento básico

set -e

echo "➕ Capibara6 - Iniciando Servicios Opcionales..."
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directorio del script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Archivo para almacenar PIDs
PID_FILE="$SCRIPT_DIR/.optional_service_pids"
> "$PID_FILE"  # Limpiar archivo de PIDs

# Función para verificar si un puerto está en uso
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Puerto en uso
    else
        return 1  # Puerto libre
    fi
}

# Función para iniciar un servicio
start_service() {
    local name=$1
    local script=$2
    local port=$3

    echo -e "${YELLOW}▶️  Verificando $name (Puerto $port)...${NC}"

    if check_port $port; then
        echo -e "${RED}⚠️  Puerto $port ya está en uso. Omitiendo $name${NC}"
        return 1
    fi

    if [ ! -f "$script" ]; then
        echo -e "${RED}❌ Archivo no encontrado: $script${NC}"
        return 1
    fi

    echo -e "${GREEN}✓ Iniciando $name...${NC}"
    python3 "$script" > "logs/${name}.log" 2>&1 &
    local pid=$!

    # Guardar PID
    echo "$name:$pid:$port" >> "$PID_FILE"

    # Esperar un momento para verificar que inició correctamente
    sleep 2

    if ps -p $pid > /dev/null; then
        echo -e "${GREEN}✅ $name iniciado (PID: $pid)${NC}"
        return 0
    else
        echo -e "${RED}❌ $name falló al iniciar. Ver logs/${name}.log${NC}"
        return 1
    fi
}

# Crear directorio de logs si no existe
mkdir -p logs

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SERVICIOS OPCIONALES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "ℹ️  Estos servicios mejoran la funcionalidad pero NO son requeridos"
echo ""

# 1. MCP Server (Contexto inteligente y RAG)
echo -e "${YELLOW}1. MCP Server - Model Context Protocol${NC}"
echo "   Proporciona: RAG, contexto inteligente, herramientas (calculadora, verificación)"
start_service "mcp_server" "mcp_server.py" 5003

echo ""

# 2. Consensus Server (Multi-modelo)
echo -e "${YELLOW}2. Consensus Server - Multi-modelo${NC}"
echo "   Proporciona: Consenso entre múltiples modelos de IA"
start_service "consensus_server" "consensus_server.py" 5005

echo ""

# 3. Smart MCP Server (RAG selectivo alternativo)
echo -e "${YELLOW}3. Smart MCP Server - RAG selectivo${NC}"
echo "   Proporciona: Alternativa ligera a MCP con RAG selectivo"
echo -e "${YELLOW}   ⚠️  Nota: Solo iniciar si NO está usando mcp_server.py${NC}"

read -p "¿Iniciar Smart MCP Server? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    start_service "smart_mcp_server" "smart_mcp_server.py" 5010
else
    echo -e "${YELLOW}   ⏭️  Omitiendo Smart MCP Server${NC}"
fi

echo ""

# 4. FastAPI Server (Alternativa a server_gptoss.py)
echo -e "${YELLOW}4. FastAPI Server - API alternativa con E2B${NC}"
echo "   Proporciona: Endpoints alternativos, integración E2B"
echo -e "${YELLOW}   ⚠️  Nota: El frontend usa server_gptoss.py por defecto${NC}"

read -p "¿Iniciar FastAPI Server? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # FastAPI usa uvicorn, no python3 directo
    echo -e "${GREEN}✓ Iniciando FastAPI Server...${NC}"
    uvicorn main:app --host 0.0.0.0 --port 8000 > "logs/fastapi_server.log" 2>&1 &
    local pid=$!
    echo "fastapi_server:$pid:8000" >> "$PID_FILE"

    sleep 2

    if ps -p $pid > /dev/null; then
        echo -e "${GREEN}✅ FastAPI Server iniciado (PID: $pid)${NC}"
    else
        echo -e "${RED}❌ FastAPI Server falló al iniciar. Ver logs/fastapi_server.log${NC}"
    fi
else
    echo -e "${YELLOW}   ⏭️  Omitiendo FastAPI Server${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Resumen
if [ -f "$PID_FILE" ] && [ -s "$PID_FILE" ]; then
    echo -e "${GREEN}✅ Servicios opcionales iniciados${NC}"
    echo ""
    echo "📋 Servicios activos:"
    cat "$PID_FILE" | while IFS=: read name pid port; do
        echo "   • $name (PID: $pid, Puerto: $port)"
    done

    echo ""
    echo "📝 Logs disponibles en: $SCRIPT_DIR/logs/"
    echo ""
    echo "🔍 Para verificar estado: ./check-services.sh"
    echo "🛑 Para detener servicios: ./stop-all-services.sh"
else
    echo -e "${YELLOW}ℹ️  No se iniciaron servicios opcionales${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
