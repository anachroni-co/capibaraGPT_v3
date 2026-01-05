#!/bin/bash
# Capibara6 - Script para detener todos los servicios

set -e

echo "🛑 Capibara6 - Deteniendo Servicios..."
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directorio del script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/.service_pids"

# Función para detener un servicio por nombre de proceso
stop_by_name() {
    local name=$1
    local script=$2

    echo -e "${YELLOW}Buscando procesos: $script${NC}"

    # Buscar PIDs
    pids=$(pgrep -f "$script" || true)

    if [ -z "$pids" ]; then
        echo -e "${YELLOW}  ℹ️  No hay procesos de $name corriendo${NC}"
        return 0
    fi

    # Detener cada PID
    for pid in $pids; do
        echo -e "${GREEN}  ⏹️  Deteniendo $name (PID: $pid)${NC}"
        kill -TERM $pid 2>/dev/null || true

        # Esperar a que termine (máximo 5 segundos)
        local count=0
        while ps -p $pid > /dev/null 2>&1 && [ $count -lt 5 ]; do
            sleep 1
            count=$((count + 1))
        done

        # Si aún está corriendo, forzar
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${RED}  ⚠️  Forzando detención de PID: $pid${NC}"
            kill -9 $pid 2>/dev/null || true
        fi
    done

    echo -e "${GREEN}  ✓ $name detenido${NC}"
}

# Detener servicios principales
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DETENIENDO SERVICIOS PRINCIPALES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

stop_by_name "Backend Principal" "server_gptoss.py"
stop_by_name "TTS Server" "kyutai_tts_server.py"
stop_by_name "Auth Server" "auth_server.py"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DETENIENDO SERVICIOS OPCIONALES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

stop_by_name "MCP Server" "mcp_server.py"
stop_by_name "Consensus Server" "consensus_server.py"
stop_by_name "Smart MCP Server" "smart_mcp_server.py"
stop_by_name "FastAPI Server" "main.py"

# Limpiar archivo de PIDs
if [ -f "$PID_FILE" ]; then
    rm "$PID_FILE"
    echo -e "${GREEN}✓ Archivo de PIDs limpiado${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ Todos los servicios detenidos${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Verificar que no haya procesos colgados
remaining=$(pgrep -f "server_gptoss.py|kyutai_tts_server.py|mcp_server.py|auth_server.py|consensus_server.py|smart_mcp_server.py" || true)
if [ -n "$remaining" ]; then
    echo -e "${RED}⚠️  Advertencia: Algunos procesos aún están corriendo:${NC}"
    echo "$remaining"
    echo ""
    echo "Para forzar detención: pkill -9 -f 'server'"
else
    echo -e "${GREEN}✓ No hay procesos residuales${NC}"
fi
