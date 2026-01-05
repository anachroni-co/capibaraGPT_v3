#!/bin/bash
# Capibara6 - Script de inicio de servicios principales
# Este script inicia los servicios CORE necesarios para el funcionamiento básico
# Para servicios opcionales, usar: start-optional-services.sh

set -e

echo "🚀 Capibara6 - Iniciando Servicios Principales..."
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
PID_FILE="$SCRIPT_DIR/.service_pids"
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
echo "  SERVICIOS PRINCIPALES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. Backend Principal (REQUERIDO)
start_service "server_gptoss" "server_gptoss.py" 5001

# 2. TTS Server (REQUERIDO para síntesis de voz)
start_service "kyutai_tts" "kyutai_tts_server.py" 5002

# 3. Auth Server (REQUERIDO para login)
start_service "auth_server" "auth_server.py" 5004

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Resumen
echo -e "${GREEN}✅ Servicios principales iniciados${NC}"
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
echo "➕ Para servicios opcionales: ./start-optional-services.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Mantener el script corriendo (opcional)
# Descomentar para que el script no termine hasta que se detengan los servicios
# echo ""
# echo "Presiona Ctrl+C para detener todos los servicios..."
# wait
