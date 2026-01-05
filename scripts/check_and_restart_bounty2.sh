#!/bin/bash
# Script para verificar y reiniciar servicios en Bounty2
# Ejecutar este script dentro de la VM Bounty2

set -e

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "================================================================================"
echo "VERIFICACION Y REINICIO DE SERVICIOS - BOUNTY2"
echo "================================================================================"
echo ""

# Función para verificar si un puerto está en uso
check_port() {
    local port=$1
    if lsof -ti:$port > /dev/null 2>&1 || netstat -tuln 2>/dev/null | grep -q ":$port "; then
        return 0
    else
        return 1
    fi
}

# Función para verificar servicio HTTP
check_http_service() {
    local url=$1
    local name=$2
    
    if curl -s --connect-timeout 3 "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name: ACTIVO${NC}"
        return 0
    else
        echo -e "${RED}❌ $name: INACTIVO${NC}"
        return 1
    fi
}

echo "1. Verificando servicios corriendo..."
echo "--------------------------------------------------------------------------------"

# Verificar Ollama (puerto 11434)
OLLAMA_ACTIVE=false
if check_port 11434; then
    echo -e "${GREEN}✅ Puerto 11434 (Ollama): EN USO${NC}"
    PID=$(lsof -ti:11434 2>/dev/null || netstat -tuln 2>/dev/null | grep ":11434 " | awk '{print $7}' | cut -d'/' -f1 | head -1)
    if [ -n "$PID" ]; then
        echo "   PID: $PID"
        ps -p $PID -o cmd= 2>/dev/null | head -c 80
        echo ""
    fi
    
    if check_http_service "http://localhost:11434/api/tags" "Ollama API"; then
        OLLAMA_ACTIVE=true
    fi
else
    echo -e "${RED}❌ Puerto 11434 (Ollama): LIBRE${NC}"
fi

# Verificar Backend Flask (puerto 5001)
BACKEND_ACTIVE=false
if check_port 5001; then
    echo -e "${GREEN}✅ Puerto 5001 (Backend Flask): EN USO${NC}"
    PID=$(lsof -ti:5001 2>/dev/null || netstat -tuln 2>/dev/null | grep ":5001 " | awk '{print $7}' | cut -d'/' -f1 | head -1)
    if [ -n "$PID" ]; then
        echo "   PID: $PID"
        ps -p $PID -o cmd= 2>/dev/null | head -c 80
        echo ""
    fi
    
    if check_http_service "http://localhost:5001/api/health" "Backend Flask"; then
        BACKEND_ACTIVE=true
    fi
else
    echo -e "${RED}❌ Puerto 5001 (Backend Flask): LIBRE${NC}"
fi

echo ""
echo "2. Verificando procesos relacionados..."
echo "--------------------------------------------------------------------------------"

# Buscar procesos Ollama
OLLAMA_PROCESSES=$(ps aux | grep -E "ollama" | grep -v grep || echo "")
if [ -n "$OLLAMA_PROCESSES" ]; then
    echo -e "${GREEN}✅ Procesos Ollama encontrados:${NC}"
    echo "$OLLAMA_PROCESSES" | head -3
else
    echo -e "${RED}❌ No se encontraron procesos Ollama${NC}"
fi

# Buscar procesos Python relacionados con backend
BACKEND_PROCESSES=$(ps aux | grep -E "python.*server|python.*backend|python.*capibara6|flask" | grep -v grep || echo "")
if [ -n "$BACKEND_PROCESSES" ]; then
    echo -e "${GREEN}✅ Procesos Backend encontrados:${NC}"
    echo "$BACKEND_PROCESSES" | head -3
else
    echo -e "${RED}❌ No se encontraron procesos Backend${NC}"
fi

echo ""
echo "================================================================================"
echo "REINICIANDO SERVICIOS"
echo "================================================================================"
echo ""

# Reiniciar Ollama si no está activo
if [ "$OLLAMA_ACTIVE" = false ]; then
    echo "Iniciando Ollama..."
    
    # Detener Ollama existente
    if check_port 11434; then
        PID=$(lsof -ti:11434 2>/dev/null || echo "")
        if [ -n "$PID" ]; then
            kill -9 $PID 2>/dev/null || true
            sleep 2
        fi
    fi
    
    # Iniciar Ollama
    if command -v ollama > /dev/null; then
        if command -v screen > /dev/null; then
            screen -dmS ollama bash -c "ollama serve; exec bash" || {
                nohup ollama serve > /tmp/ollama.log 2>&1 &
            }
        else
            nohup ollama serve > /tmp/ollama.log 2>&1 &
        fi
        echo -e "${GREEN}✅ Ollama iniciado${NC}"
        echo "   Logs: /tmp/ollama.log"
        echo "   Screen: screen -r ollama"
    else
        echo -e "${RED}❌ Ollama no está instalado${NC}"
    fi
fi

# Reiniciar Backend Flask si no está activo
if [ "$BACKEND_ACTIVE" = false ]; then
    echo ""
    echo "Iniciando Backend Flask..."
    
    # Detener backend existente
    if check_port 5001; then
        PID=$(lsof -ti:5001 2>/dev/null || echo "")
        if [ -n "$PID" ]; then
            echo "   Matando proceso en puerto 5001 (PID: $PID)..."
            kill -9 $PID 2>/dev/null || true
            sleep 2
        fi
    fi
    
    # Buscar script de inicio
    BACKEND_START_SCRIPT=""
    if [ -f "backend/capibara6_integrated_server.py" ]; then
        BACKEND_START_SCRIPT="python3 backend/capibara6_integrated_server.py"
    elif [ -f "backend/server.py" ]; then
        BACKEND_START_SCRIPT="python3 backend/server.py"
    elif [ -f "backend/server_gptoss.py" ]; then
        BACKEND_START_SCRIPT="python3 backend/server_gptoss.py"
    elif [ -f "backend/start_integrated_server.py" ]; then
        BACKEND_START_SCRIPT="python3 backend/start_integrated_server.py"
    fi
    
    if [ -n "$BACKEND_START_SCRIPT" ]; then
        echo "   Script encontrado: $BACKEND_START_SCRIPT"
        
        # Asegurar que escucha en 0.0.0.0
        export FLASK_HOST=0.0.0.0
        export FLASK_PORT=5001
        
        if command -v screen > /dev/null; then
            screen -dmS backend bash -c "cd $(pwd) && $BACKEND_START_SCRIPT; exec bash" || {
                nohup $BACKEND_START_SCRIPT > /tmp/backend.log 2>&1 &
            }
        else
            nohup $BACKEND_START_SCRIPT > /tmp/backend.log 2>&1 &
        fi
        
        echo -e "${GREEN}✅ Backend iniciado${NC}"
        echo "   Logs: /tmp/backend.log"
        echo "   Screen: screen -r backend"
    else
        echo -e "${YELLOW}⚠️  No se encontró script de inicio de backend${NC}"
        echo "   Buscando archivos relacionados..."
        
        BACKEND_FILES=$(find backend -name "*server*.py" -o -name "*backend*.py" 2>/dev/null | head -5 || echo "")
        if [ -n "$BACKEND_FILES" ]; then
            echo "   Archivos encontrados:"
            echo "$BACKEND_FILES" | while read file; do
                echo "     - $file"
            done
        fi
    fi
fi

echo ""
echo "3. Esperando que los servicios inicien..."
echo "--------------------------------------------------------------------------------"

sleep 5

# Verificar nuevamente
if check_port 11434; then
    if check_http_service "http://localhost:11434/api/tags" "Ollama"; then
        echo -e "${GREEN}✅ Ollama está respondiendo correctamente${NC}"
    fi
fi

if check_port 5001; then
    if check_http_service "http://localhost:5001/api/health" "Backend Flask"; then
        echo -e "${GREEN}✅ Backend Flask está respondiendo correctamente${NC}"
    else
        echo -e "${YELLOW}⚠️  Puerto 5001 ocupado pero servicio no responde aún${NC}"
        echo "   Espera unos segundos más y verifica: curl http://localhost:5001/api/health"
    fi
fi

echo ""
echo "================================================================================"
echo "RESUMEN"
echo "================================================================================"
echo ""

# Estado final
check_http_service "http://localhost:11434/api/tags" "Ollama (puerto 11434)"
check_http_service "http://localhost:5001/api/health" "Backend Flask (puerto 5001)"

echo ""
echo "Comandos útiles:"
echo "  - Ver logs backend: tail -f /tmp/backend.log"
echo "  - Ver logs Ollama: tail -f /tmp/ollama.log"
echo "  - Ver sesión screen backend: screen -r backend"
echo "  - Ver sesión screen Ollama: screen -r ollama"
echo "  - Verificar salud: curl http://localhost:5001/api/health"
echo "  - Ver puertos: sudo netstat -tuln | grep LISTEN"
echo ""

