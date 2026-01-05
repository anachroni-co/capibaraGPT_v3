#!/bin/bash
# Script para verificar y reiniciar servicios en rag3
# Ejecutar este script dentro de la VM rag3

set -e

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "================================================================================"
echo "VERIFICACION Y REINICIO DE SERVICIOS - RAG3"
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

# Verificar RAG API (puerto 8000)
RAG_API_ACTIVE=false
if check_port 8000; then
    echo -e "${GREEN}✅ Puerto 8000 (RAG API): EN USO${NC}"
    PID=$(lsof -ti:8000 2>/dev/null || netstat -tuln 2>/dev/null | grep ":8000 " | awk '{print $7}' | cut -d'/' -f1 | head -1)
    if [ -n "$PID" ]; then
        echo "   PID: $PID"
        ps -p $PID -o cmd= 2>/dev/null | head -c 80
        echo ""
    fi
    
    if check_http_service "http://localhost:8000/health" "RAG API Health"; then
        RAG_API_ACTIVE=true
    fi
else
    echo -e "${RED}❌ Puerto 8000 (RAG API): LIBRE${NC}"
fi

# Verificar PostgreSQL (puerto 5432)
if check_port 5432; then
    echo -e "${GREEN}✅ Puerto 5432 (PostgreSQL): EN USO${NC}"
else
    echo -e "${YELLOW}⚠️  Puerto 5432 (PostgreSQL): LIBRE${NC}"
fi

# Verificar Redis (puerto 6379)
if check_port 6379; then
    echo -e "${GREEN}✅ Puerto 6379 (Redis): EN USO${NC}"
else
    echo -e "${YELLOW}⚠️  Puerto 6379 (Redis): LIBRE${NC}"
fi

echo ""
echo "2. Verificando procesos relacionados..."
echo "--------------------------------------------------------------------------------"

# Buscar procesos Python relacionados con RAG
RAG_PROCESSES=$(ps aux | grep -E "python.*rag|rag.*python|uvicorn.*rag|fastapi.*rag" | grep -v grep || echo "")
if [ -n "$RAG_PROCESSES" ]; then
    echo -e "${GREEN}✅ Procesos RAG encontrados:${NC}"
    echo "$RAG_PROCESSES" | while read line; do
        echo "   $line"
    done
else
    echo -e "${RED}❌ No se encontraron procesos RAG${NC}"
fi

echo ""
echo "3. Verificando sesiones de screen..."
echo "--------------------------------------------------------------------------------"

SCREEN_SESSIONS=$(screen -ls 2>/dev/null | grep -E "rag|RAG" || echo "")
if [ -n "$SCREEN_SESSIONS" ]; then
    echo -e "${GREEN}✅ Sesiones de screen encontradas:${NC}"
    echo "$SCREEN_SESSIONS"
else
    echo -e "${YELLOW}⚠️  No hay sesiones de screen activas${NC}"
fi

echo ""
echo "================================================================================"
echo "REINICIANDO SERVICIOS"
echo "================================================================================"
echo ""

# Detener servicios existentes si están corriendo
if [ "$RAG_API_ACTIVE" = false ] || [ -z "$RAG_PROCESSES" ]; then
    echo "Deteniendo servicios RAG existentes..."
    
    # Matar procesos en puerto 8000
    if check_port 8000; then
        PID=$(lsof -ti:8000 2>/dev/null || echo "")
        if [ -n "$PID" ]; then
            echo "   Matando proceso en puerto 8000 (PID: $PID)..."
            kill -9 $PID 2>/dev/null || true
            sleep 2
        fi
    fi
    
    # Matar procesos Python relacionados con RAG
    pkill -f "python.*rag" 2>/dev/null || true
    pkill -f "uvicorn.*rag" 2>/dev/null || true
    pkill -f "fastapi.*rag" 2>/dev/null || true
    
    echo -e "${GREEN}✅ Servicios detenidos${NC}"
fi

echo ""
echo "Iniciando servicios RAG..."
echo "--------------------------------------------------------------------------------"

# Buscar script de inicio de RAG
RAG_START_SCRIPT=""
if [ -f "backend/start_rag_server.sh" ]; then
    RAG_START_SCRIPT="backend/start_rag_server.sh"
elif [ -f "start_rag.sh" ]; then
    RAG_START_SCRIPT="start_rag.sh"
elif [ -f "scripts/start_rag.sh" ]; then
    RAG_START_SCRIPT="scripts/start_rag.sh"
elif [ -f "backend/rag_server.py" ]; then
    RAG_START_SCRIPT="python3 backend/rag_server.py"
elif [ -f "rag_server.py" ]; then
    RAG_START_SCRIPT="python3 rag_server.py"
fi

if [ -n "$RAG_START_SCRIPT" ]; then
    echo "   Script encontrado: $RAG_START_SCRIPT"
    
    # Intentar iniciar en screen
    if command -v screen > /dev/null; then
        echo "   Iniciando en sesión de screen 'rag-api'..."
        screen -dmS rag-api bash -c "cd $(pwd) && $RAG_START_SCRIPT; exec bash" || {
            echo "   Intentando iniciar directamente..."
            nohup $RAG_START_SCRIPT > /tmp/rag_api.log 2>&1 &
        }
    else
        echo "   Iniciando en background..."
        nohup $RAG_START_SCRIPT > /tmp/rag_api.log 2>&1 &
    fi
    
    echo -e "${GREEN}✅ Servicio iniciado${NC}"
    echo "   Logs: /tmp/rag_api.log"
    echo "   Screen: screen -r rag-api"
else
    echo -e "${YELLOW}⚠️  No se encontró script de inicio de RAG${NC}"
    echo "   Buscando archivos relacionados..."
    
    # Buscar archivos Python relacionados con RAG
    RAG_FILES=$(find . -name "*rag*server*.py" -o -name "*rag*api*.py" 2>/dev/null | head -5 || echo "")
    if [ -n "$RAG_FILES" ]; then
        echo "   Archivos encontrados:"
        echo "$RAG_FILES" | while read file; do
            echo "     - $file"
        done
        echo ""
        echo "   Intenta iniciar manualmente:"
        echo "     screen -S rag-api"
        echo "     python3 [ARCHIVO_ENCONTRADO]"
        echo "     Ctrl+A, D para salir"
    fi
fi

echo ""
echo "4. Esperando que los servicios inicien..."
echo "--------------------------------------------------------------------------------"

sleep 5

# Verificar nuevamente
if check_port 8000; then
    echo -e "${GREEN}✅ Puerto 8000 ahora está en uso${NC}"
    
    if check_http_service "http://localhost:8000/health" "RAG API"; then
        echo -e "${GREEN}✅ RAG API está respondiendo correctamente${NC}"
    else
        echo -e "${YELLOW}⚠️  Puerto ocupado pero servicio no responde aún${NC}"
        echo "   Espera unos segundos más y verifica: curl http://localhost:8000/health"
    fi
else
    echo -e "${RED}❌ Puerto 8000 aún no está en uso${NC}"
    echo "   Revisa los logs: tail -f /tmp/rag_api.log"
fi

echo ""
echo "================================================================================"
echo "RESUMEN"
echo "================================================================================"
echo ""

# Estado final
echo "Estado de servicios:"
check_http_service "http://localhost:8000/health" "RAG API (puerto 8000)"

if check_port 5432; then
    echo -e "${GREEN}✅ PostgreSQL (puerto 5432): ACTIVO${NC}"
else
    echo -e "${YELLOW}⚠️  PostgreSQL (puerto 5432): INACTIVO${NC}"
fi

if check_port 6379; then
    echo -e "${GREEN}✅ Redis (puerto 6379): ACTIVO${NC}"
else
    echo -e "${YELLOW}⚠️  Redis (puerto 6379): INACTIVO${NC}"
fi

echo ""
echo "Comandos útiles:"
echo "  - Ver logs: tail -f /tmp/rag_api.log"
echo "  - Ver sesión screen: screen -r rag-api"
echo "  - Verificar salud: curl http://localhost:8000/health"
echo "  - Ver puertos: sudo netstat -tuln | grep LISTEN"
echo ""

