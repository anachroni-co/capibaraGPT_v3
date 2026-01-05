#!/bin/bash
# Capibara6 - Script para verificar el estado de todos los servicios

set -e

echo "🔍 Capibara6 - Verificando Estado de Servicios..."
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directorio del script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Función para verificar puerto
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Puerto en uso
    else
        return 1  # Puerto libre
    fi
}

# Función para verificar health endpoint
check_health() {
    local url=$1
    local timeout=${2:-3}

    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time $timeout "$url" 2>/dev/null || echo "000")
    if [ "$response" = "200" ]; then
        return 0
    else
        return 1
    fi
}

# Función para verificar un servicio completo
check_service() {
    local name=$1
    local port=$2
    local health_url=$3
    local required=$4

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}🔍 $name (Puerto $port)${NC}"

    # Verificar puerto
    if check_port $port; then
        echo -e "${GREEN}  ✓ Puerto $port: ACTIVO${NC}"

        # Verificar health endpoint si está definido
        if [ -n "$health_url" ]; then
            echo -e "${YELLOW}  → Probando health endpoint: $health_url${NC}"

            if check_health "$health_url"; then
                echo -e "${GREEN}  ✓ Health check: OK${NC}"
                echo -e "${GREEN}  ✅ $name: FUNCIONANDO CORRECTAMENTE${NC}"
            else
                echo -e "${RED}  ✗ Health check: FALLÓ${NC}"
                echo -e "${YELLOW}  ⚠️  $name: Puerto activo pero health check falló${NC}"
            fi
        else
            echo -e "${GREEN}  ✅ $name: ACTIVO (sin health check)${NC}"
        fi
    else
        if [ "$required" = "required" ]; then
            echo -e "${RED}  ✗ Puerto $port: NO ESTÁ ESCUCHANDO${NC}"
            echo -e "${RED}  ❌ $name: NO DISPONIBLE (REQUERIDO)${NC}"
        else
            echo -e "${YELLOW}  ✗ Puerto $port: NO ESTÁ ESCUCHANDO${NC}"
            echo -e "${YELLOW}  ℹ️  $name: NO DISPONIBLE (Opcional)${NC}"
        fi
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SERVICIOS PRINCIPALES (REQUERIDOS)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Servicios principales
check_service "Backend Principal (server_gptoss.py)" 5001 "http://localhost:5001/api/health" "required"
check_service "TTS Server (kyutai_tts_server.py)" 5002 "http://localhost:5002/health" "required"
check_service "Auth Server (auth_server.py)" 5004 "http://localhost:5004/health" "required"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SERVICIOS OPCIONALES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Servicios opcionales
check_service "MCP Server (mcp_server.py)" 5003 "http://localhost:5003/api/mcp/health" "optional"
check_service "Consensus Server (consensus_server.py)" 5005 "http://localhost:5005/api/consensus/health" "optional"
check_service "Smart MCP Server (smart_mcp_server.py)" 5010 "http://localhost:5010/health" "optional"
check_service "FastAPI Server (main.py)" 8000 "http://localhost:8000/health" "optional"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BASES DE DATOS (Docker)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Bases de datos Docker
check_service "PostgreSQL" 5432 "" "optional"
check_service "TimescaleDB" 5433 "" "optional"
check_service "Redis" 6379 "" "optional"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SERVICIOS REMOTOS (VMs)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Servicios en VMs (solo verificar si son accesibles)
echo -e "${YELLOW}🔍 TTS Server VM (34.175.136.104:5002)${NC}"
if check_health "http://34.175.136.104:5002/health" 5; then
    echo -e "${GREEN}  ✅ TTS VM: ACCESIBLE${NC}"
else
    echo -e "${YELLOW}  ℹ️  TTS VM: NO ACCESIBLE (puede estar en otra red)${NC}"
fi

echo ""
echo -e "${YELLOW}🔍 MCP Server VM (34.175.136.104:5003)${NC}"
if check_health "http://34.175.136.104:5003/api/mcp/health" 5; then
    echo -e "${GREEN}  ✅ MCP VM: ACCESIBLE${NC}"
else
    echo -e "${YELLOW}  ℹ️  MCP VM: NO ACCESIBLE (puede estar en otra red)${NC}"
fi

echo ""
echo -e "${YELLOW}🔍 N8N (34.175.136.104:5678) - Requiere VPN${NC}"
if check_health "http://34.175.136.104:5678/healthz" 3; then
    echo -e "${GREEN}  ✅ N8N: ACCESIBLE${NC}"
else
    echo -e "${YELLOW}  ℹ️  N8N: NO ACCESIBLE (esperado sin VPN)${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  VM rag3 - SISTEMA RAG COMPLETO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Nota: Reemplaza 'rag3' con la IP real de la VM rag3 si está disponible
# Por ahora usamos el hostname interno
RAG3_HOST="rag3"

echo -e "${YELLOW}🔍 Bridge API (capibara6-api) - Puerto 8000${NC}"
if check_health "http://$RAG3_HOST:8000/health" 5; then
    echo -e "${GREEN}  ✅ capibara6-api: ACCESIBLE${NC}"
else
    echo -e "${YELLOW}  ℹ️  capibara6-api: NO ACCESIBLE (puede requerir acceso interno)${NC}"
fi

echo ""
echo -e "${YELLOW}🔍 Milvus Vector Database - Puerto 19530${NC}"
echo -e "${YELLOW}  → Probando conexión TCP...${NC}"
if timeout 3 bash -c "cat < /dev/null > /dev/tcp/$RAG3_HOST/19530" 2>/dev/null; then
    echo -e "${GREEN}  ✅ Milvus: PUERTO ACCESIBLE${NC}"
else
    echo -e "${YELLOW}  ℹ️  Milvus: NO ACCESIBLE (puede requerir acceso interno)${NC}"
fi

echo ""
echo -e "${YELLOW}🔍 Nebula Graph Query Service - Puerto 9669${NC}"
echo -e "${YELLOW}  → Probando conexión TCP...${NC}"
if timeout 3 bash -c "cat < /dev/null > /dev/tcp/$RAG3_HOST/9669" 2>/dev/null; then
    echo -e "${GREEN}  ✅ Nebula Graph: PUERTO ACCESIBLE${NC}"
else
    echo -e "${YELLOW}  ℹ️  Nebula Graph: NO ACCESIBLE (puede requerir acceso interno)${NC}"
fi

echo ""
echo -e "${YELLOW}🔍 Nebula Graph Studio (UI) - Puerto 7001${NC}"
if check_health "http://$RAG3_HOST:7001" 3; then
    echo -e "${GREEN}  ✅ Nebula Studio: ACCESIBLE${NC}"
    echo -e "${GREEN}  → UI disponible en: http://$RAG3_HOST:7001${NC}"
else
    echo -e "${YELLOW}  ℹ️  Nebula Studio: NO ACCESIBLE${NC}"
fi

echo ""
echo -e "${YELLOW}🔍 PostgreSQL - Puerto 5432${NC}"
if timeout 2 bash -c "cat < /dev/null > /dev/tcp/$RAG3_HOST/5432" 2>/dev/null; then
    echo -e "${GREEN}  ✅ PostgreSQL: PUERTO ACCESIBLE${NC}"
else
    echo -e "${YELLOW}  ℹ️  PostgreSQL: NO ACCESIBLE${NC}"
fi

echo ""
echo -e "${YELLOW}🔍 Redis - Puerto 6379${NC}"
if timeout 2 bash -c "cat < /dev/null > /dev/tcp/$RAG3_HOST/6379" 2>/dev/null; then
    echo -e "${GREEN}  ✅ Redis: PUERTO ACCESIBLE${NC}"
else
    echo -e "${YELLOW}  ℹ️  Redis: NO ACCESIBLE${NC}"
fi

echo ""
echo -e "${YELLOW}🔍 Monitoring Stack${NC}"
echo -e "${YELLOW}  - Grafana (3000)${NC}"
if check_health "http://$RAG3_HOST:3000" 3; then
    echo -e "${GREEN}    ✅ Grafana: ACCESIBLE → http://$RAG3_HOST:3000${NC}"
else
    echo -e "${YELLOW}    ℹ️  Grafana: NO ACCESIBLE${NC}"
fi

echo -e "${YELLOW}  - Prometheus (9090)${NC}"
if check_health "http://$RAG3_HOST:9090" 3; then
    echo -e "${GREEN}    ✅ Prometheus: ACCESIBLE → http://$RAG3_HOST:9090${NC}"
else
    echo -e "${YELLOW}    ℹ️  Prometheus: NO ACCESIBLE${NC}"
fi

echo -e "${YELLOW}  - Jaeger (16686)${NC}"
if check_health "http://$RAG3_HOST:16686" 3; then
    echo -e "${GREEN}    ✅ Jaeger: ACCESIBLE → http://$RAG3_HOST:16686${NC}"
else
    echo -e "${YELLOW}    ℹ️  Jaeger: NO ACCESIBLE${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Resumen final
echo -e "${BLUE}📊 RESUMEN${NC}"
echo ""

required_count=0
required_ok=0

# Contar servicios requeridos activos
for port in 5001 5002 5004; do
    required_count=$((required_count + 1))
    if check_port $port; then
        required_ok=$((required_ok + 1))
    fi
done

if [ $required_ok -eq $required_count ]; then
    echo -e "${GREEN}✅ Todos los servicios requeridos están activos ($required_ok/$required_count)${NC}"
    echo -e "${GREEN}✅ El sistema está OPERATIVO${NC}"
else
    echo -e "${RED}⚠️  Servicios requeridos activos: $required_ok/$required_count${NC}"
    echo -e "${RED}❌ El sistema NO está completamente operativo${NC}"
    echo ""
    echo "Para iniciar servicios: ./start-all-services.sh"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Mostrar procesos Python relacionados
echo ""
echo -e "${BLUE}🔍 Procesos Python activos:${NC}"
ps aux | grep -E "server_gptoss|kyutai_tts|mcp_server|auth_server|consensus_server|smart_mcp|main.py" | grep -v grep | awk '{printf "  PID: %-6s CPU: %-5s MEM: %-5s CMD: %s\n", $2, $3"%", $4"%", $11}' || echo "  (ninguno)"
echo ""
