#!/bin/bash
# Script para verificar servicios corriendo en una VM
# Ejecutar este script dentro de cada VM para verificar qué servicios están activos

echo "🔍 Verificando servicios en esta VM..."
echo "========================================"
echo ""

# Información del sistema
echo "📊 Información del Sistema:"
echo "  Hostname: $(hostname)"
echo "  IP Interna: $(hostname -I | awk '{print $1}')"
echo "  Zona: $(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null | awk -F'/' '{print $NF}' || echo 'N/A')"
echo ""

# Verificar puertos en escucha
echo "🔌 Puertos en escucha:"
echo "----------------------"
netstat -tuln 2>/dev/null | grep LISTEN | awk '{printf "  Puerto %-6s %s\n", $4, $7}' | sort -u || \
ss -tuln 2>/dev/null | grep LISTEN | awk '{printf "  Puerto %-6s\n", $5}' | sort -u || \
echo "  ⚠️  No se pudo obtener información de puertos"
echo ""

# Verificar procesos específicos
echo "🔄 Procesos activos:"
echo "-------------------"

# Ollama
if pgrep -x ollama > /dev/null; then
    OLLAMA_PID=$(pgrep -x ollama | head -1)
    echo "  ✅ Ollama corriendo (PID: $OLLAMA_PID)"
    OLLAMA_PORT=$(sudo lsof -i -P -n | grep ollama | grep LISTEN | awk '{print $9}' | cut -d: -f2 | head -1)
    if [ -n "$OLLAMA_PORT" ]; then
        echo "     Puerto: $OLLAMA_PORT"
    fi
else
    echo "  ❌ Ollama no está corriendo"
fi

# Python servers
PYTHON_PROCS=$(pgrep -f "python.*server" | wc -l)
if [ "$PYTHON_PROCS" -gt 0 ]; then
    echo "  ✅ Servidores Python corriendo ($PYTHON_PROCS procesos)"
    pgrep -f "python.*server" | while read pid; do
        CMD=$(ps -p $pid -o cmd= | head -1)
        PORT=$(sudo lsof -i -P -n | grep "python.*$pid" | grep LISTEN | awk '{print $9}' | cut -d: -f2 | head -1)
        echo "     PID $pid: $CMD"
        if [ -n "$PORT" ]; then
            echo "       Puerto: $PORT"
        fi
    done
else
    echo "  ⚠️  No se encontraron servidores Python activos"
fi

# Node servers
if pgrep -f "node.*server" > /dev/null; then
    echo "  ✅ Servidor Node corriendo"
    pgrep -f "node.*server" | while read pid; do
        CMD=$(ps -p $pid -o cmd= | head -1)
        PORT=$(sudo lsof -i -P -n | grep "node.*$pid" | grep LISTEN | awk '{print $9}' | cut -d: -f2 | head -1)
        echo "     PID $pid: $CMD"
        if [ -n "$PORT" ]; then
            echo "       Puerto: $PORT"
        fi
    done
else
    echo "  ⚠️  No se encontraron servidores Node activos"
fi

# N8n
if pgrep -f n8n > /dev/null; then
    echo "  ✅ N8n corriendo"
    N8N_PORT=$(sudo lsof -i -P -n | grep n8n | grep LISTEN | awk '{print $9}' | cut -d: -f2 | head -1)
    if [ -n "$N8N_PORT" ]; then
        echo "     Puerto: $N8N_PORT"
    fi
else
    echo "  ❌ N8n no está corriendo"
fi

# Verificar servicios específicos por puerto
echo ""
echo "🌐 Verificación de servicios por puerto:"
echo "----------------------------------------"

check_port() {
    local port=$1
    local service=$2
    
    if sudo lsof -i :$port > /dev/null 2>&1 || netstat -tuln 2>/dev/null | grep -q ":$port " || ss -tuln 2>/dev/null | grep -q ":$port "; then
        echo "  ✅ Puerto $port ($service): ACTIVO"
        PROCESS=$(sudo lsof -i :$port 2>/dev/null | tail -1 | awk '{print $1, $2}' || echo "N/A")
        echo "     Proceso: $PROCESS"
    else
        echo "  ❌ Puerto $port ($service): INACTIVO"
    fi
}

# Puertos comunes según la VM
VM_NAME=$(hostname | tr '[:upper:]' '[:lower:]')

if [[ "$VM_NAME" == *"bounty"* ]] || [[ "$VM_NAME" == *"bounty2"* ]]; then
    echo "  📍 Detectada VM: Bounty2 (Ollama)"
    check_port 11434 "Ollama"
    check_port 5001 "Backend Flask"
    check_port 5000 "Backend Flask (alternativo)"
elif [[ "$VM_NAME" == *"rag"* ]] || [[ "$VM_NAME" == *"rag3"* ]]; then
    echo "  📍 Detectada VM: rag3 (RAG Database)"
    check_port 8000 "RAG API"
    check_port 5432 "PostgreSQL"
    check_port 6379 "Redis"
elif [[ "$VM_NAME" == *"gpt-oss"* ]] || [[ "$VM_NAME" == *"gptoss"* ]]; then
    echo "  📍 Detectada VM: gpt-oss-20b (Servicios)"
    check_port 5000 "Bridge"
    check_port 5002 "TTS"
    check_port 5003 "MCP"
    check_port 5010 "MCP (alternativo)"
    check_port 5678 "N8n"
fi

echo ""
echo "✅ Verificación completada"
echo ""
echo "💡 Para ver logs de un servicio específico:"
echo "   - Ollama: journalctl -u ollama -f"
echo "   - Python: tail -f /path/to/logs/*.log"
echo "   - N8n: docker logs n8n (si está en Docker)"

