#!/bin/bash
# Script para verificar el estado del backend de Capibara6
# Ejecutar en la VM: gcloud compute ssh --zone "europe-southwest1-b" "gpt-oss-20b" --project "mamba-001"

echo "🔍 Verificando Estado del Backend Capibara6"
echo "=========================================="

echo "📊 Información del sistema:"
echo "Usuario: $(whoami)"
echo "Fecha: $(date)"
echo "Uptime: $(uptime)"

echo ""
echo "💾 Espacio en disco:"
df -h | grep -E "(Filesystem|/dev/)"

echo ""
echo "🔄 Procesos Python activos:"
ps aux | grep -E "(python|flask)" | grep -v grep || echo "No hay procesos Python activos"

echo ""
echo "🌐 Puertos en uso:"
echo "Puerto 5001 (Servidor integrado):"
lsof -i :5001 2>/dev/null || netstat -tuln | grep :5001 || echo "❌ Puerto 5001 no está en uso"

echo ""
echo "Puerto 5002 (TTS Server):"
lsof -i :5002 2>/dev/null || netstat -tuln | grep :5002 || echo "❌ Puerto 5002 no está en uso"

echo ""
echo "Puerto 5003 (MCP Server):"
lsof -i :5003 2>/dev/null || netstat -tuln | grep :5003 || echo "❌ Puerto 5003 no está en uso"

echo ""
echo "Puerto 8080 (GPT-OSS-20B llama-server):"
lsof -i :8080 2>/dev/null || netstat -tuln | grep :8080 || echo "❌ Puerto 8080 no está en uso"

echo ""
echo "🧠 Verificando modelo GPT-OSS-20B:"
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Modelo GPT-OSS-20B responde en puerto 8080"
    curl -s http://localhost:8080/health | head -5
else
    echo "❌ Modelo GPT-OSS-20B no responde en puerto 8080"
fi

echo ""
echo "🔌 Verificando servidor integrado (puerto 5001):"
if curl -s http://localhost:5001/health > /dev/null 2>&1; then
    echo "✅ Servidor integrado responde en puerto 5001"
    curl -s http://localhost:5001/health | head -5
else
    echo "❌ Servidor integrado no responde en puerto 5001"
fi

echo ""
echo "🎤 Verificando servidor TTS (puerto 5002):"
if curl -s http://localhost:5002/health > /dev/null 2>&1; then
    echo "✅ Servidor TTS responde en puerto 5002"
    curl -s http://localhost:5002/health | head -5
else
    echo "❌ Servidor TTS no responde en puerto 5002"
fi

echo ""
echo "🧠 Verificando servidor MCP (puerto 5003):"
if curl -s http://localhost:5003/health > /dev/null 2>&1; then
    echo "✅ Servidor MCP responde en puerto 5003"
    curl -s http://localhost:5003/health | head -5
else
    echo "❌ Servidor MCP no responde en puerto 5003"
fi

echo ""
echo "📁 Verificando archivos del backend:"
if [ -f "/home/$(whoami)/capibara6/backend/capibara6_integrated_server.py" ]; then
    echo "✅ Servidor integrado encontrado"
    BACKEND_DIR="/home/$(whoami)/capibara6/backend"
elif [ -f "$(pwd)/backend/capibara6_integrated_server.py" ]; then
    echo "✅ Servidor integrado encontrado en $(pwd)/backend"
    BACKEND_DIR="$(pwd)/backend"
else
    echo "⚠️ Buscando servidor integrado..."
    BACKEND_DIR=$(find ~ -name "capibara6_integrated_server.py" 2>/dev/null | head -1 | xargs dirname)
    if [ -n "$BACKEND_DIR" ]; then
        echo "✅ Servidor integrado encontrado en $BACKEND_DIR"
    else
        echo "❌ Servidor integrado no encontrado"
    fi
fi

echo ""
echo "📋 Logs recientes del servidor:"
if [ -d "$BACKEND_DIR/logs" ]; then
    echo "Últimas líneas de capibara6.log:"
    tail -20 "$BACKEND_DIR/logs/capibara6.log" 2>/dev/null || echo "No hay logs disponibles"
    echo ""
    echo "Errores recientes:"
    tail -10 "$BACKEND_DIR/logs/errors.log" 2>/dev/null || echo "No hay errores recientes"
else
    echo "⚠️ Directorio de logs no encontrado"
fi

echo ""
echo "🔧 Verificando servicios systemd:"
if systemctl list-units | grep -E "(llama|capibara|gpt)" > /dev/null 2>&1; then
    echo "Servicios systemd relacionados:"
    systemctl list-units | grep -E "(llama|capibara|gpt)"
else
    echo "No hay servicios systemd relacionados"
fi

echo ""
echo "📊 Verificando procesos en tmux o screen:"
if command -v tmux > /dev/null 2>&1; then
    tmux list-sessions 2>/dev/null || echo "No hay sesiones tmux activas"
fi
if command -v screen > /dev/null 2>&1; then
    screen -list 2>/dev/null || echo "No hay sesiones screen activas"
fi

echo ""
echo "🎯 Resumen del estado:"
echo "======================"
if lsof -i :8080 > /dev/null 2>&1; then
    echo "✅ Modelo GPT-OSS-20B: ACTIVO"
else
    echo "❌ Modelo GPT-OSS-20B: INACTIVO"
fi

if lsof -i :5001 > /dev/null 2>&1; then
    echo "✅ Servidor Integrado: ACTIVO"
else
    echo "❌ Servidor Integrado: INACTIVO"
fi

if lsof -i :5002 > /dev/null 2>&1; then
    echo "✅ Servidor TTS: ACTIVO"
else
    echo "❌ Servidor TTS: INACTIVO"
fi

if lsof -i :5003 > /dev/null 2>&1; then
    echo "✅ Servidor MCP: ACTIVO"
else
    echo "❌ Servidor MCP: INACTIVO"
fi
