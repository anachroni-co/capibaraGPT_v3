#!/bin/bash
# Script para iniciar servicios en las VMs
# Ejecutar desde cada VM después de conectarse via SSH

set -e

echo "🚀 Iniciando Servicios en VMs"
echo "=============================="
echo ""

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Detectar en qué VM estamos
VM_NAME=$(hostname)
echo -e "${BLUE}VM detectada: $VM_NAME${NC}"
echo ""

case $VM_NAME in
    *models-europe*)
        echo -e "${YELLOW}Iniciando servicios en models-europe...${NC}"
        echo ""

        cd ~/capibara6/backend 2>/dev/null || cd /ruta/a/tu/proyecto/backend

        # Verificar si ya está corriendo
        if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo "⚠️  Puerto 5001 ya en uso"
            echo "Deteniendo proceso existente..."
            lsof -ti:5001 | xargs kill -9 2>/dev/null || true
            sleep 2
        fi

        # Iniciar backend
        echo "🚀 Iniciando Backend en puerto 5001..."
        screen -dmS backend bash -c "cd $(pwd) && python3 capibara6_integrated_server.py || python3 server.py"

        sleep 3

        # Verificar
        if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Backend iniciado correctamente${NC}"
        else
            echo -e "${YELLOW}⚠️  Backend iniciado pero no responde aún${NC}"
        fi

        echo ""
        echo "📋 Screens activos:"
        screen -ls
        ;;

    *rag-europe*)
        echo -e "${YELLOW}Iniciando servicios en rag-europe...${NC}"
        echo ""

        cd ~/capibara6/backend 2>/dev/null || cd /ruta/a/tu/proyecto/backend

        # Iniciar RAG API según tu configuración
        echo "🚀 Iniciando RAG API..."
        # Ajustar según tu configuración específica
        screen -dmS rag python3 api_server.py || echo "⚠️  Ajusta el comando según tu configuración"

        echo ""
        echo "📋 Screens activos:"
        screen -ls
        ;;

    *services*)
        echo -e "${YELLOW}Iniciando servicios en services...${NC}"
        echo ""

        cd ~/capibara6 2>/dev/null || cd /ruta/a/tu/proyecto

        # Usar script de inicio si existe
        if [ -f "check_and_start_gpt_oss_20b.sh" ]; then
            chmod +x check_and_start_gpt_oss_20b.sh
            ./check_and_start_gpt_oss_20b.sh
        else
            echo "⚠️  Script no encontrado, iniciando manualmente..."
            cd backend

            # Bridge (5000)
            if ! lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1; then
                screen -dmS bridge python3 server_gptoss.py || python3 server.py
                echo "✅ Bridge iniciado"
            fi

            # MCP (5003)
            if ! lsof -Pi :5003 -sTCP:LISTEN -t >/dev/null 2>&1; then
                screen -dmS mcp python3 smart_mcp_server.py
                echo "✅ MCP iniciado"
            fi

            # MCP Alt (5010) - modificar puerto
            if ! lsof -Pi :5010 -sTCP:LISTEN -t >/dev/null 2>&1; then
                screen -dmS mcp-alt bash -c "python3 -c \"
import sys
sys.path.insert(0, '.')
from smart_mcp_server import app
app.run(host='0.0.0.0', port=5010, debug=False)
\""
                echo "✅ MCP Alt iniciado"
            fi

            # TTS (5002)
            if ! lsof -Pi :5002 -sTCP:LISTEN -t >/dev/null 2>&1; then
                screen -dmS tts python3 kyutai_tts_server.py || python3 coqui_tts_server.py
                echo "✅ TTS iniciado"
            fi
        fi

        echo ""
        echo "📋 Screens activos:"
        screen -ls
        ;;

    *)
        echo "⚠️  VM no reconocida: $VM_NAME"
        echo "Ejecuta los comandos manualmente según tu VM"
        ;;
esac

echo ""
echo -e "${GREEN}✅ Script completado${NC}"

