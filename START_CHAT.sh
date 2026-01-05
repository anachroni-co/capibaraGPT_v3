#!/bin/bash
# Script de inicio para Capibara6 Chat
# Configurado para conectarse a VMs de Google Cloud

echo "🦫 =========================================="
echo "🦫   CAPIBARA6 CHAT - INICIO RÁPIDO"
echo "🦫 =========================================="
echo ""

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Verificar que estamos en el directorio correcto
if [ ! -d "backend" ] || [ ! -d "web" ]; then
    echo -e "${RED}❌ Error: Este script debe ejecutarse desde el directorio raíz de capibara6${NC}"
    exit 1
fi

echo "📊 Configuración actual:"
echo "  - VM de modelos (bounty): 34.12.166.76"
echo "  - VM de servicios: 34.175.136.104"
echo "  - Puerto backend local: 5001"
echo ""

# Verificar si existe .env
if [ ! -f "backend/.env" ]; then
    echo -e "${YELLOW}⚠️  Creando archivo .env...${NC}"
    cat > backend/.env << 'EOF'
# Configuración del backend capibara6
PORT=5001

# VM de MODELOS (bounty) - 34.12.166.76
GPT_OSS_URL=http://34.12.166.76:8080
GPT_OSS_TIMEOUT=60

# VM de SERVICIOS - 34.175.136.104
SERVICES_VM_URL=http://34.175.136.104
TTS_URL=http://34.175.136.104:5002
MCP_URL=http://34.175.136.104:5003
N8N_URL=http://34.175.136.104:5678

# SMTP (si es necesario)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=marco@anachroni.co
FROM_EMAIL=marco@anachroni.co
EOF
    echo -e "${GREEN}✅ Archivo .env creado${NC}"
else
    echo -e "${GREEN}✅ Archivo .env existe${NC}"
fi

echo ""
echo "🚀 Iniciando backend..."
echo ""

# Ir al directorio backend
cd backend

# Verificar que existe server_gptoss.py
if [ ! -f "server_gptoss.py" ]; then
    echo -e "${RED}❌ Error: No se encuentra server_gptoss.py${NC}"
    exit 1
fi

# Iniciar el servidor
echo -e "${GREEN}▶️  Ejecutando server_gptoss.py en puerto 5001...${NC}"
echo ""
echo "================================================"
echo "  Conexiones configuradas:"
echo "  📡 Modelos GPT-OSS: http://34.12.166.76:8080"
echo "  🔧 Servicios: http://34.175.136.104"
echo "  💻 Backend local: http://localhost:5001"
echo "================================================"
echo ""
echo "Para acceder al chat:"
echo "  1. Abre otra terminal"
echo "  2. cd ~/capibara6/web"
echo "  3. python3 -m http.server 8000"
echo "  4. Abre: http://localhost:8000/chat.html"
echo ""
echo "Presiona Ctrl+C para detener el servidor"
echo ""

# Ejecutar el servidor
python3 server_gptoss.py
