#!/bin/bash
# Script para verificar el estado de las VMs de Capibara6

echo "🦫 =========================================="
echo "   CAPIBARA6 - Verificación de VMs"
echo "=========================================="
echo ""

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar estado de VMs
echo "📊 Estado de las VMs:"
echo ""
gcloud compute instances list --project "mamba-001" --filter="name:(bounty2 OR gpt-oss-20b)"

echo ""
echo "=========================================="
echo "🔍 Probando conectividad..."
echo "=========================================="
echo ""

# Probar VM de Modelos (bounty2)
echo -n "VM de Modelos (bounty2 - 34.12.166.76:8080): "
if curl -s --connect-timeout 5 http://34.12.166.76:8080/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ACCESIBLE${NC}"
else
    echo -e "${RED}✗ NO ACCESIBLE${NC}"
    echo "  → Puede que esté apagada o el puerto 8080 bloqueado"
fi

echo -n "Backend API (34.12.166.76:5001): "
if curl -s --connect-timeout 5 http://34.12.166.76:5001/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ACCESIBLE${NC}"
else
    echo -e "${RED}✗ NO ACCESIBLE${NC}"
    echo "  → Puede que el servicio no esté corriendo"
fi

echo ""

# Probar VM de Servicios (gpt-oss-20b)
echo -n "VM de Servicios (gpt-oss-20b - 34.175.136.104:5002): "
if curl -s --connect-timeout 5 http://34.175.136.104:5002/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ACCESIBLE${NC}"
else
    echo -e "${RED}✗ NO ACCESIBLE${NC}"
    echo "  → Verifica que TTS esté corriendo"
fi

echo -n "MCP Service (34.175.136.104:5003): "
if curl -s --connect-timeout 5 http://34.175.136.104:5003/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ACCESIBLE${NC}"
else
    echo -e "${RED}✗ NO ACCESIBLE${NC}"
    echo "  → Verifica que MCP esté corriendo"
fi

echo -n "N8N (34.175.136.104:5678): "
if curl -s --connect-timeout 5 http://34.175.136.104:5678 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ACCESIBLE${NC}"
else
    echo -e "${RED}✗ NO ACCESIBLE${NC}"
    echo "  → Verifica que N8N esté corriendo"
fi

echo ""
echo "=========================================="
echo "📝 Comandos útiles:"
echo "=========================================="
echo ""
echo "Conectarse a VM de modelos:"
echo "  ./ssh-bounty2.sh"
echo ""
echo "Conectarse a VM de servicios:"
echo "  ./ssh-services.sh"
echo ""
echo "Verificar puertos en firewall:"
echo "  gcloud compute firewall-rules list --project mamba-001"
echo ""
