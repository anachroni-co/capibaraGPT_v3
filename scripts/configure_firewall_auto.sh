#!/bin/bash
# Script automático para configurar firewall (sin interacción)
# Usa este script si quieres configurar todo automáticamente

set -e

PROJECT="mamba-001"

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "================================================================================"
echo "CONFIGURACION AUTOMATICA DE FIREWALL - CAPIBARA6"
echo "================================================================================"
echo ""

# Obtener IP pública del usuario
echo "Obteniendo tu IP pública..."
USER_IP=$(curl -s https://api.ipify.org 2>/dev/null || echo "")
if [ -z "$USER_IP" ]; then
    echo -e "${YELLOW}No se pudo obtener tu IP pública automáticamente${NC}"
    echo -e "${YELLOW}Se configurará solo comunicación interna${NC}"
fi

# Obtener red VPC (usar default o la primera disponible)
NETWORK=$(gcloud compute networks list --project=$PROJECT --format="value(name)" 2>/dev/null | head -1 || echo "default")

echo -e "${BLUE}Red VPC: $NETWORK${NC}"
if [ -n "$USER_IP" ]; then
    echo -e "${BLUE}Tu IP pública: $USER_IP${NC}"
fi
echo ""

# Puertos
PORTS="11434,5000,5001,5002,5003,5010,5678,8000"

# Eliminar reglas existentes si existen
echo "Verificando reglas existentes..."
EXISTING_RULES=$(gcloud compute firewall-rules list --project=$PROJECT --filter="name~allow-capibara6" --format="value(name)" 2>/dev/null || echo "")

if [ -n "$EXISTING_RULES" ]; then
    echo -e "${YELLOW}Eliminando reglas existentes...${NC}"
    for rule in $EXISTING_RULES; do
        gcloud compute firewall-rules delete "$rule" --project=$PROJECT --quiet 2>/dev/null || true
    done
    echo -e "${GREEN}Reglas eliminadas${NC}"
fi

# Crear regla interna
echo ""
echo "Creando regla para comunicación interna..."
RULE_NAME_INTERNAL="allow-capibara6-internal"

gcloud compute firewall-rules create "$RULE_NAME_INTERNAL" \
    --project=$PROJECT \
    --network=$NETWORK \
    --allow=tcp:$PORTS \
    --source-ranges=10.0.0.0/8 \
    --description="Permitir comunicación interna entre VMs de Capibara6" \
    --direction=INGRESS \
    --priority=1000 \
    --quiet 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Regla interna creada: $RULE_NAME_INTERNAL${NC}"
else
    echo -e "${RED}❌ Error creando regla interna${NC}"
fi

# Crear regla externa si hay IP
if [ -n "$USER_IP" ]; then
    echo ""
    echo "Creando regla para acceso externo..."
    RULE_NAME_EXTERNAL="allow-capibara6-external-dev"
    
    gcloud compute firewall-rules create "$RULE_NAME_EXTERNAL" \
        --project=$PROJECT \
        --network=$NETWORK \
        --allow=tcp:$PORTS \
        --source-ranges=$USER_IP/32 \
        --description="Permitir acceso a servicios Capibara6 desde desarrollo local (IP: $USER_IP)" \
        --direction=INGRESS \
        --priority=1000 \
        --quiet 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Regla externa creada: $RULE_NAME_EXTERNAL${NC}"
    else
        echo -e "${RED}❌ Error creando regla externa${NC}"
    fi
fi

echo ""
echo "================================================================================"
echo "RESUMEN"
echo "================================================================================"
echo ""
echo "Reglas creadas:"
gcloud compute firewall-rules list \
    --project=$PROJECT \
    --filter="name~allow-capibara6" \
    --format="table(name,network,direction,sourceRanges.list():label=SOURCE_RANGES,allowed[].map().firewall_rule().list():label=ALLOW)" \
    2>/dev/null || echo "No se pudieron listar las reglas"

echo ""
echo -e "${GREEN}✅ Configuración completada${NC}"
echo ""
echo "Próximos pasos:"
echo "  1. Verifica que los servicios están corriendo en cada VM"
echo "  2. Prueba las conexiones: bash scripts/verify_all_services.sh"
echo ""

