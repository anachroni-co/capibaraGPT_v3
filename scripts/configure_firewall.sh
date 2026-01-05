#!/bin/bash
# Script para configurar firewall rules en GCloud para Capibara6
# Permite comunicación entre VMs y acceso desde desarrollo local

set -e

PROJECT="mamba-001"

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "================================================================================"
echo "CONFIGURACION DE FIREWALL - CAPIBARA6"
echo "================================================================================"
echo ""

# Obtener IP pública del usuario (para acceso desde desarrollo local)
echo "Obteniendo tu IP pública..."
USER_IP=$(curl -s https://api.ipify.org 2>/dev/null || echo "")
if [ -z "$USER_IP" ]; then
    echo -e "${YELLOW}No se pudo obtener tu IP pública automáticamente${NC}"
    read -p "Ingresa tu IP pública (o presiona Enter para omitir acceso externo): " USER_IP
fi

if [ -n "$USER_IP" ]; then
    echo -e "${GREEN}IP pública detectada: $USER_IP${NC}"
else
    echo -e "${YELLOW}No se configurará acceso externo${NC}"
fi

echo ""

# Obtener información de redes VPC
echo "Obteniendo información de redes VPC..."
NETWORKS=$(gcloud compute networks list --project=$PROJECT --format="value(name)" 2>/dev/null || echo "default")

if [ -z "$NETWORKS" ]; then
    NETWORK="default"
else
    # Usar la primera red disponible o preguntar
    NETWORK=$(echo "$NETWORKS" | head -1)
    echo -e "${BLUE}Red VPC detectada: $NETWORK${NC}"
fi

echo ""

# Puertos que necesitan estar abiertos
PORTS_INTERNAL="11434,5000,5001,5002,5003,5010,5678,8000"
PORTS_EXTERNAL="11434,5000,5001,5002,5003,5010,5678,8000"

echo "Puertos a configurar:"
echo "  - Ollama: 11434"
echo "  - Backend Flask: 5001"
echo "  - Bridge: 5000"
echo "  - TTS: 5002"
echo "  - MCP: 5003"
echo "  - MCP Alt: 5010"
echo "  - N8n: 5678"
echo "  - RAG API: 8000"
echo ""

# Verificar reglas existentes
echo "Verificando reglas de firewall existentes..."
EXISTING_RULE=$(gcloud compute firewall-rules list --project=$PROJECT --filter="name~allow-capibara6" --format="value(name)" 2>/dev/null || echo "")

if [ -n "$EXISTING_RULE" ]; then
    echo -e "${YELLOW}Se encontraron reglas existentes:${NC}"
    echo "$EXISTING_RULE"
    echo ""
    read -p "¿Deseas eliminar las reglas existentes y crear nuevas? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Eliminando reglas existentes..."
        for rule in $EXISTING_RULE; do
            gcloud compute firewall-rules delete "$rule" --project=$PROJECT --quiet 2>/dev/null || true
        done
        echo -e "${GREEN}Reglas eliminadas${NC}"
    fi
fi

echo ""

# Crear regla para comunicación interna entre VMs
echo "================================================================================"
echo "1. Creando regla para comunicación INTERNA entre VMs"
echo "================================================================================"

RULE_NAME_INTERNAL="allow-capibara6-internal"
echo "Nombre de regla: $RULE_NAME_INTERNAL"
echo "Red: $NETWORK"
echo "Puertos: $PORTS_INTERNAL"
echo "Origen: 10.0.0.0/8 (red interna de GCloud)"
echo ""

read -p "¿Crear esta regla? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creando regla de firewall interna..."
    
    gcloud compute firewall-rules create "$RULE_NAME_INTERNAL" \
        --project=$PROJECT \
        --network=$NETWORK \
        --allow=tcp:$PORTS_INTERNAL \
        --source-ranges=10.0.0.0/8 \
        --description="Permitir comunicación interna entre VMs de Capibara6" \
        --direction=INGRESS \
        --priority=1000 \
        2>&1 | while read line; do
            if [[ $line == *"Created"* ]] || [[ $line == *"already exists"* ]]; then
                echo -e "${GREEN}$line${NC}"
            else
                echo "$line"
            fi
        done
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✅ Regla interna creada exitosamente${NC}"
    else
        echo -e "${RED}❌ Error creando regla interna${NC}"
    fi
else
    echo -e "${YELLOW}Omitiendo creación de regla interna${NC}"
fi

echo ""

# Crear regla para acceso externo (si se proporcionó IP)
if [ -n "$USER_IP" ]; then
    echo "================================================================================"
    echo "2. Creando regla para acceso EXTERNO desde desarrollo local"
    echo "================================================================================"
    
    RULE_NAME_EXTERNAL="allow-capibara6-external-dev"
    echo "Nombre de regla: $RULE_NAME_EXTERNAL"
    echo "Red: $NETWORK"
    echo "Puertos: $PORTS_EXTERNAL"
    echo "Origen: $USER_IP/32 (tu IP pública)"
    echo ""
    
    read -p "¿Crear esta regla? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creando regla de firewall externa..."
        
        gcloud compute firewall-rules create "$RULE_NAME_EXTERNAL" \
            --project=$PROJECT \
            --network=$NETWORK \
            --allow=tcp:$PORTS_EXTERNAL \
            --source-ranges=$USER_IP/32 \
            --description="Permitir acceso a servicios Capibara6 desde desarrollo local (IP: $USER_IP)" \
            --direction=INGRESS \
            --priority=1000 \
            2>&1 | while read line; do
                if [[ $line == *"Created"* ]] || [[ $line == *"already exists"* ]]; then
                    echo -e "${GREEN}$line${NC}"
                else
                    echo "$line"
                fi
            done
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo -e "${GREEN}✅ Regla externa creada exitosamente${NC}"
        else
            echo -e "${RED}❌ Error creando regla externa${NC}"
        fi
    else
        echo -e "${YELLOW}Omitiendo creación de regla externa${NC}"
    fi
fi

echo ""

# Verificar reglas creadas
echo "================================================================================"
echo "Reglas de firewall configuradas:"
echo "================================================================================"

gcloud compute firewall-rules list \
    --project=$PROJECT \
    --filter="name~allow-capibara6" \
    --format="table(name,network,direction,priority,sourceRanges.list():label=SOURCE_RANGES,allowed[].map().firewall_rule().list():label=ALLOW,targetTags.list():label=TARGET_TAGS)" \
    2>/dev/null || echo "No se pudieron listar las reglas"

echo ""

# Verificar que las VMs pueden comunicarse
echo "================================================================================"
echo "Verificando configuración de VMs..."
echo "================================================================================"

# Obtener IPs internas de las VMs
echo "Obteniendo IPs internas de las VMs..."

BOUNTY2_INTERNAL=$(gcloud compute instances describe bounty2 \
    --zone=europe-west4-a \
    --project=$PROJECT \
    --format="value(networkInterfaces[0].networkIP)" 2>/dev/null || echo "")

RAG3_INTERNAL=$(gcloud compute instances describe rag3 \
    --zone=europe-west2-c \
    --project=$PROJECT \
    --format="value(networkInterfaces[0].networkIP)" 2>/dev/null || echo "")

GPTOSS_INTERNAL=$(gcloud compute instances describe gpt-oss-20b \
    --zone=europe-southwest1-b \
    --project=$PROJECT \
    --format="value(networkInterfaces[0].networkIP)" 2>/dev/null || echo "")

if [ -n "$BOUNTY2_INTERNAL" ]; then
    echo -e "${GREEN}Bounty2 IP interna: $BOUNTY2_INTERNAL${NC}"
fi

if [ -n "$RAG3_INTERNAL" ]; then
    echo -e "${GREEN}rag3 IP interna: $RAG3_INTERNAL${NC}"
fi

if [ -n "$GPTOSS_INTERNAL" ]; then
    echo -e "${GREEN}gpt-oss-20b IP interna: $GPTOSS_INTERNAL${NC}"
fi

echo ""

# Resumen
echo "================================================================================"
echo "RESUMEN DE CONFIGURACION"
echo "================================================================================"
echo ""
echo -e "${GREEN}✅ Configuración completada${NC}"
echo ""
echo "Reglas de firewall creadas:"
echo "  1. $RULE_NAME_INTERNAL - Comunicación interna entre VMs"
if [ -n "$USER_IP" ]; then
    echo "  2. $RULE_NAME_EXTERNAL - Acceso desde tu IP ($USER_IP)"
fi
echo ""
echo "Próximos pasos:"
echo "  1. Verifica que los servicios están corriendo en cada VM"
echo "  2. Prueba las conexiones con: bash scripts/verify_all_services.sh"
echo "  3. Si necesitas agregar más IPs, edita las reglas o crea nuevas"
echo ""
echo "Para ver todas las reglas:"
echo "  gcloud compute firewall-rules list --project=$PROJECT"
echo ""
echo "Para eliminar una regla:"
echo "  gcloud compute firewall-rules delete NOMBRE_REGLA --project=$PROJECT"
echo ""

