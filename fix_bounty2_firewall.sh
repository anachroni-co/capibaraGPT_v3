#!/bin/bash
# Script para configurar firewall de bounty2

echo "🔧 Configurando firewall para bounty2..."
echo ""

# Añadir tag a la VM si no lo tiene
echo "📋 Verificando tags de bounty2..."
TAGS=$(gcloud compute instances describe bounty2 --zone=europe-west4-a --project=mamba-001 --format="get(tags.items)" 2>/dev/null)

if [ -z "$TAGS" ] || ! echo "$TAGS" | grep -q "bounty2"; then
    echo "➕ Añadiendo tag 'bounty2' a la VM..."
    gcloud compute instances add-tags bounty2 \
        --zone=europe-west4-a \
        --tags=bounty2 \
        --project=mamba-001
    echo "✅ Tag añadido"
else
    echo "✅ Tag 'bounty2' ya existe"
fi

echo ""
echo "🔥 Creando reglas de firewall..."

# Crear regla para puerto 5001 (backend principal)
echo "Creando regla para puerto 5001..."
gcloud compute firewall-rules create allow-bounty2-backend-5001 \
    --allow tcp:5001 \
    --source-ranges 0.0.0.0/0 \
    --target-tags bounty2 \
    --project=mamba-001 \
    --description="Permitir acceso externo al backend de Capibara6 en puerto 5001" \
    2>&1 | grep -v "already exists" || echo "Regla ya existe o creada"

# Crear regla para puerto 5000 (alternativo)
echo "Creando regla para puerto 5000..."
gcloud compute firewall-rules create allow-bounty2-backend-5000 \
    --allow tcp:5000 \
    --source-ranges 0.0.0.0/0 \
    --target-tags bounty2 \
    --project=mamba-001 \
    --description="Permitir acceso externo al backend de Capibara6 en puerto 5000" \
    2>&1 | grep -v "already exists" || echo "Regla ya existe o creada"

echo ""
echo "✅ Firewall configurado"
echo ""
echo "📋 Reglas creadas:"
gcloud compute firewall-rules list --project=mamba-001 --filter="targetTags:bounty2" --format="table(name,allowed,sourceRanges)" 2>&1

