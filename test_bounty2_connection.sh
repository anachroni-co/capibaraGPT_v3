#!/bin/bash
# Script para probar conexión a bounty2 desde local

BOUNTY2_IP="34.12.166.76"

echo "🔍 Probando conexión a bounty2 ($BOUNTY2_IP)..."
echo ""

# Probar diferentes puertos comunes
PORTS=(5000 5001 5002 8000 8080 11434)

for port in "${PORTS[@]}"; do
    echo -n "Probando puerto $port... "
    if timeout 3 bash -c "echo > /dev/tcp/$BOUNTY2_IP/$port" 2>/dev/null; then
        echo "✅ PUERTO ABIERTO"
        # Intentar hacer una petición HTTP
        echo "  Probando HTTP..."
        response=$(curl -s --connect-timeout 3 "http://$BOUNTY2_IP:$port/health" 2>&1 || curl -s --connect-timeout 3 "http://$BOUNTY2_IP:$port/" 2>&1)
        if [ $? -eq 0 ]; then
            echo "  ✅ Responde HTTP: ${response:0:100}"
        else
            echo "  ⚠️  Puerto abierto pero no responde HTTP"
        fi
    else
        echo "❌ Cerrado o no accesible"
    fi
done

echo ""
echo "=========================================="
echo "Verificando firewall de Google Cloud..."
echo "=========================================="
echo ""

# Verificar reglas de firewall
echo "Reglas de firewall para bounty2:"
gcloud compute firewall-rules list --project=mamba-001 --filter="targetTags:bounty2 OR targetTags:*" --format="table(name,targetTags,allowed,sourceRanges)" 2>&1 | head -20

