#!/bin/bash
# Script completo para diagnosticar y probar conexión con bounty2

BOUNTY2_IP="34.12.166.76"
ZONE="europe-west4-a"
PROJECT="mamba-001"

echo "🔍 DIAGNÓSTICO COMPLETO - Frontend ↔ Backend bounty2"
echo "=================================================="
echo ""

# 1. Probar conexión a Ollama (sabemos que funciona)
echo "1️⃣ Probando Ollama (puerto 11434)..."
if curl -s --connect-timeout 5 "http://$BOUNTY2_IP:11434/api/tags" > /dev/null 2>&1; then
    echo "   ✅ Ollama responde correctamente"
    MODELS=$(curl -s "http://$BOUNTY2_IP:11434/api/tags" | grep -o '"name":"[^"]*"' | head -3)
    echo "   📋 Modelos disponibles: $MODELS"
else
    echo "   ❌ Ollama no responde"
fi
echo ""

# 2. Probar puertos del backend
echo "2️⃣ Probando puertos del backend..."
PORTS=(5000 5001 5002 8000 8080)
for port in "${PORTS[@]}"; do
    echo -n "   Puerto $port: "
    if timeout 3 bash -c "echo > /dev/tcp/$BOUNTY2_IP/$port" 2>/dev/null; then
        echo "✅ ABIERTO"
        # Intentar health check
        response=$(curl -s --connect-timeout 2 "http://$BOUNTY2_IP:$port/health" 2>&1)
        if echo "$response" | grep -q "ok\|status\|health"; then
            echo "      ✅ Responde HTTP: ${response:0:50}..."
        else
            echo "      ⚠️  Abierto pero no responde /health"
        fi
    else
        echo "❌ CERRADO o NO ACCESIBLE"
    fi
done
echo ""

# 3. Verificar firewall
echo "3️⃣ Verificando firewall..."
echo "   Reglas de firewall para bounty2:"
gcloud compute firewall-rules list --project=$PROJECT \
    --filter="targetTags~bounty2 OR name~bounty2" \
    --format="table(name,allowed,sourceRanges,targetTags)" 2>&1 | head -10
echo ""

# 4. Verificar tags de la VM
echo "4️⃣ Verificando tags de la VM..."
TAGS=$(gcloud compute instances describe bounty2 --zone=$ZONE --project=$PROJECT --format="get(tags.items)" 2>&1)
if [ -z "$TAGS" ]; then
    echo "   ⚠️  No se encontraron tags. La VM puede no tener tags configurados."
else
    echo "   Tags: $TAGS"
fi
echo ""

# 5. Intentar obtener información de procesos dentro de la VM
echo "5️⃣ Información de procesos en bounty2..."
echo "   (Esto puede tardar unos segundos...)"
PROCESSES=$(gcloud compute ssh --zone=$ZONE bounty2 --project=$PROJECT \
    --command="ps aux | grep -E 'python.*(server|flask|capibara6)' | grep -v grep | head -5" 2>&1)
if [ $? -eq 0 ] && [ ! -z "$PROCESSES" ]; then
    echo "   Procesos encontrados:"
    echo "$PROCESSES" | sed 's/^/      /'
else
    echo "   ⚠️  No se encontraron procesos de backend corriendo"
    echo "   O no se pudo conectar a la VM"
fi
echo ""

# 6. Verificar puertos abiertos dentro de la VM
echo "6️⃣ Puertos abiertos en bounty2..."
PORTS_OPEN=$(gcloud compute ssh --zone=$ZONE bounty2 --project=$PROJECT \
    --command="sudo ss -tulnp 2>/dev/null | grep LISTEN | grep -E '(5000|5001|8000|8080|11434)'" 2>&1)
if [ $? -eq 0 ] && [ ! -z "$PORTS_OPEN" ]; then
    echo "   Puertos abiertos:"
    echo "$PORTS_OPEN" | sed 's/^/      /'
else
    echo "   ⚠️  No se encontraron puertos relevantes abiertos"
    echo "   O no se pudo conectar a la VM"
fi
echo ""

# 7. Resumen y recomendaciones
echo "=================================================="
echo "📋 RESUMEN Y RECOMENDACIONES"
echo "=================================================="
echo ""

if curl -s --connect-timeout 2 "http://$BOUNTY2_IP:5001/health" > /dev/null 2>&1; then
    echo "✅ El backend está accesible en el puerto 5001"
    echo "   El frontend debería poder conectarse correctamente"
elif curl -s --connect-timeout 2 "http://$BOUNTY2_IP:5000/health" > /dev/null 2>&1; then
    echo "✅ El backend está accesible en el puerto 5000"
    echo "   Actualiza la configuración del frontend para usar el puerto 5000"
else
    echo "❌ El backend NO está accesible desde fuera"
    echo ""
    echo "🔧 ACCIONES NECESARIAS:"
    echo "   1. Verificar que el backend esté corriendo en bounty2"
    echo "   2. Crear reglas de firewall para permitir acceso externo"
    echo "   3. Verificar que la VM tenga los tags correctos"
    echo ""
    echo "   Ejecuta estos comandos:"
    echo "   bash fix_bounty2_firewall.sh"
    echo ""
    echo "   O manualmente:"
    echo "   gcloud compute firewall-rules create allow-bounty2-backend-5001 \\"
    echo "       --allow tcp:5001 \\"
    echo "       --source-ranges 0.0.0.0/0 \\"
    echo "       --target-tags bounty2 \\"
    echo "       --project=$PROJECT"
fi

