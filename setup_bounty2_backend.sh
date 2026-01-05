#!/bin/bash
# Script completo para configurar y verificar el backend en bounty2

BOUNTY2_IP="34.12.166.76"
ZONE="europe-west4-a"
PROJECT="mamba-001"

echo "🚀 CONFIGURACIÓN DEL BACKEND EN BOUNTY2"
echo "========================================"
echo ""

# 1. Verificar tags de la VM
echo "1️⃣ Verificando tags de la VM..."
TAGS=$(gcloud compute instances describe bounty2 --zone=$ZONE --project=$PROJECT --format="get(tags.items)" 2>&1)
if [ -z "$TAGS" ] || ! echo "$TAGS" | grep -q "bounty2"; then
    echo "   ➕ Añadiendo tag 'bounty2'..."
    gcloud compute instances add-tags bounty2 \
        --zone=$ZONE \
        --tags=bounty2 \
        --project=$PROJECT
    echo "   ✅ Tag añadido"
else
    echo "   ✅ Tag 'bounty2' ya existe: $TAGS"
fi
echo ""

# 2. Crear reglas de firewall
echo "2️⃣ Configurando firewall..."
echo "   Creando regla para puerto 5001..."
gcloud compute firewall-rules create allow-bounty2-backend-5001 \
    --allow tcp:5001 \
    --source-ranges 0.0.0.0/0 \
    --target-tags bounty2 \
    --project=$PROJECT \
    --description="Permitir acceso externo al backend de Capibara6 en puerto 5001" \
    2>&1 | grep -v "already exists" || echo "   ✅ Regla ya existe"

echo "   Creando regla para puerto 5000..."
gcloud compute firewall-rules create allow-bounty2-backend-5000 \
    --allow tcp:5000 \
    --source-ranges 0.0.0.0/0 \
    --target-tags bounty2 \
    --project=$PROJECT \
    --description="Permitir acceso externo al backend de Capibara6 en puerto 5000" \
    2>&1 | grep -v "already exists" || echo "   ✅ Regla ya existe"
echo ""

# 3. Verificar procesos dentro de la VM
echo "3️⃣ Verificando procesos en bounty2..."
echo "   (Conectando a la VM, esto puede tardar...)"
PROCESSES=$(gcloud compute ssh --zone=$ZONE bounty2 --project=$PROJECT \
    --command="ps aux | grep -E 'python.*(server|flask|capibara6|integrated)' | grep -v grep" 2>&1)

if echo "$PROCESSES" | grep -q "python"; then
    echo "   ✅ Procesos encontrados:"
    echo "$PROCESSES" | sed 's/^/      /'
else
    echo "   ⚠️  No se encontraron procesos de backend corriendo"
    echo "   📝 Necesitas iniciar el backend manualmente"
fi
echo ""

# 4. Verificar puertos abiertos dentro de la VM
echo "4️⃣ Verificando puertos abiertos en bounty2..."
PORTS=$(gcloud compute ssh --zone=$ZONE bounty2 --project=$PROJECT \
    --command="sudo ss -tulnp 2>/dev/null | grep LISTEN | grep -E '(5000|5001|8000|8080|11434)'" 2>&1)

if echo "$PORTS" | grep -q "LISTEN"; then
    echo "   ✅ Puertos abiertos:"
    echo "$PORTS" | sed 's/^/      /'
else
    echo "   ⚠️  No se encontraron puertos relevantes abiertos"
fi
echo ""

# 5. Probar conexión desde local
echo "5️⃣ Probando conexión desde local..."
echo "   Probando puerto 5001..."
if timeout 3 curl -s "http://$BOUNTY2_IP:5001/health" > /dev/null 2>&1; then
    echo "   ✅ Puerto 5001 ACCESIBLE"
    RESPONSE=$(curl -s "http://$BOUNTY2_IP:5001/health")
    echo "   Respuesta: $RESPONSE"
elif timeout 3 bash -c "echo > /dev/tcp/$BOUNTY2_IP/5001" 2>/dev/null; then
    echo "   ⚠️  Puerto 5001 abierto pero no responde /health"
else
    echo "   ❌ Puerto 5001 NO ACCESIBLE"
fi

echo "   Probando puerto 5000..."
if timeout 3 curl -s "http://$BOUNTY2_IP:5000/health" > /dev/null 2>&1; then
    echo "   ✅ Puerto 5000 ACCESIBLE"
    RESPONSE=$(curl -s "http://$BOUNTY2_IP:5000/health")
    echo "   Respuesta: $RESPONSE"
elif timeout 3 bash -c "echo > /dev/tcp/$BOUNTY2_IP/5000" 2>/dev/null; then
    echo "   ⚠️  Puerto 5000 abierto pero no responde /health"
else
    echo "   ❌ Puerto 5000 NO ACCESIBLE"
fi
echo ""

# 6. Resumen y próximos pasos
echo "========================================"
echo "📋 RESUMEN"
echo "========================================"
echo ""

if timeout 2 curl -s "http://$BOUNTY2_IP:5001/health" > /dev/null 2>&1; then
    echo "✅ El backend está ACCESIBLE en el puerto 5001"
    echo "   El frontend debería poder conectarse correctamente"
    echo ""
    echo "🔗 URL del backend: http://$BOUNTY2_IP:5001"
elif timeout 2 curl -s "http://$BOUNTY2_IP:5000/health" > /dev/null 2>&1; then
    echo "✅ El backend está ACCESIBLE en el puerto 5000"
    echo "   Actualiza la configuración del frontend para usar el puerto 5000"
    echo ""
    echo "🔗 URL del backend: http://$BOUNTY2_IP:5000"
else
    echo "❌ El backend NO está accesible"
    echo ""
    echo "🔧 PRÓXIMOS PASOS:"
    echo ""
    echo "1. Conectarse a bounty2:"
    echo "   gcloud compute ssh --zone $ZONE bounty2 --project $PROJECT"
    echo ""
    echo "2. Verificar si el backend está corriendo:"
    echo "   ps aux | grep python | grep server"
    echo "   sudo ss -tulnp | grep -E '(5000|5001)'"
    echo ""
    echo "3. Si no está corriendo, iniciarlo:"
    echo "   cd ~/capibara6/backend"
    echo "   python3 server.py"
    echo "   # O"
    echo "   python3 capibara6_integrated_server_ollama.py"
    echo ""
    echo "4. Verificar que el firewall esté configurado:"
    echo "   gcloud compute firewall-rules list --project=$PROJECT --filter='targetTags:bounty2'"
fi

