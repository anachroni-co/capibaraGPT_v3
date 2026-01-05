#!/bin/bash

# Script para verificar la integración de Acontext real con Capibara6
# Este script prueba que todos los servicios están comunicándose correctamente

echo "🔍 Verificando integración de Acontext real con Capibara6..."

# Verificar que Acontext esté corriendo
echo "   1. Verificando que Acontext API esté corriendo..."
if curl -f -s http://localhost:8029/health > /dev/null; then
  echo "   ✅ Acontext API está corriendo"
else
  echo "   ❌ Acontext API no está accesible"
  echo "   📋 Asegúrese de que Acontext esté iniciado con: ./deploy_acontext_real.sh"
  exit 1
fi

# Verificar que el gateway server esté corriendo
echo "   2. Verificando que Gateway Server esté corriendo..."
if curl -f -s http://localhost:8080/api/health > /dev/null; then
  echo "   ✅ Gateway Server está corriendo"
else
  echo "   ❌ Gateway Server no está accesible"
  echo "   📋 Asegúrese de que Gateway Server esté iniciado"
  exit 1
fi

# Verificar la conexión entre Gateway y Acontext
echo "   3. Verificando conexión Gateway → Acontext..."
if curl -f -s "http://localhost:8080/api/acontext/status" | grep -q '"status":"connected"'; then
  echo "   ✅ Conexión Gateway → Acontext funcionando"
else
  echo "   ❌ Problema en la conexión Gateway → Acontext"
  RESPONSE=$(curl -s "http://localhost:8080/api/acontext/status")
  echo "   📊 Respuesta recibida: $RESPONSE"
  exit 1
fi

# Crear una sesión de prueba
echo "   4. Probando creación de sesión en Acontext..."
SESSION_RESPONSE=$(curl -s -X POST "http://localhost:8080/api/acontext/session/create" -H "Content-Type: application/json" -d '{}')
if echo "$SESSION_RESPONSE" | grep -q '"status":"created"'; then
  SESSION_ID=$(echo "$SESSION_RESPONSE" | jq -r '.session_id' 2>/dev/null || echo "no_id")
  echo "   ✅ Sesión creada exitosamente: $SESSION_ID"
else
  echo "   ❌ Error creando sesión"
  echo "   📊 Respuesta recibida: $SESSION_RESPONSE"
  exit 1
fi

# Probar búsqueda
echo "   5. Probando búsqueda en espacio (simulada)..."
SEARCH_RESPONSE=$(curl -s -X POST "http://localhost:8080/api/acontext/search?query=test&space_id=1234&mode=fast")
if echo "$SEARCH_RESPONSE" | grep -q '"cited_blocks"'; then
  echo "   ✅ Búsqueda funcionando correctamente"
else
  echo "   ⚠️ Advertencia: Posible problema con búsqueda"
  echo "   📊 Respuesta recibida: $SEARCH_RESPONSE"
fi

# Probar creación de agentes
echo "   6. Probando endpoints de agentes..."
AGENT_RESPONSE=$(curl -s "http://localhost:8080/api/agents")
if echo "$AGENT_RESPONSE" | grep -q '"agents"'; then
  echo "   ✅ Endpoints de agentes funcionando"
else
  echo "   ❌ Problema con endpoints de agentes"
  echo "   📊 Respuesta recibida: $AGENT_RESPONSE"
  exit 1
fi

echo ""
echo "🎉 ¡Todas las verificaciones pasaron exitosamente!"
echo ""
echo "🔧 Sistema integrado operativo:"
echo "   - Acontext Server: ✅ Corriendo en http://localhost:8029"
echo "   - Gateway Server: ✅ Corriendo en http://localhost:8080"
echo "   - Conexión: ✅ Gateway conectado a Acontext"
echo "   - Sesiones: ✅ Creación de sesiones funciona"
echo "   - Agentes: ✅ Endpoints de agentes funcionando"
echo ""
echo "🚀 ¡Acontext real está completamente integrado con Capibara6!"