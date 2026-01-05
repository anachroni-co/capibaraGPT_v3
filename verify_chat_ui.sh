#!/bin/bash

# Script para verificar que la interfaz de chat esté funcionando correctamente
# Ahora con las mejoras implementadas

echo "🔍 Verificando la interfaz de chat actualizada..."
echo ""

# Mostrar estado actual de los servicios
echo "📡 Estado de los servicios:"
if pgrep -f "gateway_server.py" > /dev/null; then
    echo "✅ Gateway Server está corriendo"
else
    echo "⚠️ Gateway Server no detectado - debe iniciarse manualmente"
fi

if curl -f -s http://localhost:8080/api/health > /dev/null 2>&1; then
    echo "✅ Gateway Server está respondiendo"
else
    echo "⚠️ Gateway Server no está accesible"
fi

echo ""
echo "🔧 Cambios implementados:"
echo "   ✅ Constructor de Capibara6ChatPage ahora registra actividad"
echo "   ✅ Función init() ahora registra cada paso de inicialización"
echo "   ✅ Eventos click ahora registran cuando son disparados"
echo "   ✅ Función sendMessage ahora registra cada paso"
echo "   ✅ Manejo seguro de inicialización con comprobación de elementos"
echo "   ✅ Todos los botones tienen handlers con logs"
echo "   ✅ Inicialización del DOM más robusta"
echo ""
echo "📋 Para probar la funcionalidad:"
echo "   1. Abra el archivo web/chat.html en su navegador"
echo "   2. Abra la consola del navegador (F12)"
echo "   3. Intente los siguientes elementos:"
echo "      - Escribir y enviar un mensaje"
echo "      - Clickear botones de sidebar"
echo "      - Abrir/cerrar menús y modales"
echo "      - Verificar que los logs aparezcan en la consola"
echo ""
echo "🔍 Lo que debería ver en la consola:"
echo "   - '📊 Estado del DOM: ...'"
echo "   - '🔧 Constructor de Capibara6ChatPage llamado'"
echo "   - '🚀 Iniciando función init()'"
echo "   - '🔍 Elementos DOM obtenidos: {...}'"
echo "   - '🖱️ Botón de enviar clickeado' al clickear enviar"
echo "   - '📤 sendMessage() llamado' al enviar mensaje"
echo ""
echo "💡 Si los logs aparecen en la consola, la UI está funcionando correctamente!"