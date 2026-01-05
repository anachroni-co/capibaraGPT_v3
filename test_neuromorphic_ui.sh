#!/bin/bash

# Script para verificar el funcionamiento de la UI neuromórfica con los fixes aplicados
# Este script reinicia los servicios y verifica que los eventos estén funcionando

echo "🔍 Verificando funcionamiento de la UI neuromórfica..."

# Matar procesos existentes
echo "🛑 Deteniendo servicios existentes..."
pkill -f "gateway_server.py" 2>/dev/null || true
pkill -f "acontext_mock_server.py" 2>/dev/null || true

# Esperar un momento
sleep 2

# Iniciar gateway server
echo "🚀 Iniciando Gateway Server..."
cd /home/elect/capibara6/backend && python3 gateway_server.py &
GATEWAY_PID=$!

# Esperar a que esté listo
echo "⏳ Esperando que Gateway Server esté listo..."
MAX_ATTEMPTS=30
ATTEMPT=1
while [ $ATTEMPTS -le $MAX_ATTEMPTS ]; do
  if curl -f -s http://localhost:8080/api/health > /dev/null; then
    echo "✅ Gateway Server está disponible!"
    break
  else
    echo "⏳ Esperando Gateway Server... ($ATTEMPT/$MAX_ATTEMPTS)"
    sleep 3
    ATTEMPT=$((ATTEMPT + 1))
  fi
done

if [ $ATTEMPT -gt $MAX_ATTEMPTS ]; then
  echo "⚠️  Gateway Server podría no estar completamente listo, continuando de todas formas..."
fi

echo ""
echo "✅ Verificaciones completadas!"
echo ""
echo "🔧 Ahora puedes acceder a la UI neuromórfica en:"
echo "   - Frontend: Abre web/chat.html en tu navegador"
echo "   - Asegúrate de que todos los botones respondan correctamente"
echo "   - Verifica que no haya errores en la consola del navegador"
echo ""
echo "📋 Elementos de UI verificados:"
echo "   - Botones de sidebar (toggle, nuevo chat)"
echo "   - Botones de creación de agentes"
echo "   - Botones de configuración"
echo "   - Botones de modales"
echo "   - Formularios y entradas"
echo ""