#!/bin/bash
# Script para diagnosticar el error 500 en el backend de Capibara6

echo "🔍 Diagnóstico del Error 500 en Backend"
echo "======================================"

echo ""
echo "1️⃣ Verificando puertos activos:"
echo "Puerto 8080 (GPT-OSS-20B):"
sudo lsof -i :8080 || sudo netstat -tuln | grep :8080 || echo "❌ Puerto 8080 no activo"

echo ""
echo "Puerto 5001 (Servidor Integrado):"
sudo lsof -i :5001 || sudo netstat -tuln | grep :5001 || echo "❌ Puerto 5001 no activo"

echo ""
echo "2️⃣ Verificando respuestas de servicios:"
echo "Modelo GPT-OSS-20B:"
curl -v http://localhost:8080/health 2>&1 | head -10

echo ""
echo "Servidor Integrado:"
curl -v http://localhost:5001/health 2>&1 | head -10

echo ""
echo "3️⃣ Verificando logs recientes:"
if [ -d "backend/logs" ]; then
    echo "Últimos errores:"
    tail -30 backend/logs/errors.log 2>/dev/null || echo "No hay archivo de errores"
    echo ""
    echo "Últimas líneas del servidor:"
    tail -30 backend/logs/capibara6.log 2>/dev/null || echo "No hay logs del servidor"
elif [ -d "/home/$(whoami)/capibara6/backend/logs" ]; then
    echo "Últimos errores:"
    tail -30 /home/$(whoami)/capibara6/backend/logs/errors.log 2>/dev/null || echo "No hay archivo de errores"
    echo ""
    echo "Últimas líneas del servidor:"
    tail -30 /home/$(whoami)/capibara6/backend/logs/capibara6.log 2>/dev/null || echo "No hay logs del servidor"
else
    echo "⚠️ Buscando logs..."
    find ~ -name "capibara6.log" -o -name "errors.log" 2>/dev/null | head -2 | while read logfile; do
        echo "Log encontrado: $logfile"
        tail -20 "$logfile"
    done
fi

echo ""
echo "4️⃣ Verificando procesos del servidor:"
ps aux | grep -E "(capibara6_integrated|python.*5001|flask)" | grep -v grep

echo ""
echo "5️⃣ Verificando conexión del servidor al modelo:"
curl -v http://34.175.215.109:8080/health 2>&1 | head -10

echo ""
echo "6️⃣ Verificando archivo del servidor:"
if [ -f "backend/capibara6_integrated_server.py" ]; then
    echo "✅ Servidor encontrado en backend/"
    ls -lh backend/capibara6_integrated_server.py
elif [ -f "/home/$(whoami)/capibara6/backend/capibara6_integrated_server.py" ]; then
    echo "✅ Servidor encontrado en ~/capibara6/backend/"
    ls -lh /home/$(whoami)/capibara6/backend/capibara6_integrated_server.py
else
    echo "⚠️ Buscando archivo del servidor..."
    find ~ -name "capibara6_integrated_server.py" 2>/dev/null | head -1
fi

echo ""
echo "7️⃣ Verificando errores de Python:"
python3 -c "import flask; import requests; print('✅ Dependencias básicas OK')" 2>&1

echo ""
echo "8️⃣ Probando endpoint /api/chat directamente:"
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"test","conversation":[]}' \
  2>&1 | head -20

echo ""
echo "🎯 Diagnóstico completado"
