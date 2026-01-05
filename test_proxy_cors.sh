#!/bin/bash
# Script para probar el proxy CORS local

echo "🧪 Probando Proxy CORS Local (puerto 8001)"
echo "=========================================="
echo ""

# 1. Probar health check del proxy
echo "1️⃣ Probando health check del proxy..."
curl -s "http://localhost:8001/" | python3 -m json.tool 2>/dev/null || curl -s "http://localhost:8001/"
echo ""
echo ""

# 2. Probar /health a través del proxy
echo "2️⃣ Probando /health a través del proxy..."
curl -s "http://localhost:8001/health" | python3 -m json.tool 2>/dev/null || curl -s "http://localhost:8001/health"
echo ""
echo ""

# 3. Probar /api/health a través del proxy
echo "3️⃣ Probando /api/health a través del proxy..."
curl -s "http://localhost:8001/api/health" | python3 -m json.tool 2>/dev/null || curl -s "http://localhost:8001/api/health"
echo ""
echo ""

# 4. Probar conexión directa al backend (para comparar)
echo "4️⃣ Probando conexión directa al backend (comparación)..."
curl -s "http://34.12.166.76:5001/health" | tail -1 | python3 -m json.tool 2>/dev/null || curl -s "http://34.12.166.76:5001/health" | tail -1
echo ""
echo ""

echo "=========================================="
echo "✅ Pruebas completadas"
echo "=========================================="

