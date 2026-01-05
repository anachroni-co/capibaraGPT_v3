#!/bin/bash
# Script para verificar servicios en bounty2
# Ejecutar desde bounty2 o desde local con gcloud

set -e

echo "🔍 Verificando servicios en bounty2..."
echo ""

# Verificar procesos Python corriendo
echo "📋 Procesos Python activos:"
ps aux | grep python | grep -v grep || echo "  No hay procesos Python activos"

echo ""
echo "📋 Procesos Ollama activos:"
ps aux | grep ollama | grep -v grep || echo "  No hay procesos Ollama activos"

echo ""
echo "📋 Procesos Node activos:"
ps aux | grep node | grep -v grep || echo "  No hay procesos Node activos"

echo ""
echo "🔌 Puertos escuchando:"
if command -v netstat &> /dev/null; then
    sudo netstat -tuln | grep -E "(5001|5000|11434|8000)" || echo "  No se encontraron puertos relevantes"
elif command -v ss &> /dev/null; then
    sudo ss -tuln | grep -E "(5001|5000|11434|8000)" || echo "  No se encontraron puertos relevantes"
else
    echo "  netstat y ss no disponibles"
fi

echo ""
echo "🧪 Probando servicios localmente:"

# Probar puerto 5001
echo -n "  Puerto 5001: "
if curl -s --connect-timeout 2 http://localhost:5001/api/health > /dev/null 2>&1; then
    echo "✅ ACTIVO"
    curl -s http://localhost:5001/api/health | head -5
else
    echo "❌ INACTIVO"
fi

# Probar puerto 5000
echo -n "  Puerto 5000: "
if curl -s --connect-timeout 2 http://localhost:5000/api/health > /dev/null 2>&1; then
    echo "✅ ACTIVO"
    curl -s http://localhost:5000/api/health | head -5
else
    echo "❌ INACTIVO"
fi

# Probar puerto 11434 (Ollama)
echo -n "  Puerto 11434 (Ollama): "
if curl -s --connect-timeout 2 http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ ACTIVO"
    curl -s http://localhost:11434/api/tags | head -5
else
    echo "❌ INACTIVO"
fi

# Probar puerto 8000
echo -n "  Puerto 8000: "
if curl -s --connect-timeout 2 http://localhost:8000 > /dev/null 2>&1; then
    echo "✅ ACTIVO"
else
    echo "❌ INACTIVO"
fi

echo ""
echo "🌐 IPs de la VM:"
hostname -I 2>/dev/null || ip addr show | grep "inet " | grep -v "127.0.0.1"

echo ""
echo "✅ Verificación completada"

