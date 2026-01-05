#!/bin/bash
# Script específico para verificar servicios en bounty2 usando gcloud

PROJECT="mamba-001"
VM="bounty2"
ZONE="europe-west4-a"

echo "🔍 Verificando servicios en $VM..."
echo ""

# Ejecutar comando remoto para verificar servicios
gcloud compute ssh "$VM" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --command="
        echo '📋 Procesos Python activos:'
        ps aux | grep python | grep -v grep | head -10
        echo ''
        echo '📋 Procesos Ollama activos:'
        ps aux | grep ollama | grep -v grep | head -10
        echo ''
        echo '📋 Procesos Node activos:'
        ps aux | grep node | grep -v grep | head -5
        echo ''
        echo '🔌 Puertos escuchando (5000, 5001, 11434, 8000):'
        if command -v ss &> /dev/null; then
            sudo ss -tuln | grep -E ':(500[01]|11434|8000)' || echo '  No se encontraron puertos relevantes'
        elif command -v netstat &> /dev/null; then
            sudo netstat -tuln | grep -E ':(500[01]|11434|8000)' || echo '  No se encontraron puertos relevantes'
        fi
        echo ''
        echo '🧪 Probando servicios:'
        echo -n '  Puerto 11434 (Ollama): '
        if curl -s --connect-timeout 2 http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo '✅ ACTIVO'
            curl -s http://localhost:11434/api/tags | head -3
        else
            echo '❌ INACTIVO'
        fi
        echo ''
        echo -n '  Puerto 5001 (Backend): '
        if curl -s --connect-timeout 2 http://localhost:5001/api/health > /dev/null 2>&1; then
            echo '✅ ACTIVO'
            curl -s http://localhost:5001/api/health
        elif curl -s --connect-timeout 2 http://localhost:5001/health > /dev/null 2>&1; then
            echo '✅ ACTIVO (endpoint /health)'
            curl -s http://localhost:5001/health
        else
            echo '❌ INACTIVO'
        fi
        echo ''
        echo -n '  Puerto 5000 (Backend alt): '
        if curl -s --connect-timeout 2 http://localhost:5000/api/health > /dev/null 2>&1; then
            echo '✅ ACTIVO'
        elif curl -s --connect-timeout 2 http://localhost:5000/health > /dev/null 2>&1; then
            echo '✅ ACTIVO'
        else
            echo '❌ INACTIVO'
        fi
        echo ''
        echo -n '  Puerto 8000: '
        if curl -s --connect-timeout 2 http://localhost:8000 > /dev/null 2>&1; then
            echo '✅ ACTIVO'
        else
            echo '❌ INACTIVO'
        fi
        echo ''
        echo '🌐 IPs de la VM:'
        hostname -I 2>/dev/null || ip addr show | grep 'inet ' | grep -v '127.0.0.1' | head -3
        echo ''
        echo '📁 Directorio actual:'
        pwd
        echo ''
        echo '📁 Archivos en backend (si existe):'
        ls -la backend/*.py 2>/dev/null | head -10 || echo '  Directorio backend no encontrado'
        echo ''
        echo '📁 Archivos de inicio disponibles:'
        ls -la backend/start*.sh backend/start*.py 2>/dev/null | head -10 || echo '  No se encontraron scripts de inicio'
    " 2>&1

echo ""
echo "✅ Verificación completada"

