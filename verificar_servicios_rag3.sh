#!/bin/bash
# Script específico para verificar servicios en rag3 usando gcloud

PROJECT="mamba-001"
VM="rag3"
ZONE="europe-west2-c"

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
        echo '📋 Procesos relacionados con RAG/Base de datos:'
        ps aux | grep -E '(rag|postgres|chroma|vector)' | grep -v grep | head -10
        echo ''
        echo '🔌 Puertos escuchando (8000, 5432, 6379):'
        if command -v ss &> /dev/null; then
            sudo ss -tuln | grep -E ':(8000|5432|6379)' || echo '  No se encontraron puertos relevantes'
        elif command -v netstat &> /dev/null; then
            sudo netstat -tuln | grep -E ':(8000|5432|6379)' || echo '  No se encontraron puertos relevantes'
        fi
        echo ''
        echo '🧪 Probando servicios:'
        echo -n '  Puerto 8000 (RAG API): '
        if curl -s --connect-timeout 2 http://localhost:8000/health > /dev/null 2>&1; then
            echo '✅ ACTIVO'
            curl -s http://localhost:8000/health
        elif curl -s --connect-timeout 2 http://localhost:8000/api/health > /dev/null 2>&1; then
            echo '✅ ACTIVO (endpoint /api/health)'
            curl -s http://localhost:8000/api/health
        else
            echo '❌ INACTIVO'
        fi
        echo ''
        echo -n '  Puerto 5432 (PostgreSQL): '
        if sudo ss -tuln | grep -q ':5432' || sudo netstat -tuln | grep -q ':5432'; then
            echo '✅ Puerto abierto'
        else
            echo '❌ Puerto cerrado'
        fi
        echo ''
        echo -n '  Puerto 6379 (Redis): '
        if sudo ss -tuln | grep -q ':6379' || sudo netstat -tuln | grep -q ':6379'; then
            echo '✅ Puerto abierto'
        else
            echo '❌ Puerto cerrado'
        fi
        echo ''
        echo '🌐 IPs de la VM:'
        hostname -I 2>/dev/null || ip addr show | grep 'inet ' | grep -v '127.0.0.1' | head -3
        echo ''
        echo '📁 Directorio actual:'
        pwd
        echo ''
        echo '📁 Archivos Python relacionados con RAG:'
        find . -name '*rag*.py' -o -name '*RAG*.py' 2>/dev/null | head -10 || echo '  No se encontraron archivos RAG'
        echo ''
        echo '📁 Servicios Docker (si aplica):'
        docker ps 2>/dev/null | head -10 || echo '  Docker no disponible o sin contenedores'
    " 2>&1

echo ""
echo "✅ Verificación completada"

