#!/bin/bash
# Script específico para verificar servicios en gpt-oss-20b usando gcloud

PROJECT="mamba-001"
VM="gpt-oss-20b"
ZONE="europe-southwest1-b"

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
        echo '🔌 Puertos escuchando (5000-5010, 5678):'
        if command -v ss &> /dev/null; then
            sudo ss -tuln | grep -E ':(500[0-9]|5010|5678)' || echo '  No se encontraron puertos'
        elif command -v netstat &> /dev/null; then
            sudo netstat -tuln | grep -E ':(500[0-9]|5010|5678)' || echo '  No se encontraron puertos'
        fi
        echo ''
        echo '🧪 Probando servicios:'
        for port in 5000 5001 5002 5003 5010 5678; do
            if curl -s --connect-timeout 2 http://localhost:\$port/health > /dev/null 2>&1 || \
               curl -s --connect-timeout 2 http://localhost:\$port/api/health > /dev/null 2>&1; then
                echo \"  ✅ Puerto \$port - ACTIVO\"
            else
                echo \"  ❌ Puerto \$port - INACTIVO\"
            fi
        done
        echo ''
        echo '📁 Directorio actual:'
        pwd
        echo ''
        echo '📁 Archivos en backend (si existe):'
        ls -la backend/*.py 2>/dev/null | head -10 || echo '  Directorio backend no encontrado'
    " 2>&1

echo ""
echo "✅ Verificación completada"

