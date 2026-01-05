#!/bin/bash
# Script para explorar la VM TPU y buckets existentes
# Ejecutar: gcloud compute tpus tpu-vm ssh --zone "us-central1-a" "tx-5-oss-20b" --project "mamba-001"

echo "🔍 Explorando VM TPU y Buckets Existentes"
echo "========================================="

echo "📊 Información del sistema:"
echo "Usuario: $(whoami)"
echo "Sistema: $(uname -a)"
echo "Fecha: $(date)"

echo ""
echo "💾 Discos disponibles:"
lsblk

echo ""
echo "📁 Espacio en sistema:"
df -h

echo ""
echo "☁️ Verificando autenticación en Google Cloud:"
gcloud auth list

echo ""
echo "📦 Explorando buckets disponibles en proyecto mamba-001:"
echo "========================================================"

echo ""
echo "🔍 Listando todos los buckets:"
gsutil ls

echo ""
echo "📊 Explorando bucket datasets-training_9b:"
if gsutil ls gs://datasets-training_9b/ > /dev/null 2>&1; then
    echo "✅ Bucket datasets-training_9b accesible"
    echo "📁 Contenido:"
    gsutil ls gs://datasets-training_9b/ | head -20
    echo ""
    echo "📊 Tamaño del bucket:"
    gsutil du -sh gs://datasets-training_9b/
else
    echo "❌ No se puede acceder a datasets-training_9b"
fi

echo ""
echo "🤖 Explorando bucket gpt-oss-20b-models:"
if gsutil ls gs://gpt-oss-20b-models/ > /dev/null 2>&1; then
    echo "✅ Bucket gpt-oss-20b-models accesible"
    echo "📁 Contenido:"
    gsutil ls gs://gpt-oss-20b-models/ | head -20
    echo ""
    echo "📊 Tamaño del bucket:"
    gsutil du -sh gs://gpt-oss-20b-models/
else
    echo "❌ No se puede acceder a gpt-oss-20b-models"
fi

echo ""
echo "🤖 Explorando bucket gpt-oss-120b-models:"
if gsutil ls gs://gpt-oss-120b-models/ > /dev/null 2>&1; then
    echo "✅ Bucket gpt-oss-120b-models accesible"
    echo "📁 Contenido:"
    gsutil ls gs://gpt-oss-120b-models/ | head -20
    echo ""
    echo "📊 Tamaño del bucket:"
    gsutil du -sh gs://gpt-oss-120b-models/
else
    echo "❌ No se puede acceder a gpt-oss-120b-models"
fi

echo ""
echo "🔍 Buscando otros buckets con 'gpt' o 'model':"
gsutil ls | grep -E "(gpt|model|dataset)" || echo "No se encontraron buckets relacionados"

echo ""
echo "🐍 Verificando Python y dependencias:"
python3 --version 2>/dev/null || echo "❌ Python3 no instalado"
pip3 --version 2>/dev/null || echo "❌ pip3 no instalado"

echo ""
echo "📦 Verificando si JAX está instalado:"
python3 -c "import jax; print('✅ JAX version:', jax.__version__)" 2>/dev/null || echo "❌ JAX no instalado"

echo ""
echo "🔧 Verificando si T5X está instalado:"
python3 -c "import t5x; print('✅ T5X disponible')" 2>/dev/null || echo "❌ T5X no instalado"

echo ""
echo "💾 Verificando espacio disponible para montar disco:"
if [ -d "/mnt" ]; then
    echo "✅ Directorio /mnt existe"
    ls -la /mnt/
else
    echo "❌ Directorio /mnt no existe"
fi

echo ""
echo "🎯 Resumen de la exploración:"
echo "============================="
echo "📊 Discos disponibles: $(lsblk | grep -c disk)"
echo "💾 Espacio total: $(df -h / | tail -1 | awk '{print $2}')"
echo "☁️ Buckets accesibles: $(gsutil ls | wc -l)"
echo "🐍 Python disponible: $(python3 --version 2>/dev/null || echo 'No')"
echo "🧠 JAX disponible: $(python3 -c 'import jax' 2>/dev/null && echo 'Sí' || echo 'No')"
