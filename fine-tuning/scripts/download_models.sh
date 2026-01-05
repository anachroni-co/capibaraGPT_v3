#!/bin/bash
# Script para descargar modelos desde GCS al disco local de 1PB
# Ejecutar en la VM TPU después de configurar el disco

set -e

# Configuración
LOCAL_STORAGE="/mnt/1pb-storage"
BUCKET_20B="gs://gpt-oss-20b-models"
BUCKET_120B="gs://gpt-oss-120b-models"
BUCKET_DATASETS="gs://datasets-training_9b"

echo "📥 Descargando modelos y datasets al disco local"
echo "==============================================="

# Verificar que el disco esté montado
if [ ! -d "${LOCAL_STORAGE}" ]; then
    echo "❌ Error: Disco de 1PB no está montado en ${LOCAL_STORAGE}"
    echo "   Ejecuta primero: ./setup_1pb_disk.sh"
    exit 1
fi

echo "🔍 Verificando espacio disponible..."
df -h "${LOCAL_STORAGE}"

echo ""
echo "📦 Descargando modelo GPT-OSS-20B..."
if gsutil -m cp -r "${BUCKET_20B}/*" "${LOCAL_STORAGE}/models/gpt-oss-20b/"; then
    echo "✅ Modelo 20B descargado correctamente"
else
    echo "⚠️ Error descargando modelo 20B, continuando..."
fi

echo ""
echo "📦 Descargando modelo GPT-OSS-120B..."
if gsutil -m cp -r "${BUCKET_120B}/*" "${LOCAL_STORAGE}/models/gpt-oss-120b/"; then
    echo "✅ Modelo 120B descargado correctamente"
else
    echo "⚠️ Error descargando modelo 120B, continuando..."
fi

echo ""
echo "📊 Descargando datasets..."
if gsutil -m cp -r "${BUCKET_DATASETS}/datasets/*" "${LOCAL_STORAGE}/datasets/"; then
    echo "✅ Datasets descargados correctamente"
else
    echo "⚠️ Error descargando datasets, continuando..."
fi

echo ""
echo "📋 Descargando vocabularios..."
if gsutil -m cp "${BUCKET_20B}/vocab/*" "${LOCAL_STORAGE}/vocab/"; then
    echo "✅ Vocabulario 20B descargado"
fi

if gsutil -m cp "${BUCKET_120B}/vocab/*" "${LOCAL_STORAGE}/vocab/"; then
    echo "✅ Vocabulario 120B descargado"
fi

echo ""
echo "📊 Verificando descargas..."
echo "Modelo 20B:"
ls -la "${LOCAL_STORAGE}/models/gpt-oss-20b/" | head -5

echo ""
echo "Modelo 120B:"
ls -la "${LOCAL_STORAGE}/models/gpt-oss-120b/" | head -5

echo ""
echo "Datasets:"
ls -la "${LOCAL_STORAGE}/datasets/" | head -5

echo ""
echo "💾 Espacio usado después de descargas:"
df -h "${LOCAL_STORAGE}"

echo ""
echo "✅ Descarga de modelos completada"
echo "📍 Modelos disponibles en: ${LOCAL_STORAGE}/models/"
echo "📊 Datasets disponibles en: ${LOCAL_STORAGE}/datasets/"
