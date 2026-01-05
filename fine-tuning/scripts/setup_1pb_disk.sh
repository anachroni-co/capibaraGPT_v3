#!/bin/bash
# Script para configurar el disco de 1 Petabyte en la VM TPU
# Ejecutar en la VM: gcloud compute tpus tpu-vm ssh --zone "us-central1-a" "tx-5-oss-20b" --project "mamba-001"

set -e

echo "💾 Configurando disco de 1 Petabyte para modelos grandes"
echo "========================================================"

# Variables
DISK_DEVICE="/dev/sdb"
MOUNT_POINT="/mnt/1pb-storage"
STORAGE_USER="$(whoami)"

echo "🔍 Verificando disco disponible..."
lsblk

echo ""
echo "📁 Creando punto de montaje..."
sudo mkdir -p "${MOUNT_POINT}"

echo "🔧 Formateando disco (si es necesario)..."
# Verificar si el disco ya está formateado
if ! sudo blkid "${DISK_DEVICE}" > /dev/null 2>&1; then
    echo "⚠️ Disco no formateado. Formateando con ext4..."
    sudo mkfs.ext4 -F "${DISK_DEVICE}"
else
    echo "✅ Disco ya está formateado"
fi

echo "🔗 Montando disco..."
sudo mount -o discard,defaults "${DISK_DEVICE}" "${MOUNT_POINT}"

echo "👤 Configurando permisos..."
sudo chown -R "${STORAGE_USER}:${STORAGE_USER}" "${MOUNT_POINT}"

echo "📂 Creando estructura de directorios..."
mkdir -p "${MOUNT_POINT}/models/gpt-oss-20b"
mkdir -p "${MOUNT_POINT}/models/gpt-oss-120b"
mkdir -p "${MOUNT_POINT}/checkpoints/base"
mkdir -p "${MOUNT_POINT}/checkpoints/finetuned"
mkdir -p "${MOUNT_POINT}/datasets/original"
mkdir -p "${MOUNT_POINT}/datasets/processed"
mkdir -p "${MOUNT_POINT}/datasets/eval"
mkdir -p "${MOUNT_POINT}/logs/training"
mkdir -p "${MOUNT_POINT}/logs/tensorboard"
mkdir -p "${MOUNT_POINT}/vocab"

echo "💾 Configurando montaje automático..."
echo "${DISK_DEVICE} ${MOUNT_POINT} ext4 discard,defaults 0 2" | sudo tee -a /etc/fstab

echo "📊 Verificando espacio disponible..."
df -h "${MOUNT_POINT}"

echo ""
echo "✅ Disco de 1PB configurado correctamente"
echo "📍 Ubicación: ${MOUNT_POINT}"
echo "💾 Espacio disponible: $(df -h ${MOUNT_POINT} | tail -1 | awk '{print $4}')"
echo ""
echo "🚀 Próximos pasos:"
echo "   1. Descargar modelos desde GCS a disco local"
echo "   2. Configurar datasets en ${MOUNT_POINT}/datasets/"
echo "   3. Iniciar entrenamiento con disco local"
