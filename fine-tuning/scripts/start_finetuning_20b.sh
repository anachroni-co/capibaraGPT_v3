#!/bin/bash
# Script para iniciar fine-tuning GPT-OSS-20B desde máquina local
# Ejecuta la configuración completa en la VM TPU

set -e

echo "🚀 Iniciando Fine-tuning GPT-OSS-20B"
echo "==================================="

# Variables
TPU_NAME="tx-5-oss-20b"
PROJECT_ID="mamba-001"
ZONE="us-central1-a"

echo "🔍 Verificando autenticación en Google Cloud..."
gcloud auth list || {
    echo "❌ Error: No estás autenticado en Google Cloud"
    echo "Ejecuta: gcloud auth login"
    exit 1
}

echo "📋 Verificando estado de la TPU..."
gcloud compute tpus describe ${TPU_NAME} --zone=${ZONE} --project=${PROJECT_ID} || {
    echo "❌ Error: TPU ${TPU_NAME} no encontrada"
    exit 1
}

echo "🔗 Conectando a la VM TPU y configurando entorno..."
gcloud compute tpus tpu-vm ssh \
    --zone ${ZONE} \
    --project ${PROJECT_ID} \
    ${TPU_NAME} \
    --command="
        echo '📥 Descargando script de configuración...'
        
        # Crear directorio de trabajo
        mkdir -p ~/capibara6-finetuning
        cd ~/capibara6-finetuning
        
        # Descargar script de configuración desde GitHub o copiarlo
        cat > setup_complete_environment.sh << 'SCRIPT_EOF'
$(cat fine-tuning/scripts/setup_complete_environment.sh)
SCRIPT_EOF
        
        chmod +x setup_complete_environment.sh
        
        echo '🚀 Ejecutando configuración completa...'
        ./setup_complete_environment.sh
        
        echo '✅ Configuración completada. Iniciando entrenamiento...'
        source ~/.bashrc
        /mnt/1pb-storage/launch_training.sh
    "

echo ""
echo "✅ Fine-tuning iniciado en la VM TPU"
echo "📊 Para monitorear el progreso:"
echo "   gcloud compute tpus tpu-vm ssh --zone ${ZONE} --project ${PROJECT_ID} ${TPU_NAME}"
echo "   tensorboard --logdir=/mnt/1pb-storage/logs/tensorboard"
echo ""
echo "⚠️ El entrenamiento puede tomar varias horas. Revisa los logs regularmente."
