#!/bin/bash
# Script de lanzamiento para fine-tuning GPT-OSS-20B en TPU v5-64

set -e

# Configuración
TPU_NAME="tx-5-oss-20b"
PROJECT_ID="mamba-001"
ZONE="us-central1-a"
BUCKET="gs://datasets-training_9b"
MODEL_DIR="${BUCKET}/gpt_oss_20b_finetune_model_dir"
GIN_FILE="${BUCKET}/configs/gpt_oss_20b_finetune.gin"

echo "🚀 Iniciando fine-tuning GPT-OSS-20B en TPU v5-64"
echo "================================================"
echo "TPU: ${TPU_NAME}"
echo "Proyecto: ${PROJECT_ID}"
echo "Zona: ${ZONE}"
echo "Bucket: ${BUCKET}"
echo "Model Dir: ${MODEL_DIR}"
echo ""

# Verificar que el bucket existe
echo "🔍 Verificando bucket GCS..."
gsutil ls "${BUCKET}" > /dev/null || {
    echo "❌ Error: No se puede acceder al bucket ${BUCKET}"
    echo "   Verifica que el bucket existe y tienes permisos"
    exit 1
}

# Verificar que el archivo .gin existe
echo "🔍 Verificando archivo de configuración..."
gsutil ls "${GIN_FILE}" > /dev/null || {
    echo "❌ Error: No se encuentra el archivo ${GIN_FILE}"
    echo "   Sube el archivo gpt_oss_20b_finetune.gin al bucket"
    exit 1
}

# Conectar a la VM TPU
echo "🔗 Conectando a la VM TPU..."
gcloud compute tpus tpu-vm ssh \
    --zone "${ZONE}" \
    --project "${PROJECT_ID}" \
    "${TPU_NAME}" \
    --command="
        echo '📋 Configurando entorno en la VM...'
        
        # Variables de entorno
        export TPU_NAME='${TPU_NAME}'
        export JAX_PROCESS_COUNT=64
        export JAX_PROCESS_INDEX=0
        export XLA_FLAGS='--xla_backend=host --xla_tpu_mesh_shape=(8,8) --xla_gspmd_enable=true'
        
        # Verificar que T5X está instalado
        python3 -c 'import t5x; print(\"✅ T5X disponible\")' || {
            echo '❌ Error: T5X no está instalado'
            exit 1
        }
        
        # Verificar que SeqIO está disponible
        python3 -c 'import seqio; print(\"✅ SeqIO disponible\")' || {
            echo '❌ Error: SeqIO no está instalado'
            exit 1
        }
        
        # Crear directorio de logs
        mkdir -p logs
        
        echo '🚀 Iniciando entrenamiento...'
        echo '📊 Monitoreo disponible en: ${MODEL_DIR}/tensorboard'
        echo '💾 Checkpoints se guardan en: ${MODEL_DIR}'
        echo ''
        
        # Ejecutar entrenamiento
        python3 -m t5x.train \
            --gin_file='${GIN_FILE}' \
            --gin.TRAINER.model_dir='${MODEL_DIR}' \
            --jax_backend_target='grpc://\${TPU_NAME}:8470' \
            --alsologtostderr \
            --tensorboard_dir='${MODEL_DIR}/tensorboard' \
            --profile_dir='${MODEL_DIR}/profiles' \
            2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log
    "

echo ""
echo "✅ Script de lanzamiento completado"
echo "📊 Para monitorear el entrenamiento:"
echo "   gcloud compute tpus tpu-vm ssh --zone ${ZONE} --project ${PROJECT_ID} ${TPU_NAME}"
echo "   tensorboard --logdir=${MODEL_DIR}/tensorboard"
