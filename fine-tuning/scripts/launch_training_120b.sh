#!/bin/bash
# Script de lanzamiento para fine-tuning GPT-OSS-120B en TPU v5-64
# Usa disco local de 1PB para modelos grandes

set -e

# Configuración
TPU_NAME="tx-5-oss-20b"
PROJECT_ID="mamba-001"
ZONE="us-central1-a"
LOCAL_STORAGE="/mnt/1pb-storage"
MODEL_DIR="${LOCAL_STORAGE}/checkpoints/gpt_oss_120b_finetune_model_dir"
GIN_FILE="${LOCAL_STORAGE}/configs/gpt_oss_120b_finetune.gin"

echo "🚀 Iniciando fine-tuning GPT-OSS-120B en TPU v5-64"
echo "================================================="
echo "TPU: ${TPU_NAME}"
echo "Proyecto: ${PROJECT_ID}"
echo "Zona: ${ZONE}"
echo "Almacenamiento local: ${LOCAL_STORAGE}"
echo "Model Dir: ${MODEL_DIR}"
echo ""

# Verificar que el disco esté montado
if [ ! -d "${LOCAL_STORAGE}" ]; then
    echo "❌ Error: Disco de 1PB no está montado en ${LOCAL_STORAGE}"
    echo "   Ejecuta primero: ./setup_1pb_disk.sh"
    exit 1
fi

# Verificar que el modelo esté disponible
if [ ! -d "${LOCAL_STORAGE}/models/gpt-oss-120b" ]; then
    echo "❌ Error: Modelo 120B no encontrado en ${LOCAL_STORAGE}/models/gpt-oss-120b"
    echo "   Ejecuta primero: ./download_models.sh"
    exit 1
fi

# Verificar que el archivo .gin existe
if [ ! -f "${GIN_FILE}" ]; then
    echo "❌ Error: No se encuentra el archivo ${GIN_FILE}"
    echo "   Copia el archivo gpt_oss_120b_finetune.gin al disco local"
    exit 1
fi

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
        export XLA_FLAGS='--xla_backend=host --xla_tpu_mesh_shape=(8,8) --xla_gspmd_enable=true --xla_gpu_enable_async_all_gather=false'
        export JAX_PLATFORM_NAME=tpu
        export JAX_USE_PJRT_C_API_ON_TPU=1
        
        # Usar almacenamiento local
        export LOCAL_STORAGE='${LOCAL_STORAGE}'
        export MODEL_20B_LOCAL=\"\${LOCAL_STORAGE}/models/gpt-oss-20b\"
        export MODEL_120B_LOCAL=\"\${LOCAL_STORAGE}/models/gpt-oss-120b\"
        export CHECKPOINTS_LOCAL=\"\${LOCAL_STORAGE}/checkpoints\"
        export DATASETS_LOCAL=\"\${LOCAL_STORAGE}/datasets\"
        
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
        
        # Verificar espacio en disco
        echo '💾 Espacio disponible en disco local:'
        df -h \${LOCAL_STORAGE}
        
        # Crear directorio de logs
        mkdir -p \${LOCAL_STORAGE}/logs/training
        
        echo '🚀 Iniciando entrenamiento GPT-OSS-120B...'
        echo '📊 Monitoreo disponible en: \${LOCAL_STORAGE}/logs/tensorboard'
        echo '💾 Checkpoints se guardan en: ${MODEL_DIR}'
        echo '⚠️ Modelo grande: entrenamiento puede ser lento'
        echo ''
        
        # Ejecutar entrenamiento
        python3 -m t5x.train \
            --gin_file='${GIN_FILE}' \
            --gin.TRAINER.model_dir='${MODEL_DIR}' \
            --jax_backend_target='grpc://\${TPU_NAME}:8470' \
            --alsologtostderr \
            --tensorboard_dir='\${LOCAL_STORAGE}/logs/tensorboard' \
            --profile_dir='\${LOCAL_STORAGE}/logs/profiles' \
            2>&1 | tee \${LOCAL_STORAGE}/logs/training/training_120b_\$(date +%Y%m%d_%H%M%S).log
    "

echo ""
echo "✅ Script de lanzamiento completado"
echo "📊 Para monitorear el entrenamiento:"
echo "   gcloud compute tpus tpu-vm ssh --zone ${ZONE} --project ${PROJECT_ID} ${TPU_NAME}"
echo "   tensorboard --logdir=${LOCAL_STORAGE}/logs/tensorboard"
echo ""
echo "⚠️ Nota: El modelo 120B es muy grande y el entrenamiento puede tomar mucho tiempo"
