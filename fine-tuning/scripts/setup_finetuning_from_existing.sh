#!/bin/bash
# Script de fine-tuning basado en la configuración existente de Capibara6
# Usa la configuración de GPT-OSS-20B que ya tienes funcionando

set -e

echo "🚀 Configurando Fine-tuning GPT-OSS-20B basado en configuración existente"
echo "======================================================================="

# Variables basadas en tu configuración existente
PROJECT_ID="mamba-001"
TPU_NAME="tx-5-oss-20b"
ZONE="us-central1-a"
VM_IP="34.175.215.109"  # IP de tu VM existente
BUCKET_DATASETS="gs://datasets-training_9b"
BUCKET_MODELS="gs://gpt-oss-20b-models"
LOCAL_STORAGE="/mnt/1pb-storage"
VENV_PATH="/home/$(whoami)/venv-capibara6-finetuning"

echo "📋 Configuración detectada:"
echo "VM IP: ${VM_IP}"
echo "Proyecto: ${PROJECT_ID}"
echo "TPU: ${TPU_NAME}"
echo "Zona: ${ZONE}"

echo ""
echo "🔧 Paso 1: Verificar sistema y discos..."
echo "Sistema: $(uname -a)"
echo "Discos disponibles:"
lsblk
echo ""
echo "Espacio disponible:"
df -h

echo ""
echo "🐍 Paso 2: Configurar Python 3.11 y entorno virtual..."
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
sudo apt install -y build-essential cmake git wget curl
sudo apt install -y libhdf5-dev pkg-config

# Crear entorno virtual específico para fine-tuning
python3.11 -m venv ${VENV_PATH}
source ${VENV_PATH}/bin/activate

echo ""
echo "📦 Paso 3: Instalar dependencias de fine-tuning..."
pip install --upgrade pip setuptools wheel

# Instalar JAX para TPU (basado en tu configuración)
pip install --upgrade "jax[tpu]>=0.4.30" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Instalar dependencias de ML
pip install flax>=0.8.3
pip install optax>=0.2.3
pip install tensorflow>=2.15
pip install tensorstore
pip install gin-config

# Instalar T5X y SeqIO
pip install git+https://github.com/google-research/t5x.git
pip install git+https://github.com/google/seqio.git

# Dependencias adicionales
pip install numpy scipy matplotlib seaborn
pip install tqdm wandb

echo ""
echo "💾 Paso 4: Configurar disco de 1PB..."
if [ ! -d "${LOCAL_STORAGE}" ]; then
    echo "Configurando disco de 1PB..."
    sudo mkdir -p ${LOCAL_STORAGE}
    
    # Buscar disco disponible (buscar discos grandes)
    DISK_DEVICE=$(lsblk -d -n -o NAME,SIZE | grep -E '[0-9]+T' | head -1 | awk '{print "/dev/"$1}')
    
    if [ -n "${DISK_DEVICE}" ]; then
        echo "Montando disco ${DISK_DEVICE} en ${LOCAL_STORAGE}"
        sudo mkfs.ext4 -F ${DISK_DEVICE}
        sudo mount -o discard,defaults ${DISK_DEVICE} ${LOCAL_STORAGE}
        sudo chown -R $(whoami):$(whoami) ${LOCAL_STORAGE}
        
        # Configurar montaje automático
        echo "${DISK_DEVICE} ${LOCAL_STORAGE} ext4 discard,defaults 0 2" | sudo tee -a /etc/fstab
    else
        echo "⚠️ No se encontró disco de 1PB, usando almacenamiento local"
        LOCAL_STORAGE="/tmp/capibara6-storage"
        mkdir -p ${LOCAL_STORAGE}
    fi
else
    echo "✅ Disco de 1PB ya está montado"
fi

echo ""
echo "📁 Paso 5: Crear estructura de directorios..."
mkdir -p ${LOCAL_STORAGE}/models/gpt-oss-20b
mkdir -p ${LOCAL_STORAGE}/checkpoints
mkdir -p ${LOCAL_STORAGE}/datasets
mkdir -p ${LOCAL_STORAGE}/logs/training
mkdir -p ${LOCAL_STORAGE}/logs/tensorboard
mkdir -p ${LOCAL_STORAGE}/vocab

echo ""
echo "☁️ Paso 6: Verificar acceso a buckets existentes..."
echo "Verificando bucket de datasets..."
if gsutil ls ${BUCKET_DATASETS}/ > /dev/null 2>&1; then
    echo "✅ Acceso a ${BUCKET_DATASETS}"
    echo "Contenido:"
    gsutil ls ${BUCKET_DATASETS}/ | head -10
else
    echo "❌ No se puede acceder a ${BUCKET_DATASETS}"
fi

echo ""
echo "Verificando bucket de modelos..."
if gsutil ls ${BUCKET_MODELS}/ > /dev/null 2>&1; then
    echo "✅ Acceso a ${BUCKET_MODELS}"
    echo "Contenido:"
    gsutil ls ${BUCKET_MODELS}/ | head -10
else
    echo "❌ No se puede acceder a ${BUCKET_MODELS}"
fi

echo ""
echo "📥 Paso 7: Descargar modelo GPT-OSS-20B existente..."
echo "Descargando desde ${BUCKET_MODELS}..."
if gsutil -m cp -r ${BUCKET_MODELS}/* ${LOCAL_STORAGE}/models/gpt-oss-20b/; then
    echo "✅ Modelo GPT-OSS-20B descargado"
else
    echo "⚠️ Error descargando modelo, continuando..."
fi

echo ""
echo "📊 Paso 8: Descargar datasets..."
echo "Descargando desde ${BUCKET_DATASETS}..."
if gsutil -m cp -r ${BUCKET_DATASETS}/datasets/* ${LOCAL_STORAGE}/datasets/; then
    echo "✅ Datasets descargados"
else
    echo "⚠️ Error descargando datasets, continuando..."
fi

echo ""
echo "🔧 Paso 9: Crear configuración .gin basada en tu setup..."
cat > ${LOCAL_STORAGE}/gpt_oss_20b_finetune.gin << 'EOF'
############################
# gpt_oss_20b_finetune.gin
# Configuración basada en Capibara6 existente
############################

TRAINER = @t5x.train.Trainer()

# Directorios (usando disco local)
TRAINER.model_dir = "/mnt/1pb-storage/checkpoints/gpt_oss_20b_finetune_model_dir"
TRAINER.init_checkpoint = "/mnt/1pb-storage/models/gpt-oss-20b/checkpoints/base"
TRAINER.checkpoint_save_interval_steps = 500
TRAINER.checkpoint_keep = 10

# Checkpointing
TRAINER.checkpointer_class = @t5x.checkpoints.tensorstore_checkpointer.TensorstoreCheckpointer()
TRAINER.checkpointer_class.path = TRAINER.model_dir

# TPU Configuration (basado en tu VM)
TRAINER.compute_dtype = 'bfloat16'
TRAINER.parameter_partition_spec = '2d'
TRAINER.activation_partition_spec = '2d'
TRAINER.gradient_accumulation_steps = 4

# Model Configuration (basado en tu GPT-OSS-20B)
TRAINER.model_ctor = @models.GPTOSS20BModel
TRAINER.model_ctor.vocab_size = 50257
TRAINER.model_ctor.d_model = 2048
TRAINER.model_ctor.n_layers = 36
TRAINER.model_ctor.n_heads = 32
TRAINER.model_ctor.dropout_rate = 0.0
TRAINER.model_ctor.init_scale = 0.02

# Optimizer (basado en tu configuración optimizada)
TRAINER.optimizer_ctor = @t5x.optimizers.Adafactor
TRAINER.optimizer_ctor.learning_rate = 1e-4
TRAINER.optimizer_ctor.beta2 = 0.999
TRAINER.optimizer_ctor.weight_decay = 0.0
TRAINER.optimizer_ctor.eps = 1e-30
TRAINER.optimizer_ctor.clipping_threshold = 1.0

# Learning Rate Schedule
TRAINER.lr_schedule = @t5x.lr_schedules.PolynomialDecaySchedule
TRAINER.lr_schedule.init_value = 1e-4
TRAINER.lr_schedule.end_value = 1e-5
TRAINER.lr_schedule.decay_steps = 200000
TRAINER.lr_schedule.warmup_steps = 2000

# Loss and Metrics
TRAINER.loss_fn = @t5x.losses.cross_entropy_loss
TRAINER.loss_fn.label_smoothing = 0.0
TRAINER.eval_metrics_fns = @t5x.metrics.default_metrics_fns

# Data Configuration
TRAINER.train_task_name = "gpt_oss_20b_finetune:train"
TRAINER.eval_task_name = "gpt_oss_20b_finetune:validation"
TRAINER.seqio_deterministic = True
TRAINER.seqio_shard_type = "mod"
TRAINER.global_batch_size = 1024
TRAINER.eval_batch_size = 512

# Logging
TRAINER.log_eval_steps = 1000
TRAINER.eval_interval_steps = 2000
TRAINER.save_eval_checkpoints = True
TRAINER.profile = False

# Performance
TRAINER.prefetch_to_device = 2
TRAINER.tf_data_service = None
EOF

echo ""
echo "🔧 Paso 10: Configurar variables de entorno..."
cat >> ~/.bashrc << EOF

# Capibara6 Fine-tuning Environment
export VENV_PATH="${VENV_PATH}"
export LOCAL_STORAGE="${LOCAL_STORAGE}"
export MODEL_20B_PATH="${LOCAL_STORAGE}/models/gpt-oss-20b"
export CHECKPOINTS_PATH="${LOCAL_STORAGE}/checkpoints"
export DATASETS_PATH="${LOCAL_STORAGE}/datasets"
export TPU_NAME="${TPU_NAME}"
export JAX_PROCESS_COUNT=64
export JAX_PROCESS_INDEX=0
export XLA_FLAGS="--xla_backend=host --xla_tpu_mesh_shape=(8,8) --xla_gspmd_enable=true"
export JAX_PLATFORM_NAME=tpu
export JAX_USE_PJRT_C_API_ON_TPU=1

# Activar entorno virtual
source \${VENV_PATH}/bin/activate
EOF

echo ""
echo "🚀 Paso 11: Crear script de lanzamiento..."
cat > ${LOCAL_STORAGE}/launch_training.sh << 'EOF'
#!/bin/bash
source ~/.bashrc

echo "🚀 Iniciando fine-tuning GPT-OSS-20B (Capibara6)"
echo "==============================================="
echo "Model Dir: ${CHECKPOINTS_PATH}/gpt_oss_20b_finetune_model_dir"
echo "TPU: ${TPU_NAME}"
echo "VM IP: 34.175.215.109"

# Verificar que el modelo esté disponible
if [ ! -d "${MODEL_20B_PATH}" ]; then
    echo "❌ Error: Modelo no encontrado en ${MODEL_20B_PATH}"
    exit 1
fi

# Ejecutar entrenamiento
python3 -m t5x.train \
  --gin_file="${LOCAL_STORAGE}/gpt_oss_20b_finetune.gin" \
  --gin.TRAINER.model_dir="${CHECKPOINTS_PATH}/gpt_oss_20b_finetune_model_dir" \
  --jax_backend_target="grpc://${TPU_NAME}:8470" \
  --alsologtostderr \
  --tensorboard_dir="${LOCAL_STORAGE}/logs/tensorboard" \
  2>&1 | tee "${LOCAL_STORAGE}/logs/training/training_$(date +%Y%m%d_%H%M%S).log"
EOF

chmod +x ${LOCAL_STORAGE}/launch_training.sh

echo ""
echo "📊 Paso 12: Verificar instalación..."
python3 -c "
import jax
import flax
import t5x
import seqio
print('✅ JAX version:', jax.__version__)
print('✅ Flax version:', flax.__version__)
print('✅ T5X disponible')
print('✅ SeqIO disponible')
print('✅ Dispositivos JAX:', jax.devices())
"

echo ""
echo "💾 Paso 13: Verificar espacio disponible..."
df -h ${LOCAL_STORAGE}

echo ""
echo "🎉 ¡Configuración completada!"
echo "=============================="
echo "📁 Almacenamiento: ${LOCAL_STORAGE}"
echo "🐍 Entorno virtual: ${VENV_PATH}"
echo "🤖 Modelo: GPT-OSS-20B (basado en tu configuración)"
echo "🌐 VM IP: ${VM_IP}"
echo ""
echo "🚀 Para iniciar entrenamiento:"
echo "   source ~/.bashrc"
echo "   ${LOCAL_STORAGE}/launch_training.sh"
echo ""
echo "📊 Para monitorear:"
echo "   tensorboard --logdir=${LOCAL_STORAGE}/logs/tensorboard"
echo ""
echo "⚠️ Nota: El entrenamiento puede tomar varias horas"
