#!/bin/bash
# Script completo para configurar el entorno de fine-tuning GPT-OSS-20B
# Ejecutar en la VM TPU: gcloud compute tpus tpu-vm ssh --zone "us-central1-a" "tx-5-oss-20b" --project "mamba-001"

set -e

echo "🚀 Configurando entorno completo para fine-tuning GPT-OSS-20B"
echo "=========================================================="

# Variables
PROJECT_ID="mamba-001"
BUCKET_DATASETS="gs://datasets-training_9b"
BUCKET_MODELS="gs://gpt-oss-20b-models"
LOCAL_STORAGE="/mnt/1pb-storage"
VENV_PATH="/home/$(whoami)/venv-capibara6"

echo "📋 Información del sistema:"
echo "Usuario: $(whoami)"
echo "Directorio: $(pwd)"
echo "Fecha: $(date)"

echo ""
echo "🔧 Paso 1: Actualizar sistema Ubuntu..."
sudo apt update && sudo apt upgrade -y

echo ""
echo "🐍 Paso 2: Instalar Python 3.11 y dependencias del sistema..."
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
sudo apt install -y build-essential cmake git wget curl
sudo apt install -y libhdf5-dev pkg-config

echo ""
echo "📦 Paso 3: Crear entorno virtual..."
python3.11 -m venv ${VENV_PATH}
source ${VENV_PATH}/bin/activate

echo ""
echo "🔧 Paso 4: Actualizar pip y instalar dependencias base..."
pip install --upgrade pip setuptools wheel

echo ""
echo "📚 Paso 5: Instalar JAX para TPU..."
pip install --upgrade "jax[tpu]>=0.4.30" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

echo ""
echo "🧠 Paso 6: Instalar Flax y dependencias de ML..."
pip install flax>=0.8.3
pip install optax>=0.2.3
pip install tensorflow>=2.15
pip install tensorstore
pip install gin-config

echo ""
echo "🔬 Paso 7: Instalar T5X y SeqIO..."
pip install git+https://github.com/google-research/t5x.git
pip install git+https://github.com/google/seqio.git

echo ""
echo "📊 Paso 8: Instalar dependencias adicionales..."
pip install numpy scipy matplotlib seaborn
pip install tqdm wandb  # Para logging y monitoreo

echo ""
echo "💾 Paso 9: Configurar disco de 1PB (si no está montado)..."
if [ ! -d "${LOCAL_STORAGE}" ]; then
    echo "Configurando disco de 1PB..."
    sudo mkdir -p ${LOCAL_STORAGE}
    
    # Buscar disco disponible
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
echo "📁 Paso 10: Crear estructura de directorios..."
mkdir -p ${LOCAL_STORAGE}/models/gpt-oss-20b
mkdir -p ${LOCAL_STORAGE}/checkpoints
mkdir -p ${LOCAL_STORAGE}/datasets
mkdir -p ${LOCAL_STORAGE}/logs/training
mkdir -p ${LOCAL_STORAGE}/logs/tensorboard
mkdir -p ${LOCAL_STORAGE}/vocab

echo ""
echo "☁️ Paso 11: Autenticarse en Google Cloud..."
gcloud auth list
if [ $? -ne 0 ]; then
    echo "❌ Error: No estás autenticado en Google Cloud"
    echo "Ejecuta: gcloud auth login"
    exit 1
fi

echo ""
echo "📥 Paso 12: Descargar modelo GPT-OSS-20B..."
echo "Descargando desde ${BUCKET_MODELS}..."
gsutil -m cp -r ${BUCKET_MODELS}/* ${LOCAL_STORAGE}/models/gpt-oss-20b/ || {
    echo "⚠️ Error descargando modelo, continuando con configuración..."
}

echo ""
echo "📊 Paso 13: Descargar datasets..."
echo "Descargando desde ${BUCKET_DATASETS}..."
gsutil -m cp -r ${BUCKET_DATASETS}/datasets/* ${LOCAL_STORAGE}/datasets/ || {
    echo "⚠️ Error descargando datasets, continuando con configuración..."
}

echo ""
echo "🔧 Paso 14: Configurar variables de entorno..."
cat >> ~/.bashrc << EOF

# Capibara6 Fine-tuning Environment
export VENV_PATH="${VENV_PATH}"
export LOCAL_STORAGE="${LOCAL_STORAGE}"
export MODEL_20B_PATH="${LOCAL_STORAGE}/models/gpt-oss-20b"
export CHECKPOINTS_PATH="${LOCAL_STORAGE}/checkpoints"
export DATASETS_PATH="${LOCAL_STORAGE}/datasets"
export TPU_NAME="tx-5-oss-20b"
export JAX_PROCESS_COUNT=64
export JAX_PROCESS_INDEX=0
export XLA_FLAGS="--xla_backend=host --xla_tpu_mesh_shape=(8,8) --xla_gspmd_enable=true"
export JAX_PLATFORM_NAME=tpu
export JAX_USE_PJRT_C_API_ON_TPU=1

# Activar entorno virtual
source \${VENV_PATH}/bin/activate
EOF

echo ""
echo "📋 Paso 15: Crear archivo de configuración .gin..."
cat > ${LOCAL_STORAGE}/gpt_oss_20b_finetune.gin << 'EOF'
############################
# gpt_oss_20b_finetune.gin
# Configuración para fine-tuning GPT-OSS-20B
############################

TRAINER = @t5x.train.Trainer()

# Directorios
TRAINER.model_dir = "/mnt/1pb-storage/checkpoints/gpt_oss_20b_finetune_model_dir"
TRAINER.init_checkpoint = "/mnt/1pb-storage/models/gpt-oss-20b/checkpoints/base"
TRAINER.checkpoint_save_interval_steps = 500
TRAINER.checkpoint_keep = 10

# Checkpointing
TRAINER.checkpointer_class = @t5x.checkpoints.tensorstore_checkpointer.TensorstoreCheckpointer()
TRAINER.checkpointer_class.path = TRAINER.model_dir

# TPU Configuration
TRAINER.compute_dtype = 'bfloat16'
TRAINER.parameter_partition_spec = '2d'
TRAINER.activation_partition_spec = '2d'
TRAINER.gradient_accumulation_steps = 4

# Model Configuration
TRAINER.model_ctor = @models.GPTOSS20BModel
TRAINER.model_ctor.vocab_size = 50257
TRAINER.model_ctor.d_model = 2048
TRAINER.model_ctor.n_layers = 36
TRAINER.model_ctor.n_heads = 32
TRAINER.model_ctor.dropout_rate = 0.0
TRAINER.model_ctor.init_scale = 0.02

# Optimizer
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
echo "🎯 Paso 16: Crear script de lanzamiento..."
cat > ${LOCAL_STORAGE}/launch_training.sh << 'EOF'
#!/bin/bash
source ~/.bashrc

echo "🚀 Iniciando fine-tuning GPT-OSS-20B..."
echo "Model Dir: ${TRAINER.model_dir}"
echo "TPU: ${TPU_NAME}"

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
echo "📊 Paso 17: Verificar instalación..."
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
echo "💾 Paso 18: Verificar espacio disponible..."
df -h ${LOCAL_STORAGE}

echo ""
echo "🎉 ¡Configuración completada!"
echo "=============================="
echo "📁 Almacenamiento: ${LOCAL_STORAGE}"
echo "🐍 Entorno virtual: ${VENV_PATH}"
echo "🚀 Para iniciar entrenamiento:"
echo "   source ~/.bashrc"
echo "   ${LOCAL_STORAGE}/launch_training.sh"
echo ""
echo "📊 Para monitorear:"
echo "   tensorboard --logdir=${LOCAL_STORAGE}/logs/tensorboard"
echo ""
echo "⚠️ Nota: El entrenamiento puede tomar varias horas"
