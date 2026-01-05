# Fine-tuning GPT-OSS-20B con T5X + SeqIO

Este directorio contiene todo lo necesario para realizar fine-tuning del modelo GPT-OSS-20B usando T5X y SeqIO en una TPU v5-64.

## 🏗️ Estructura

```
fine-tuning/
├── README.md                    # Esta guía
├── configs/
│   ├── gpt_oss_20b_finetune.gin # Configuración principal
│   └── example_values.txt       # Valores de ejemplo
├── scripts/
│   ├── launch_training.sh      # Script de lanzamiento
│   ├── validate_setup.py       # Validación del setup
│   └── monitor_training.py     # Monitoreo en tiempo real
├── datasets/
│   ├── seqio_tasks.py          # Configuración de SeqIO
│   └── data_preprocessing.py   # Preprocesamiento de datos
└── t5x/                        # Código T5X (movido desde backend/finetuning/)
```

## 🚀 Inicio Rápido

### 1. Preparar el entorno

```bash
# Conectar a la VM TPU
gcloud compute tpus tpu-vm ssh --zone "us-central1-a" "tx-5-oss-20b" --project "mamba-001"

# En la VM, instalar dependencias
pip install jax[tpu] flax optax t5x seqio tensorflow tensorstore gin-config

# Clonar el repositorio
git clone <tu-repo>
cd capibara6/fine-tuning
```

### 2. Configurar disco de 1 Petabyte

```bash
# Configurar disco local para modelos grandes
chmod +x scripts/setup_1pb_disk.sh
./scripts/setup_1pb_disk.sh

# Descargar modelos desde GCS
chmod +x scripts/download_models.sh
./scripts/download_models.sh
```

### 3. Configurar parámetros

Edita `configs/gpt_oss_20b_finetune.gin` y reemplaza los placeholders:

```gin
# Reemplazar estos valores:
TRAINER.model_dir = "gs://tu-bucket/gpt_oss_20b_finetune_model_dir"
TRAINER.model_ctor.vocab_size = 50257
TRAINER.model_ctor.d_model = 2048
TRAINER.model_ctor.n_layers = 36
TRAINER.model_ctor.n_heads = 32
```

### 4. Preparar datos

```bash
# Preprocesar tu dataset
python datasets/data_preprocessing.py /path/to/raw/data /path/to/processed/data

# Subir a GCS
gsutil -m cp -r /path/to/processed/data gs://tu-bucket/datasets/
```

### 5. Configurar SeqIO

Edita `datasets/seqio_tasks.py` con las rutas reales de tus datos:

```python
split_to_filepattern={
    "train": "gs://tu-bucket/datasets/processed/*.txt",
    "validation": "gs://tu-bucket/datasets/processed/validation/*.txt"
}
```

### 6. Validar setup

```bash
python scripts/validate_setup.py
```

### 7. Lanzar entrenamiento

```bash
# Opción 1: Modelo 20B (script automático)
./scripts/launch_training.sh

# Opción 2: Modelo 120B (requiere disco de 1PB)
./scripts/launch_training_120b.sh

# Opción 3: Manual (20B)
export TPU_NAME="tx-5-oss-20b"
export JAX_PROCESS_COUNT=64
export JAX_PROCESS_INDEX=0

python3 -m t5x.train \
  --gin_file="configs/gpt_oss_20b_finetune.gin" \
  --gin.TRAINER.model_dir="'gs://datasets-training_9b/gpt_oss_20b_finetune_model_dir'" \
  --jax_backend_target="grpc://${TPU_NAME}:8470" \
  --alsologtostderr
```

### 8. Monitorear entrenamiento

```bash
# En otra terminal
python scripts/monitor_training.py gs://datasets-training_9b/gpt_oss_20b_finetune_model_dir

# O usar TensorBoard
tensorboard --logdir=/mnt/1pb-storage/logs/tensorboard
```

## ⚙️ Configuración Detallada

### Parámetros del Modelo

- **VOCAB_SIZE**: Tamaño del vocabulario (ej: 50257 para GPT-2)
- **D_MODEL**: Dimensión del modelo (ej: 2048)
- **N_LAYERS**: Número de capas (ej: 36)
- **N_HEADS**: Número de cabezas de atención (ej: 32)

### Configuración de TPU

- **Mesh Shape**: (8, 8) = 64 chips
- **Precisión**: bfloat16
- **Partitioning**: 2D para parámetros y activaciones

### Optimizador

- **Tipo**: Adafactor
- **Learning Rate**: 1e-4 inicial, decay a 1e-5
- **Warmup**: 1% del total de steps
- **Gradient Accumulation**: 4 steps

### Datos

- **Global Batch Size**: 1024 tokens
- **Mixture**: 99.2% dataset original + 0.8% nuevo dataset
- **Checkpointing**: Cada 500 steps

## 📊 Métricas Esperadas

| Métrica | Valor Objetivo |
|---------|----------------|
| Throughput | 550-650k tokens/s |
| Utilización TPU | ≥85% |
| Pérdida inicial | 1.2-1.5 |
| Pérdida final | ≤1.0 |
| Tiempo por checkpoint | 8-10 min |

## 🔧 Troubleshooting

### Error: "No se puede acceder al bucket GCS"
```bash
# Verificar autenticación
gcloud auth list
gcloud config set project mamba-001

# Verificar permisos
gsutil ls gs://tu-bucket
```

### Error: "T5X no está instalado"
```bash
pip install t5x[gcp]
```

### Error: "TPU no disponible"
```bash
# Verificar estado de TPU
gcloud compute tpus list --filter="name:tx-5-oss-20b"

# Reiniciar si es necesario
gcloud compute tpus stop tx-5-oss-20b --zone=us-central1-a
gcloud compute tpus start tx-5-oss-20b --zone=us-central1-a
```

### Error: "Out of memory"
- Reducir `global_batch_size` en el .gin
- Aumentar `gradient_accumulation_steps`
- Verificar que el modelo cabe en memoria

## 📚 Recursos Adicionales

- [Documentación T5X](https://github.com/google-research/t5x)
- [Documentación SeqIO](https://github.com/google/seqio)
- [Guía TPU v5](https://cloud.google.com/tpu/docs/tpu-v5)
- [JAX Documentation](https://jax.readthedocs.io/)

## 🆘 Soporte

Si encuentras problemas:

1. Revisa los logs en `gs://tu-bucket/gpt_oss_20b_finetune_model_dir/logs/`
2. Verifica el estado de la TPU
3. Ejecuta `python scripts/validate_setup.py`
4. Consulta la documentación de T5X y SeqIO

---

**Nota**: Este setup está optimizado para TPU v5-64. Para otros hardware, ajusta los parámetros de mesh y batch size según corresponda.
