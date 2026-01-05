# Fine-tuning with T5X + SeqIO

This directory contains everything needed to fine-tune language models using T5X and SeqIO on a TPU v5-64.

## Structure

```
fine-tuning/
├── README.md                    # This guide
├── configs/
│   ├── gpt_oss_20b_finetune.gin # Main configuration
│   └── example_values.txt       # Example values
├── scripts/
│   ├── launch_training.sh      # Launch script
│   ├── validate_setup.py       # Setup validation
│   └── monitor_training.py     # Real-time monitoring
├── datasets/
│   ├── seqio_tasks.py          # SeqIO configuration
│   └── data_preprocessing.py   # Data preprocessing
└── t5x/                        # T5X code
```

## Quick Start

### 1. Prepare the Environment

```bash
# Connect to TPU VM
gcloud compute tpus tpu-vm ssh --zone "us-central1-a" "tx-5-oss-20b" --project "your-project"

# On the VM, install dependencies
pip install jax[tpu] flax optax t5x seqio tensorflow tensorstore gin-config

# Clone the repository
git clone <your-repo>
cd capibaraGPT_v3/fine-tuning
```

### 2. Configure Large Storage Disk

```bash
# Configure local disk for large models
chmod +x scripts/setup_1pb_disk.sh
./scripts/setup_1pb_disk.sh

# Download models from GCS
chmod +x scripts/download_models.sh
./scripts/download_models.sh
```

### 3. Configure Parameters

Edit `configs/gpt_oss_20b_finetune.gin` and replace placeholders:

```gin
# Replace these values:
TRAINER.model_dir = "gs://your-bucket/model_dir"
TRAINER.model_ctor.vocab_size = 50257
TRAINER.model_ctor.d_model = 2048
TRAINER.model_ctor.n_layers = 36
TRAINER.model_ctor.n_heads = 32
```

### 4. Prepare Data

```bash
# Preprocess your dataset
python datasets/data_preprocessing.py /path/to/raw/data /path/to/processed/data

# Upload to GCS
gsutil -m cp -r /path/to/processed/data gs://your-bucket/datasets/
```

### 5. Configure SeqIO

Edit `datasets/seqio_tasks.py` with actual paths:

```python
split_to_filepattern={
    "train": "gs://your-bucket/datasets/processed/*.txt",
    "validation": "gs://your-bucket/datasets/processed/validation/*.txt"
}
```

### 6. Validate Setup

```bash
python scripts/validate_setup.py
```

### 7. Launch Training

```bash
# Option 1: 20B model (automatic script)
./scripts/launch_training.sh

# Option 2: 120B model (requires 1PB disk)
./scripts/launch_training_120b.sh

# Option 3: Manual (20B)
export TPU_NAME="tx-5-oss-20b"
export JAX_PROCESS_COUNT=64
export JAX_PROCESS_INDEX=0

python3 -m t5x.train \
  --gin_file="configs/gpt_oss_20b_finetune.gin" \
  --gin.TRAINER.model_dir="'gs://your-bucket/model_dir'" \
  --jax_backend_target="grpc://${TPU_NAME}:8470" \
  --alsologtostderr
```

### 8. Monitor Training

```bash
# In another terminal
python scripts/monitor_training.py gs://your-bucket/model_dir

# Or use TensorBoard
tensorboard --logdir=/mnt/1pb-storage/logs/tensorboard
```

## Detailed Configuration

### Model Parameters

- **VOCAB_SIZE**: Vocabulary size (e.g., 50257 for GPT-2)
- **D_MODEL**: Model dimension (e.g., 2048)
- **N_LAYERS**: Number of layers (e.g., 36)
- **N_HEADS**: Number of attention heads (e.g., 32)

### TPU Configuration

- **Mesh Shape**: (8, 8) = 64 chips
- **Precision**: bfloat16
- **Partitioning**: 2D for parameters and activations

### Optimizer

- **Type**: Adafactor
- **Learning Rate**: 1e-4 initial, decay to 1e-5
- **Warmup**: 1% of total steps
- **Gradient Accumulation**: 4 steps

### Data

- **Global Batch Size**: 1024 tokens
- **Mixture**: 99.2% original dataset + 0.8% new dataset
- **Checkpointing**: Every 500 steps

## Expected Metrics

| Metric | Target Value |
|--------|--------------|
| Throughput | 550-650k tokens/s |
| TPU Utilization | ≥85% |
| Initial Loss | 1.2-1.5 |
| Final Loss | ≤1.0 |
| Time per checkpoint | 8-10 min |

## Troubleshooting

### Error: "Cannot access GCS bucket"
```bash
# Verify authentication
gcloud auth list
gcloud config set project your-project

# Verify permissions
gsutil ls gs://your-bucket
```

### Error: "T5X not installed"
```bash
pip install t5x[gcp]
```

### Error: "TPU not available"
```bash
# Check TPU status
gcloud compute tpus list --filter="name:your-tpu"

# Restart if necessary
gcloud compute tpus stop your-tpu --zone=us-central1-a
gcloud compute tpus start your-tpu --zone=us-central1-a
```

### Error: "Out of memory"
- Reduce `global_batch_size` in .gin file
- Increase `gradient_accumulation_steps`
- Verify model fits in memory

## Additional Resources

- [T5X Documentation](https://github.com/google-research/t5x)
- [SeqIO Documentation](https://github.com/google/seqio)
- [TPU v5 Guide](https://cloud.google.com/tpu/docs/tpu-v5)
- [JAX Documentation](https://jax.readthedocs.io/)

## Support

If you encounter problems:

1. Check logs in `gs://your-bucket/model_dir/logs/`
2. Verify TPU status
3. Run `python scripts/validate_setup.py`
4. Consult T5X and SeqIO documentation

---

**Note**: This setup is optimized for TPU v5-64. For other hardware, adjust mesh and batch size parameters accordingly.
