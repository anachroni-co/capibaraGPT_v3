#!/bin/bash
#SBATCH --job-name=capibara_7b
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=336:00:00
#SBATCH --partition=booster
#SBATCH --account=YOUR_PROJECT
#SBATCH --output=logs/capibara_7b_%j.out
#SBATCH --error=logs/capibara_7b_%j.err

# ============================================================================
# CapibaraGPT 7B Training Script - JUWELS Booster (FZJ Germany)
#
# System: JUWELS Booster @ Forschungszentrum Jülich
# GPUs: 16 nodes x 4 NVIDIA A100 80GB = 64 GPUs
# Estimated time: 14 days for 1.05T tokens
# ============================================================================

echo "=========================================="
echo "CapibaraGPT 7B Training - JUWELS Booster"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs: $((SLURM_NNODES * 4))"
echo "Start: $(date)"
echo "=========================================="

# Load modules (JUWELS uses different naming)
module purge
module load Stages/2024
module load GCC/12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load NCCL/2.18.3-CUDA-12.1.1
module load Python/3.11.3

# InfiniBand HDR200 optimization
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=ib0

# Distributed
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 4))

# CUDA
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

mkdir -p checkpoints/capibara_7b_juwels
mkdir -p logs
mkdir -p output/deepspeed_configs

# DeepSpeed config for A100 80GB (can use larger batches)
cat > output/deepspeed_configs/ds_config_7b_juwels.json << 'DSEOF'
{
    "train_batch_size": 512,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,

    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    },

    "bf16": {
        "enabled": true
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },

    "scheduler": {
        "type": "WarmupCosineDecay",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-4,
            "warmup_num_steps": 2000,
            "total_num_steps": 250000
        }
    },

    "activation_checkpointing": {
        "partition_activations": true,
        "contiguous_memory_optimization": true,
        "cpu_checkpointing": false
    }
}
DSEOF

srun --kill-on-bad-exit=1 \
    python -u scripts/train_eurohpc.py \
    --system juwels \
    --nodes $SLURM_NNODES \
    --model-size 7B \
    --backend deepspeed \
    --zero-stage 2 \
    --batch-size 8 \
    --grad-accum 1 \
    --lr 3e-4 \
    --max-steps 250000 \
    --checkpoint-dir checkpoints/capibara_7b_juwels \
    --run

echo "Training completed: $(date)"
