#!/bin/bash
#SBATCH --job-name=capibara_7b
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=336:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=logs/capibara_7b_%j.out
#SBATCH --error=logs/capibara_7b_%j.err

# ============================================================================
# CapibaraGPT 7B Training Script - Leonardo (CINECA)
#
# System: Leonardo @ CINECA, Italy
# GPUs: 16 nodes x 4 NVIDIA A100 64GB = 64 GPUs
# Estimated time: 14 days for 1.05T tokens
# ============================================================================

echo "=========================================="
echo "CapibaraGPT 7B Training - Leonardo"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs: $((SLURM_NNODES * 4))"
echo "Start: $(date)"
echo "=========================================="

# Load modules
module purge
module load cuda/12.1
module load cudnn/8.9.7-cuda-12.1
module load nccl/2.18.3-cuda-12.1
module load openmpi/4.1.6--nvhpc--23.11
module load python/3.11.6

# Environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 4))
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=ib0

# CUDA optimizations
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Disable tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

# Working directory
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

# Create output directories
mkdir -p checkpoints/capibara_7b
mkdir -p logs
mkdir -p output/deepspeed_configs

# Generate DeepSpeed config
cat > output/deepspeed_configs/ds_config_7b.json << 'DSEOF'
{
    "train_batch_size": 512,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 2,
    "gradient_clipping": 1.0,
    "steps_per_print": 100,
    "wall_clock_breakdown": false,

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
    },

    "flops_profiler": {
        "enabled": true,
        "profile_step": 100,
        "module_depth": -1,
        "top_modules": 3
    }
}
DSEOF

echo "DeepSpeed config generated"

# Training command
srun --kill-on-bad-exit=1 \
    python -u scripts/train_eurohpc.py \
    --system leonardo \
    --nodes $SLURM_NNODES \
    --model-size 7B \
    --backend deepspeed \
    --zero-stage 2 \
    --batch-size 4 \
    --grad-accum 2 \
    --lr 3e-4 \
    --max-steps 250000 \
    --warmup-steps 2000 \
    --checkpoint-dir checkpoints/capibara_7b \
    --save-steps 2500 \
    --output-dir output/capibara_7b \
    --run

echo "=========================================="
echo "Training completed: $(date)"
echo "=========================================="
