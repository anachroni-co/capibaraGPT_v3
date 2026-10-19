#!/bin/bash
#SBATCH --job-name=capibara_7b
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --time=336:00:00
#SBATCH --partition=standard-g
#SBATCH --account=YOUR_PROJECT
#SBATCH --output=logs/capibara_7b_%j.out
#SBATCH --error=logs/capibara_7b_%j.err

# ============================================================================
# CapibaraGPT 7B Training Script - LUMI (CSC Finland)
#
# System: LUMI-G @ CSC, Finland
# GPUs: 16 nodes x 8 AMD MI250X (128 GCDs)
# Estimated time: 12 days for 1.05T tokens
# ============================================================================

echo "=========================================="
echo "CapibaraGPT 7B Training - LUMI"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GCDs: $((SLURM_NNODES * 8))"
echo "Start: $(date)"
echo "=========================================="

# Load LUMI modules
module purge
module load LUMI/23.09
module load partition/G
module load rocm/5.6.1
module load cray-python/3.10.10

# AMD ROCm settings
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HSA_FORCE_FINE_GRAIN_PCIE=1

# RCCL (ROCm NCCL) settings
export NCCL_DEBUG=INFO
export RCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export FI_CXI_ATS=0
export MPICH_GPU_SUPPORT_ENABLED=1

# Distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 8))

# Memory optimization
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_SYSTEM_DB_PATH=/tmp/miopen_sys_${SLURM_JOB_ID}
mkdir -p $MIOPEN_USER_DB_PATH $MIOPEN_SYSTEM_DB_PATH

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

mkdir -p checkpoints/capibara_7b_lumi
mkdir -p logs

# DeepSpeed config for AMD MI250X
cat > output/deepspeed_configs/ds_config_7b_lumi.json << 'DSEOF'
{
    "train_batch_size": 1024,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 2,
    "gradient_clipping": 1.0,

    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8
    },

    "bf16": {
        "enabled": true
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1
        }
    },

    "scheduler": {
        "type": "WarmupCosineDecay",
        "params": {
            "warmup_max_lr": 3e-4,
            "warmup_num_steps": 2000,
            "total_num_steps": 250000
        }
    },

    "activation_checkpointing": {
        "partition_activations": true,
        "contiguous_memory_optimization": true
    }
}
DSEOF

# Launch with srun
srun --kill-on-bad-exit=1 \
    python -u scripts/train_eurohpc.py \
    --system lumi \
    --nodes $SLURM_NNODES \
    --model-size 7B \
    --backend deepspeed \
    --zero-stage 2 \
    --batch-size 4 \
    --grad-accum 2 \
    --lr 3e-4 \
    --max-steps 250000 \
    --checkpoint-dir checkpoints/capibara_7b_lumi \
    --run

echo "Training completed: $(date)"
