#!/bin/bash
#SBATCH --job-name=capibara_7b_sft
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=logs/capibara_7b_sft_%j.out
#SBATCH --error=logs/capibara_7b_sft_%j.err

# ============================================================================
# CapibaraGPT 7B Post-Training: SFT + RLAIF + DPO
#
# This script runs the complete post-training pipeline:
# 1. Supervised Fine-Tuning (SFT) - 2 days
# 2. RLAIF Multi-objective alignment - 3 days
# 3. DPO preference optimization - 2 days
# ============================================================================

echo "=========================================="
echo "CapibaraGPT 7B Post-Training Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

module purge
module load cuda/12.1 cudnn/8.9.7-cuda-12.1 nccl/2.18.3-cuda-12.1 python/3.11.6

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 4))

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

# ============================================================================
# PHASE 1: Supervised Fine-Tuning (SFT)
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 1: SFT - Starting $(date)"
echo "=========================================="

cat > output/deepspeed_configs/ds_config_sft.json << 'DSEOF'
{
    "train_batch_size": 128,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 2,
    "gradient_clipping": 1.0,

    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true
    },

    "bf16": {"enabled": true},

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01
        }
    },

    "scheduler": {
        "type": "WarmupCosineDecay",
        "params": {
            "warmup_max_lr": 2e-5,
            "warmup_num_steps": 100,
            "total_num_steps": 10000
        }
    }
}
DSEOF

srun --kill-on-bad-exit=1 \
    python -u scripts/post_training.py \
    --phase sft \
    --base-model checkpoints/capibara_7b/latest \
    --output-dir checkpoints/capibara_7b_sft \
    --dataset data/sft/instructions.jsonl \
    --max-steps 10000 \
    --deepspeed output/deepspeed_configs/ds_config_sft.json

echo "SFT completed: $(date)"

# ============================================================================
# PHASE 2: RLAIF Multi-Objective Alignment
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 2: RLAIF - Starting $(date)"
echo "=========================================="

cat > output/deepspeed_configs/ds_config_rlaif.json << 'DSEOF'
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 2,
    "gradient_clipping": 1.0,

    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true
    },

    "bf16": {"enabled": true},

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-6,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01
        }
    }
}
DSEOF

srun --kill-on-bad-exit=1 \
    python -u scripts/post_training.py \
    --phase rlaif \
    --base-model checkpoints/capibara_7b_sft/latest \
    --output-dir checkpoints/capibara_7b_rlaif \
    --objectives helpfulness,harmlessness,honesty,factuality \
    --constitution configs/constitution.json \
    --num-critique-revisions 2 \
    --max-steps 5000 \
    --deepspeed output/deepspeed_configs/ds_config_rlaif.json

echo "RLAIF completed: $(date)"

# ============================================================================
# PHASE 3: Direct Preference Optimization (DPO)
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 3: DPO - Starting $(date)"
echo "=========================================="

cat > output/deepspeed_configs/ds_config_dpo.json << 'DSEOF'
{
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 2,
    "gradient_clipping": 1.0,

    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "offload_optimizer": {"device": "cpu"}
    },

    "bf16": {"enabled": true},

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-7,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01
        }
    }
}
DSEOF

srun --kill-on-bad-exit=1 \
    python -u scripts/post_training.py \
    --phase dpo \
    --base-model checkpoints/capibara_7b_rlaif/latest \
    --ref-model checkpoints/capibara_7b_sft/latest \
    --output-dir checkpoints/capibara_7b_final \
    --preference-data data/preferences/human_preferences.jsonl \
    --beta 0.1 \
    --max-steps 3000 \
    --deepspeed output/deepspeed_configs/ds_config_dpo.json

echo "DPO completed: $(date)"

# ============================================================================
# PHASE 4: Expert Pruning (Optimization)
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 4: Expert Pruning - Starting $(date)"
echo "=========================================="

python -u scripts/optimize_model.py \
    --model checkpoints/capibara_7b_final/latest \
    --output-dir checkpoints/capibara_7b_optimized \
    --prune-experts \
    --target-compression 0.5 \
    --strategy cluster \
    --calibration-samples 1000

echo "=========================================="
echo "Post-Training Pipeline Complete!"
echo "Final model: checkpoints/capibara_7b_optimized"
echo "Finished: $(date)"
echo "=========================================="
