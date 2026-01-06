# capibara/training - Training System

The **training** module implements the advanced training system for capibaraGPT-v2, including **Meta-Consensus**, **TPU v6e optimization**, and multiple distributed training strategies.

## Table of Contents

1. [Overview](#overview)
2. [Main Components](#main-components)
3. [Meta-Consensus System](#meta-consensus-system)
4. [Training Strategies](#training-strategies)
5. [TPU v6e Training](#tpu-v6e-training)
6. [Quick Start](#quick-start)
7. [Advanced Configuration](#advanced-configuration)
8. [Optimizations](#optimizations)

---

## Overview

The capibaraGPT-v2 training system implements advanced strategies for training high-quality language models:

### Key Features

- **TPU v6e Optimized**: Ultra-fast training on Google Cloud TPU v6e-64/256
- **Meta-Consensus**: Consensus system for combining multiple models/strategies
- **Distributed Training**: Data parallelism + Model parallelism + Expert parallelism
- **Multiple Strategies**: Hierarchical, Convexity, Incremental Soup, HuggingFace Integration
- **Consensus Algorithms**: Byzantine-tolerant, Convex optimization, Federated learning
- **Cython Kernels**: Optimized C++ kernels for critical operations
- **Advanced Monitoring**: Real-time dashboard with detailed metrics

### System Architecture

```
┌──────────────────────────────────────────────────────────┐
│            Meta-Consensus Training System                │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Consensus Coordinator                   │   │
│  │  - Byzantine-tolerant voting                    │   │
│  │  - Model merging strategies                     │   │
│  │  - Quality gating                               │   │
│  └───────────────┬─────────────────────────────────┘   │
│                  │                                       │
│    ┌─────────────┼─────────────┐                       │
│    ▼             ▼             ▼                       │
│  ┌─────┐     ┌─────┐       ┌──────┐                   │
│  │Node1│     │Node2│  ...  │NodeN │                   │
│  │TPU  │     │TPU  │       │ TPU  │                   │
│  └─────┘     └─────┘       └──────┘                   │
│                                                           │
│  Each Node:                                             │
│  ├─> Training Strategy (Hierarchical/Convexity/Soup)   │
│  ├─> Local Model Updates                               │
│  ├─> Gradient Synchronization                          │
│  └─> Consensus Participation                           │
└──────────────────────────────────────────────────────────┘
```

---

## Main Components

| Component | File | Purpose |
|------------|---------|-----------|
| **Meta-Consensus System** | `meta_consensus_system.py` | Coordinates distributed training with consensus |
| **TPU v6e Trainer** | `tpu_v6e_trainer.py` | Optimized trainer for TPU v6e |
| **TPU v6e Config** | `tpu_v6e_config.py` | TPU v6e specific configuration |
| **Consensus Algorithms** | `advanced_consensus_algorithms.py` | Byzantine-tolerant algorithms |
| **Hierarchical Strategy** | `hierarchical_training_strategy.py` | Multi-level hierarchical training |
| **Convexity Strategy** | `convexity_training_strategy.py` | Convex optimization for training |
| **Incremental Soup** | `incremental_soup_strategy.py` | Incremental model soups |
| **HuggingFace Integration** | `huggingface_consensus_strategy.py` | Integration with HF Transformers |
| **Hybrid Expert Router** | `hybrid_expert_router.py` | Expert routing during training |
| **Monitoring Dashboard** | `monitoring_dashboard.py` | Live metrics dashboard |
| **Cython Kernels** | `cython_kernels/` | Optimized C++ kernels |
| **Federated Consensus** | `federated_consensus/` | Distributed federated learning |

---

## Meta-Consensus System

The **Meta-Consensus System** is the core of distributed training, enabling training of multiple model variants and combining them through consensus.

### Meta-Consensus Architecture

```python
from capibara.training import MetaConsensusSystem, ConsensusConfig

# Configure consensus system
consensus_config = ConsensusConfig(
    num_nodes=8,              # 8 TPUs participating
    voting_strategy="byzantine_tolerant",
    merge_strategy="weighted_average",
    quality_threshold=0.85,   # Minimum quality score
    consensus_threshold=0.75  # 75% of nodes must agree
)

# Create system
consensus_system = MetaConsensusSystem(consensus_config)

# Train with consensus
results = consensus_system.train(
    data_path="gs://capibara-data/",
    num_epochs=10,
    strategies=["hierarchical", "convexity", "incremental_soup"]
)
```

### Available Consensus Algorithms

#### 1. Byzantine-Tolerant Voting

Tolerates up to (n-1)/3 failed or malicious nodes:

```python
from capibara.training.advanced_consensus_algorithms import ByzantineConsensus

consensus = ByzantineConsensus(
    fault_tolerance=0.33,  # Tolerates up to 33% failed nodes
    verification_rounds=3
)

# Byzantine voting
consensus_model = consensus.vote(models=[model1, model2, model3])
```

#### 2. Convex Optimization Consensus

Convex optimization for model merging:

```python
from capibara.training.convexity_controller import ConvexityController

controller = ConvexityController(
    lambda_reg=0.01,
    max_iterations=100
)

# Find optimal combination
optimal_weights = controller.optimize(
    models=models,
    validation_data=val_data
)

merged_model = controller.merge(models, optimal_weights)
```

#### 3. Federated Consensus

Federated learning with differential privacy:

```python
from capibara.training.federated_consensus import FederatedConsensusStrategy

federated = FederatedConsensusStrategy(
    num_clients=20,
    privacy_budget=1.0,  # Epsilon for differential privacy
    aggregation_method="fedavg"
)

# Federated training round
global_model = federated.train_round(
    global_model=model,
    client_data=client_datasets
)
```

---

## Training Strategies

### 1. Hierarchical Training Strategy

Multi-level training with level-specific specialization:

```python
from capibara.training.hierarchical_strategy import HierarchicalTrainingPipeline

# Configure hierarchical pipeline
pipeline = HierarchicalTrainingPipeline(
    levels=[
        {  # Level 1: General knowledge
            "name": "general",
            "datasets": ["wikipedia", "books"],
            "epochs": 3
        },
        {  # Level 2: Domain specialization
            "name": "specialized",
            "datasets": ["academic", "legal", "medical"],
            "epochs": 2
        },
        {  # Level 3: Fine-tuning
            "name": "finetuning",
            "datasets": ["specific_task"],
            "epochs": 1
        }
    ]
)

# Execute hierarchical training
final_model = pipeline.train(base_model)
```

### 2. Convexity Training Strategy

Optimization with convexity guarantees:

```python
from capibara.training import ConvexityTrainingStrategy

strategy = ConvexityTrainingStrategy(
    regularization=0.01,
    convex_constraint=True,
    use_proximal_gradient=True
)

# Training with convexity constraints
model = strategy.train(
    model=model,
    train_data=train_data,
    epochs=10
)
```

### 3. Incremental Soup Strategy

Incremental model soups (combines checkpoints):

```python
from capibara.training import IncrementalSoupStrategy

soup = IncrementalSoupStrategy(
    merging_method="uniform",  # uniform, weighted, greedy
    evaluation_metric="perplexity"
)

# Add checkpoints to soup
soup.add_checkpoint("checkpoint_epoch_1.pkl")
soup.add_checkpoint("checkpoint_epoch_2.pkl")
soup.add_checkpoint("checkpoint_epoch_3.pkl")

# Create soup model
souped_model = soup.merge(validation_data=val_data)
```

### 4. HuggingFace Consensus Strategy

Integration with HuggingFace Transformers:

```python
from capibara.training import HuggingFaceConsensusStrategy

hf_strategy = HuggingFaceConsensusStrategy(
    model_name="bert-base-uncased",
    num_consensus_models=5,
    merge_method="weighted_average"
)

# Training with HF integration
model = hf_strategy.train(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    epochs=3
)
```

---

## TPU v6e Training

### TPU v6e-64 Configuration

```python
from capibara.training import TPUv6eTrainer, TPUv6eConfig

# Configure TPU v6e
tpu_config = TPUv6eConfig(
    tpu_type="v6e-64",
    mesh_shape=(8, 8),  # 64 chips in 8x8 mesh
    batch_size=128,
    use_bf16=True,
    enable_flash_attention=True,
    xla_flags={
        "xla_tpu_enable_data_parallelism": True,
        "xla_tpu_enable_async_collective_fusion": True
    }
)

# Create trainer
trainer = TPUv6eTrainer(tpu_config)

# Training on TPU
results = trainer.train(
    model=model,
    train_data="gs://capibara-data/training/",
    eval_data="gs://capibara-data/validation/",
    num_steps=100000,
    checkpoint_every=1000
)
```

### TPU v6e Optimizations

```python
from capibara.training import TPUv6ConsensusOptimizer

# Specialized optimizer for TPU v6e
optimizer = TPUv6ConsensusOptimizer(
    learning_rate=3e-4,
    use_bf16_accumulation=True,
    gradient_clipping=1.0,
    enable_xla_fusion=True
)

# Features:
# - Native BFloat16 accumulation
# - Automatic XLA fusion
# - Optimized collective operations
# - Memory-efficient gradient checkpointing
```

---

## Quick Start

### Basic Training (CPU/GPU)

```python
from capibara.training import UnifiedTrainer
from capibara.core import ModularCapibaraModel, ModularConfig

# 1. Configure model
config = ModularConfig.from_toml("config/development/config.toml")
model = ModularCapibaraModel(config)

# 2. Create trainer
trainer = UnifiedTrainer(
    model=model,
    learning_rate=3e-4,
    batch_size=32
)

# 3. Train
trainer.train(
    train_data="data/train/",
    eval_data="data/eval/",
    num_epochs=10,
    checkpoint_dir="checkpoints/"
)
```

### Training with Meta-Consensus (TPU)

```python
from capibara.training import MetaConsensusSystem, ConsensusConfig

# 1. Configure consensus
consensus_config = ConsensusConfig(
    num_nodes=8,  # 8 TPUs
    voting_strategy="byzantine_tolerant",
    merge_strategy="convex_optimization"
)

# 2. Create consensus system
consensus = MetaConsensusSystem(consensus_config)

# 3. Distributed training with consensus
results = consensus.train(
    model=model,
    data_path="gs://capibara-data/",
    strategies=["hierarchical", "convexity"],
    num_epochs=20
)

# 4. Get consensus model
final_model = results.consensus_model
```

---

## Advanced Configuration

### Training with All Strategies

```python
from capibara.training import IntegratedConsensusStrategy

# Integrate multiple strategies
integrated = IntegratedConsensusStrategy(
    strategies={
        "hierarchical": {
            "enabled": True,
            "weight": 0.4
        },
        "convexity": {
            "enabled": True,
            "weight": 0.3
        },
        "incremental_soup": {
            "enabled": True,
            "weight": 0.3
        }
    },
    consensus_threshold=0.8
)

# Integrated training
model = integrated.train(
    base_model=model,
    train_data=train_data,
    epochs=15
)
```

### Real-Time Monitoring

```python
from capibara.training import MonitoringDashboard

# Create dashboard
dashboard = MonitoringDashboard(
    port=8080,
    update_frequency=10,  # seconds
    metrics=[
        "loss",
        "perplexity",
        "gradient_norm",
        "consensus_score",
        "tpu_utilization",
        "memory_usage"
    ]
)

# Start dashboard
dashboard.start()

# Training with monitoring
trainer = UnifiedTrainer(model=model, dashboard=dashboard)
trainer.train(...)

# View dashboard at http://localhost:8080
```

### Data Preprocessing

```python
from capibara.training.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(
    tokenizer=tokenizer,
    max_length=2048,
    padding="max_length",
    num_workers=8
)

# Preprocess data
processed_data = preprocessor.process(
    input_path="data/raw/",
    output_path="data/processed/",
    cache=True
)
```

---

## Optimizations

### Cython Kernels

Optimized C++ kernels for critical operations:

```python
from capibara.training.cython_kernels import (
    fast_attention,
    optimized_matmul,
    sparse_softmax
)

# Use optimized kernels
# 10-50x faster than pure Python implementations
attn_output = fast_attention(q, k, v)
```

### Gradient Checkpointing

Reduces memory at the cost of some computation:

```python
from capibara.training import enable_gradient_checkpointing

# Enable gradient checkpointing
model = enable_gradient_checkpointing(
    model=model,
    checkpoint_every_n_layers=2
)

# Reduces memory ~50% with ~20% more computation
```

### Mixed Precision Training

```python
from capibara.training import MixedPrecisionTrainer

trainer = MixedPrecisionTrainer(
    model=model,
    precision="bf16",  # bf16, fp16, or mixed
    loss_scaling=True,
    dynamic_loss_scale=True
)

# 2-3x faster training, ~50% less memory
```

---

## Metrics and Evaluation

### Available Metrics

```python
from capibara.training import TrainingMetrics

metrics = TrainingMetrics(
    track=[
        "loss",              # Training loss
        "perplexity",        # Language model perplexity
        "gradient_norm",     # Gradient norm
        "learning_rate",     # Current LR
        "consensus_score",   # Meta-consensus agreement
        "expert_utilization",  # MoE expert usage
        "throughput",        # Samples/second
        "memory_usage"       # GPU/TPU memory
    ]
)

# Log during training
for batch in dataloader:
    loss = train_step(batch)
    metrics.log(loss=loss, step=step)

# Export metrics
metrics.export("metrics.json")
```

### Integration with Weights & Biases

```python
import wandb
from capibara.training import WandbIntegration

# Setup W&B
wandb.init(project="capibaraGPT-v2", name="tpu-training-run-001")

# Integrate with trainer
trainer = UnifiedTrainer(
    model=model,
    wandb_project="capibaraGPT-v2"
)

# Metrics are automatically logged to W&B
```

---

## Debugging and Troubleshooting

### Error: "TPU not found"

```bash
# Verify TPU availability
python -c "import jax; print(jax.devices())"

# Configure environment variables
export JAX_PLATFORMS=tpu
export TPU_NAME=your-tpu-name
```

### Error: "Consensus timeout"

```python
# Increase timeout
consensus_config = ConsensusConfig(
    consensus_timeout=600,  # 10 minutes
    max_retries=5
)
```

### Error: "Out of memory"

Solutions:
1. Reduce `batch_size`
2. Enable `gradient_checkpointing`
3. Use `accumulation_steps` to simulate larger batch size
4. Reduce `sequence_length`

```python
trainer = UnifiedTrainer(
    batch_size=16,  # Reduced from 128
    gradient_accumulation_steps=8,  # Simulates batch_size=128
    use_gradient_checkpointing=True
)
```

---

## Subdirectories

- **`cython_kernels/`**: Optimized C++/Cython kernels
- **`data_lineage/`**: Data lineage tracking
- **`data_preprocessing/`**: Data preprocessing
- **`federated_consensus/`**: Federated learning
- **`hierarchical_strategy/`**: Hierarchical training
- **`optimizations/`**: Training optimizations

---

## References

- [TPU v6e Trainer](tpu_v6e_trainer.py) - TPU v6e trainer
- [Meta-Consensus System](meta_consensus_system.py) - Consensus system
- [Consensus Algorithms](advanced_consensus_algorithms.py) - Byzantine algorithms
- [Monitoring Dashboard](monitoring_dashboard.py) - Metrics dashboard
- [Config Manager](config_manager.py) - Configuration management

---

## Support

For training issues:
1. Check logs in `logs/training.log`
2. Verify monitoring dashboard
3. Check consensus metrics
4. Open GitHub issue with complete logs

---

**Last updated**: 2025-11-16
**System version**: v2.0.0
