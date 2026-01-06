# capibara/core - Core Components

The **core** module is the architectural heart of capibaraGPT-v2, containing all the fundamental components of the system.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Detailed Components](#detailed-components)
6. [Integration Patterns](#integration-patterns)
7. [Performance Optimization](#performance-optimization)

---

## 🎯 Overview

The `core/` directory contains **19 subdirectories** with the essential components:

| Component | Purpose | Documentation |
|------------|-----------|---------------|
| **moe/** | Mixture of Experts (32 experts) | [README](moe/README.md) |
| **cot/** | Chain-of-Thought reasoning | [README](cot/README.md) |
| **routers/** | Routing strategies (Base, Adaptive, BTO, TTS) | [README](routers/README.md) |
| **optimizers/** | Optimizers (Adam, AdamW, Schedulers) | [README](optimizers/README.md) |
| **encoders/** | Multimodal encoders (Vision, Video) | [README](encoders/README.md) |
| **adapters/** | Parameter-efficient fine-tuning | [README](adapters/README.md) |
| **distributed/** | Distributed training (TPU mesh, sharding) | [README](distributed/README.md) |
| **tpu/** | TPU optimizations | [README](tpu/README.md) |
| **monitoring/** | Monitoring and alerting system | [README](monitoring/README.md) |
| **observers/** | Observer pattern implementation | [README](observers/README.md) |
| **experts/** | Expert system control | [README](experts/README.md) |
| **pipelines/** | RAG 2.0, Multimodal, TTS pipelines | [README](pipelines/README.md) |
| **kernels/** | Optimized kernels (Flash Attention, MatMul) | [README](kernels/README.md) |
| **activations/** | Contextual activations | [README](activations/README.md) |
| **arm_optimizations/** | ARM CPU optimizations (SVE, NEON) | [README](arm_optimizations/README.md) |
| **age_adaptation/** | Age and cultural adaptation | [README](age_adaptation/README.md) |
| **inference_ttc/** | Time-to-completion optimization | [README](inference_ttc/README.md) |
| **verification/** | Model verification utilities | - |

### Base Components

Additionally, `core/` contains essential base files:

- **`modular_model.py`**: Modular model that integrates all components
- **`router.py`**: Main routing system
- **`config.py`**: Model configuration
- **`optimization.py`**: Metrics and training state

---

## 🏗️ Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  ModularCapibaraModel                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input → Encoders → Router → [Mamba/Transformer]       │
│                       │                                  │
│                       ├──> MoE (32 experts)             │
│                       │                                  │
│                       ├──> CoT (Chain-of-Thought)       │
│                       │                                  │
│                       └──> Pipelines (RAG/TTS)          │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Adapters (LoRA, Prefix, Adapter Layers)        │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Optimizers (AdamW + LR Schedulers)              │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Monitoring (Metrics, Alerts, Dashboards)        │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

Infrastructure Layer:
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Distributed  │  │     TPU      │  │   Kernels    │
│  (Sharding)  │  │(Optimizations│  │  (Flash Attn)│
└──────────────┘  └──────────────┘  └──────────────┘
```

### Execution Flow

```
1. Input Processing
   └─> Encoders (Vision/Video/Text) → Embeddings

2. Routing Decision
   └─> Router → Selects strategy (Mamba vs Transformer)

3. Core Processing
   ├─> Mamba/Transformer Attention
   ├─> MoE Layer (Top-2 of 32 experts)
   └─> Optional: CoT Reasoning

4. Pipeline Processing (Optional)
   ├─> RAG 2.0 (Retrieval)
   ├─> Multimodal Fusion
   └─> TTS Generation

5. Output Generation
   └─> Adapters (if fine-tuning) → Output
```

---

## 🚀 Quick Start

### Basic Usage of Modular Model

```python
from capibara.core import ModularCapibaraModel, ModularConfig

# Configuration
config = ModularConfig(
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    vocab_size=50000,
    use_moe=True,
    num_experts=32,
    num_active_experts=2
)

# Create model
model = ModularCapibaraModel(config)

# Forward pass
import jax.numpy as jnp
inputs = jnp.ones((2, 128), dtype=jnp.int32)  # (batch, seq_len)
outputs = model(inputs, training=False)
```

### Usage with Advanced Router

```python
from capibara.core import create_enhanced_router, RouterType

# Create hybrid router
router = create_enhanced_router(
    router_type=RouterType.HYBRID,
    config={
        "mamba_threshold": 512,
        "memory_threshold": 0.8
    }
)

# Routing decision
decision = router.route(
    inputs=inputs,
    sequence_length=128,
    memory_pressure=0.5
)
# decision.strategy = "mamba" or "transformer"
```

### Usage with MoE

```python
from capibara.core.moe import DynamicMoE, MoEConfig

# Configure MoE
moe_config = MoEConfig(
    num_experts=32,
    num_active_experts=2,
    expert_capacity_factor=1.25,
    load_balancing_weight=0.01
)

# Create MoE layer
moe = DynamicMoE(moe_config)

# Forward pass
moe_outputs = moe(inputs, training=True)
```

### Usage with Adapters

```python
from capibara.core.adapters import AdapterSystem, AdapterConfig

# Configure adapter (LoRA)
adapter_config = AdapterConfig(
    adapter_type="lora",
    r=8,  # Rank
    alpha=16,
    dropout=0.1
)

# Create adapter system
adapter_system = AdapterSystem(adapter_config)

# Apply adapter
adapted_outputs = adapter_system.apply(outputs)
```

---

## 🔧 Detailed Components

### 1. MoE (Mixture of Experts)

**Location**: `core/moe/`

The MoE system provides 32 specialized experts with Top-2 routing:

```python
from capibara.core.moe import DynamicMoE

# Expert specialization
expert_types = {
    "general": 8,          # Generalist experts
    "reasoning": 8,        # Logical reasoning
    "languages": 8,        # Minority languages
    "creative": 8          # Creative generation
}

# Automatic load balancing
# Capacity factor to avoid overload
# Auxiliary loss for uniform distribution
```

See [moe/README.md](moe/README.md) for full details.

### 2. Chain-of-Thought (CoT)

**Location**: `core/cot/`

Multi-step reasoning system with self-verification:

```python
from capibara.core import ChainOfThought, create_cot_handler

# Create CoT handler
cot = create_cot_handler(
    max_steps=5,
    use_process_rewards=True,
    enable_self_reflection=True
)

# Execute reasoning
result = cot.reason(
    problem="What is 15% of 80?",
    context={"domain": "mathematics"}
)
# result.steps = ["Step 1: ...", "Step 2: ...", ...]
# result.answer = "12"
```

See [cot/README.md](cot/README.md) for full details.

### 3. Routers

**Location**: `core/routers/`

Multiple routing strategies:

- **BaseRouter**: Simple rule-based routing
- **AdaptiveRouter**: Adaptive routing based on real-time metrics
- **BTORouter**: Bayesian Task-Oriented routing
- **TTSRouter**: Specialized routing for Text-to-Speech

```python
from capibara.core.routers import AdaptiveRouter

router = AdaptiveRouter(
    learning_rate=0.01,
    adaptation_window=100
)

# Router learns from past metrics
router.update_metrics(latency=0.5, quality=0.9)
decision = router.route(inputs)
```

See [routers/README.md](routers/README.md) for details.

### 4. Optimizers

**Location**: `core/optimizers/`

Optimizers with advanced LR schedulers:

```python
from capibara.core.optimizers import create_optimizer

optimizer = create_optimizer(
    optimizer_type="adamw",
    learning_rate=3e-4,
    weight_decay=0.01,
    schedule="cosine",
    warmup_steps=1000
)
```

See [optimizers/README.md](optimizers/README.md) for details.

### 5. Multimodal Encoders

**Location**: `core/encoders/`

Encoders for vision, video, and multimodal fusion:

```python
from capibara.core.encoders import VisionEncoder, MultimodalCombiner

# Vision encoder
vision_encoder = VisionEncoder(hidden_size=768)
image_embeddings = vision_encoder(images)

# Multimodal combiner
combiner = MultimodalCombiner(fusion_type="attention")
fused = combiner(text_emb=text, vision_emb=images)
```

See [encoders/README.md](encoders/README.md) for details.

### 6. Distributed Training

**Location**: `core/distributed/`

Configuration for distributed training on TPU:

```python
from capibara.core.distributed import create_mesh_config

# Configure mesh for TPU v5e-64
mesh_config = create_mesh_config(
    mesh_shape=(8, 8),  # 64 chips
    data_parallel=8,
    model_parallel=8
)
```

See [distributed/README.md](distributed/README.md) for details.

---

## 🔗 Integration Patterns

### Pattern 1: Complete Model with All Components

```python
from capibara.core import ModularCapibaraModel, ModularConfig
from capibara.core.adapters import AdapterSystem
from capibara.core.monitoring import create_monitor

# 1. Configure model
config = ModularConfig.from_toml("config/production/config.toml")

# 2. Create model
model = ModularCapibaraModel(config)

# 3. Add adapters (optional - for fine-tuning)
adapter_system = AdapterSystem.from_config(config.adapter_config)
model = adapter_system.wrap_model(model)

# 4. Setup monitoring
monitor = create_monitor(model, metrics=["loss", "perplexity", "moe_balance"])

# 5. Training loop
for batch in dataloader:
    outputs = model(batch.inputs, training=True)
    loss = compute_loss(outputs, batch.targets)

    # Monitor metrics
    monitor.log_step(loss=loss, outputs=outputs)
```

### Pattern 2: Inference with Hybrid Routing

```python
from capibara.core import create_enhanced_router, RouterType
from capibara.sub_models.mamba import MambaModule
from capibara.sub_models.hybrid import HybridAttentionModule

# Setup router
router = create_enhanced_router(RouterType.HYBRID)

# Setup modules
mamba = MambaModule(config.mamba_config)
transformer = TransformerModule(config.transformer_config)

# Inference
def infer(inputs):
    decision = router.route(inputs, len(inputs))

    if decision.strategy == "mamba":
        return mamba(inputs)
    else:
        return transformer(inputs)
```

### Pattern 3: RAG + CoT Pipeline

```python
from capibara.core.pipelines import RAGPipeline
from capibara.core.cot import ChainOfThought

# Setup RAG
rag = RAGPipeline(
    index_path="data/index/",
    retrieval_k=10
)

# Setup CoT
cot = ChainOfThought(max_steps=5)

# Inference with RAG + CoT
def rag_cot_infer(query):
    # 1. Retrieve context
    context = rag.retrieve(query)

    # 2. Reason with CoT
    reasoning = cot.reason(
        problem=query,
        context=context
    )

    # 3. Generate final answer
    answer = model.generate(
        query=query,
        context=context,
        reasoning_steps=reasoning.steps
    )

    return answer
```

---

## ⚡ Performance Optimization

### TPU Optimizations

```python
from capibara.core.tpu import configure_tpu_environment
from capibara.core.distributed import create_mesh_config

# Configure TPU environment
configure_tpu_environment(
    use_bf16=True,
    enable_flash_attention=True,
    xla_flags="--xla_tpu_enable_data_parallelism=true"
)

# Configure mesh
mesh_config = create_mesh_config(
    mesh_shape=(8, 8),  # TPU v5e-64
    data_parallel=8,
    model_parallel=8
)
```

### Kernel Optimizations

```python
from capibara.core.kernels import FlashAttention, OptimizedMatMul

# Use Flash Attention (reduces memory O(n²) → O(n))
flash_attn = FlashAttention()
attn_output = flash_attn(q, k, v)

# Use optimized MatMul
matmul = OptimizedMatMul(use_tpu_optimizations=True)
output = matmul(a, b)
```

### Monitoring for Optimization

```python
from capibara.core.monitoring import PerformanceMonitor

monitor = PerformanceMonitor(
    track_memory=True,
    track_latency=True,
    track_throughput=True
)

with monitor.track("forward_pass"):
    outputs = model(inputs)

# View metrics
metrics = monitor.get_metrics()
print(f"Latency: {metrics['forward_pass']['latency']:.3f}s")
print(f"Memory: {metrics['forward_pass']['memory_mb']:.1f}MB")
```

---

## 📊 Metrics and Monitoring

The core system includes extensive monitoring:

```python
from capibara.core.monitoring import create_dashboard

# Create dashboard
dashboard = create_dashboard(
    model=model,
    port=8080,
    enable_grafana=True
)

# Available metrics:
# - Training: loss, perplexity, gradient norm
# - MoE: expert utilization, load balance, routing entropy
# - Performance: latency, throughput, memory usage
# - System: CPU/TPU utilization, temperature
```

See [monitoring/README.md](monitoring/README.md) for complete dashboard.

---

## 🔍 Testing and Validation

```python
from capibara.core.verification import validate_model

# Validate that the model works correctly
validation_result = validate_model(
    model=model,
    test_inputs=test_data,
    checks=[
        "forward_pass",
        "gradient_flow",
        "moe_routing",
        "memory_usage"
    ]
)

assert validation_result.all_passed()
```

---

## 📚 References

- [Modular Model](modular_model.py) - Model implementation
- [Main Router](router.py) - Routing system
- [Configuration](config.py) - Model config
- [MoE README](moe/README.md) - Mixture of Experts
- [CoT README](cot/README.md) - Chain-of-Thought
- [Distributed README](distributed/README.md) - Distributed training
- [TPU README](tpu/README.md) - TPU optimizations

---

## 🆘 Troubleshooting

### Error: "Module 'ultra_core_integration' not found"

```python
from capibara.core import ULTRA_CORE_AVAILABLE

if not ULTRA_CORE_AVAILABLE:
    print("Ultra Core Integration not available")
    # Use standard components instead
```

### Error: "Out of Memory on TPU"

- Reduce `batch_size` in config
- Enable `gradient_checkpointing`
- Reduce number of active experts in MoE
- See [tpu/README.md](tpu/README.md) for more optimizations

### Slow Performance

- Verify TPU/GPU is being used
- Enable Flash Attention in kernels
- Review monitoring metrics
- Optimize data loading with prefetching

---

**Last updated**: 2025-11-16
**System version**: v2.0.0
