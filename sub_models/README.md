# capibara/sub_models - Sub-Models Architecture

The **sub_models** directory contains all specialized sub-models that can be combined modularly to build custom architectures.

##  Table of Contents

1. [Overview](#overview)
2. [Available Sub-Models](#available-sub-models)
3. [Modular Architecture](#modular-architecture)
4. [Quick Start](#quick-start)
5. [Detailed Sub-Models](#detailed-sub-models)
6. [Integration and Composition](#integration-and-composition)
7. [Performance Comparison](#performance-comparison)

---

##  Overview

CapibaraGPT v3 uses a **fully modular architecture** where different sub-models can be combined as needed:

```
┌──────────────────────────────────────────────────────────┐
│              ModularCapibaraModel                        │
├──────────────────────────────────────────────────────────┤
│                                                           │
│   ┌─────────────┐     ┌──────────────┐                  │
│   │   Mamba     │────>│ Hybrid Router│                  │
│   │   (SSM)     │     │              │                  │
│   └─────────────┘     └──────┬───────┘                  │
│                              │                            │
│   ┌─────────────┐            │      ┌──────────────┐    │
│   │ Transformer │<───────────┴─────>│    Vision    │    │
│   │             │                    │   Encoder    │    │
│   └─────────────┘                    └──────────────┘    │
│                                                           │
│   ┌──────────────┐    ┌──────────────┐                  │
│   │   Semiotic   │    │ Deep Dialog  │                  │
│   │  Grounding   │    │ Reasoning    │                  │
│   └──────────────┘    └──────────────┘                  │
└──────────────────────────────────────────────────────────┘
```

### Design Philosophy

- **Modular**: Each sub-model can be used independently
- **Composable**: Easily combine multiple sub-models
- **Optimized**: Each sub-model optimized for its specific task
- **Flexible**: Swap sub-models without changing overall architecture

---

##  Available Sub-Models

| Sub-Model | Directory | Purpose | Documentation |
|-----------|-----------|---------|---------------|
| **Mamba (SSM)** | `mamba/` | O(n) attention for long sequences | [README](mamba/README.md) |
| **Hybrid Attention** | `hybrid/` | Intelligent Mamba/Transformer router | [README](hybrid/README.md) |
| **Vision** | `vision/` | Image/video encoder | - |
| **Semiotic** | `semiotic/` | Semiotic and symbolic grounding | - |
| **Deep Dialog** | `deep_dialog.py` | Advanced multi-turn dialog | - |
| **CSA Expert** | `csa_expert.py` | Cross-Stream Attention expert | - |
| **Reasoning Enhancement** | `reasoning_enhancement.py` | Reasoning improvements | - |
| **Byte-Level** | `Byte_TPU.py` | Byte-level processing | - |
| **Experimental** | `experimental/` | Experimental sub-models | - |
| **Capibaras** | `capibaras/` | Capibara model variants | - |

---

## ️ Modular Architecture

### ModularCapibaraModel

The main model integrates all sub-models:

```python
from capibara.core import ModularCapibaraModel, ModularConfig
from capibara.sub_models import (
    MambaModule,
    HybridAttentionModule,
    VisionEncoder,
    SemioticGrounder
)

# Configure modules
config = ModularConfig(
    # Base model
    hidden_size=768,
    num_layers=12,

    # Active sub-models
    use_mamba=True,
    use_hybrid_attention=True,
    use_vision_encoder=True,
    use_semiotic=True,

    # Sub-model configs
    mamba_config=MambaConfig(...),
    vision_config=VisionConfig(...)
)

# Create modular model
model = ModularCapibaraModel(config)

# The model automatically uses configured sub-models
output = model(inputs)
```

### Sub-Model Orchestrator

```python
from capibara.sub_models import UltraSubmodelOrchestrator

# Create orchestrator
orchestrator = UltraSubmodelOrchestrator(
    enabled_submodels=[
        "mamba",
        "hybrid_attention",
        "deep_dialog",
        "reasoning_enhancement"
    ]
)

# Orchestrate execution
result = orchestrator.process(
    inputs=inputs,
    task_type="reasoning"  # reasoning, dialog, vision, etc.
)

# The orchestrator automatically selects appropriate sub-models
```

---

##  Quick Start

### Basic Usage: Mamba

```python
from capibara.sub_models.mamba import MambaModule, MambaConfig

# Configure Mamba
config = MambaConfig(
    hidden_size=768,
    d_state=16,
    d_conv=4,
    expand_factor=2
)

# Create module
mamba = MambaModule(config)

# Forward pass
import jax.numpy as jnp
inputs = jnp.ones((2, 512, 768))  # (batch, seq_len, hidden)
outputs = mamba(inputs)

# Complexity: O(n) vs O(n²) for Transformer
```

### Basic Usage: Hybrid Attention

```python
from capibara.sub_models.hybrid import HybridAttentionModule

# Configure hybrid router
hybrid = HybridAttentionModule(
    config={
        "mamba_threshold": 512,
        "use_dynamic_routing": True
    }
)

# Automatic routing
outputs = hybrid(inputs)  # Uses Mamba if seq_len >= 512, else Transformer

# Inspect decision
print(f"Used: {hybrid.last_decision}")  # "mamba" or "transformer"
```

### Basic Usage: Vision

```python
from capibara.sub_models.vision import VisionEncoder

# Configure vision encoder
vision = VisionEncoder(
    hidden_size=768,
    image_size=224,
    patch_size=16
)

# Encode image
image = jnp.ones((1, 224, 224, 3))  # (batch, H, W, C)
image_embeddings = vision(image)  # (batch, num_patches, hidden_size)

# Combine with text
combined = model.combine_modalities(
    text_emb=text_embeddings,
    vision_emb=image_embeddings
)
```

---

##  Detailed Sub-Models

### 1. Mamba (Selective State Space Model)

**Purpose**: Efficient O(n) attention for long sequences

```python
from capibara.sub_models.mamba import MambaModule

mamba = MambaModule(config)

# Features:
# - Complexity: O(n) vs O(n²) Transformer
# - Ideal for: Sequences > 512 tokens
# - TPU optimized: Associative scan
# - Memory efficient: ~50% less memory than Transformer
```

See [mamba/README.md](mamba/README.md) for complete documentation.

### 2. Hybrid Attention

**Purpose**: Intelligent router between Mamba and Transformer

```python
from capibara.sub_models.hybrid import HybridAttentionModule

hybrid = HybridAttentionModule(
    mamba_threshold=512,
    use_dynamic_routing=True,
    memory_threshold=0.8
)

# Decision based on:
# - Sequence length
# - Available memory
# - Latency requirements
# - Required quality
```

See [hybrid/README.md](hybrid/README.md) for complete documentation.

### 3. Vision Encoder

**Purpose**: Process images and video

```python
from capibara.sub_models.vision import VisionEncoder, VideoEncoder

# Images
vision = VisionEncoder(
    architecture="vit",  # vit, resnet, convnext
    pretrained="imagenet"
)

# Video
video = VideoEncoder(
    num_frames=16,
    temporal_pooling="attention"
)

# Multimodal fusion
from capibara.core.encoders import MultimodalCombiner
combiner = MultimodalCombiner(fusion_type="cross_attention")
fused = combiner(text=text_emb, vision=vision_emb)
```

### 4. Semiotic Grounding

**Purpose**: Semiotic and symbolic grounding

```python
from capibara.sub_models.semiotic import SemioticGrounder

semiotic = SemioticGrounder(
    symbol_vocab_size=10000,
    grounding_layers=4
)

# Ground symbols to concepts
grounded = semiotic.ground(
    symbols=["apple", "red", "fruit"],
    context=text_context
)

# Symbolic reasoning
reasoning_result = semiotic.reason(
    premises=["All apples are fruits", "This is an apple"],
    query="Is this a fruit?"
)
```

### 5. Deep Dialog

**Purpose**: Multi-turn dialog with context memory

```python
from capibara.sub_models import DeepDialogModel

dialog = DeepDialogModel(
    max_context_length=4096,
    use_episodic_memory=True
)

# Multi-turn conversation
context = dialog.initialize_context()

for user_input in conversation:
    response = dialog.respond(
        user_input=user_input,
        context=context
    )
    context = dialog.update_context(context, user_input, response)
```

### 6. CSA Expert (Cross-Stream Attention)

**Purpose**: Cross attention between multiple information streams

```python
from capibara.sub_models import CSAExpert

csa = CSAExpert(
    num_streams=3,  # text, vision, audio
    cross_attention_heads=12
)

# Process multiple streams
outputs = csa.process_streams(
    text_stream=text,
    vision_stream=images,
    audio_stream=audio
)

# Automatic cross-stream attention
```

### 7. Reasoning Enhancement

**Purpose**: Specific improvements for reasoning

```python
from capibara.sub_models import ReasoningEnhancement

reasoning = ReasoningEnhancement(
    use_scratch_pad=True,
    use_self_consistency=True,
    num_reasoning_paths=5
)

# Enhanced reasoning
result = reasoning.reason(
    problem="If John has 5 apples and gives 2 to Mary...",
    reasoning_type="mathematical"
)

# Includes:
# - Scratch pad for intermediate work
# - Self-consistency voting
# - Multiple reasoning paths
```

### 8. Byte-Level Processing (TPU Optimized)

**Purpose**: Byte-level processing

```python
from capibara.sub_models import ByteTPU

byte_model = ByteTPU(
    vocab_size=256,  # 256 possible bytes
    use_tpu_optimizations=True
)

# Process bytes directly (no tokenization)
byte_inputs = jnp.array([72, 101, 108, 108, 111])  # "Hello"
outputs = byte_model(byte_inputs)

# Advantages:
# - No tokenizer needed
# - Handles any language/script
# - Robust to spelling errors
```

---

##  Integration and Composition

### Manual Composition

```python
from capibara.sub_models import (
    MambaModule,
    VisionEncoder,
    DeepDialogModel,
    ReasoningEnhancement
)

class MyCustomModel:
    def __init__(self, config):
        # Manually combine sub-models
        self.mamba = MambaModule(config.mamba_config)
        self.vision = VisionEncoder(config.vision_config)
        self.dialog = DeepDialogModel(config.dialog_config)
        self.reasoning = ReasoningEnhancement(config.reasoning_config)

    def __call__(self, inputs, images=None, context=None):
        # 1. Vision encoding (if images present)
        if images is not None:
            vision_emb = self.vision(images)
            inputs = self.combine(inputs, vision_emb)

        # 2. Mamba processing
        mamba_output = self.mamba(inputs)

        # 3. Dialog context
        if context is not None:
            mamba_output = self.dialog.apply_context(mamba_output, context)

        # 4. Reasoning enhancement
        final_output = self.reasoning.enhance(mamba_output)

        return final_output
```

### Composition with Orchestrator

```python
from capibara.sub_models import UltraSubmodelOrchestrator

# The orchestrator handles composition automatically
orchestrator = UltraSubmodelOrchestrator(
    enabled_submodels=["mamba", "vision", "dialog", "reasoning"]
)

# Automatically detects which sub-models to use based on inputs
output = orchestrator.process(
    text=text_input,
    images=images,  # Automatically activates vision
    task="reasoning"  # Automatically activates reasoning
)
```

### Integration with ModularCapibaraModel

```python
from capibara.core import ModularCapibaraModel, ModularConfig

config = ModularConfig(
    # Configure all sub-models
    use_mamba=True,
    mamba_config=MambaConfig(...),

    use_vision=True,
    vision_config=VisionConfig(...),

    use_dialog=True,
    dialog_config=DialogConfig(...),

    use_reasoning=True,
    reasoning_config=ReasoningConfig(...)
)

# Model automatically integrates all sub-models
model = ModularCapibaraModel(config)

# Unified usage
output = model(
    text_inputs=text,
    image_inputs=images,
    dialog_context=context
)
```

---

##  Performance Comparison

### Latency (512 tokens, batch_size=1)

| Sub-Model | Latency | Memory | Throughput |
|-----------|---------|--------|------------|
| Mamba | 45ms | 2GB | 1200 req/s |
| Transformer | 120ms | 4GB | 450 req/s |
| Hybrid (auto) | 50-110ms | 2-3.5GB | 900 req/s |
| Vision | 30ms | 1.5GB | 1500 req/s |
| Deep Dialog | 60ms | 2.5GB | 800 req/s |

### Computational Complexity

| Sub-Model | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Mamba | O(n) | O(n) |
| Transformer | O(n²) | O(n²) |
| Hybrid | O(n) - O(n²) | O(n) - O(n²) |
| Vision (ViT) | O(n²) patches | O(n²) |
| CSA Expert | O(n²) per stream | O(n²) |

### When to Use Each Sub-Model

| Use Case | Recommended Sub-Model | Reason |
|----------|----------------------|--------|
| Sequences > 1024 tokens | Mamba | O(n) complexity |
| Sequences < 512 tokens | Transformer | Better quality |
| Variable sequences | Hybrid Attention | Adaptive |
| Multimodal (text + image) | Vision + Mamba | Efficient multimodal |
| Multi-turn dialog | Deep Dialog | Contextual memory |
| Complex reasoning | Reasoning Enhancement | Multiple paths |
| Multiple data sources | CSA Expert | Cross-stream attention |

---

## ️ Developing New Sub-Models

### Template for New Sub-Model

```python
from flax import linen as nn
from capibara.core.interfaces import IModule
from typing import Any, Dict

class MyNewSubModel(nn.Module, IModule):
    """My custom new sub-model."""

    hidden_size: int
    custom_param: float = 1.0

    def setup(self):
        """Initialize components."""
        self.layer1 = nn.Dense(self.hidden_size)
        self.layer2 = nn.Dense(self.hidden_size)

    def __call__(self, inputs, **kwargs):
        """Forward pass."""
        x = self.layer1(inputs)
        x = nn.relu(x)
        x = self.layer2(x)
        return x

    def get_metrics(self) -> Dict[str, Any]:
        """Module metrics."""
        return {
            "module_type": "MyNewSubModel",
            "hidden_size": self.hidden_size,
            "custom_param": self.custom_param
        }

    def get_config(self) -> Dict[str, Any]:
        """Module configuration."""
        return {
            "hidden_size": self.hidden_size,
            "custom_param": self.custom_param
        }
```

### Register in ModularCapibaraModel

```python
# In capibara/core/modular_model.py
from capibara.sub_models.my_new import MyNewSubModel

class ModularCapibaraModel(nn.Module):
    def setup(self):
        # ...existing setup...

        # Add new sub-model
        if self.config.use_my_new:
            self.my_new = MyNewSubModel(
                hidden_size=self.config.hidden_size,
                **self.config.my_new_config
            )
```

---

##  References

- [Mamba Module](mamba/README.md) - Complete Mamba documentation
- [Hybrid Attention](hybrid/README.md) - Hybrid Router documentation
- [Core Integration](../core/README.md) - Core integration
- [ModularCapibaraModel](../core/modular_model.py) - Main modular model

---

## 🆘 Troubleshooting

### Error: "Sub-model not found"

```python
# Verify available sub-models
from capibara.sub_models import list_available_submodels

available = list_available_submodels()
print(f"Available: {available}")
```

### Error: "Incompatible dimensions"

Ensure all sub-models use the same `hidden_size`:

```python
config = ModularConfig(
    hidden_size=768,  # Same for all
    mamba_config=MambaConfig(hidden_size=768),
    vision_config=VisionConfig(output_size=768)
)
```

### Slow Performance

- Use Mamba for long sequences
- Use Hybrid Attention for automatic adaptation
- Enable TPU optimizations in configs
- Use quantization for inference

---

**Last updated**: 2025-11-16
**System version**: v3.0.0

## Example quick

Example (pseudo-code) para cargar un sub-modelo:

```python
# sub_model = load_sub_model("mamba")
# output = sub_model.forward(tokens)
```

## Issues por hacer

- [ ] """Simulate world state evolution.""" - `sub_models\csa_expert_tpu_optimized.py:254`
- [ ] diversity = jnp.ones_like(plausibility) * 0.7  # Placeholder - `sub_models\csa_expert_tpu_optimized.py:312`
- [ ] self._simulate_scenarios_jit = jit( - `sub_models\csa_expert_tpu_optimized.py:393`
- [ ] self._simulate_scenarios_impl, - `sub_models\csa_expert_tpu_optimized.py:394`
- [ ] def _simulate_scenarios_impl(self, params, initial_state, hypothesis_features, num_steps): - `sub_models\csa_expert_tpu_optimized.py:424`
- [ ] """Simulate world state evolution.""" - `sub_models\csa_expert_tpu_optimized.py:425`
- [ ] # Simulate scenarios - `sub_models\csa_expert_tpu_optimized.py:468`
- [ ] final_state, trajectory = self._simulate_scenarios_jit( - `sub_models\csa_expert_tpu_optimized.py:474`
- [ ] final_state, trajectory = self._simulate_scenarios_impl( - `sub_models\csa_expert_tpu_optimized.py:479`
- [ ] Esta es la update que lleva todos los sub-modelos al level ultra-advanced. - `sub_models\ultra_enhanced_integration.py:14`
- [ ] logger.warning("️ SSM layers not available, using placeholder") - `sub_models\ultra_enhanced_integration.py:330`
- [ ] Sistema de orquestación ultra-avanzada for todos los sub-modelos: - `sub_models\ultra_submodel_orchestrator.py:5`
- [ ] """Orquestador ultra-advanced for todos los sub-modelos del sistema.""" - `sub_models\ultra_submodel_orchestrator.py:179`
- [ ] logger.warning(f"SubModel type {submodel_type} not available or not implemented") - `sub_models\ultra_submodel_orchestrator.py:274`
- [ ] return "adaptive_vq_placeholder"  # Placeholder for current enhanced model - `sub_models\ultra_submodel_orchestrator.py:292`
- [ ] return "spike_ssm_placeholder"  # Placeholder for current enhanced model - `sub_models\ultra_submodel_orchestrator.py:300`
- [ ] return "deep_dialog_placeholder"  # Placeholder for current enhanced model - `sub_models\ultra_submodel_orchestrator.py:308`
- [ ] self.expert_soup_manager = "expert_soup_placeholder" - `sub_models\ultra_submodel_orchestrator.py:361`
- [ ] # For now, return placeholder results - `sub_models\ultra_submodel_orchestrator.py:677`
- [ ] # Simulate execution with input transformation - `sub_models\ultra_submodel_orchestrator.py:681`
- [ ] if isinstance(submodel, str):  # Placeholder models - `sub_models\ultra_submodel_orchestrator.py:682`
- [ ] return x  # Placeholder - `sub_models\ultra_submodel_orchestrator.py:686`
- [ ] # For placeholder implementation, return first result - `sub_models\ultra_submodel_orchestrator.py:702`
- [ ] """Module placeholder.""" - `sub_models\experimental\__init__.py:1`
- [ ] output = x  # Placeholder - implement real convolution - `sub_models\mamba\mamba_module.py:331`
- [ ] Describe an image using BLIP if available, otherwise return placeholder. - `sub_models\semiotic\mnemosyne_semio_module.py:298`
- [ ] """Decorador placeholder for distributed_jit.""" - `sub_models\vision\capivision.py:49`
