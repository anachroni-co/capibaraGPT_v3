# CapibaraGPT v3 - Core Modules Export

This directory contains the recovered and documented core modules for CapibaraGPT v3.
All code is documented in English with comprehensive docstrings and type hints.

## Directory Structure

```
v3_export/
├── core/
│   ├── ssm/                    # State Space Models
│   │   ├── ssm_tpu.py          # Classic SSM optimized for TPU
│   │   ├── spike_ssm.py        # Bio-inspired Spiking SSM
│   │   └── __init__.py
│   │
│   ├── routers/                # Routing Mechanisms
│   │   ├── self_modifying_router.py  # Nested Learning Router
│   │   └── __init__.py
│   │
│   ├── optimizations/          # Hardware Optimizations
│   │   ├── tpu_v6e.py          # TPU v6e-64 specific optimizations
│   │   └── __init__.py
│   │
│   ├── vq/                     # Vector Quantization
│   │   ├── vq_arm_axion.py     # ARM Axion VQ optimizer
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── layers/                     # Neural Network Layers (future)
├── training/                   # Training Systems (future)
├── inference/                  # Inference Engines (future)
├── utils/                      # Utilities (future)
└── README.md
```

## Module Descriptions

### SSM (State Space Models)

- **SSMBlock**: Classic state space model with O(n) complexity
  - TPU-optimized using `jax.lax.scan`
  - Stable initialization for state transition matrix
  - Support for bidirectional processing

- **SpikeSSM**: Bio-inspired spiking neural network SSM
  - Surrogate gradient learning for spike discontinuities
  - Leaky integrate-and-fire neuron dynamics
  - Energy-efficient sparse activations

- **AdaptiveSpikeSSM**: Advanced spiking SSM with:
  - Adaptive thresholds based on activity
  - Multiple decay timescales
  - Lateral inhibition for competition

### Routers

- **SelfModifyingRouter**: Nested Learning router (NeurIPS 2025 inspired)
  - Two-level optimization:
    - Level 1: Routing decisions (fast, every step)
    - Level 2: Meta-routing policy (slow, every 100 steps)
  - Self-modification of temperature, top-k, exploration rate
  - Automatic adaptation to changing conditions

### Optimizations

- **TPUv6eOptimizer**: Google Cloud TPU v6e-64 optimizations
  - Optimized attention with einsum operations
  - Mamba SSM optimized for TPU scan operations
  - Mixed precision (bfloat16) support
  - Distributed training with PMAP

### Vector Quantization

- **ARMAxionOptimizer**: ARM processor optimizations
  - SVE2 vectorization support (512-bit)
  - Memory-efficient chunked attention
  - Efficient VQ codebook lookups
  - Portable fallback for non-ARM systems

## Installation

Copy these modules to your CapibaraGPT v3 project:

```bash
# From v2 repository root
cp -r v3_export/* /path/to/capibaraGPT_v3/capibara/
```

## Dependencies

```
jax>=0.4.0
flax>=0.7.0
optax>=0.1.5
torch>=2.0.0  # Optional, for ARM optimizations
numpy>=1.24.0
```

## Usage Example

```python
# SSM usage
from core.ssm import SSMBlock, SpikeSSM

ssm = SSMBlock(hidden_size=256, state_dim=64)
spike_ssm = SpikeSSM(hidden_size=256, threshold=0.5)

# Router usage
from core.routers import create_self_modifying_router

router = create_self_modifying_router()
routes, metadata = router.route(inputs)
router.update_with_feedback(routes, performance=0.85, step=100)

# TPU optimization
from core.optimizations import TPUv6eOptimizer

optimizer = TPUv6eOptimizer()
attention = optimizer.create_optimized_attention(1024, 16, 2048)

# ARM VQ optimization
from core.vq import create_arm_axion_optimizer

vq_opt = create_arm_axion_optimizer()
quantized, indices = vq_opt.optimize_vq_forward(x, codebook)
```

## References

- Gu, A., Goel, K., & Re, C. (2021). Efficiently Modeling Long Sequences with Structured State Spaces. arXiv:2111.00396
- Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks. IEEE Signal Processing Magazine.
- Behrouz, A.; Razaviyayn, M.; Mirrokni, V.; Zhong, P. (2025). Nested Learning: The Illusion of Deep Learning Architectures. NeurIPS 2025.

## License

MIT License - See main repository for details.

## Example quick

Example (pseudo-code) de uso del paquete principal:

```python
from capibara.cli import main

# Simula el arranque del CLI principal
main()
```
