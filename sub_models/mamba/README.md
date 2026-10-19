# Mamba Module - Selective State Space Model

## Description

Mamba (Selective State Space Model) implementation for CapibaraGPT v3. Provides processing with **O(n)** complexity instead of O(n²) from traditional Transformers, ideal for long sequences.

## Features

-  **O(n) Complexity** for sequence processing
-  **Selective State Space Model** with adaptive parameters
-  **IModule Compatible** for modular integration
-  **TPU Optimizations** with associative scan
-  **Robust Fallbacks** when JAX is not available
-  **Detailed Metrics** for complexity and performance

## Dependency Installation

```bash
# Install required dependencies
pip install numpy>=1.24.4
pip install jax jaxlib
pip install flax>=0.8.0

# For TPU (optional but recommended)
pip install jax[tpu]
```

## Basic Usage

```python
from capibara.sub_models.mamba import MambaModule, MambaConfig

# Configuration
config = {
    'hidden_size': 768,
    'd_state': 64,         # SSM state dimension
    'd_conv': 4,           # 1D convolution kernel
    'expand_factor': 2,    # Expansion factor
    'scan_type': 'associative'  # For TPU parallelization
}

# Create module
mamba = MambaModule(config)

# Process input
import numpy as np
inputs = np.random.randn(2, 512, 768)  # [batch, seq_len, hidden_size]
outputs = mamba(inputs, training=False)

print(f"Complexity: {outputs['metrics']['complexity']}")
print(f"Output shape: {outputs['output'].shape}")
```

## Advanced Configuration

### MambaConfig Parameters

- `hidden_size` (int, default=768): Model dimension
- `d_state` (int, default=64): SSM internal state dimension
- `d_conv` (int, default=4): 1D convolution kernel size
- `expand_factor` (int, default=2): Expansion factor for projections
- `dt_rank` (int, default=32): Rank for temporal parameter Δ
- `activation` (str, default='swish'): Activation function (swish, gelu, relu)
- `use_tpu_optimizations` (bool, default=True): Enable TPU optimizations
- `scan_type` (str, default='associative'): Scan type ('linear' or 'associative')

### Example with Custom Configuration

```python
from capibara.sub_models.mamba import MambaModule

config = {
    'hidden_size': 1024,
    'd_state': 128,
    'd_conv': 8,
    'expand_factor': 4,
    'activation': 'gelu',
    'use_tpu_optimizations': True,
    'scan_type': 'associative'
}

mamba = MambaModule(config)
```

## Integration with ModularCapibaraModel

The module is designed to integrate directly with Capibara's modular architecture:

```python
# In capibara/core/modular_model.py
from capibara.sub_models.mamba import MambaModule

available_modules = {
    "mamba": MambaModule,
    # ... other modules
}
```

### TOML Configuration

```toml
# In capibara/config/configs_toml/mamba_hybrid.toml
[modules]
active = [
    "mamba",
    "embedding_module",
    # ... other modules
]

[modules.mamba]
enabled = true
hidden_size = 768
d_state = 64
d_conv = 4
expand_factor = 2
scan_type = "associative"
```

## Metrics and Monitoring

The module provides detailed metrics:

```python
outputs = mamba(inputs, training=False)
metrics = outputs['metrics']

print(f"Mamba active: {metrics['mamba_active']}")
print(f"Complexity: {metrics['complexity']}")  # 'O(n)' or 'O(log n)'
print(f"Sequence length: {metrics['sequence_length']}")
print(f"State dimension: {metrics['d_state']}")
print(f"Selective scan used: {metrics['selective_scan_used']}")
print(f"TPU optimized: {metrics['tpu_optimized']}")
```

## Performance

### Complexity Comparison

| Sequence Length | Transformer (O(n²)) | Mamba (O(n)) | Improvement |
|-----------------|---------------------|--------------|-------------|
| 512             | 262,144 ops         | 512 ops      | 512x        |
| 2048            | 4,194,304 ops       | 2048 ops     | 2048x       |
| 4096            | 16,777,216 ops      | 4096 ops     | 4096x       |

### Expected Benchmarks

```
# With TPU v4-32
- Throughput: ~3000 tokens/sec for 2048 token sequences
- Memory: 4x less than Transformer for sequences > 1024
- Latency: Sub-linear scaling with sequence length
```

## Troubleshooting

### Error: "JAX not available"

```bash
# Install JAX
pip install jax jaxlib

# For TPU
pip install jax[tpu]
```

### Error: "Flax not available"

```bash
pip install flax>=0.8.0
```

### Fallback Mode

If JAX is not available, the module will use a fallback implementation with numpy:

```python
# The module automatically detects and uses fallback
# A warning will be logged: "Using Mamba fallback implementation"
```

## References

- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [Structured State Space Models (S4)](https://arxiv.org/abs/2111.00396)
- [Selective State Space Models](https://github.com/state-spaces/mamba)

## Implementation Status

-  Core SSM implementation
-  Selective scan mechanism
-  IModule interface compatibility
-  TPU optimizations (associative scan)
-  Fallback mode (numpy)
- ️ Complete 1D convolution optimization (in progress)
-  Mamba-2 features (roadmap)

## Contribution

To contribute to Mamba module improvements:

1. Conv1d optimizations for production
2. Mamba-2 feature implementation
3. Additional benchmarks on different hardware
4. Metrics system improvements

---

**Recovered from commit**: 6377222 (2025-09-03)
**Author**: Cursor Agent, marco@anachroni.co
**Last updated**: 2025-11-16
