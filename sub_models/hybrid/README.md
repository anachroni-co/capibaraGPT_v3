# Hybrid Attention Module - Intelligent Routing

## Description

Intelligent hybrid module that automatically selects between Transformer (O(n²)) and Mamba (O(n)) based on input characteristics. Provides the best balance between precision and efficiency.

## Features

-  **Intelligent Routing** automatic between Mamba and Transformer
-  **Configurable Threshold** for routing decisions
-  **Detailed Decision Metrics**
-  **Intelligent Caching** of decisions
-  **Reason Logging** for debugging
-  **IModule Compatible**

## Decision Logic

```python
if sequence_length >= mamba_threshold:  # Default: 512
    use_mamba = True  # O(n) for efficiency
    reason = "long_sequence_efficiency"
else:
    use_transformer = True  # O(n²) for precision
    reason = "short_sequence_precision"
```

## Dependency Installation

```bash
# Required dependencies
pip install numpy>=1.24.4
pip install jax jaxlib
pip install flax>=0.8.0

# For TPU (recommended)
pip install jax[tpu]
```

## Basic Usage

```python
from capibara.sub_models.hybrid import HybridAttentionModule, HybridConfig

# Configuration
config = {
    'hidden_size': 768,
    'num_heads': 12,
    'mamba_threshold': 512,  # Threshold to use Mamba
    'transformer_max_length': 2048,
    'collect_metrics': True,
    'log_decisions': True
}

# Create hybrid module
hybrid = HybridAttentionModule(config)

# Process inputs of different lengths
import numpy as np

# Short sequence (will use Transformer)
short_input = np.random.randn(2, 256, 768)
output_short = hybrid(short_input, training=False)
print(f"Module used: {output_short['metrics']['selected_module']}")  # 'transformer'

# Long sequence (will use Mamba)
long_input = np.random.randn(2, 1024, 768)
output_long = hybrid(long_input, training=False)
print(f"Module used: {output_long['metrics']['selected_module']}")  # 'mamba'
```

## Advanced Configuration

### HybridConfig Parameters

#### Hybrid Decision
- `mamba_threshold` (int, default=512): Minimum length to use Mamba
- `transformer_max_length` (int, default=2048): Maximum length for Transformer

#### Architecture
- `hidden_size` (int, default=768): Model dimension
- `num_heads` (int, default=12): Number of attention heads
- `intermediate_size` (int, default=3072): Intermediate FFN size

#### Mamba Configuration
- `mamba_config` (dict, optional): Custom configuration for MambaModule

#### Transformer
- `dropout_rate` (float, default=0.1): Dropout rate
- `layer_norm_eps` (float, default=1e-12): Epsilon for layer normalization

#### Optimizations
- `use_tpu_optimizations` (bool, default=True): TPU optimizations
- `use_mixed_precision` (bool, default=True): Mixed precision
- `enable_caching` (bool, default=True): Decision caching

#### Metrics and Logging
- `collect_metrics` (bool, default=True): Collect metrics
- `log_decisions` (bool, default=False): Detailed logging

### Advanced Example

```python
from capibara.sub_models.hybrid import HybridAttentionModule

config = {
    'hidden_size': 1024,
    'num_heads': 16,
    'mamba_threshold': 1024,  # Higher threshold
    'transformer_max_length': 4096,

    # Custom Mamba configuration
    'mamba_config': {
        'd_state': 128,
        'd_conv': 8,
        'expand_factor': 4
    },

    # Metrics and debugging
    'collect_metrics': True,
    'log_decisions': True,
    'enable_caching': True
}

hybrid = HybridAttentionModule(config)
```

## Metrics and Monitoring

### Available Metrics

```python
outputs = hybrid(inputs, training=False)
metrics = outputs['metrics']

print(f"Selected module: {metrics['selected_module']}")  # 'mamba' or 'transformer'
print(f"Reason: {metrics['selection_reason']}")
print(f"Complexity: {metrics['complexity']}")  # 'O(n)' or 'O(n²)'
print(f"Sequence length: {metrics['sequence_length']}")
print(f"Threshold used: {metrics['mamba_threshold']}")
print(f"Decision confidence: {metrics['decision_confidence']}")

# Accumulated statistics
stats = metrics['routing_statistics']
print(f"Total decisions: {stats['total_decisions']}")
print(f"Mamba count: {stats['mamba_count']}")
print(f"Transformer count: {stats['transformer_count']}")
```

### Decision Caching

```python
# The module caches decisions for similar sequences
cache_stats = metrics['cache_statistics']
print(f"Cache size: {cache_stats['cache_size']}")
print(f"Cache hits: {cache_stats['cache_hits']}")
print(f"Cache misses: {cache_stats['cache_misses']}")
print(f"Hit rate: {cache_stats['hit_rate']:.2%}")
```

## Integration with ModularCapibaraModel

```python
# In capibara/core/modular_model.py
from capibara.sub_models.hybrid import HybridAttentionModule

available_modules = {
    "hybrid_attention": HybridAttentionModule,
    # ... other modules
}
```

### TOML Configuration

```toml
# capibara/config/configs_toml/mamba_hybrid.toml
[modules]
active = [
    "core_transformer",
    "mamba",
    "hybrid_attention",  # ← Intelligent routing
    "embedding_module"
]

[modules.hybrid_attention]
enabled = true
hidden_size = 768
num_heads = 12
mamba_threshold = 512
transformer_max_length = 2048
collect_metrics = true
log_decisions = false
enable_caching = true
```

## Use Cases

### 1. Mixed Processing

```python
# Batch with sequences of different lengths
# The module automatically uses the optimal strategy for each

batch = {
    'short_docs': np.random.randn(4, 128, 768),   # Transformer
    'medium_docs': np.random.randn(4, 512, 768),  # Hybrid/Mamba
    'long_docs': np.random.randn(4, 2048, 768)    # Mamba
}

for name, inputs in batch.items():
    outputs = hybrid(inputs, training=False)
    print(f"{name}: {outputs['metrics']['selected_module']}")
```

### 2. Resource Optimization

```python
# Configure threshold dynamically based on available resources
import psutil

available_memory_gb = psutil.virtual_memory().available / (1024**3)

if available_memory_gb < 8:
    threshold = 256  # Use Mamba earlier to save memory
else:
    threshold = 1024  # Use Transformer longer

config['mamba_threshold'] = threshold
hybrid = HybridAttentionModule(config)
```

### 3. A/B Testing

```python
# Compare performance of different thresholds
thresholds = [256, 512, 1024, 2048]
results = {}

for threshold in thresholds:
    config['mamba_threshold'] = threshold
    hybrid = HybridAttentionModule(config)

    # Process test dataset
    outputs = hybrid(test_data, training=False)

    results[threshold] = {
        'quality': outputs['metrics']['quality_score'],
        'latency': outputs['metrics']['processing_time_ms'],
        'mamba_usage': outputs['metrics']['routing_statistics']['mamba_count']
    }
```

## Benefits

### Adaptive Performance

| Sequence Length | Module Used | Complexity | Memory |
|-----------------|-------------|------------|--------|
| < 512           | Transformer | O(n²)      | Moderate |
| 512-2048        | Mamba       | O(n)       | Low |
| > 2048          | Mamba       | O(n)       | Very Low |

### Advantages

-  **Best of both worlds**: Transformer precision + Mamba efficiency
-  **Automatic**: No manual configuration per input
-  **Adaptive**: Adjusts to data characteristics
-  **Efficient**: Automatically optimizes resources
-  **Transparent**: Detailed decision metrics

## Troubleshooting

### Problem: "Always uses Transformer"

**Solution**: Reduce `mamba_threshold`

```python
config['mamba_threshold'] = 256  # Lower value
```

### Problem: "Quality degraded with Mamba"

**Solution**: Increase threshold or adjust Mamba configuration

```python
config['mamba_threshold'] = 1024  # Use Transformer longer

# Or improve Mamba configuration
config['mamba_config'] = {
    'd_state': 128,  # Higher capacity
    'expand_factor': 4  # More expressiveness
}
```

### Problem: "High memory usage"

**Solution**: Reduce threshold to use Mamba earlier

```python
config['mamba_threshold'] = 128
config['transformer_max_length'] = 512
```

## References

- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)
- [Hybrid Architectures for LLMs](https://arxiv.org/abs/2401.00000)

## Implementation Status

-  Basic intelligent routing
-  Metrics and monitoring
-  Decision caching
-  IModule compatibility
- ️ Content-based routing (in progress)
-  Adaptive thresholds (roadmap)
-  Multi-dimensional routing (roadmap)

---

**Recovered from commit**: 6377222 (2025-09-03)
**Author**: Cursor Agent, marco@anachroni.co
**Last updated**: 2025-11-16
