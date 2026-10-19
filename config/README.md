# Configuration Module

This module manages all CapibaraGPT system configurations, including model, training, optimization, and deployment configurations.

##  Main Components

### ConfigManager (`config_manager.py`)
Central manager for loading and handling TOML configurations.

```python
from capibara.config import ConfigManager

# Initialize manager
config_manager = ConfigManager("config")

# Load configuration
config = config_manager.load_config("model")

# Get specific values
hidden_size = config_manager.get_value("model", "model.hidden_size", default=768)

# Validate configuration
is_valid = config_manager.validate_config("model", schema)
```

### AdaptiveConfig (`adaptive_config.py`)
Configuration for adaptive computation and expert routing.

```python
from capibara.config import AdaptiveConfig

# Create adaptive configuration
config = AdaptiveConfig(
    hidden_size=1024,
    num_experts=16,
    routing_type='top_k',
    adaptive_routing=True,
    device='tpu',
    precision='bfloat16'
)

# Load from JSON
config = AdaptiveConfig.from_json("adaptive_config.json")

# Automatic validation in __post_init__
```

### Specific Configurations

#### ModelConfig (`config_schema.py`)
Defines model architecture with Pydantic validation.

```python
from capibara.config import ModelConfig

model_config = ModelConfig(
    hidden_size=768,
    seq_len=2048,
    num_layers=12,
    num_heads=12,
    dropout_rate=0.1,
    use_mixture=True,
    use_bitnet_quantizer=True,
    bit_width=8
)
```

#### TrainingConfig (`training_config.py`)
Training parameter configuration.

```python
from capibara.config import TrainingConfig

training_config = TrainingConfig(
    train_data_path="data/train.jsonl",
    val_data_path="data/val.jsonl",
    batch_size=32,
    learning_rate=0.001,
    num_epochs=10,
    vocab_size=32000
)
```

#### MemoryConfig (`memory_config.py`)
Memory and optimization configuration management.

```python
from capibara.config import MemoryConfig

memory_config = MemoryConfig(
    enable_gradient_checkpointing=True,
    max_memory_gb=32,
    offload_to_cpu=False,
    memory_efficient_attention=True
)
```

##  Specialized Configurations

### Chain of Thought (`cot_config.py`)
Configuration for step-by-step reasoning.

```python
from capibara.config import CoTConfig

cot_config = CoTConfig(
    enable_cot=True,
    max_reasoning_steps=8,
    reasoning_temperature=0.7,
    step_validation=True
)
```

### Convexity (`convexity_config.py`)
Configuration for convex optimization.

```python
from capibara.config import ConvexityConfig

convexity_config = ConvexityConfig(
    enable_convex_optimization=True,
    constraint_tolerance=1e-6,
    max_iterations=1000
)
```

### Multimodal Training (Config Example)
Configuration example for enabling audio/video encoders in a training setup.

```python
multimodal_training_config = {
    "modalities": ["text", "image", "audio", "video"],
    "encoders": {
        "audio": {
            "sample_rate": 16000,
            "n_fft": 512,
            "output_dim": 256
        },
        "video": {
            "width": 224,
            "height": 224,
            "fps": 30,
            "output_dim": 512
        }
    },
    "fusion": {
        "weights": {"text": 0.35, "image": 0.25, "audio": 0.2, "video": 0.2}
    }
}
```

### Scaling (unified)
Distributed scaling and parallelism are managed from `unified_model_config.py` (e.g., `MemoryOptimizationConfig`, submeshes, and `ModularModelConfig`).

##  Directory Structure

```
config/
├── configs_toml/          # TOML configuration files
├── conversion/            # Conversion utilities
├── __init__.py           # Main exports
├── adaptive_config.py    # Adaptive configuration
├── config_manager.py     # Central manager
├── config_schema.py      # Pydantic schemas
├── config_settings.py   # Generatel settings
├── config_validator.py   # Validators
├── config_validators.py  # Additional validators
├── convexity_config.py   # Convex configuration
├── cot_config.py         # Chain of Thought
├── memory_config.py      # Memory configuration
├── model_config.py       # Compatibility wrapper (re-exports unified)
├── config_types.py       # Compatibility wrapper (re-exports types)
├── config_semiotic.py    # Compatibility stub
├── training_config.py    # Training configuration
└── unified_model_config.py # Unified configuration
```

##  Advanced Usage

### Complete Configuration
```python
from capibara.config import CapibaraConfig

# Create complete configuration
config = CapibaraConfig(
    model=ModelConfig(...),
    training=TrainingConfig(...),
    pruning=PruningConfig(...),
    wandb=WandbConfig(...),
    modules=ModulesConfig(...),
    paths=PathsConfig(...)
)

# Load from YAML
config = CapibaraConfig.from_yaml("config.yaml")

# Validate configuration
warnings = config.validate()
if warnings:
    for warning in warnings:
        print(f"Warning: {warning}")

# Convert to dictionary
config_dict = config.to_dict()
```

### Custom Validation
```python
from capibara.config.config_validators import (
    estimate_model_memory,
    validate_device_compatibility,
    check_data_paths
)

# Estimate model memory
memory_gb = estimate_model_memory(config_dict) / 1e9
print(f"Estimated memory: {memory_gb:.2f} GB")

# Validate device compatibility
is_compatible = validate_device_compatibility(config.device)

# Verify data paths
valid_paths = check_data_paths(config.training)
```

## ️ Key Features

- **Automatic validation**: Using Pydantic for type and value validation
- **Flexible loading**: Support for TOML, JSON, and YAML
- **Hierarchical configuration**: Access to nested values with dot notation
- **Memory management**: Automatic estimation of required resources
- **Specialized configurations**: For different system components
- **Cross-validation**: Compatibility verification between modules
- **Hot reload**: Runtime configuration reloading

##  Available Validators

- `ModelConfig`: Validates model architecture and dimension compatibility
- `TrainingConfig`: Verifies data paths and training parameters
- `MemoryConfig`: Estimates memory usage and hardware compatibility
- `AdaptiveConfig`: Validates adaptive routing configurations
- `ConvexityConfig`: Verifies convex optimization parameters

##  Configuration Examples

See the `configs_toml/` directory for complete configuration examples in TOML format.

## Example quick

Example (pseudo-code) para cargar una configuration TOML:

```python
import toml

config = toml.load("config/configs_toml/default.toml")
print(config["model"]["hidden_size"])
```
