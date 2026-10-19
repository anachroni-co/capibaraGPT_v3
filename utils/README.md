# Utils Module

**Core Utility Functions and Tools for CapibaraGPT**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UTILS MODULE ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Core Utilities                                 │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │   │
│  │  │   Config   │  │    Data    │  │   String   │  │    Math    │     │   │
│  │  │   Utils    │  │   Utils    │  │   Utils    │  │   Utils    │     │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Infrastructure                                   │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │   │
│  │  │Checkpoint  │  │   Cache    │  │  Logging   │  │ Monitoring │     │   │
│  │  │  Manager   │  │  Manager   │  │   Utils    │  │   System   │     │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       Validation                                      │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │   │
│  │  │ Validation │  │  Validator │  │    ARM     │  │   Error    │     │   │
│  │  │   Utils    │  │  (Pydantic)│  │ Validator  │  │  Handling  │     │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Overview

The utils module provides essential utility functions, tools, and helpers used throughout CapibaraGPT. These utilities handle common tasks like configuration management, data processing, caching, checkpointing, logging, and validation.

## Module Structure

```
utils/
├── __init__.py                    # Module exports
├── config_utils.py                # Configuration file handling
├── data_utils.py                  # Data manipulation utilities
├── data_processing.py             # Batch data processing
├── string_utils.py                # String manipulation
├── math_utils.py                  # Mathematical utilities
├── format_utils.py                # Formatting utilities
├── checkpoint_manager.py          # Model checkpoint management
├── cache_standalone.py            # TPU-optimized caching
├── logging.py                     # Core logging setup
├── logging_utils.py               # Advanced logging utilities
├── monitoring.py                  # Real-time monitoring
├── system_info.py                 # System information
├── validation_utils.py            # Input validation
├── validator.py                   # Pydantic validators
├── arm_compatibility_validator.py # ARM hardware validation
├── error_handling.py              # Error handling decorators
├── smart_utils_contracts.py       # Smart contract utilities
├── ultra_utils_orchestrator.py    # Advanced orchestration
└── version.py                     # Version information
```

## Quick Start

### Import Utilities

```python
from utils import (
    # Configuration
    load_config_file,
    merge_configs,

    # Data
    flatten_dict,
    unflatten_dict,
    process_data,

    # Caching
    CacheManager,
    create_cache,

    # Checkpoints
    CheckpointManager,

    # Logging
    setup_logger,
    get_logger,

    # Monitoring
    RealTimeMonitor,
    SystemMonitor,

    # Validation
    is_valid_url,
    is_valid_email,
)
```

## Component Documentation

### Configuration Utils

Load and manage configuration files (JSON, YAML).

```python
from utils.config_utils import load_config_file, merge_configs, save_config

# Load configuration
config = load_config_file("config/model.yaml")

# Merge multiple configs (later overrides earlier)
base_config = load_config_file("config/base.yaml")
env_config = load_config_file("config/production.yaml")
final_config = merge_configs(base_config, env_config)

# Save configuration
save_config(final_config, "config/merged.yaml")
```

**Functions:**

| Function | Description |
|----------|-------------|
| `load_config_file(path)` | Load JSON or YAML config |
| `merge_configs(*configs)` | Merge multiple configurations |
| `save_config(config, path)` | Save configuration to file |
| `validate_config(config, schema)` | Validate against schema |

---

### Data Utils

Utilities for data manipulation and transformation.

```python
from utils.data_utils import flatten_dict, unflatten_dict, deep_merge

# Flatten nested dictionary
nested = {
    "model": {
        "hidden_size": 768,
        "layers": {"count": 12}
    }
}
flat = flatten_dict(nested)
# {"model.hidden_size": 768, "model.layers.count": 12}

# Unflatten back
original = unflatten_dict(flat)

# Deep merge dictionaries
base = {"a": 1, "b": {"c": 2}}
override = {"b": {"d": 3}}
merged = deep_merge(base, override)
# {"a": 1, "b": {"c": 2, "d": 3}}
```

**Functions:**

| Function | Description |
|----------|-------------|
| `flatten_dict(d, sep)` | Flatten nested dict to single level |
| `unflatten_dict(d, sep)` | Restore nested structure |
| `deep_merge(*dicts)` | Recursively merge dictionaries |
| `safe_get(d, path, default)` | Safely get nested value |

---

### Data Processing

Process batches of data for training and inference.

```python
from utils.data_processing import process_data, create_batch

# Process text batch
batch = process_data(
    batch=["Hello world", "How are you?"],
    tokenizer=tokenizer,
    max_length=512,
    padding="max_length",
    truncation=True
)
# Returns: {"input_ids": [...], "attention_mask": [...]}

# Create training batch
train_batch = create_batch(
    texts=texts,
    labels=labels,
    tokenizer=tokenizer,
    device=backend.device
)
```

---

### String Utils

String manipulation and cleaning utilities.

```python
from utils.string_utils import clean_text, truncate, slugify, remove_html

# Clean text
dirty = "  Hello   World  \n\n  "
clean = clean_text(dirty)  # "Hello World"

# Truncate with ellipsis
long_text = "This is a very long text that needs to be truncated"
short = truncate(long_text, max_length=20)  # "This is a very lo..."

# Create URL-safe slug
title = "Hello World! How are you?"
slug = slugify(title)  # "hello-world-how-are-you"

# Remove HTML tags
html = "<p>Hello <b>World</b></p>"
text = remove_html(html)  # "Hello World"
```

**Functions:**

| Function | Description |
|----------|-------------|
| `clean_text(text)` | Remove unwanted characters |
| `truncate(text, length)` | Truncate with ellipsis |
| `slugify(text)` | Create URL-safe slug |
| `remove_html(text)` | Strip HTML tags |
| `normalize_whitespace(text)` | Normalize spaces |

---

### Math Utils

Mathematical helper functions.

```python
from utils.math_utils import safe_divide, clamp, moving_average, softmax

# Safe division (handles zero)
result = safe_divide(10, 0, default=0.0)  # Returns 0.0

# Clamp value to range
value = clamp(150, min_val=0, max_val=100)  # Returns 100

# Calculate moving average
values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
avg = moving_average(values, window=3)  # [2.0, 3.0, 4.0, ...]

# Softmax (pure Python)
logits = [1.0, 2.0, 3.0]
probs = softmax(logits)  # [0.09, 0.24, 0.67]
```

**Functions:**

| Function | Description |
|----------|-------------|
| `safe_divide(a, b, default)` | Division without ZeroDivisionError |
| `clamp(value, min, max)` | Restrict value to range |
| `moving_average(values, window)` | Calculate moving average |
| `softmax(values)` | Softmax normalization |
| `cosine_similarity(a, b)` | Cosine similarity |

---

### Format Utils

Formatting utilities for display and logging.

```python
from utils.format_utils import format_bytes, format_time, format_number

# Format bytes
size = format_bytes(1536000000)  # "1.43 GB"

# Format time duration
duration = format_time(3725)  # "1h 2m 5s"

# Format large numbers
number = format_number(1234567)  # "1,234,567"
number = format_number(1234567, suffix=True)  # "1.23M"
```

**Functions:**

| Function | Description |
|----------|-------------|
| `format_bytes(bytes)` | Human-readable bytes |
| `format_time(seconds)` | Human-readable duration |
| `format_number(num)` | Formatted number with separators |
| `format_percentage(value)` | Format as percentage |

---

### Checkpoint Manager

Save and load model checkpoints.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Checkpoint Manager                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  checkpoints/                                                    │
│  ├── checkpoint-1000/                                           │
│  │   ├── model.safetensors                                      │
│  │   ├── optimizer.pt                                           │
│  │   └── metadata.json                                          │
│  ├── checkpoint-2000/                                           │
│  │   └── ...                                                    │
│  └── latest -> checkpoint-2000/                                 │
│                                                                  │
│  Features:                                                       │
│  • Automatic cleanup (keep N most recent)                       │
│  • Async saving (non-blocking)                                  │
│  • Compression support                                          │
│  • TPU-optimized serialization                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
from utils.checkpoint_manager import CheckpointManager
from core.config import CheckpointConfig

# Initialize manager
config = CheckpointConfig(
    base_dir="./checkpoints",
    max_to_keep=5,
    save_optimizer=True,
    compression=True
)
manager = CheckpointManager(config)

# Save checkpoint
manager.save(
    step=1000,
    model_state=model.state_dict(),
    optimizer_state=optimizer.state_dict(),
    metadata={"loss": 0.5, "epoch": 2}
)

# Load latest checkpoint
state = manager.load_latest()
model.load_state_dict(state["model"])
optimizer.load_state_dict(state["optimizer"])

# Load specific checkpoint
state = manager.load(step=1000)

# List available checkpoints
checkpoints = manager.list_checkpoints()
# [1000, 2000, 3000, 4000, 5000]
```

---

### Cache Manager

TPU-optimized caching system.

```python
from utils.cache_standalone import CacheManager, create_cache

# Create cache instance
cache = create_cache(
    max_size=1000,
    ttl=3600,  # 1 hour
    backend="memory"  # or "disk", "redis"
)

# Basic operations
cache.set("key", value)
value = cache.get("key")
cache.delete("key")

# With TTL
cache.set("temp_key", value, ttl=60)  # 60 seconds

# Decorator for function caching
@cache.memoize(ttl=300)
def expensive_computation(x, y):
    return x ** y

# Batch operations
cache.set_many({"a": 1, "b": 2, "c": 3})
values = cache.get_many(["a", "b", "c"])
```

---

### Logging Utils

Configure and manage logging.

```python
from utils.logging_utils import setup_logger, get_logger, LogLevel

# Setup root logger
setup_logger(
    name="capibara",
    level=LogLevel.INFO,
    log_file="logs/capibara.log",
    format_string="%(asctime)s | %(levelname)s | %(message)s",
    include_console=True
)

# Get module logger
logger = get_logger(__name__)
logger.info("Training started")
logger.debug("Batch processed", extra={"batch_size": 32})

# Structured logging
logger.info("Metrics", extra={
    "loss": 0.5,
    "accuracy": 0.95,
    "step": 1000
})
```

**Log Levels:**

```
┌────────────────────────────────────────────────────────────┐
│  Level    │ Value │ Use Case                               │
├───────────┼───────┼────────────────────────────────────────┤
│  DEBUG    │  10   │ Detailed diagnostic information       │
│  INFO     │  20   │ Generatel operational messages          │
│  WARNING  │  30   │ Potential issues                      │
│  ERROR    │  40   │ Errors that need attention            │
│  CRITICAL │  50   │ Critical failures                     │
└────────────────────────────────────────────────────────────┘
```

---

### Monitoring

Real-time system monitoring.

```python
from utils.monitoring import RealTimeMonitor, MetricsCollector

# Start monitor
monitor = RealTimeMonitor(
    interval=1.0,
    metrics=["cpu", "memory", "gpu", "disk"]
)
monitor.start()

# Get current metrics
metrics = monitor.get_metrics()
# {
#     "cpu_percent": 45.2,
#     "memory_percent": 62.8,
#     "memory_used_gb": 24.5,
#     "disk_percent": 35.0,
#     "gpu_utilization": 87.3,
#     "gpu_memory_used": 38.2
# }

# Collect training metrics
collector = MetricsCollector()
collector.add("loss", 0.5)
collector.add("accuracy", 0.95)
summary = collector.summarize()
# {"loss_mean": 0.5, "accuracy_mean": 0.95, ...}

# Stop monitor
monitor.stop()
```

---

### System Info

System information utilities.

```python
from utils.system_info import SystemMonitor, get_system_info

# Get system overview
info = get_system_info()
print(info)
# {
#     "platform": "Linux",
#     "python_version": "3.10.0",
#     "cpu_count": 8,
#     "memory_total_gb": 32.0,
#     "gpu_available": True,
#     "gpu_count": 2,
#     "gpu_names": ["NVIDIA A100", "NVIDIA A100"]
# }

# Continuous monitoring
monitor = SystemMonitor()
while training:
    stats = monitor.get_stats()
    log_metrics(stats)
```

---

### Validation Utils

Input validation utilities.

```python
from utils.validation_utils import (
    is_valid_url,
    is_valid_email,
    is_valid_ipv4,
    validate_range,
    validate_type
)

# URL validation
is_valid_url("https://example.com")  # True
is_valid_url("not-a-url")  # False

# Email validation
is_valid_email("user@example.com")  # True

# IP validation
is_valid_ipv4("192.168.1.1")  # True
is_valid_ipv4("999.999.999.999")  # False

# Range validation
validate_range(value=5, min_val=0, max_val=10)  # True
validate_range(value=15, min_val=0, max_val=10)  # Raises ValueError

# Type validation
validate_type(value="hello", expected=str)  # True
validate_type(value=123, expected=str)  # Raises TypeError
```

---

### Error Handling

Error handling decorators and utilities.

```python
from utils.error_handling import handle_error, DataProcessingError, retry

# Handle specific errors
@handle_error(DataProcessingError)
def process_batch(batch):
    # Processing code
    pass

# Retry on failure
@retry(max_attempts=3, delay=1.0, backoff=2.0)
def unreliable_operation():
    # May fail occasionally
    pass

# Custom error handler
@handle_error(ValueError, handler=lambda e: default_value)
def safe_parse(text):
    return int(text)
```

---

### Version

Access version information.

```python
from utils.version import __version__, get_version, get_version_info

print(__version__)  # "2.1.8"
print(get_version())  # "2.1.8"
print(get_version_info())  # (2, 1, 8)
```

## Usage Patterns

### Configuration Loading Pattern

```python
from utils import load_config_file, merge_configs
from pathlib import Path

def load_training_config(config_name: str, overrides: dict = None):
    """Load training configuration with environment overrides."""

    # Base config
    base = load_config_file("config/base.yaml")

    # Environment-specific
    env = os.getenv("ENV", "development")
    env_config = load_config_file(f"config/{env}.yaml")

    # Named config
    named_config = load_config_file(f"config/{config_name}.yaml")

    # Merge all
    config = merge_configs(base, env_config, named_config)

    # Apply runtime overrides
    if overrides:
        config = merge_configs(config, overrides)

    return config
```

### Logging Setup Pattern

```python
from utils import setup_logger, get_logger
import logging

def setup_training_logging(output_dir: str, debug: bool = False):
    """Setup logging for training run."""

    level = logging.DEBUG if debug else logging.INFO

    # Root logger
    setup_logger(
        name="capibara",
        level=level,
        log_file=f"{output_dir}/training.log",
        include_console=True
    )

    # Reduce noise from libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)

    return get_logger("capibara.training")
```

### Checkpoint Pattern

```python
from utils import CheckpointManager
from core.config import CheckpointConfig

class Trainer:
    def __init__(self, output_dir: str):
        self.checkpoint_manager = CheckpointManager(
            CheckpointConfig(
                base_dir=f"{output_dir}/checkpoints",
                max_to_keep=3
            )
        )

    def save_checkpoint(self, step: int):
        self.checkpoint_manager.save(
            step=step,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            metadata={
                "step": step,
                "loss": self.current_loss,
                "learning_rate": self.current_lr
            }
        )

    def resume_from_checkpoint(self):
        state = self.checkpoint_manager.load_latest()
        if state:
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            return state["metadata"]["step"]
        return 0
```

## Best Practices

1. **Use Type Hints**: All utility functions support type hints
2. **Handle Errors Gracefully**: Use error handling decorators
3. **Log Appropriately**: Use structured logging with context
4. **Cache Expensive Operations**: Use caching for repeated computations
5. **Validate Inputs**: Validate external inputs at boundaries

## See Also

- [Core Module](../core/README.md)
- [Configuration](../config/README.md)
- [Training Module](../training/README.md)

## Example quick

Example (pseudo-code) para usar utilidades:

```python
# from utils.logging_utils import setup_logging
# logger = setup_logging()
# logger.info("utils listo")
```

## Issues por hacer

- [ ] logger.warning(f"State is missing required attribute: {attr}") - `utils\checkpoint_manager.py:288`
- [ ] Crea un logger que incluye contexto adicional en todos los mensajes. - `utils\logging_utils.py:447`
- [ ] DLLs) may be missing. - `utils\memory_profiler.py:56`
- [ ] # verify todos los contratos - `utils\smart_utils_contracts.py:671`
- [ ] 'success_rate': 95.0,  # Placeholder - `utils\smart_utils_contracts.py:716`
- [ ] 'error_rate_percent': 2.0,  # Placeholder - `utils\smart_utils_contracts.py:717`
- [ ] 'availability_percent': 99.5,  # Placeholder - `utils\smart_utils_contracts.py:718`
- [ ] 'execution_time_ms': 50.0  # Placeholder - `utils\smart_utils_contracts.py:719`
- [ ] """Obtiene estado global de todos los contratos.""" - `utils\smart_utils_contracts.py:736`
- [ ] # Reemplazar todos los tipos de espacios en blanco with espacios normales - `utils\string_utils.py:127`
- [ ] TODO: Add detailed description. - `utils\system_info.py:23`
