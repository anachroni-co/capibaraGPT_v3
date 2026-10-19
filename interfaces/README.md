# Interfaces

**Abstract Interfaces and Protocol Definitions**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INTERFACE ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                          Interface Layer                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │   ICache        ISubModel       IProcessor       IBackend              │ │
│  │     │              │               │                │                  │ │
│  │     │              │               │                │                  │ │
│  └─────┼──────────────┼───────────────┼────────────────┼──────────────────┘ │
│        │              │               │                │                    │
│        ▼              ▼               ▼                ▼                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                     Implementation Layer                                 ││
│  │                                                                          ││
│  │  MemoryCache   MambaModel    TextProcessor    CPUBackend               ││
│  │  DiskCache     VisionModel   CodeProcessor    GPUBackend               ││
│  │  RedisCache    HybridModel   AudioProcessor   TPUBackend               ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Overview

The interfaces module defines abstract base classes and protocols that establish contracts between different components of CapibaraGPT. This enables loose coupling, easier testing, and flexible implementations.

## Module Structure

```
interfaces/
├── __init__.py               # Interface exports
├── icache.py                 # Cache interface
├── isubmodel.py              # Sub-model interface
├── isub_models.py            # Sub-model collection interface
└── ultra_interface_system.py # Advanced interface system
```

## Core Interfaces

### ICache

Interface for caching implementations.

```python
from abc import ABC, abstractmethod
from typing import Any, Optional

class ICache(ABC):
    """Abstract interface for cache implementations."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
```

**Implementations:**

```python
from interfaces import ICache

class MemoryCache(ICache):
    """In-memory cache implementation."""

    def __init__(self, max_size: int = 1000):
        self._cache = {}
        self._max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
        self._cache[key] = value

    # ... other methods

class DiskCache(ICache):
    """Disk-based cache implementation."""
    pass

class RedisCache(ICache):
    """Redis-based cache implementation."""
    pass
```

### ISubModel

Interface for sub-model components.

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class ISubModel(ABC):
    """Abstract interface for sub-models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return model name."""
        pass

    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Return model configuration."""
        pass

    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        """Forward pass through the model."""
        pass

    @abstractmethod
    def load_weights(self, path: str) -> None:
        """Load model weights from path."""
        pass

    @abstractmethod
    def save_weights(self, path: str) -> None:
        """Save model weights to path."""
        pass
```

**Example Implementation:**

```python
from interfaces import ISubModel

class MambaSubModel(ISubModel):
    """Mamba-based sub-model implementation."""

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._name = "mamba"
        self._model = self._build_model()

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def forward(self, inputs: Any) -> Any:
        return self._model(inputs)

    def load_weights(self, path: str) -> None:
        # Load implementation
        pass

    def save_weights(self, path: str) -> None:
        # Save implementation
        pass
```

### ISubModels

Interface for managing collections of sub-models.

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class ISubModels(ABC):
    """Interface for sub-model collections."""

    @abstractmethod
    def register(self, model: ISubModel) -> None:
        """Register a sub-model."""
        pass

    @abstractmethod
    def get(self, name: str) -> ISubModel:
        """Get sub-model by name."""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """List all registered model names."""
        pass

    @abstractmethod
    def forward_all(self, inputs: Any) -> Dict[str, Any]:
        """Run forward pass on all models."""
        pass

    @abstractmethod
    def forward_selected(
        self,
        inputs: Any,
        model_names: List[str]
    ) -> Dict[str, Any]:
        """Run forward pass on selected models."""
        pass
```

## Protocol Definitions

For duck-typing support using Python protocols:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Configurable(Protocol):
    """Protocol for configurable components."""

    def get_config(self) -> Dict[str, Any]:
        """Return configuration."""
        ...

    def update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration."""
        ...

@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable components."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Serializable":
        """Deserialize from dictionary."""
        ...

@runtime_checkable
class Trainable(Protocol):
    """Protocol for trainable components."""

    def train(self) -> None:
        """Set to training mode."""
        ...

    def eval(self) -> None:
        """Set to evaluation mode."""
        ...

    def parameters(self) -> Iterator[Any]:
        """Return trainable parameters."""
        ...
```

## Usage Patterns

### Dependency Injection

```python
class ModelTrainer:
    """Trainer that depends on interfaces, not implementations."""

    def __init__(
        self,
        model: ISubModel,
        cache: ICache,
        logger: ILogger
    ):
        self.model = model
        self.cache = cache
        self.logger = logger

    def train(self, data):
        # Works with any implementation
        cached_data = self.cache.get("preprocessed")
        if cached_data is None:
            cached_data = preprocess(data)
            self.cache.set("preprocessed", cached_data)

        output = self.model.forward(cached_data)
        self.logger.log(f"Training output: {output}")
```

### Factory Pattern

```python
from interfaces import ICache, ISubModel

class CacheFactory:
    """Factory for creating cache instances."""

    @staticmethod
    def create(cache_type: str, **kwargs) -> ICache:
        if cache_type == "memory":
            return MemoryCache(**kwargs)
        elif cache_type == "disk":
            return DiskCache(**kwargs)
        elif cache_type == "redis":
            return RedisCache(**kwargs)
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")

class SubModelFactory:
    """Factory for creating sub-model instances."""

    _registry = {}

    @classmethod
    def register(cls, name: str, model_class: type):
        cls._registry[name] = model_class

    @classmethod
    def create(cls, name: str, config: dict) -> ISubModel:
        if name not in cls._registry:
            raise ValueError(f"Unknown model: {name}")
        return cls._registry[name](config)
```

### Type Checking

```python
from interfaces import ICache, Configurable

def process_with_cache(cache: ICache, data: Any) -> Any:
    """Function that requires ICache interface."""
    key = hash(data)
    result = cache.get(key)
    if result is None:
        result = expensive_computation(data)
        cache.set(key, result)
    return result

# Type checker will verify interface compliance
cache = MemoryCache()
result = process_with_cache(cache, input_data)  #  Valid

# Runtime check with protocols
if isinstance(component, Configurable):
    config = component.get_config()
```

## Interface Guidelines

### Designing Interfaces

1. **Single Responsibility**: Each interface should have one purpose
2. **Interface Segregation**: Prefer many small interfaces over few large ones
3. **Dependency Inversion**: Depend on abstractions, not concretions
4. **Liskov Substitution**: Implementations must be substitutable

### Documentation

```python
class IProcessor(ABC):
    """
    Interface for data processors.

    All processor implementations must:
    1. Accept input data in the specified format
    2. Return processed data in the specified format
    3. Be stateless or thread-safe if stateful
    4. Handle errors gracefully

    Example:
        >>> processor = TextProcessor()
        >>> result = processor.process("Hello World")
        >>> assert isinstance(result, ProcessedData)
    """

    @abstractmethod
    def process(self, data: InputData) -> ProcessedData:
        """
        Process input data.

        Args:
            data: Input data to process

        Returns:
            Processed data

        Raises:
            ProcessingError: If processing fails
        """
        pass
```

## See Also

- [Sub-Models](../sub_models/README.md)
- [Core Backends](../core/backends/README.md)
- [Utils Cache](../utils/README.md#cache-manager)

## Example quick

Example (pseudo-code) para implementar una interfaz:

```python
from interfaces.base import BaseInterface

class MiInterface(BaseInterface):
    def run(self, payload):
        return {"result": payload}
```

## Issues por hacer

- [ ] missing_methods = [] - `interfaces\ultra_interface_system.py:651`
- [ ] missing_methods.append(method_name) - `interfaces\ultra_interface_system.py:655`
- [ ] if missing_methods: - `interfaces\ultra_interface_system.py:661`
- [ ] validation_result["errors"].extend([f"Missing method: {m}" for m in missing_methods]) - `interfaces\ultra_interface_system.py:662`
- [ ] quality_score = 0.9  # Simulated quality assessment - `interfaces\ultra_interface_system.py:923`
