# Code Standards

## Generatel principles

- All code must pass `pytest` before merging
- Factorization guardrails are enforced in CI (`tests/unit/test_factorization.py`)
- Use centralized utilities instead of inline boilerplate

## Import conventions

### JAX/Flax in `layers/`

Use `layers.jax_compat`:

```python
# Correct
from layers.jax_compat import jax, jnp, nn, JAX_AVAILABLE

# Wrong — will fail factorization tests
try:
    import jax
    from flax import linen as nn
except ImportError:
    ...
```

### Safe imports in `core/`

Use `core.import_utils.safe_import`:

```python
# Correct
from core.import_utils import safe_import
Router = safe_import("core.router", "Router")

# Wrong in core/__init__.py
try:
    from core.router import Router
except ImportError:
    Router = None
```

### Lazy backend imports

Use `core.backends.lazy_import`:

```python
# Correct
from core.backends.lazy_import import ensure_torch
torch, nn, F = ensure_torch()

# Wrong in backends/
try:
    import torch
except ImportError:
    ...
```

## Dataclass fields

Use `utils.field_helpers` for common defaults:

```python
from utils.field_helpers import dict_field, list_field

@dataclass
class Config:
    metadata: dict = dict_field()     # not field(default_factory=dict)
    items: list = list_field()        # not field(default_factory=list)
```

## Wrapper classes

Use `CompositeWrapper` + `functools.partial` instead of defining new wrapper
classes:

```python
# Correct
MyWrapper = partial(CompositeWrapper, label="My", use_residual=True)

# Wrong — will fail factorization tests
class MyWrapper(BaseLayer):
    def __init__(self, base_layer, component): ...
```

## Testing

Every PR must pass:

1. `pytest tests/unit/test_factorization.py` — factorization guardrails
2. `pytest` — full test suite (256+ tests)
