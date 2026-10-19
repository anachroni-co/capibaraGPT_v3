# capibara/core – Core Components (v3)

The `core/` package is the **real integration layer** for CapibaraGPT v3.  
It provides routing, modular composition, configuration utilities, and
lightweight primitives that are CPU‑safe by default.

## What’s Here (Real Modules)

- `config.py` – core configuration helpers
- `modular_model.py` – modular model composition
- `router.py` / `routing.py` – routing primitives (async + simple)
- `optimization.py` – training metrics + state helpers (JAX/Flax optional)
- `kernels/` – TPU v4 wrapper utilities
- `pipelines/` – RAG pipeline helpers (minimal)
- `routers/` – base/adaptive/BTO routers
- `distributed/` – minimal mesh config helpers
- `tpu/` – TPU config + v6e helpers (optional deps)
- `cot/` – chain‑of‑thought tooling
- `monitoring/`, `verification/`, `observers/` – diagnostics/verification utilities
- `activations/` – minimal contextual activations (real, CPU‑friendly)

> Some submodules rely on optional deps (JAX, Flax, Optax, etc.).  
> Missing deps should fail **explicitly** rather than silently.

## Quick Start

```python
from capibara.core import ModularCapibaraModel, ModularConfig

config = ModularConfig()
model = ModularCapibaraModel(config)
```

```python
from capibara.core.router import EnhancedRouter, RouterConfig

router = EnhancedRouter(RouterConfig())
```

## Package Import

You can import via either namespace:

```python
import core
import capibara.core
```

## Notes

- This README describes **what exists today**.
- If you add new core features, document them here and in the module README.

## TODOs

See `core/TODOs.md` for the current list of tracked TODOs in this package.
