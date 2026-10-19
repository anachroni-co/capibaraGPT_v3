# Routers Module (Core)

This folder contains **legacy routing primitives** used by CapibaraGPT v3.
The **primary router** for the project lives in `core/router.py`
(`EnhancedRouter`).

## Available Routers (Legacy)
- `BaseRouter` / `BaseRouterV2` (`base.py`)
- `BtoRouterV2` (`bto.py`) - simple Best Time Optimization baseline
- `AdaptiveRouter` (`adaptive_router.py`) - adapts based on config

## Primary Router
Use `EnhancedRouter` for new work:
```python
from capibara.core.router import EnhancedRouter, RouterConfig

router = EnhancedRouter(RouterConfig())
```

## Example (Legacy)
```python
from capibara.core.routers import AdaptiveRouter
from capibara.core.config import ModularModelConfig

router = AdaptiveRouter(ModularModelConfig())
result = router.route({"path": "/default"})
print(result)
```

## Notes
- This module is intentionally minimal and dependency-light.
- For new work, prefer `capibara.core.router.EnhancedRouter`.
- If you need additional router variants, add them in new files and export
  them via `core/routers/__init__.py` with tests.
