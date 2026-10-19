"""
Legacy routers subpackage for CapibaraGPT core.

Use `capibara.core.router.EnhancedRouter` as the primary router.
"""

from .base import BaseRouter, BaseRouterV2
from .bto import BtoRouterV2
from .adaptive_router import AdaptiveRouter, AdtoptiveRouter

# Compatibility aliases
BTORouter = BtoRouterV2

__all__ = [
    "BaseRouter",
    "BaseRouterV2",
    "BtoRouterV2",
    "BTORouter",
    "AdaptiveRouter",
    "AdtoptiveRouter",
]
