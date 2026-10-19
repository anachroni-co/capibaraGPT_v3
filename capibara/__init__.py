"""
Core Module for CapibaraGPT v3.

This package contains the core components of CapibaraGPT:

Submodules:
    - ssm: State Space Models (SSM, SpikeSSM, AdaptiveSpikeSSM)
    - routers: Routing mechanisms (SelfModifyingRouter with Nested Learning)
    - optimizations: Hardware optimizations (TPU v6e-64)
    - vq: Vector Quantization (ARM Axion optimizations)
"""

import logging

logger = logging.getLogger(__name__)

try:
    from . import ssm
    SSM_AVAILABLE = True
except Exception as e:
    ssm = None
    SSM_AVAILABLE = False
    logger.debug("capibara.ssm unavailable: %s", e)

try:
    from . import routers
    ROUTERS_AVAILABLE = True
except Exception as e:
    routers = None
    ROUTERS_AVAILABLE = False
    logger.debug("capibara.routers unavailable: %s", e)

try:
    from . import optimizations
    OPTIMIZATIONS_AVAILABLE = True
except Exception as e:
    optimizations = None
    OPTIMIZATIONS_AVAILABLE = False
    logger.debug("capibara.optimizations unavailable: %s", e)

try:
    from . import vq
    VQ_AVAILABLE = True
except Exception as e:
    vq = None
    VQ_AVAILABLE = False
    logger.debug("capibara.vq unavailable: %s", e)

__all__ = [
    "ssm",
    "routers",
    "optimizations",
    "vq",
    "SSM_AVAILABLE",
    "ROUTERS_AVAILABLE",
    "OPTIMIZATIONS_AVAILABLE",
    "VQ_AVAILABLE",
]
