"""
Vector Quantization Module - CapibaraGPT v3

Advanced vector quantization system supporting:
- vqbit_layer: VQbit layer implementation
- vq_arm_axion: ARM Axion optimizations
"""

import logging

logger = logging.getLogger(__name__)

# VQbit package (submodule) availability
try:
    from . import vqbit  # noqa: F401
    VQBIT_PACKAGE_AVAILABLE = True
except Exception as e:
    vqbit = None  # type: ignore
    VQBIT_PACKAGE_AVAILABLE = False
    logger.debug("VQ subpackage 'vqbit' unavailable: %s", e)

# VQbit layer
try:
    from .vqbit_layer import VQbitLayer
    VQBIT_AVAILABLE = True
except Exception as e:
    VQBIT_AVAILABLE = False
    VQbitLayer = None
    logger.debug("VQbit layer unavailable: %s", e)

# ARM Axion optimizations
try:
    from .vq_arm_axion import VQArmAxion
    ARM_AVAILABLE = True
except Exception as e:
    ARM_AVAILABLE = False
    VQArmAxion = None
    logger.debug("ARM Axion VQ unavailable: %s", e)


__all__ = [
    # Subpackage
    "vqbit",
    "VQBIT_PACKAGE_AVAILABLE",
    # Orchestrator
    # VQbit
    "VQbitLayer",
    # Multi-modal
    # Adaptive
    # ARM
    "VQArmAxion",
    # Flags
    "VQBIT_AVAILABLE",
    "ARM_AVAILABLE",
]
