"""
Sparsity layers module.

This module provides sparsity and quantization layers for the Capibara model.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check for dependencies
try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    logger.warning("JAX/Flax not available - sparsity layers will use fallback implementations")

# Import all sparsity modules
if JAX_AVAILABLE:
    try:
        from .bitnet import BitNet158, Conv1DBlock
        BITNET_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"BitNet components not available - using fallback implementations: {e}")
        BITNET_AVAILABLE = False
    
    try:
        from .sparse_capibara import SparseCapibara
        SPARSE_CAPIBARA_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"SparseCapibara not available - using fallback implementation: {e}")
        SPARSE_CAPIBARA_AVAILABLE = False
    
    try:
        from .affine_quantizer import AffineQuantizer
        AFFINE_QUANTIZER_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"AffineQuantizer not available - using fallback implementation: {e}")
        AFFINE_QUANTIZER_AVAILABLE = False
    
    try:
        from .mixture_of_rookies import MixtureOfRookies
        MIXTURE_OF_ROOKIES_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"MixtureOfRookies not available - using fallback implementation: {e}")
        MIXTURE_OF_ROOKIES_AVAILABLE = False
else:
    BITNET_AVAILABLE = False
    SPARSE_CAPIBARA_AVAILABLE = False
    AFFINE_QUANTIZER_AVAILABLE = False
    MIXTURE_OF_ROOKIES_AVAILABLE = False

# Fallback implementations when JAX is not available or imports fail
if not BITNET_AVAILABLE:
    class BitNet158:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("BitNet158 not implemented - JAX/Flax required")
    
    class Conv1DBlock:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Conv1DBlock not implemented - JAX/Flax required")

if not SPARSE_CAPIBARA_AVAILABLE:
    class SparseCapibara:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("SparseCapibara not implemented - JAX/Flax required")

if not AFFINE_QUANTIZER_AVAILABLE:
    class AffineQuantizer:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("AffineQuantizer not implemented - JAX/Flax required")

if not MIXTURE_OF_ROOKIES_AVAILABLE:
    class MixtureOfRookies:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("MixtureOfRookies not implemented - JAX/Flax required")

__all__ = [
    "BitNet158",
    "Conv1DBlock", 
    "SparseCapibara",
    "AffineQuantizer",
    "MixtureOfRookies",
    "BITNET_AVAILABLE",
    "SPARSE_CAPIBARA_AVAILABLE", 
    "AFFINE_QUANTIZER_AVAILABLE",
    "MIXTURE_OF_ROOKIES_AVAILABLE"
]