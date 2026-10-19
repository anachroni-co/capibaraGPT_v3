"""
JAX Neural Network Module - CapibaraGPT v3

Native JAX neural network implementations including:
- activations: Activation functions (relu, gelu, swish)
- layers: Core layers (Dense, Embedding, LayerNorm)
- advanced: Advanced Flax-style layers
- initializers: Weight initialization functions
- attention: Attention mechanisms
- decorators: JIT compilation decorators
"""

import logging

logger = logging.getLogger(__name__)

# Import submodules with fallbacks
try:
    from . import activations
    ACTIVATIONS_AVAILABLE = True
except ImportError:
    ACTIVATIONS_AVAILABLE = False
    activations = None

try:
    from . import layers
    LAYERS_AVAILABLE = True
except ImportError:
    LAYERS_AVAILABLE = False
    layers = None

try:
    from . import advanced
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    advanced = None

try:
    from . import initializers
    INITIALIZERS_AVAILABLE = True
except ImportError:
    INITIALIZERS_AVAILABLE = False
    initializers = None

try:
    from . import attention
    ATTENTION_AVAILABLE = True
except ImportError:
    ATTENTION_AVAILABLE = False
    attention = None

try:
    from . import decorators
    DECORATORS_AVAILABLE = True
except ImportError:
    DECORATORS_AVAILABLE = False
    decorators = None

try:
    from . import utils
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    utils = None

try:
    from .activations import (
        softmax,
        log_softmax,
        sigmoid,
        relu,
        gelu,
        swish,
        silu,
        tanh,
        glu,
        swiglu,
        geglu,
        reglu,
        mish,
        leaky_relu,
        elu,
        selu,
        hard_sigmoid,
        hard_swish,
        hard_tanh,
    )
    _ACTIVATION_EXPORTS_AVAILABLE = True
except Exception:
    _ACTIVATION_EXPORTS_AVAILABLE = False


__all__ = [
    "activations",
    "layers",
    "advanced",
    "initializers",
    "attention",
    "decorators",
    "utils",
    "ACTIVATIONS_AVAILABLE",
    "LAYERS_AVAILABLE",
    "ADVANCED_AVAILABLE",
    "INITIALIZERS_AVAILABLE",
    "ATTENTION_AVAILABLE",
    "DECORATORS_AVAILABLE",
    "UTILS_AVAILABLE",
]

if _ACTIVATION_EXPORTS_AVAILABLE:
    __all__.extend(
        [
            "softmax",
            "log_softmax",
            "sigmoid",
            "relu",
            "gelu",
            "swish",
            "silu",
            "tanh",
            "glu",
            "swiglu",
            "geglu",
            "reglu",
            "mish",
            "leaky_relu",
            "elu",
            "selu",
            "hard_sigmoid",
            "hard_swish",
            "hard_tanh",
        ]
    )
