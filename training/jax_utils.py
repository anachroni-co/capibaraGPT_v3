"""
JAX Utilities - CapibaraGPT v3

Shared JAX imports with numpy/math fallbacks for training modules.
"""

import logging

logger = logging.getLogger(__name__)

JAX_AVAILABLE = False
jax = None
jnp = None
optax = None

# Try JAX first
try:
    import jax as jax
    import jax.numpy as jnp
    from jax import random, grad, jit, vmap
    from jax.nn import softmax, relu, sigmoid
    from jax.tree_util import tree_map
    import optax
    np = jnp
    JAX_AVAILABLE = True
    logger.debug("Using capibara.jax with optax")
except ImportError as exc:
    logger.debug("JAX/optax not available, will use numpy fallback: %s", exc)

# Fallback to numpy
if not JAX_AVAILABLE:
    try:
        import numpy as np
        import math

        jnp = np

        def softmax(x, axis=-1):
            """Softmax with numpy."""
            exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

        def relu(x):
            """ReLU with numpy."""
            return np.maximum(0, x)

        def sigmoid(x):
            """Sigmoid with numpy."""
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

        def tree_map(func, tree):
            """Simple tree_map fallback."""
            return func(tree)

        def grad(func):
            """Dummy grad for compatibility."""
            return func

        def jit(func):
            """Dummy jit for compatibility."""
            return func

        def vmap(func, *args, **kwargs):
            """Dummy vmap for compatibility."""
            return func

        class random:
            """Dummy random module."""
            @staticmethod
            def PRNGKey(seed):
                return seed
            @staticmethod
            def split(key, num=2):
                return [key] * num

        class optax:
            """Dummy optax for compatibility."""
            @staticmethod
            def adam(learning_rate):
                return {"lr": learning_rate}
            @staticmethod
            def apply_updates(params, updates):
                if isinstance(params, dict) and isinstance(updates, dict):
                    return {k: params[k] + updates.get(k, 0) for k in params}
                return params + updates
            @staticmethod
            def chain(*args):
                return {}

        logger.debug("Using numpy fallback")
    except ImportError:
        # Ultimate fallback - pure Python
        import math

        class DummyArray:
            """Minimal array-like for basic operations."""
            def __init__(self, data):
                self.data = data if isinstance(data, list) else [data]
            def __getitem__(self, idx):
                return self.data[idx] if idx < len(self.data) else 0
            def __len__(self):
                return len(self.data)
            def __iter__(self):
                return iter(self.data)

        class np:
            """Minimal numpy-like interface."""
            @staticmethod
            def array(data):
                return DummyArray(data)
            @staticmethod
            def mean(data):
                d = list(data) if hasattr(data, '__iter__') else [data]
                return sum(d) / len(d) if d else 0
            @staticmethod
            def exp(x):
                if isinstance(x, (int, float)):
                    return math.exp(x)
                return DummyArray([math.exp(i) for i in x])
            @staticmethod
            def maximum(a, b):
                if isinstance(a, (int, float)):
                    return max(a, b)
                return DummyArray([max(i, b) for i in a])
            @staticmethod
            def clip(x, lo, hi):
                if isinstance(x, (int, float)):
                    return max(lo, min(hi, x))
                return DummyArray([max(lo, min(hi, i)) for i in x])

        jnp = np

        def softmax(x, axis=-1):
            return x

        def relu(x):
            return max(0, x) if isinstance(x, (int, float)) else x

        def sigmoid(x):
            if isinstance(x, (int, float)):
                return 1 / (1 + math.exp(-max(-500, min(500, x))))
            return x

        def tree_map(func, tree):
            return func(tree)

        def grad(func):
            return func

        def jit(func):
            return func

        def vmap(func, *args, **kwargs):
            return func

        class random:
            @staticmethod
            def PRNGKey(seed):
                return seed
            @staticmethod
            def split(key, num=2):
                return [key] * num

        class optax:
            @staticmethod
            def adam(learning_rate):
                return {}
            @staticmethod
            def apply_updates(params, updates):
                if isinstance(params, dict) and isinstance(updates, dict):
                    return {k: params[k] + updates.get(k, 0) for k in params}
                try:
                    return params + updates
                except TypeError:
                    return params
            @staticmethod
            def chain(*args):
                return {}

        logger.warning("Using pure Python fallback (no numpy)")


# ARM optimizations (optional)
try:
    from capibara.core.arm_optimizations import ARMAxionInferenceOptimizer, ARM_CAPABILITIES
    ARM_OPTIMIZATIONS_AVAILABLE = bool(ARM_CAPABILITIES.get("total_features", 0))
except ImportError:
    ARM_OPTIMIZATIONS_AVAILABLE = False

    class ARMAxionInferenceOptimizer:
        """Compatibility stub when ARM optimizations are not installed."""
        def __init__(self, *args, **kwargs):
            self.enabled = False
        def optimize_inference(self, *args, **kwargs):
            return args[0] if args else None


__all__ = [
    "np",
    "jnp",
    "jax",
    "optax",
    "softmax",
    "relu",
    "sigmoid",
    "tree_map",
    "grad",
    "jit",
    "vmap",
    "random",
    "JAX_AVAILABLE",
    "ARM_OPTIMIZATIONS_AVAILABLE",
    "ARMAxionInferenceOptimizer",
]
