"""
TPU Backend Implementation (JAX/Flax)

Optimized for TPU v4-32 and v6e-64 pods.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .base import BackendConfig, ComputeBackend, DType, TensorLike

from .lazy_import import ensure_jax

# Module-level aliases populated on first _ensure_jax() call
jax = None
jnp = None
flax = None
optax = None


def _ensure_jax():
    """Lazy import JAX modules."""
    global jax, jnp, flax, optax
    if jax is None:
        jax, jnp, flax, optax = ensure_jax()


# Dtype mapping
DTYPE_MAP = {
    DType.FLOAT32: "float32",
    DType.FLOAT16: "float16",
    DType.BFLOAT16: "bfloat16",
    DType.INT32: "int32",
    DType.INT64: "int64",
    DType.BOOL: "bool",
}


class TPUBackend(ComputeBackend):
    """
    TPU Backend using JAX/Flax.

    Optimized for:
    - TPU v4-32 (32 cores, 4x4x2 topology)
    - TPU v6e-64 (64 cores, 4x4x4 topology)

    Features:
    - Automatic XLA compilation with jax.jit
    - Efficient pmap for data parallelism
    - bfloat16 mixed precision
    - Efficient checkpointing with orbax
    """

    def __init__(self, config: Optional[BackendConfig] = None):
        super().__init__(config)
        self._rng_key = None
        self._devices = None
        self._mesh = None

    @property
    def name(self) -> str:
        return "tpu"

    @property
    def is_available(self) -> bool:
        """Check if TPU is available."""
        try:
            _ensure_jax()
            devices = jax.devices("tpu")
            return len(devices) > 0
        except Exception:
            return False

    def initialize(self) -> None:
        """Initialize JAX for TPU."""
        _ensure_jax()

        # Configure JAX
        jax.config.update("jax_enable_x64", False)  # Use 32-bit by default

        # Get TPU devices
        self._devices = jax.devices("tpu") if self.is_available else jax.devices()
        self._device = self._devices[0] if self._devices else None

        # Initialize RNG
        self._rng_key = jax.random.PRNGKey(42)

        # Set up mesh for distributed training
        if self.config.distributed and len(self._devices) > 1:
            from jax.sharding import Mesh, PartitionSpec, NamedSharding

            # Create device mesh based on topology
            if self.config.tpu_topology == "v4-32":
                mesh_shape = (4, 4, 2)  # 4x4x2 = 32 cores
            elif self.config.tpu_topology == "v6e-64":
                mesh_shape = (4, 4, 4)  # 4x4x4 = 64 cores
            else:
                mesh_shape = (len(self._devices),)

            devices_array = np.array(self._devices).reshape(mesh_shape)
            self._mesh = Mesh(devices_array, axis_names=("data", "model", "pipeline"))

        self._initialized = True

    def shutdown(self) -> None:
        """Clean up JAX resources."""
        if jax is not None:
            jax.clear_caches()
        self._initialized = False

    # ==================== Tensor Operations ====================

    def create_tensor(
        self,
        data: Any,
        dtype: Optional[DType] = None,
        device: Optional[str] = None,
        requires_grad: bool = False,
    ) -> TensorLike:
        _ensure_jax()
        jax_dtype = DTYPE_MAP.get(dtype, "float32") if dtype else None
        arr = jnp.array(data, dtype=jax_dtype)
        if device:
            arr = jax.device_put(arr, jax.devices(device)[0])
        return arr

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> TensorLike:
        _ensure_jax()
        jax_dtype = DTYPE_MAP.get(dtype, "float32") if dtype else None
        return jnp.zeros(shape, dtype=jax_dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> TensorLike:
        _ensure_jax()
        jax_dtype = DTYPE_MAP.get(dtype, "float32") if dtype else None
        return jnp.ones(shape, dtype=jax_dtype)

    def randn(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> TensorLike:
        _ensure_jax()
        jax_dtype = DTYPE_MAP.get(dtype, "float32") if dtype else None
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return jax.random.normal(subkey, shape, dtype=jax_dtype)

    def to_device(self, tensor: TensorLike, device: str) -> TensorLike:
        _ensure_jax()
        target_device = jax.devices(device)[0]
        return jax.device_put(tensor, target_device)

    def to_numpy(self, tensor: TensorLike) -> np.ndarray:
        _ensure_jax()
        return np.asarray(tensor)

    # ==================== Math Operations ====================

    def matmul(self, a: TensorLike, b: TensorLike) -> TensorLike:
        _ensure_jax()
        return jnp.matmul(a, b)

    def add(self, a: TensorLike, b: TensorLike) -> TensorLike:
        _ensure_jax()
        return jnp.add(a, b)

    def mul(self, a: TensorLike, b: TensorLike) -> TensorLike:
        _ensure_jax()
        return jnp.multiply(a, b)

    def softmax(self, x: TensorLike, axis: int = -1) -> TensorLike:
        _ensure_jax()
        return jax.nn.softmax(x, axis=axis)

    def layer_norm(
        self,
        x: TensorLike,
        normalized_shape: Tuple[int, ...],
        weight: Optional[TensorLike] = None,
        bias: Optional[TensorLike] = None,
        eps: float = 1e-5,
    ) -> TensorLike:
        _ensure_jax()
        # Compute mean and variance
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.var(x, axis=-1, keepdims=True)

        # Normalize
        x_norm = (x - mean) / jnp.sqrt(variance + eps)

        # Apply affine transformation
        if weight is not None:
            x_norm = x_norm * weight
        if bias is not None:
            x_norm = x_norm + bias

        return x_norm

    def gelu(self, x: TensorLike) -> TensorLike:
        _ensure_jax()
        return jax.nn.gelu(x, approximate=True)

    def silu(self, x: TensorLike) -> TensorLike:
        _ensure_jax()
        return jax.nn.silu(x)

    # ==================== Attention Operations ====================

    def scaled_dot_product_attention(
        self,
        query: TensorLike,
        key: TensorLike,
        value: TensorLike,
        mask: Optional[TensorLike] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ) -> TensorLike:
        """
        TPU-optimized attention using XLA.

        For TPU, we use JAX's native attention which is automatically
        optimized by XLA for the TPU architecture.
        """
        _ensure_jax()

        # Get dimensions
        batch_size, num_heads, seq_len, head_dim = query.shape
        scale = scale or (head_dim ** -0.5)

        # Compute attention scores
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", query, key) * scale

        # Apply causal mask if needed
        if is_causal:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            attn_weights = jnp.where(
                causal_mask[None, None, :, :] == 0,
                float("-inf"),
                attn_weights,
            )

        # Apply provided mask
        if mask is not None:
            attn_weights = jnp.where(mask == 0, float("-inf"), attn_weights)

        # Softmax
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Apply dropout (during training)
        if dropout_p > 0.0:
            self._rng_key, dropout_key = jax.random.split(self._rng_key)
            keep_prob = 1.0 - dropout_p
            dropout_mask = jax.random.bernoulli(dropout_key, keep_prob, attn_weights.shape)
            attn_weights = jnp.where(dropout_mask, attn_weights / keep_prob, 0.0)

        # Compute output: contract over k (key/value sequence dimension)
        output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, value)

        return output

    # ==================== Gradient Operations ====================

    def backward(self, loss: TensorLike) -> None:
        """In JAX, gradients are computed explicitly via jax.grad."""
        pass  # JAX uses functional gradients

    def zero_grad(self, parameters: Sequence[TensorLike]) -> None:
        """In JAX, gradients are not accumulated in-place."""
        pass  # Not needed in JAX

    def clip_grad_norm(
        self,
        parameters: Sequence[TensorLike],
        max_norm: float,
    ) -> TensorLike:
        """Clip gradients using optax."""
        _ensure_jax()
        # In JAX, this is typically done via optax.clip_by_global_norm
        total_norm = jnp.sqrt(sum(jnp.sum(p ** 2) for p in parameters))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef = jnp.minimum(clip_coef, 1.0)
        return [p * clip_coef for p in parameters], total_norm

    # ==================== JIT Compilation ====================

    def jit_compile(
        self,
        fn: Callable,
        static_argnums: Optional[Tuple[int, ...]] = None,
    ) -> Callable:
        """JIT compile using jax.jit."""
        _ensure_jax()
        return jax.jit(fn, static_argnums=static_argnums)

    # ==================== Distributed Operations ====================

    def all_reduce(
        self,
        tensor: TensorLike,
        op: str = "sum",
    ) -> TensorLike:
        """All-reduce using JAX's psum."""
        _ensure_jax()
        op_fn = {
            "sum": jax.lax.psum,
            "mean": lambda x, axis: jax.lax.pmean(x, axis),
            "max": jax.lax.pmax,
            "min": jax.lax.pmin,
        }.get(op, jax.lax.psum)

        return op_fn(tensor, axis_name="data")

    def broadcast(
        self,
        tensor: TensorLike,
        src: int = 0,
    ) -> TensorLike:
        """Broadcast using JAX."""
        _ensure_jax()
        return jax.lax.broadcast(tensor, ())

    # ==================== Memory Management ====================

    def memory_allocated(self) -> int:
        """Return memory usage."""
        _ensure_jax()
        try:
            devices = jax.devices("tpu") if self.is_available else jax.devices()
            if not devices:
                return 0
            stats = {}
            memory_stats_fn = getattr(devices[0], "memory_stats", None)
            if callable(memory_stats_fn):
                stats = memory_stats_fn() or {}
            return int(
                stats.get("bytes_in_use")
                or stats.get("bytes_used")
                or stats.get("allocated_bytes")
                or 0
            )
        except Exception:
            return 0

    def memory_reserved(self) -> int:
        """Return reserved memory."""
        _ensure_jax()
        try:
            devices = jax.devices("tpu") if self.is_available else jax.devices()
            if not devices:
                return 0
            stats = {}
            memory_stats_fn = getattr(devices[0], "memory_stats", None)
            if callable(memory_stats_fn):
                stats = memory_stats_fn() or {}
            return int(
                stats.get("bytes_limit")
                or stats.get("memory_limit")
                or stats.get("bytes_reserved")
                or 0
            )
        except Exception:
            return 0

    def empty_cache(self) -> None:
        """Clear JAX caches."""
        _ensure_jax()
        jax.clear_caches()

    def synchronize(self) -> None:
        """Block until all operations complete."""
        _ensure_jax()
        jax.block_until_ready

    # ==================== Context Managers ====================

    @contextmanager
    def no_grad(self):
        """JAX doesn't track gradients by default."""
        yield  # No-op in JAX

    @contextmanager
    def autocast(self, dtype: Optional[DType] = None):
        """Mixed precision context using jax.default_matmul_precision."""
        _ensure_jax()
        # JAX handles precision via config
        old_precision = jax.config.jax_default_matmul_precision
        try:
            jax.config.update("jax_default_matmul_precision", "bfloat16")
            yield
        finally:
            jax.config.update("jax_default_matmul_precision", old_precision)

    # ==================== Checkpoint Operations ====================

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        path: str,
    ) -> None:
        """Save checkpoint using orbax."""
        _ensure_jax()
        try:
            import orbax.checkpoint as ocp

            checkpointer = ocp.PyTreeCheckpointer()
            checkpointer.save(path, state)
        except ImportError:
            # Fallback to basic saving
            import pickle
            with open(path, "wb") as f:
                pickle.dump(jax.device_get(state), f)

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint."""
        _ensure_jax()
        try:
            import orbax.checkpoint as ocp

            checkpointer = ocp.PyTreeCheckpointer()
            return checkpointer.restore(path)
        except ImportError:
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)  # nosec B301 — trusted checkpoint

    # ==================== TPU-Specific Methods ====================

    def get_device_count(self) -> int:
        """Get number of TPU cores."""
        _ensure_jax()
        return jax.device_count()

    def pmap(self, fn: Callable, axis_name: str = "data") -> Callable:
        """Parallel map across TPU cores."""
        _ensure_jax()
        return jax.pmap(fn, axis_name=axis_name)

    def vmap(self, fn: Callable, in_axes: int = 0) -> Callable:
        """Vectorized map for batching."""
        _ensure_jax()
        return jax.vmap(fn, in_axes=in_axes)

    def value_and_grad(self, fn: Callable, argnums: int = 0) -> Callable:
        """Get function value and gradients."""
        _ensure_jax()
        return jax.value_and_grad(fn, argnums=argnums)

    def scan(
        self,
        fn: Callable,
        init: TensorLike,
        xs: TensorLike,
    ) -> Tuple[TensorLike, TensorLike]:
        """Efficient sequential scan (for SSM/Mamba)."""
        _ensure_jax()
        return jax.lax.scan(fn, init, xs)
