"""
CPU Backend Implementation (NumPy)

For development, testing, and debugging.
No GPU/TPU dependencies required.
"""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .base import BackendConfig, ComputeBackend, DType, TensorLike
from core.decorators import get_causal_mask

# Dtype mapping
DTYPE_MAP = {
    DType.FLOAT32: np.float32,
    DType.FLOAT16: np.float16,
    DType.BFLOAT16: np.float32,  # NumPy doesn't support bfloat16, use float32
    DType.INT32: np.int32,
    DType.INT64: np.int64,
    DType.BOOL: np.bool_,
}


class CPUBackend(ComputeBackend):
    """
    CPU Backend using NumPy.

    Used for:
    - Development and debugging
    - Unit testing
    - Environments without GPU/TPU
    - Reference implementation

    Not optimized for production training.
    """

    def __init__(self, config: Optional[BackendConfig] = None):
        super().__init__(config)
        self._rng = np.random.default_rng(42)

    @property
    def name(self) -> str:
        return "cpu"

    @property
    def is_available(self) -> bool:
        """CPU is always available."""
        return True

    def initialize(self) -> None:
        """Initialize CPU backend."""
        self._device = "cpu"
        self._initialized = True

    def shutdown(self) -> None:
        """No cleanup needed for CPU."""
        self._initialized = False

    # ==================== Tensor Operations ====================

    def create_tensor(
        self,
        data: Any,
        dtype: Optional[DType] = None,
        device: Optional[str] = None,
        requires_grad: bool = False,
    ) -> TensorLike:
        np_dtype = DTYPE_MAP.get(dtype, np.float32) if dtype else None
        return np.array(data, dtype=np_dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> TensorLike:
        np_dtype = DTYPE_MAP.get(dtype, np.float32) if dtype else np.float32
        return np.zeros(shape, dtype=np_dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> TensorLike:
        np_dtype = DTYPE_MAP.get(dtype, np.float32) if dtype else np.float32
        return np.ones(shape, dtype=np_dtype)

    def randn(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> TensorLike:
        np_dtype = DTYPE_MAP.get(dtype, np.float32) if dtype else np.float32
        return self._rng.standard_normal(shape).astype(np_dtype)

    def to_device(self, tensor: TensorLike, device: str) -> TensorLike:
        return tensor  # No-op for CPU

    def to_numpy(self, tensor: TensorLike) -> np.ndarray:
        return np.asarray(tensor)

    # ==================== Math Operations ====================

    def matmul(self, a: TensorLike, b: TensorLike) -> TensorLike:
        return np.matmul(a, b)

    def add(self, a: TensorLike, b: TensorLike) -> TensorLike:
        return np.add(a, b)

    def mul(self, a: TensorLike, b: TensorLike) -> TensorLike:
        return np.multiply(a, b)

    def softmax(self, x: TensorLike, axis: int = -1) -> TensorLike:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def layer_norm(
        self,
        x: TensorLike,
        normalized_shape: Tuple[int, ...],
        weight: Optional[TensorLike] = None,
        bias: Optional[TensorLike] = None,
        eps: float = 1e-5,
    ) -> TensorLike:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + eps)

        if weight is not None:
            x_norm = x_norm * weight
        if bias is not None:
            x_norm = x_norm + bias

        return x_norm

    def gelu(self, x: TensorLike) -> TensorLike:
        # Numerically stable GELU approximation with input clipping.
        x = np.asarray(x)
        x_clip = np.clip(x, -10.0, 10.0)
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_clip + 0.044715 * x_clip ** 3)))

    def silu(self, x: TensorLike) -> TensorLike:
        return x * (1 / (1 + np.exp(-x)))

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
        """Reference implementation of scaled dot-product attention."""
        batch_size, num_heads, seq_len, head_dim = query.shape
        scale = scale or (head_dim ** -0.5)

        # Compute attention scores
        attn_weights = np.einsum("bhqd,bhkd->bhqk", query, key) * scale

        # Apply causal mask (uses cached mask for common sequence lengths)
        if is_causal:
            causal_mask = get_causal_mask(seq_len)
            attn_weights = np.where(
                causal_mask[None, None, :, :] == 0,
                -np.inf,
                attn_weights,
            )

        # Apply provided mask
        if mask is not None:
            attn_weights = np.where(mask == 0, -np.inf, attn_weights)

        # Softmax
        attn_weights = self.softmax(attn_weights, axis=-1)

        # Dropout (random drop during training)
        if dropout_p > 0.0:
            dropout_mask = self._rng.random(attn_weights.shape) > dropout_p
            attn_weights = attn_weights * dropout_mask / (1 - dropout_p)

        # Compute output: contract over k (key/value sequence dimension)
        return np.einsum("bhqk,bhkd->bhqd", attn_weights, value)

    # ==================== Gradient Operations ====================

    def backward(self, loss: TensorLike) -> None:
        """No automatic gradients in NumPy."""
        pass

    def zero_grad(self, parameters: Sequence[TensorLike]) -> None:
        """No gradients in NumPy."""
        pass

    def clip_grad_norm(
        self,
        parameters: Sequence[TensorLike],
        max_norm: float,
    ) -> TensorLike:
        """No gradients in NumPy."""
        return 0.0

    # ==================== JIT Compilation ====================

    def jit_compile(
        self,
        fn: Callable,
        static_argnums: Optional[Tuple[int, ...]] = None,
    ) -> Callable:
        """No JIT for NumPy, return function as-is."""
        return fn

    # ==================== Distributed Operations ====================

    def all_reduce(
        self,
        tensor: TensorLike,
        op: str = "sum",
    ) -> TensorLike:
        """No distributed for CPU."""
        return tensor

    def broadcast(
        self,
        tensor: TensorLike,
        src: int = 0,
    ) -> TensorLike:
        """No distributed for CPU."""
        return tensor

    # ==================== Memory Management ====================

    def memory_allocated(self) -> int:
        """Return 0 for CPU."""
        return 0

    def memory_reserved(self) -> int:
        """Return 0 for CPU."""
        return 0

    def empty_cache(self) -> None:
        """No-op for CPU."""
        pass

    def synchronize(self) -> None:
        """No-op for CPU."""
        pass

    # ==================== Context Managers ====================

    @contextmanager
    def no_grad(self):
        """No-op for CPU."""
        yield

    @contextmanager
    def autocast(self, dtype: Optional[DType] = None):
        """No-op for CPU."""
        yield

    # ==================== Checkpoint Operations ====================

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        path: str,
    ) -> None:
        """Save using NumPy."""
        np.savez(path, **state)

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load NumPy checkpoint.

        Warning: Uses allow_pickle=True. Only load checkpoints from trusted sources.
        """
        data = np.load(path, allow_pickle=True)  # nosec B301 — trusted checkpoint
        return dict(data)

    # ==================== Utility Methods ====================

    def get_device_count(self) -> int:
        """CPU has 1 'device'."""
        return 1
