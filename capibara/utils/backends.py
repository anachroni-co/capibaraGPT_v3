"""
Unified tensor backend for numpy and torch operations.

Provides a singleton that detects available backends once and offers
a consistent interface for tensor operations across the codebase.
"""

from typing import Any, List, Optional, Set


class TensorBackend:
    """
    Unified tensor operations for numpy and torch.

    Detects available backend once and provides consistent interface.
    Thread-safe singleton pattern.
    """

    _instance = None
    _np = None
    _torch = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_backends()
        return cls._instance

    def _init_backends(self):
        """Initialize available backends."""
        try:
            import numpy as np
            self._np = np
        except ImportError:
            self._np = None

        try:
            import torch
            self._torch = torch
        except ImportError:
            self._torch = None

    @property
    def np(self):
        """Get numpy module if available."""
        return self._np

    @property
    def torch(self):
        """Get torch module if available."""
        return self._torch

    @property
    def has_numpy(self) -> bool:
        return self._np is not None

    @property
    def has_torch(self) -> bool:
        return self._torch is not None

    def is_numpy(self, tensor: Any) -> bool:
        """Check if tensor is numpy array."""
        return self._np is not None and isinstance(tensor, self._np.ndarray)

    def is_torch(self, tensor: Any) -> bool:
        """Check if tensor is torch tensor."""
        return self._torch is not None and hasattr(tensor, 'is_cuda')

    def concat(self, a: Any, b: Any, dim: int = 2) -> Any:
        """Concatenate tensors along dimension."""
        if self.is_numpy(a):
            return self._np.concatenate([a, b], axis=dim)
        elif self.is_torch(a):
            return self._torch.cat([a, b], dim=dim)
        return b

    def slice(self, tensor: Any, start: int, end: int, dim: int = 2) -> Any:
        """Slice tensor along dimension."""
        if tensor is None:
            return None
        if self.is_numpy(tensor):
            if dim == 2:
                return tensor[:, :, start:end, :]
            return tensor
        elif self.is_torch(tensor):
            if dim == 2:
                return tensor[:, :, start:end, :]
            return tensor
        return tensor

    def index(self, tensor: Any, indices: List[int], dim: int = 2) -> Any:
        """Index tensor along dimension."""
        if tensor is None:
            return None
        if self.is_numpy(tensor):
            return tensor[:, :, indices, :]
        elif self.is_torch(tensor):
            return tensor[:, :, indices, :]
        return tensor

    def zeros_like(self, tensor: Any) -> Any:
        """Create zeros with same shape/type."""
        if self.is_numpy(tensor):
            return self._np.zeros_like(tensor)
        elif self.is_torch(tensor):
            return self._torch.zeros_like(tensor)
        return None

    def get_shape(self, tensor: Any, dim: int) -> int:
        """Get size along dimension."""
        if tensor is None:
            return 0
        if hasattr(tensor, 'shape'):
            return tensor.shape[dim]
        return 0

    def average_pool(self, tensor: Any, factor: int, dim: int = 2) -> Any:
        """Average pool along dimension."""
        if tensor is None or factor <= 1:
            return tensor

        if self.is_numpy(tensor):
            seq_len = tensor.shape[dim]
            new_len = seq_len // factor
            if new_len == 0:
                return tensor

            shape = list(tensor.shape)
            shape[dim] = new_len
            shape.insert(dim + 1, factor)
            reshaped = tensor[:, :, :new_len * factor, :].reshape(shape)
            return reshaped.mean(axis=dim + 1)

        elif self.is_torch(tensor):
            seq_len = tensor.shape[dim]
            new_len = seq_len // factor
            if new_len == 0:
                return tensor

            shape = list(tensor.shape)
            shape[dim] = new_len
            shape.insert(dim + 1, factor)
            reshaped = tensor[:, :, :new_len * factor, :].reshape(shape)
            return reshaped.mean(dim=dim + 1)

        return tensor

    def percentile(self, array: Any, p: float) -> float:
        """Compute percentile."""
        if self.is_numpy(array):
            return float(self._np.percentile(array, p))
        elif self.is_torch(array):
            k = int((1 - p / 100) * len(array))
            return float(self._torch.kthvalue(array.flatten(), k).values)
        return 0.0

    def copy(self, tensor: Any) -> Any:
        """Create a copy of the tensor."""
        if self.is_numpy(tensor):
            return tensor.copy()
        elif self.is_torch(tensor):
            return tensor.clone()
        return tensor

    def boost_indices(
        self, logits: Any, indices: Set[int], delta: float
    ) -> Any:
        """Boost logits at specified indices by delta."""
        if logits is None:
            return logits

        modified = self.copy(logits)
        is_batched = len(modified.shape) > 1

        for idx in indices:
            if is_batched:
                modified[:, idx] += delta
            else:
                modified[idx] += delta

        return modified

    def randint(
        self, low: int, high: int, shape: tuple, device: str = 'cpu'
    ) -> Optional[Any]:
        """Generate random integers in range [low, high)."""
        if self.has_numpy:
            return self._np.random.randint(low, high, shape)
        elif self.has_torch:
            return self._torch.randint(low, high, shape, device=device)
        return None

    def randn(self, *shape, device: str = 'cpu') -> Optional[Any]:
        """Generate random normal values."""
        if self.has_numpy:
            return self._np.random.randn(*shape)
        elif self.has_torch:
            return self._torch.randn(*shape, device=device)
        return None

    def concatenate(self, arrays: List[Any], axis: int = 0) -> Optional[Any]:
        """Concatenate arrays along axis."""
        if not arrays:
            return None
        first = arrays[0]
        if self.is_numpy(first):
            return self._np.concatenate(arrays, axis=axis)
        elif self.is_torch(first):
            return self._torch.cat(arrays, dim=axis)
        return None

    def to_numpy(self, tensor: Any) -> Optional[Any]:
        """Convert tensor to numpy array."""
        if self.is_numpy(tensor):
            return tensor
        elif self.is_torch(tensor):
            return tensor.detach().cpu().numpy()
        elif self.has_numpy and hasattr(tensor, '__array__'):
            return self._np.array(tensor)
        return None


# Global singleton instance
_backend: Optional[TensorBackend] = None


def get_backend() -> TensorBackend:
    """Get the global TensorBackend instance."""
    global _backend
    if _backend is None:
        _backend = TensorBackend()
    return _backend


def boost_logits(logits: Any, token_ids: Set[int], delta: float) -> Any:
    """
    Boost logits for specified token IDs.

    Convenience function that uses the global backend.

    Args:
        logits: Logit tensor (numpy or torch)
        token_ids: Set of token indices to boost
        delta: Amount to add to selected logits

    Returns:
        Modified logits tensor
    """
    backend = get_backend()
    if not backend.has_numpy and not backend.has_torch:
        return logits
    return backend.boost_indices(logits, token_ids, delta)
