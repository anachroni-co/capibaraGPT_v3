"""
DataLoader for CapibaraGPT v3.
"""

from typing import Any, Iterator, List, Optional
import numpy as np

from .dataset import Dataset


class DataLoader:
    """DataLoader for batching datasets."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False,
        **kwargs: Any
    ):
        """
        Initialize DataLoader.

        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle data
            drop_last: Whether to drop incomplete last batch
            **kwargs: Additional arguments
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._indices: Optional[np.ndarray] = None

    def __len__(self) -> int:
        """Return number of batches."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator:
        """Iterate over batches."""
        n = len(self.dataset)
        indices = np.arange(n)

        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, n, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            yield self._get_batch(batch_indices)

    def _get_batch(self, indices: np.ndarray) -> List[Any]:
        """Get batch by indices."""
        return [self.dataset[i] for i in indices]
