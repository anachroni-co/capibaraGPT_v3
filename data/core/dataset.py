"""
Dataset base class for CapibaraGPT v3.
"""

from typing import Any, Dict, Iterator, List, Optional, Union
import numpy as np


class Dataset:
    """Base class for datasets in CapibaraGPT."""

    def __init__(
        self,
        data: Optional[Union[List, np.ndarray]] = None,
        **kwargs: Any
    ):
        """
        Initialize a dataset.

        Args:
            data: Dataset content
            **kwargs: Additional arguments
        """
        self.data = data if data is not None else []
        self._length = len(self.data) if hasattr(self.data, '__len__') else 0

    def __len__(self) -> int:
        """Return dataset length."""
        return self._length

    def __getitem__(self, idx: int) -> Any:
        """Get item by index."""
        return self.data[idx]

    def __iter__(self) -> Iterator:
        """Iterate over dataset."""
        return iter(self.data)
