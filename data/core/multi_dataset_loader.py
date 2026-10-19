"""
Loader for multiple datasets in CapibaraGPT v3.
"""

import numpy as np
from .dataset import Dataset
from .data_loader import DataLoader
from typing import Any, Dict, List, Optional, Union

class MultiDatasetLoader:
    """Class for loading multiple datasets."""

    def __init__(
        self,
        datasets: List[Dataset],
        batch_sizes: Optional[List[int]] = None,
        weights: Optional[List[float]] = None,
        **kwargs: Any
    ):
        """
        Initialize a new MultiDatasetLoader.

        Args:
            datasets: List of datasets
            batch_sizes: List of batch sizes
            weights: Weights for each dataset
            **kwargs: Additional arguments
        """
        self.datasets = datasets
        self.num_datasets = len(datasets)

        # Configure batch sizes
        if batch_sizes is None:
            batch_sizes = [32] * self.num_datasets
        self.batch_sizes = batch_sizes

        # Configure weights
        if weights is None:
            weights = [1.0 / self.num_datasets] * self.num_datasets
        self.weights = np.array(weights) / sum(weights)

        # Create loaders
        self.loaders = [
            DataLoader(dataset, batch_size, **kwargs)
            for dataset, batch_size in zip(datasets, batch_sizes)
        ]

        # Iterators
        self.iterators = None
        self.reset_iterators()

    def reset_iterators(self) -> None:
        """Reset the iterators."""
        self.iterators = [iter(loader) for loader in self.loaders]

    def __iter__(self):
        """Allows iteration over mixed batches."""
        self.reset_iterators()
        return self

    def __next__(self) -> Dict[str, Any]:
        """
        Get the next mixed batch.

        Returns:
            Mixed batch
        """
        # Select dataset based on weights
        dataset_idx = np.random.choice(
            self.num_datasets,
            p=self.weights
        )

        try:
            return next(self.iterators[dataset_idx])
        except StopIteration:
            # Reset iterator and retry
            self.iterators[dataset_idx] = iter(self.loaders[dataset_idx])
            return next(self.iterators[dataset_idx])

    def __len__(self) -> int:
        """
        Return the total number of batches.

        Returns:
            Total number of batches
        """
        return sum(len(loader) for loader in self.loaders)

    def get_weights(self) -> np.ndarray:
        """
        Return the normalized weights.

        Returns:
            Array with weights
        """
        return self.weights.copy()

    def set_weights(
        self,
        weights: List[float]
    ) -> None:
        """
        Update the weights.

        Args:
            weights: New weights
        """
        if len(weights) != self.num_datasets:
            raise ValueError("Invalid number of weights")
        self.weights = np.array(weights) / sum(weights)
