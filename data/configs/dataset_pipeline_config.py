"""Dataset configuration integrated with the existing training system."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

class ModelScale(str, Enum):
    """Supported model scales."""
    MICRO_300M = "300M"
    SMALL_3B = "3B"
    SMALL_7B = "7B"
    MEDIUM_30B = "30B"

@dataclass
class DatasetScale:
    """Dataset configuration for each model scale."""
    model_scale: ModelScale
    dataset_names: List[str]
    total_tokens: str
    batch_size: int
    sequence_length: int
    shuffle_buffer: int
    cache_strategy: str
    preprocessing_workers: int

    # Integration with consensus distillation
    synthetic_data_ratio: float = 0.0
    quality_filtering_threshold: float = 0.0

DATASET_CONFIGURATIONS = {
    ModelScale.MICRO_300M: DatasetScale(
        model_scale=ModelScale.MICRO_300M,
        dataset_names=["wikipedia_small", "simple_books"],
        total_tokens="1B",
        batch_size=512,
        sequence_length=512,
        shuffle_buffer=10000,
        cache_strategy="memory",
        preprocessing_workers=4,
        synthetic_data_ratio=0.0,
        quality_filtering_threshold=0.7
    ),

    ModelScale.SMALL_3B: DatasetScale(
        model_scale=ModelScale.SMALL_3B,
        dataset_names=["redpajama_subset", "books3", "github_code_clean"],
        total_tokens="100B",
        batch_size=1024,
        sequence_length=2048,
        shuffle_buffer=50000,
        cache_strategy="disk",
        preprocessing_workers=8,
        synthetic_data_ratio=0.1,  # For consensus distillation
        quality_filtering_threshold=0.8
    ),

    ModelScale.SMALL_7B: DatasetScale(
        model_scale=ModelScale.SMALL_7B,
        dataset_names=["redpajama_full", "pile_subset", "arxiv_papers"],
        total_tokens="300B",
        batch_size=2048,
        sequence_length=2048,
        shuffle_buffer=100000,
        cache_strategy="disk",
        preprocessing_workers=16,
        synthetic_data_ratio=0.2,
        quality_filtering_threshold=0.85
    ),

    ModelScale.MEDIUM_30B: DatasetScale(
        model_scale=ModelScale.MEDIUM_30B,
        dataset_names=["pile_full", "refinedweb", "dolma_subset", "s2orc"],
        total_tokens="1T",
        batch_size=4096,
        sequence_length=4096,
        shuffle_buffer=200000,
        cache_strategy="distributed",
        preprocessing_workers=32,
        synthetic_data_ratio=0.3,  # More synthetic data
        quality_filtering_threshold=0.9
    ),
}
