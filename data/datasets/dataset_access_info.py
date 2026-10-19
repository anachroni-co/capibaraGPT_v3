"""Compatibility shim for dataset access metadata."""

from data.configs.dataset_access_info import (  # noqa: F401
    AccessType,
    DatasetAccess,
    PSYCHOLOGY_DATASETS,
    LEGAL_DATASETS,
    PHYSICS_DATASETS,
    LINUX_DATASETS,
    get_dataset_access_info,
    get_datasets_by_category,
    get_preprocessing_pipeline,
)

DataAccess = DatasetAccess

__all__ = [
    "AccessType",
    "DataAccess",
    "DatasetAccess",
    "PSYCHOLOGY_DATASETS",
    "LEGAL_DATASETS",
    "PHYSICS_DATASETS",
    "LINUX_DATASETS",
    "get_dataset_access_info",
    "get_datasets_by_category",
    "get_preprocessing_pipeline",
]
