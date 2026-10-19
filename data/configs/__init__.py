"""
CapibaraGPT v3 Data Configurations
Configurations and metadata for datasets
"""

import importlib
from typing import Any

_LAZY_MODULES = {
    "dataset_access_config": "data.configs.dataset_access_config",
    "dataset_pipeline_config": "data.configs.dataset_pipeline_config",
    "dataset_access_info": "data.configs.dataset_access_info",
}

__all__ = list(_LAZY_MODULES.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
