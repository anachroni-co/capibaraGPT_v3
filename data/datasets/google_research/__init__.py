"""
Google Research Datasets for CapibaraGPT v3

Access to Google Research datasets including multimodal and specialized datasets.
"""

import importlib
from typing import Any

_LAZY_MODULES = {
    "GoogleResearchDatasets": ("data.datasets.google_research.google_research_datasets", "GoogleResearchDatasets"),
}

__all__ = list(_LAZY_MODULES.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        module_name, attr_name = _LAZY_MODULES[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
