"""
CapibaraGPT v3 Academic Datasets
Academic, educational and research datasets
"""

import importlib
from typing import Any

_LAZY_MODULES = {
    "academic_code_datasets": "data.datasets.academic.academic_code_datasets",
    "institutional_datasets": "data.datasets.academic.institutional_datasets",
    "wiki_datasets": "data.datasets.academic.wiki_datasets",
    "psychology_datasets": "data.datasets.academic.psychology_datasets",
}

__all__ = list(_LAZY_MODULES.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
