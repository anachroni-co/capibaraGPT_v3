"""
Data Management Module - CapibaraGPT v3

Provides dataset registry and data orchestration capabilities.

Components:
- dataset_registry: Functions to load and query available datasets
- ultra_data_orchestrator: Advanced data orchestration system
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sequence packing (pretraining stream packing + SFT example packing)
from .packing import (
    pack_token_streams,
    pack_examples,
    PackedExample,
    PackedDataset,
    packing_efficiency,
)

# Core dataset registry functions
from .dataset_registry import (
    load_registry,
    get_dataset_info,
    list_available_datasets,
)

# Ultra data orchestrator (optional advanced features)
ORCHESTRATOR_AVAILABLE = False
UltraDataOrchestrator = None
UltraDataConfig = None

try:
    from .ultra_data_orchestrator import (
        UltraDataOrchestrator,
        UltraDataConfig,
        create_ultra_data_system,
    )
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    create_ultra_data_system = None


def get_data_status() -> Dict[str, Any]:
    """Get status of data module components."""
    return {
        "registry_available": True,
        "orchestrator_available": ORCHESTRATOR_AVAILABLE,
        "available_datasets": list_available_datasets(),
    }


__all__ = [
    # Registry functions
    "load_registry",
    "get_dataset_info",
    "list_available_datasets",
    # Orchestrator
    "UltraDataOrchestrator",
    "UltraDataConfig",
    "create_ultra_data_system",
    "ORCHESTRATOR_AVAILABLE",
    # Status
    "get_data_status",
]
