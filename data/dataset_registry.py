#!/usr/bin/env python3
"""
Dataset Registry Loader for CapibaraGPT
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_registry(registry_path: str = "capibara/data/dataset_registry.json") -> Dict[str, Any]:
    """Load the dataset registry from JSON file."""
    try:
        path = Path(registry_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            logger.info(f"Loaded dataset registry with {len(registry)} entries")
            return registry
        else:
            logger.warning(f"Registry file not found: {registry_path}")
            return {}
    except Exception as e:
        logger.error(f"Failed to load registry: {e}")
        return {}

def get_dataset_info(dataset_name: str, registry_path: str = "capibara/data/dataset_registry.json") -> Dict[str, Any]:
    """Get information about a specific dataset."""
    registry = load_registry(registry_path)
    return registry.get(dataset_name, {})

def list_available_datasets(registry_path: str = "capibara/data/dataset_registry.json") -> list:
    """List all available datasets in the registry."""
    registry = load_registry(registry_path)
    return list(registry.keys())