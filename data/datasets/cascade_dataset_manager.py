#!/usr/bin/env python3
"""
Cascade Dataset Manager for CapibaraGPT
Manages multiple datasets for training
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

class CascadeDatasetManager:
    """Manages cascade of datasets for training."""
    
    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = registry_path or "capibara/data/dataset_registry.json"
        self.datasets = {}
        self.loaded_data = []
        
    def load_registry(self) -> Dict[str, Any]:
        """Load the dataset registry."""
        try:
            registry_path = Path(self.registry_path)
            if registry_path.exists():
                with open(registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Registry file not found: {registry_path}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return {}
    
    def load_datasets(self) -> List[str]:
        """Load all available datasets."""
        registry = self.load_registry()
        all_texts = []
        
        for dataset_name, config in registry.items():
            try:
                logger.info(f"Loading dataset: {dataset_name}")
                
                if config.get('type') == 'huggingface':
                    texts = self._load_huggingface_dataset(config)
                    all_texts.extend(texts)
                    logger.info(f"Loaded {len(texts)} samples from {dataset_name}")
                    
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
                continue
        
        self.loaded_data = all_texts
        logger.info(f"Total loaded samples: {len(all_texts)}")
        return all_texts
    
    def _load_huggingface_dataset(self, config: Dict[str, Any]) -> List[str]:
        """Load a HuggingFace dataset."""
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(
                config['identifier'],
                split=config.get('split', 'train[:1000]'),
                trust_remote_code=True
            )
            
            text_column = config.get('text_column', 'text')
            if text_column in dataset.column_names:
                return [item[text_column] for item in dataset if item.get(text_column)]
            else:
                logger.warning(f"Text column '{text_column}' not found in dataset")
                return []
                
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset: {e}")
            return []
    
    def get_training_data(self) -> List[str]:
        """Get training data."""
        if not self.loaded_data:
            return self.load_datasets()
        return self.loaded_data