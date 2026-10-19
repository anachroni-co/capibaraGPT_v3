"""
Spanish Jokes Datasets for CapibaraGPT v3
==========================================

Specialized datasets for jokes and humor in Spanish.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import datasets
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

class SpanishJokesDataset:
    """Manager for Spanish jokes datasets."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "capibara" / "humor")
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def load_chistes_spanish_jokes(self) -> Dataset:
        """
        Load the CHISTES_spanish_jokes dataset with 2,419 jokes in Spanish.

        Returns:
            Dataset: Dataset with Spanish jokes
        """
        try:
            dataset = datasets.load_dataset(
                "mrm8488/CHISTES_spanish_jokes",
                cache_dir=self.cache_dir
            )
            logger.info(f"Loaded CHISTES_spanish_jokes dataset: {len(dataset['train'])} jokes")
            return dataset['train']
        except Exception as e:
            logger.error(f"Error loading CHISTES_spanish_jokes: {e}")
            raise

    def load_barcenas_humor_negro(self) -> Dataset:
        """
        Load the Barcenas-HumorNegro dataset with 500 dark humor jokes.

        Returns:
            Dataset: Dataset with dark humor jokes and explanations
        """
        try:
            dataset = datasets.load_dataset(
                "Danielbrdz/Barcenas-HumorNegro",
                cache_dir=self.cache_dir
            )
            logger.info(f"Loaded Barcenas-HumorNegro dataset: {len(dataset['train'])} jokes")
            return dataset['train']
        except Exception as e:
            logger.error(f"Error loading Barcenas-HumorNegro: {e}")
            raise

    def load_humor_qa(self) -> Dataset:
        """
        Load the HumorQA dataset with jokes categorized by humor type.

        Returns:
            Dataset: Dataset with jokes and humor type labels
        """
        try:
            dataset = datasets.load_dataset(
                "LenguajeNaturalAI/HumorQA",
                cache_dir=self.cache_dir
            )
            logger.info(f"Loaded HumorQA dataset: {len(dataset['train'])} categorized jokes")
            return dataset['train']
        except Exception as e:
            logger.error(f"Error loading HumorQA: {e}")
            raise

    def get_combined_dataset(self) -> Dataset:
        """
        Combine all joke datasets into one.

        Returns:
            Dataset: Combined dataset with all jokes
        """
        datasets_list = []

        # Load main jokes dataset
        try:
            chistes = self.load_chistes_spanish_jokes()
            # Normalize columns
            chistes = chistes.map(lambda x: {
                'joke': x.get('chiste', x.get('text', '')),
                'type': 'general',
                'source': 'CHISTES_spanish_jokes',
                'explanation': None
            })
            datasets_list.append(chistes)
        except Exception as e:
            logger.warning(f"Could not load CHISTES_spanish_jokes: {e}")

        # Load dark humor dataset
        try:
            humor_negro = self.load_barcenas_humor_negro()
            humor_negro = humor_negro.map(lambda x: {
                'joke': x.get('chiste', x.get('joke', '')),
                'type': 'dark_humor',
                'source': 'Barcenas-HumorNegro',
                'explanation': x.get('explicacion', x.get('explanation', None))
            })
            datasets_list.append(humor_negro)
        except Exception as e:
            logger.warning(f"Could not load Barcenas-HumorNegro: {e}")

        # Load HumorQA dataset
        try:
            humor_qa = self.load_humor_qa()
            humor_qa = humor_qa.map(lambda x: {
                'joke': x.get('chiste', x.get('joke', '')),
                'type': x.get('tipo_humor', x.get('humor_type', 'general')),
                'source': 'HumorQA',
                'explanation': x.get('explicacion', None)
            })
            datasets_list.append(humor_qa)
        except Exception as e:
            logger.warning(f"Could not load HumorQA: {e}")

        if not datasets_list:
            raise RuntimeError("Could not load any joke dataset")

        # Combine datasets
        combined = datasets.concatenate_datasets(datasets_list)
        logger.info(f"Combined dataset created: {len(combined)} total jokes")

        return combined

    def get_humor_categories(self) -> Dict[str, List[str]]:
        """
        Get available humor categories.

        Returns:
            Dict: Dictionary with categories and examples
        """
        return {
            'general': [
                'Traditional jokes',
                'Family humor',
                'Short jokes'
            ],
            'dark_humor': [
                'Dark humor',
                'Sarcasm',
                'Dark irony'
            ],
            'wordplay': [
                'Wordplay',
                'Puns',
                'Humorous tongue twisters'
            ],
            'comparison': [
                'Exaggerated comparisons',
                'Humorous metaphors'
            ],
            'rule_of_three': [
                'Rule of three structure',
                'Narrative patterns'
            ],
            'animation': [
                'Animating the inanimate',
                'Humorous personification'
            ]
        }

    def filter_by_type(self, dataset: Dataset, humor_type: str) -> Dataset:
        """
        Filter dataset by humor type.

        Args:
            dataset: Dataset to filter
            humor_type: Humor type to filter by

        Returns:
            Dataset: Filtered dataset
        """
        return dataset.filter(lambda x: x.get('type', '').lower() == humor_type.lower())

    def get_dataset_stats(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Args:
            dataset: Dataset to analyze

        Returns:
            Dict: Dataset statistics
        """
        stats = {
            'total_jokes': len(dataset),
            'humor_types': {},
            'sources': {},
            'avg_joke_length': 0,
            'with_explanation': 0
        }

        humor_types = {}
        sources = {}
        total_length = 0
        with_explanation = 0

        for item in dataset:
            # Count humor types
            humor_type = item.get('type', 'unknown')
            humor_types[humor_type] = humor_types.get(humor_type, 0) + 1

            # Count sources
            source = item.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1

            # Average length
            joke_text = item.get('joke', '')
            total_length += len(joke_text)

            # With explanation
            if item.get('explanation'):
                with_explanation += 1

        stats['humor_types'] = humor_types
        stats['sources'] = sources
        stats['avg_joke_length'] = total_length / len(dataset) if len(dataset) > 0 else 0
        stats['with_explanation'] = with_explanation

        return stats


# Convenience functions
def load_chistes_spanish_jokes(cache_dir: Optional[str] = None) -> Dataset:
    """Load the main Spanish jokes dataset."""
    manager = SpanishJokesDataset(cache_dir)
    return manager.load_chistes_spanish_jokes()

def load_barcenas_humor_negro(cache_dir: Optional[str] = None) -> Dataset:
    """Load the dark humor dataset."""
    manager = SpanishJokesDataset(cache_dir)
    return manager.load_barcenas_humor_negro()

def load_humor_qa(cache_dir: Optional[str] = None) -> Dataset:
    """Load the HumorQA dataset."""
    manager = SpanishJokesDataset(cache_dir)
    return manager.load_humor_qa()

def get_humor_categories() -> Dict[str, List[str]]:
    """Get available humor categories."""
    manager = SpanishJokesDataset()
    return manager.get_humor_categories()

# Dataset configuration for registry
spanish_jokes_datasets = {
    "chistes_spanish_jokes": {
        "type": "huggingface",
        "identifier": "mrm8488/CHISTES_spanish_jokes",
        "split": "train",
        "text_column": "chiste",
        "description": "2,419 Spanish jokes for humor model training",
        "category": "humor",
        "language": "es",
        "size_mb": 1.2,
        "num_samples": 2419
    },
    "barcenas_humor_negro": {
        "type": "huggingface",
        "identifier": "Danielbrdz/Barcenas-HumorNegro",
        "split": "train",
        "text_column": "chiste",
        "explanation_column": "explicacion",
        "description": "500 dark humor jokes in Spanish with explanations",
        "category": "humor",
        "subcategory": "dark_humor",
        "language": "es",
        "size_mb": 0.3,
        "num_samples": 500
    },
    "humor_qa": {
        "type": "huggingface",
        "identifier": "LenguajeNaturalAI/HumorQA",
        "split": "train",
        "text_column": "chiste",
        "type_column": "tipo_humor",
        "description": "Jokes categorized by humor type (wordplay, comparisons, etc.)",
        "category": "humor",
        "subcategory": "categorized",
        "language": "es",
        "size_mb": 0.8,
        "humor_types": ["wordplay", "comparison", "rule_of_three", "animation"]
    }
}
