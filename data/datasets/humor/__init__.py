"""
CapibaraGPT v3 Humor & Jokes Datasets
====================================

Specialized datasets for humor, jokes, and comedic content in Spanish.

Available Datasets:
- CHISTES_spanish_jokes: 2,419 jokes in Spanish
- Barcenas-HumorNegro: 500 dark humor jokes with explanations
- HumorQA: Jokes categorized by humor type
- Twitter_Humor_ES: Annotated humorous tweets

Humor Categories:
- Traditional jokes
- Dark humor
- Wordplay
- Comparisons/exaggerations
- Rule of three
- Animating the inanimate
"""

from .spanish_jokes import (
    SpanishJokesDataset,
    load_chistes_spanish_jokes,
    load_barcenas_humor_negro,
    load_humor_qa,
    get_humor_categories,
    spanish_jokes_datasets,
)
from .humor_analysis import (
    HumorType,
    HumorAnalysis,
    HumorAnalyzer,
    HumorMetrics,
    analyze_humor_type,
    get_humor_distribution,
    humor_analysis_datasets,
)

__all__ = [
    'spanish_jokes_datasets',
    'humor_analysis_datasets',
    'load_chistes_spanish_jokes',
    'load_barcenas_humor_negro',
    'load_humor_qa',
    'get_humor_categories',
    'analyze_humor_type',
]
