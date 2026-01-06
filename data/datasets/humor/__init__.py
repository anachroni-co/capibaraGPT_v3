"""
CapibaraGPT-v2 Humor & Jokes Datasets
====================================

Datasets especializados en humor, chistes y contenido cómico en español.

Datasets Disponibles:
- CHISTES_spanish_jokes: 2,419 chistes en español
- Barcenas-HumorNegro: 500 chistes de humor negro con explicaciones
- HumorQA: Chistes categorizados por tipo de humor
- Twitter_Humor_ES: Tweets humorísticos anotados

Categorías de Humor:
- Chistes tradicionales
- Humor negro
- Juegos de palabras
- Comparaciones/exageraciones
- Regla de tres
- Animar lo inanimado
"""

from .spanish_jokes import *
from .humor_analysis import *

__all__ = [
    'spanish_jokes_datasets',
    'humor_analysis_datasets',
    'load_chistes_spanish_jokes',
    'load_barcenas_humor_negro',
    'load_humor_qa',
    'get_humor_categories',
    'analyze_humor_type',
]