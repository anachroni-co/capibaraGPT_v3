"""
Spanish Jokes Datasets for CapibaraGPT-v2
==========================================

Datasets especializados en chistes y humor en español.
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
    """Gestor para datasets de chistes en español."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "capibara" / "humor")
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
    def load_chistes_spanish_jokes(self) -> Dataset:
        """
        Carga el dataset CHISTES_spanish_jokes con 2,419 chistes en español.
        
        Returns:
            Dataset: Dataset con chistes en español
        """
        try:
            dataset = datasets.load_dataset(
                "mrm8488/CHISTES_spanish_jokes",
                cache_dir=self.cache_dir
            )
            logger.info(f"Cargado dataset CHISTES_spanish_jokes: {len(dataset['train'])} chistes")
            return dataset['train']
        except Exception as e:
            logger.error(f"Error cargando CHISTES_spanish_jokes: {e}")
            raise
    
    def load_barcenas_humor_negro(self) -> Dataset:
        """
        Carga el dataset Barcenas-HumorNegro con 500 chistes de humor negro.
        
        Returns:
            Dataset: Dataset con chistes de humor negro y explicaciones
        """
        try:
            dataset = datasets.load_dataset(
                "Danielbrdz/Barcenas-HumorNegro",
                cache_dir=self.cache_dir
            )
            logger.info(f"Cargado dataset Barcenas-HumorNegro: {len(dataset['train'])} chistes")
            return dataset['train']
        except Exception as e:
            logger.error(f"Error cargando Barcenas-HumorNegro: {e}")
            raise
    
    def load_humor_qa(self) -> Dataset:
        """
        Carga el dataset HumorQA con chistes categorizados por tipo de humor.
        
        Returns:
            Dataset: Dataset con chistes y etiquetas de tipo de humor
        """
        try:
            dataset = datasets.load_dataset(
                "LenguajeNaturalAI/HumorQA",
                cache_dir=self.cache_dir
            )
            logger.info(f"Cargado dataset HumorQA: {len(dataset['train'])} chistes categorizados")
            return dataset['train']
        except Exception as e:
            logger.error(f"Error cargando HumorQA: {e}")
            raise
    
    def get_combined_dataset(self) -> Dataset:
        """
        Combina todos los datasets de chistes en uno solo.
        
        Returns:
            Dataset: Dataset combinado con todos los chistes
        """
        datasets_list = []
        
        # Cargar dataset principal de chistes
        try:
            chistes = self.load_chistes_spanish_jokes()
            # Normalizar columnas
            chistes = chistes.map(lambda x: {
                'joke': x.get('chiste', x.get('text', '')),
                'type': 'general',
                'source': 'CHISTES_spanish_jokes',
                'explanation': None
            })
            datasets_list.append(chistes)
        except Exception as e:
            logger.warning(f"No se pudo cargar CHISTES_spanish_jokes: {e}")
        
        # Cargar dataset de humor negro
        try:
            humor_negro = self.load_barcenas_humor_negro()
            humor_negro = humor_negro.map(lambda x: {
                'joke': x.get('chiste', x.get('joke', '')),
                'type': 'humor_negro',
                'source': 'Barcenas-HumorNegro',
                'explanation': x.get('explicacion', x.get('explanation', None))
            })
            datasets_list.append(humor_negro)
        except Exception as e:
            logger.warning(f"No se pudo cargar Barcenas-HumorNegro: {e}")
        
        # Cargar dataset HumorQA
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
            logger.warning(f"No se pudo cargar HumorQA: {e}")
        
        if not datasets_list:
            raise RuntimeError("No se pudo cargar ningún dataset de chistes")
        
        # Combinar datasets
        combined = datasets.concatenate_datasets(datasets_list)
        logger.info(f"Dataset combinado creado: {len(combined)} chistes totales")
        
        return combined
    
    def get_humor_categories(self) -> Dict[str, List[str]]:
        """
        Obtiene las categorías de humor disponibles.
        
        Returns:
            Dict: Diccionario con categorías y ejemplos
        """
        return {
            'general': [
                'Chistes tradicionales',
                'Humor familiar',
                'Chistes cortos'
            ],
            'humor_negro': [
                'Humor negro',
                'Sarcasmo',
                'Ironía oscura'
            ],
            'juego_palabras': [
                'Juegos de palabras',
                'Calambures',
                'Trabalenguas humorísticos'
            ],
            'comparacion': [
                'Comparaciones exageradas',
                'Metáforas humorísticas'
            ],
            'regla_tres': [
                'Estructura de regla de tres',
                'Patrones narrativos'
            ],
            'animacion': [
                'Animar lo inanimado',
                'Personificación humorística'
            ]
        }
    
    def filter_by_type(self, dataset: Dataset, humor_type: str) -> Dataset:
        """
        Filtra el dataset por tipo de humor.
        
        Args:
            dataset: Dataset a filtrar
            humor_type: Tipo de humor a filtrar
            
        Returns:
            Dataset: Dataset filtrado
        """
        return dataset.filter(lambda x: x.get('type', '').lower() == humor_type.lower())
    
    def get_dataset_stats(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Obtiene statistics del dataset.
        
        Args:
            dataset: Dataset a analizar
            
        Returns:
            Dict: Estadísticas del dataset
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
            # Contar tipos de humor
            humor_type = item.get('type', 'unknown')
            humor_types[humor_type] = humor_types.get(humor_type, 0) + 1
            
            # Contar fuentes
            source = item.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
            
            # Longitud promedio
            joke_text = item.get('joke', '')
            total_length += len(joke_text)
            
            # Con explicación
            if item.get('explanation'):
                with_explanation += 1
        
        stats['humor_types'] = humor_types
        stats['sources'] = sources
        stats['avg_joke_length'] = total_length / len(dataset) if len(dataset) > 0 else 0
        stats['with_explanation'] = with_explanation
        
        return stats


# Funciones de conveniencia
def load_chistes_spanish_jokes(cache_dir: Optional[str] = None) -> Dataset:
    """Loads el dataset principal de chistes españoles."""
    manager = SpanishJokesDataset(cache_dir)
    return manager.load_chistes_spanish_jokes()

def load_barcenas_humor_negro(cache_dir: Optional[str] = None) -> Dataset:
    """Loads el dataset de humor negro."""
    manager = SpanishJokesDataset(cache_dir)
    return manager.load_barcenas_humor_negro()

def load_humor_qa(cache_dir: Optional[str] = None) -> Dataset:
    """Loads el dataset HumorQA."""
    manager = SpanishJokesDataset(cache_dir)
    return manager.load_humor_qa()

def get_humor_categories() -> Dict[str, List[str]]:
    """Gets las categorías de humor disponibles."""
    manager = SpanishJokesDataset()
    return manager.get_humor_categories()

# Dataset configuration for registry
spanish_jokes_datasets = {
    "chistes_spanish_jokes": {
        "type": "huggingface",
        "identifier": "mrm8488/CHISTES_spanish_jokes",
        "split": "train",
        "text_column": "chiste",
        "description": "2,419 chistes en español para entrenamiento de modelos de humor",
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
        "description": "500 chistes de humor negro en español con explicaciones",
        "category": "humor",
        "subcategory": "humor_negro",
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
        "description": "Chistes categorizados por tipo de humor (juegos de palabras, comparaciones, etc.)",
        "category": "humor",
        "subcategory": "categorized",
        "language": "es",
        "size_mb": 0.8,
        "humor_types": ["juego_palabras", "comparacion", "regla_tres", "animacion"]
    }
}