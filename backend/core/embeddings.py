#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo de embeddings para análisis de queries y similitud semántica.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Modelo de embeddings para análisis de queries y similitud semántica.
    Incluye caché y optimizaciones para uso en producción.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 cache_dir: str = "backend/data/embeddings_cache"):
        """
        Inicializa el modelo de embeddings.
        
        Args:
            model_name: Nombre del modelo de sentence-transformers
            cache_dir: Directorio para caché de embeddings
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Caché en memoria
        self._memory_cache = {}
        self._cache_file = self.cache_dir / "embeddings_cache.pkl"
        
        # Cargar modelo
        self.model = self._load_model()
        
        # Cargar caché persistente
        self._load_cache()
        
        logger.info(f"EmbeddingModel inicializado con {model_name}")
    
    def _load_model(self) -> SentenceTransformer:
        """Carga el modelo de sentence-transformers."""
        try:
            model = SentenceTransformer(self.model_name)
            logger.info(f"Modelo {self.model_name} cargado exitosamente")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo de embeddings: {e}")
            raise
    
    def _load_cache(self):
        """Carga el caché persistente desde disco."""
        try:
            if self._cache_file.exists():
                with open(self._cache_file, 'rb') as f:
                    self._memory_cache = pickle.load(f)
                logger.info(f"Caché cargado: {len(self._memory_cache)} embeddings")
            else:
                self._memory_cache = {}
                logger.info("Caché vacío, se creará nuevo")
        except Exception as e:
            logger.error(f"Error cargando caché: {e}")
            self._memory_cache = {}
    
    def _save_cache(self):
        """Guarda el caché persistente en disco."""
        try:
            with open(self._cache_file, 'wb') as f:
                pickle.dump(self._memory_cache, f)
            logger.debug(f"Caché guardado: {len(self._memory_cache)} embeddings")
        except Exception as e:
            logger.error(f"Error guardando caché: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Genera clave de caché para un texto."""
        return f"{self.model_name}:{hash(text)}"
    
    def encode(self, texts: List[str], batch_size: int = 32, 
               use_cache: bool = True) -> np.ndarray:
        """
        Codifica textos a embeddings.
        
        Args:
            texts: Lista de textos a codificar
            batch_size: Tamaño del batch para procesamiento
            use_cache: Si usar caché para acelerar
            
        Returns:
            Array numpy con embeddings
        """
        if not texts:
            return np.array([])
        
        # Normalizar entrada
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        texts_to_encode = []
        indices_to_encode = []
        
        # Verificar caché
        if use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._memory_cache:
                    embeddings.append(self._memory_cache[cache_key])
                else:
                    texts_to_encode.append(text)
                    indices_to_encode.append(i)
                    embeddings.append(None)  # Placeholder
        else:
            texts_to_encode = texts
            indices_to_encode = list(range(len(texts)))
            embeddings = [None] * len(texts)
        
        # Codificar textos no cacheados
        if texts_to_encode:
            try:
                new_embeddings = self.model.encode(
                    texts_to_encode, 
                    batch_size=batch_size,
                    show_progress_bar=False
                )
                
                # Actualizar resultados y caché
                for i, embedding in enumerate(new_embeddings):
                    original_idx = indices_to_encode[i]
                    embeddings[original_idx] = embedding
                    
                    if use_cache:
                        cache_key = self._get_cache_key(texts_to_encode[i])
                        self._memory_cache[cache_key] = embedding
                
                # Guardar caché si hay nuevos embeddings
                if use_cache and texts_to_encode:
                    self._save_cache()
                    
            except Exception as e:
                logger.error(f"Error codificando embeddings: {e}")
                raise
        
        # Convertir a numpy array
        result = np.array(embeddings)
        
        logger.debug(f"Embeddings generados: {len(texts)} textos, "
                    f"{len(texts_to_encode)} nuevos")
        
        return result
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similitud coseno entre dos textos.
        
        Args:
            text1: Primer texto
            text2: Segundo texto
            
        Returns:
            Similitud coseno (0.0 - 1.0)
        """
        try:
            embeddings = self.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculando similitud: {e}")
            return 0.0
    
    def find_most_similar(self, query: str, candidates: List[str], 
                         top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Encuentra los candidatos más similares a la query.
        
        Args:
            query: Texto de consulta
            candidates: Lista de textos candidatos
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de tuplas (texto, similitud) ordenadas por similitud
        """
        try:
            if not candidates:
                return []
            
            # Codificar query y candidatos
            query_embedding = self.encode([query])[0]
            candidate_embeddings = self.encode(candidates)
            
            # Calcular similitudes
            similarities = np.dot(candidate_embeddings, query_embedding)
            
            # Obtener top_k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(candidates):
                    results.append((candidates[idx], float(similarities[idx])))
            
            return results
            
        except Exception as e:
            logger.error(f"Error en find_most_similar: {e}")
            return []
    
    def cluster_texts(self, texts: List[str], n_clusters: int = 3) -> Dict[int, List[str]]:
        """
        Agrupa textos en clusters basado en similitud semántica.
        
        Args:
            texts: Lista de textos a agrupar
            n_clusters: Número de clusters deseados
            
        Returns:
            Diccionario con clusters {cluster_id: [textos]}
        """
        try:
            if len(texts) < n_clusters:
                # Si hay menos textos que clusters, cada texto es un cluster
                return {i: [text] for i, text in enumerate(texts)}
            
            # Codificar textos
            embeddings = self.encode(texts)
            
            # K-means clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Agrupar resultados
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(texts[i])
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error en cluster_texts: {e}")
            # Fallback: agrupar secuencialmente
            return {i: [text] for i, text in enumerate(texts)}
    
    def get_embedding_dimension(self) -> int:
        """Retorna la dimensión de los embeddings."""
        try:
            # Codificar un texto de prueba para obtener dimensión
            test_embedding = self.encode(["test"])
            return test_embedding.shape[1]
        except Exception as e:
            logger.error(f"Error obteniendo dimensión: {e}")
            return 384  # Dimensión típica de all-MiniLM-L6-v2
    
    def clear_cache(self):
        """Limpia el caché de embeddings."""
        self._memory_cache.clear()
        if self._cache_file.exists():
            self._cache_file.unlink()
        logger.info("Caché de embeddings limpiado")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del caché."""
        return {
            'cache_size': len(self._memory_cache),
            'cache_file_size_mb': self._cache_file.stat().st_size / (1024 * 1024) 
                                 if self._cache_file.exists() else 0,
            'model_name': self.model_name,
            'embedding_dimension': self.get_embedding_dimension()
        }


class DomainEmbeddingAnalyzer:
    """
    Analizador especializado para embeddings de dominios.
    """
    
    def __init__(self, embedding_model: EmbeddingModel):
        """
        Inicializa el analizador con un modelo de embeddings.
        
        Args:
            embedding_model: Instancia de EmbeddingModel
        """
        self.embedding_model = embedding_model
        self.domain_embeddings = {}
        self.domain_keywords = {}
        
        # Dominios predefinidos
        self._initialize_domains()
    
    def _initialize_domains(self):
        """Inicializa dominios conocidos con sus keywords."""
        self.domain_keywords = {
            'programming': [
                'python', 'javascript', 'java', 'c++', 'sql', 'html', 'css',
                'django', 'flask', 'react', 'node', 'api', 'database', 'algorithm',
                'function', 'class', 'variable', 'loop', 'condition', 'debug'
            ],
            'data_science': [
                'machine learning', 'neural network', 'deep learning', 'ai',
                'data analysis', 'statistics', 'pandas', 'numpy', 'tensorflow',
                'pytorch', 'model', 'training', 'prediction', 'classification'
            ],
            'web_development': [
                'html', 'css', 'javascript', 'react', 'vue', 'angular', 'node',
                'express', 'api', 'rest', 'graphql', 'frontend', 'backend',
                'responsive', 'bootstrap', 'sass', 'webpack'
            ],
            'devops': [
                'docker', 'kubernetes', 'ci/cd', 'jenkins', 'git', 'deployment',
                'monitoring', 'logging', 'infrastructure', 'cloud', 'aws',
                'azure', 'gcp', 'terraform', 'ansible'
            ],
            'general': [
                'what', 'how', 'why', 'when', 'where', 'explain', 'describe',
                'help', 'question', 'answer', 'information', 'guide', 'tutorial'
            ]
        }
        
        # Generar embeddings para cada dominio
        for domain, keywords in self.domain_keywords.items():
            domain_text = f"{domain}: {' '.join(keywords)}"
            embedding = self.embedding_model.encode([domain_text])[0]
            self.domain_embeddings[domain] = embedding
    
    def analyze_domain_similarity(self, query: str) -> Dict[str, float]:
        """
        Analiza la similitud de una query con diferentes dominios.
        
        Args:
            query: Query a analizar
            
        Returns:
            Diccionario con similitudes por dominio
        """
        try:
            query_embedding = self.embedding_model.encode([query])[0]
            similarities = {}
            
            for domain, domain_embedding in self.domain_embeddings.items():
                similarity = np.dot(query_embedding, domain_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(domain_embedding)
                )
                similarities[domain] = float(similarity)
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error analizando similitud de dominio: {e}")
            return {}
    
    def get_primary_domain(self, query: str) -> Tuple[str, float]:
        """
        Obtiene el dominio principal de una query.
        
        Args:
            query: Query a analizar
            
        Returns:
            Tupla (dominio, confianza)
        """
        similarities = self.analyze_domain_similarity(query)
        
        if not similarities:
            return 'general', 0.0
        
        # Encontrar dominio con mayor similitud
        primary_domain = max(similarities, key=similarities.get)
        confidence = similarities[primary_domain]
        
        return primary_domain, confidence
    
    def is_domain_specific(self, query: str, threshold: float = 0.3) -> bool:
        """
        Determina si una query es específica de un dominio.
        
        Args:
            query: Query a analizar
            threshold: Umbral de confianza
            
        Returns:
            True si es específica de dominio
        """
        _, confidence = self.get_primary_domain(query)
        return confidence > threshold


# Funciones de conveniencia
def create_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    """Crea una instancia de EmbeddingModel."""
    return EmbeddingModel(model_name)


def create_domain_analyzer(embedding_model: EmbeddingModel = None) -> DomainEmbeddingAnalyzer:
    """Crea una instancia de DomainEmbeddingAnalyzer."""
    if embedding_model is None:
        embedding_model = create_embedding_model()
    return DomainEmbeddingAnalyzer(embedding_model)


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.INFO)
    
    # Crear modelo
    model = create_embedding_model()
    
    # Test similitud
    text1 = "¿Cómo implementar una API REST en Python?"
    text2 = "Crear un endpoint web con Flask"
    similarity = model.similarity(text1, text2)
    print(f"Similitud: {similarity:.3f}")
    
    # Test dominio
    analyzer = create_domain_analyzer(model)
    domain, confidence = analyzer.get_primary_domain(text1)
    print(f"Dominio: {domain}, Confianza: {confidence:.3f}")
    
    # Stats
    stats = model.get_cache_stats()
    print(f"Stats: {stats}")
