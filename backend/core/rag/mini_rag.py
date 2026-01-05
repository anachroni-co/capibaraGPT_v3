#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniRAG - Búsqueda rápida superficial con timeout estricto (<50ms).
"""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import signal
import threading
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Excepción para timeouts."""
    pass


@contextmanager
def timeout(seconds):
    """Context manager para timeout compatible con Windows."""
    if hasattr(signal, 'SIGALRM'):
        # Unix/Linux/Mac
        def signal_handler(signum, frame):
            raise TimeoutError(f"Operación excedió {seconds} segundos")
        
        old_handler = signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(int(seconds))
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows - simplificado, solo verificar tiempo
        start_time = time.time()
        yield
        elapsed = time.time() - start_time
        if elapsed > seconds:
            raise TimeoutError(f"Operación excedió {seconds} segundos")


class MiniRAG:
    """
    Búsqueda rápida superficial con timeout estricto.
    Optimizado para latencia <50ms.
    """
    
    def __init__(self, vector_store=None, embedding_model=None, 
                 timeout_ms: int = 50, max_results: int = 5):
        """
        Inicializa MiniRAG.
        
        Args:
            vector_store: Instancia de VectorStore
            embedding_model: Modelo de embeddings
            timeout_ms: Timeout en milisegundos
            max_results: Máximo número de resultados
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.timeout_ms = timeout_ms
        self.max_results = max_results
        
        # Cache para queries recientes
        self.query_cache = {}
        self.cache_size = 100
        self.cache_ttl = 300  # 5 minutos
        
        # Métricas
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'timeout_queries': 0,
            'cache_hits': 0,
            'avg_latency_ms': 0,
            'total_latency_ms': 0
        }
        
        logger.info(f"MiniRAG inicializado: timeout={timeout_ms}ms, max_results={max_results}")
    
    def search(self, query: str, k: int = None, 
               filter_metadata: Dict[str, Any] = None) -> List[Any]:
        """
        Búsqueda superficial con límite de tiempo.
        
        Args:
            query: Query de búsqueda
            k: Número de resultados (usa max_results si no se especifica)
            filter_metadata: Filtros de metadata
            
        Returns:
            Lista de documentos encontrados
        """
        start_time = time.time()
        
        try:
            if k is None:
                k = self.max_results
            
            # Verificar caché
            cache_key = self._get_cache_key(query, k, filter_metadata)
            if cache_key in self.query_cache:
                cache_entry = self.query_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    self.stats['cache_hits'] += 1
                    logger.debug(f"Cache hit para query: {query[:50]}...")
                    return cache_entry['results']
            
            # Búsqueda con timeout
            results = self._search_with_timeout(query, k, filter_metadata)
            
            # Actualizar caché
            self._update_cache(cache_key, results)
            
            # Actualizar estadísticas
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(latency_ms, success=True)
            
            logger.debug(f"MiniRAG search: {len(results)} resultados, {latency_ms:.1f}ms")
            return results
            
        except TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(latency_ms, success=False, timeout=True)
            logger.warning(f"MiniRAG timeout: {query[:50]}... ({latency_ms:.1f}ms)")
            return []
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(latency_ms, success=False)
            logger.error(f"Error en MiniRAG search: {e}")
            return []
    
    def _search_with_timeout(self, query: str, k: int, 
                           filter_metadata: Dict[str, Any] = None) -> List[Any]:
        """Búsqueda con timeout estricto."""
        try:
            with timeout(self.timeout_ms / 1000):
                # Generar embedding de la query
                if not self.embedding_model:
                    logger.warning("No hay modelo de embeddings disponible")
                    return []
                
                query_embedding = self.embedding_model.encode([query])[0]
                
                # Buscar en vector store
                if not self.vector_store:
                    logger.warning("No hay vector store disponible")
                    return []
                
                results = self.vector_store.similarity_search(
                    query_embedding, k, filter_metadata
                )
                
                return results
                
        except TimeoutError:
            raise
        except Exception as e:
            logger.error(f"Error en búsqueda con timeout: {e}")
            return []
    
    def _get_cache_key(self, query: str, k: int, 
                      filter_metadata: Dict[str, Any] = None) -> str:
        """Genera clave de caché."""
        import hashlib
        import json
        
        key_data = {
            'query': query,
            'k': k,
            'filter': filter_metadata or {}
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_cache(self, cache_key: str, results: List[Any]):
        """Actualiza caché de queries."""
        try:
            # Limpiar caché si está lleno
            if len(self.query_cache) >= self.cache_size:
                # Eliminar entrada más antigua
                oldest_key = min(
                    self.query_cache.keys(),
                    key=lambda k: self.query_cache[k]['timestamp']
                )
                del self.query_cache[oldest_key]
            
            # Agregar nueva entrada
            self.query_cache[cache_key] = {
                'results': results,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error actualizando caché: {e}")
    
    def _update_stats(self, latency_ms: float, success: bool, timeout: bool = False):
        """Actualiza estadísticas."""
        try:
            self.stats['total_queries'] += 1
            self.stats['total_latency_ms'] += latency_ms
            
            if success:
                self.stats['successful_queries'] += 1
            elif timeout:
                self.stats['timeout_queries'] += 1
            
            # Actualizar latencia promedio
            self.stats['avg_latency_ms'] = (
                self.stats['total_latency_ms'] / self.stats['total_queries']
            )
            
        except Exception as e:
            logger.error(f"Error actualizando estadísticas: {e}")
    
    def get_fast_results(self, query: str, max_results: int = 3) -> List[Any]:
        """
        Obtiene resultados rápidos con timeout más estricto.
        
        Args:
            query: Query de búsqueda
            max_results: Máximo número de resultados
            
        Returns:
            Lista de resultados rápidos
        """
        try:
            # Usar timeout más estricto para resultados rápidos
            original_timeout = self.timeout_ms
            self.timeout_ms = 25  # 25ms para resultados rápidos
            
            results = self.search(query, max_results)
            
            # Restaurar timeout original
            self.timeout_ms = original_timeout
            
            return results
            
        except Exception as e:
            logger.error(f"Error obteniendo resultados rápidos: {e}")
            return []
    
    def is_available(self) -> bool:
        """Verifica si MiniRAG está disponible."""
        return (self.vector_store is not None and 
                self.embedding_model is not None)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de performance."""
        try:
            if self.stats['total_queries'] == 0:
                return {
                    'total_queries': 0,
                    'success_rate': 0.0,
                    'timeout_rate': 0.0,
                    'cache_hit_rate': 0.0,
                    'avg_latency_ms': 0.0
                }
            
            return {
                'total_queries': self.stats['total_queries'],
                'success_rate': self.stats['successful_queries'] / self.stats['total_queries'],
                'timeout_rate': self.stats['timeout_queries'] / self.stats['total_queries'],
                'cache_hit_rate': self.stats['cache_hits'] / self.stats['total_queries'],
                'avg_latency_ms': round(self.stats['avg_latency_ms'], 2)
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas: {e}")
            return {}
    
    def clear_cache(self):
        """Limpia el caché de queries."""
        try:
            self.query_cache.clear()
            logger.info("Caché de MiniRAG limpiado")
        except Exception as e:
            logger.error(f"Error limpiando caché: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas completas."""
        return {
            'stats': self.stats.copy(),
            'performance_metrics': self.get_performance_metrics(),
            'cache_size': len(self.query_cache),
            'cache_ttl': self.cache_ttl,
            'timeout_ms': self.timeout_ms,
            'max_results': self.max_results,
            'available': self.is_available()
        }
    
    def reset_stats(self):
        """Resetea estadísticas."""
        try:
            self.stats = {
                'total_queries': 0,
                'successful_queries': 0,
                'timeout_queries': 0,
                'cache_hits': 0,
                'avg_latency_ms': 0,
                'total_latency_ms': 0
            }
            logger.info("Estadísticas de MiniRAG reseteadas")
        except Exception as e:
            logger.error(f"Error reseteando estadísticas: {e}")


# Función de conveniencia
def create_mini_rag(vector_store=None, embedding_model=None, 
                   timeout_ms: int = 50) -> MiniRAG:
    """Crea una instancia de MiniRAG."""
    return MiniRAG(vector_store, embedding_model, timeout_ms)


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.INFO)
    
    # Crear MiniRAG
    mini_rag = create_mini_rag()
    
    # Test búsqueda
    query = "python programming"
    results = mini_rag.search(query, k=3)
    
    print("=== Test MiniRAG ===")
    print(f"Query: {query}")
    print(f"Resultados: {len(results)}")
    
    # Test métricas
    metrics = mini_rag.get_performance_metrics()
    print(f"Métricas: {metrics}")
    
    # Test estadísticas
    stats = mini_rag.get_stats()
    print(f"Estadísticas: {stats}")
