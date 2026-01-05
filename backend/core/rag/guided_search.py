#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guided Search - Coordinación entre MiniRAG y FullRAG.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class GuidedSearch:
    """
    Coordinación entre MiniRAG y FullRAG para búsqueda optimizada.
    """
    
    def __init__(self, mini_rag=None, full_rag=None, 
                 expansion_threshold: float = 0.3):
        """
        Inicializa GuidedSearch.
        
        Args:
            mini_rag: Instancia de MiniRAG
            full_rag: Instancia de FullRAG
            expansion_threshold: Umbral para expandir a FullRAG
        """
        self.mini_rag = mini_rag
        self.full_rag = full_rag
        self.expansion_threshold = expansion_threshold
        
        # Criterios de expansión
        self.expansion_criteria = {
            'min_results': 2,  # Mínimo de resultados de MiniRAG
            'max_latency_ms': 100,  # Máxima latencia para MiniRAG
            'relevance_threshold': 0.5,  # Umbral de relevancia
            'query_complexity_threshold': 0.6  # Umbral de complejidad
        }
        
        # Métricas
        self.stats = {
            'total_queries': 0,
            'mini_rag_only': 0,
            'full_rag_triggered': 0,
            'avg_latency_ms': 0,
            'total_latency_ms': 0,
            'avg_results': 0,
            'total_results': 0
        }
        
        logger.info(f"GuidedSearch inicializado: expansion_threshold={expansion_threshold}")
    
    def search(self, query: str, use_full: bool = False, 
               filter_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Coordinación entre MiniRAG y FullRAG.
        
        Args:
            query: Query de búsqueda
            use_full: Forzar uso de FullRAG
            filter_metadata: Filtros de metadata
            
        Returns:
            Diccionario con resultados y metadata
        """
        start_time = time.time()
        
        try:
            # Paso 1: Búsqueda rápida con MiniRAG
            mini_results = []
            mini_latency = 0
            
            if self.mini_rag and not use_full:
                mini_start = time.time()
                mini_results = self.mini_rag.search(query, filter_metadata=filter_metadata)
                mini_latency = (time.time() - mini_start) * 1000
            
            # Paso 2: Decidir si expandir a FullRAG
            should_expand = self._should_expand(query, mini_results, mini_latency, use_full)
            
            # Paso 3: Búsqueda profunda si es necesario
            full_results = []
            full_latency = 0
            expansion_reason = None
            
            if should_expand and self.full_rag:
                full_start = time.time()
                full_results = self.full_rag.search(query, mini_results, filter_metadata)
                full_latency = (time.time() - full_start) * 1000
                expansion_reason = self._get_expansion_reason(query, mini_results, mini_latency)
            
            # Paso 4: Combinar y rankear resultados
            final_results = self._combine_results(mini_results, full_results, query)
            
            # Calcular latencia total
            total_latency = (time.time() - start_time) * 1000
            
            # Actualizar estadísticas
            self._update_stats(total_latency, len(final_results), should_expand)
            
            result = {
                'results': final_results,
                'total_results': len(final_results),
                'mini_rag_results': len(mini_results),
                'full_rag_results': len(full_results),
                'mini_rag_latency_ms': mini_latency,
                'full_rag_latency_ms': full_latency,
                'total_latency_ms': total_latency,
                'expansion_triggered': should_expand,
                'expansion_reason': expansion_reason,
                'search_strategy': self._get_search_strategy(mini_results, full_results),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"GuidedSearch: {len(final_results)} resultados, "
                        f"{total_latency:.1f}ms, expansión: {should_expand}")
            
            return result
            
        except Exception as e:
            total_latency = (time.time() - start_time) * 1000
            self._update_stats(total_latency, 0, False)
            logger.error(f"Error en GuidedSearch: {e}")
            return {
                'results': [],
                'total_results': 0,
                'mini_rag_results': 0,
                'full_rag_results': 0,
                'mini_rag_latency_ms': 0,
                'full_rag_latency_ms': 0,
                'total_latency_ms': total_latency,
                'expansion_triggered': False,
                'expansion_reason': 'error',
                'search_strategy': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _should_expand(self, query: str, mini_results: List[Any], 
                      mini_latency: float, force_full: bool) -> bool:
        """
        Decide si expandir a FullRAG.
        
        Args:
            query: Query original
            mini_results: Resultados de MiniRAG
            mini_latency: Latencia de MiniRAG
            force_full: Forzar expansión
            
        Returns:
            True si debe expandir
        """
        try:
            if force_full:
                return True
            
            if not self.full_rag:
                return False
            
            # Criterio 1: Pocos resultados de MiniRAG
            if len(mini_results) < self.expansion_criteria['min_results']:
                return True
            
            # Criterio 2: Latencia alta de MiniRAG (posible timeout)
            if mini_latency > self.expansion_criteria['max_latency_ms']:
                return True
            
            # Criterio 3: Baja relevancia de resultados
            if mini_results:
                avg_relevance = self._calculate_avg_relevance(mini_results, query)
                if avg_relevance < self.expansion_criteria['relevance_threshold']:
                    return True
            
            # Criterio 4: Query compleja
            query_complexity = self._calculate_query_complexity(query)
            if query_complexity > self.expansion_criteria['query_complexity_threshold']:
                return True
            
            # Criterio 5: Patrones específicos que requieren búsqueda profunda
            if self._requires_deep_search(query):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error decidiendo expansión: {e}")
            return False
    
    def _calculate_avg_relevance(self, results: List[Any], query: str) -> float:
        """Calcula relevancia promedio de resultados."""
        try:
            if not results:
                return 0.0
            
            total_relevance = 0.0
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            for result in results:
                # Extraer contenido
                if hasattr(result, 'content'):
                    content = result.content.lower()
                elif isinstance(result, dict):
                    content = result.get('content', '').lower()
                else:
                    content = str(result).lower()
                
                # Calcular relevancia simple (coincidencias de palabras)
                content_words = set(content.split())
                matches = len(query_words.intersection(content_words))
                relevance = matches / len(query_words) if query_words else 0.0
                
                total_relevance += relevance
            
            return total_relevance / len(results)
            
        except Exception as e:
            logger.error(f"Error calculando relevancia promedio: {e}")
            return 0.0
    
    def _calculate_query_complexity(self, query: str) -> float:
        """Calcula complejidad de la query."""
        try:
            complexity = 0.0
            
            # Factor 1: Longitud
            length_score = min(len(query) / 100, 1.0)  # Normalizar por 100 chars
            complexity += length_score * 0.2
            
            # Factor 2: Número de palabras
            word_count = len(query.split())
            word_score = min(word_count / 10, 1.0)  # Normalizar por 10 palabras
            complexity += word_score * 0.2
            
            # Factor 3: Términos técnicos
            technical_terms = [
                'algorithm', 'architecture', 'optimization', 'implementation',
                'algoritmo', 'arquitectura', 'optimización', 'implementación',
                'framework', 'library', 'api', 'database', 'security'
            ]
            technical_count = sum(1 for term in technical_terms if term in query.lower())
            technical_score = min(technical_count / 3, 1.0)  # Normalizar por 3 términos
            complexity += technical_score * 0.3
            
            # Factor 4: Múltiples conceptos (palabras únicas)
            unique_words = len(set(query.lower().split()))
            concept_score = min(unique_words / 8, 1.0)  # Normalizar por 8 conceptos
            complexity += concept_score * 0.3
            
            return min(complexity, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculando complejidad: {e}")
            return 0.5
    
    def _requires_deep_search(self, query: str) -> bool:
        """Verifica si la query requiere búsqueda profunda."""
        try:
            query_lower = query.lower()
            
            # Patrones que requieren búsqueda profunda
            deep_search_patterns = [
                'compare', 'comparar', 'vs', 'versus',
                'difference', 'diferencia', 'advantages', 'ventajas',
                'best practice', 'mejores prácticas', 'recommendation', 'recomendación',
                'tutorial', 'guide', 'guía', 'how to', 'cómo',
                'example', 'ejemplo', 'sample', 'muestra',
                'comprehensive', 'completo', 'detailed', 'detallado',
                'research', 'investigar', 'analyze', 'analizar'
            ]
            
            return any(pattern in query_lower for pattern in deep_search_patterns)
            
        except Exception as e:
            logger.error(f"Error verificando búsqueda profunda: {e}")
            return False
    
    def _get_expansion_reason(self, query: str, mini_results: List[Any], 
                            mini_latency: float) -> str:
        """Obtiene razón de expansión."""
        try:
            if len(mini_results) < self.expansion_criteria['min_results']:
                return 'insufficient_results'
            
            if mini_latency > self.expansion_criteria['max_latency_ms']:
                return 'high_latency'
            
            if mini_results:
                avg_relevance = self._calculate_avg_relevance(mini_results, query)
                if avg_relevance < self.expansion_criteria['relevance_threshold']:
                    return 'low_relevance'
            
            query_complexity = self._calculate_query_complexity(query)
            if query_complexity > self.expansion_criteria['query_complexity_threshold']:
                return 'complex_query'
            
            if self._requires_deep_search(query):
                return 'deep_search_required'
            
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Error obteniendo razón de expansión: {e}")
            return 'error'
    
    def _combine_results(self, mini_results: List[Any], full_results: List[Any], 
                        query: str) -> List[Any]:
        """Combina y rankea resultados de MiniRAG y FullRAG."""
        try:
            # Si no hay resultados de FullRAG, usar solo MiniRAG
            if not full_results:
                return mini_results
            
            # Si no hay resultados de MiniRAG, usar solo FullRAG
            if not mini_results:
                return full_results
            
            # Combinar resultados evitando duplicados
            combined_results = []
            seen_doc_ids = set()
            
            # Agregar resultados de MiniRAG primero (prioridad)
            for result in mini_results:
                doc_id = getattr(result, 'doc_id', id(result))
                if doc_id not in seen_doc_ids:
                    combined_results.append(result)
                    seen_doc_ids.add(doc_id)
            
            # Agregar resultados de FullRAG
            for result in full_results:
                doc_id = getattr(result, 'doc_id', id(result))
                if doc_id not in seen_doc_ids:
                    combined_results.append(result)
                    seen_doc_ids.add(doc_id)
            
            # Rankear por relevancia
            ranked_results = self._rank_results(combined_results, query)
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Error combinando resultados: {e}")
            return mini_results + full_results
    
    def _rank_results(self, results: List[Any], query: str) -> List[Any]:
        """Rankea resultados por relevancia."""
        try:
            if not results:
                return []
            
            # Calcular scores de relevancia
            scored_results = []
            for result in results:
                score = self._calculate_result_score(result, query)
                scored_results.append((score, result))
            
            # Ordenar por score
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # Retornar solo los resultados (sin scores)
            return [result for score, result in scored_results]
            
        except Exception as e:
            logger.error(f"Error rankeando resultados: {e}")
            return results
    
    def _calculate_result_score(self, result: Any, query: str) -> float:
        """Calcula score de relevancia para un resultado."""
        try:
            score = 0.0
            
            # Extraer contenido
            if hasattr(result, 'content'):
                content = result.content
            elif isinstance(result, dict):
                content = result.get('content', '')
            else:
                content = str(result)
            
            content_lower = content.lower()
            query_lower = query.lower()
            query_words = query_lower.split()
            
            # Score por coincidencias exactas
            exact_matches = sum(1 for word in query_words if word in content_lower)
            score += exact_matches * 2.0
            
            # Score por coincidencias parciales
            partial_matches = sum(1 for word in query_words 
                                if any(word in content_word for content_word in content_lower.split()))
            score += partial_matches * 1.0
            
            # Score por metadata
            if hasattr(result, 'metadata') and result.metadata:
                metadata = result.metadata
                for word in query_words:
                    for key, value in metadata.items():
                        if isinstance(value, str) and word in value.lower():
                            score += 0.5
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, str) and word in item.lower():
                                    score += 0.3
            
            # Bonus por longitud del contenido (contenido más detallado)
            content_length = len(content.split())
            if content_length > 50:  # Contenido sustancial
                score += 0.5
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculando score: {e}")
            return 0.0
    
    def _get_search_strategy(self, mini_results: List[Any], full_results: List[Any]) -> str:
        """Obtiene estrategia de búsqueda utilizada."""
        try:
            if not mini_results and not full_results:
                return 'no_results'
            elif mini_results and not full_results:
                return 'mini_rag_only'
            elif not mini_results and full_results:
                return 'full_rag_only'
            else:
                return 'hybrid'
                
        except Exception as e:
            logger.error(f"Error obteniendo estrategia: {e}")
            return 'unknown'
    
    def _update_stats(self, latency_ms: float, results_count: int, expanded: bool):
        """Actualiza estadísticas."""
        try:
            self.stats['total_queries'] += 1
            self.stats['total_latency_ms'] += latency_ms
            self.stats['total_results'] += results_count
            
            if expanded:
                self.stats['full_rag_triggered'] += 1
            else:
                self.stats['mini_rag_only'] += 1
            
            # Actualizar promedios
            self.stats['avg_latency_ms'] = (
                self.stats['total_latency_ms'] / self.stats['total_queries']
            )
            self.stats['avg_results'] = (
                self.stats['total_results'] / self.stats['total_queries']
            )
            
        except Exception as e:
            logger.error(f"Error actualizando estadísticas: {e}")
    
    def get_optimized_results(self, query: str, 
                            filter_metadata: Dict[str, Any] = None) -> List[Any]:
        """
        Obtiene resultados optimizados como lista simple.
        
        Args:
            query: Query de búsqueda
            filter_metadata: Filtros de metadata
            
        Returns:
            Lista de resultados
        """
        try:
            result = self.search(query, filter_metadata=filter_metadata)
            return result.get('results', [])
        except Exception as e:
            logger.error(f"Error obteniendo resultados optimizados: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de performance."""
        try:
            if self.stats['total_queries'] == 0:
                return {
                    'total_queries': 0,
                    'mini_rag_only_rate': 0.0,
                    'full_rag_triggered_rate': 0.0,
                    'avg_latency_ms': 0.0,
                    'avg_results': 0.0
                }
            
            return {
                'total_queries': self.stats['total_queries'],
                'mini_rag_only_rate': self.stats['mini_rag_only'] / self.stats['total_queries'],
                'full_rag_triggered_rate': self.stats['full_rag_triggered'] / self.stats['total_queries'],
                'avg_latency_ms': round(self.stats['avg_latency_ms'], 2),
                'avg_results': round(self.stats['avg_results'], 2)
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas completas."""
        return {
            'stats': self.stats.copy(),
            'performance_metrics': self.get_performance_metrics(),
            'expansion_threshold': self.expansion_threshold,
            'expansion_criteria': self.expansion_criteria.copy(),
            'components_available': {
                'mini_rag': self.mini_rag is not None,
                'full_rag': self.full_rag is not None
            }
        }
    
    def reset_stats(self):
        """Resetea estadísticas."""
        try:
            self.stats = {
                'total_queries': 0,
                'mini_rag_only': 0,
                'full_rag_triggered': 0,
                'avg_latency_ms': 0,
                'total_latency_ms': 0,
                'avg_results': 0,
                'total_results': 0
            }
            logger.info("Estadísticas de GuidedSearch reseteadas")
        except Exception as e:
            logger.error(f"Error reseteando estadísticas: {e}")


# Función de conveniencia
def create_guided_search(mini_rag=None, full_rag=None, 
                        expansion_threshold: float = 0.3) -> GuidedSearch:
    """Crea una instancia de GuidedSearch."""
    return GuidedSearch(mini_rag, full_rag, expansion_threshold)


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.INFO)
    
    # Crear GuidedSearch
    guided_search = create_guided_search()
    
    # Test búsqueda
    query = "python web development"
    result = guided_search.search(query)
    
    print("=== Test GuidedSearch ===")
    print(f"Query: {query}")
    print(f"Resultados: {result['total_results']}")
    print(f"Estrategia: {result['search_strategy']}")
    print(f"Expansión: {result['expansion_triggered']}")
    print(f"Latencia: {result['total_latency_ms']:.1f}ms")
    
    # Test métricas
    metrics = guided_search.get_performance_metrics()
    print(f"Métricas: {metrics}")
    
    # Test estadísticas
    stats = guided_search.get_stats()
    print(f"Estadísticas: {stats}")
