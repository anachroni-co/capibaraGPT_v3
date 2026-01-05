#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FullRAG - Búsqueda profunda guiada por resultados de MiniRAG.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class FullRAG:
    """
    Búsqueda profunda guiada por resultados de MiniRAG.
    Optimizado para completitud y relevancia.
    """
    
    def __init__(self, vector_store=None, embedding_model=None, 
                 mini_rag=None, max_results: int = 10, 
                 expansion_factor: float = 2.0):
        """
        Inicializa FullRAG.
        
        Args:
            vector_store: Instancia de VectorStore
            embedding_model: Modelo de embeddings
            mini_rag: Instancia de MiniRAG
            max_results: Máximo número de resultados
            expansion_factor: Factor de expansión para queries
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.mini_rag = mini_rag
        self.max_results = max_results
        self.expansion_factor = expansion_factor
        
        # Estrategias de expansión
        self.expansion_strategies = {
            'synonym_expansion': True,
            'concept_expansion': True,
            'context_expansion': True,
            'semantic_expansion': True
        }
        
        # Métricas
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'expanded_queries': 0,
            'avg_latency_ms': 0,
            'total_latency_ms': 0,
            'avg_results_per_query': 0,
            'total_results': 0
        }
        
        logger.info(f"FullRAG inicializado: max_results={max_results}, "
                   f"expansion_factor={expansion_factor}")
    
    def search(self, query: str, mini_results: List[Any] = None, 
               filter_metadata: Dict[str, Any] = None) -> List[Any]:
        """
        Búsqueda profunda expandiendo resultados iniciales.
        
        Args:
            query: Query original
            mini_results: Resultados de MiniRAG (opcional)
            filter_metadata: Filtros de metadata
            
        Returns:
            Lista de documentos encontrados
        """
        start_time = time.time()
        
        try:
            # Si no hay resultados de MiniRAG, usar MiniRAG primero
            if mini_results is None and self.mini_rag:
                mini_results = self.mini_rag.search(query, k=3)
            
            # Expandir query basado en resultados de MiniRAG
            expanded_queries = self._expand_query(query, mini_results)
            
            # Realizar búsquedas expandidas
            all_results = []
            seen_doc_ids = set()
            
            # Búsqueda original
            original_results = self._deep_search(query, filter_metadata)
            all_results.extend(original_results)
            seen_doc_ids.update(doc.doc_id for doc in original_results)
            
            # Búsquedas expandidas
            for expanded_query in expanded_queries:
                expanded_results = self._deep_search(expanded_query, filter_metadata)
                
                # Agregar solo resultados nuevos
                for doc in expanded_results:
                    if doc.doc_id not in seen_doc_ids:
                        all_results.append(doc)
                        seen_doc_ids.add(doc.doc_id)
            
            # Ordenar por relevancia y limitar resultados
            final_results = self._rank_and_limit_results(all_results, query)
            
            # Actualizar estadísticas
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(latency_ms, len(final_results), len(expanded_queries) > 0)
            
            logger.debug(f"FullRAG search: {len(final_results)} resultados, "
                        f"{latency_ms:.1f}ms, {len(expanded_queries)} queries expandidas")
            
            return final_results
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(latency_ms, 0, False)
            logger.error(f"Error en FullRAG search: {e}")
            return []
    
    def _expand_query(self, original_query: str, mini_results: List[Any]) -> List[str]:
        """
        Expande query basado en resultados de MiniRAG.
        
        Args:
            original_query: Query original
            mini_results: Resultados de MiniRAG
            
        Returns:
            Lista de queries expandidas
        """
        try:
            expanded_queries = []
            
            # Extraer términos relevantes de resultados de MiniRAG
            relevant_terms = self._extract_relevant_terms(mini_results)
            
            # Estrategia 1: Expansión por sinónimos
            if self.expansion_strategies['synonym_expansion']:
                synonym_queries = self._create_synonym_queries(original_query, relevant_terms)
                expanded_queries.extend(synonym_queries)
            
            # Estrategia 2: Expansión por conceptos
            if self.expansion_strategies['concept_expansion']:
                concept_queries = self._create_concept_queries(original_query, relevant_terms)
                expanded_queries.extend(concept_queries)
            
            # Estrategia 3: Expansión por contexto
            if self.expansion_strategies['context_expansion']:
                context_queries = self._create_context_queries(original_query, mini_results)
                expanded_queries.extend(context_queries)
            
            # Estrategia 4: Expansión semántica
            if self.expansion_strategies['semantic_expansion']:
                semantic_queries = self._create_semantic_queries(original_query, relevant_terms)
                expanded_queries.extend(semantic_queries)
            
            # Limitar número de queries expandidas
            max_expanded = int(len(original_query.split()) * self.expansion_factor)
            expanded_queries = expanded_queries[:max_expanded]
            
            logger.debug(f"Query expandida: {len(expanded_queries)} variaciones")
            return expanded_queries
            
        except Exception as e:
            logger.error(f"Error expandiendo query: {e}")
            return []
    
    def _extract_relevant_terms(self, mini_results: List[Any]) -> List[str]:
        """Extrae términos relevantes de resultados de MiniRAG."""
        try:
            terms = []
            
            for result in mini_results:
                if hasattr(result, 'content'):
                    content = result.content
                elif isinstance(result, dict):
                    content = result.get('content', '')
                else:
                    content = str(result)
                
                # Extraer palabras clave (palabras de 4+ caracteres)
                words = re.findall(r'\b\w{4,}\b', content.lower())
                terms.extend(words)
            
            # Contar frecuencia y retornar más comunes
            term_counts = {}
            for term in terms:
                term_counts[term] = term_counts.get(term, 0) + 1
            
            # Ordenar por frecuencia y retornar top 10
            sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
            return [term for term, count in sorted_terms[:10]]
            
        except Exception as e:
            logger.error(f"Error extrayendo términos relevantes: {e}")
            return []
    
    def _create_synonym_queries(self, original_query: str, relevant_terms: List[str]) -> List[str]:
        """Crea queries con sinónimos."""
        try:
            # Diccionario básico de sinónimos
            synonyms = {
                'programming': ['coding', 'development', 'software'],
                'python': ['python3', 'py'],
                'javascript': ['js', 'node'],
                'database': ['db', 'data storage'],
                'api': ['interface', 'endpoint'],
                'web': ['internet', 'online'],
                'mobile': ['phone', 'app'],
                'cloud': ['server', 'hosting'],
                'security': ['protection', 'safety'],
                'performance': ['speed', 'optimization']
            }
            
            expanded_queries = []
            
            for term in relevant_terms[:5]:  # Solo top 5 términos
                if term in synonyms:
                    for synonym in synonyms[term]:
                        # Reemplazar término en query original
                        expanded_query = original_query.replace(term, synonym)
                        if expanded_query != original_query:
                            expanded_queries.append(expanded_query)
            
            return expanded_queries
            
        except Exception as e:
            logger.error(f"Error creando queries de sinónimos: {e}")
            return []
    
    def _create_concept_queries(self, original_query: str, relevant_terms: List[str]) -> List[str]:
        """Crea queries con conceptos relacionados."""
        try:
            # Conceptos relacionados
            related_concepts = {
                'python': ['django', 'flask', 'numpy', 'pandas'],
                'javascript': ['react', 'vue', 'angular', 'node'],
                'database': ['sql', 'postgresql', 'mysql', 'mongodb'],
                'api': ['rest', 'graphql', 'json', 'http'],
                'web': ['html', 'css', 'frontend', 'backend'],
                'mobile': ['ios', 'android', 'react native'],
                'cloud': ['aws', 'azure', 'gcp', 'docker'],
                'security': ['authentication', 'authorization', 'encryption'],
                'performance': ['caching', 'optimization', 'scalability']
            }
            
            expanded_queries = []
            
            for term in relevant_terms[:3]:  # Solo top 3 términos
                if term in related_concepts:
                    for concept in related_concepts[term][:2]:  # Solo 2 conceptos por término
                        expanded_query = f"{original_query} {concept}"
                        expanded_queries.append(expanded_query)
            
            return expanded_queries
            
        except Exception as e:
            logger.error(f"Error creando queries de conceptos: {e}")
            return []
    
    def _create_context_queries(self, original_query: str, mini_results: List[Any]) -> List[str]:
        """Crea queries basadas en contexto de resultados."""
        try:
            expanded_queries = []
            
            # Extraer contexto de metadata
            for result in mini_results[:2]:  # Solo 2 resultados
                if hasattr(result, 'metadata') and result.metadata:
                    metadata = result.metadata
                    
                    # Crear query con categoría
                    if 'category' in metadata:
                        category_query = f"{original_query} {metadata['category']}"
                        expanded_queries.append(category_query)
                    
                    # Crear query con tags
                    if 'tags' in metadata and isinstance(metadata['tags'], list):
                        for tag in metadata['tags'][:2]:  # Solo 2 tags
                            tag_query = f"{original_query} {tag}"
                            expanded_queries.append(tag_query)
            
            return expanded_queries
            
        except Exception as e:
            logger.error(f"Error creando queries de contexto: {e}")
            return []
    
    def _create_semantic_queries(self, original_query: str, relevant_terms: List[str]) -> List[str]:
        """Crea queries con expansión semántica."""
        try:
            expanded_queries = []
            
            # Patrones de expansión semántica
            semantic_patterns = [
                f"{original_query} tutorial",
                f"{original_query} examples",
                f"{original_query} best practices",
                f"how to {original_query}",
                f"{original_query} implementation"
            ]
            
            # Agregar términos relevantes a patrones
            for term in relevant_terms[:2]:  # Solo 2 términos
                semantic_patterns.append(f"{original_query} {term}")
            
            expanded_queries.extend(semantic_patterns)
            
            return expanded_queries
            
        except Exception as e:
            logger.error(f"Error creando queries semánticas: {e}")
            return []
    
    def _deep_search(self, query: str, filter_metadata: Dict[str, Any] = None) -> List[Any]:
        """Realiza búsqueda profunda en vector store."""
        try:
            if not self.embedding_model or not self.vector_store:
                return []
            
            # Generar embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Buscar con más resultados para mayor cobertura
            search_k = min(self.max_results * 2, 20)  # Buscar más para mejor ranking
            results = self.vector_store.similarity_search(
                query_embedding, search_k, filter_metadata
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda profunda: {e}")
            return []
    
    def _rank_and_limit_results(self, all_results: List[Any], original_query: str) -> List[Any]:
        """Rankea y limita resultados finales."""
        try:
            if not all_results:
                return []
            
            # Calcular scores de relevancia
            scored_results = []
            for doc in all_results:
                score = self._calculate_relevance_score(doc, original_query)
                scored_results.append((score, doc))
            
            # Ordenar por score
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # Limitar resultados
            final_results = [doc for score, doc in scored_results[:self.max_results]]
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error rankeando resultados: {e}")
            return all_results[:self.max_results]
    
    def _calculate_relevance_score(self, doc: Any, query: str) -> float:
        """Calcula score de relevancia para un documento."""
        try:
            score = 0.0
            
            # Extraer contenido
            if hasattr(doc, 'content'):
                content = doc.content
            elif isinstance(doc, dict):
                content = doc.get('content', '')
            else:
                content = str(doc)
            
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
            if hasattr(doc, 'metadata') and doc.metadata:
                metadata = doc.metadata
                for word in query_words:
                    for key, value in metadata.items():
                        if isinstance(value, str) and word in value.lower():
                            score += 0.5
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, str) and word in item.lower():
                                    score += 0.3
            
            # Normalizar por longitud del contenido
            content_length = len(content.split())
            if content_length > 0:
                score = score / (content_length / 100)  # Normalizar por 100 palabras
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculando score de relevancia: {e}")
            return 0.0
    
    def _update_stats(self, latency_ms: float, results_count: int, expanded: bool):
        """Actualiza estadísticas."""
        try:
            self.stats['total_queries'] += 1
            self.stats['total_latency_ms'] += latency_ms
            self.stats['total_results'] += results_count
            
            if results_count > 0:
                self.stats['successful_queries'] += 1
            
            if expanded:
                self.stats['expanded_queries'] += 1
            
            # Actualizar promedios
            self.stats['avg_latency_ms'] = (
                self.stats['total_latency_ms'] / self.stats['total_queries']
            )
            self.stats['avg_results_per_query'] = (
                self.stats['total_results'] / self.stats['total_queries']
            )
            
        except Exception as e:
            logger.error(f"Error actualizando estadísticas: {e}")
    
    def get_comprehensive_results(self, query: str, 
                                filter_metadata: Dict[str, Any] = None) -> List[Any]:
        """
        Obtiene resultados comprensivos sin usar MiniRAG.
        
        Args:
            query: Query de búsqueda
            filter_metadata: Filtros de metadata
            
        Returns:
            Lista de resultados comprensivos
        """
        try:
            # Realizar múltiples búsquedas con diferentes estrategias
            all_results = []
            seen_doc_ids = set()
            
            # Búsqueda original
            original_results = self._deep_search(query, filter_metadata)
            all_results.extend(original_results)
            seen_doc_ids.update(doc.doc_id for doc in original_results)
            
            # Búsquedas con variaciones de la query
            query_variations = [
                f"{query} tutorial",
                f"{query} examples",
                f"how to {query}",
                f"{query} best practices"
            ]
            
            for variation in query_variations:
                variation_results = self._deep_search(variation, filter_metadata)
                for doc in variation_results:
                    if doc.doc_id not in seen_doc_ids:
                        all_results.append(doc)
                        seen_doc_ids.add(doc.doc_id)
            
            # Rankear y limitar
            final_results = self._rank_and_limit_results(all_results, query)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error obteniendo resultados comprensivos: {e}")
            return []
    
    def is_available(self) -> bool:
        """Verifica si FullRAG está disponible."""
        return (self.vector_store is not None and 
                self.embedding_model is not None)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de performance."""
        try:
            if self.stats['total_queries'] == 0:
                return {
                    'total_queries': 0,
                    'success_rate': 0.0,
                    'expansion_rate': 0.0,
                    'avg_latency_ms': 0.0,
                    'avg_results_per_query': 0.0
                }
            
            return {
                'total_queries': self.stats['total_queries'],
                'success_rate': self.stats['successful_queries'] / self.stats['total_queries'],
                'expansion_rate': self.stats['expanded_queries'] / self.stats['total_queries'],
                'avg_latency_ms': round(self.stats['avg_latency_ms'], 2),
                'avg_results_per_query': round(self.stats['avg_results_per_query'], 2)
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas completas."""
        return {
            'stats': self.stats.copy(),
            'performance_metrics': self.get_performance_metrics(),
            'max_results': self.max_results,
            'expansion_factor': self.expansion_factor,
            'expansion_strategies': self.expansion_strategies.copy(),
            'available': self.is_available()
        }
    
    def reset_stats(self):
        """Resetea estadísticas."""
        try:
            self.stats = {
                'total_queries': 0,
                'successful_queries': 0,
                'expanded_queries': 0,
                'avg_latency_ms': 0,
                'total_latency_ms': 0,
                'avg_results_per_query': 0,
                'total_results': 0
            }
            logger.info("Estadísticas de FullRAG reseteadas")
        except Exception as e:
            logger.error(f"Error reseteando estadísticas: {e}")


# Función de conveniencia
def create_full_rag(vector_store=None, embedding_model=None, mini_rag=None,
                   max_results: int = 10) -> FullRAG:
    """Crea una instancia de FullRAG."""
    return FullRAG(vector_store, embedding_model, mini_rag, max_results)


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.INFO)
    
    # Crear FullRAG
    full_rag = create_full_rag()
    
    # Test búsqueda
    query = "python web development"
    results = full_rag.search(query)
    
    print("=== Test FullRAG ===")
    print(f"Query: {query}")
    print(f"Resultados: {len(results)}")
    
    # Test métricas
    metrics = full_rag.get_performance_metrics()
    print(f"Métricas: {metrics}")
    
    # Test estadísticas
    stats = full_rag.get_stats()
    print(f"Estadísticas: {stats}")
