#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FullCAG - Context-Aware Generation para modelo 120B (32K tokens max).
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class FullCAG:
    """
    Context-Aware Generation para modelo 120B con límite de 32K tokens.
    Optimizado para profundidad y completitud.
    """
    
    def __init__(self, static_cache=None, dynamic_context=None, 
                 awareness_gate=None, mini_rag=None, full_rag=None,
                 max_tokens: int = 32000):
        """
        Inicializa FullCAG.
        
        Args:
            static_cache: Instancia de StaticCache
            dynamic_context: Instancia de DynamicContext
            awareness_gate: Instancia de AwarenessGate
            mini_rag: Instancia de MiniRAG
            full_rag: Instancia de FullRAG
            max_tokens: Máximo tokens (32K para modelo 120B)
        """
        self.static_cache = static_cache
        self.dynamic_context = dynamic_context
        self.awareness_gate = awareness_gate
        self.mini_rag = mini_rag
        self.full_rag = full_rag
        self.max_tokens = max_tokens
        
        # Límites específicos para FullCAG (más generosos)
        self.token_limits = {
            'static_cache': int(max_tokens * 0.25),  # 25% para conocimiento estático
            'dynamic_context': int(max_tokens * 0.20),  # 20% para contexto dinámico
            'mini_rag': int(max_tokens * 0.20),  # 20% para MiniRAG
            'full_rag': int(max_tokens * 0.35)  # 35% para FullRAG (búsqueda profunda)
        }
        
        # Métricas
        self.stats = {
            'total_queries': 0,
            'avg_latency_ms': 0,
            'token_usage': {
                'static_cache': 0,
                'dynamic_context': 0,
                'mini_rag': 0,
                'full_rag': 0
            },
            'source_usage': {
                'static_cache': 0,
                'dynamic_context': 0,
                'mini_rag': 0,
                'full_rag': 0
            },
            'rag_escalations': 0
        }
        
        logger.info(f"FullCAG inicializado con límite de {max_tokens} tokens")
    
    def generate_context(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Genera contexto completo para modelo 120B.
        
        Args:
            query: Query del usuario
            context: Contexto adicional
            
        Returns:
            Diccionario con contexto generado y metadata
        """
        start_time = time.time()
        
        try:
            if context is None:
                context = {}
            
            # Decidir fuentes usando AwarenessGate
            if self.awareness_gate:
                source_decision = self.awareness_gate.decide_sources(
                    query, self.max_tokens, context
                )
            else:
                # Fallback: usar todas las fuentes disponibles
                source_decision = {
                    'sources': {
                        'static_cache': True,
                        'dynamic_context': True,
                        'rag': True
                    },
                    'token_budget': {
                        'static_cache': self.token_limits['static_cache'],
                        'dynamic_context': self.token_limits['dynamic_context'],
                        'rag': self.token_limits['mini_rag'] + self.token_limits['full_rag']
                    },
                    'confidence': 0.7
                }
            
            # Generar contexto de cada fuente
            context_parts = []
            actual_tokens = 0
            source_metadata = {}
            
            # Static Cache
            if source_decision['sources'].get('static_cache', False) and self.static_cache:
                static_tokens = source_decision['token_budget'].get('static_cache',
                                                                   self.token_limits['static_cache'])
                static_context = self.static_cache.retrieve(query, static_tokens)
                
                if static_context:
                    context_parts.append(f"**Conocimiento Base:**\n{static_context}")
                    actual_tokens += len(static_context.split())
                    source_metadata['static_cache'] = {
                        'tokens_used': len(static_context.split()),
                        'content_length': len(static_context)
                    }
                    self.stats['source_usage']['static_cache'] += 1
            
            # Dynamic Context
            if source_decision['sources'].get('dynamic_context', False) and self.dynamic_context:
                dynamic_tokens = source_decision['token_budget'].get('dynamic_context',
                                                                    self.token_limits['dynamic_context'])
                dynamic_context = self.dynamic_context.get_context(query, dynamic_tokens)
                
                if dynamic_context:
                    context_parts.append(dynamic_context)
                    actual_tokens += len(dynamic_context.split())
                    source_metadata['dynamic_context'] = {
                        'tokens_used': len(dynamic_context.split()),
                        'content_length': len(dynamic_context)
                    }
                    self.stats['source_usage']['dynamic_context'] += 1
            
            # RAG (MiniRAG + FullRAG)
            if source_decision['sources'].get('rag', False):
                rag_context, rag_metadata = self._generate_rag_context(query, source_decision)
                if rag_context:
                    context_parts.append(rag_context)
                    actual_tokens += len(rag_context.split())
                    source_metadata.update(rag_metadata)
            
            # Combinar contexto
            final_context = "\n\n".join(context_parts)
            
            # Verificar límite de tokens
            if actual_tokens > self.max_tokens:
                # Truncar inteligentemente (mantener contexto más relevante)
                final_context = self._intelligent_truncate(final_context, self.max_tokens)
                actual_tokens = len(final_context.split())
            
            # Calcular latencia
            latency_ms = (time.time() - start_time) * 1000
            
            # Actualizar estadísticas
            self._update_stats(actual_tokens, latency_ms, source_metadata)
            
            result = {
                'context': final_context,
                'tokens_used': actual_tokens,
                'tokens_available': self.max_tokens - actual_tokens,
                'sources_used': list(source_metadata.keys()),
                'source_metadata': source_metadata,
                'latency_ms': latency_ms,
                'confidence': source_decision.get('confidence', 0.7),
                'rag_escalated': 'full_rag' in source_metadata,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"FullCAG generó contexto: {actual_tokens} tokens, "
                        f"{latency_ms:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generando contexto en FullCAG: {e}")
            return {
                'context': "",
                'tokens_used': 0,
                'tokens_available': self.max_tokens,
                'sources_used': [],
                'source_metadata': {},
                'latency_ms': (time.time() - start_time) * 1000,
                'confidence': 0.0,
                'rag_escalated': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_rag_context(self, query: str, source_decision: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Genera contexto usando RAG (MiniRAG + FullRAG)."""
        try:
            rag_tokens = source_decision['token_budget'].get('rag', 
                                                           self.token_limits['mini_rag'] + self.token_limits['full_rag'])
            
            context_parts = []
            metadata = {}
            
            # MiniRAG (búsqueda rápida)
            if self.mini_rag:
                mini_tokens = min(self.token_limits['mini_rag'], rag_tokens // 2)
                mini_results = self.mini_rag.search(query, k=5)
                
                if mini_results:
                    mini_context = self._format_rag_results(mini_results, mini_tokens)
                    if mini_context:
                        context_parts.append(f"**Información Rápida:**\n{mini_context}")
                        metadata['mini_rag'] = {
                            'tokens_used': len(mini_context.split()),
                            'results_count': len(mini_results)
                        }
                        self.stats['source_usage']['mini_rag'] += 1
            
            # FullRAG (búsqueda profunda) - solo si MiniRAG no fue suficiente
            if self.full_rag and len(context_parts) == 0:
                full_tokens = min(self.token_limits['full_rag'], rag_tokens)
                full_results = self.full_rag.search(query, mini_results if 'mini_results' in locals() else [])
                
                if full_results:
                    full_context = self._format_rag_results(full_results, full_tokens)
                    if full_context:
                        context_parts.append(f"**Información Detallada:**\n{full_context}")
                        metadata['full_rag'] = {
                            'tokens_used': len(full_context.split()),
                            'results_count': len(full_results)
                        }
                        self.stats['source_usage']['full_rag'] += 1
                        self.stats['rag_escalations'] += 1
            
            final_context = "\n\n".join(context_parts)
            return final_context, metadata
            
        except Exception as e:
            logger.error(f"Error generando contexto RAG: {e}")
            return "", {}
    
    def _format_rag_results(self, results: List[Any], max_tokens: int) -> str:
        """Formatea resultados de RAG respetando límite de tokens."""
        try:
            if not results:
                return ""
            
            formatted_parts = []
            current_tokens = 0
            
            for i, result in enumerate(results):
                # Extraer contenido del resultado
                if hasattr(result, 'page_content'):
                    content = result.page_content
                elif isinstance(result, dict):
                    content = result.get('content', str(result))
                else:
                    content = str(result)
                
                # Formatear resultado
                formatted_result = f"**Resultado {i+1}:**\n{content}\n"
                result_tokens = len(formatted_result.split())
                
                if current_tokens + result_tokens <= max_tokens:
                    formatted_parts.append(formatted_result)
                    current_tokens += result_tokens
                else:
                    # Truncar último resultado si es necesario
                    remaining_tokens = max_tokens - current_tokens
                    if remaining_tokens > 50:  # Solo si queda espacio significativo
                        truncated_content = ' '.join(content.split()[:remaining_tokens])
                        formatted_parts.append(f"**Resultado {i+1}:**\n{truncated_content}...\n")
                    break
            
            return "\n".join(formatted_parts)
            
        except Exception as e:
            logger.error(f"Error formateando resultados RAG: {e}")
            return ""
    
    def _intelligent_truncate(self, context: str, max_tokens: int) -> str:
        """Trunca contexto inteligentemente manteniendo la información más relevante."""
        try:
            # Dividir por secciones
            sections = context.split('\n\n')
            
            # Priorizar secciones (RAG > Dynamic > Static)
            section_priorities = {
                'Información Detallada': 3,
                'Información Rápida': 2,
                'Contexto Dinámico': 2,
                'Contexto Histórico': 1,
                'Conocimiento Base': 1
            }
            
            # Ordenar secciones por prioridad
            prioritized_sections = []
            for section in sections:
                priority = 0
                for key, value in section_priorities.items():
                    if key in section:
                        priority = value
                        break
                prioritized_sections.append((priority, section))
            
            prioritized_sections.sort(key=lambda x: x[0], reverse=True)
            
            # Construir contexto truncado
            truncated_parts = []
            current_tokens = 0
            
            for priority, section in prioritized_sections:
                section_tokens = len(section.split())
                if current_tokens + section_tokens <= max_tokens:
                    truncated_parts.append(section)
                    current_tokens += section_tokens
                else:
                    # Truncar última sección si es necesario
                    remaining_tokens = max_tokens - current_tokens
                    if remaining_tokens > 50:
                        truncated_section = ' '.join(section.split()[:remaining_tokens])
                        truncated_parts.append(truncated_section + "...")
                    break
            
            return '\n\n'.join(truncated_parts)
            
        except Exception as e:
            logger.error(f"Error en truncamiento inteligente: {e}")
            # Fallback: truncamiento simple
            words = context.split()
            return ' '.join(words[:max_tokens])
    
    def _update_stats(self, tokens_used: int, latency_ms: float, 
                     source_metadata: Dict[str, Any]):
        """Actualiza estadísticas."""
        try:
            self.stats['total_queries'] += 1
            
            # Actualizar latencia promedio
            current_avg = self.stats['avg_latency_ms']
            total_queries = self.stats['total_queries']
            self.stats['avg_latency_ms'] = (
                (current_avg * (total_queries - 1) + latency_ms) / total_queries
            )
            
            # Actualizar uso de tokens por fuente
            for source, metadata in source_metadata.items():
                if source in self.stats['token_usage']:
                    self.stats['token_usage'][source] += metadata['tokens_used']
            
        except Exception as e:
            logger.error(f"Error actualizando estadísticas: {e}")
    
    def get_comprehensive_context(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Obtiene contexto comprensivo como string simple.
        
        Args:
            query: Query del usuario
            context: Contexto adicional
            
        Returns:
            Contexto como string
        """
        try:
            result = self.generate_context(query, context)
            return result.get('context', '')
        except Exception as e:
            logger.error(f"Error obteniendo contexto comprensivo: {e}")
            return ""
    
    def should_use_full_rag(self, query: str, context: Dict[str, Any] = None) -> bool:
        """
        Decide si usar FullRAG basado en complejidad de la query.
        
        Args:
            query: Query del usuario
            context: Contexto adicional
            
        Returns:
            True si debe usar FullRAG
        """
        try:
            # Criterios para usar FullRAG
            full_rag_criteria = {
                'research_query': any(term in query.lower() for term in [
                    'research', 'investigate', 'analyze', 'comprehensive',
                    'investigar', 'analizar', 'completo', 'detallado'
                ]),
                'technical_depth': any(term in query.lower() for term in [
                    'architecture', 'design pattern', 'best practice',
                    'arquitectura', 'patrón', 'mejores prácticas'
                ]),
                'comparison': any(term in query.lower() for term in [
                    'compare', 'vs', 'versus', 'difference',
                    'comparar', 'diferencia', 'ventajas'
                ]),
                'long_query': len(query) > 100,
                'multiple_concepts': len(query.split()) > 20
            }
            
            # Usar FullRAG si se cumple cualquier criterio
            should_use = any(full_rag_criteria.values())
            
            logger.debug(f"Decisión FullRAG: {should_use}, criterios: {full_rag_criteria}")
            
            return should_use
            
        except Exception as e:
            logger.error(f"Error decidiendo uso de FullRAG: {e}")
            return False
    
    def get_context_analysis(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Obtiene análisis detallado del contexto sin generar el contenido completo.
        
        Args:
            query: Query del usuario
            context: Contexto adicional
            
        Returns:
            Análisis del contexto
        """
        try:
            if self.awareness_gate:
                source_decision = self.awareness_gate.decide_sources(
                    query, self.max_tokens, context
                )
            else:
                source_decision = {
                    'sources': {
                        'static_cache': True,
                        'dynamic_context': True,
                        'rag': True
                    },
                    'token_budget': {
                        'static_cache': self.token_limits['static_cache'],
                        'dynamic_context': self.token_limits['dynamic_context'],
                        'rag': self.token_limits['mini_rag'] + self.token_limits['full_rag']
                    },
                    'confidence': 0.7
                }
            
            # Analizar disponibilidad de fuentes
            source_analysis = {}
            
            if self.static_cache:
                source_analysis['static_cache'] = {
                    'available': True,
                    'estimated_tokens': source_decision['token_budget'].get('static_cache', 0),
                    'categories': self.static_cache.get_categories()
                }
            
            if self.dynamic_context:
                source_analysis['dynamic_context'] = {
                    'available': True,
                    'estimated_tokens': source_decision['token_budget'].get('dynamic_context', 0),
                    'active_entries': self.dynamic_context.get_stats().get('active_entries', 0)
                }
            
            if self.mini_rag or self.full_rag:
                source_analysis['rag'] = {
                    'available': True,
                    'mini_rag_available': self.mini_rag is not None,
                    'full_rag_available': self.full_rag is not None,
                    'estimated_tokens': source_decision['token_budget'].get('rag', 0),
                    'should_use_full_rag': self.should_use_full_rag(query, context)
                }
            
            return {
                'sources_analysis': source_analysis,
                'total_estimated_tokens': sum(
                    budget for budget in source_decision['token_budget'].values()
                ),
                'confidence': source_decision.get('confidence', 0.7),
                'complexity_score': self._calculate_complexity_score(query),
                'recommended_approach': self._get_recommended_approach(query, context)
            }
            
        except Exception as e:
            logger.error(f"Error analizando contexto: {e}")
            return {
                'sources_analysis': {},
                'total_estimated_tokens': 0,
                'confidence': 0.0,
                'complexity_score': 0.0,
                'recommended_approach': 'fallback',
                'error': str(e)
            }
    
    def _calculate_complexity_score(self, query: str) -> float:
        """Calcula puntuación de complejidad de la query."""
        try:
            score = 0.0
            
            # Factores de complejidad
            factors = {
                'length': min(len(query) / 200, 1.0),  # Normalizar por 200 chars
                'word_count': min(len(query.split()) / 30, 1.0),  # Normalizar por 30 palabras
                'technical_terms': len([term for term in [
                    'algorithm', 'architecture', 'optimization', 'implementation',
                    'algoritmo', 'arquitectura', 'optimización', 'implementación'
                ] if term in query.lower()]) / 4,  # Normalizar por 4 términos
                'question_marks': min(query.count('?') / 3, 1.0),  # Normalizar por 3 preguntas
                'complex_words': len([word for word in query.split() if len(word) > 8]) / 5
            }
            
            # Peso de factores
            weights = [0.2, 0.2, 0.3, 0.15, 0.15]
            
            for factor, weight in zip(factors.values(), weights):
                score += factor * weight
            
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculando puntuación de complejidad: {e}")
            return 0.5
    
    def _get_recommended_approach(self, query: str, context: Dict[str, Any]) -> str:
        """Obtiene enfoque recomendado basado en análisis."""
        try:
            complexity_score = self._calculate_complexity_score(query)
            
            if complexity_score > 0.8:
                return 'comprehensive'  # Usar todas las fuentes
            elif complexity_score > 0.5:
                return 'balanced'  # Usar fuentes principales
            else:
                return 'focused'  # Usar solo fuentes más relevantes
                
        except Exception as e:
            logger.error(f"Error obteniendo enfoque recomendado: {e}")
            return 'balanced'
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de FullCAG."""
        try:
            return {
                'total_queries': self.stats['total_queries'],
                'avg_latency_ms': round(self.stats['avg_latency_ms'], 2),
                'token_usage': self.stats['token_usage'].copy(),
                'source_usage': self.stats['source_usage'].copy(),
                'rag_escalations': self.stats['rag_escalations'],
                'max_tokens': self.max_tokens,
                'token_limits': self.token_limits.copy(),
                'components_available': {
                    'static_cache': self.static_cache is not None,
                    'dynamic_context': self.dynamic_context is not None,
                    'awareness_gate': self.awareness_gate is not None,
                    'mini_rag': self.mini_rag is not None,
                    'full_rag': self.full_rag is not None
                }
            }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}
    
    def reset_stats(self):
        """Resetea estadísticas."""
        try:
            self.stats = {
                'total_queries': 0,
                'avg_latency_ms': 0,
                'token_usage': {
                    'static_cache': 0,
                    'dynamic_context': 0,
                    'mini_rag': 0,
                    'full_rag': 0
                },
                'source_usage': {
                    'static_cache': 0,
                    'dynamic_context': 0,
                    'mini_rag': 0,
                    'full_rag': 0
                },
                'rag_escalations': 0
            }
            logger.info("Estadísticas de FullCAG reseteadas")
        except Exception as e:
            logger.error(f"Error reseteando estadísticas: {e}")


# Función de conveniencia
def create_full_cag(static_cache=None, dynamic_context=None, awareness_gate=None,
                   mini_rag=None, full_rag=None, max_tokens: int = 32000) -> FullCAG:
    """Crea una instancia de FullCAG."""
    return FullCAG(static_cache, dynamic_context, awareness_gate, 
                  mini_rag, full_rag, max_tokens)


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.INFO)
    
    # Crear FullCAG
    full_cag = create_full_cag()
    
    # Test generación de contexto
    query = "Explica la arquitectura completa de microservicios con patrones de comunicación"
    result = full_cag.generate_context(query)
    
    print("=== Test FullCAG ===")
    print(f"Query: {query}")
    print(f"Contexto: {result['context'][:200]}...")
    print(f"Tokens usados: {result['tokens_used']}")
    print(f"Latencia: {result['latency_ms']:.1f}ms")
    print(f"Fuentes: {result['sources_used']}")
    print(f"RAG escalado: {result['rag_escalated']}")
    
    # Test análisis de contexto
    analysis = full_cag.get_context_analysis(query)
    print(f"\nAnálisis:")
    print(f"Puntuación de complejidad: {analysis['complexity_score']:.2f}")
    print(f"Enfoque recomendado: {analysis['recommended_approach']}")
    
    # Test estadísticas
    stats = full_cag.get_stats()
    print(f"\nEstadísticas: {stats}")
