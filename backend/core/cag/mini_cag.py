#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniCAG - Context-Aware Generation para modelo 20B (8K tokens max).
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class MiniCAG:
    """
    Context-Aware Generation para modelo 20B con límite de 8K tokens.
    Optimizado para velocidad y eficiencia.
    """
    
    def __init__(self, static_cache=None, dynamic_context=None, 
                 awareness_gate=None, max_tokens: int = 8000):
        """
        Inicializa MiniCAG.
        
        Args:
            static_cache: Instancia de StaticCache
            dynamic_context: Instancia de DynamicContext
            awareness_gate: Instancia de AwarenessGate
            max_tokens: Máximo tokens (8K para modelo 20B)
        """
        self.static_cache = static_cache
        self.dynamic_context = dynamic_context
        self.awareness_gate = awareness_gate
        self.max_tokens = max_tokens
        
        # Límites específicos para MiniCAG
        self.token_limits = {
            'static_cache': int(max_tokens * 0.4),  # 40% para conocimiento estático
            'dynamic_context': int(max_tokens * 0.3),  # 30% para contexto dinámico
            'rag': int(max_tokens * 0.3)  # 30% para RAG (si se usa)
        }
        
        # Métricas
        self.stats = {
            'total_queries': 0,
            'avg_latency_ms': 0,
            'token_usage': {
                'static_cache': 0,
                'dynamic_context': 0,
                'rag': 0
            },
            'source_usage': {
                'static_cache': 0,
                'dynamic_context': 0,
                'rag': 0
            }
        }
        
        logger.info(f"MiniCAG inicializado con límite de {max_tokens} tokens")
    
    def generate_context(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Genera contexto optimizado para modelo 20B.
        
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
                # Fallback: usar solo static_cache
                source_decision = {
                    'sources': {'static_cache': True},
                    'token_budget': {'static_cache': self.max_tokens},
                    'confidence': 0.5
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
            
            # Combinar contexto
            final_context = "\n\n".join(context_parts)
            
            # Verificar límite de tokens
            if actual_tokens > self.max_tokens:
                # Truncar si es necesario
                words = final_context.split()
                truncated_words = words[:self.max_tokens]
                final_context = " ".join(truncated_words) + "..."
                actual_tokens = len(truncated_words)
            
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
                'confidence': source_decision.get('confidence', 0.5),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"MiniCAG generó contexto: {actual_tokens} tokens, "
                        f"{latency_ms:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generando contexto en MiniCAG: {e}")
            return {
                'context': "",
                'tokens_used': 0,
                'tokens_available': self.max_tokens,
                'sources_used': [],
                'source_metadata': {},
                'latency_ms': (time.time() - start_time) * 1000,
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
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
    
    def get_optimal_context(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Obtiene contexto óptimo como string simple.
        
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
            logger.error(f"Error obteniendo contexto óptimo: {e}")
            return ""
    
    def should_escalate_to_full_cag(self, query: str, context: Dict[str, Any] = None) -> bool:
        """
        Decide si escalar a FullCAG basado en complejidad.
        
        Args:
            query: Query del usuario
            context: Contexto adicional
            
        Returns:
            True si debe escalar a FullCAG
        """
        try:
            # Criterios para escalación
            escalation_criteria = {
                'query_length': len(query) > 200,
                'complex_terms': any(term in query.lower() for term in [
                    'complex', 'advanced', 'detailed', 'comprehensive',
                    'complejo', 'avanzado', 'detallado', 'completo'
                ]),
                'multiple_questions': query.count('?') > 1,
                'technical_depth': any(term in query.lower() for term in [
                    'architecture', 'optimization', 'performance', 'scalability',
                    'arquitectura', 'optimización', 'rendimiento', 'escalabilidad'
                ])
            }
            
            # Escalar si se cumple cualquier criterio
            should_escalate = any(escalation_criteria.values())
            
            logger.debug(f"Decisión de escalación: {should_escalate}, "
                        f"criterios: {escalation_criteria}")
            
            return should_escalate
            
        except Exception as e:
            logger.error(f"Error decidiendo escalación: {e}")
            return False
    
    def get_context_summary(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Obtiene resumen del contexto sin generar el contenido completo.
        
        Args:
            query: Query del usuario
            context: Contexto adicional
            
        Returns:
            Resumen del contexto
        """
        try:
            if self.awareness_gate:
                source_decision = self.awareness_gate.decide_sources(
                    query, self.max_tokens, context
                )
            else:
                source_decision = {
                    'sources': {'static_cache': True},
                    'token_budget': {'static_cache': self.max_tokens},
                    'confidence': 0.5
                }
            
            # Estimar disponibilidad de fuentes
            source_availability = {}
            
            if self.static_cache:
                source_availability['static_cache'] = {
                    'available': True,
                    'estimated_tokens': source_decision['token_budget'].get('static_cache', 0)
                }
            
            if self.dynamic_context:
                source_availability['dynamic_context'] = {
                    'available': True,
                    'estimated_tokens': source_decision['token_budget'].get('dynamic_context', 0)
                }
            
            return {
                'sources_available': source_availability,
                'total_estimated_tokens': sum(
                    budget for budget in source_decision['token_budget'].values()
                ),
                'confidence': source_decision.get('confidence', 0.5),
                'escalation_recommended': self.should_escalate_to_full_cag(query, context)
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo resumen de contexto: {e}")
            return {
                'sources_available': {},
                'total_estimated_tokens': 0,
                'confidence': 0.0,
                'escalation_recommended': False,
                'error': str(e)
            }
    
    def optimize_for_speed(self, query: str) -> Dict[str, Any]:
        """
        Optimiza contexto para máxima velocidad.
        
        Args:
            query: Query del usuario
            
        Returns:
            Contexto optimizado para velocidad
        """
        try:
            # Usar solo static_cache para máxima velocidad
            if not self.static_cache:
                return {'context': '', 'tokens_used': 0, 'latency_ms': 0}
            
            start_time = time.time()
            
            # Usar solo una fracción de tokens para velocidad
            speed_tokens = int(self.max_tokens * 0.3)  # Solo 30% para velocidad
            context = self.static_cache.retrieve(query, speed_tokens)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                'context': context,
                'tokens_used': len(context.split()),
                'latency_ms': latency_ms,
                'optimized_for': 'speed'
            }
            
        except Exception as e:
            logger.error(f"Error optimizando para velocidad: {e}")
            return {'context': '', 'tokens_used': 0, 'latency_ms': 0, 'error': str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de MiniCAG."""
        try:
            return {
                'total_queries': self.stats['total_queries'],
                'avg_latency_ms': round(self.stats['avg_latency_ms'], 2),
                'token_usage': self.stats['token_usage'].copy(),
                'source_usage': self.stats['source_usage'].copy(),
                'max_tokens': self.max_tokens,
                'token_limits': self.token_limits.copy(),
                'components_available': {
                    'static_cache': self.static_cache is not None,
                    'dynamic_context': self.dynamic_context is not None,
                    'awareness_gate': self.awareness_gate is not None
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
                    'rag': 0
                },
                'source_usage': {
                    'static_cache': 0,
                    'dynamic_context': 0,
                    'rag': 0
                }
            }
            logger.info("Estadísticas de MiniCAG reseteadas")
        except Exception as e:
            logger.error(f"Error reseteando estadísticas: {e}")


# Función de conveniencia
def create_mini_cag(static_cache=None, dynamic_context=None, 
                   awareness_gate=None, max_tokens: int = 8000) -> MiniCAG:
    """Crea una instancia de MiniCAG."""
    return MiniCAG(static_cache, dynamic_context, awareness_gate, max_tokens)


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.INFO)
    
    # Crear MiniCAG
    mini_cag = create_mini_cag()
    
    # Test generación de contexto
    query = "¿Qué es Python y cómo se usa?"
    result = mini_cag.generate_context(query)
    
    print("=== Test MiniCAG ===")
    print(f"Query: {query}")
    print(f"Contexto: {result['context'][:200]}...")
    print(f"Tokens usados: {result['tokens_used']}")
    print(f"Latencia: {result['latency_ms']:.1f}ms")
    print(f"Fuentes: {result['sources_used']}")
    
    # Test escalación
    complex_query = "Explica la arquitectura completa de microservicios con patrones avanzados de comunicación"
    should_escalate = mini_cag.should_escalate_to_full_cag(complex_query)
    print(f"\nEscalación recomendada: {should_escalate}")
    
    # Test estadísticas
    stats = mini_cag.get_stats()
    print(f"\nEstadísticas: {stats}")
