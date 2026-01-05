#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AwarenessGate - Decide qué fuentes de contexto utilizar.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class AwarenessGate:
    """
    Decide qué fuentes de contexto utilizar basado en análisis de la query.
    """
    
    def __init__(self):
        """Inicializa el AwarenessGate."""
        # Patrones para detectar necesidades de contexto
        self.context_patterns = {
            'static_cache': [
                r'\b(what|qué|qué es|definir|definición)\b',
                r'\b(how|how to|cómo|cómo hacer|tutorial|guide)\b',
                r'\b(explain|explicar|explica)\b',
                r'\b(basics|básico|fundamentos|introduction)\b'
            ],
            'dynamic_context': [
                r'\b(recent|reciente|último|latest|nuevo)\b',
                r'\b(conversation|conversación|chat|historial)\b',
                r'\b(continue|continuar|sigue|follow up)\b',
                r'\b(context|contexto|anterior|before)\b'
            ],
            'rag': [
                r'\b(specific|específico|detallado|detailed)\b',
                r'\b(research|investigar|buscar|find)\b',
                r'\b(documentation|documentación|docs|manual)\b',
                r'\b(example|ejemplo|ejemplos|sample)\b',
                r'\b(code|código|implementation|implementación)\b'
            ]
        }
        
        # Dominios que requieren fuentes específicas
        self.domain_requirements = {
            'programming': ['static_cache', 'rag'],
            'api': ['static_cache', 'rag'],
            'database': ['static_cache', 'rag'],
            'general': ['static_cache'],
            'conversation': ['dynamic_context'],
            'research': ['rag', 'static_cache']
        }
        
        # Umbrales de confianza
        self.confidence_thresholds = {
            'static_cache': 0.3,
            'dynamic_context': 0.4,
            'rag': 0.5
        }
        
        logger.info("AwarenessGate inicializado")
    
    def decide_sources(self, query: str, available_tokens: int, 
                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Decide qué fuentes de contexto utilizar.
        
        Args:
            query: Query del usuario
            available_tokens: Tokens disponibles
            context: Contexto adicional
            
        Returns:
            Diccionario con decisión de fuentes y presupuesto de tokens
        """
        try:
            if context is None:
                context = {}
            
            # Analizar query
            analysis = self._analyze_query(query, context)
            
            # Decidir fuentes basado en análisis
            sources = self._select_sources(analysis)
            
            # Asignar presupuesto de tokens
            token_budget = self._allocate_token_budget(sources, available_tokens)
            
            decision = {
                'sources': sources,
                'token_budget': token_budget,
                'analysis': analysis,
                'confidence': self._calculate_confidence(analysis, sources),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"Decisión de fuentes: {sources}")
            return decision
            
        except Exception as e:
            logger.error(f"Error decidiendo fuentes: {e}")
            # Fallback: usar solo static_cache
            return {
                'sources': {'static_cache': True},
                'token_budget': {'static_cache': available_tokens},
                'analysis': {'error': str(e)},
                'confidence': 0.5,
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza la query para determinar necesidades de contexto."""
        try:
            query_lower = query.lower()
            
            analysis = {
                'query_length': len(query),
                'word_count': len(query.split()),
                'pattern_matches': {},
                'domain_hints': [],
                'complexity_indicators': [],
                'context_indicators': []
            }
            
            # Detectar patrones
            for source, patterns in self.context_patterns.items():
                matches = []
                for pattern in patterns:
                    if re.search(pattern, query_lower):
                        matches.append(pattern)
                analysis['pattern_matches'][source] = matches
            
            # Detectar dominios
            analysis['domain_hints'] = self._detect_domains(query_lower)
            
            # Detectar indicadores de complejidad
            analysis['complexity_indicators'] = self._detect_complexity(query_lower)
            
            # Detectar indicadores de contexto
            analysis['context_indicators'] = self._detect_context_indicators(query_lower, context)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando query: {e}")
            return {'error': str(e)}
    
    def _detect_domains(self, query_lower: str) -> List[str]:
        """Detecta dominios en la query."""
        domains = []
        
        domain_keywords = {
            'programming': ['python', 'javascript', 'java', 'code', 'programming', 'function', 'class'],
            'api': ['api', 'rest', 'endpoint', 'request', 'response', 'http'],
            'database': ['sql', 'database', 'query', 'table', 'select', 'insert'],
            'general': ['what', 'how', 'why', 'when', 'where', 'explain'],
            'conversation': ['continue', 'previous', 'before', 'earlier', 'conversation'],
            'research': ['research', 'find', 'search', 'investigate', 'analyze']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                domains.append(domain)
        
        return domains
    
    def _detect_complexity(self, query_lower: str) -> List[str]:
        """Detecta indicadores de complejidad."""
        indicators = []
        
        complexity_patterns = {
            'multi_part': r'\b(and|y|also|también|additionally|además)\b',
            'conditional': r'\b(if|si|when|cuando|unless|a menos que)\b',
            'comparison': r'\b(compare|comparar|vs|versus|difference|diferencia)\b',
            'technical': r'\b(algorithm|algoritmo|architecture|arquitectura|optimization|optimización)\b',
            'specific': r'\b(specific|específico|exact|exacto|precise|preciso)\b'
        }
        
        for indicator, pattern in complexity_patterns.items():
            if re.search(pattern, query_lower):
                indicators.append(indicator)
        
        return indicators
    
    def _detect_context_indicators(self, query_lower: str, context: Dict[str, Any]) -> List[str]:
        """Detecta indicadores de necesidad de contexto."""
        indicators = []
        
        # Indicadores de contexto histórico
        if any(word in query_lower for word in ['previous', 'anterior', 'before', 'antes']):
            indicators.append('historical_context')
        
        # Indicadores de contexto de conversación
        if any(word in query_lower for word in ['continue', 'continuar', 'follow up', 'seguir']):
            indicators.append('conversation_context')
        
        # Indicadores de contexto específico
        if any(word in query_lower for word in ['based on', 'basado en', 'according to', 'según']):
            indicators.append('reference_context')
        
        # Verificar si hay contexto disponible
        if context.get('conversation_history'):
            indicators.append('has_conversation_history')
        
        if context.get('user_context'):
            indicators.append('has_user_context')
        
        return indicators
    
    def _select_sources(self, analysis: Dict[str, Any]) -> Dict[str, bool]:
        """Selecciona fuentes basado en análisis."""
        try:
            sources = {
                'static_cache': False,
                'dynamic_context': False,
                'rag': False
            }
            
            # Decisión basada en patrones
            pattern_matches = analysis.get('pattern_matches', {})
            for source, matches in pattern_matches.items():
                if matches and len(matches) > 0:
                    sources[source] = True
            
            # Decisión basada en dominios
            domain_hints = analysis.get('domain_hints', [])
            for domain in domain_hints:
                if domain in self.domain_requirements:
                    for required_source in self.domain_requirements[domain]:
                        sources[required_source] = True
            
            # Decisión basada en complejidad
            complexity_indicators = analysis.get('complexity_indicators', [])
            if 'technical' in complexity_indicators or 'specific' in complexity_indicators:
                sources['rag'] = True
            
            if 'multi_part' in complexity_indicators or 'comparison' in complexity_indicators:
                sources['static_cache'] = True
            
            # Decisión basada en indicadores de contexto
            context_indicators = analysis.get('context_indicators', [])
            if any(indicator in context_indicators for indicator in 
                   ['historical_context', 'conversation_context', 'has_conversation_history']):
                sources['dynamic_context'] = True
            
            # Fallback: siempre incluir static_cache si no hay otras fuentes
            if not any(sources.values()):
                sources['static_cache'] = True
            
            return sources
            
        except Exception as e:
            logger.error(f"Error seleccionando fuentes: {e}")
            return {'static_cache': True, 'dynamic_context': False, 'rag': False}
    
    def _allocate_token_budget(self, sources: Dict[str, bool], 
                             available_tokens: int) -> Dict[str, int]:
        """Asigna presupuesto de tokens a cada fuente."""
        try:
            # Contar fuentes activas
            active_sources = [source for source, active in sources.items() if active]
            
            if not active_sources:
                return {'static_cache': available_tokens}
            
            # Asignación base
            base_allocation = available_tokens // len(active_sources)
            
            # Ajustes por prioridad
            priority_weights = {
                'static_cache': 1.0,
                'dynamic_context': 0.8,
                'rag': 1.2  # RAG necesita más tokens para búsquedas profundas
            }
            
            budget = {}
            total_weighted = sum(priority_weights[source] for source in active_sources)
            
            for source in active_sources:
                weight = priority_weights.get(source, 1.0)
                allocation = int((weight / total_weighted) * available_tokens)
                budget[source] = max(100, allocation)  # Mínimo 100 tokens
            
            # Ajustar para que no exceda el total
            total_allocated = sum(budget.values())
            if total_allocated > available_tokens:
                # Reducir proporcionalmente
                ratio = available_tokens / total_allocated
                for source in budget:
                    budget[source] = int(budget[source] * ratio)
            
            return budget
            
        except Exception as e:
            logger.error(f"Error asignando presupuesto: {e}")
            return {'static_cache': available_tokens}
    
    def _calculate_confidence(self, analysis: Dict[str, Any], 
                            sources: Dict[str, bool]) -> float:
        """Calcula confianza en la decisión."""
        try:
            confidence = 0.5  # Base
            
            # Aumentar confianza por patrones detectados
            pattern_matches = analysis.get('pattern_matches', {})
            total_patterns = sum(len(matches) for matches in pattern_matches.values())
            confidence += min(0.3, total_patterns * 0.05)
            
            # Aumentar confianza por dominios detectados
            domain_hints = analysis.get('domain_hints', [])
            confidence += min(0.2, len(domain_hints) * 0.1)
            
            # Aumentar confianza por indicadores de contexto
            context_indicators = analysis.get('context_indicators', [])
            confidence += min(0.2, len(context_indicators) * 0.05)
            
            # Penalizar si no se detectaron fuentes claras
            active_sources = sum(sources.values())
            if active_sources == 0:
                confidence -= 0.3
            elif active_sources == 1 and 'static_cache' in sources and sources['static_cache']:
                confidence -= 0.1  # Solo fallback
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculando confianza: {e}")
            return 0.5
    
    def should_use_rag(self, query: str, context: Dict[str, Any] = None) -> bool:
        """Decide si usar RAG basado en la query."""
        try:
            decision = self.decide_sources(query, 1000, context)
            return decision['sources'].get('rag', False)
        except Exception as e:
            logger.error(f"Error decidiendo uso de RAG: {e}")
            return False
    
    def should_use_dynamic_context(self, query: str, context: Dict[str, Any] = None) -> bool:
        """Decide si usar contexto dinámico basado en la query."""
        try:
            decision = self.decide_sources(query, 1000, context)
            return decision['sources'].get('dynamic_context', False)
        except Exception as e:
            logger.error(f"Error decidiendo uso de contexto dinámico: {e}")
            return False
    
    def get_optimal_token_distribution(self, query: str, available_tokens: int,
                                     context: Dict[str, Any] = None) -> Dict[str, int]:
        """Obtiene distribución óptima de tokens."""
        try:
            decision = self.decide_sources(query, available_tokens, context)
            return decision['token_budget']
        except Exception as e:
            logger.error(f"Error obteniendo distribución de tokens: {e}")
            return {'static_cache': available_tokens}
    
    def update_patterns(self, source: str, patterns: List[str]):
        """Actualiza patrones para una fuente."""
        try:
            if source in self.context_patterns:
                self.context_patterns[source] = patterns
                logger.info(f"Patrones actualizados para {source}")
            else:
                logger.warning(f"Fuente {source} no encontrada")
        except Exception as e:
            logger.error(f"Error actualizando patrones: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del AwarenessGate."""
        return {
            'total_patterns': sum(len(patterns) for patterns in self.context_patterns.values()),
            'sources': list(self.context_patterns.keys()),
            'domains': list(self.domain_requirements.keys()),
            'confidence_thresholds': self.confidence_thresholds
        }


# Función de conveniencia
def create_awareness_gate() -> AwarenessGate:
    """Crea una instancia de AwarenessGate."""
    return AwarenessGate()


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.INFO)
    
    # Crear AwarenessGate
    gate = create_awareness_gate()
    
    # Test queries
    test_queries = [
        "¿Qué es Python?",
        "¿Cómo implementar una API REST?",
        "Continúa con la conversación anterior",
        "Busca documentación específica sobre Flask",
        "Explica la diferencia entre SQL y NoSQL"
    ]
    
    print("=== Test AwarenessGate ===")
    for query in test_queries:
        decision = gate.decide_sources(query, 2000)
        print(f"Query: {query}")
        print(f"Fuentes: {decision['sources']}")
        print(f"Presupuesto: {decision['token_budget']}")
        print(f"Confianza: {decision['confidence']:.2f}")
        print("-" * 50)
    
    # Test estadísticas
    stats = gate.get_stats()
    print(f"Estadísticas: {stats}")
