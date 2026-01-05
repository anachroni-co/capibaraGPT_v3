#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Router inteligente que decide cuándo escalar de 20B a 120B
basado en complejidad y confianza de dominio.
"""

import logging
import hashlib
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RouterModel20B:
    """
    Router inteligente que decide cuándo escalar de 20B a 120B
    basado en complejidad y confianza de dominio.
    """
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializa el router con modelo de embeddings y umbrales.
        
        Args:
            embedding_model_name: Nombre del modelo de embeddings a usar
        """
        self.embedding_model = self._load_embedding_model(embedding_model_name)
        self.complexity_threshold = 0.7
        self.domain_confidence_threshold = 0.6
        
        # Dominios conocidos para cálculo de confianza
        self.known_domains = {
            'programming': [
                'python', 'javascript', 'java', 'c++', 'sql', 'html', 'css',
                'django', 'flask', 'react', 'node', 'api', 'database', 'algorithm'
            ],
            'science': [
                'physics', 'chemistry', 'biology', 'mathematics', 'theory',
                'research', 'experiment', 'hypothesis', 'analysis'
            ],
            'business': [
                'marketing', 'finance', 'strategy', 'management', 'sales',
                'revenue', 'profit', 'investment', 'market'
            ],
            'general': [
                'what', 'how', 'why', 'when', 'where', 'explain', 'describe',
                'help', 'question', 'answer'
            ]
        }
        
        # Cache para embeddings de dominios
        self._domain_embeddings = None
        self._initialize_domain_embeddings()
        
        logger.info(f"RouterModel20B inicializado con umbrales: "
                   f"complejidad={self.complexity_threshold}, "
                   f"confianza={self.domain_confidence_threshold}")
    
    def _load_embedding_model(self, model_name: str) -> SentenceTransformer:
        """Carga el modelo de embeddings."""
        try:
            model = SentenceTransformer(model_name)
            logger.info(f"Modelo de embeddings {model_name} cargado exitosamente")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo de embeddings: {e}")
            raise
    
    def _initialize_domain_embeddings(self):
        """Inicializa embeddings para dominios conocidos."""
        try:
            domain_texts = []
            for domain, keywords in self.known_domains.items():
                domain_text = f"{domain}: {' '.join(keywords)}"
                domain_texts.append(domain_text)
            
            self._domain_embeddings = self.embedding_model.encode(domain_texts)
            logger.info("Embeddings de dominios inicializados")
        except Exception as e:
            logger.error(f"Error inicializando embeddings de dominios: {e}")
            self._domain_embeddings = None
    
    def should_escalate(self, query: str, context: Dict[str, Any] = None) -> bool:
        """
        Decide si escalar de 20B a 120B basado en complejidad y confianza.
        
        Args:
            query: Query del usuario
            context: Contexto adicional (historial, metadata, etc.)
            
        Returns:
            True si debe escalar a 120B, False si usar 20B
        """
        if context is None:
            context = {}
        
        try:
            # Calcular métricas
            complexity = self._calculate_complexity(query)
            domain_conf = self._calculate_domain_confidence(query, context)
            
            # Decisión basada en umbrales
            decision = (complexity > self.complexity_threshold or 
                       domain_conf < self.domain_confidence_threshold)
            
            # Logging de la decisión
            self._log_decision(query, complexity, domain_conf, decision, context)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error en should_escalate: {e}")
            # En caso de error, usar 20B por defecto
            return False
    
    def _calculate_complexity(self, query: str) -> float:
        """
        Calcula la complejidad de la query (0.0 - 1.0).
        
        Factores considerados:
        - Longitud del texto
        - Número de entidades técnicas
        - Estructura sintáctica compleja
        - Presencia de múltiples conceptos
        """
        try:
            # Normalizar query
            query_lower = query.lower().strip()
            
            # Factor 1: Longitud (normalizada)
            length_score = min(len(query) / 500, 1.0)  # Max 500 chars = 1.0
            
            # Factor 2: Entidades técnicas
            technical_terms = [
                'algorithm', 'complexity', 'optimization', 'architecture',
                'implementation', 'framework', 'library', 'api', 'database',
                'machine learning', 'neural network', 'deep learning',
                'quantum', 'cryptography', 'blockchain', 'distributed'
            ]
            technical_count = sum(1 for term in technical_terms if term in query_lower)
            technical_score = min(technical_count / 5, 1.0)  # Max 5 términos = 1.0
            
            # Factor 3: Estructura sintáctica
            # Contar preguntas complejas, condicionales, etc.
            complex_patterns = [
                r'\?.*\?',  # Múltiples preguntas
                r'if.*then',  # Condicionales
                r'compare.*with',  # Comparaciones
                r'explain.*and.*how',  # Explicaciones complejas
                r'what.*why.*how'  # Múltiples interrogativos
            ]
            pattern_count = sum(1 for pattern in complex_patterns 
                              if re.search(pattern, query_lower))
            structure_score = min(pattern_count / 3, 1.0)  # Max 3 patrones = 1.0
            
            # Factor 4: Múltiples conceptos (usando embeddings)
            concept_score = self._calculate_concept_diversity(query)
            
            # Peso de factores
            weights = [0.2, 0.3, 0.2, 0.3]  # length, technical, structure, concepts
            scores = [length_score, technical_score, structure_score, concept_score]
            
            complexity = sum(w * s for w, s in zip(weights, scores))
            
            return min(max(complexity, 0.0), 1.0)  # Clamp entre 0 y 1
            
        except Exception as e:
            logger.error(f"Error calculando complejidad: {e}")
            return 0.5  # Valor por defecto
    
    def _calculate_concept_diversity(self, query: str) -> float:
        """Calcula diversidad de conceptos usando embeddings."""
        try:
            if self._domain_embeddings is None:
                return 0.5
            
            # Embedding de la query
            query_embedding = self.embedding_model.encode([query])
            
            # Calcular similitudes con dominios
            similarities = np.dot(query_embedding, self._domain_embeddings.T)[0]
            
            # Diversidad = 1 - similitud máxima (más diverso = menos similar a un dominio específico)
            max_similarity = np.max(similarities)
            diversity = 1.0 - max_similarity
            
            return diversity
            
        except Exception as e:
            logger.error(f"Error calculando diversidad de conceptos: {e}")
            return 0.5
    
    def _calculate_domain_confidence(self, query: str, context: Dict[str, Any]) -> float:
        """
        Calcula la confianza de dominio (0.0 - 1.0).
        
        Alta confianza = query se ajusta bien a dominios conocidos
        Baja confianza = query es ambigua o fuera de dominios conocidos
        """
        try:
            if self._domain_embeddings is None:
                return 0.5
            
            # Embedding de la query
            query_embedding = self.embedding_model.encode([query])
            
            # Calcular similitudes con dominios conocidos
            similarities = np.dot(query_embedding, self._domain_embeddings.T)[0]
            
            # Confianza = similitud máxima (qué tan bien se ajusta a un dominio)
            max_similarity = np.max(similarities)
            
            # Ajustar por contexto si está disponible
            if 'previous_domain' in context:
                # Bonus si el dominio coincide con consultas anteriores
                domain_names = list(self.known_domains.keys())
                if context['previous_domain'] in domain_names:
                    domain_idx = domain_names.index(context['previous_domain'])
                    domain_similarity = similarities[domain_idx]
                    max_similarity = max(max_similarity, domain_similarity * 1.1)
            
            return min(max(max_similarity, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculando confianza de dominio: {e}")
            return 0.5
    
    def _log_decision(self, query: str, complexity: float, domain_conf: float, 
                     decision: bool, context: Dict[str, Any]):
        """Registra la decisión de routing para análisis."""
        try:
            # Hash de la query para privacidad
            query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
            
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'query_hash': query_hash,
                'query_length': len(query),
                'complexity': round(complexity, 3),
                'domain_confidence': round(domain_conf, 3),
                'decision': 'escalate' if decision else 'use_20b',
                'model_selected': '120B' if decision else '20B',
                'context_keys': list(context.keys()) if context else []
            }
            
            logger.info(f"Routing decision: {log_data}")
            
            # Aquí se podría enviar a un sistema de métricas
            # self._send_to_metrics(log_data)
            
        except Exception as e:
            logger.error(f"Error logging decision: {e}")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del router."""
        return {
            'complexity_threshold': self.complexity_threshold,
            'domain_confidence_threshold': self.domain_confidence_threshold,
            'known_domains': list(self.known_domains.keys()),
            'embedding_model_loaded': self.embedding_model is not None,
            'domain_embeddings_loaded': self._domain_embeddings is not None
        }
    
    def update_thresholds(self, complexity_threshold: float = None, 
                         domain_confidence_threshold: float = None):
        """Actualiza los umbrales de decisión."""
        if complexity_threshold is not None:
            self.complexity_threshold = max(0.0, min(1.0, complexity_threshold))
            logger.info(f"Umbral de complejidad actualizado: {self.complexity_threshold}")
        
        if domain_confidence_threshold is not None:
            self.domain_confidence_threshold = max(0.0, min(1.0, domain_confidence_threshold))
            logger.info(f"Umbral de confianza de dominio actualizado: {self.domain_confidence_threshold}")


# Función de conveniencia para uso directo
def create_router(complexity_threshold: float = 0.7, 
                 domain_confidence_threshold: float = 0.6) -> RouterModel20B:
    """
    Crea una instancia del router con umbrales personalizados.
    
    Args:
        complexity_threshold: Umbral de complejidad (0.0-1.0)
        domain_confidence_threshold: Umbral de confianza de dominio (0.0-1.0)
        
    Returns:
        Instancia de RouterModel20B
    """
    router = RouterModel20B()
    router.update_thresholds(complexity_threshold, domain_confidence_threshold)
    return router


if __name__ == "__main__":
    # Test básico del router
    logging.basicConfig(level=logging.INFO)
    
    router = create_router()
    
    # Test queries
    test_queries = [
        "¿Qué es Python?",
        "Explica la teoría de cuerdas y su relación con la física cuántica",
        "¿Cómo implementar un algoritmo de machine learning distribuido?",
        "Ayuda con mi código",
        "Compara las arquitecturas de microservicios vs monolíticas"
    ]
    
    print("=== Test RouterModel20B ===")
    for query in test_queries:
        decision = router.should_escalate(query)
        print(f"Query: {query[:50]}...")
        print(f"Decisión: {'120B' if decision else '20B'}")
        print("-" * 50)
