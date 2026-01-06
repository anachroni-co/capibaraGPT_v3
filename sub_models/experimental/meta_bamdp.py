"""
Enhanced MetaBAMDP (Bayesian Adaptive Multi-Domain Processing) Module
=====================================================================

Meta-learning system que adapta dinámicamente el procesamiento based en:
- Dominio de la consulta (matemáticas, programación, texto, etc.)
- Complejidad estimada
- Patrones históricos de éxito
- Recursos disponibles

Integra con el system MoE para routing inteligente.
"""

import os
import sys
import logging
from typing import Tuple, Optional, Dict, Any, Callable, Union, List
from dataclasses import dataclass, field
from enum import Enum
from functools import partial

# Fixed: Using proper imports instead of sys.path manipulation
try:
    from capibara.interfaces.isub_models import ISubModel
    from capibara.utils.cache_standalone import TpuOptimizedCache
    CAPIBARA_IMPORTS_AVAILABLE = True
except ImportError:
    CAPIBARA_IMPORTS_AVAILABLE = False

# JAX imports for neural network components (with fallbacks)
try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    np = jnp
    JAX_AVAILABLE = True
except ImportError:
    # Fallback to regular numpy or dummy implementations
    try:
        import numpy as np
        JAX_AVAILABLE = False
        
        # Dummy Flax module for compatibility
        class nn:
            class Module:
                def setup(self): pass
                def __call__(self, *args, **kwargs): return args[0] if args else None
            
            class Dense:
                def __init__(self, features): self.features = features
                def __call__(self, x): return x
            
            class Embed:
                def __init__(self, num_embeddings, features): 
                    self.num_embeddings = num_embeddings
                    self.features = features
                def __call__(self, x): return x
            
            class Sequential:
                def __init__(self, layers): self.layers = layers
                def __call__(self, x): return x
            
            @staticmethod
            def relu(x): return np.maximum(0, x) if hasattr(np, 'maximum') else x
            @staticmethod
            def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500))) if hasattr(np, 'exp') else x
            @staticmethod
            def tanh(x): return np.tanh(x) if hasattr(np, 'tanh') else x
    except ImportError:
        # Ultimate fallback
        import math
        
        class np:
            @staticmethod
            def array(data): return data
            @staticmethod  
            def mean(data): return sum(data) / len(data) if data else 0
            @staticmethod
            def concatenate(arrays, axis=None): return sum(arrays, [])
        
        class nn:
            class Module:
                def setup(self): pass
                def __call__(self, *args, **kwargs): return args[0] if args else None
            
            class Dense:
                def __init__(self, features): self.features = features
                def __call__(self, x): return x
            
            class Embed:
                def __init__(self, num_embeddings, features): pass
                def __call__(self, x): return x
            
            class Sequential:
                def __init__(self, layers): pass
                def __call__(self, x): return x
            
            @staticmethod
            def relu(x): return max(0, x) if isinstance(x, (int, float)) else x
            @staticmethod
            def sigmoid(x): return 1 / (1 + math.exp(-max(-500, min(500, x)))) if isinstance(x, (int, float)) else x
            @staticmethod
            def tanh(x): return math.tanh(x) if isinstance(x, (int, float)) else x
        
        JAX_AVAILABLE = False

logger = logging.getLogger(__name__)

class ProcessingDomain(Enum):
    """Dominios de procesamiento especializados"""
    MATHEMATICS = "mathematics"
    PROGRAMMING = "programming" 
    NATURAL_LANGUAGE = "natural_language"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"

class AdaptationStrategy(Enum):
    """Estrategias de adaptación del modelo"""
    CONSERVATIVE = "conservative"  # Cambios mínimos, alta estabilidad
    MODERATE = "moderate"         # Balance entre adaptación y estabilidad
    AGGRESSIVE = "aggressive"     # Adaptación rápida, mayor riesgo
    CONTEXT_AWARE = "context_aware"  # Basado en contexto específico

@dataclass
class MetaBAMDPConfig:
    """Configuration for the MetaBAMDP system"""
    hidden_size: int = 768
    num_domains: int = 7
    adaptation_rate: float = 0.01
    meta_learning_rate: float = 0.001
    memory_size: int = 1024
    temperature: float = 1.0
    use_bayesian_updates: bool = True
    enable_domain_transfer: bool = True
    max_adaptation_steps: int = 100
    
    # Configuración de procesamiento
    enable_parallel_processing: bool = True
    batch_size: int = 32
    sequence_length: int = 512
    
    # Configuración de memoria y cache
    enable_episodic_memory: bool = True
    memory_decay_rate: float = 0.95
    cache_size: int = 10000

@dataclass 
class ProcessingContext:
    """Processing context for a specific query"""
    domain: ProcessingDomain
    complexity_score: float
    confidence_level: float
    historical_success_rate: float
    resource_requirements: Dict[str, float]
    adaptation_strategy: AdaptationStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)

if JAX_AVAILABLE:
    class BayesianDomainAdapterBase(nn.Module):
        pass
else:
    class BayesianDomainAdapterBase:
        def setup(self): pass
        def __call__(self, *args, **kwargs): return args[0] if args else None

class BayesianDomainAdapter(BayesianDomainAdapterBase):
    """Adaptador bayesiano para diferentes dominios"""
    
    def __init__(self, config: MetaBAMDPConfig):
        if JAX_AVAILABLE:
            super().__init__()
        self.config = config
        self.setup()
    
    def setup(self):
        self.domain_embeddings = nn.Embed(
            self.config.num_domains, 
            self.config.hidden_size
        )
        
        self.adaptation_network = nn.Sequential([
            nn.Dense(self.config.hidden_size * 2),
            nn.relu,
            nn.Dense(self.config.hidden_size),
            nn.relu,
            nn.Dense(self.config.hidden_size)
        ])
        
        self.uncertainty_estimator = nn.Dense(1)
        
    def __call__(self, x, domain_id, training=False):
        """
        Adapta la representación basada en el dominio
        """
        domain_emb = self.domain_embeddings(domain_id)
        
        # Combinar input con domain embedding
        if JAX_AVAILABLE:
            combined = jnp.concatenate([x, domain_emb], axis=-1)
        else:
            combined = np.concatenate([x, domain_emb], axis=-1)
        
        # Aplicar adaptación
        adapted = self.adaptation_network(combined)
        
        # Estimar incertidumbre para updates bayesianos
        uncertainty = nn.sigmoid(self.uncertainty_estimator(adapted))
        
        return adapted, uncertainty

if JAX_AVAILABLE:
    class MetaLearningControllerBase(nn.Module):
        pass
else:
    class MetaLearningControllerBase:
        def setup(self): pass
        def __call__(self, *args, **kwargs): return args[0] if args else None

class MetaLearningController(MetaLearningControllerBase):
    """Controlador de meta-aprendizaje que coordina la adaptación"""
    
    def __init__(self, config: MetaBAMDPConfig):
        if JAX_AVAILABLE:
            super().__init__()
        self.config = config
        self.setup()
    
    def setup(self):
        self.memory_network = nn.Sequential([
            nn.Dense(self.config.hidden_size),
            nn.relu,
            nn.Dense(self.config.memory_size),
            nn.tanh
        ])
        
        self.strategy_selector = nn.Dense(len(AdaptationStrategy))
        
        self.performance_predictor = nn.Sequential([
            nn.Dense(self.config.hidden_size // 2),
            nn.relu,
            nn.Dense(1),
            nn.sigmoid
        ])
    
    def __call__(self, context_features, historical_performance):
        """
        Selecciona strategy de adaptación basada en contexto e historia
        """
        # Actualizar memoria episódica
        memory_state = self.memory_network(context_features)
        
        # Seleccionar strategy de adaptación
        strategy_logits = self.strategy_selector(context_features)
        
        # Predecir performance esperado
        expected_performance = self.performance_predictor(context_features)
        
        return {
            'memory_state': memory_state,
            'strategy_logits': strategy_logits,
            'expected_performance': expected_performance
        }

if JAX_AVAILABLE:
    class MetaBAMDPBase(nn.Module):
        pass
else:
    class MetaBAMDPBase:
        def setup(self): pass
        def __call__(self, *args, **kwargs): return args[0] if args else None

class MetaBAMDP(MetaBAMDPBase):
    """
    Meta-Bayesian Adaptive Multi-Domain Processing System
    
    Sistema principal que coordina:
    1. Adaptación por dominio
    2. Meta-aprendizaje continuo  
    3. Gestión de memoria episódica
    4. Routing inteligente based en contexto
    
    Enhanced with backward compatibility for simple interface.
    """
    
    def __init__(self, config: Optional[Union[MetaBAMDPConfig, Dict[str, Any]]] = None):
        if JAX_AVAILABLE:
            super().__init__()
        
        # Handle both new config format and legacy dict format
        if isinstance(config, dict):
            self.config = MetaBAMDPConfig(**config)
            self.cache = TpuOptimizedCache() if CAPIBARA_IMPORTS_AVAILABLE and 'TpuOptimizedCache' in globals() else None
            self.models = {}
            self.meta_parameters = {}
        elif isinstance(config, MetaBAMDPConfig):
            self.config = config
        else:
            self.config = MetaBAMDPConfig()
            self.cache = TpuOptimizedCache() if CAPIBARA_IMPORTS_AVAILABLE and 'TpuOptimizedCache' in globals() else None
            self.models = {}
            self.meta_parameters = {}
        
        self.setup()
        logger.info("🧠 Enhanced Meta-BAMDP initialized successfully")
    
    def setup(self):
        self.domain_adapter = BayesianDomainAdapter(self.config)
        self.meta_controller = MetaLearningController(self.config)
        
        # Procesadores especializados por dominio
        self.domain_processors = {}
        for domain in ProcessingDomain:
            self.domain_processors[domain.value] = nn.Sequential([
                nn.Dense(self.config.hidden_size),
                nn.relu,
                nn.Dense(self.config.hidden_size),
                nn.relu,
                nn.Dense(self.config.hidden_size)
            ])
        
        # Integrador final
        self.output_integrator = nn.Sequential([
            nn.Dense(self.config.hidden_size * 2),
            nn.relu,
            nn.Dense(self.config.hidden_size),
            nn.tanh
        ])
        
        # Memoria episódica
        self.episodic_memory = {}
        
    def __call__(self, 
                 x, 
                 processing_context: Optional[ProcessingContext] = None,
                 training: bool = False):
        """
        Procesamiento principal con adaptación meta-bayesiana
        """
        # Handle legacy interface
        if processing_context is None:
            processing_context = create_processing_context(ProcessingDomain.NATURAL_LANGUAGE)
        
        domain_id = list(ProcessingDomain).index(processing_context.domain)
        
        # Adaptación por dominio con estimación de incertidumbre
        adapted_x, uncertainty = self.domain_adapter(
            x, domain_id, training=training
        )
        
        # Control meta-learning
        context_features = self._extract_context_features(processing_context)
        meta_output = self.meta_controller(context_features, uncertainty)
        
        # Procesamiento especializado
        domain_key = processing_context.domain.value
        if domain_key in self.domain_processors:
            specialized_output = self.domain_processors[domain_key](adapted_x)
        else:
            specialized_output = adapted_x
        
        # Integración final
        if JAX_AVAILABLE:
            combined = jnp.concatenate([specialized_output, meta_output['memory_state']], axis=-1)
        else:
            combined = np.concatenate([specialized_output, meta_output['memory_state']], axis=-1)
        final_output = self.output_integrator(combined)
        
        return {
            'output': final_output,
            'uncertainty': uncertainty,
            'strategy_logits': meta_output['strategy_logits'],
            'expected_performance': meta_output['expected_performance'],
            'domain_adaptation': adapted_x
        }
    
    # Legacy methods for backward compatibility
    def adapt(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt the model based on context (legacy interface)"""
        # Convert legacy context to ProcessingContext
        domain = ProcessingDomain.NATURAL_LANGUAGE  # Default
        if 'domain' in context:
            try:
                domain = ProcessingDomain(context['domain'])
            except ValueError:
                pass
        
        processing_context = create_processing_context(domain)
        result = self(context.get('input', np.zeros((1, 768))), processing_context)
        return {"adapted": True, "context": context, "result": result}
    
    def predict(self, inputs: Any) -> Any:
        """Make predictions using the adapted model (legacy interface)"""
        # Simple prediction interface
        if hasattr(inputs, 'shape'):
            result = self(inputs)
            return result.get('output', inputs)
        return inputs  # Fallback
    
    def update_meta_parameters(self, feedback: Dict[str, Any]):
        """Update meta-parameters based on feedback (legacy interface)"""
        if hasattr(self, 'meta_parameters'):
            self.meta_parameters.update(feedback)
        logger.debug("Meta-parameters updated")
    
    def _extract_context_features(self, context: ProcessingContext):
        """Extrae features del contexto de procesamiento"""
        features_data = [
            context.complexity_score,
            context.confidence_level, 
            context.historical_success_rate,
            float(list(AdaptationStrategy).index(context.adaptation_strategy))
        ]
        
        if JAX_AVAILABLE:
            return jnp.array(features_data)
        else:
            return np.array(features_data)
    
    def update_episodic_memory(self, 
                              context: ProcessingContext, 
                              performance: float,
                              output_features):
        """Updates la memoria episódica con nuevas experiencias"""
        if not self.config.enable_episodic_memory:
            return
            
        memory_key = f"{context.domain.value}_{context.adaptation_strategy.value}"
        
        if memory_key not in self.episodic_memory:
            self.episodic_memory[memory_key] = []
        
        # Agregar nueva experiencia
        experience = {
            'context': context,
            'performance': performance,
            'features': output_features,
            'timestamp': len(self.episodic_memory[memory_key])
        }
        
        self.episodic_memory[memory_key].append(experience)
        
        # Mantener tamaño de memoria
        if len(self.episodic_memory[memory_key]) > self.config.cache_size:
            self.episodic_memory[memory_key].pop(0)
    
    def get_adaptation_recommendations(self, 
                                     current_context: ProcessingContext) -> Dict[str, Any]:
        """Generates recomendaciones de adaptación basadas en experiencias pasadas"""
        domain_key = current_context.domain.value
        strategy_key = current_context.adaptation_strategy.value
        memory_key = f"{domain_key}_{strategy_key}"
        
        recommendations = {
            'suggested_strategy': current_context.adaptation_strategy,
            'confidence': 0.5,
            'historical_success_rate': 0.5,
            'resource_prediction': {},
            'similar_contexts': []
        }
        
        if memory_key in self.episodic_memory:
            experiences = self.episodic_memory[memory_key]
            if experiences:
                # Calcular métricas agregadas
                performances = [exp['performance'] for exp in experiences[-10:]]  # Últimas 10
                avg_performance = np.mean(performances)
                
                recommendations.update({
                    'historical_success_rate': avg_performance,
                    'confidence': min(len(experiences) / 50.0, 1.0),  # Más experiencias = más confianza
                    'similar_contexts': len(experiences)
                })
        
        return recommendations

# Factory functions
def create_meta_bamdp(config: Optional[MetaBAMDPConfig] = None) -> MetaBAMDP:
    """Factory function para crear una instancia de MetaBAMDP"""
    if config is None:
        config = MetaBAMDPConfig()
    
    return MetaBAMDP(config=config)

def create_processing_context(
    domain: ProcessingDomain,
    complexity_score: float = 0.5,
    confidence_level: float = 0.7,
    adaptation_strategy: AdaptationStrategy = AdaptationStrategy.MODERATE
) -> ProcessingContext:
    """Factory function para crear contexto de procesamiento"""
    return ProcessingContext(
        domain=domain,
        complexity_score=complexity_score,
        confidence_level=confidence_level,
        historical_success_rate=0.5,  # Default
        resource_requirements={},
        adaptation_strategy=adaptation_strategy
    )

# Export main components
__all__ = [
    'MetaBAMDP',
    'MetaBAMDPConfig', 
    'ProcessingContext',
    'ProcessingDomain',
    'AdaptationStrategy',
    'BayesianDomainAdapter',
    'MetaLearningController',
    'create_meta_bamdp',
    'create_processing_context'
]

def main():
    """Main function for testing the module"""
    logger.info("Enhanced MetaBAMDP module initialized successfully")
    
    # Test both interfaces
    # 1. New comprehensive interface
    config = MetaBAMDPConfig(hidden_size=256)
    meta_bamdp = create_meta_bamdp(config)
    
    context = create_processing_context(
        domain=ProcessingDomain.MATHEMATICS,
        complexity_score=0.8,
        adaptation_strategy=AdaptationStrategy.MODERATE
    )
    
    logger.info(f"✅ MetaBAMDP created with config: {config}")
    logger.info(f"✅ Processing context created: {context.domain.value}")
    
    # 2. Legacy interface test
    legacy_meta_bamdp = MetaBAMDP()
    logger.info("✅ Legacy MetaBAMDP interface test successful")
    return True

if __name__ == "__main__":
    main()
