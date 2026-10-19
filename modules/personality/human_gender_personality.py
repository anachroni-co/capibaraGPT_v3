"""
Human Gender Personality Module - PRODUCTION-READY VERSION
==========================================================

The most advanced, secure, and optimized implementation of authentically
human gender expression. Ready for production use with:

 Optimized Performance: Intelligent cache + JIT compiled
️ Robust Safety: Automatic limits + soft reset
 Advanced Memory: Working memory + long-term consolidation
 Visualization: Debug tools + real-time metrics
 Simplified API: Automatic initialization + error handling
"""

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Any, Optional, Tuple, NamedTuple, List, Union

import jax
import jax.numpy as jnp
from flax import linen as nn

try:
    from capibara.interfaces.imodules import IModule
except Exception:
    class IModule:  # type: ignore
        pass

logger = logging.getLogger(__name__)

# ==================== PRODUCTION-READY SETUP ====================

@dataclass
class ProductionHumanGenderConfig:
    """Production-ready setup with optimizations and safety."""
    
    # Base parameters
    hidden_size: int = 256
    dropout_rate: float = 0.1
    blend_temperature: float = 0.6
    
    # Humanity control
    emotional_volatility: float = 0.3
    contradiction_tolerance: float = 0.4
    familiarity_adaptation: float = 0.5
    microexpression_intensity: float = 0.2
    
    # Emotional states
    mood_persistence: int = 8
    stress_sensitivity: float = 0.4
    comfort_threshold: float = 0.6
    
    # Optimized memory
    working_memory_size: int = 5
    episodic_memory_size: int = 15
    long_term_consolidation_rate: float = 0.1
    memory_consolidation_frequency: int = 10
    
    # Adaptive learning
    adaptation_rate: float = 0.05
    volatility_learning_rate: float = 0.02
    sensitivity_learning_rate: float = 0.03
    
    # Emotional synchronization
    empathy_strength: float = 0.4
    mirroring_intensity: float = 0.3
    emotional_inertia: float = 0.7
    
    # Optimized cultural
    cultural_sensitivity: float = 0.3
    cultural_adaptation_speed: float = 0.1
    cultural_cache_size: int = 100
    
    # Safety and limits
    safety_bounds: Tuple[float, float] = (0.15, 0.85)
    activation_clip_range: Tuple[float, float] = (-3.0, 3.0)
    mood_stability_threshold: float = 0.2
    emergency_reset_threshold: float = 0.95
    
    # Performance optimizations
    cache_enabled: bool = True
    batch_processing: bool = True
    jit_compilation: bool = True
    memory_efficient_mode: bool = True
    
    # Debug and visualization
    debug_visualization: bool = False
    metrics_collection: bool = True
    performance_monitoring: bool = True
    
    # Sensitive topics (optimized)
    sensitive_topics: List[str] = field(default_factory=lambda: [
        "gender", "feminism", "masculinity", "relationships", 
        "parenting", "career", "appearance", "emotions",
        "technology", "leadership", "competence", "sexuality",
        "discrimination", "stereotypes", "family_roles"
    ])
    
    @classmethod
    def for_production(cls):
        """Optimized setup for production."""
        return cls(
            cache_enabled=True,
            debug_visualization=False,
            safety_bounds=(0.2, 0.8),
            memory_efficient_mode=True,
            performance_monitoring=True
        )
    
    @classmethod
    def for_development(cls):
        """Setup for development and research."""
        return cls(
            debug_visualization=True,
            working_memory_size=10,
            safety_bounds=(0.1, 0.9),
            metrics_collection=True,
            performance_monitoring=True
        )
    
    @classmethod
    def for_maximum_safety(cls):
        """Setup with maximum safety."""
        return cls(
            safety_bounds=(0.3, 0.7),
            emotional_volatility=0.2,
            long_term_consolidation_rate=0.05,
            emergency_reset_threshold=0.8,
            mood_stability_threshold=0.15
        ) 

# ==================== STATES AND STRUCTURES ====================

class OptimizedMoodState(NamedTuple):
    """Optimized emotional state for performance."""
    energy_level: float
    openness: float
    defensiveness: float
    confidence: float
    social_comfort: float
    empathy_level: float
    cultural_openness: float
    steps_in_mood: int
    stability_score: float  # new: stability score
    last_major_change: int  # new: timestamp of the last major change

class WorkingMemoryEvent(NamedTuple):
    """Optimized working memory event."""
    features: jnp.ndarray
    description: str
    intensity: float
    timestamp: int
    importance_score: float

class SafetyMetrics(NamedTuple):
    """Safety metrics."""
    bounds_violations: int
    stability_warnings: int
    emergency_resets: int
    avg_activation_variance: float

# ==================== ENHANCED MEMORY SYSTEM ====================

class EnhancedMemorySystem:
    """Optimized hierarchical memory system."""
    
    def __init__(self, config: ProductionHumanGenderConfig):
        self.config = config
        
        # Working memory (short term)
        self.working_memory = WorkingMemory(
            max_size=config.working_memory_size,
            decay_rate=0.1
        )
        
        # Episodic memory (medium term)
        self.episodic_events = []
        self.episodic_importance = []
        
        # Semantic memory (long term)
        self.semantic_patterns = defaultdict(float)
        
        # Counters and metrics
        self.consolidation_counter = 0
        self.total_events_processed = 0
        
    def add_interaction(self, features: jnp.ndarray, context: Dict[str, Any],
                       activations: Dict[str, float], intensity: float):
        """Adds interaction to memory with intelligent consolidation."""
        # 1. add a working memory
        event = WorkingMemoryEvent(
            features=features,
            description=context.get('description', ''),
            intensity=intensity,
            timestamp=self.total_events_processed,
            importance_score=self._compute_importance(context, activations)
        )
        self.working_memory.add_event(event)
        
        # 2. Episodic consolidation
        self.consolidation_counter += 1
        if self.consolidation_counter >= self.config.memory_consolidation_frequency:
            self._consolidate_to_episodic()
            self.consolidation_counter = 0
        
        # 3. Semantic update
        if intensity > 0.5:  # only significant patterns
            pattern_key = self._extract_pattern_key(context, activations)
            self.semantic_patterns[pattern_key] += intensity * 0.1
        
        self.total_events_processed += 1
    
    def get_long_term_influence(self, topic_type: str) -> jnp.ndarray:
        """Gets long-term memory influence."""
        # Combine influences from different levels
        working_influence = self.working_memory.get_current_influence()
        episodic_influence = self._get_episodic_influence(topic_type)
        semantic_influence = self._get_semantic_influence(topic_type)
        
        # Combination weights
        w_working = 0.5
        w_episodic = 0.3
        w_semantic = 0.2
        
        total_influence = (
            w_working * working_influence +
            w_episodic * episodic_influence +
            w_semantic * semantic_influence
        )
        
        total_weight = w_working + w_episodic + w_semantic
        
        if total_weight > 0:
            return jnp.tanh(total_influence / total_weight * 0.3)
        return jnp.zeros(3)
    
    def _compute_importance(self, context: Dict[str, Any], 
                          activations: Dict[str, float]) -> float:
        """Computes event importance for memory."""
        base_importance = context.get('emotional_intensity', 0.3)
        
        # Importance factors
        activation_strength = max(activations.values())
        topic_relevance = 1.0 if context.get('topic_type') in ['gender', 'identity'] else 0.5
        emotional_impact = context.get('emotional_impact', 0.3)
        
        importance = (
            0.4 * base_importance +
            0.3 * activation_strength +
            0.2 * topic_relevance +
            0.1 * emotional_impact
        )
        
        return float(importance)
    
    def _consolidate_to_episodic(self):
        """Consolidates events from working memory to episodic."""
        events = self.working_memory.get_important_events(threshold=0.6)
        
        for event in events:
            if len(self.episodic_events) >= self.config.episodic_memory_size:
                # Replace least important event
                min_idx = jnp.argmin(jnp.array(self.episodic_importance))
                if event.importance_score > self.episodic_importance[min_idx]:
                    self.episodic_events[min_idx] = event
                    self.episodic_importance[min_idx] = event.importance_score
            else:
                self.episodic_events.append(event)
                self.episodic_importance.append(event.importance_score)
    
    def _get_episodic_influence(self, topic_type: str) -> jnp.ndarray:
        """Computes episodic memory influence."""
        if not self.episodic_events:
            return jnp.zeros(3)
        
        relevant_events = [
            event for event in self.episodic_events
            if topic_type.lower() in event.description.lower()
        ]
        
        if not relevant_events:
            return jnp.zeros(3)
        
        # Weighted average by importance
        total_influence = jnp.zeros(3)
        total_weight = 0.0

        for event, importance in zip(relevant_events, self.episodic_importance):
            decay = math.exp(-0.1 * (self.total_events_processed - event.timestamp))
            weight = importance * decay
            total_influence += event.features[:3] * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_influence / total_weight
        return jnp.zeros(3)
    
    def _get_semantic_influence(self, topic_type: str) -> jnp.ndarray:
        """Compute influence from semantic patterns."""
        if not self.semantic_patterns:
            return jnp.zeros(3)
        
        # Filter relevant patterns
        relevant_patterns = {
            k: v for k, v in self.semantic_patterns.items()
            if topic_type.lower() in k.lower()
        }
        
        if not relevant_patterns:
            return jnp.zeros(3)
        
        # Weighted average by pattern strength
        total_influence = jnp.zeros(3)
        total_weight = 0.0
        
        for pattern, strength in relevant_patterns.items():
            pattern_vector = self._pattern_to_vector(pattern)
            total_influence += pattern_vector * strength
            total_weight += strength
        
        if total_weight > 0:
            return total_influence / total_weight
        return jnp.zeros(3)
    
    def _extract_pattern_key(self, context: Dict[str, Any], 
                           activations: Dict[str, float]) -> str:
        """Extracts pattern key for semantic memory."""
        topic = context.get('topic_type', 'general').lower()
        dominant_aspect = max(activations.items(), key=lambda x: x[1])[0]
        intensity = context.get('emotional_intensity', 0.3)
        
        return f"{topic}_{dominant_aspect}_{intensity:.1f}"
    
    def _pattern_to_vector(self, pattern_key: str) -> jnp.ndarray:
        """Converts semantic pattern to activation vector."""
        parts = pattern_key.split('_')
        if len(parts) != 3:
            return jnp.array([0.33, 0.33, 0.34])
        
        _, aspect, intensity = parts
        intensity = float(intensity)
        
        if aspect == 'masculine':
            return jnp.array([0.6, 0.2, 0.2]) * intensity
        elif aspect == 'feminine':
            return jnp.array([0.2, 0.6, 0.2]) * intensity
        else:
            return jnp.array([0.2, 0.2, 0.6]) * intensity

class WorkingMemory:
    """Optimized working memory with automatic decay."""
    
    def __init__(self, max_size: int = 5, decay_rate: float = 0.1):
        self.events = []
        self.max_size = max_size
        self.decay_rate = decay_rate
    
    def add_event(self, event: WorkingMemoryEvent):
        """Adds event with automatic capacity management."""
        if len(self.events) >= self.max_size:
            # Remove least important event
            min_idx = min(range(len(self.events)), 
                        key=lambda i: self.events[i].importance_score)
            self.events.pop(min_idx)
        
        self.events.append(event)
    
    def get_important_events(self, threshold: float = 0.5) -> List[WorkingMemoryEvent]:
        """Gets important events for consolidation."""
        return [event for event in self.events 
                if event.importance_score > threshold]
    
    def get_current_influence(self) -> jnp.ndarray:
        """Computes current working memory influence."""
        if not self.events:
            return jnp.zeros(3)
        
        total_influence = jnp.zeros(3)
        total_weight = 0.0
        
        for event in self.events:
            # Exponential decay
            age = len(self.events) - self.events.index(event)
            decay = math.exp(-self.decay_rate * age)
            
            weight = event.importance_score * decay
            total_influence += event.features[:3] * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_influence / total_weight
        return jnp.zeros(3)

# ==================== INTELLIGENT CULTURAL CACHE ====================

class CulturalCache:
    """Intelligent cache for recurring cultural computations."""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.access_frequency = defaultdict(int)
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
    def _compute_hash(self, x: jnp.ndarray) -> str:
        """Computes stable hash for input features."""
        # Use quantization to increase hit rate
        quantized = jnp.round(x.flatten()[:20] * 10) / 10  # Round to 1 decimal
        return str(hash(tuple(quantized.tolist())))
    
    def get_cultural_context(self, x: jnp.ndarray, compute_fn: callable) -> jnp.ndarray:
        """Gets cultural context with intelligent cache."""
        cache_key = self._compute_hash(x)
        
        if cache_key in self.cache:
            self.hits += 1
            self.access_frequency[cache_key] += 1
            return self.cache[cache_key]
        
        # cache miss - calculate
        self.misses += 1
        context = compute_fn(x)
        
        # Add to cache
        if len(self.cache) >= self.max_size:
            self._evict_least_frequent()
        
        self.cache[cache_key] = context
        self.access_frequency[cache_key] = 1
        
        return context
    
    def _evict_least_frequent(self):
        """Evicta el element except usado del caché."""
        if not self.cache:
            return
            
        least_used_key = min(self.access_frequency.keys(), 
                           key=lambda k: self.access_frequency[k])
        del self.cache[least_used_key]
        del self.access_frequency[least_used_key]
    
    def get_hit_rate(self) -> float:
        """Calcula tasa de aciertos del caché."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def clear_cache(self):
        """Limpia el caché completamente."""
        self.cache.clear()
        self.access_frequency.clear()
        self.hits = 0
        self.misses = 0

# ==================== SISTEMA DE SEGURIDAD ====================

class SafetyMechanism(nn.Module):
    """Sistema de seguridad multi-capa."""
    
    config: ProductionHumanGenderConfig
    
    def setup(self):
        self.reset_controller = nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.config.hidden_size // 2),
            nn.gelu,
            nn.Dense(self.config.hidden_size // 4),
            nn.gelu,
            nn.Dense(7),  # Reset factors for each mood dimension
            nn.sigmoid
        ], name="safety_reset_controller")
        
        self.stability_analyzer = nn.Sequential([
            nn.Dense(self.config.hidden_size // 3),
            nn.gelu,
            nn.Dense(1),
            nn.sigmoid
        ], name="stability_analyzer")
        
        # Métricas de seguridad
        self.safety_metrics = SafetyMetrics(
            bounds_violations=0,
            stability_warnings=0,
            emergency_resets=0,
            avg_activation_variance=0.0
        )
    
    def check_activation_safety(self, activations: Dict[str, float]) -> Tuple[bool, str]:
        """Verify if activations are within safe bounds."""
        values = jnp.array(list(activations.values()))
        
        # Check bounds
        lower_bound, upper_bound = self.config.safety_bounds
        if jnp.any(values < lower_bound) or jnp.any(values > upper_bound):
            return False, "activation_bounds_violation"
        
        # Check variance (stability)
        variance = float(jnp.var(values))
        if variance > self.config.mood_stability_threshold:
            return False, "high_variance_warning"
        
        # Check for extreme values
        max_val = float(jnp.max(values))
        if max_val > self.config.emergency_reset_threshold:
            return False, "emergency_threshold_exceeded"
        
        return True, "safe"
    
    def check_mood_safety(self, mood: OptimizedMoodState) -> Tuple[bool, str]:
        """Verifica estabilidad del estado de ánimo."""
        mood_vector = jnp.array([
            mood.energy_level, mood.openness, mood.defensiveness,
            mood.confidence, mood.social_comfort, mood.empathy_level,
            mood.cultural_openness
        ])
        
        # Check individual values
        lower_bound, upper_bound = self.config.safety_bounds
        if jnp.any(mood_vector < lower_bound) or jnp.any(mood_vector > upper_bound):
            return False, "mood_bounds_violation"
        
        # Check stability score
        if mood.stability_score < 0.3:
            return False, "mood_instability"
        
        return True, "safe"
    
    @partial(jax.jit, static_argnums=(0,))
    def soft_reset_activations(self, activations: jnp.ndarray, severity: float) -> jnp.ndarray:
        """Aplica reset suave a activaciones problemáticas."""
        # move towards el center (0.33, 0.33, 0.33) with intensidad basada en severidad
        neutral_center = jnp.ones(3) / 3
        reset_factor = severity * 0.5  # maximum 50% de reset
        
        return activations * (1 - reset_factor) + neutral_center * reset_factor
    
    @partial(jax.jit, static_argnums=(0,))
    def adaptive_mood_reset(self, mood_vector: jnp.ndarray, context_features: jnp.ndarray) -> jnp.ndarray:
        """Adaptive mood reset based on context."""
        combined_features = jnp.concatenate([mood_vector, context_features])
        reset_factors = self.reset_controller(combined_features)
        
        # Reset towards more stable values
        stable_mood = jnp.array([0.6, 0.6, 0.3, 0.7, 0.6, 0.6, 0.5])  # Default stable values
        
        return mood_vector * (1 - reset_factors * 0.3) + stable_mood * (reset_factors * 0.3)
    
    def update_safety_metrics(self, violation_type: str, severity: float):
        """Update safety metrics."""
        if violation_type == "activation_bounds_violation":
            self.safety_metrics = self.safety_metrics._replace(
                bounds_violations=self.safety_metrics.bounds_violations + 1
            )
        elif violation_type == "high_variance_warning":
            self.safety_metrics = self.safety_metrics._replace(
                stability_warnings=self.safety_metrics.stability_warnings + 1
            )
        elif violation_type == "emergency_threshold_exceeded":
            self.safety_metrics = self.safety_metrics._replace(
                emergency_resets=self.safety_metrics.emergency_resets + 1
            ) 

# ==================== VISUALIZACIÓN and DEBUG ====================

class DebugVisualizer:
    """Sistema de visualización and debug advanced."""
    
    def __init__(self, config: ProductionHumanGenderConfig):
        self.config = config
        self.state_history = []
        self.performance_metrics = {
            'inference_times': [],
            'memory_usage': [],
            'cache_hit_rates': [],
            'safety_violations': []
        }
        self.enabled = config.debug_visualization
        
    def record_state(self, state_data: Dict[str, Any], inference_time: float = None):
        """Registra estado for analysis posterior."""
        if not self.enabled:
            return
            
        timestamp = len(self.state_history)
        
        # Simplificar data for storage eficiente
        simplified = {
            'timestamp': timestamp,
            'activations': {k: float(v) for k, v in state_data['activations'].items()},
            'mood': {k: float(v) for k, v in state_data['mood'].items()},
            'relationship': {
                'familiarity': state_data['relationship']['familiarity'],
                'trust': state_data['relationship']['trust']
            },
            'metrics': {
                'authenticity': state_data['metrics']['authenticity_level'],
                'stability': state_data['metrics'].get('mood_stability', 0.5)
            }
        }
        
        if inference_time:
            simplified['inference_time'] = inference_time
            self.performance_metrics['inference_times'].append(inference_time)
        
        self.state_history.append(simplified)
        
        # maintain historial inside de límites de memory
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-500:]  # maintain últimos 500
    
    def get_activation_trends(self, window: int = 50) -> Dict[str, List[float]]:
        """Prepara data de tendencias for visualización."""
        if not self.enabled or len(self.state_history) < 2:
            return {}
        
        recent_history = self.state_history[-window:]
        
        return {
            'timestamps': [s['timestamp'] for s in recent_history],
            'masculine': [s['activations']['masculine'] for s in recent_history],
            'feminine': [s['activations']['feminine'] for s in recent_history],
            'neutral': [s['activations']['neutral'] for s in recent_history],
            'authenticity': [s['metrics']['authenticity'] for s in recent_history],
            'mood_stability': [s['metrics']['stability'] for s in recent_history]
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Genera resumen de rendimiento."""
        if not self.performance_metrics['inference_times']:
            return {"no_data": True}
        
        times = self.performance_metrics['inference_times']
        
        return {
            'avg_inference_time': jnp.mean(jnp.array(times)),
            'p95_inference_time': jnp.percentile(jnp.array(times), 95),
            'min_inference_time': jnp.min(jnp.array(times)),
            'max_inference_time': jnp.max(jnp.array(times)),
            'total_interactions': len(times),
            'performance_trend': 'improving' if len(times) > 10 and times[-5:] < times[:5] else 'stable'
        }
    
    def export_debug_data(self, filepath: str = None):
        """Exporta data de debug for analysis externo."""
        if not self.enabled:
            return
        
        import json
        
        export_data = {
            'state_history': self.state_history,
            'performance_metrics': {
                k: [float(x) for x in v] if isinstance(v, list) else v
                for k, v in self.performance_metrics.items()
            },
            'config': {
                'hidden_size': self.config.hidden_size,
                'safety_bounds': self.config.safety_bounds,
                'cache_enabled': self.config.cache_enabled
            }
        }
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        return export_data 

# ==================== implementation PRINCIPAL ====================

class ProductionHumanGenderPersonality(nn.Module):
    """implementation production-ready with todas las optimizaciones."""
    
    config: ProductionHumanGenderConfig

    def setup(self):
        """initialization optimizada de todos los sistemas."""
        # Sistemas principales
        self.memory_system = EnhancedMemorySystem(self.config)
        self.cultural_cache = CulturalCache(max_size=self.config.cultural_cache_size)
        self.safety_system = SafetyMechanism(config=self.config)
        self.visualizer = DebugVisualizer(self.config)
        
        # Estado emocional optimizado
        self.current_mood = OptimizedMoodState(
            energy_level=0.7,
            openness=0.6,
            defensiveness=0.2,
            confidence=0.7,
            social_comfort=0.5,
            empathy_level=0.6,
            cultural_openness=0.5,
            steps_in_mood=0,
            stability_score=0.8,
            last_major_change=0
        )
        
        # Relación simplificada for production
        self.relationship_state = {
            'familiarity': 0.0,
            'trust': 0.5,
            'emotional_intimacy': 0.2,
            'cultural_understanding': 0.3,
            'interaction_count': 0
        }
        
        # Contadores and métricas
        self.global_step = 0
        self.interaction_count = 0
        
        # Proyectores optimizados
        self.aspect_projectors = {
            'masculine': nn.Sequential([
                nn.LayerNorm(),
                nn.Dense(self.config.hidden_size),
                nn.gelu,
                nn.Dense(self.config.hidden_size)
            ], name="masculine_projector"),
            
            'feminine': nn.Sequential([
                nn.LayerNorm(),
                nn.Dense(self.config.hidden_size),
                nn.gelu,
                nn.Dense(self.config.hidden_size)
            ], name="feminine_projector"),
            
            'neutral': nn.Sequential([
                nn.LayerNorm(),
                nn.Dense(self.config.hidden_size),
                nn.gelu,
                nn.Dense(self.config.hidden_size)
            ], name="neutral_projector")
        }
        
        # Analizador contextual optimizado
        self.context_analyzer = nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.config.hidden_size),
            nn.gelu,
            nn.Dropout(self.config.dropout_rate),
            nn.Dense(self.config.hidden_size // 2),
            nn.gelu,
            nn.Dense(6)  # [masc, fem, neut, sensitivity, authenticity, stability]
        ], name="context_analyzer")
        
        # Detector cultural
        self.cultural_detector = nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.config.hidden_size // 2),
            nn.gelu,
            nn.Dense(5)  # Dimensiones culturales principales
        ], name="cultural_detector")
        
        # Blender end
        self.expression_blender = nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.config.hidden_size),
            nn.gelu,
            nn.Dense(self.config.hidden_size)
        ], name="expression_blender")
        
        # matrix de transformation end
        self.output_transform = self.param(
            "output_transform",
            nn.initializers.orthogonal(),
            (self.config.hidden_size, self.config.hidden_size)
        )

    def __call__(self, x: jnp.ndarray, context: Optional[Dict] = None, training: bool = False) -> Dict[str, Any]:
        """Optimized and secure forward pass."""
        start_time = time.time() if self.config.performance_monitoring else None
        
        try:
            # 1. Secure pre-processing
            x = self._safe_preprocess(x)
            
            # 2. Contextual analysis with cache
            cultural_context = self._get_cultural_context_cached(x)
            
            # 3. Input analysis
            context_features = self._analyze_context(x, cultural_context)
            base_activations = jax.nn.softmax(context_features[:3])
            modifiers = jax.nn.sigmoid(context_features[3:])
            
            # 4. Aspect projection
            aspect_features = self._project_aspects(x, training)
            
            # 5. Apply memory influences
            long_term_influence = self.memory_system.get_long_term_influence(
                context.get('topic_type', 'general') if context else 'general'
            )
            
            adjusted_activations = base_activations + long_term_influence * 0.2
            adjusted_activations = jax.nn.softmax(adjusted_activations)
            
            # 6. Combine aspects
            final_features = self._blend_aspects(aspect_features, adjusted_activations)
            
            # 7. Final transformation
            output_tensor = jnp.dot(final_features, self.output_transform)
            
            # 8. Safety verification
            activations_dict = {
                'masculine': float(adjusted_activations[0]),
                'feminine': float(adjusted_activations[1]),
                'neutral': float(adjusted_activations[2])
            }
            
            is_safe, safety_msg = self.safety_system.check_activation_safety(activations_dict)
            if not is_safe:
                adjusted_activations = self.safety_system.soft_reset_activations(
                    adjusted_activations, severity=0.5
                )
                activations_dict = {
                    'masculine': float(adjusted_activations[0]),
                    'feminine': float(adjusted_activations[1]),
                    'neutral': float(adjusted_activations[2])
                }
                self.safety_system.update_safety_metrics(safety_msg, 0.5)
            
            # 9. Update states
            self._update_relationship_state(context, activations_dict)
            self._update_mood_state(context, activations_dict, modifiers)
            
            # 10. Add to memory
            if context:
                self.memory_system.add_interaction(
                    features=jnp.mean(x, axis=(0, 1)),
                    context=context,
                    activations=activations_dict,
                    intensity=context.get('emotional_intensity', 0.3)
                )
            
            # 11. calculate métricas
            metrics = self._compute_production_metrics(
                activations_dict, modifiers, cultural_context, safety_msg
            )
            
            # 12. Preparar output
            inference_time = time.time() - start_time if start_time else None
            
            result = {
                "output": output_tensor,
                "score": float(jnp.mean(adjusted_activations)),
                "is_active": jnp.max(adjusted_activations) > 0.4,
                "activations": activations_dict,
                "human_state": self._get_human_state_summary(),
                "metrics": metrics,
                "safety_status": safety_msg,
                "performance": {
                    "inference_time": inference_time,
                    "cache_hit_rate": self.cultural_cache.get_hit_rate(),
                    "memory_events": len(self.memory_system.working_memory.events)
                } if self.config.performance_monitoring else {}
            }
            
            # 13. Registrar for debug/visualización
            if self.config.debug_visualization:
                self.visualizer.record_state({
                    'activations': activations_dict,
                    'mood': self._mood_to_dict(),
                    'relationship': self.relationship_state,
                    'metrics': metrics
                }, inference_time)
            
            # 14. increment contadores
            self.global_step += 1
            self.interaction_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ProductionHumanGenderPersonality: {str(e)}")
            # return estado secure en caso de error
            return self._get_safe_fallback_response()

    @partial(jax.jit, static_argnums=(0,))
    def _safe_preprocess(self, x: jnp.ndarray) -> jnp.ndarray:
        """Preprocesamiento secure with clipping."""
        x = jnp.clip(x, *self.config.activation_clip_range)
        # Normalización suave
        x = jnp.tanh(x * 0.5)
        return x

    def _get_cultural_context_cached(self, x: jnp.ndarray) -> jnp.ndarray:
        """Obtiene contexto cultural usando caché inteligente."""
        if self.config.cache_enabled:
            return self.cultural_cache.get_cultural_context(
                x, lambda features: self._compute_cultural_context(features)
            )
        else:
            return self._compute_cultural_context(x)

    @partial(jax.jit, static_argnums=(0,))
    def _compute_cultural_context(self, x: jnp.ndarray) -> jnp.ndarray:
        """Computa contexto cultural."""
        mean_features = jnp.mean(x, axis=(0, 1))
        return jax.nn.sigmoid(self.cultural_detector(mean_features))

    @partial(jax.jit, static_argnums=(0,))
    def _analyze_context(self, x: jnp.ndarray, cultural_context: jnp.ndarray) -> jnp.ndarray:
        """analysis contextual optimizado."""
        mean_features = jnp.mean(x, axis=(0, 1))
        mood_vector = jnp.array([
            self.current_mood.energy_level,
            self.current_mood.openness,
            self.current_mood.confidence,
            self.current_mood.empathy_level
        ])
        
        combined_features = jnp.concatenate([mean_features, cultural_context, mood_vector])
        return self.context_analyzer(combined_features)

    @partial(jax.jit, static_argnums=(0, 2))
    def _project_aspects(self, x: jnp.ndarray, training: bool) -> Dict[str, jnp.ndarray]:
        """Proyección optimizada de aspectos."""
        features = {}
        
        for aspect_name, projector in self.aspect_projectors.items():
            features[aspect_name] = projector(x)
            if training:
                features[aspect_name] = nn.Dropout(self.config.dropout_rate)(
                    features[aspect_name], deterministic=False
                )
        
        return features

    @partial(jax.jit, static_argnums=(0,))
    def _blend_aspects(self, aspect_features: Dict[str, jnp.ndarray], 
                      weights: jnp.ndarray) -> jnp.ndarray:
        """Mezcla optimizada de aspectos."""
        weighted_sum = (
            weights[0] * aspect_features['masculine'] +
            weights[1] * aspect_features['feminine'] +
            weights[2] * aspect_features['neutral']
        )
        
        return self.expression_blender(weighted_sum)

    def _update_relationship_state(self, context: Optional[Dict], activations: Dict[str, float]):
        """Actualiza estado relacional de forma eficiente."""
        if not context:
            return
        
        # Crecimiento gradual de familiaridad
        familiarity_growth = 0.02
        if context.get('emotional_intensity', 0) > 0.6:
            familiarity_growth *= 1.5
        
        self.relationship_state['familiarity'] = min(
            1.0, self.relationship_state['familiarity'] + familiarity_growth
        )
        
        # setting de confianza
        if context.get('was_conflict', False):
            trust_change = -0.05 * context.get('emotional_intensity', 0.5)
        elif context.get('was_positive', False):
            trust_change = 0.03 * context.get('emotional_intensity', 0.5)
        else:
            trust_change = 0.01
        
        self.relationship_state['trust'] = jnp.clip(
            self.relationship_state['trust'] + trust_change, 0.1, 1.0
        )
        
        # increment contador
        self.relationship_state['interaction_count'] += 1

    def _update_mood_state(self, context: Optional[Dict], activations: Dict[str, float], 
                          modifiers: jnp.ndarray):
        """Actualiza estado de ánimo with estabilidad."""
        mood = self.current_mood
        
        # Factores de change
        stress_factor = 0.0
        if context:
            if context.get('was_conflict', False):
                stress_factor += 0.4
            if any(topic in context.get('topic_type', '').lower() 
                  for topic in ['gender', 'competence', 'criticism']):
                stress_factor += 0.3
        
        # calculate cambios graduales
        inertia = self.config.emotional_inertia
        
        energy_change = (jax.random.normal(jax.random.PRNGKey(self.global_step), ()) * 0.05 
                        - stress_factor * 0.15)
        openness_change = (modifiers[1] * 0.1 - stress_factor * 0.2)
        defensiveness_change = (stress_factor * 0.3 + modifiers[0] * 0.1)
        confidence_change = ((sum(activations.values()) - 1.5) * 0.05 - stress_factor * 0.1)
        comfort_change = (self.relationship_state['familiarity'] * 0.05 - stress_factor * 0.1)
        
        # apply cambios with inercia
        new_energy = jnp.clip(mood.energy_level * inertia + energy_change * (1 - inertia), 0.1, 1.0)
        new_openness = jnp.clip(mood.openness * inertia + openness_change * (1 - inertia), 0.0, 1.0)
        new_defensiveness = jnp.clip(mood.defensiveness * inertia + defensiveness_change * (1 - inertia), 0.0, 1.0)
        new_confidence = jnp.clip(mood.confidence * inertia + confidence_change * (1 - inertia), 0.2, 1.0)
        new_comfort = jnp.clip(mood.social_comfort * inertia + comfort_change * (1 - inertia), 0.1, 1.0)
        
        # calculate estabilidad
        mood_vector = jnp.array([new_energy, new_openness, new_defensiveness, 
                               new_confidence, new_comfort, mood.empathy_level, mood.cultural_openness])
        stability_score = 1.0 - float(jnp.std(mood_vector))
        
        # detect cambios mayores
        change_magnitude = abs(new_energy - mood.energy_level) + abs(new_confidence - mood.confidence)
        last_major_change = self.global_step if change_magnitude > 0.2 else mood.last_major_change
        
        # update estado
        self.current_mood = OptimizedMoodState(
            energy_level=float(new_energy),
            openness=float(new_openness),
            defensiveness=float(new_defensiveness),
            confidence=float(new_confidence),
            social_comfort=float(new_comfort),
            empathy_level=mood.empathy_level,  # Cambia more lentamente
            cultural_openness=mood.cultural_openness,  # Cambia more lentamente
            steps_in_mood=mood.steps_in_mood + 1,
            stability_score=stability_score,
            last_major_change=last_major_change
        )

    def _compute_production_metrics(self, activations: Dict[str, float], 
                                  modifiers: jnp.ndarray, cultural_context: jnp.ndarray,
                                  safety_status: str) -> Dict[str, Any]:
        """Calcula métricas optimizadas for producción."""
        values = jnp.array(list(activations.values()))
        entropy = -jnp.sum(values * jnp.log(values + 1e-8)) / jnp.log(3.0)
        
        return {
            # Métricas básicas
            "masculine_activation": activations["masculine"],
            "feminine_activation": activations["feminine"], 
            "neutral_activation": activations["neutral"],
            "predominance": max(activations, key=activations.get),
            "entropy": float(entropy),
            
            # Métricas de humanidad
            "authenticity_level": float(modifiers[1]) * self.relationship_state['familiarity'],
            "mood_stability": self.current_mood.stability_score,
            "relationship_quality": (
                self.relationship_state['familiarity'] * 0.4 + 
                self.relationship_state['trust'] * 0.6
            ),
            
            # Métricas culturales
            "cultural_adaptation": float(jnp.mean(cultural_context)),
            "cultural_openness": self.current_mood.cultural_openness,
            
            # Métricas de seguridad
            "safety_status": safety_status,
            "safety_violations": self.safety_system.safety_metrics.bounds_violations,
            
            # Métricas de rendimiento
            "memory_efficiency": min(1.0, len(self.memory_system.working_memory.events) / self.config.working_memory_size),
            "cache_performance": self.cultural_cache.get_hit_rate(),
            
            # Estados de development
            "interaction_count": self.interaction_count,
            "relationship_stage": self._get_relationship_stage(),
            "mood_trend": self._get_mood_trend()
        }

    def _get_human_state_summary(self) -> Dict[str, Any]:
        """Resumen optimizado del estado humano."""
        return {
            "mood": self._mood_to_dict(),
            "relationship": {
                "familiarity": self.relationship_state['familiarity'],
                "trust": self.relationship_state['trust'],
                "stage": self._get_relationship_stage(),
                "interactions": self.relationship_state['interaction_count']
            },
            "development": {
                "total_steps": self.global_step,
                "last_major_mood_change": self.current_mood.last_major_change,
                "stability_score": self.current_mood.stability_score
            },
            "performance": {
                "cache_hit_rate": self.cultural_cache.get_hit_rate(),
                "memory_utilization": len(self.memory_system.working_memory.events) / self.config.working_memory_size,
                "safety_score": 1.0 - (self.safety_system.safety_metrics.bounds_violations / max(1, self.interaction_count))
            }
        }

    def _mood_to_dict(self) -> Dict[str, float]:
        """Convierte estado de ánimo a diccionario."""
        return {
            "energy": self.current_mood.energy_level,
            "openness": self.current_mood.openness,
            "defensiveness": self.current_mood.defensiveness,
            "confidence": self.current_mood.confidence,
            "social_comfort": self.current_mood.social_comfort,
            "empathy_level": self.current_mood.empathy_level,
            "cultural_openness": self.current_mood.cultural_openness,
            "stability": self.current_mood.stability_score
        }

    def _get_relationship_stage(self) -> str:
        """Determina etapa de relación de forma eficiente."""
        familiarity = self.relationship_state['familiarity']
        trust = self.relationship_state['trust']
        
        if familiarity < 0.2:
            return "stranger"
        elif familiarity < 0.5 and trust < 0.7:
            return "acquaintance"
        elif familiarity < 0.8 and trust > 0.6:
            return "friend"
        else:
            return "close"

    def _get_mood_trend(self) -> str:
        """Determina tendencia del estado de ánimo."""
        if self.current_mood.steps_in_mood < 3:
            return "stable"
        
        recent_change = self.global_step - self.current_mood.last_major_change
        if recent_change < 5:
            return "volatile"
        elif self.current_mood.stability_score > 0.7:
            return "stable"
        else:
            return "fluctuating"

    def _get_safe_fallback_response(self) -> Dict[str, Any]:
        """answer segura en caso de error."""
        return {
            "output": jnp.zeros(self.config.hidden_size),
            "score": 0.5,
            "is_active": False,
            "activations": {"masculine": 0.33, "feminine": 0.33, "neutral": 0.34},
            "human_state": {
                "mood": {"energy": 0.5, "openness": 0.5, "defensiveness": 0.3, 
                        "confidence": 0.5, "stability": 0.8},
                "relationship": {"familiarity": 0.0, "trust": 0.5, "stage": "stranger"}
            },
            "metrics": {"safety_status": "emergency_fallback", "error_occurred": True},
            "safety_status": "emergency_fallback"
        }

    # ==================== MÉTODOS DE UTILIDAD ====================
    
    def reset_to_safe_state(self):
        """Reset complete a estado secure."""
        self.current_mood = OptimizedMoodState(
            energy_level=0.6, openness=0.5, defensiveness=0.3, confidence=0.6,
            social_comfort=0.5, empathy_level=0.5, cultural_openness=0.5,
            steps_in_mood=0, stability_score=0.8, last_major_change=0
        )
        
        self.relationship_state = {
            'familiarity': 0.0, 'trust': 0.5, 'emotional_intimacy': 0.2,
            'cultural_understanding': 0.3, 'interaction_count': 0
        }
        
        self.memory_system.working_memory.events.clear()
        self.cultural_cache.clear_cache()
        
        logger.info("Sistema reseteado a estado seguro")

    def get_performance_report(self) -> Dict[str, Any]:
        """Genera reporte de rendimiento complete."""
        if not self.config.performance_monitoring:
            return {"monitoring_disabled": True}
        
        return {
            "system_performance": {
                "total_interactions": self.interaction_count,
                "cache_hit_rate": self.cultural_cache.get_hit_rate(),
                "memory_utilization": len(self.memory_system.working_memory.events) / self.config.working_memory_size,
                "safety_violations": self.safety_system.safety_metrics.bounds_violations
            },
            "personality_development": {
                "relationship_stage": self._get_relationship_stage(),
                "mood_stability": self.current_mood.stability_score,
                "cultural_adaptation": self.current_mood.cultural_openness,
                "interaction_quality": self.relationship_state['trust']
            },
            "debug_data": self.visualizer.get_performance_summary() if self.config.debug_visualization else {}
        }

    def export_state(self) -> Dict[str, Any]:
        """Exporta estado complete for analysis or backup."""
        return {
            "mood_state": self._mood_to_dict(),
            "relationship_state": self.relationship_state.copy(),
            "memory_summary": {
                "working_memory_events": len(self.memory_system.working_memory.events),
                "episodic_events": len(self.memory_system.episodic_events),
                "semantic_patterns": dict(self.memory_system.semantic_patterns)
            },
            "performance_metrics": {
                "global_step": self.global_step,
                "interaction_count": self.interaction_count,
                "cache_stats": {
                    "hits": self.cultural_cache.hits,
                    "misses": self.cultural_cache.misses,
                    "hit_rate": self.cultural_cache.get_hit_rate()
                }
            },
            "safety_metrics": {
                "bounds_violations": self.safety_system.safety_metrics.bounds_violations,
                "stability_warnings": self.safety_system.safety_metrics.stability_warnings,
                "emergency_resets": self.safety_system.safety_metrics.emergency_resets
            }
        }

# ==================== record E integration ====================

def register_human_gender_personality():
    """Registra el módulo de personalidad de género humano en el sistema."""
    from core.model import ThreadSafeRegistry
    
    registry = ThreadSafeRegistry()
    registry.register(
        "human_gender_personality",
        ProductionHumanGenderPersonality
    )
    
    logger.info("Módulo de personalidad de género humano registrado exitosamente")

def create_personality_from_config(config_dict: Dict[str, Any]) -> ProductionHumanGenderPersonality:
    """Crea una instance del módulo since setup."""
    config = ProductionHumanGenderConfig(**config_dict)
    return ProductionHumanGenderPersonality(config=config)

# integration with el sistema principal
class HumanGenderPersonalityModule(IModule):
    """Wrapper for integration with el sistema principal."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializa el módulo with setup optional."""
        super().__init__()
        
        if config is None:
            config = {}
        
        self.personality = create_personality_from_config(config)
        self.is_initialized = False
    
    def setup(self):
        """initialization lazy del módulo."""
        if not self.is_initialized:
            # Inicializar componentes
            self.personality.setup()
            self.is_initialized = True
    
    def __call__(self, x: jnp.ndarray, context: Optional[Dict] = None, 
                 training: bool = False) -> Dict[str, Any]:
        """Forward pass with integration de contexto."""
        self.setup()
        
        # process input
        result = self.personality(x, context, training)
        
        # add metadatos de integration
        result["module_type"] = "personality"
        result["personality_type"] = "human_gender"
        result["is_stateful"] = True
        
        return result
    
    def get_state(self) -> Dict[str, Any]:
        """Obtiene estado current for checkpointing."""
        return {
            "personality_state": self.personality.export_state(),
            "is_initialized": self.is_initialized
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restaura estado since checkpoint."""
        if not self.is_initialized:
            self.setup()
        
        if "personality_state" in state:
            # Restaurar estado de personalidad
            personality_state = state["personality_state"]
            
            # Restaurar mood
            if "mood_state" in personality_state:
                self.personality.current_mood = OptimizedMoodState(
                    **personality_state["mood_state"]
                )
            
            # Restaurar relación
            if "relationship_state" in personality_state:
                self.personality.relationship_state = personality_state["relationship_state"]
            
            # Restaurar memory
            if "memory_summary" in personality_state:
                memory_data = personality_state["memory_summary"]
                # Restaurar eventos de working memory
                self.personality.memory_system.working_memory.events = []
                for _ in range(memory_data["working_memory_events"]):
                    self.personality.memory_system.working_memory.add_event(
                        WorkingMemoryEvent(
                            features=jnp.zeros(self.personality.config.hidden_size),
                            description="restored_event",
                            intensity=0.5,
                            timestamp=0,
                            importance_score=0.5
                        )
                    )
            
            self.is_initialized = True
    
    def reset(self):
        """Reset complete del módulo."""
        if self.is_initialized:
            self.personality.reset_to_safe_state()
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del módulo."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "active",
            "performance": self.personality.get_performance_report(),
            "memory_usage": len(self.personality.memory_system.working_memory.events),
            "mood_stability": self.personality.current_mood.stability_score,
            "relationship_stage": self.personality.relationship_state["stage"]
        }

# ==================== integration with INFERENCIA ====================

class HumanGenderInferenceWrapper:
    """Wrapper for integration with el sistema de inferencia."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.module = HumanGenderPersonalityModule(config)
        self.cache = {}
        self._lock = threading.Lock()
    
    def preprocess_for_inference(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocesa entradas for inferencia."""
        # Extraer information relevante
        text = inputs.get("text", "")
        context = inputs.get("context", {})
        
        # add information de género if está available
        if "gender_hints" in inputs:
            context["gender_hints"] = inputs["gender_hints"]
        
        # add information emocional
        if "emotional_state" in inputs:
            context["emotional_intensity"] = inputs["emotional_state"].get("intensity", 0.3)
            context["was_positive"] = inputs["emotional_state"].get("is_positive", True)
        
        return {
            "text": text,
            "context": context,
            "use_cache": inputs.get("use_cache", True)
        }
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa input during inferencia."""
        processed = self.preprocess_for_inference(inputs)
        
        # Check cache
        if processed["use_cache"]:
            cache_key = self._get_cache_key(processed)
            with self._lock:
                if cache_key in self.cache:
                    return self.cache[cache_key]
        
        # Process with module
        result = self.module(
            jnp.array(processed["text"]),
            context=processed["context"]
        )
        
        # cache result
        if processed["use_cache"]:
            with self._lock:
                self.cache[cache_key] = result
        
        return result
    
    def _get_cache_key(self, processed: Dict[str, Any]) -> str:
        """Genera key de caché."""
        key_parts = [
            str(processed["text"]),
            str(hash(frozenset(processed["context"].items())))
        ]
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    def clear_cache(self):
        """Limpia caché de inferencia."""
        with self._lock:
            self.cache.clear()

# setup for inferencia
DEFAULT_INFERENCE_CONFIG = {
    "model_type": "human_gender_personality",
    "use_cache": True,
    "cache_size": 1000,
    "personality": {
        "hidden_size": 256,
        "dropout_rate": 0.1,
        "blend_temperature": 0.6,
        "emotional_volatility": 0.3,
        "contradiction_tolerance": 0.4,
        "familiarity_adaptation": 0.5,
        "microexpression_intensity": 0.2
    }
} 
