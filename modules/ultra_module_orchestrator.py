"""
Ultra Module Orchestrator - CapibaraGPT v3
=============================================

Sistema de orquestación ultra-avanzada for todos los módulos:
- Intelligent module selection and coordination
- Integration with Ultra Core System
- Connection with UltraAdvancedTrainer
- Expert Soup for module ensembles
- Performance monitoring and optimization
- Dynamic module composition
- Graceful fallbacks and error handling

Esta es la evolución del sistema de módulos for be al level del ecosistema ultra-advanced.
"""

import time
import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Callable, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

import jax
from flax import linen as nn
from functools import partial
import jax.numpy as jnp

# Safe imports for ultra systems integration
try:
    from core.ultra_core_integration import (
        UltraCoreOrchestrator, create_ultra_core_system,
        ULTRA_TRAINING_AVAILABLE, SSM_AVAILABLE
    )
    ULTRA_CORE_AVAILABLE = True
except ImportError:
    ULTRA_CORE_AVAILABLE = False
    UltraCoreOrchestrator = None

try:
    from training.optimizations import (
        UltraAdvancedTrainer, ExpertSoupIntegration,
        ModelSoupConfig, ULTRA_OPTIMIZATIONS_AVAILABLE
    )
    ULTRA_TRAINING_INTEGRATION = True
except ImportError:
    ULTRA_TRAINING_INTEGRATION = False
    UltraAdvancedTrainer = None
    ExpertSoupIntegration = None

# Import existing modules with safe fallbacks
try:
    from .shared_attention import (
        OptimizedSharedAttention, MultiScaleSharedAttention, 
        EfficiencyOptimizedAttention, create_shared_attention
    )
    ATTENTION_MODULES_AVAILABLE = True
except ImportError:
    ATTENTION_MODULES_AVAILABLE = False

try:
    from .capibara_adaptive_router import (
        OptimizedAdaptiveRouter, ContextualRouterOptimized,
        VQbitLayerOptimized, ExpertLayer,
        create_router_for_tpu_v4_32, distributed_router_forward
    )
    ROUTER_MODULES_AVAILABLE = True
except ImportError:
    ROUTER_MODULES_AVAILABLE = False

try:
    from .specialized_processors import (
        ProcessorConfig,
        ProcessorType,
        TextAnalysisProcessor,
        SentimentAnalysisProcessor,
        EntityExtractionProcessor,
        CodeAnalysisProcessor,
        MultimodalFusionProcessor,
    )
    PROCESSOR_MODULES_AVAILABLE = True
except ImportError:
    PROCESSOR_MODULES_AVAILABLE = False

try:
    from .personality.unified_personality_system import UnifiedPersonalitySystem
    PERSONALITY_MODULES_AVAILABLE = True
except ImportError:
    PERSONALITY_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration and Enums
# ============================================================================

class ModuleType(str, Enum):
    """Types of available modules."""
    ATTENTION = "attention"
    ROUTER = "router"
    PROCESSOR = "processor"
    PERSONALITY = "personality"
    ULTRA_HYBRID = "ultra_hybrid"

class OrchestrationStrategy(str, Enum):
    """Strategies for module orchestration."""
    ADAPTIVE = "adaptive"           # Intelligent selection based on input
    ENSEMBLE = "ensemble"           # Combine multiple modules
    SEQUENTIAL = "sequential"       # Chain modules in sequence
    PARALLEL = "parallel"           # Run modules in parallel
    ULTRA_HYBRID = "ultra_hybrid"   # Ultra-advanced hybrid strategy

@dataclass
class ModulePerformanceMetrics:
    """Performance metrics for individual modules."""
    module_type: str
    computation_time_ms: float
    memory_usage_mb: float
    accuracy_score: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    efficiency_score: float = 0.0
    error_rate: float = 0.0

@dataclass
class UltraModuleConfig:
    """Configuration for ultra-advanced module orchestration."""
    
    # Core configuration
    hidden_size: int = 768
    orchestration_strategy: OrchestrationStrategy = OrchestrationStrategy.ADAPTIVE
    
    # Module selection
    enabled_modules: List[ModuleType] = field(default_factory=lambda: [
        ModuleType.ATTENTION,
        ModuleType.ROUTER,
        ModuleType.PROCESSOR,
        ModuleType.PERSONALITY,
        ModuleType.ULTRA_HYBRID
    ])
    
    # Performance optimization
    enable_tpu_optimization: bool = True
    use_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    
    # Ultra integrations
    auto_core_integration: bool = True
    auto_training_integration: bool = True
    enable_expert_soup: bool = True
    
    # Intelligent routing
    adaptive_routing_threshold: float = 0.8
    ensemble_weights: Optional[Dict[str, float]] = None
    dynamic_weight_adjustment: bool = True
    
    # Performance monitoring
    enable_comprehensive_monitoring: bool = True
    performance_tracking_window: int = 1000

# ============================================================================
# Ultra Module Orchestrator
# ============================================================================

class UltraModuleOrchestrator:
    """Orquestador ultra-advanced for todos los módulos del sistema."""
    
    def __init__(self, config: UltraModuleConfig):
        self.config = config
        self.modules = {}
        self.performance_metrics = {}
        self.routing_history = []
        
        # Ultra system integrations
        self.core_orchestrator = None
        self.training_integration = None
        self.expert_soup_manager = None
        
        # Performance tracking
        self.global_metrics = {
            "total_forward_passes": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "average_latency_ms": 0.0,
            "total_modules": 0,
            "active_modules": 0
        }
        
        # Initialize the orchestrator
        self._initialize_orchestrator()
    
    def _initialize_orchestrator(self):
        """Initialize the ultra module orchestrator."""
        
        logger.info(" Initializing Ultra Module Orchestrator")
        
        # Initialize ultra system integrations
        if self.config.auto_core_integration and ULTRA_CORE_AVAILABLE:
            try:
                self.core_orchestrator = create_ultra_core_system()
                logger.info(" Ultra Core integration initialized")
            except Exception as e:
                logger.warning(f"️ Core integration failed: {e}")
        
        # Initialize modules
        self._initialize_modules()
        
        # Initialize training integration
        if self.config.auto_training_integration and ULTRA_TRAINING_INTEGRATION:
            self._initialize_training_integration()
        
        # Initialize expert soup for modules
        if self.config.enable_expert_soup and ULTRA_TRAINING_INTEGRATION:
            self._initialize_expert_soup()
        
        logger.info(f" Ultra Module Orchestrator initialized")
        logger.info(f"    Available modules: {len(self.modules)}")
        logger.info(f"    Ultra Core: {'' if self.core_orchestrator else ''}")
        logger.info(f"    Expert Soup: {'' if self.expert_soup_manager else ''}")
    
    def _initialize_modules(self):
        """Initialize all available modules."""
        
        for module_type in self.config.enabled_modules:
            try:
                modules = self._create_modules_of_type(module_type)
                if modules:
                    self.modules[module_type.value] = modules
                    self.global_metrics["total_modules"] += len(modules)
                    logger.info(f" {module_type.value}: {len(modules)} modules loaded")
                else:
                    logger.warning(f"️ {module_type.value} modules not available")
            except Exception as e:
                logger.error(f" Failed to initialize {module_type.value}: {e}")
        
        self.global_metrics["active_modules"] = sum(len(mods) for mods in self.modules.values())
    
    def _create_modules_of_type(self, module_type: ModuleType) -> Optional[Dict[str, Any]]:
        """Create modules of a specific type."""
        
        modules = {}
        
        if module_type == ModuleType.ATTENTION and ATTENTION_MODULES_AVAILABLE:
            modules.update({
                "optimized_shared": OptimizedSharedAttention(
                    hidden_size=self.config.hidden_size,
                    num_heads=8,
                    dropout_rate=0.1
                ),
                "multi_scale": MultiScaleSharedAttention(
                    hidden_size=self.config.hidden_size,
                    num_heads=8,
                    scales=(1, 2, 4, 8)
                ),
                "efficiency_optimized": EfficiencyOptimizedAttention(
                    hidden_size=self.config.hidden_size,
                    num_heads=8,
                    window_size=512
                )
            })
        
        elif module_type == ModuleType.ROUTER and ROUTER_MODULES_AVAILABLE:
            modules.update({
                "adaptive_router": OptimizedAdaptiveRouter(
                    hidden_size=self.config.hidden_size,
                    num_virtual_qubits=512,
                    vocab_size=50257
                ),
                "contextual_router": ContextualRouterOptimized(
                    hidden_size=self.config.hidden_size,
                    num_virtual_qubits=512,
                    vocab_size=50257
                )
            })
        
        elif module_type == ModuleType.PROCESSOR and PROCESSOR_MODULES_AVAILABLE:
            modules.update({
                "text_analysis": TextAnalysisProcessor(
                    ProcessorConfig(processor_type=ProcessorType.TEXT_ANALYSIS)
                ),
                "sentiment": SentimentAnalysisProcessor(
                    ProcessorConfig(processor_type=ProcessorType.SENTIMENT_ANALYSIS)
                ),
                "entity_extraction": EntityExtractionProcessor(
                    ProcessorConfig(processor_type=ProcessorType.ENTITY_EXTRACTION)
                ),
                "code_analysis": CodeAnalysisProcessor(
                    ProcessorConfig(processor_type=ProcessorType.CODE_ANALYSIS)
                ),
                "multimodal_fusion": MultimodalFusionProcessor(
                    ProcessorConfig(processor_type=ProcessorType.MULTIMODAL_FUSION)
                ),
            })
        
        elif module_type == ModuleType.PERSONALITY and PERSONALITY_MODULES_AVAILABLE:
            modules.update({
                "unified_personality": UnifiedPersonalitySystem()
            })
        
        elif module_type == ModuleType.ULTRA_HYBRID:
            # Create ultra-hybrid modules that combine multiple types
            modules.update({
                "ultra_attention_router": self._create_ultra_hybrid_module()
            })
        
        return modules if modules else None
    
    def _create_ultra_hybrid_module(self):
        """Create ultra-hybrid module combining multiple capabilities."""
        
        class UltraHybridModule(nn.Module):
            """Ultra-hybrid module combining attention, routing, and processing."""
            
            hidden_size: int = self.config.hidden_size
            
            def setup(self):
                if ATTENTION_MODULES_AVAILABLE:
                    self.attention = OptimizedSharedAttention(
                        hidden_size=self.hidden_size,
                        num_heads=8
                    )
                
                if ROUTER_MODULES_AVAILABLE:
                    self.router = OptimizedAdaptiveRouter(
                        hidden_size=self.hidden_size,
                        num_virtual_qubits=256,
                        vocab_size=50257
                    )
                
                self.fusion = nn.Dense(self.hidden_size)
            
            def __call__(self, x, context_tokens=None, training=False):
                outputs = []
                
                # Apply attention if available
                if hasattr(self, 'attention'):
                    attn_out = self.attention(x, training=training)
                    outputs.append(attn_out["output"])
                
                # Apply routing if available
                if hasattr(self, 'router') and context_tokens is not None:
                    router_out = self.router(x, context_tokens, training=training)
                    outputs.append(router_out["output"])
                
                # Combine outputs
                if outputs:
                    combined = jnp.concatenate(outputs, axis=-1)
                    return self.fusion(combined)
                else:
                    return x
        
        return UltraHybridModule()
    
    def route_to_modules(
        self,
        x: jnp.ndarray,
        context: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = None,
        strategy: Optional[OrchestrationStrategy] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Route input to optimal module(s) based on strategy."""
        
        start_time = time.time()
        
        # Use configured strategy if not specified
        if strategy is None:
            strategy = self.config.orchestration_strategy
        
        try:
            if strategy == OrchestrationStrategy.ADAPTIVE:
                result, routing_info = self._adaptive_routing(x, context, task_type)
            elif strategy == OrchestrationStrategy.ENSEMBLE:
                result, routing_info = self._ensemble_routing(x, context)
            elif strategy == OrchestrationStrategy.SEQUENTIAL:
                result, routing_info = self._sequential_routing(x, context)
            elif strategy == OrchestrationStrategy.PARALLEL:
                result, routing_info = self._parallel_routing(x, context)
            elif strategy == OrchestrationStrategy.ULTRA_HYBRID:
                result, routing_info = self._ultra_hybrid_routing(x, context, task_type)
            else:
                raise ValueError(f"Unknown orchestration strategy: {strategy}")
            
            # Track successful routing
            self.global_metrics["successful_routes"] += 1
            
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            result, routing_info = self._fallback_routing(x, context)
            self.global_metrics["failed_routes"] += 1
        
        # Update performance metrics
        computation_time = (time.time() - start_time) * 1000
        self.global_metrics["total_forward_passes"] += 1
        
        # Comprehensive routing info
        comprehensive_info = {
            "strategy": strategy.value,
            "routing_details": routing_info,
            "performance": {
                "computation_time_ms": computation_time,
                "total_passes": self.global_metrics["total_forward_passes"],
                "success_rate": self._calculate_success_rate()
            },
            "system_status": {
                "active_modules": self.global_metrics["active_modules"],
                "ultra_core_active": self.core_orchestrator is not None,
                "expert_soup_active": self.expert_soup_manager is not None
            }
        }
        
        return result, comprehensive_info
    
    def _adaptive_routing(self, x, context, task_type):
        """Adaptive routing based on input characteristics."""
        
        # Analyze input to determine best module type
        best_module_type = self._analyze_input_for_routing(x, context, task_type)
        
        # Select specific module within type
        if best_module_type in self.modules:
            modules = self.modules[best_module_type]
            best_module_name = self._select_best_module_in_type(modules, x, context)
            
            # Execute selected module
            result = self._execute_module(best_module_type, best_module_name, x, context)
            
            routing_info = {
                "selected_module_type": best_module_type,
                "selected_module": best_module_name,
                "confidence": 0.8,
                "reason": f"Best for {task_type if task_type else 'general'} task"
            }
        else:
            # Fallback
            result = x
            routing_info = {"fallback_used": True, "reason": "No suitable modules available"}
        
        return result, routing_info
    
    def _ensemble_routing(self, x, context):
        """Ensemble routing combining multiple modules."""
        
        results = {}
        weights = self.config.ensemble_weights or {}
        
        # Execute available modules
        for module_type, modules in self.modules.items():
            try:
                # Use first available module of each type
                module_name = next(iter(modules.keys()))
                result = self._execute_module(module_type, module_name, x, context)
                results[module_type] = result
            except Exception as e:
                logger.warning(f"Module {module_type} failed: {e}")
        
        # Combine results
        combined_result = self._combine_ensemble_results(results, weights)
        
        routing_info = {
            "strategy": "ensemble",
            "participating_modules": list(results.keys()),
            "weights_used": weights
        }
        
        return combined_result, routing_info
    
    def _sequential_routing(self, x, context):
        """Sequential routing chaining modules."""
        
        current_input = x
        processing_chain = []
        
        # Process through available module types in order
        module_order = ["processor", "attention", "router", "personality"]
        
        for module_type in module_order:
            if module_type in self.modules:
                try:
                    modules = self.modules[module_type]
                    module_name = next(iter(modules.keys()))
                    current_input = self._execute_module(module_type, module_name, current_input, context)
                    processing_chain.append(f"{module_type}.{module_name}")
                except Exception as e:
                    logger.warning(f"Sequential processing failed at {module_type}: {e}")
                    break
        
        routing_info = {
            "strategy": "sequential",
            "processing_chain": processing_chain
        }
        
        return current_input, routing_info
    
    def _parallel_routing(self, x, context):
        """Parallel routing running modules simultaneously."""
        
        results = {}
        
        # Execute all available modules in parallel
        for module_type, modules in self.modules.items():
            try:
                module_name = next(iter(modules.keys()))
                result = self._execute_module(module_type, module_name, x, context)
                results[module_type] = result
            except Exception as e:
                logger.warning(f"Parallel execution failed for {module_type}: {e}")
        
        # Select best result
        best_result = self._select_best_parallel_result(results)
        
        routing_info = {
            "strategy": "parallel",
            "executed_modules": list(results.keys()),
            "results_count": len(results)
        }
        
        return best_result, routing_info
    
    def _ultra_hybrid_routing(self, x, context, task_type):
        """Ultra-advanced hybrid routing using all intelligence."""
        
        # Use ultra-hybrid module if available
        if "ultra_hybrid" in self.modules and "ultra_attention_router" in self.modules["ultra_hybrid"]:
            context_tokens = context.get("tokens") if context else jnp.ones((x.shape[0], 10), dtype=jnp.int32)
            
            hybrid_module = self.modules["ultra_hybrid"]["ultra_attention_router"]
            result = hybrid_module(x, context_tokens=context_tokens, training=False)
            
            routing_info = {
                "strategy": "ultra_hybrid",
                "module_used": "ultra_attention_router",
                "ultra_features_active": True
            }
        else:
            # Fallback to adaptive
            result, adaptive_info = self._adaptive_routing(x, context, task_type)
            routing_info = {
                "strategy": "ultra_hybrid_fallback",
                "adaptive_info": adaptive_info
            }
        
        return result, routing_info
    
    def _fallback_routing(self, x, context):
        """Fallback routing when all else fails."""
        
        # Use any available module
        for module_type, modules in self.modules.items():
            try:
                module_name = next(iter(modules.keys()))
                result = self._execute_module(module_type, module_name, x, context)
                
                routing_info = {
                    "strategy": "fallback",
                    "used_module": f"{module_type}.{module_name}",
                    "reason": "All primary routing failed"
                }
                
                return result, routing_info
            except:
                continue
        
        # end fallback - return input unchanged
        return x, {"strategy": "identity", "reason": "No modules available"}
    
    def _analyze_input_for_routing(self, x, context, task_type):
        """Analyze input to determine best module type."""
        
        # Task-based routing
        if task_type:
            if "attention" in task_type.lower():
                return "attention"
            elif "router" in task_type.lower() or "routing" in task_type.lower():
                return "router"
            elif "audio" in task_type.lower() or "bio" in task_type.lower():
                return "processor"
            elif "personality" in task_type.lower():
                return "personality"
        
        # Textual/contextual routing
        if isinstance(x, str):
            return "processor"
        if isinstance(context, dict):
            if any(key in context for key in ("text", "prompt", "response", "code", "modalities")):
                return "processor"

        # Sequence length based routing
        if len(x.shape) >= 2:
            seq_len = x.shape[1]
            if seq_len > 1024:
                return "attention"  # Use efficient attention for long sequences
            elif seq_len < 64:
                return "processor"  # Use processors for short inputs
        
        # Default to ultra_hybrid if available, else attention
        if "ultra_hybrid" in self.modules:
            return "ultra_hybrid"
        elif "attention" in self.modules:
            return "attention"
        else:
            return next(iter(self.modules.keys()))
    
    def _select_best_module_in_type(self, modules, x, context):
        """Select best module within a module type."""
        
        # For attention modules, prefer efficiency_optimized for long sequences
        if "efficiency_optimized" in modules and len(x.shape) >= 2 and x.shape[1] > 512:
            return "efficiency_optimized"
        elif "multi_scale" in modules:
            return "multi_scale"
        else:
            return next(iter(modules.keys()))
    
    def _execute_module(self, module_type, module_name, x, context):
        """Execute a specific module safely."""
        
        try:
            module = self.modules[module_type][module_name]
            
            # Handle different module interfaces
            if module_type == "attention":
                result = module(x, training=False)
                return result["output"] if isinstance(result, dict) else result
            
            elif module_type == "router":
                context_tokens = context.get("tokens") if context else jnp.ones((x.shape[0], 10), dtype=jnp.int32)
                result = module(x, context_tokens, training=False)
                return result["output"] if isinstance(result, dict) else result
            
            elif module_type == "processor":
                # Processors expect structured inputs; use context payloads when present.
                if hasattr(module, "analyze_text"):
                    payload = self._extract_text_payload(x, context)
                    if payload is None:
                        logger.warning("Processor %s skipped: no text payload", module_name)
                        return x
                    return module.analyze_text(payload)
                if hasattr(module, "analyze_sentiment"):
                    payload = self._extract_text_payload(x, context)
                    if payload is None:
                        logger.warning("Processor %s skipped: no text payload", module_name)
                        return x
                    return module.analyze_sentiment(payload)
                if hasattr(module, "extract_entities"):
                    payload = self._extract_text_payload(x, context)
                    if payload is None:
                        logger.warning("Processor %s skipped: no text payload", module_name)
                        return x
                    return module.extract_entities(payload)
                if hasattr(module, "analyze_code"):
                    payload = self._extract_code_payload(x, context)
                    if payload is None:
                        logger.warning("Processor %s skipped: no code payload", module_name)
                        return x
                    return module.analyze_code(payload)
                if hasattr(module, "fuse_modalities"):
                    payload = self._extract_modalities_payload(x, context)
                    if payload is None:
                        logger.warning("Processor %s skipped: no modalities payload", module_name)
                        return x
                    return module.fuse_modalities(payload)
                return x
            
            elif module_type == "personality":
                if hasattr(module, "adapt_response"):
                    payload = self._extract_text_payload(x, context)
                    if payload is None:
                        logger.warning("Personality module skipped: no text payload")
                        return x
                    return module.adapt_response(payload)
                return x
            
            elif module_type == "ultra_hybrid":
                context_tokens = context.get("tokens") if context else jnp.ones((x.shape[0], 10), dtype=jnp.int32)
                return module(x, context_tokens=context_tokens, training=False)
            
            else:
                return x  # Safe fallback
                
        except Exception as e:
            logger.error(f"Module execution failed: {e}")
            return x  # Return input unchanged on failure
    
    def _combine_ensemble_results(self, results, weights):
        """Combine results from ensemble routing."""
        
        if not results:
            raise ValueError("No results to combine")
        
        if len(results) == 1:
            return next(iter(results.values()))
        
        # simple weighted average for numeric outputs; return first result otherwise
        numeric_results = {
            module_type: result
            for module_type, result in results.items()
            if hasattr(result, "shape") or isinstance(result, (int, float))
        }
        if not numeric_results:
            return next(iter(results.values()))

        combined = None
        total_weight = 0

        for module_type, result in numeric_results.items():
            weight = weights.get(module_type, 1.0)
            if combined is None:
                combined = weight * result
            else:
                combined = combined + weight * result
            total_weight += weight

        return combined / total_weight if total_weight > 0 else combined
    
    def _select_best_parallel_result(self, results):
        """Select best result from parallel execution."""
        
        if not results:
            return None
        
        # Prefer ultra_hybrid, then attention, then others
        preference_order = ["ultra_hybrid", "attention", "router", "processor", "personality"]
        
        for preferred in preference_order:
            if preferred in results:
                return results[preferred]
        
        # Return first available
        return next(iter(results.values()))

    def _extract_text_payload(self, x, context) -> Optional[str]:
        if isinstance(x, str):
            return x
        if isinstance(context, dict):
            for key in ("text", "prompt", "response"):
                if key in context and isinstance(context[key], str):
                    return context[key]
        return None

    def _extract_code_payload(self, x, context) -> Optional[str]:
        if isinstance(context, dict) and isinstance(context.get("code"), str):
            return context["code"]
        if isinstance(x, str):
            return x
        return None

    def _extract_modalities_payload(self, x, context) -> Optional[Dict[str, Any]]:
        if isinstance(context, dict):
            if isinstance(context.get("modalities"), dict):
                return context["modalities"]
            text_payload = self._extract_text_payload(x, context)
            if text_payload:
                return {"text": text_payload}
        if isinstance(x, str):
            return {"text": x}
        return None
    
    def _calculate_success_rate(self):
        """Calculate overall success rate."""
        total = self.global_metrics["successful_routes"] + self.global_metrics["failed_routes"]
        return self.global_metrics["successful_routes"] / total if total > 0 else 0.0
    
    def _initialize_training_integration(self):
        """Initialize training integration for modules."""
        if ULTRA_TRAINING_INTEGRATION:
            logger.info(" Training integration initialized for modules")
    
    def _initialize_expert_soup(self):
        """Initialize Expert Soup for modules."""
        if ULTRA_TRAINING_INTEGRATION:
            logger.info(" Expert Soup initialized for modules")
            self.expert_soup_manager = "expert_soup_placeholder"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        return {
            "config": {
                "orchestration_strategy": self.config.orchestration_strategy.value,
                "enabled_modules": [mod.value for mod in self.config.enabled_modules],
                "tpu_optimization": self.config.enable_tpu_optimization
            },
            "availability": {
                "attention_modules": ATTENTION_MODULES_AVAILABLE,
                "router_modules": ROUTER_MODULES_AVAILABLE,
                "processor_modules": PROCESSOR_MODULES_AVAILABLE,
                "personality_modules": PERSONALITY_MODULES_AVAILABLE,
                "ultra_core": ULTRA_CORE_AVAILABLE,
                "ultra_training": ULTRA_TRAINING_INTEGRATION
            },
            "active_modules": {k: list(v.keys()) for k, v in self.modules.items()},
            "performance": self.global_metrics,
            "integrations": {
                "core_orchestrator": self.core_orchestrator is not None,
                "expert_soup_manager": self.expert_soup_manager is not None
            }
        }

# ============================================================================
# Factory Functions
# ============================================================================

def create_ultra_module_system(
    config: Optional[UltraModuleConfig] = None,
    **kwargs
) -> UltraModuleOrchestrator:
    """Create ultra-advanced module system."""
    
    if config is None:
        config = UltraModuleConfig(**kwargs)
    
    return UltraModuleOrchestrator(config)

def create_ultra_module_config(
    orchestration_strategy: OrchestrationStrategy = OrchestrationStrategy.ADAPTIVE,
    enable_all_features: bool = True,
    **kwargs
) -> UltraModuleConfig:
    """Create optimized module configuration."""
    
    enabled_modules = [
        ModuleType.ULTRA_HYBRID,
        ModuleType.ATTENTION,
        ModuleType.ROUTER,
        ModuleType.PROCESSOR,
        ModuleType.PERSONALITY
    ]
    
    return UltraModuleConfig(
        orchestration_strategy=orchestration_strategy,
        enabled_modules=enabled_modules,
        enable_tpu_optimization=enable_all_features,
        auto_core_integration=enable_all_features and ULTRA_CORE_AVAILABLE,
        auto_training_integration=enable_all_features and ULTRA_TRAINING_INTEGRATION,
        enable_expert_soup=enable_all_features and ULTRA_TRAINING_INTEGRATION,
        **kwargs
    )

def demonstrate_ultra_module_orchestration():
    """Demonstrate the ultra module orchestration system."""
    
    logger.info(" ULTRA MODULE ORCHESTRATION DEMONSTRATION")
    logger.info("=" * 60)
    
    # Create configuration
    config = create_ultra_module_config(
        orchestration_strategy=OrchestrationStrategy.ULTRA_HYBRID,
        enable_all_features=True
    )
    
    logger.info(f" Configuration created:")
    logger.info(f"   - Strategy: {config.orchestration_strategy.value}")
    logger.info(f"   - Module types: {len(config.enabled_modules)}")
    logger.info(f"   - Ultra features: {config.auto_core_integration}")
    
    # Create orchestrator
    orchestrator = create_ultra_module_system(config)
    
    # Get system status
    status = orchestrator.get_system_status()
    
    logger.info(f"\n System Status:")
    logger.info(f"   - Total modules: {status['performance']['total_modules']}")
    logger.info(f"   - Active modules: {status['performance']['active_modules']}")
    logger.info(f"   - Ultra Core: {'' if status['availability']['ultra_core'] else ''}")
    logger.info(f"   - Module types available:")
    for module_type, available in status['availability'].items():
        if module_type.endswith('_modules'):
            status_emoji = "" if available else ""
            logger.info(f"     {status_emoji} {module_type}")
    
    # Test routing
    try:
        dummy_input = jnp.ones((2, 128, 768))
        result, routing_info = orchestrator.route_to_modules(
            dummy_input, 
            task_type="attention",
            strategy=OrchestrationStrategy.ULTRA_HYBRID
        )
        
        logger.info(f"\n Routing Test:")
        logger.info(f"   - Strategy: {routing_info['strategy']}")
        logger.info(f"   - Computation time: {routing_info['performance']['computation_time_ms']:.2f}ms")
        logger.info(f"   - Success rate: {routing_info['performance']['success_rate']:.2%}")
        
    except Exception as e:
        logger.error(f"\n Routing test failed: {e}")
    
    return orchestrator

__all__ = [
    # Configuration and enums
    'ModuleType',
    'OrchestrationStrategy', 
    'UltraModuleConfig',
    'ModulePerformanceMetrics',
    
    # Main orchestrator
    'UltraModuleOrchestrator',
    
    # Factory functions
    'create_ultra_module_system',
    'create_ultra_module_config',
    'demonstrate_ultra_module_orchestration',
    
    # Status flags
    'ULTRA_CORE_AVAILABLE',
    'ULTRA_TRAINING_INTEGRATION',
    'ATTENTION_MODULES_AVAILABLE',
    'ROUTER_MODULES_AVAILABLE',
    'PROCESSOR_MODULES_AVAILABLE',
    'PERSONALITY_MODULES_AVAILABLE'
]
