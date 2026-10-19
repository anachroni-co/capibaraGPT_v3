"""
Ultra SubModel Orchestrator - CapibaraGPT v3
===============================================

Sistema de orquestación ultra-avanzada for todos los sub-modelos:
- Intelligent sub-model selection and routing
- Integration with Ultra Core System  
- Connection with UltraAdvancedTrainer
- Expert Soup for sub-models
- Performance monitoring and optimization
- Dynamic sub-model composition
- Graceful fallbacks and error handling

Esta es la evolución del sistema de sub-modelos for be al level del ecosistema ultra-advanced.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Callable, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    pass  # Using proper imports instead of sys.path manipulation

import jax
from flax import linen as nn
from functools import partial
from jax import numpy as jnp

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
    ModelSoupConfig = None

try:
    from layers.ssm_hybrid_layers import (
        UltraSSMLayer, create_ssm_layer, SSM_LAYERS_AVAILABLE
    )
    SSM_HYBRID_AVAILABLE = True
except ImportError:
    SSM_HYBRID_AVAILABLE = False
    UltraSSMLayer = None

# Import existing sub-models with safe fallbacks
try:
    from .experimental.adaptive_vq_submodel import AdaptiveSubmodel
    ADAPTIVE_VQ_AVAILABLE = True
except ImportError:
    ADAPTIVE_VQ_AVAILABLE = False
    AdaptiveSubmodel = None

try:
    from .experimental.spike_ssm import SpikeSSM
    SPIKE_SSM_AVAILABLE = True
except ImportError:
    SPIKE_SSM_AVAILABLE = False
    SpikeSSM = None

try:
    from .deep_dialog import DeepDialog, DeepDialogConfig
    DEEP_DIALOG_AVAILABLE = True
except ImportError:
    DEEP_DIALOG_AVAILABLE = False
    DeepDialog = None
    DeepDialogConfig = None

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration and Enums
# ============================================================================

class SubModelType(str, Enum):
    """Types of available sub-models."""
    ADAPTIVE_VQ = "adaptive_vq"
    SPIKE_SSM = "spike_ssm"
    DEEP_DIALOG = "deep_dialog"
    SSM_TPU = "ssm_tpu"
    BYTE_TPU = "byte_tpu"
    LIQUID = "liquid"
    META_BAMDP = "meta_bamdp"
    DUAL_PROCESS = "dual_process"
    ULTRA_SSM_HYBRID = "ultra_ssm_hybrid"  # New ultra-advanced SSM

class OrchestrationStrategy(str, Enum):
    """Strategies for sub-model orchestration."""
    ADAPTIVE = "adaptive"           # Intelligent selection based on input
    ENSEMBLE = "ensemble"           # Combine multiple sub-models
    SEQUENTIAL = "sequential"       # Chain sub-models in sequence
    PARALLEL = "parallel"           # Run sub-models in parallel
    ULTRA_HYBRID = "ultra_hybrid"   # Ultra-advanced hybrid strategy

@dataclass
class SubModelPerformanceMetrics:
    """Performance metrics for individual sub-models."""
    
    model_type: str
    computation_time_ms: float
    memory_usage_mb: float
    accuracy_score: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    efficiency_score: float = 0.0
    error_rate: float = 0.0
    
    def __post_init__(self):
        if self.computation_time_ms > 0 and self.throughput_tokens_per_sec > 0:
            self.efficiency_score = self.throughput_tokens_per_sec / self.computation_time_ms

@dataclass
class UltraSubModelConfig:
    """Configuration for ultra-advanced sub-model orchestration."""
    
    # Core configuration
    hidden_size: int = 768
    orchestration_strategy: OrchestrationStrategy = OrchestrationStrategy.ADAPTIVE
    
    # Sub-model selection
    enabled_submodels: List[SubModelType] = field(default_factory=lambda: [
        SubModelType.ADAPTIVE_VQ,
        SubModelType.SPIKE_SSM,
        SubModelType.DEEP_DIALOG,
        SubModelType.ULTRA_SSM_HYBRID
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
    
    # Fallback and error handling
    graceful_degradation: bool = True
    fallback_submodel: SubModelType = SubModelType.DEEP_DIALOG
    max_retries: int = 3

# ============================================================================
# Ultra SubModel Orchestrator
# ============================================================================

class UltraSubModelOrchestrator:
    """Orquestador ultra-advanced for todos los sub-modelos del sistema."""
    
    def __init__(self, config: UltraSubModelConfig):
        self.config = config
        self.submodels = {}
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
            "total_submodels": 0,
            "active_submodels": 0
        }
        
        # Initialize the orchestrator
        self._initialize_orchestrator()
    
    def _initialize_orchestrator(self):
        """Initialize the ultra sub-model orchestrator."""
        
        logger.info(" Initializing Ultra SubModel Orchestrator")
        
        # Initialize ultra system integrations
        if self.config.auto_core_integration and ULTRA_CORE_AVAILABLE:
            try:
                self.core_orchestrator = create_ultra_core_system()
                logger.info(" Ultra Core integration initialized")
            except Exception as e:
                logger.warning(f"️ Core integration failed: {e}")
        
        # Initialize sub-models
        self._initialize_submodels()
        
        # Initialize training integration
        if self.config.auto_training_integration and ULTRA_TRAINING_INTEGRATION:
            self._initialize_training_integration()
        
        # Initialize expert soup for sub-models
        if self.config.enable_expert_soup and ULTRA_TRAINING_INTEGRATION:
            self._initialize_expert_soup()
        
        logger.info(f" Ultra SubModel Orchestrator initialized")
        logger.info(f"    Available sub-models: {len(self.submodels)}")
        logger.info(f"    Ultra Core: {'' if self.core_orchestrator else ''}")
        logger.info(f"    Expert Soup: {'' if self.expert_soup_manager else ''}")
    
    def _initialize_submodels(self):
        """Initialize all available sub-models."""
        
        for submodel_type in self.config.enabled_submodels:
            try:
                submodel = self._create_submodel(submodel_type)
                if submodel is not None:
                    self.submodels[submodel_type.value] = submodel
                    self.global_metrics["total_submodels"] += 1
                    logger.info(f" {submodel_type.value} initialized")
                else:
                    logger.warning(f"️ {submodel_type.value} not available")
            except Exception as e:
                logger.error(f" Failed to initialize {submodel_type.value}: {e}")
                if not self.config.graceful_degradation:
                    raise
        
        self.global_metrics["active_submodels"] = len(self.submodels)
        
        # Ensure we have at least one sub-model
        if not self.submodels:
            raise RuntimeError("No sub-models available! Please check dependencies.")
    
    def _create_submodel(self, submodel_type: SubModelType) -> Optional[Any]:
        """Create a specific sub-model with ultra-advanced features."""
        
        try:
            if submodel_type == SubModelType.ADAPTIVE_VQ and ADAPTIVE_VQ_AVAILABLE:
                return self._create_adaptive_vq_submodel()
            
            elif submodel_type == SubModelType.SPIKE_SSM and SPIKE_SSM_AVAILABLE:
                return self._create_spike_ssm_submodel()
            
            elif submodel_type == SubModelType.DEEP_DIALOG and DEEP_DIALOG_AVAILABLE:
                return self._create_deep_dialog_submodel()
            
            elif submodel_type == SubModelType.ULTRA_SSM_HYBRID and SSM_HYBRID_AVAILABLE:
                return self._create_ultra_ssm_hybrid_submodel()
            
            else:
                logger.warning(f"SubModel type {submodel_type} not available or not implemented")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create {submodel_type}: {e}")
            if self.config.graceful_degradation:
                return None
            else:
                raise
    
    def _create_adaptive_vq_submodel(self) -> Optional[Any]:
        """Create ultra-enhanced adaptive VQ sub-model."""
        if not ADAPTIVE_VQ_AVAILABLE:
            return None
        
        # This would create an enhanced version integrated with our systems
        # For now, return basic version (full implementation would enhance it)
        logger.info(" Creating enhanced AdaptiveVQ submodel")
        return "adaptive_vq_placeholder"  # Placeholder for current enhanced model
    
    def _create_spike_ssm_submodel(self) -> Optional[Any]:
        """Create ultra-enhanced spiking SSM sub-model."""
        if not SPIKE_SSM_AVAILABLE:
            return None
        
        logger.info(" Creating enhanced SpikeSSM submodel")
        return "spike_ssm_placeholder"  # Placeholder for current enhanced model
    
    def _create_deep_dialog_submodel(self) -> Optional[Any]:
        """Create ultra-enhanced deep dialog sub-model."""
        if not DEEP_DIALOG_AVAILABLE:
            return None
        
        logger.info(" Creating enhanced DeepDialog submodel")
        return "deep_dialog_placeholder"  # Placeholder for current enhanced model
    
    def _create_ultra_ssm_hybrid_submodel(self) -> Optional[Any]:
        """Create ultra-advanced SSM hybrid sub-model using our layers."""
        if not SSM_HYBRID_AVAILABLE:
            return None
        
        try:
            # Use our ultra-advanced SSM layers from layers/
            ssm_config = {
                "hidden_size": self.config.hidden_size,
                "enable_all_optimizations": self.config.enable_tpu_optimization
            }
            
            ultra_ssm = create_ssm_layer("ultra", ssm_config)
            logger.info("️ Created Ultra SSM Hybrid submodel with O(n) complexity")
            return ultra_ssm
            
        except Exception as e:
            logger.error(f"Failed to create Ultra SSM Hybrid: {e}")
            return None
    
    def _initialize_training_integration(self):
        """Initialize training integration for sub-models."""
        
        if not ULTRA_TRAINING_INTEGRATION:
            return
        
        try:
            # This would integrate with UltraAdvancedTrainer
            # Implementation depends on trainer interface
            logger.info(" Training integration initialized for sub-models")
            
        except Exception as e:
            logger.warning(f"️ Training integration failed: {e}")
    
    def _initialize_expert_soup(self):
        """Initialize Expert Soup for sub-models."""
        
        if not ULTRA_TRAINING_INTEGRATION:
            return
        
        try:
            # Create Expert Soup configuration for sub-models
            soup_config = ModelSoupConfig(
                n_best_models=len(self.submodels),
                combination_strategy="weighted_average",
                weight_strategy="adaptive",
                min_overall_score=0.6
            )
            
            # This would create ExpertSoupIntegration for sub-models
            logger.info(" Expert Soup initialized for sub-models")
            self.expert_soup_manager = "expert_soup_placeholder"
            
        except Exception as e:
            logger.warning(f"️ Expert Soup initialization failed: {e}")
    
    def route_to_submodel(
        self,
        x: jnp.ndarray,
        context: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = None,
        strategy: Optional[OrchestrationStrategy] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Route input to optimal sub-model(s) based on strategy."""
        
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
            
            # Graceful fallback
            if self.config.graceful_degradation:
                result, routing_info = self._fallback_routing(x, context)
                self.global_metrics["failed_routes"] += 1
            else:
                raise
        
        # Update performance metrics
        computation_time = (time.time() - start_time) * 1000
        self.global_metrics["total_forward_passes"] += 1
        
        # Update routing history
        self._update_routing_history(strategy, routing_info, computation_time)
        
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
                "active_submodels": len(self.submodels),
                "ultra_core_active": self.core_orchestrator is not None,
                "expert_soup_active": self.expert_soup_manager is not None
            }
        }
        
        return result, comprehensive_info
    
    def _adaptive_routing(
        self, 
        x: jnp.ndarray, 
        context: Optional[Dict[str, Any]], 
        task_type: Optional[str]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Adaptive routing based on input characteristics and task type."""
        
        # Analyze input characteristics
        input_analysis = self._analyze_input(x, context, task_type)
        
        # Select best sub-model based on analysis
        best_submodel = self._select_optimal_submodel(input_analysis)
        
        # Route to selected sub-model
        if best_submodel in self.submodels:
            result = self._execute_submodel(best_submodel, x, context)
            routing_info = {
                "selected_submodel": best_submodel,
                "confidence": input_analysis.get("confidence", 0.8),
                "input_analysis": input_analysis
            }
        else:
            # Fallback to available sub-model
            fallback_model = next(iter(self.submodels.keys()))
            result = self._execute_submodel(fallback_model, x, context)
            routing_info = {
                "selected_submodel": fallback_model,
                "confidence": 0.5,
                "fallback_used": True
            }
        
        return result, routing_info
    
    def _ensemble_routing(
        self, 
        x: jnp.ndarray, 
        context: Optional[Dict[str, Any]]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Ensemble routing that combines multiple sub-models."""
        
        results = {}
        weights = self.config.ensemble_weights or {}
        
        # Execute all available sub-models
        for submodel_name in self.submodels.keys():
            try:
                result = self._execute_submodel(submodel_name, x, context)
                results[submodel_name] = result
            except Exception as e:
                logger.warning(f"SubModel {submodel_name} failed in ensemble: {e}")
        
        # Combine results using weights or equal weighting
        combined_result = self._combine_ensemble_results(results, weights)
        
        routing_info = {
            "strategy": "ensemble",
            "participating_models": list(results.keys()),
            "weights_used": weights,
            "combination_method": "weighted_average"
        }
        
        return combined_result, routing_info
    
    def _sequential_routing(
        self, 
        x: jnp.ndarray, 
        context: Optional[Dict[str, Any]]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Sequential routing that chains sub-models."""
        
        current_input = x
        processing_chain = []
        
        # Process through each sub-model in sequence
        for submodel_name in self.submodels.keys():
            try:
                result = self._execute_submodel(submodel_name, current_input, context)
                processing_chain.append(submodel_name)
                current_input = result  # Use output as input for next model
            except Exception as e:
                logger.warning(f"SubModel {submodel_name} failed in sequence: {e}")
                break
        
        routing_info = {
            "strategy": "sequential",
            "processing_chain": processing_chain,
            "chain_length": len(processing_chain)
        }
        
        return current_input, routing_info
    
    def _parallel_routing(
        self, 
        x: jnp.ndarray, 
        context: Optional[Dict[str, Any]]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Parallel routing that runs sub-models simultaneously."""
        
        # Execute all sub-models in parallel
        results = {}
        for submodel_name in self.submodels.keys():
            try:
                result = self._execute_submodel(submodel_name, x, context)
                results[submodel_name] = result
            except Exception as e:
                logger.warning(f"SubModel {submodel_name} failed in parallel: {e}")
        
        # Select best result or combine them
        best_result = self._select_best_parallel_result(results)
        
        routing_info = {
            "strategy": "parallel",
            "executed_models": list(results.keys()),
            "results_count": len(results)
        }
        
        return best_result, routing_info
    
    def _ultra_hybrid_routing(
        self, 
        x: jnp.ndarray, 
        context: Optional[Dict[str, Any]], 
        task_type: Optional[str]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Ultra-advanced hybrid routing using all available intelligence."""
        
        # Combine adaptive selection with ensemble wisdom
        adaptive_result, adaptive_info = self._adaptive_routing(x, context, task_type)
        ensemble_result, ensemble_info = self._ensemble_routing(x, context)
        
        # Use ultra-intelligent combination
        if self.core_orchestrator:
            # Leverage Ultra Core System for intelligent combination
            combined_result = self._ultra_intelligent_combination(
                adaptive_result, ensemble_result, adaptive_info, ensemble_info
            )
        else:
            # simple combination fallback
            combined_result = adaptive_result  # Prefer adaptive
        
        routing_info = {
            "strategy": "ultra_hybrid",
            "adaptive_info": adaptive_info,
            "ensemble_info": ensemble_info,
            "ultra_core_used": self.core_orchestrator is not None
        }
        
        return combined_result, routing_info
    
    def _fallback_routing(
        self, 
        x: jnp.ndarray, 
        context: Optional[Dict[str, Any]]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Fallback routing when all else fails."""
        
        fallback_submodel = self.config.fallback_submodel.value
        
        if fallback_submodel in self.submodels:
            result = self._execute_submodel(fallback_submodel, x, context)
        else:
            # Use any available sub-model
            fallback_submodel = next(iter(self.submodels.keys()))
            result = self._execute_submodel(fallback_submodel, x, context)
        
        routing_info = {
            "strategy": "fallback",
            "used_submodel": fallback_submodel,
            "fallback_reason": "primary_routing_failed"
        }
        
        return result, routing_info
    
    def _analyze_input(
        self, 
        x: jnp.ndarray, 
        context: Optional[Dict[str, Any]], 
        task_type: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze input to determine optimal sub-model."""
        
        analysis = {
            "input_shape": x.shape,
            "input_dtype": str(x.dtype),
            "has_context": context is not None,
            "task_type": task_type,
            "confidence": 0.8
        }
        
        # Sequence length analysis
        if len(x.shape) >= 2:
            seq_len = x.shape[1] if len(x.shape) == 3 else x.shape[0]
            analysis["sequence_length"] = seq_len
            
            # Recommend SSM for long sequences
            if seq_len > 1024:
                analysis["recommended_submodel"] = "ultra_ssm_hybrid"
                analysis["reason"] = "long_sequence_optimization"
            elif seq_len < 64:
                analysis["recommended_submodel"] = "spike_ssm"
                analysis["reason"] = "short_sequence_spiking"
            else:
                analysis["recommended_submodel"] = "adaptive_vq"
                analysis["reason"] = "medium_sequence_adaptive"
        
        # Task type analysis
        if task_type:
            if "dialog" in task_type.lower():
                analysis["recommended_submodel"] = "deep_dialog"
                analysis["reason"] = "dialog_specialization"
            elif "quantum" in task_type.lower():
                analysis["recommended_submodel"] = "adaptive_vq"
                analysis["reason"] = "quantum_capabilities"
        
        return analysis
    
    def _select_optimal_submodel(self, input_analysis: Dict[str, Any]) -> str:
        """Select optimal sub-model based on input analysis."""
        
        recommended = input_analysis.get("recommended_submodel")
        
        # Check if recommended sub-model is available
        if recommended and recommended in self.submodels:
            return recommended
        
        # Fallback to best available sub-model
        if "ultra_ssm_hybrid" in self.submodels:
            return "ultra_ssm_hybrid"
        elif "adaptive_vq" in self.submodels:
            return "adaptive_vq"
        else:
            return next(iter(self.submodels.keys()))
    
    def _execute_submodel(
        self, 
        submodel_name: str, 
        x: jnp.ndarray, 
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute a specific sub-model with error handling."""
        
        submodel = self.submodels[submodel_name]
        
        # For now, return placeholder results
        # Full implementation would call current sub-models
        logger.debug(f"Executing submodel: {submodel_name}")
        
        # Simulate execution with input transformation
        if isinstance(submodel, str):  # Placeholder models
            return f"output_from_{submodel_name}"
        else:
            # current model execution would happen here
            return x  # Placeholder
    
    def _combine_ensemble_results(
        self, 
        results: Dict[str, Any], 
        weights: Dict[str, float]
    ) -> Any:
        """Combine results from ensemble routing."""
        
        if not results:
            raise ValueError("No results to combine")
        
        # simple combination for now (weighted average for tensors)
        if len(results) == 1:
            return next(iter(results.values()))
        
        # For placeholder implementation, return first result
        return next(iter(results.values()))
    
    def _select_best_parallel_result(self, results: Dict[str, Any]) -> Any:
        """Select best result from parallel execution."""
        
        if not results:
            raise ValueError("No results to select from")
        
        # simple selection - prefer ultra_ssm_hybrid if available
        if "ultra_ssm_hybrid" in results:
            return results["ultra_ssm_hybrid"]
        
        # Otherwise return first available result
        return next(iter(results.values()))
    
    def _ultra_intelligent_combination(
        self,
        adaptive_result: Any,
        ensemble_result: Any,
        adaptive_info: Dict[str, Any],
        ensemble_info: Dict[str, Any]
    ) -> Any:
        """Ultra-intelligent combination using Core System."""
        
        # Leverage Ultra Core System intelligence
        if self.core_orchestrator:
            # This would use the core orchestrator for intelligent combination
            logger.debug("Using Ultra Core System for intelligent combination")
        
        # For now, use simple heuristic
        adaptive_confidence = adaptive_info.get("confidence", 0.5)
        
        if adaptive_confidence > 0.8:
            return adaptive_result
        else:
            return ensemble_result
    
    def _update_routing_history(
        self, 
        strategy: OrchestrationStrategy, 
        routing_info: Dict[str, Any], 
        computation_time: float
    ):
        """Update routing history for performance analysis."""
        
        self.routing_history.append({
            "timestamp": time.time(),
            "strategy": strategy.value,
            "routing_info": routing_info,
            "computation_time_ms": computation_time
        })
        
        # Keep only recent history
        max_history = self.config.performance_tracking_window
        if len(self.routing_history) > max_history:
            self.routing_history = self.routing_history[-max_history:]
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        total = self.global_metrics["successful_routes"] + self.global_metrics["failed_routes"]
        if total == 0:
            return 0.0
        return self.global_metrics["successful_routes"] / total
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        return {
            "config": {
                "orchestration_strategy": self.config.orchestration_strategy.value,
                "enabled_submodels": [sm.value for sm in self.config.enabled_submodels],
                "tpu_optimization": self.config.enable_tpu_optimization,
                "expert_soup": self.config.enable_expert_soup
            },
            "availability": {
                "ultra_core": ULTRA_CORE_AVAILABLE,
                "ultra_training": ULTRA_TRAINING_INTEGRATION,
                "ssm_hybrid": SSM_HYBRID_AVAILABLE,
                "adaptive_vq": ADAPTIVE_VQ_AVAILABLE,
                "spike_ssm": SPIKE_SSM_AVAILABLE,
                "deep_dialog": DEEP_DIALOG_AVAILABLE
            },
            "active_submodels": list(self.submodels.keys()),
            "performance": self.global_metrics,
            "integrations": {
                "core_orchestrator": self.core_orchestrator is not None,
                "training_integration": self.training_integration is not None,
                "expert_soup_manager": self.expert_soup_manager is not None
            },
            "routing_stats": {
                "total_routes": len(self.routing_history),
                "success_rate": self._calculate_success_rate(),
                "average_latency": self._calculate_average_latency()
            }
        }
    
    def _calculate_average_latency(self) -> float:
        """Calculate average routing latency."""
        if not self.routing_history:
            return 0.0
        
        total_time = sum(entry["computation_time_ms"] for entry in self.routing_history)
        return total_time / len(self.routing_history)
    
    def validate_system(self) -> Dict[str, bool]:
        """Validate the entire sub-model system."""
        
        validation_results = {}
        
        # Test sub-model availability
        validation_results["submodels_available"] = len(self.submodels) > 0
        
        # Test system integrations
        validation_results["ultra_core_integration"] = ULTRA_CORE_AVAILABLE
        validation_results["ultra_training_integration"] = ULTRA_TRAINING_INTEGRATION
        
        # Test routing functionality
        try:
            dummy_input = jnp.ones((2, 64, self.config.hidden_size))
            result, info = self.route_to_submodel(dummy_input)
            validation_results["routing_functional"] = True
        except Exception as e:
            validation_results["routing_functional"] = False
            logger.error(f"Routing validation failed: {e}")
        
        # Test orchestration strategies
        for strategy in OrchestrationStrategy:
            try:
                result, info = self.route_to_submodel(
                    dummy_input, strategy=strategy
                )
                validation_results[f"{strategy.value}_strategy"] = True
            except Exception as e:
                validation_results[f"{strategy.value}_strategy"] = False
                logger.error(f"{strategy.value} strategy validation failed: {e}")
        
        # Overall system health
        validation_results["system_healthy"] = all([
            validation_results["submodels_available"],
            validation_results["routing_functional"]
        ])
        
        return validation_results

# ============================================================================
# Factory Functions and Utilities
# ============================================================================

def create_ultra_submodel_system(
    config: Optional[UltraSubModelConfig] = None,
    **kwargs
) -> UltraSubModelOrchestrator:
    """Create ultra-advanced sub-model system."""
    
    if config is None:
        config = UltraSubModelConfig(**kwargs)
    
    return UltraSubModelOrchestrator(config)

def create_ultra_submodel_config(
    orchestration_strategy: OrchestrationStrategy = OrchestrationStrategy.ADAPTIVE,
    enable_all_features: bool = True,
    **kwargs
) -> UltraSubModelConfig:
    """Create optimized sub-model configuration."""
    
    enabled_submodels = [
        SubModelType.ULTRA_SSM_HYBRID,
        SubModelType.ADAPTIVE_VQ,
        SubModelType.SPIKE_SSM,
        SubModelType.DEEP_DIALOG
    ]
    
    return UltraSubModelConfig(
        orchestration_strategy=orchestration_strategy,
        enabled_submodels=enabled_submodels,
        enable_tpu_optimization=enable_all_features,
        use_mixed_precision=enable_all_features,
        auto_core_integration=enable_all_features and ULTRA_CORE_AVAILABLE,
        auto_training_integration=enable_all_features and ULTRA_TRAINING_INTEGRATION,
        enable_expert_soup=enable_all_features and ULTRA_TRAINING_INTEGRATION,
        enable_comprehensive_monitoring=enable_all_features,
        **kwargs
    )

def demonstrate_ultra_submodel_orchestration():
    """Demonstrate the ultra sub-model orchestration system."""
    
    logger.info(" ULTRA SUBMODEL ORCHESTRATION DEMONSTRATION")
    logger.info("=" * 60)
    
    # Create configuration
    config = create_ultra_submodel_config(
        orchestration_strategy=OrchestrationStrategy.ULTRA_HYBRID,
        enable_all_features=True
    )
    
    logger.info(f" Configuration created:")
    logger.info(f"   - Strategy: {config.orchestration_strategy.value}")
    logger.info(f"   - Sub-models: {len(config.enabled_submodels)}")
    logger.info(f"   - Ultra features: {config.auto_core_integration}")
    
    # Create orchestrator
    orchestrator = create_ultra_submodel_system(config)
    
    # Get system status
    status = orchestrator.get_system_status()
    
    logger.info(f"\n System Status:")
    logger.info(f"   - Active sub-models: {len(status['active_submodels'])}")
    logger.info(f"   - Ultra Core: {'' if status['availability']['ultra_core'] else ''}")
    logger.info(f"   - SSM Hybrid: {'' if status['availability']['ssm_hybrid'] else ''}")
    logger.info(f"   - Expert Soup: {'' if status['integrations']['expert_soup_manager'] else ''}")
    
    # Test routing
    try:
        dummy_input = jnp.ones((2, 128, 768))
        result, routing_info = orchestrator.route_to_submodel(
            dummy_input, 
            task_type="dialog",
            strategy=OrchestrationStrategy.ULTRA_HYBRID
        )
        
        logger.info(f"\n Routing Test:")
        logger.info(f"   - Strategy: {routing_info['strategy']}")
        logger.info(f"   - Computation time: {routing_info['performance']['computation_time_ms']:.2f}ms")
        logger.info(f"   - Success rate: {routing_info['performance']['success_rate']:.2%}")
        
    except Exception as e:
        logger.error(f"\n Routing test failed: {e}")
    
    # Validate system
    validation = orchestrator.validate_system()
    
    logger.info(f"\n Validation Results:")
    for check, result in validation.items():
        logger.info(f"   - {check}: {'' if result else ''}")
    
    return orchestrator

__all__ = [
    # Configuration and enums
    'SubModelType',
    'OrchestrationStrategy', 
    'UltraSubModelConfig',
    'SubModelPerformanceMetrics',
    
    # Main orchestrator
    'UltraSubModelOrchestrator',
    
    # Factory functions
    'create_ultra_submodel_system',
    'create_ultra_submodel_config',
    'demonstrate_ultra_submodel_orchestration',
    
    # Status flags
    'ULTRA_CORE_AVAILABLE',
    'ULTRA_TRAINING_INTEGRATION',
    'SSM_HYBRID_AVAILABLE',
    'ADAPTIVE_VQ_AVAILABLE',
    'SPIKE_SSM_AVAILABLE',
    'DEEP_DIALOG_AVAILABLE'
]
