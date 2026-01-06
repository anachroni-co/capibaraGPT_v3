#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Core Integration Module - CapibaraGPT

Enhanced integration between core and training modules with proper encoding handling,
comprehensive training integration, and optimized performance.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
import os

# Core imports
try:
    from capibara.core.router import EnhancedRouter, RouterConfig, RouterType
    from capibara.core.routing import CoreHttpRouter
    try:
        from capibara.core.config import Config
    except ImportError:
        # Create a simple Config class if not available
        class Config:
            def __init__(self):
                pass
    CORE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Core modules not available: {e}")
    CORE_AVAILABLE = False

# Training imports
try:
    from capibara.training.expanded_expert_cores_strategy import (
        ExpandedExpertCoresStrategy, 
        ExpertCoreType,
        ExpertCoreConfig
    )
    from capibara.training.huggingface_consensus_strategy import (
        HuggingFaceConsensusStrategy
    )
    from capibara.training.cascade_training_integration import (
        CascadeTrainingIntegration,
        StageTrainingConfig
    )
    TRAINING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Training modules not available: {e}")
    TRAINING_AVAILABLE = False

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    from capibara.jax import nn
    JAX_AVAILABLE = True
except ImportError as e:
    logging.warning(f"JAX not available: {e}")
    JAX_AVAILABLE = False

logger = logging.getLogger(__name__)

class IntegrationType(Enum):
    """Types of integration available."""
    FULL = "full"
    CORE_ONLY = "core_only"
    TRAINING_ONLY = "training_only"
    HYBRID = "hybrid"
    MINIMAL = "minimal"

@dataclass
class IntegrationConfig:
    """Configuration for ultra core integration."""
    integration_type: IntegrationType = IntegrationType.FULL
    encoding: str = "utf-8"
    enable_caching: bool = True
    cache_size: int = 1000
    enable_monitoring: bool = True
    enable_metrics: bool = True
    training_integration_enabled: bool = True
    expert_cores_enabled: bool = True
    consensus_enabled: bool = True
    config_path: Optional[str] = None
    log_level: str = "INFO"

@dataclass
class IntegrationMetrics:
    """Metrics for integration performance."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    training_integration_used: int = 0
    expert_cores_used: int = 0
    consensus_used: int = 0
    average_response_time: float = 0.0
    last_update: float = 0.0

@dataclass
class IntegrationResult:
    """Result of integration operation."""
    success: bool
    result: Any
    integration_type: str
    processing_time: float
    cache_hit: bool = False
    training_integration_used: bool = False
    expert_cores_used: List[str] = None
    consensus_used: bool = False
    error_message: Optional[str] = None

class UltraCoreIntegration:
    """
    Ultra Core Integration - Complete integration between core and training modules.
    
    This class provides:
    1. Enhanced router integration with training strategies
    2. Comprehensive caching and performance optimization
    3. Metrics and monitoring capabilities
    4. Proper encoding handling
    5. Fallback mechanisms for missing components
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        """Initialize the ultra core integration."""
        self.config = config or IntegrationConfig()
        self._setup_logging()
        self._initialize_components()
        self._setup_metrics()
        
    def _setup_logging(self):
        """Setup logging with proper encoding."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            encoding=self.config.encoding
        )
        logger.info(f"Ultra Core Integration initialized with type: {self.config.integration_type.value}")
        
    def _initialize_components(self):
        """Initialize all integration components."""
        self.components = {}
        self.cache = {}
        
        # Initialize core components
        if CORE_AVAILABLE:
            self._initialize_core_components()
            
        # Initialize training components
        if TRAINING_AVAILABLE and self.config.training_integration_enabled:
            self._initialize_training_components()
            
        # Initialize JAX components
        if JAX_AVAILABLE:
            self._initialize_jax_components()
            
        logger.info(f"Components initialized: {list(self.components.keys())}")
        
    def _initialize_core_components(self):
        """Initialize core components."""
        try:
            # Enhanced router
            router_config = RouterConfig(
                router_type=RouterType.HYBRID,
                use_training_integration=self.config.training_integration_enabled,
                expert_cores_enabled=self.config.expert_cores_enabled,
                consensus_enabled=self.config.consensus_enabled,
                cache_size=self.config.cache_size,
                encoding=self.config.encoding
            )
            self.components['enhanced_router'] = EnhancedRouter(router_config)
            
            # Core router
            self.components['core_router'] = CoreHttpRouter()
            
            # Configuration
            self.components['config'] = Config()
            
            logger.info("Core components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing core components: {e}")
            
    def _initialize_training_components(self):
        """Initialize training components."""
        try:
            # Expert cores strategy
            if self.config.expert_cores_enabled:
                self.components['expert_strategy'] = ExpandedExpertCoresStrategy({})
                logger.info("Expert cores strategy initialized")
                
            # Consensus strategy
            if self.config.consensus_enabled:
                self.components['consensus_strategy'] = HuggingFaceConsensusStrategy({})
                logger.info("Consensus strategy initialized")
                
            # Cascade training integration
            self.components['cascade_integration'] = CascadeTrainingIntegration()
            logger.info("Cascade training integration initialized")
            
        except Exception as e:
            logger.error(f"Error initializing training components: {e}")
            
    def _initialize_jax_components(self):
        """Initialize JAX components."""
        try:
            # JAX configuration - no x64 on TPU v6e (use BF16/FP32)
            # jax.config.update('jax_enable_x64', True)  # Disabled for TPU v6e
            self.components['jax_available'] = True
            logger.info("JAX components initialized for TPU v6e (BF16/FP32)")
            
        except Exception as e:
            logger.error(f"Error initializing JAX components: {e}")
            self.components['jax_available'] = False
            
    def _setup_metrics(self):
        """Setup metrics tracking."""
        self.metrics = IntegrationMetrics()
        self.metrics.last_update = time.time()
        
    async def process_request(
        self, 
        request_data: Union[str, Dict[str, Any]], 
        request_type: str = "general",
        use_cache: bool = None,
        context: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """Process a request through the integrated system."""
        start_time = time.time()
        
        try:
            # Update metrics
            self.metrics.total_requests += 1
            
            # Determine cache usage
            use_cache = use_cache if use_cache is not None else self.config.enable_caching
            
            # Check cache
            cache_key = self._compute_cache_key(request_data, request_type)
            if use_cache and cache_key in self.cache:
                self.metrics.cache_hits += 1
                cached_result = self.cache[cache_key]
                cached_result.cache_hit = True
                return cached_result
                
            self.metrics.cache_misses += 1
            
            # Process based on integration type
            if self.config.integration_type == IntegrationType.FULL:
                result = await self._process_full_integration(request_data, request_type, context)
            elif self.config.integration_type == IntegrationType.CORE_ONLY:
                result = await self._process_core_only(request_data, request_type, context)
            elif self.config.integration_type == IntegrationType.TRAINING_ONLY:
                result = await self._process_training_only(request_data, request_type, context)
            elif self.config.integration_type == IntegrationType.HYBRID:
                result = await self._process_hybrid(request_data, request_type, context)
            else:  # MINIMAL
                result = await self._process_minimal(request_data, request_type, context)
                
            # Update cache
            if use_cache:
                self._update_cache(cache_key, result)
                
            # Update metrics
            self._update_metrics(result, time.time() - start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            self.metrics.failed_requests += 1
            
            return IntegrationResult(
                success=False,
                result=None,
                integration_type=self.config.integration_type.value,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
            
    async def _process_full_integration(
        self, 
        request_data: Union[str, Dict[str, Any]], 
        request_type: str,
        context: Optional[Dict[str, Any]]
    ) -> IntegrationResult:
        """Process request with full integration."""
        try:
            # Use enhanced router with training integration
            if 'enhanced_router' in self.components:
                router_result = await self.components['enhanced_router'].route_request(
                    request_data, context, use_cache=True, training_mode=False
                )
                
                if router_result.success:
                    return IntegrationResult(
                        success=True,
                        result=router_result,
                        integration_type="full_enhanced_router",
                        processing_time=router_result.processing_time,
                        training_integration_used=router_result.training_integration_used,
                        expert_cores_used=router_result.expert_cores_used
                    )
                    
            # Fallback to core router
            return await self._process_core_only(request_data, request_type, context)
            
        except Exception as e:
            logger.error(f"Error in full integration: {e}")
            return await self._process_core_only(request_data, request_type, context)
            
    async def _process_core_only(
        self, 
        request_data: Union[str, Dict[str, Any]], 
        request_type: str,
        context: Optional[Dict[str, Any]]
    ) -> IntegrationResult:
        """Process request with core components only."""
        try:
            if 'core_router' in self.components:
                result = self.components['core_router'].route(request_type, **context or {})
                return IntegrationResult(
                    success=True,
                    result=result,
                    integration_type="core_only",
                    processing_time=0.0
                )
            else:
                return IntegrationResult(
                    success=False,
                    result=None,
                    integration_type="core_only",
                    processing_time=0.0,
                    error_message="Core router not available"
                )
                
        except Exception as e:
            logger.error(f"Error in core only processing: {e}")
            return IntegrationResult(
                success=False,
                result=None,
                integration_type="core_only",
                processing_time=0.0,
                error_message=str(e)
            )
            
    async def _process_training_only(
        self, 
        request_data: Union[str, Dict[str, Any]], 
        request_type: str,
        context: Optional[Dict[str, Any]]
    ) -> IntegrationResult:
        """Process request with training components only."""
        try:
            if isinstance(request_data, str):
                prompt = request_data
            else:
                prompt = request_data.get("text", str(request_data))
                
            # Try expert cores first
            if 'expert_strategy' in self.components:
                expert_result = await self.components['expert_strategy'].get_expanded_consensus_response(
                    prompt=prompt,
                    max_cores=2,
                    max_models_per_core=1
                )
                
                return IntegrationResult(
                    success=True,
                    result=expert_result,
                    integration_type="training_expert_cores",
                    processing_time=expert_result.get("processing_time", 0.0),
                    training_integration_used=True,
                    expert_cores_used=expert_result.get("cores_used", [])
                )
                
            # Try consensus
            elif 'consensus_strategy' in self.components:
                consensus_result = await self.components['consensus_strategy'].get_consensus_response(prompt)
                
                return IntegrationResult(
                    success=True,
                    result=consensus_result,
                    integration_type="training_consensus",
                    processing_time=consensus_result.get("processing_time", 0.0),
                    training_integration_used=True,
                    consensus_used=True
                )
                
            else:
                return IntegrationResult(
                    success=False,
                    result=None,
                    integration_type="training_only",
                    processing_time=0.0,
                    error_message="No training components available"
                )
                
        except Exception as e:
            logger.error(f"Error in training only processing: {e}")
            return IntegrationResult(
                success=False,
                result=None,
                integration_type="training_only",
                processing_time=0.0,
                error_message=str(e)
            )
            
    async def _process_hybrid(
        self, 
        request_data: Union[str, Dict[str, Any]], 
        request_type: str,
        context: Optional[Dict[str, Any]]
    ) -> IntegrationResult:
        """Process request with hybrid approach."""
        try:
            # Try training first
            training_result = await self._process_training_only(request_data, request_type, context)
            if training_result.success:
                return training_result
                
            # Fallback to core
            return await self._process_core_only(request_data, request_type, context)
            
        except Exception as e:
            logger.error(f"Error in hybrid processing: {e}")
            return await self._process_core_only(request_data, request_type, context)
            
    async def _process_minimal(
        self, 
        request_data: Union[str, Dict[str, Any]], 
        request_type: str,
        context: Optional[Dict[str, Any]]
    ) -> IntegrationResult:
        """Process request with minimal processing."""
        try:
            # Simple echo response
            result = {
                "input": request_data,
                "type": request_type,
                "context": context,
                "processed": True
            }
            
            return IntegrationResult(
                success=True,
                result=result,
                integration_type="minimal",
                processing_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Error in minimal processing: {e}")
            return IntegrationResult(
                success=False,
                result=None,
                integration_type="minimal",
                processing_time=0.0,
                error_message=str(e)
            )
            
    def _compute_cache_key(self, data: Union[str, Dict[str, Any]], request_type: str) -> str:
        """Compute cache key with proper encoding."""
        try:
            if isinstance(data, str):
                key_data = {"text": data, "type": request_type}
            else:
                key_data = {"data": data, "type": request_type}
                
            return json.dumps(key_data, sort_keys=True, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error computing cache key: {e}")
            return str(hash(str(data) + request_type))
            
    def _update_cache(self, key: str, result: IntegrationResult):
        """Update cache with size management."""
        if len(self.cache) >= self.config.cache_size:
            self._cleanup_cache()
        self.cache[key] = result
        
    def _cleanup_cache(self):
        """Clean up cache when it reaches the limit."""
        num_to_remove = len(self.cache) - self.config.cache_size + 100
        for _ in range(num_to_remove):
            if self.cache:
                self.cache.pop(next(iter(self.cache)))
                
    def _update_metrics(self, result: IntegrationResult, processing_time: float):
        """Update metrics with result information."""
        if result.success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
            
        if result.training_integration_used:
            self.metrics.training_integration_used += 1
            
        if result.expert_cores_used:
            self.metrics.expert_cores_used += 1
            
        if result.consensus_used:
            self.metrics.consensus_used += 1
            
        # Update average response time
        total_time = self.metrics.average_response_time * (self.metrics.successful_requests - 1) + processing_time
        self.metrics.average_response_time = total_time / self.metrics.successful_requests if self.metrics.successful_requests > 0 else 0
        
        self.metrics.last_update = time.time()
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return asdict(self.metrics)
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "max_size": self.config.cache_size,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "hit_rate": self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses) if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0
        }
        
    def get_integration_info(self) -> Dict[str, Any]:
        """Get integration information."""
        return {
            "integration_type": self.config.integration_type.value,
            "encoding": self.config.encoding,
            "components_available": list(self.components.keys()),
            "core_available": CORE_AVAILABLE,
            "training_available": TRAINING_AVAILABLE,
            "jax_available": JAX_AVAILABLE,
            "cache_enabled": self.config.enable_caching,
            "monitoring_enabled": self.config.enable_monitoring
        }
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of all components."""
        health_status = {
            "overall": "healthy",
            "components": {},
            "timestamp": time.time()
        }
        
        # Check core components
        if CORE_AVAILABLE:
            health_status["components"]["core"] = "available"
        else:
            health_status["components"]["core"] = "unavailable"
            health_status["overall"] = "degraded"
            
        # Check training components
        if TRAINING_AVAILABLE:
            health_status["components"]["training"] = "available"
        else:
            health_status["components"]["training"] = "unavailable"
            health_status["overall"] = "degraded"
            
        # Check JAX
        if JAX_AVAILABLE:
            health_status["components"]["jax"] = "available"
        else:
            health_status["components"]["jax"] = "unavailable"
            health_status["overall"] = "degraded"
            
        return health_status

# Factory functions
def create_ultra_core_integration(config: Optional[IntegrationConfig] = None) -> UltraCoreIntegration:
    """Create an ultra core integration instance."""
    return UltraCoreIntegration(config)

def create_minimal_integration() -> UltraCoreIntegration:
    """Create a minimal integration instance."""
    config = IntegrationConfig(integration_type=IntegrationType.MINIMAL)
    return UltraCoreIntegration(config)

def create_full_integration() -> UltraCoreIntegration:
    """Create a full integration instance."""
    config = IntegrationConfig(integration_type=IntegrationType.FULL)
    return UltraCoreIntegration(config)

# Legacy compatibility
def main():
    # Main function for this module.
    logger.info("Module ultra_core_integration.py starting")
    
    # Create integration instance
    integration = create_full_integration()
    
    # Perform health check
    health_status = asyncio.run(integration.health_check())
    logger.info(f"Health status: {health_status}")
    
    return True

if __name__ == "__main__":
    main()
