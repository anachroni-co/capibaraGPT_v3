#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Training Bridge - CapibaraGPT

Enhanced bridge module for seamless integration between core and training modules.
Provides proper encoding handling, comprehensive integration capabilities, and
optimized performance for the complete system.
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

# Core imports with proper error handling
try:
    from capibara.core.router import EnhancedRouter, RouterConfig, RouterType
    from capibara.core.ultra_core_integration import UltraCoreIntegration, IntegrationConfig
    from capibara.core.routing import Router as CoreRouter
    CORE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Core modules not available: {e}")
    CORE_AVAILABLE = False

# Training imports with proper error handling
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

# JAX imports with proper error handling
try:
    import jax
    import jax.numpy as jnp
    from capibara.jax import nn
    JAX_AVAILABLE = True
except ImportError as e:
    logging.warning(f"JAX not available: {e}")
    JAX_AVAILABLE = False

logger = logging.getLogger(__name__)

class BridgeMode(Enum):
    """Modes of operation for the bridge."""
    FULL_INTEGRATION = "full_integration"
    CORE_TO_TRAINING = "core_to_training"
    TRAINING_TO_CORE = "training_to_core"
    BIDIRECTIONAL = "bidirectional"
    MONITORING_ONLY = "monitoring_only"

@dataclass
class BridgeConfig:
    """Configurestion for the core training bridge."""
    mode: BridgeMode = BridgeMode.FULL_INTEGRATION
    encoding: str = "utf-8"
    enable_caching: bool = True
    cache_size: int = 1000
    enable_monitoring: bool = True
    enable_metrics: bool = True
    enable_health_checks: bool = True
    log_level: str = "INFO"
    config_path: Optional[str] = None
    max_retries: int = 3
    timeout: float = 30.0

@dataclass
class BridgeMetrics:
    """Metrics for bridge performance."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    core_to_training_calls: int = 0
    training_to_core_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_response_time: float = 0.0
    last_update: float = 0.0
    health_check_count: int = 0

@dataclass
class BridgeResult:
    """Result of bridge operation."""
    success: bool
    result: Any
    mode: str
    processing_time: float
    cache_hit: bool = False
    core_integration_used: bool = False
    training_integration_used: bool = False
    error_message: Optional[str] = None
    retry_count: int = 0

class CoreTrainingBridge:
    """
    Core Training Bridge - Seamless integration between core and training modules.
    
    This class provides:
    1. Bidirectional communication between core and training modules
    2. Proper encoding handling for all data transfers
    3. Comprehensive caching and performance optimization
    4. Health monitoring and metrics collection
    5. Fallback mechanisms and error handling
    6. Async/await support for high-performance operations
    """
    
    def __init__(self, config: Optional[BridgeConfig] = None):
        """Initialize the core training bridge."""
        self.config = config or BridgeConfig()
        self._setup_logging()
        self._initialize_components()
        self._setup_metrics()
        self._setup_cache()
        
    def _setup_logging(self):
        """Setup logging with proper encoding."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            encoding=self.config.encoding
        )
        logger.info(f"Core Training Bridge initialized with mode: {self.config.mode.value}")
        
    def _initialize_components(self):
        """Initialize all bridge components."""
        self.components = {}
        self.health_status = {}
        
        # Initialize core components
        if CORE_AVAILABLE:
            self._initialize_core_components()
            
        # Initialize training components
        if TRAINING_AVAILABLE:
            self._initialize_training_components()
            
        # Initialize JAX components
        if JAX_AVAILABLE:
            self._initialize_jax_components()
            
        logger.info(f"Bridge components initialized: {list(self.components.keys())}")
        
    def _initialize_core_components(self):
        """Initialize core components."""
        try:
            # Enhanced router
            router_config = RouterConfig(
                router_type=RouterType.HYBRID,
                use_training_integration=True,
                expert_cores_enabled=True,
                consensus_enabled=True,
                cache_size=self.config.cache_size,
                encoding=self.config.encoding
            )
            self.components['enhanced_router'] = EnhancedRouter(router_config)
            
            # Ultra core integration - avoid circular import
            self.components['ultra_integration'] = None  # Will be initialized later if needed
            
            # Core router
            self.components['core_router'] = CoreRouter()
            
            logger.info("Core components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing core components: {e}")
            
    def _initialize_training_components(self):
        """Initialize training components."""
        try:
            # Expert cores strategy
            self.components['expert_strategy'] = ExpandedExpertCoresStrategy({})
            
            # Consensus strategy
            self.components['consensus_strategy'] = HuggingFaceConsensusStrategy({})
            
            # Cascade training integration
            self.components['cascade_integration'] = CascadeTrainingIntegration()
            
            logger.info("Training components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing training components: {e}")
            
    def _initialize_jax_components(self):
        """Initialize JAX components."""
        try:
            # JAX configuration
            jax.config.update('jax_enable_x64', True)
            self.components['jax_available'] = True
            logger.info("JAX components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing JAX components: {e}")
            self.components['jax_available'] = False
            
    def _setup_metrics(self):
        """Setup metrics tracking."""
        self.metrics = BridgeMetrics()
        self.metrics.last_update = time.time()
        
    def _setup_cache(self):
        """Setup caching system."""
        self.cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "size": 0,
            "max_size": self.config.cache_size
        }
        
    async def process_request(
        self, 
        request_data: Union[str, Dict[str, Any]], 
        request_type: str = "general",
        direction: str = "bidirectional",
        use_cache: bool = None,
        context: Optional[Dict[str, Any]] = None
    ) -> BridgeResult:
        """Process a request through the bridge."""
        start_time = time.time()
        
        try:
            # Update metrics
            self.metrics.total_requests += 1
            
            # Determine cache usage
            use_cache = use_cache if use_cache is not None else self.config.enable_caching
            
            # Check cache
            cache_key = self._compute_cache_key(request_data, request_type, direction)
            if use_cache and cache_key in self.cache:
                self.metrics.cache_hits += 1
                self.cache_stats["hits"] += 1
                cached_result = self.cache[cache_key]
                cached_result.cache_hit = True
                return cached_result
                
            self.metrics.cache_misses += 1
            self.cache_stats["misses"] += 1
            
            # Process based on bridge mode
            if self.config.mode == BridgeMode.FULL_INTEGRATION:
                result = await self._process_full_integration(request_data, request_type, direction, context)
            elif self.config.mode == BridgeMode.CORE_TO_TRAINING:
                result = await self._process_core_to_training(request_data, request_type, context)
            elif self.config.mode == BridgeMode.TRAINING_TO_CORE:
                result = await self._process_training_to_core(request_data, request_type, context)
            elif self.config.mode == BridgeMode.BIDIRECTIONAL:
                result = await self._process_bidirectional(request_data, request_type, context)
            else:  # MONITORING_ONLY
                result = await self._process_monitoring_only(request_data, request_type, context)
                
            # Update cache
            if use_cache:
                self._update_cache(cache_key, result)
                
            # Update metrics
            self._update_metrics(result, time.time() - start_time, direction)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            self.metrics.failed_requests += 1
            
            return BridgeResult(
                success=False,
                result=None,
                mode=self.config.mode.value,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
            
    async def _process_full_integration(
        self, 
        request_data: Union[str, Dict[str, Any]], 
        request_type: str,
        direction: str,
        context: Optional[Dict[str, Any]]
    ) -> BridgeResult:
        """Process request with full integration."""
        try:
            # Use enhanced router directly for full integration
            if 'enhanced_router' in self.components:
                router_result = await self.components['enhanced_router'].route_request(
                    request_data, context, use_cache=True, training_mode=False
                )
                
                return BridgeResult(
                    success=router_result.success,
                    result=router_result,
                    mode="full_integration",
                    processing_time=router_result.processing_time,
                    core_integration_used=True,
                    training_integration_used=router_result.training_integration_used
                )
                
            # Fallback to bidirectional processing
            return await self._process_bidirectional(request_data, request_type, context)
            
        except Exception as e:
            logger.error(f"Error in full integration: {e}")
            return await self._process_bidirectional(request_data, request_type, context)
            
    async def _process_core_to_training(
        self, 
        request_data: Union[str, Dict[str, Any]], 
        request_type: str,
        context: Optional[Dict[str, Any]]
    ) -> BridgeResult:
        """Process request from core to training."""
        try:
            self.metrics.core_to_training_calls += 1
            
            # Use enhanced router to route to training components
            if 'enhanced_router' in self.components:
                router_result = await self.components['enhanced_router'].route_request(
                    request_data, context, use_cache=True, training_mode=True
                )
                
                return BridgeResult(
                    success=router_result.success,
                    result=router_result,
                    mode="core_to_training",
                    processing_time=router_result.processing_time,
                    core_integration_used=True,
                    training_integration_used=router_result.training_integration_used
                )
                
            # Fallback to direct training call
            return await self._call_training_directly(request_data, request_type, context)
            
        except Exception as e:
            logger.error(f"Error in core to training processing: {e}")
            return BridgeResult(
                success=False,
                result=None,
                mode="core_to_training",
                processing_time=0.0,
                error_message=str(e)
            )
            
    async def _process_training_to_core(
        self, 
        request_data: Union[str, Dict[str, Any]], 
        request_type: str,
        context: Optional[Dict[str, Any]]
    ) -> BridgeResult:
        """Process request from training to core."""
        try:
            self.metrics.training_to_core_calls += 1
            
            # Use core router to process training results
            if 'core_router' in self.components:
                result = self.components['core_router'].route(request_type, **context or {})
                
                return BridgeResult(
                    success=True,
                    result=result,
                    mode="training_to_core",
                    processing_time=0.0,
                    core_integration_used=True
                )
                
            # Fallback to minimal processing
            return BridgeResult(
                success=True,
                result={"processed": True, "data": request_data},
                mode="training_to_core",
                processing_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Error in training to core processing: {e}")
            return BridgeResult(
                success=False,
                result=None,
                mode="training_to_core",
                processing_time=0.0,
                error_message=str(e)
            )
            
    async def _process_bidirectional(
        self, 
        request_data: Union[str, Dict[str, Any]], 
        request_type: str,
        context: Optional[Dict[str, Any]]
    ) -> BridgeResult:
        """Process request bidirectionally."""
        try:
            # First, try core to training
            core_to_training_result = await self._process_core_to_training(request_data, request_type, context)
            
            if core_to_training_result.success:
                # Then, try training to core with the result
                training_to_core_result = await self._process_training_to_core(
                    core_to_training_result.result, f"{request_type}_processed", context
                )
                
                return BridgeResult(
                    success=training_to_core_result.success,
                    result=training_to_core_result.result,
                    mode="bidirectional",
                    processing_time=core_to_training_result.processing_time + training_to_core_result.processing_time,
                    core_integration_used=True,
                    training_integration_used=True
                )
                
            return core_to_training_result
            
        except Exception as e:
            logger.error(f"Error in bidirectional processing: {e}")
            return BridgeResult(
                success=False,
                result=None,
                mode="bidirectional",
                processing_time=0.0,
                error_message=str(e)
            )
            
    async def _process_monitoring_only(
        self, 
        request_data: Union[str, Dict[str, Any]], 
        request_type: str,
        context: Optional[Dict[str, Any]]
    ) -> BridgeResult:
        """Process request with monitoring only."""
        try:
            # Just monitor and return status
            result = {
                "monitored": True,
                "request_type": request_type,
                "data_size": len(str(request_data)),
                "timestamp": time.time(),
                "bridge_status": self.get_bridge_status()
            }
            
            return BridgeResult(
                success=True,
                result=result,
                mode="monitoring_only",
                processing_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Error in monitoring only processing: {e}")
            return BridgeResult(
                success=False,
                result=None,
                mode="monitoring_only",
                processing_time=0.0,
                error_message=str(e)
            )
            
    async def _call_training_directly(
        self, 
        request_data: Union[str, Dict[str, Any]], 
        request_type: str,
        context: Optional[Dict[str, Any]]
    ) -> BridgeResult:
        """Call training components directly."""
        try:
            if isinstance(request_data, str):
                prompt = request_data
            else:
                prompt = request_data.get("text", str(request_data))
                
            # Try expert cores strategy
            if 'expert_strategy' in self.components:
                expert_result = await self.components['expert_strategy'].get_expanded_consensus_response(
                    prompt=prompt,
                    max_cores=2,
                    max_models_per_core=1
                )
                
                return BridgeResult(
                    success=True,
                    result=expert_result,
                    mode="direct_training",
                    processing_time=expert_result.get("processing_time", 0.0),
                    training_integration_used=True
                )
                
            # Try consensus strategy
            elif 'consensus_strategy' in self.components:
                consensus_result = await self.components['consensus_strategy'].get_consensus_response(prompt)
                
                return BridgeResult(
                    success=True,
                    result=consensus_result,
                    mode="direct_training",
                    processing_time=consensus_result.get("processing_time", 0.0),
                    training_integration_used=True
                )
                
            else:
                return BridgeResult(
                    success=False,
                    result=None,
                    mode="direct_training",
                    processing_time=0.0,
                    error_message="No training components available"
                )
                
        except Exception as e:
            logger.error(f"Error calling training directly: {e}")
            return BridgeResult(
                success=False,
                result=None,
                mode="direct_training",
                processing_time=0.0,
                error_message=str(e)
            )
            
    def _compute_cache_key(self, data: Union[str, Dict[str, Any]], request_type: str, direction: str) -> str:
        """Compute cache key with proper encoding."""
        try:
            if isinstance(data, str):
                key_data = {"text": data, "type": request_type, "direction": direction}
            else:
                key_data = {"data": data, "type": request_type, "direction": direction}
                
            return json.dumps(key_data, sort_keys=True, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error computing cache key: {e}")
            return str(hash(str(data) + request_type + direction))
            
    def _update_cache(self, key: str, result: BridgeResult):
        """Update cache with size management."""
        if len(self.cache) >= self.config.cache_size:
            self._cleanup_cache()
        self.cache[key] = result
        self.cache_stats["size"] = len(self.cache)
        
    def _cleanup_cache(self):
        """Clean up cache when it reaches the limit."""
        num_to_remove = len(self.cache) - self.config.cache_size + 100
        for _ in range(num_to_remove):
            if self.cache:
                self.cache.pop(next(iter(self.cache)))
                
    def _update_metrics(self, result: BridgeResult, processing_time: float, direction: str):
        """Update metrics with result information."""
        if result.success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
            
        # Update average response time
        total_time = self.metrics.average_response_time * (self.metrics.successful_requests - 1) + processing_time
        self.metrics.average_response_time = total_time / self.metrics.successful_requests if self.metrics.successful_requests > 0 else 0
        
        self.metrics.last_update = time.time()
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return asdict(self.metrics)
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache_stats.copy()
        
    def get_bridge_status(self) -> Dict[str, Any]:
        """Get bridge status information."""
        return {
            "mode": self.config.mode.value,
            "encoding": self.config.encoding,
            "components_available": list(self.components.keys()),
            "core_available": CORE_AVAILABLE,
            "training_available": TRAINING_AVAILABLE,
            "jax_available": JAX_AVAILABLE,
            "cache_enabled": self.config.enable_caching,
            "monitoring_enabled": self.config.enable_monitoring,
            "health_checks_enabled": self.config.enable_health_checks
        }
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of all bridge components."""
        if not self.config.enable_health_checks:
            return {"status": "health_checks_disabled"}
            
        self.metrics.health_check_count += 1
        
        health_status = {
            "overall": "healthy",
            "components": {},
            "timestamp": time.time(),
            "bridge_mode": self.config.mode.value
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
            
        # Check cache
        if self.config.enable_caching:
            health_status["components"]["cache"] = "enabled"
        else:
            health_status["components"]["cache"] = "disabled"
            
        self.health_status = health_status
        return health_status

# Factory functions
def create_core_training_bridge(config: Optional[BridgeConfig] = None) -> CoreTrainingBridge:
    """Creates a core training bridge instance."""
    return CoreTrainingBridge(config)

def create_full_integration_bridge() -> CoreTrainingBridge:
    """Creates a full integration bridge instance."""
    config = BridgeConfig(mode=BridgeMode.FULL_INTEGRATION)
    return CoreTrainingBridge(config)

def create_bidirectional_bridge() -> CoreTrainingBridge:
    """Creates a bidirectional bridge instance."""
    config = BridgeConfig(mode=BridgeMode.BIDIRECTIONAL)
    return CoreTrainingBridge(config)

def create_monitoring_bridge() -> CoreTrainingBridge:
    """Creates a monitoring only bridge instance."""
    config = BridgeConfig(mode=BridgeMode.MONITORING_ONLY)
    return CoreTrainingBridge(config)

# Main function for testing
async def main():
    """Main function for testing the bridge."""
    logger.info("Core Training Bridge starting")
    
    # Create bridge instance
    bridge = create_full_integration_bridge()
    
    # Perform health check
    health_status = await bridge.health_check()
    logger.info(f"Health status: {health_status}")
    
    # Test processing
    test_data = "Test request for bridge processing"
    result = await bridge.process_request(test_data, "test", "bidirectional")
    logger.info(f"Test result: {result}")
    
    return True

if __name__ == "__main__":
    asyncio.run(main())