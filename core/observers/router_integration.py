"""
Router Integration for Observer Pattern
======================================

This module integrates the Observer pattern with the existing router system,
allowing for seamless dynamic expert activation through the routing process.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from .expert_activation_manager import ExpertActivationManager, ActivationStrategy
from .observers import (
    RequestPatternObserver,
    ComplexityObserver, 
    DomainSpecificObserver,
    PerformanceObserver,
    AdaptiveObserver
)
from .request_observer import (
    RequestEvent,
    RequestEventType,
    create_request_received_event,
    create_complexity_detected_event,
    create_domain_identified_event
)

# Import existing router components
try:
    from capibara.core.router import EnhancedRouter, RouterConfig, RoutingResult
    from capibara.core.routers.adaptive_router import AdaptiveRouter
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False
    logging.warning("Router components not available")

logger = logging.getLogger(__name__)


class RoutingMode(Enum):
    """Routing modes for observer integration."""
    TRADITIONAL = "traditional"      # Use traditional routing only
    OBSERVER_ENHANCED = "observer_enhanced"  # Traditional + observer activation
    OBSERVER_FIRST = "observer_first"        # Observer activation takes priority
    HYBRID = "hybrid"                        # Dynamic switching between modes


@dataclass
class DynamicRoutingDecision:
    """Enhanced routing decision with observer information."""
    # Traditional routing info
    success: bool
    selected_module: str
    confidence: float
    processing_time: float
    
    # Observer integration info
    observers_activated: List[str] = field(default_factory=list)
    experts_activated: List[str] = field(default_factory=list)
    observer_confidence: float = 0.0
    activation_strategy_used: str = ""
    
    # Combined results
    expert_results: Dict[str, Any] = field(default_factory=dict)
    routing_mode: RoutingMode = RoutingMode.TRADITIONAL
    
    # Performance metrics
    observer_processing_time: float = 0.0
    expert_processing_time: float = 0.0
    total_processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "success": self.success,
            "selected_module": self.selected_module,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "observers_activated": self.observers_activated,
            "experts_activated": self.experts_activated,
            "observer_confidence": self.observer_confidence,
            "activation_strategy": self.activation_strategy_used,
            "expert_results": self.expert_results,
            "routing_mode": self.routing_mode.value,
            "performance": {
                "observer_processing_time": self.observer_processing_time,
                "expert_processing_time": self.expert_processing_time,
                "total_processing_time": self.total_processing_time
            }
        }


class RouterObserverIntegration:
    """
    Integration layer between routers and the observer pattern.
    """
    
    def __init__(
        self,
        activation_strategy: ActivationStrategy = ActivationStrategy.THRESHOLD_BASED,
        routing_mode: RoutingMode = RoutingMode.OBSERVER_ENHANCED
    ):
        self.routing_mode = routing_mode
        self.activation_manager = ExpertActivationManager(activation_strategy)
        
        # Initialize default observers
        self._setup_default_observers()
        
        # Performance tracking
        self.integration_metrics = {
            "total_requests": 0,
            "observer_activations": 0,
            "expert_activations": 0,
            "routing_mode_usage": {mode.value: 0 for mode in RoutingMode},
            "average_processing_time": 0.0
        }
    
    def _setup_default_observers(self):
        """Setup default observers for the integration."""
        # Add pattern observer (highest priority)
        pattern_observer = RequestPatternObserver("PatternObserver", priority=1)
        self.activation_manager.add_observer(pattern_observer)
        
        # Add complexity observer
        complexity_observer = ComplexityObserver("ComplexityObserver", priority=2)
        self.activation_manager.add_observer(complexity_observer)
        
        # Add domain-specific observer
        domain_observer = DomainSpecificObserver("DomainObserver", priority=2)
        self.activation_manager.add_observer(domain_observer)
        
        # Add performance observer (lower priority)
        performance_observer = PerformanceObserver("PerformanceObserver", priority=3)
        self.activation_manager.add_observer(performance_observer)
        
        # Add adaptive observer (learns from patterns)
        adaptive_observer = AdaptiveObserver("AdaptiveObserver", priority=1)
        self.activation_manager.add_observer(adaptive_observer)
        
        logger.info("Setup default observers for router integration")
    
    async def enhanced_route_request(
        self,
        input_data: Union[str, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        traditional_router: Optional[Any] = None,
        request_id: Optional[str] = None
    ) -> DynamicRoutingDecision:
        """
        Enhanced routing that combines traditional routing with observer pattern.
        
        Args:
            input_data: Input data for routing
            context: Optional context information
            traditional_router: Optional traditional router instance
            request_id: Optional request identifier
            
        Returns:
            Dynamic routing decision with observer information
        """
        start_time = time.time()
        self.integration_metrics["total_requests"] += 1
        self.integration_metrics["routing_mode_usage"][self.routing_mode.value] += 1
        
        # Generate request ID if not provided
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000)}"
        
        # Extract text from input
        request_text = input_data if isinstance(input_data, str) else input_data.get("text", str(input_data))
        
        try:
            if self.routing_mode == RoutingMode.TRADITIONAL:
                return await self._traditional_routing(
                    input_data, context, traditional_router, request_id, start_time
                )
            elif self.routing_mode == RoutingMode.OBSERVER_FIRST:
                return await self._observer_first_routing(
                    request_text, request_id, context, traditional_router, start_time
                )
            elif self.routing_mode == RoutingMode.OBSERVER_ENHANCED:
                return await self._observer_enhanced_routing(
                    input_data, request_text, request_id, context, traditional_router, start_time
                )
            elif self.routing_mode == RoutingMode.HYBRID:
                return await self._hybrid_routing(
                    input_data, request_text, request_id, context, traditional_router, start_time
                )
            else:
                # Fallback to traditional
                return await self._traditional_routing(
                    input_data, context, traditional_router, request_id, start_time
                )
                
        except Exception as e:
            logger.error(f"Enhanced routing failed: {e}")
            return DynamicRoutingDecision(
                success=False,
                selected_module="error",
                confidence=0.0,
                processing_time=time.time() - start_time,
                routing_mode=self.routing_mode
            )
    
    async def _traditional_routing(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]],
        traditional_router: Optional[Any],
        request_id: str,
        start_time: float
    ) -> DynamicRoutingDecision:
        """Traditional routing without observer integration."""
        if traditional_router and hasattr(traditional_router, 'route_request'):
            try:
                result = await traditional_router.route_request(input_data, context)
                return DynamicRoutingDecision(
                    success=result.success,
                    selected_module=result.selected_module,
                    confidence=result.confidence,
                    processing_time=result.processing_time,
                    routing_mode=RoutingMode.TRADITIONAL,
                    total_processing_time=time.time() - start_time
                )
            except Exception as e:
                logger.error(f"Traditional routing failed: {e}")
        
        # Fallback simple routing
        return DynamicRoutingDecision(
            success=True,
            selected_module="default_module",
            confidence=0.5,
            processing_time=time.time() - start_time,
            routing_mode=RoutingMode.TRADITIONAL,
            total_processing_time=time.time() - start_time
        )
    
    async def _observer_first_routing(
        self,
        request_text: str,
        request_id: str,
        context: Optional[Dict[str, Any]],
        traditional_router: Optional[Any],
        start_time: float
    ) -> DynamicRoutingDecision:
        """Observer-first routing - observers take priority."""
        observer_start = time.time()
        
        # Process with observers
        observer_result = await self.activation_manager.process_request_with_observers(
            request_text, request_id, context
        )
        
        observer_time = time.time() - observer_start
        self.integration_metrics["observer_activations"] += 1
        
        if observer_result["experts_activated"]:
            self.integration_metrics["expert_activations"] += len(observer_result["experts_activated"])
            
            # Use observer results as primary
            return DynamicRoutingDecision(
                success=True,
                selected_module="observer_experts",
                confidence=self._calculate_combined_confidence(observer_result["results"]),
                processing_time=observer_result["processing_time"],
                observers_activated=list(self.activation_manager.observer_manager.observers),
                experts_activated=observer_result["experts_activated"],
                observer_confidence=self._calculate_combined_confidence(observer_result["results"]),
                activation_strategy_used=observer_result["activation_strategy"],
                expert_results=observer_result["results"],
                routing_mode=RoutingMode.OBSERVER_FIRST,
                observer_processing_time=observer_time,
                expert_processing_time=observer_result["processing_time"],
                total_processing_time=time.time() - start_time
            )
        else:
            # Fallback to traditional routing
            traditional_result = await self._traditional_routing(
                request_text, context, traditional_router, request_id, start_time
            )
            traditional_result.routing_mode = RoutingMode.OBSERVER_FIRST
            traditional_result.observer_processing_time = observer_time
            return traditional_result
    
    async def _observer_enhanced_routing(
        self,
        input_data: Any,
        request_text: str,
        request_id: str,
        context: Optional[Dict[str, Any]],
        traditional_router: Optional[Any],
        start_time: float
    ) -> DynamicRoutingDecision:
        """Observer-enhanced routing - combine traditional and observer results."""
        
        # Run traditional routing and observer activation in parallel
        traditional_task = asyncio.create_task(
            self._traditional_routing(input_data, context, traditional_router, request_id, start_time)
        )
        
        observer_task = asyncio.create_task(
            self.activation_manager.process_request_with_observers(request_text, request_id, context)
        )
        
        # Wait for both to complete
        traditional_result, observer_result = await asyncio.gather(
            traditional_task, observer_task, return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(traditional_result, Exception):
            logger.error(f"Traditional routing failed: {traditional_result}")
            traditional_result = DynamicRoutingDecision(
                success=False, selected_module="error", confidence=0.0, processing_time=0.0
            )
        
        if isinstance(observer_result, Exception):
            logger.error(f"Observer activation failed: {observer_result}")
            observer_result = {"experts_activated": [], "results": {}, "processing_time": 0.0}
        
        # Combine results
        if observer_result["experts_activated"]:
            self.integration_metrics["observer_activations"] += 1
            self.integration_metrics["expert_activations"] += len(observer_result["experts_activated"])
        
        # Create enhanced decision
        enhanced_decision = DynamicRoutingDecision(
            success=traditional_result.success or len(observer_result["experts_activated"]) > 0,
            selected_module=traditional_result.selected_module,
            confidence=max(traditional_result.confidence, self._calculate_combined_confidence(observer_result["results"])),
            processing_time=traditional_result.processing_time,
            observers_activated=[obs.name for obs in self.activation_manager.observer_manager.observers],
            experts_activated=observer_result["experts_activated"],
            observer_confidence=self._calculate_combined_confidence(observer_result["results"]),
            activation_strategy_used=observer_result.get("activation_strategy", ""),
            expert_results=observer_result["results"],
            routing_mode=RoutingMode.OBSERVER_ENHANCED,
            observer_processing_time=observer_result["processing_time"],
            expert_processing_time=observer_result["processing_time"],
            total_processing_time=time.time() - start_time
        )
        
        return enhanced_decision
    
    async def _hybrid_routing(
        self,
        input_data: Any,
        request_text: str,
        request_id: str,
        context: Optional[Dict[str, Any]],
        traditional_router: Optional[Any],
        start_time: float
    ) -> DynamicRoutingDecision:
        """Hybrid routing - dynamically choose best approach."""
        
        # Quick complexity analysis to decide approach
        complexity_score = self._quick_complexity_analysis(request_text)
        
        if complexity_score > 0.7:
            # High complexity - use observer-first approach
            return await self._observer_first_routing(
                request_text, request_id, context, traditional_router, start_time
            )
        elif complexity_score > 0.4:
            # Medium complexity - use observer-enhanced approach
            return await self._observer_enhanced_routing(
                input_data, request_text, request_id, context, traditional_router, start_time
            )
        else:
            # Low complexity - use traditional approach
            result = await self._traditional_routing(
                input_data, context, traditional_router, request_id, start_time
            )
            result.routing_mode = RoutingMode.HYBRID
            return result
    
    def _quick_complexity_analysis(self, text: str) -> float:
        """Quick complexity analysis for hybrid routing decisions."""
        if not text:
            return 0.0
        
        complexity_score = 0.0
        
        # Length-based complexity
        complexity_score += min(len(text) / 500, 0.3)
        
        # Question complexity
        question_indicators = text.count('?') + len([w for w in text.lower().split() 
                                                   if w in ['what', 'how', 'why', 'when', 'where']])
        complexity_score += min(question_indicators / 3, 0.2)
        
        # Technical indicators
        technical_words = ['algorithm', 'implementation', 'system', 'analysis', 'problem', 'error']
        tech_count = sum(1 for word in technical_words if word in text.lower())
        complexity_score += min(tech_count / 5, 0.3)
        
        # Mathematical indicators
        import re
        math_expressions = len(re.findall(r'\b\d+[\+\-\*/\^=]\d+\b', text))
        complexity_score += min(math_expressions / 3, 0.2)
        
        return min(complexity_score, 1.0)
    
    def _calculate_combined_confidence(self, expert_results: Dict[str, Any]) -> float:
        """Calculate combined confidence from expert results."""
        if not expert_results:
            return 0.0
        
        confidences = []
        for result_data in expert_results.values():
            if isinstance(result_data, dict) and "confidence" in result_data:
                confidences.append(result_data["confidence"])
        
        if not confidences:
            return 0.0
        
        # Use weighted average with higher weight for higher confidences
        weighted_sum = sum(conf * conf for conf in confidences)  # Weight by confidence itself
        weight_sum = sum(confidences)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        manager_stats = self.activation_manager.get_comprehensive_statistics()
        
        # Calculate average processing time
        total_requests = self.integration_metrics["total_requests"]
        if total_requests > 0:
            self.integration_metrics["average_processing_time"] = (
                self.integration_metrics.get("total_processing_time", 0.0) / total_requests
            )
        
        return {
            "integration_metrics": self.integration_metrics,
            "routing_mode": self.routing_mode.value,
            "activation_manager_stats": manager_stats,
            "observer_count": len(self.activation_manager.observer_manager.observers),
            "expert_pool_size": len(self.activation_manager.expert_pool.available_experts)
        }


class ObserverAwareRouter:
    """
    Router that is aware of and integrates with the observer pattern.
    """
    
    def __init__(
        self,
        router_config: Optional[Any] = None,
        activation_strategy: ActivationStrategy = ActivationStrategy.THRESHOLD_BASED,
        routing_mode: RoutingMode = RoutingMode.OBSERVER_ENHANCED
    ):
        # Initialize traditional router if available
        self.traditional_router = None
        if ROUTER_AVAILABLE and router_config:
            try:
                self.traditional_router = EnhancedRouter(router_config)
            except Exception as e:
                logger.warning(f"Could not initialize traditional router: {e}")
        
        # Initialize observer integration
        self.observer_integration = RouterObserverIntegration(
            activation_strategy, routing_mode
        )
        
        # Router configuration
        self.config = router_config
        self.routing_mode = routing_mode
        
        logger.info(f"ObserverAwareRouter initialized with mode: {routing_mode.value}")
    
    async def route_request(
        self,
        input_data: Union[str, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        training_mode: bool = False,
        request_id: Optional[str] = None
    ) -> DynamicRoutingDecision:
        """
        Route request using observer-aware routing.
        
        Args:
            input_data: Input data for routing
            context: Optional context information
            use_cache: Whether to use caching (passed to traditional router)
            training_mode: Whether in training mode (passed to traditional router)
            request_id: Optional request identifier
            
        Returns:
            Dynamic routing decision
        """
        return await self.observer_integration.enhanced_route_request(
            input_data=input_data,
            context=context,
            traditional_router=self.traditional_router,
            request_id=request_id
        )
    
    def add_observer(self, observer):
        """Add a custom observer to the routing system."""
        self.observer_integration.activation_manager.add_observer(observer)
    
    def register_expert(self, name: str, expert_class, **kwargs):
        """Register a custom expert for dynamic activation."""
        self.observer_integration.activation_manager.register_expert(name, expert_class, **kwargs)
    
    def set_routing_mode(self, mode: RoutingMode):
        """Change the routing mode."""
        self.routing_mode = mode
        self.observer_integration.routing_mode = mode
        logger.info(f"Routing mode changed to: {mode.value}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        return self.observer_integration.get_integration_statistics()
    
    def provide_feedback(self, request_id: str, expert_feedback: Dict[str, bool]):
        """Provide feedback for adaptive learning."""
        # Update adaptive observer with feedback
        for observer in self.observer_integration.activation_manager.observer_manager.observers:
            if hasattr(observer, 'learn_from_feedback'):
                for expert_name, success in expert_feedback.items():
                    observer.learn_from_feedback(expert_name, [], success)
        
        # Update activation manager adaptive weights
        self.observer_integration.activation_manager.update_adaptive_weights(
            request_id, expert_feedback
        )


# Factory functions for easy setup

def create_observer_aware_router(
    routing_mode: RoutingMode = RoutingMode.OBSERVER_ENHANCED,
    activation_strategy: ActivationStrategy = ActivationStrategy.THRESHOLD_BASED,
    router_config: Optional[Any] = None
) -> ObserverAwareRouter:
    """
    Factory function to create an observer-aware router.
    
    Args:
        routing_mode: Routing mode to use
        activation_strategy: Expert activation strategy
        router_config: Optional traditional router configuration
        
    Returns:
        Configured ObserverAwareRouter instance
    """
    return ObserverAwareRouter(
        router_config=router_config,
        activation_strategy=activation_strategy,
        routing_mode=routing_mode
    )


def create_simple_observer_router(request_patterns: Optional[Dict[str, List[str]]] = None) -> ObserverAwareRouter:
    """
    Create a simple observer router with basic pattern matching.
    
    Args:
        request_patterns: Optional custom patterns for expert activation
        
    Returns:
        Configured ObserverAwareRouter with simple setup
    """
    router = create_observer_aware_router(
        routing_mode=RoutingMode.OBSERVER_FIRST,
        activation_strategy=ActivationStrategy.THRESHOLD_BASED
    )
    
    # Add custom patterns if provided
    if request_patterns:
        pattern_observer = RequestPatternObserver("CustomPatternObserver")
        # Update patterns (this would require extending the observer)
        router.add_observer(pattern_observer)
    
    return router