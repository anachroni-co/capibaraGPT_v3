"""
Expert Activation Manager
========================

This module manages the lifecycle and activation of experts based on observer
events. It provides dynamic expert pool management, activation strategies,
and performance monitoring for the observer pattern implementation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Callable, Union, Type
from collections import defaultdict, deque

from .request_observer import (
    RequestEvent, 
    ExpertActivationEvent, 
    RequestObserver, 
    ObserverManager
)

# Import expert interfaces
try:
    from capibara.interfaces.isub_models import ICounterfactualExpert, ExpertContext, ExpertResult
    from capibara.sub_models.csa_expert import CSAExpert
    EXPERTS_AVAILABLE = True
except ImportError:
    EXPERTS_AVAILABLE = False
    logging.warning("Expert interfaces not available")

logger = logging.getLogger(__name__)


class ExpertState(Enum):
    """States of expert lifecycle."""
    INACTIVE = "inactive"
    INITIALIZING = "initializing" 
    ACTIVE = "active"
    PROCESSING = "processing"
    COOLDOWN = "cooldown"
    ERROR = "error"
    TERMINATED = "terminated"


class ActivationStrategy(Enum):
    """Strategies for expert activation."""
    IMMEDIATE = "immediate"           # Activate immediately when triggered
    THRESHOLD_BASED = "threshold"     # Activate when confidence exceeds threshold
    CONSENSUS_REQUIRED = "consensus"  # Require multiple observers to agree
    LOAD_BALANCED = "load_balanced"   # Consider current system load
    ADAPTIVE = "adaptive"             # Learn optimal activation patterns


@dataclass
class ExpertMetrics:
    """Metrics for expert performance and usage."""
    total_activations: int = 0
    successful_completions: int = 0
    failed_completions: int = 0
    average_processing_time: float = 0.0
    average_confidence: float = 0.0
    last_activation_time: Optional[float] = None
    total_processing_time: float = 0.0
    activation_reasons: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successful_completions + self.failed_completions
        return self.successful_completions / total if total > 0 else 0.0
    
    def update_activation(self, reason: str):
        """Update metrics for new activation."""
        self.total_activations += 1
        self.last_activation_time = time.time()
        self.activation_reasons[reason] = self.activation_reasons.get(reason, 0) + 1
    
    def update_completion(self, success: bool, processing_time: float, confidence: float):
        """Update metrics for completion."""
        if success:
            self.successful_completions += 1
        else:
            self.failed_completions += 1
        
        # Update averages
        total_completions = self.successful_completions + self.failed_completions
        self.average_processing_time = (
            (self.average_processing_time * (total_completions - 1) + processing_time) 
            / total_completions
        )
        self.average_confidence = (
            (self.average_confidence * (total_completions - 1) + confidence)
            / total_completions
        )
        self.total_processing_time += processing_time


@dataclass
class ExpertInstance:
    """Represents an active expert instance."""
    name: str
    expert_class: Type
    instance: Any
    state: ExpertState = ExpertState.INACTIVE
    activation_time: Optional[float] = None
    last_used_time: Optional[float] = None
    current_load: int = 0
    max_concurrent_requests: int = 5
    cooldown_duration: float = 60.0  # seconds
    metrics: ExpertMetrics = field(default_factory=ExpertMetrics)
    
    def can_process_request(self) -> bool:
        """Check if expert can process a new request."""
        return (
            self.state in [ExpertState.ACTIVE, ExpertState.PROCESSING] and
            self.current_load < self.max_concurrent_requests
        )
    
    def is_in_cooldown(self) -> bool:
        """Check if expert is in cooldown period."""
        if self.state != ExpertState.COOLDOWN:
            return False
        if self.last_used_time is None:
            return False
        return (time.time() - self.last_used_time) < self.cooldown_duration


class ExpertPool:
    """
    Pool of available experts that can be dynamically activated.
    """
    
    def __init__(self):
        self.available_experts: Dict[str, Type] = {}
        self.active_instances: Dict[str, ExpertInstance] = {}
        self.activation_queue: deque = deque()
        self.global_metrics = {
            "total_activations": 0,
            "total_requests_processed": 0,
            "average_pool_utilization": 0.0,
            "peak_concurrent_experts": 0
        }
    
    def register_expert(self, name: str, expert_class: Type, 
                       max_concurrent: int = 5, cooldown_duration: float = 60.0):
        """Register an expert class for dynamic activation."""
        self.available_experts[name] = expert_class
        logger.info(f"Registered expert: {name} with max_concurrent={max_concurrent}")
    
    async def activate_expert(self, name: str, activation_event: ExpertActivationEvent) -> bool:
        """
        Activate an expert instance.
        
        Args:
            name: Name of the expert to activate
            activation_event: Event that triggered the activation
            
        Returns:
            True if activation successful, False otherwise
        """
        if name not in self.available_experts:
            logger.error(f"Expert {name} not registered in pool")
            return False
        
        # Check if already active
        if name in self.active_instances:
            instance = self.active_instances[name]
            if instance.state == ExpertState.ACTIVE:
                logger.debug(f"Expert {name} already active")
                return True
            elif instance.is_in_cooldown():
                logger.debug(f"Expert {name} in cooldown, skipping activation")
                return False
        
        try:
            # Create new instance
            expert_class = self.available_experts[name]
            expert_instance = expert_class()
            
            # Create expert instance wrapper
            instance = ExpertInstance(
                name=name,
                expert_class=expert_class,
                instance=expert_instance,
                state=ExpertState.INITIALIZING,
                activation_time=time.time()
            )
            
            # Initialize expert if needed
            if hasattr(expert_instance, 'initialize'):
                await expert_instance.initialize()
            
            # Mark as active
            instance.state = ExpertState.ACTIVE
            self.active_instances[name] = instance
            
            # Update metrics
            instance.metrics.update_activation(activation_event.activation_reason)
            self.global_metrics["total_activations"] += 1
            
            # Update peak concurrent experts
            current_active = len([i for i in self.active_instances.values() 
                                if i.state == ExpertState.ACTIVE])
            self.global_metrics["peak_concurrent_experts"] = max(
                self.global_metrics["peak_concurrent_experts"], 
                current_active
            )
            
            logger.info(f"Successfully activated expert: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate expert {name}: {e}")
            if name in self.active_instances:
                self.active_instances[name].state = ExpertState.ERROR
            return False
    
    async def deactivate_expert(self, name: str, force: bool = False):
        """
        Deactivate an expert instance.
        
        Args:
            name: Name of the expert to deactivate
            force: Force deactivation even if processing requests
        """
        if name not in self.active_instances:
            return
        
        instance = self.active_instances[name]
        
        # Check if expert is currently processing
        if instance.current_load > 0 and not force:
            logger.info(f"Expert {name} has active requests, moving to cooldown")
            instance.state = ExpertState.COOLDOWN
            instance.last_used_time = time.time()
            return
        
        # Cleanup expert
        try:
            if hasattr(instance.instance, 'cleanup'):
                await instance.instance.cleanup()
        except Exception as e:
            logger.error(f"Error during expert {name} cleanup: {e}")
        
        # Remove from active instances
        instance.state = ExpertState.TERMINATED
        del self.active_instances[name]
        
        logger.info(f"Deactivated expert: {name}")
    
    async def process_with_expert(self, expert_name: str, context: Any) -> Optional[Any]:
        """
        Process a request with a specific expert.
        
        Args:
            expert_name: Name of the expert to use
            context: Context or data to process
            
        Returns:
            Result from expert processing, or None if failed
        """
        if expert_name not in self.active_instances:
            logger.error(f"Expert {expert_name} not active")
            return None
        
        instance = self.active_instances[expert_name]
        
        if not instance.can_process_request():
            logger.warning(f"Expert {expert_name} cannot process request (load: {instance.current_load})")
            return None
        
        # Update state and load
        instance.state = ExpertState.PROCESSING
        instance.current_load += 1
        instance.last_used_time = time.time()
        
        start_time = time.time()
        try:
            # Process with expert
            if hasattr(instance.instance, 'process'):
                result = await instance.instance.process(context)
            elif hasattr(instance.instance, '__call__'):
                result = await instance.instance(context)
            else:
                logger.error(f"Expert {expert_name} has no process method")
                return None
            
            # Update metrics for successful completion
            processing_time = time.time() - start_time
            confidence = getattr(result, 'confidence', 0.5) if result else 0.0
            instance.metrics.update_completion(True, processing_time, confidence)
            
            self.global_metrics["total_requests_processed"] += 1
            
            logger.debug(f"Expert {expert_name} processed request in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            # Update metrics for failed completion
            processing_time = time.time() - start_time
            instance.metrics.update_completion(False, processing_time, 0.0)
            
            logger.error(f"Expert {expert_name} processing failed: {e}")
            return None
            
        finally:
            # Update state and load
            instance.current_load -= 1
            if instance.current_load == 0:
                instance.state = ExpertState.ACTIVE
    
    def get_active_experts(self) -> List[str]:
        """Get list of currently active expert names."""
        return [
            name for name, instance in self.active_instances.items()
            if instance.state in [ExpertState.ACTIVE, ExpertState.PROCESSING]
        ]
    
    def get_expert_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a specific expert."""
        if name not in self.active_instances:
            return None
        
        instance = self.active_instances[name]
        return {
            "name": name,
            "state": instance.state.value,
            "current_load": instance.current_load,
            "max_concurrent": instance.max_concurrent_requests,
            "activation_time": instance.activation_time,
            "last_used_time": instance.last_used_time,
            "is_in_cooldown": instance.is_in_cooldown(),
            "metrics": {
                "total_activations": instance.metrics.total_activations,
                "success_rate": instance.metrics.success_rate,
                "average_processing_time": instance.metrics.average_processing_time,
                "average_confidence": instance.metrics.average_confidence,
                "activation_reasons": instance.metrics.activation_reasons
            }
        }
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        active_count = len([i for i in self.active_instances.values() 
                          if i.state == ExpertState.ACTIVE])
        processing_count = len([i for i in self.active_instances.values() 
                              if i.state == ExpertState.PROCESSING])
        
        # Calculate utilization
        total_capacity = sum(i.max_concurrent_requests for i in self.active_instances.values())
        current_load = sum(i.current_load for i in self.active_instances.values())
        utilization = current_load / total_capacity if total_capacity > 0 else 0.0
        
        return {
            "registered_experts": len(self.available_experts),
            "active_instances": len(self.active_instances),
            "experts_processing": processing_count,
            "experts_ready": active_count,
            "current_utilization": utilization,
            "total_capacity": total_capacity,
            "current_load": current_load,
            "global_metrics": self.global_metrics
        }


class ExpertActivationManager:
    """
    Main manager for coordinating observer events with expert activation.
    """
    
    def __init__(self, activation_strategy: ActivationStrategy = ActivationStrategy.THRESHOLD_BASED):
        self.observer_manager = ObserverManager()
        self.expert_pool = ExpertPool()
        self.activation_strategy = activation_strategy
        
        # Strategy parameters
        self.strategy_config = {
            "confidence_threshold": 0.6,
            "consensus_required_votes": 2,
            "max_concurrent_activations": 3,
            "load_balance_threshold": 0.8
        }
        
        # Adaptive learning parameters
        self.adaptive_weights = defaultdict(float)
        self.activation_history = deque(maxlen=1000)
        
        # Register default experts if available
        if EXPERTS_AVAILABLE:
            self._register_default_experts()
    
    def _register_default_experts(self):
        """Register default experts that are available."""
        try:
            self.expert_pool.register_expert("CSA", CSAExpert, max_concurrent=3)
            logger.info("Registered default CSA expert")
        except Exception as e:
            logger.warning(f"Could not register default experts: {e}")
    
    def add_observer(self, observer: RequestObserver):
        """Add an observer to the manager."""
        self.observer_manager.add_observer(observer)
    
    def register_expert(self, name: str, expert_class: Type, **kwargs):
        """Register an expert for dynamic activation."""
        self.expert_pool.register_expert(name, expert_class, **kwargs)
    
    async def process_request_with_observers(
        self, 
        request_text: str, 
        request_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a request using the observer pattern for dynamic expert activation.
        
        Args:
            request_text: The text of the request
            request_id: Unique identifier for the request
            metadata: Optional metadata about the request
            
        Returns:
            Dictionary containing results from activated experts
        """
        start_time = time.time()
        
        # Create request event
        from .request_observer import create_request_received_event
        event = create_request_received_event(request_id, request_text, metadata)
        
        # Notify observers and get activation events
        activation_events = await self.observer_manager.notify_observers(event)
        
        if not activation_events:
            logger.info(f"No experts activated for request {request_id}")
            return {
                "request_id": request_id,
                "experts_activated": [],
                "results": {},
                "processing_time": time.time() - start_time
            }
        
        # Apply activation strategy
        approved_activations = await self._apply_activation_strategy(activation_events)
        
        # Activate experts and process request
        results = {}
        activated_experts = []
        
        for activation_event in approved_activations:
            expert_name = activation_event.expert_name
            
            # Activate expert if not already active
            activation_success = await self.expert_pool.activate_expert(
                expert_name, activation_event
            )
            
            if activation_success:
                activated_experts.append(expert_name)
                
                # Create expert context
                if EXPERTS_AVAILABLE:
                    from capibara.interfaces.isub_models import ExpertContext
                    expert_context = ExpertContext(
                        text=request_text,
                        task_hint=activation_event.context.get("task_hint", "analysis"),
                        constraints=activation_event.context.get("constraints"),
                        flags=activation_event.context.get("flags")
                    )
                else:
                    expert_context = {
                        "text": request_text,
                        "metadata": metadata or {},
                        "activation_reason": activation_event.activation_reason
                    }
                
                # Process with expert
                result = await self.expert_pool.process_with_expert(
                    expert_name, expert_context
                )
                
                results[expert_name] = {
                    "result": result,
                    "activation_reason": activation_event.activation_reason,
                    "confidence": activation_event.confidence,
                    "processing_successful": result is not None
                }
        
        # Store activation history for adaptive learning
        self.activation_history.append({
            "request_id": request_id,
            "activation_events": activation_events,
            "approved_activations": approved_activations,
            "results": results,
            "timestamp": time.time()
        })
        
        return {
            "request_id": request_id,
            "experts_activated": activated_experts,
            "results": results,
            "processing_time": time.time() - start_time,
            "activation_strategy": self.activation_strategy.value
        }
    
    async def _apply_activation_strategy(
        self, 
        activation_events: List[ExpertActivationEvent]
    ) -> List[ExpertActivationEvent]:
        """Apply the configured activation strategy to filter events."""
        
        if self.activation_strategy == ActivationStrategy.IMMEDIATE:
            return activation_events
        
        elif self.activation_strategy == ActivationStrategy.THRESHOLD_BASED:
            threshold = self.strategy_config["confidence_threshold"]
            return [event for event in activation_events if event.confidence >= threshold]
        
        elif self.activation_strategy == ActivationStrategy.CONSENSUS_REQUIRED:
            # Group by expert name and require multiple votes
            expert_votes = defaultdict(list)
            for event in activation_events:
                expert_votes[event.expert_name].append(event)
            
            approved = []
            required_votes = self.strategy_config["consensus_required_votes"]
            for expert_name, votes in expert_votes.items():
                if len(votes) >= required_votes:
                    # Use the event with highest confidence
                    best_event = max(votes, key=lambda e: e.confidence)
                    approved.append(best_event)
            
            return approved
        
        elif self.activation_strategy == ActivationStrategy.LOAD_BALANCED:
            # Consider current system load
            pool_stats = self.expert_pool.get_pool_statistics()
            if pool_stats["current_utilization"] > self.strategy_config["load_balance_threshold"]:
                # Only activate high-confidence experts when load is high
                return [event for event in activation_events if event.confidence >= 0.8]
            else:
                return activation_events
        
        elif self.activation_strategy == ActivationStrategy.ADAPTIVE:
            # Use learned weights to filter activations
            return self._adaptive_filter(activation_events)
        
        return activation_events
    
    def _adaptive_filter(self, activation_events: List[ExpertActivationEvent]) -> List[ExpertActivationEvent]:
        """Apply adaptive filtering based on learned patterns."""
        approved = []
        
        for event in activation_events:
            # Calculate adaptive score
            base_score = event.confidence
            
            # Apply learned weights
            reason_weight = self.adaptive_weights.get(event.activation_reason, 1.0)
            expert_weight = self.adaptive_weights.get(f"expert_{event.expert_name}", 1.0)
            
            adaptive_score = base_score * reason_weight * expert_weight
            
            # Threshold based on adaptive score
            if adaptive_score >= 0.5:
                approved.append(event)
        
        return approved
    
    def update_adaptive_weights(self, request_id: str, success_feedback: Dict[str, bool]):
        """Update adaptive weights based on success feedback."""
        # Find the activation history for this request
        for history_item in reversed(self.activation_history):
            if history_item["request_id"] == request_id:
                # Update weights based on success
                for event in history_item["approved_activations"]:
                    expert_success = success_feedback.get(event.expert_name, False)
                    
                    # Adjust weights
                    learning_rate = 0.1
                    if expert_success:
                        self.adaptive_weights[event.activation_reason] += learning_rate
                        self.adaptive_weights[f"expert_{event.expert_name}"] += learning_rate
                    else:
                        self.adaptive_weights[event.activation_reason] -= learning_rate
                        self.adaptive_weights[f"expert_{event.expert_name}"] -= learning_rate
                    
                    # Keep weights in reasonable bounds
                    self.adaptive_weights[event.activation_reason] = max(
                        0.1, min(2.0, self.adaptive_weights[event.activation_reason])
                    )
                    self.adaptive_weights[f"expert_{event.expert_name}"] = max(
                        0.1, min(2.0, self.adaptive_weights[f"expert_{event.expert_name}"])
                    )
                break
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the entire system."""
        observer_stats = self.observer_manager.get_observer_statistics()
        pool_stats = self.expert_pool.get_pool_statistics()
        activation_patterns = self.observer_manager.analyze_activation_patterns()
        
        return {
            "observer_statistics": observer_stats,
            "expert_pool_statistics": pool_stats,
            "activation_patterns": activation_patterns,
            "activation_strategy": self.activation_strategy.value,
            "strategy_config": self.strategy_config,
            "adaptive_weights": dict(self.adaptive_weights),
            "total_requests_processed": len(self.activation_history)
        }


# Factory function for easy setup
def create_expert_activation_manager(
    strategy: ActivationStrategy = ActivationStrategy.THRESHOLD_BASED,
    **strategy_config
) -> ExpertActivationManager:
    """
    Factory function to create a configured ExpertActivationManager.
    
    Args:
        strategy: Activation strategy to use
        **strategy_config: Configuration parameters for the strategy
        
    Returns:
        Configured ExpertActivationManager instance
    """
    manager = ExpertActivationManager(strategy)
    
    # Update strategy configuration
    manager.strategy_config.update(strategy_config)
    
    return manager