"""
Request Observer Interface and Core Events
==========================================

This module defines the core Observer pattern interfaces for request monitoring
and expert activation. It provides the foundation for event-driven expert
management in the CapibaraGPT system.
"""

import time
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Callable, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class RequestEventType(Enum):
    """Types of request events that can trigger expert activation."""
    REQUEST_RECEIVED = "request_received"
    REQUEST_ANALYZED = "request_analyzed" 
    COMPLEXITY_DETECTED = "complexity_detected"
    DOMAIN_IDENTIFIED = "domain_identified"
    PATTERN_MATCHED = "pattern_matched"
    EXPERT_NEEDED = "expert_needed"
    ROUTING_DECISION = "routing_decision"
    PROCESSING_COMPLETE = "processing_complete"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_THRESHOLD = "performance_threshold"


@dataclass
class RequestEvent:
    """Event data structure for request-related events."""
    event_type: RequestEventType
    request_id: str
    timestamp: float = field(default_factory=time.time)
    
    # Request data
    request_text: Optional[str] = None
    request_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis results
    complexity_score: Optional[float] = None
    domain_predictions: Dict[str, float] = field(default_factory=dict)
    patterns_detected: List[str] = field(default_factory=list)
    
    # Performance metrics
    processing_time: Optional[float] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    # Context information
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary format."""
        return {
            "event_type": self.event_type.value,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "request_text": self.request_text,
            "request_metadata": self.request_metadata,
            "complexity_score": self.complexity_score,
            "domain_predictions": self.domain_predictions,
            "patterns_detected": self.patterns_detected,
            "processing_time": self.processing_time,
            "confidence_scores": self.confidence_scores,
            "context": self.context
        }


@dataclass 
class ExpertActivationEvent:
    """Event for expert activation decisions."""
    expert_name: str
    activation_reason: str
    confidence: float
    priority: int = 1  # 1=high, 2=medium, 3=low
    activation_time: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Activation metadata
    triggered_by_patterns: List[str] = field(default_factory=list)
    expected_contribution: str = ""
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


class RequestObserver(ABC):
    """
    Abstract base class for request observers.
    
    Observers monitor request events and can trigger expert activation
    based on their specific logic and patterns.
    """
    
    def __init__(self, name: str, priority: int = 1):
        self.name = name
        self.priority = priority
        self.is_active = True
        self.activation_count = 0
        self.last_activation_time = None
        self.performance_metrics = {
            "total_observations": 0,
            "successful_activations": 0,
            "false_positives": 0,
            "average_confidence": 0.0
        }
    
    @abstractmethod
    async def observe(self, event: RequestEvent) -> List[ExpertActivationEvent]:
        """
        Observe a request event and return expert activation events if needed.
        
        Args:
            event: The request event to observe
            
        Returns:
            List of expert activation events (empty if no activation needed)
        """
        pass
    
    @abstractmethod
    def should_activate(self, event: RequestEvent) -> bool:
        """
        Determine if this observer should trigger expert activation.
        
        Args:
            event: The request event to evaluate
            
        Returns:
            True if expert activation should be triggered
        """
        pass
    
    def get_supported_event_types(self) -> Set[RequestEventType]:
        """Return the set of event types this observer handles."""
        return {RequestEventType.REQUEST_RECEIVED}
    
    def update_metrics(self, activation_event: ExpertActivationEvent, success: bool):
        """Update performance metrics for this observer."""
        self.performance_metrics["total_observations"] += 1
        if success:
            self.performance_metrics["successful_activations"] += 1
        else:
            self.performance_metrics["false_positives"] += 1
        
        # Update average confidence
        current_avg = self.performance_metrics["average_confidence"]
        total_obs = self.performance_metrics["total_observations"]
        new_avg = ((current_avg * (total_obs - 1)) + activation_event.confidence) / total_obs
        self.performance_metrics["average_confidence"] = new_avg
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this observer."""
        total = self.performance_metrics["total_observations"]
        success_rate = (
            self.performance_metrics["successful_activations"] / total 
            if total > 0 else 0.0
        )
        
        return {
            "name": self.name,
            "priority": self.priority,
            "is_active": self.is_active,
            "activation_count": self.activation_count,
            "success_rate": success_rate,
            "average_confidence": self.performance_metrics["average_confidence"],
            "total_observations": total,
            "last_activation": self.last_activation_time
        }


class ObserverManager:
    """
    Manages a collection of request observers and coordinates their execution.
    """
    
    def __init__(self):
        self.observers: List[RequestObserver] = []
        self.event_history: List[RequestEvent] = []
        self.activation_history: List[ExpertActivationEvent] = []
        self.max_history_size = 1000
        
    def add_observer(self, observer: RequestObserver):
        """Add a new observer to the manager."""
        self.observers.append(observer)
        self.observers.sort(key=lambda x: x.priority)  # Sort by priority
        logger.info(f"Added observer: {observer.name} with priority {observer.priority}")
    
    def remove_observer(self, observer_name: str):
        """Remove an observer by name."""
        self.observers = [obs for obs in self.observers if obs.name != observer_name]
        logger.info(f"Removed observer: {observer_name}")
    
    def get_observer(self, name: str) -> Optional[RequestObserver]:
        """Get an observer by name."""
        for observer in self.observers:
            if observer.name == name:
                return observer
        return None
    
    async def notify_observers(self, event: RequestEvent) -> List[ExpertActivationEvent]:
        """
        Notify all relevant observers about a request event.
        
        Args:
            event: The request event to broadcast
            
        Returns:
            List of all expert activation events from observers
        """
        # Store event in history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)
        
        activation_events = []
        
        # Notify observers in priority order
        for observer in self.observers:
            if not observer.is_active:
                continue
                
            # Check if observer handles this event type
            if event.event_type not in observer.get_supported_event_types():
                continue
            
            try:
                observer_activations = await observer.observe(event)
                activation_events.extend(observer_activations)
                
                # Update observer metrics
                for activation in observer_activations:
                    observer.activation_count += 1
                    observer.last_activation_time = time.time()
                    
            except Exception as e:
                logger.error(f"Observer {observer.name} failed to process event: {e}")
                continue
        
        # Store activation events
        self.activation_history.extend(activation_events)
        if len(self.activation_history) > self.max_history_size:
            self.activation_history = self.activation_history[-self.max_history_size:]
        
        return activation_events
    
    def get_observer_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all observers."""
        return {
            "total_observers": len(self.observers),
            "active_observers": len([obs for obs in self.observers if obs.is_active]),
            "observer_performance": [obs.get_performance_summary() for obs in self.observers],
            "total_events_processed": len(self.event_history),
            "total_activations_triggered": len(self.activation_history),
            "recent_event_types": [
                event.event_type.value 
                for event in self.event_history[-10:]
            ]
        }
    
    def analyze_activation_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in expert activation events."""
        if not self.activation_history:
            return {"message": "No activation history available"}
        
        # Count activations by expert
        expert_counts = {}
        confidence_by_expert = {}
        
        for activation in self.activation_history:
            expert_name = activation.expert_name
            expert_counts[expert_name] = expert_counts.get(expert_name, 0) + 1
            
            if expert_name not in confidence_by_expert:
                confidence_by_expert[expert_name] = []
            confidence_by_expert[expert_name].append(activation.confidence)
        
        # Calculate average confidence by expert
        avg_confidence_by_expert = {
            expert: sum(confidences) / len(confidences)
            for expert, confidences in confidence_by_expert.items()
        }
        
        # Find most common activation reasons
        reason_counts = {}
        for activation in self.activation_history:
            reason = activation.activation_reason
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        return {
            "expert_activation_counts": expert_counts,
            "average_confidence_by_expert": avg_confidence_by_expert,
            "most_common_activation_reasons": sorted(
                reason_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            "total_activations": len(self.activation_history),
            "unique_experts_activated": len(expert_counts)
        }


# Utility functions for creating common event types

def create_request_received_event(
    request_id: str,
    request_text: str,
    metadata: Optional[Dict[str, Any]] = None
) -> RequestEvent:
    """Create a REQUEST_RECEIVED event."""
    return RequestEvent(
        event_type=RequestEventType.REQUEST_RECEIVED,
        request_id=request_id,
        request_text=request_text,
        request_metadata=metadata or {}
    )


def create_complexity_detected_event(
    request_id: str,
    complexity_score: float,
    patterns: Optional[List[str]] = None
) -> RequestEvent:
    """Create a COMPLEXITY_DETECTED event."""
    return RequestEvent(
        event_type=RequestEventType.COMPLEXITY_DETECTED,
        request_id=request_id,
        complexity_score=complexity_score,
        patterns_detected=patterns or []
    )


def create_domain_identified_event(
    request_id: str,
    domain_predictions: Dict[str, float]
) -> RequestEvent:
    """Create a DOMAIN_IDENTIFIED event."""
    return RequestEvent(
        event_type=RequestEventType.DOMAIN_IDENTIFIED,
        request_id=request_id,
        domain_predictions=domain_predictions
    )


def create_expert_activation_event(
    expert_name: str,
    reason: str,
    confidence: float,
    priority: int = 1,
    patterns: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None
) -> ExpertActivationEvent:
    """Create an expert activation event."""
    return ExpertActivationEvent(
        expert_name=expert_name,
        activation_reason=reason,
        confidence=confidence,
        priority=priority,
        triggered_by_patterns=patterns or [],
        context=context or {}
    )