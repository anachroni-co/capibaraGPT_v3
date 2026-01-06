"""
Observer Pattern Implementation for Dynamic Expert Activation
============================================================

This module implements the Observer pattern to enable dynamic activation of experts
based on request patterns, inputs, and system events. The pattern allows for:

- Event-driven expert activation
- Dynamic routing decisions
- Real-time adaptation to request patterns
- Decoupled expert management
- Scalable expert ecosystem

Main Components:
- RequestObserver: Interface for observing request events
- ExpertActivationManager: Manages expert lifecycle and activation
- RouterObserverIntegration: Integrates observers with routing system
- EventDrivenExpertPool: Pool of experts that can be dynamically activated

Usage Example:
    from capibara.core.observers import ExpertActivationManager, RequestPatternObserver
    
    # Create activation manager
    manager = ExpertActivationManager()
    
    # Add observers
    pattern_observer = RequestPatternObserver()
    manager.add_observer(pattern_observer)
    
    # Process request (experts will be activated based on patterns)
    result = await manager.process_with_dynamic_activation(request)
"""

from .request_observer import (
    RequestObserver,
    RequestEvent,
    RequestEventType,
    ExpertActivationEvent
)

from .expert_activation_manager import (
    ExpertActivationManager,
    ExpertPool,
    ActivationStrategy,
    ExpertState
)

from .observers import (
    RequestPatternObserver,
    ComplexityObserver,
    DomainSpecificObserver,
    PerformanceObserver,
    AdaptiveObserver
)

from .router_integration import (
    ObserverAwareRouter,
    RouterObserverIntegration,
    DynamicRoutingDecision
)

__all__ = [
    # Core interfaces
    'RequestObserver',
    'RequestEvent', 
    'RequestEventType',
    'ExpertActivationEvent',
    
    # Management
    'ExpertActivationManager',
    'ExpertPool',
    'ActivationStrategy', 
    'ExpertState',
    
    # Concrete observers
    'RequestPatternObserver',
    'ComplexityObserver',
    'DomainSpecificObserver', 
    'PerformanceObserver',
    'AdaptiveObserver',
    
    # Router integration
    'ObserverAwareRouter',
    'RouterObserverIntegration',
    'DynamicRoutingDecision'
]