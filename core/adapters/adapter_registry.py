"""
Central Adapter Registry System

Provides centralized registration and management of all system adapters,
enabling automatic selection, fallbacks, and dynamic optimization.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Type, Callable
from dataclasses import dataclass, field
import time
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

class AdapterType(Enum):
    """Available adapter types in the system."""
    KERNEL_ABSTRACTION = "kernel_abstraction"
    PERFORMANCE_OPTIMIZATION = "performance_optimization" 
    HARDWARE_COMPATIBILITY = "hardware_compatibility"
    QUANTIZATION = "quantization"
    LANGUAGE_PROCESSING = "language_processing"
    NEUROMORPHIC = "neuromorphic"
    MEMORY_MANAGEMENT = "memory_management"
    CACHING = "caching"

@dataclass
class AdapterInfo:
    """Information about a registered adapter."""
    adapter_type: AdapterType
    adapter_class: Type
    priority: int = 50  # 0-100, higher = more priority
    requirements: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    performance_score: float = 0.0
    last_used: float = 0.0
    usage_count: int = 0
    success_rate: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AdapterSelectionCriteria:
    """Criteria for automatic adapter selection."""
    adapter_type: AdapterType
    hardware_requirements: List[str] = field(default_factory=list)
    performance_requirements: Dict[str, float] = field(default_factory=dict)
    compatibility_requirements: List[str] = field(default_factory=list)
    prefer_fallback: bool = False
    max_latency_ms: Optional[float] = None
    min_throughput: Optional[float] = None

class AdapterRegistry:
    """Central registry for system adapters."""
    
    def __init__(self):
        self._adapters: Dict[AdapterType, List[AdapterInfo]] = defaultdict(list)
        self._instances: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._selection_strategies: Dict[AdapterType, Callable] = {}
        self._global_metrics = {
            'total_selections': 0,
            'successful_selections': 0,
            'fallback_selections': 0,
            'average_selection_time_ms': 0.0
        }
        
        # Setup default selection strategies
        self._setup_default_strategies()
        
    def register_adapter(self, 
                        adapter_type: AdapterType,
                        adapter_class: Type,
                        priority: int = 50,
                        requirements: List[str] = None,
                        capabilities: List[str] = None,
                        metadata: Dict[str, Any] = None) -> bool:
        """
        Registers a new adapter in the system.
        
        Args:
            adapter_type: Tipo del adapter
            adapter_class: Clase del adapter
            priority: Prioridad (0-100, mayor = más prioridad)
            requirements: Lista de requisitos del adapter
            capabilities: Lista de capacidades del adapter
            metadata: Metadatos adicionales
            
        Returns:
            True si el registro fue exitoso
        """
        try:
            with self._lock:
                adapter_info = AdapterInfo(
                    adapter_type=adapter_type,
                    adapter_class=adapter_class,
                    priority=priority,
                    requirements=requirements or [],
                    capabilities=capabilities or [],
                    metadata=metadata or {}
                )
                
                # Verify that la class tenga los methods requeridos
                if not self._validate_adapter_class(adapter_class):
                    logger.error(f"Adapter class {adapter_class.__name__} does not implement required interface")
                    return False
                
                self._adapters[adapter_type].append(adapter_info)
                
                # Ordenar por prioridad (mayor prioridad primero)
                self._adapters[adapter_type].sort(key=lambda x: x.priority, reverse=True)
                
                logger.info(f"Registered adapter {adapter_class.__name__} for type {adapter_type.value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register adapter {adapter_class.__name__}: {e}")
            return False
    
    def get_adapter(self, 
                   adapter_type: AdapterType,
                   criteria: Optional[AdapterSelectionCriteria] = None,
                   force_new_instance: bool = False) -> Optional[Any]:
        """
        Obtiene el mejor adapter disponible según los criterios.
        
        Args:
            adapter_type: Tipo de adapter requerido
            criteria: Criterios de selección
            force_new_instance: Forzar creación de nueva instancia
            
        Returns:
            Instancia del adapter seleccionado o None
        """
        start_time = time.time()
        
        try:
            with self._lock:
                self._global_metrics['total_selections'] += 1
                
                # Usar criterios por defecto si no se proporcionan
                if criteria is None:
                    criteria = AdapterSelectionCriteria(adapter_type=adapter_type)
                
                # Seleccionar el mejor adapter
                selected_adapter_info = self._select_best_adapter(adapter_type, criteria)
                
                if selected_adapter_info is None:
                    logger.warning(f"No suitable adapter found for type {adapter_type.value}")
                    self._global_metrics['fallback_selections'] += 1
                    return None
                
                # Obtener o crear instancia
                instance_key = f"{adapter_type.value}_{selected_adapter_info.adapter_class.__name__}"
                
                if force_new_instance or instance_key not in self._instances:
                    try:
                        instance = selected_adapter_info.adapter_class()
                        self._instances[instance_key] = instance
                        logger.info(f"Created new instance of {selected_adapter_info.adapter_class.__name__}")
                    except Exception as e:
                        logger.error(f"Failed to create adapter instance: {e}")
                        return None
                
                # Actualizar métricas del adapter
                selected_adapter_info.last_used = time.time()
                selected_adapter_info.usage_count += 1
                
                self._global_metrics['successful_selections'] += 1
                
                selection_time = (time.time() - start_time) * 1000
                self._update_average_selection_time(selection_time)
                
                return self._instances[instance_key]
                
        except Exception as e:
            logger.error(f"Error getting adapter: {e}")
            return None
    
    def get_available_adapters(self, adapter_type: AdapterType) -> List[AdapterInfo]:
        """Gets lista de adapters disponibles para un tipo."""
        with self._lock:
            return list(self._adapters.get(adapter_type, []))
    
    def get_adapter_metrics(self, adapter_type: Optional[AdapterType] = None) -> Dict[str, Any]:
        """Gets métricas de uso de adapters."""
        with self._lock:
            if adapter_type is None:
                # Métricas globales
                return dict(self._global_metrics)
            else:
                # Type-specific metrics
                adapters = self._adapters.get(adapter_type, [])
                return {
                    'adapter_count': len(adapters),
                    'total_usage': sum(a.usage_count for a in adapters),
                    'average_performance': sum(a.performance_score for a in adapters) / len(adapters) if adapters else 0.0,
                    'adapters': [
                        {
                            'class_name': a.adapter_class.__name__,
                            'priority': a.priority,
                            'usage_count': a.usage_count,
                            'success_rate': a.success_rate,
                            'performance_score': a.performance_score
                        } for a in adapters
                    ]
                }
    
    def update_adapter_performance(self, 
                                 adapter_type: AdapterType,
                                 adapter_class_name: str,
                                 performance_score: float,
                                 success: bool = True):
        """Updates metrics de rendimiento de un adapter."""
        with self._lock:
            for adapter_info in self._adapters.get(adapter_type, []):
                if adapter_info.adapter_class.__name__ == adapter_class_name:
                    # Actualizar performance score (promedio móvil)
                    if adapter_info.performance_score == 0.0:
                        adapter_info.performance_score = performance_score
                    else:
                        adapter_info.performance_score = (
                            adapter_info.performance_score * 0.8 + performance_score * 0.2
                        )
                    
                    # Actualizar success rate
                    total_attempts = adapter_info.usage_count
                    if total_attempts > 0:
                        current_successes = adapter_info.success_rate * (total_attempts - 1)
                        new_successes = current_successes + (1 if success else 0)
                        adapter_info.success_rate = new_successes / total_attempts
                    
                    break
    
    def set_selection_strategy(self, 
                             adapter_type: AdapterType,
                             strategy_func: Callable[[List[AdapterInfo], AdapterSelectionCriteria], Optional[AdapterInfo]]):
        """Establishes strategy for custom selection for an adapter type."""
        self._selection_strategies[adapter_type] = strategy_func
        logger.info(f"Set custom selection strategy for {adapter_type.value}")
    
    def _setup_default_strategies(self):
        """Configures strategies de selección por defecto."""
        # Estrategia por defecto: prioridad + performance + success rate
        def default_strategy(adapters: List[AdapterInfo], criteria: AdapterSelectionCriteria) -> Optional[AdapterInfo]:
            if not adapters:
                return None
            
            # Filtrar por requisitos
            suitable_adapters = []
            for adapter in adapters:
                if self._meets_requirements(adapter, criteria):
                    suitable_adapters.append(adapter)
            
            if not suitable_adapters:
                return None
            
            # Calcular score combinado
            best_adapter = None
            best_score = -1
            
            for adapter in suitable_adapters:
                score = (
                    adapter.priority * 0.4 +
                    adapter.performance_score * 0.3 +
                    adapter.success_rate * 100 * 0.3
                )
                
                if score > best_score:
                    best_score = score
                    best_adapter = adapter
            
            return best_adapter
        
        # Aplicar strategy por defecto a todos los tipos
        for adapter_type in AdapterType:
            self._selection_strategies[adapter_type] = default_strategy
    
    def _select_best_adapter(self, 
                           adapter_type: AdapterType,
                           criteria: AdapterSelectionCriteria) -> Optional[AdapterInfo]:
        """Selecciona el mejor adapter usando la strategy configurada."""
        available_adapters = self._adapters.get(adapter_type, [])
        
        if not available_adapters:
            return None
        
        strategy = self._selection_strategies.get(adapter_type)
        if strategy is None:
            logger.warning(f"No selection strategy for {adapter_type.value}, using first available")
            return available_adapters[0] if available_adapters else None
        
        return strategy(available_adapters, criteria)
    
    def _meets_requirements(self, 
                          adapter: AdapterInfo,
                          criteria: AdapterSelectionCriteria) -> bool:
        """Verifies si un adapter cumple con los requisitos."""
        # Verificar requisitos de hardware
        for hw_req in criteria.hardware_requirements:
            if hw_req not in adapter.capabilities:
                return False
        
        # Verificar requisitos de compatibilidad
        for compat_req in criteria.compatibility_requirements:
            if compat_req not in adapter.capabilities:
                return False
        
        # Verificar requisitos de performance
        if criteria.max_latency_ms is not None:
            adapter_latency = adapter.metadata.get('average_latency_ms', float('inf'))
            if adapter_latency > criteria.max_latency_ms:
                return False
        
        if criteria.min_throughput is not None:
            adapter_throughput = adapter.metadata.get('throughput', 0)
            if adapter_throughput < criteria.min_throughput:
                return False
        
        return True
    
    def _validate_adapter_class(self, adapter_class: Type) -> bool:
        """Validates that an adapter class implements the required interface."""
        # Verify that tenga los methods básicos
        required_methods = ['__init__']  # Métodos mínimos requeridos
        
        for method in required_methods:
            if not hasattr(adapter_class, method):
                return False
        
        return True
    
    def _update_average_selection_time(self, new_time_ms: float):
        """Updates the time promedio de selección."""
        current_avg = self._global_metrics['average_selection_time_ms']
        total_selections = self._global_metrics['total_selections']
        
        if total_selections == 1:
            self._global_metrics['average_selection_time_ms'] = new_time_ms
        else:
            # Promedio móvil
            self._global_metrics['average_selection_time_ms'] = (
                current_avg * 0.9 + new_time_ms * 0.1
            )
    
    def clear_instances(self):
        """Limpia todas las instancias cacheadas."""
        with self._lock:
            self._instances.clear()
            logger.info("Cleared all cached adapter instances")
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Gets the state completo del registro."""
        with self._lock:
            status = {
                'total_adapter_types': len(self._adapters),
                'total_registered_adapters': sum(len(adapters) for adapters in self._adapters.values()),
                'cached_instances': len(self._instances),
                'global_metrics': dict(self._global_metrics),
                'adapter_types': {}
            }
            
            for adapter_type, adapters in self._adapters.items():
                status['adapter_types'][adapter_type.value] = {
                    'count': len(adapters),
                    'adapters': [
                        {
                            'class_name': a.adapter_class.__name__,
                            'priority': a.priority,
                            'usage_count': a.usage_count,
                            'success_rate': a.success_rate
                        } for a in adapters
                    ]
                }
            
            return status


# Utility functions for automatic registration
def register_adapter_decorator(adapter_type: AdapterType, 
                             priority: int = 50,
                             requirements: List[str] = None,
                             capabilities: List[str] = None,
                             metadata: Dict[str, Any] = None):
    """Decorador para registro automático de adapters."""
    def decorator(cls):
        from . import adapter_registry
        adapter_registry.register_adapter(
            adapter_type=adapter_type,
            adapter_class=cls,
            priority=priority,
            requirements=requirements or [],
            capabilities=capabilities or [],
            metadata=metadata or {}
        )
        return cls
    return decorator