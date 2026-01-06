"""
Agent and Behavior Factories - Factory Pattern Implementation - CapibaraGPT v2024
==================================================================================

Implementation of pattern Factory para creación de agentes y comportamientos:
- BehaviorFactory: Fábrica para crear diferentes comportamientos de agentes
- AgentFactory: Fábrica mejorada para crear agentes con comportamientos específicos
- OrchestrationStrategyFactory: Fábrica para strategys de orquestación
- StrategyBasedAgentFactory: Fábrica que combina agentes con strategys
"""

import logging
from typing import Dict, Any, Optional, List, Type, Union
from abc import ABC, abstractmethod

# Safe imports
try:
    from ..interfaces.iagent import (
        IAgentBehavior, IAgentFactory, IBehaviorFactory, IOrchestrationStrategy,
        AgentBehaviorType, AgentCapability, AgentContext, AgentResult, IAgent
    )
    from .behaviors import (
        BaseBehavior, ReasoningBehavior, PlanningBehavior, ExecutionBehavior
    )
    from .advanced_behaviors import ResearchBehavior, CodingBehavior
    from .communication_behaviors import CommunicationBehavior, MonitoringBehavior
except ImportError:
    # Fallback imports
    from abc import ABC, abstractmethod
    from enum import Enum
    from typing import Dict, Any, Optional, List, Type
    
    class AgentBehaviorType(str, Enum):
        REASONING = "reasoning"
        PLANNING = "planning"
        EXECUTION = "execution"
        RESEARCH = "research"
        CODING = "coding"
        COMMUNICATION = "communication"
        MONITORING = "monitoring"
    
    class IAgentBehavior(ABC):
        @abstractmethod
        def execute_behavior(self, context, agent): pass
    
    class IBehaviorFactory(ABC):
        @abstractmethod
        def create_behavior(self, behavior_type, config): pass
    
    class IAgentFactory(ABC):
        @abstractmethod
        def create_agent(self, agent_type, config): pass
    
    # Placeholder classes for fallback
    class BaseBehavior:
        def __init__(self, config=None): pass
    
    ReasoningBehavior = PlanningBehavior = ExecutionBehavior = BaseBehavior
    ResearchBehavior = CodingBehavior = BaseBehavior  
    CommunicationBehavior = MonitoringBehavior = BaseBehavior

logger = logging.getLogger(__name__)


# ============================================================================
# Behavior Factory - Factory Pattern for Agent Behaviors
# ============================================================================

class BehaviorFactory(IBehaviorFactory):
    """
    Fábrica para crear diferentes tipos de comportamientos de agentes.
    
    Implementa el patrón Factory para la creación de comportamientos específicos
    que pueden ser utilizados por los agentes según sus necesidades.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._behavior_registry: Dict[AgentBehaviorType, Type[IAgentBehavior]] = {}
        self._behavior_instances: Dict[str, IAgentBehavior] = {}
        self._default_configs: Dict[AgentBehaviorType, Dict[str, Any]] = {}
        
        # Registrar comportamientos por defecto
        self._register_default_behaviors()
        
        # Factory statistics
        self.creation_stats = {
            "behaviors_created": 0,
            "behaviors_cached": 0,
            "registry_size": len(self._behavior_registry)
        }
    
    def _register_default_behaviors(self) -> None:
        """Register default behaviors."""
        
        # Registrar classs de comportamiento
        self._behavior_registry[AgentBehaviorType.REASONING] = ReasoningBehavior
        self._behavior_registry[AgentBehaviorType.PLANNING] = PlanningBehavior
        self._behavior_registry[AgentBehaviorType.EXECUTION] = ExecutionBehavior
        self._behavior_registry[AgentBehaviorType.RESEARCH] = ResearchBehavior
        self._behavior_registry[AgentBehaviorType.CODING] = CodingBehavior
        self._behavior_registry[AgentBehaviorType.COMMUNICATION] = CommunicationBehavior
        self._behavior_registry[AgentBehaviorType.MONITORING] = MonitoringBehavior
        
        # Configuraciones por defecto
        self._default_configs[AgentBehaviorType.REASONING] = {
            "reasoning_depth": 3,
            "use_formal_logic": False
        }
        
        self._default_configs[AgentBehaviorType.PLANNING] = {
            "planning_horizon": 5,
            "use_contingency_planning": True
        }
        
        self._default_configs[AgentBehaviorType.EXECUTION] = {
            "max_retries": 3,
            "timeout_seconds": 300,
            "monitor_progress": True
        }
        
        self._default_configs[AgentBehaviorType.RESEARCH] = {
            "max_sources": 10,
            "quality_threshold": 0.7,
            "use_data_integration": True
        }
        
        self._default_configs[AgentBehaviorType.CODING] = {
            "languages": ["python", "javascript", "typescript"],
            "include_tests": True,
            "include_docs": True
        }
        
        self._default_configs[AgentBehaviorType.COMMUNICATION] = {
            "max_message_history": 1000,
            "enable_broadcasting": True,
            "default_timeout": 30
        }
        
        self._default_configs[AgentBehaviorType.MONITORING] = {
            "monitoring_interval": 10,
            "alert_thresholds": {
                "response_time_ms": 1000,
                "error_rate": 0.05
            }
        }
        
        logger.info(f"Registered {len(self._behavior_registry)} default behaviors")
    
    def create_behavior(
        self, 
        behavior_type: AgentBehaviorType,
        config: Optional[Dict[str, Any]] = None
    ) -> IAgentBehavior:
        """
        Crear un comportamiento específico.
        
        Args:
            behavior_type: Tipo de comportamiento a crear
            config: Configuración opcional para el comportamiento
            
        Returns:
            IAgentBehavior: Instancia del comportamiento creado
            
        Raises:
            ValueError: Si el tipo de comportamiento no está soportado
        """
        
        if behavior_type not in self._behavior_registry:
            raise ValueError(f"Behavior type {behavior_type} not supported. "
                           f"Available types: {list(self._behavior_registry.keys())}")
        
        # Create configuration combinada
        final_config = self._create_final_config(behavior_type, config)
        
        # Verificar si usar instancia cacheada
        cache_key = self._create_cache_key(behavior_type, final_config)
        if self.config.get("enable_caching", True) and cache_key in self._behavior_instances:
            self.creation_stats["behaviors_cached"] += 1
            logger.debug(f"Returning cached behavior: {behavior_type}")
            return self._behavior_instances[cache_key]
        
        # Create new instance
        behavior_class = self._behavior_registry[behavior_type]
        
        try:
            behavior_instance = behavior_class(final_config)
            
            # Cachear si está habilitado
            if self.config.get("enable_caching", True):
                self._behavior_instances[cache_key] = behavior_instance
            
            self.creation_stats["behaviors_created"] += 1
            logger.info(f"Created behavior: {behavior_type} with config keys: {list(final_config.keys())}")
            
            return behavior_instance
            
        except Exception as e:
            logger.error(f"Failed to create behavior {behavior_type}: {e}")
            raise
    
    def _create_final_config(
        self, 
        behavior_type: AgentBehaviorType, 
        user_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create final configuration by combining defaults and user configuration."""
        
        # Start with default configuration
        final_config = self._default_configs.get(behavior_type, {}).copy()
        
        # Add global factory configuration
        if "global_behavior_config" in self.config:
            final_config.update(self.config["global_behavior_config"])
        
        # Add user-specific configuration
        if user_config:
            final_config.update(user_config)
        
        return final_config
    
    def _create_cache_key(self, behavior_type: AgentBehaviorType, config: Dict[str, Any]) -> str:
        """Create cache key based on type and configuration."""
        # Create simple configuration hash
        config_hash = hash(str(sorted(config.items())))
        return f"{behavior_type}_{config_hash}"
    
    def get_available_behaviors(self) -> List[AgentBehaviorType]:
        """Obtener lista de comportamientos disponibles."""
        return list(self._behavior_registry.keys())
    
    def register_behavior_class(
        self, 
        behavior_type: AgentBehaviorType,
        behavior_class: Type[IAgentBehavior],
        default_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registrar una nueva class de comportamiento.
        
        Args:
            behavior_type: Tipo de comportamiento
            behavior_class: Clase que implementa el comportamiento
            default_config: Configuración por defecto opcional
        """
        
        self._behavior_registry[behavior_type] = behavior_class
        
        if default_config:
            self._default_configs[behavior_type] = default_config
        
        self.creation_stats["registry_size"] = len(self._behavior_registry)
        logger.info(f"Registered behavior class: {behavior_type} -> {behavior_class.__name__}")
    
    def get_behavior_info(self, behavior_type: AgentBehaviorType) -> Dict[str, Any]:
        """Obtener información sobre un tipo de comportamiento."""
        
        if behavior_type not in self._behavior_registry:
            return {"available": False, "error": "Behavior type not registered"}
        
        behavior_class = self._behavior_registry[behavior_type]
        default_config = self._default_configs.get(behavior_type, {})
        
        return {
            "available": True,
            "class_name": behavior_class.__name__,
            "default_config": default_config,
            "required_capabilities": getattr(behavior_class, 'required_capabilities', []),
            "description": behavior_class.__doc__ or "No description available"
        }
    
    def clear_cache(self) -> None:
        """Limpiar cache de instancias de comportamiento."""
        cached_count = len(self._behavior_instances)
        self._behavior_instances.clear()
        logger.info(f"Cleared {cached_count} cached behavior instances")
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics."""
        return {
            **self.creation_stats,
            "cached_instances": len(self._behavior_instances),
            "available_behaviors": len(self._behavior_registry)
        }


# ============================================================================
# Strategy-Based Agent Factory - Enhanced Agent Factory
# ============================================================================

class StrategyBasedAgentFactory(IAgentFactory):
    """
    Fábrica de agentes mejorada que utiliza el patrón Strategy.
    
    Combina la creación de agentes con la asignación de comportamientos específicos,
    permitiendo mayor flexibilidad y reutilización de componentes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.behavior_factory = BehaviorFactory(self.config.get("behavior_factory_config", {}))
        
        # Registro de tipos de agentes soportados
        self._supported_agent_types = list(AgentBehaviorType)
        
        # Templates de agentes predefinidos
        self._agent_templates = self._create_agent_templates()
        
        # Statistics
        self.creation_stats = {
            "agents_created": 0,
            "templates_used": 0,
            "custom_agents": 0
        }
    
    def _create_agent_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create predefined agent templates."""
        
        templates = {
            "reasoning_specialist": {
                "primary_behavior": AgentBehaviorType.REASONING,
                "secondary_behaviors": [AgentBehaviorType.PLANNING],
                "config": {
                    "reasoning_depth": 5,
                    "use_formal_logic": True
                },
                "description": "Specialized agent for logical reasoning and analysis"
            },
            
            "execution_expert": {
                "primary_behavior": AgentBehaviorType.EXECUTION,
                "secondary_behaviors": [AgentBehaviorType.MONITORING],
                "config": {
                    "max_retries": 5,
                    "monitor_progress": True
                },
                "description": "Expert agent for reliable task execution"
            },
            
            "research_analyst": {
                "primary_behavior": AgentBehaviorType.RESEARCH,
                "secondary_behaviors": [AgentBehaviorType.REASONING, AgentBehaviorType.PLANNING],
                "config": {
                    "max_sources": 15,
                    "quality_threshold": 0.8
                },
                "description": "Advanced research and analysis agent"
            },
            
            "coding_developer": {
                "primary_behavior": AgentBehaviorType.CODING,
                "secondary_behaviors": [AgentBehaviorType.REASONING, AgentBehaviorType.EXECUTION],
                "config": {
                    "languages": ["python", "javascript", "typescript", "rust"],
                    "include_tests": True,
                    "include_docs": True
                },
                "description": "Full-stack development agent"
            },
            
            "communication_coordinator": {
                "primary_behavior": AgentBehaviorType.COMMUNICATION,
                "secondary_behaviors": [AgentBehaviorType.MONITORING, AgentBehaviorType.PLANNING],
                "config": {
                    "enable_broadcasting": True,
                    "max_message_history": 2000
                },
                "description": "Agent specialized in inter-agent coordination"
            },
            
            "system_monitor": {
                "primary_behavior": AgentBehaviorType.MONITORING,
                "secondary_behaviors": [AgentBehaviorType.COMMUNICATION],
                "config": {
                    "monitoring_interval": 5,
                    "alert_thresholds": {
                        "response_time_ms": 500,
                        "error_rate": 0.02
                    }
                },
                "description": "System monitoring and alerting agent"
            },
            
            "general_assistant": {
                "primary_behavior": AgentBehaviorType.REASONING,
                "secondary_behaviors": [
                    AgentBehaviorType.PLANNING,
                    AgentBehaviorType.EXECUTION,
                    AgentBehaviorType.COMMUNICATION
                ],
                "config": {
                    "reasoning_depth": 3,
                    "planning_horizon": 5
                },
                "description": "General-purpose assistant agent"
            }
        }
        
        return templates
    
    def create_agent(
        self, 
        agent_type: AgentBehaviorType,
        config: Optional[Dict[str, Any]] = None
    ) -> IAgent:
        """
        Crear un agente de un tipo específico.
        
        Args:
            agent_type: Tipo de agente a crear
            config: Configuración opcional
            
        Returns:
            IAgent: Instancia del agente creado
        """
        
        if agent_type not in self._supported_agent_types:
            raise ValueError(f"Agent type {agent_type} not supported. "
                           f"Available types: {self._supported_agent_types}")
        
        # Create main behavior
        primary_behavior = self.behavior_factory.create_behavior(agent_type, config)
        
        # Create agent with behavior
        agent = StrategyBasedAgent(
            agent_id=self._generate_agent_id(agent_type),
            agent_type=agent_type,
            primary_behavior=primary_behavior,
            config=config or {}
        )
        
        self.creation_stats["agents_created"] += 1
        logger.info(f"Created agent: {agent.agent_id} of type {agent_type}")
        
        return agent
    
    def create_agent_from_spec(self, spec: Dict[str, Any]) -> IAgent:
        """
        Crear un agente basándose en una especificación completa.
        
        Args:
            spec: Especificación del agente
            
        Returns:
            IAgent: Instancia del agente creado
        """
        
        # Extract information from specification
        agent_type = AgentBehaviorType(spec.get("type", "reasoning"))
        agent_name = spec.get("name", f"agent_{agent_type}")
        behaviors = spec.get("behaviors", [agent_type])
        config = spec.get("config", {})
        
        # Create behaviors
        primary_behavior = self.behavior_factory.create_behavior(agent_type, config)
        secondary_behaviors = []
        
        for behavior_type in behaviors[1:]:  # Comportamientos secundarios
            if isinstance(behavior_type, str):
                behavior_type = AgentBehaviorType(behavior_type)
            secondary_behavior = self.behavior_factory.create_behavior(behavior_type, config)
            secondary_behaviors.append(secondary_behavior)
        
        # Create agent
        agent = StrategyBasedAgent(
            agent_id=agent_name,
            agent_type=agent_type,
            primary_behavior=primary_behavior,
            secondary_behaviors=secondary_behaviors,
            config=config
        )
        
        self.creation_stats["custom_agents"] += 1
        logger.info(f"Created agent from spec: {agent_name}")
        
        return agent
    
    def create_agent_from_template(self, template_name: str, config: Optional[Dict[str, Any]] = None) -> IAgent:
        """
        Crear un agente usando un template predefinido.
        
        Args:
            template_name: Nombre del template a usar
            config: Configuración adicional opcional
            
        Returns:
            IAgent: Instancia del agente creado
        """
        
        if template_name not in self._agent_templates:
            available_templates = list(self._agent_templates.keys())
            raise ValueError(f"Template '{template_name}' not found. "
                           f"Available templates: {available_templates}")
        
        template = self._agent_templates[template_name]
        
        # Combine template configuration with user configuration
        final_config = template["config"].copy()
        if config:
            final_config.update(config)
        
        # Create main behavior
        primary_behavior = self.behavior_factory.create_behavior(
            template["primary_behavior"], 
            final_config
        )
        
        # Create behaviors secundarios
        secondary_behaviors = []
        for behavior_type in template.get("secondary_behaviors", []):
            behavior = self.behavior_factory.create_behavior(behavior_type, final_config)
            secondary_behaviors.append(behavior)
        
        # Create agent
        agent = StrategyBasedAgent(
            agent_id=self._generate_agent_id(template["primary_behavior"]),
            agent_type=template["primary_behavior"],
            primary_behavior=primary_behavior,
            secondary_behaviors=secondary_behaviors,
            config=final_config
        )
        
        self.creation_stats["templates_used"] += 1
        logger.info(f"Created agent from template: {template_name}")
        
        return agent
    
    def get_supported_types(self) -> List[AgentBehaviorType]:
        """Obtener tipos de agentes soportados."""
        return self._supported_agent_types.copy()
    
    def can_create(self, agent_type: AgentBehaviorType) -> bool:
        """Check if a specific agent type can be created."""
        return agent_type in self._supported_agent_types
    
    def get_available_templates(self) -> Dict[str, str]:
        """Obtener templates disponibles con sus descripciones."""
        return {
            name: template["description"] 
            for name, template in self._agent_templates.items()
        }
    
    def register_behavior(self, behavior: IAgentBehavior) -> None:
        """Register a new behavior."""
        # Delegar al behavior factory
        if hasattr(behavior, 'behavior_type'):
            self.behavior_factory.register_behavior_class(
                behavior.behavior_type,
                type(behavior)
            )
    
    def _generate_agent_id(self, agent_type: AgentBehaviorType) -> str:
        """Generate unique agent ID."""
        import time
        timestamp = int(time.time() * 1000)
        return f"{agent_type}_{timestamp}"
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics."""
        behavior_stats = self.behavior_factory.get_factory_stats()
        
        return {
            **self.creation_stats,
            "available_templates": len(self._agent_templates),
            "supported_agent_types": len(self._supported_agent_types),
            "behavior_factory_stats": behavior_stats
        }


# ============================================================================
# Strategy-Based Agent Implementation
# ============================================================================

class StrategyBasedAgent(IAgent):
    """
    Implementación de agente que utiliza el patrón Strategy.
    
    Este agente puede cambiar su comportamiento dinámicamente utilizando
    diferentes strategys de comportamiento.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentBehaviorType,
        primary_behavior: IAgentBehavior,
        secondary_behaviors: Optional[List[IAgentBehavior]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self._agent_id = agent_id
        self._agent_type = agent_type
        self._primary_behavior = primary_behavior
        self._secondary_behaviors = secondary_behaviors or []
        self.config = config or {}
        
        # Agent state
        self._status = "ready"
        self._current_behavior = primary_behavior
        self._execution_history: List[Dict[str, Any]] = []
        
        # Métricas del agente
        self._metrics = {
            "tasks_executed": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time_ms": 0.0,
            "behavior_switches": 0
        }
        
        logger.info(f"Created StrategyBasedAgent: {agent_id} with {len(self._secondary_behaviors)} secondary behaviors")
    
    @property
    def agent_id(self) -> str:
        return self._agent_id
    
    @property
    def agent_type(self) -> AgentBehaviorType:
        return self._agent_type
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        """Obtener capacidades combinadas de todos los comportamientos."""
        capabilities = []
        
        # Agregar capacidades del comportamiento principal
        if hasattr(self._primary_behavior, 'required_capabilities'):
            capabilities.extend(self._primary_behavior.required_capabilities)
        
        # Agregar capacidades de comportamientos secundarios
        for behavior in self._secondary_behaviors:
            if hasattr(behavior, 'required_capabilities'):
                capabilities.extend(behavior.required_capabilities)
        
        # Eliminar duplicados
        return list(set(capabilities))
    
    def execute(self, context: AgentContext) -> AgentResult:
        """Execute task using appropriate behavior."""
        
        start_time = time.time()
        self._metrics["tasks_executed"] += 1
        
        try:
            # Seleccionar comportamiento apropiado
            selected_behavior = self._select_behavior_for_context(context)
            
            # Cambiar comportamiento si es necesario
            if selected_behavior != self._current_behavior:
                self._switch_behavior(selected_behavior)
            
            # Execute with selected behavior
            result = self._current_behavior.execute_behavior(context, self)
            
            # Actualizar métricas
            execution_time = (time.time() - start_time) * 1000
            self._update_execution_metrics(result, execution_time)
            
            # Save in history
            self._execution_history.append({
                "context": context,
                "behavior_used": type(self._current_behavior).__name__,
                "result": result,
                "execution_time_ms": execution_time,
                "timestamp": time.time()
            })
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._metrics["failed_executions"] += 1
            self._metrics["total_execution_time_ms"] += execution_time
            
            logger.error(f"Agent {self.agent_id} execution failed: {e}")
            
            return AgentResult(
                agent_id=self.agent_id,
                status="failed",
                result=None,
                execution_time_ms=execution_time,
                error=str(e)
            )
    
    def can_handle(self, task_description: str, requirements: Dict[str, Any]) -> bool:
        """Determinar si el agente puede manejar una tarea."""
        
        # Create temporary context for evaluation
        temp_context = AgentContext(
            task_id="evaluation",
            task_description=task_description,
            requirements=requirements
        )
        
        # Verify if any behavior can handle the task
        all_behaviors = [self._primary_behavior] + self._secondary_behaviors
        
        for behavior in all_behaviors:
            if hasattr(behavior, 'validate_context'):
                if behavior.validate_context(temp_context):
                    return True
        
        # Evaluación básica basada en tipo de agente
        task_lower = task_description.lower()
        
        if self._agent_type == AgentBehaviorType.REASONING:
            return any(keyword in task_lower for keyword in ["analyze", "reason", "think", "logic"])
        elif self._agent_type == AgentBehaviorType.CODING:
            return any(keyword in task_lower for keyword in ["code", "program", "implement", "develop"])
        elif self._agent_type == AgentBehaviorType.RESEARCH:
            return any(keyword in task_lower for keyword in ["research", "investigate", "study", "find"])
        else:
            return True  # General agent can try any task
    
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado actual del agente."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "status": self._status,
            "current_behavior": type(self._current_behavior).__name__,
            "available_behaviors": [type(b).__name__ for b in [self._primary_behavior] + self._secondary_behaviors],
            "capabilities": [cap.value for cap in self.capabilities],
            "metrics": self._metrics,
            "execution_history_size": len(self._execution_history)
        }
    
    def _select_behavior_for_context(self, context: AgentContext) -> IAgentBehavior:
        """Seleccionar el comportamiento más apropiado para el contexto."""
        
        task_description = context.task_description.lower()
        
        # Evaluar comportamientos secundarios primero (más específicos)
        for behavior in self._secondary_behaviors:
            if self._is_behavior_suitable(behavior, task_description):
                return behavior
        
        # Use main behavior as fallback
        return self._primary_behavior
    
    def _is_behavior_suitable(self, behavior: IAgentBehavior, task_description: str) -> bool:
        """Determinar si un comportamiento es adecuado para una tarea."""
        
        behavior_type = getattr(behavior, 'behavior_type', None)
        if not behavior_type:
            return False
        
        # Mapeo de palabras clave a tipos de comportamiento
        keyword_mapping = {
            AgentBehaviorType.REASONING: ["analyze", "reason", "think", "logic", "deduce"],
            AgentBehaviorType.PLANNING: ["plan", "strategy", "organize", "schedule"],
            AgentBehaviorType.EXECUTION: ["execute", "run", "perform", "do", "implement"],
            AgentBehaviorType.RESEARCH: ["research", "investigate", "study", "find", "search"],
            AgentBehaviorType.CODING: ["code", "program", "develop", "script", "software"],
            AgentBehaviorType.COMMUNICATION: ["communicate", "message", "coordinate", "collaborate"],
            AgentBehaviorType.MONITORING: ["monitor", "watch", "track", "observe", "measure"]
        }
        
        keywords = keyword_mapping.get(behavior_type, [])
        return any(keyword in task_description for keyword in keywords)
    
    def _switch_behavior(self, new_behavior: IAgentBehavior) -> None:
        """Cambiar al nuevo comportamiento."""
        old_behavior_name = type(self._current_behavior).__name__
        new_behavior_name = type(new_behavior).__name__
        
        self._current_behavior = new_behavior
        self._metrics["behavior_switches"] += 1
        
        logger.info(f"Agent {self.agent_id} switched behavior: {old_behavior_name} -> {new_behavior_name}")
    
    def _update_execution_metrics(self, result: AgentResult, execution_time: float) -> None:
        """Updatesr métricas de ejecución."""
        
        self._metrics["total_execution_time_ms"] += execution_time
        
        if result.status == "success":
            self._metrics["successful_executions"] += 1
        else:
            self._metrics["failed_executions"] += 1
    
    def add_behavior(self, behavior: IAgentBehavior) -> None:
        """Agregar un nuevo comportamiento secundario."""
        if behavior not in self._secondary_behaviors:
            self._secondary_behaviors.append(behavior)
            logger.info(f"Added behavior {type(behavior).__name__} to agent {self.agent_id}")
    
    def remove_behavior(self, behavior_type: AgentBehaviorType) -> bool:
        """Remover un comportamiento secundario."""
        for i, behavior in enumerate(self._secondary_behaviors):
            if hasattr(behavior, 'behavior_type') and behavior.behavior_type == behavior_type:
                removed_behavior = self._secondary_behaviors.pop(i)
                logger.info(f"Removed behavior {type(removed_behavior).__name__} from agent {self.agent_id}")
                return True
        return False
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Obtener historial de ejecuciones."""
        if limit:
            return self._execution_history[-limit:]
        return self._execution_history.copy()


# Export all factories and agent implementations
__all__ = [
    "BehaviorFactory",
    "StrategyBasedAgentFactory", 
    "StrategyBasedAgent",
    "IBehaviorFactory",
    "IAgentFactory"
]