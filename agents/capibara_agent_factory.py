## capibara/agents/capibara_agent_factory.py - Enhanced Factory Pattern Implementation

import os
import sys
import logging
from capibara.jax.numpy import jnp

# Safe imports for Factory and Strategy patterns
try:
    from ..interfaces.iagent import (
        IAgentFactory, IAgent, AgentBehaviorType, AgentContext, AgentResult
    )
    from .factories import StrategyBasedAgentFactory, BehaviorFactory
    from .behaviors import ReasoningBehavior, PlanningBehavior, ExecutionBehavior
    ENHANCED_FACTORY_AVAILABLE = True
except ImportError:
    ENHANCED_FACTORY_AVAILABLE = False
    from abc import ABC, abstractmethod
    
    class IAgentFactory(ABC):
        @abstractmethod
        def create_agent(self, agent_type, config): pass

# Fixed: Using proper imports instead of sys.path manipulation
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

from capibara.jax import jax
from typing import Optional, List, Dict, Any, Union
from capibara.jax.experimental import mesh_utils
from capibara.jax.sharding import Mesh, PartitionSpec 

# Importaciones de tpu v4-32
from capibara.jax.tpu_v4.backend import (
    TpuV4LinalgOps,
    TpuV4SparseOps,
    TpuV4NeuralOps,
    TpuV4RandomOps,
    TpuV4PerformanceUtils,
)
from capibara.jax.tpu_v4.optimizations import (
    create_tpu_mesh,
    TpuMemoryMonitor,
    tpu_optimized_gemm,
    create_jitted_forward,
)
from capibara.jax.tpu_v4.profiling import (
    TpuProfiler,
    _uniform_fallback_weights,
    _expert_weights_with_cache,
    checkpointed_transformer_block,
)

from .tool_library import get_tool_by_name
from .capibara_agent import CapibaraAgent, CapibaraTool, CapibaraLLM, CapibaraVectorDB

logger = logging.getLogger(__name__)

# setup tpu v4-32
TPU_MESH_SHAPE = (32, 8)
TPU_MEMORY_LIMIT_GB = 32

# Stub for LLM without dependencias externas
class Ollama:
    def __init__(self, model):
        self.model = model
    
    def invoke(self, prompt):
        # implementation stub - reemplazar with LLM real
        return f"Respuesta generada por {self.model}: {prompt[:50]}..."

LLM_REGISTRY = {
    "ollama": lambda spec: CapibaraLLM(Ollama(model=spec["model"]))
}

def create_llm(llm_spec: dict) -> CapibaraLLM:
    llm_type = llm_spec["type"]
    if llm_type not in LLM_REGISTRY:
        raise ValueError(f"LLM type {llm_type} no soportado. Opciones: {list(LLM_REGISTRY.keys())}")
    try:
        llm = LLM_REGISTRY[llm_type](llm_spec)
        # configure optimizaciones tpu v4-32
        llm.mesh = create_tpu_mesh(TPU_MESH_SHAPE)
        llm.memory_monitor = TpuMemoryMonitor(limit_gb=TPU_MEMORY_LIMIT_GB)
        llm.profiler = TpuProfiler()
        return llm
    except KeyError as e:
        raise ValueError(f"Falta parameter requerido {e} en spec para LLM tipo {llm_type}")

def create_vector_db(vectordb_spec: dict) -> Optional[CapibaraVectorDB]:
    if vectordb_spec.get("type") == "qdrant":
        # Stub retriever - reemplazar with retriever real
        class StubRetriever:
            def invoke(self, query):
                return [type('Doc', (), {'page_content': f"Contexto para: {query}"})()]
        
        vectordb = CapibaraVectorDB(StubRetriever())
        # configure optimizaciones tpu v4-32
        vectordb.mesh = create_tpu_mesh(TPU_MESH_SHAPE)
        vectordb.memory_monitor = TpuMemoryMonitor(limit_gb=TPU_MEMORY_LIMIT_GB)
        return vectordb
    return None

def create_tools(tool_names: List[str]):
    tools = [get_tool_by_name(name) for name in tool_names]
    # configure optimizaciones tpu v4-32 for each herramienta
    for tool in tools:
        tool.mesh = create_tpu_mesh(TPU_MESH_SHAPE)
        tool.memory_monitor = TpuMemoryMonitor(limit_gb=TPU_MEMORY_LIMIT_GB)
    return tools

def create_agent_from_spec(spec: dict, base_model=None):
    # Agente estándar
    llm = create_llm(spec["llm"])
    vectordb = create_vector_db(spec.get("vectordb", {}))
    tools = create_tools(spec.get("tools", []))
    agent = CapibaraAgent(name=spec["name"], llm=llm, vectordb=vectordb, tools=tools)
    # configure optimizaciones tpu v4-32
    agent.mesh = create_tpu_mesh(TPU_MESH_SHAPE)
    agent.memory_monitor = TpuMemoryMonitor(limit_gb=TPU_MEMORY_LIMIT_GB)
    agent.profiler = TpuProfiler()
    return agent

def create_capibara_agent(name: str, llm_config: dict = None, tools: list = None, vectordb_config: dict = None):
    """Create a CapibaraAgent with specified configuration."""
    try:
        # Default configuration
        if llm_config is None:
            llm_config = {"type": "default", "model": "capibara-base"}
        
        if tools is None:
            tools = []
        
        if vectordb_config is None:
            vectordb_config = {"type": "memory", "dimension": 768}
        
        # Create components
        llm = create_llm(llm_config)
        vectordb = create_vector_db(vectordb_config)
        tools_list = create_tools(tools)
        
        # Create agent
        agent = CapibaraAgent(name=name, llm=llm, vectordb=vectordb, tools=tools_list)
        
        # Configure TPU optimizations
        agent.mesh = create_tpu_mesh(TPU_MESH_SHAPE)
        agent.memory_monitor = TpuMemoryMonitor(limit_gb=TPU_MEMORY_LIMIT_GB)
        agent.profiler = TpuProfiler()
        
        return agent
        
    except Exception as e:
        raise Exception(f"Failed to create CapibaraAgent: {e}")

# ============================================================================
# Enhanced Factory Pattern Implementation
# ============================================================================

class CapibaraAgentFactory(IAgentFactory):
    """
    Enhanced CapibaraAgentFactory implementing Factory Pattern with Strategy support.
    
    This factory combines the original CapibaraAgent functionality with the new
    Strategy-based agent system for maximum flexibility and backward compatibility.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize enhanced factory if available
        if ENHANCED_FACTORY_AVAILABLE:
            self.strategy_factory = StrategyBasedAgentFactory(
                self.config.get("strategy_factory_config", {})
            )
            self.behavior_factory = BehaviorFactory(
                self.config.get("behavior_factory_config", {})
            )
            logger.info("Enhanced Factory Pattern initialized")
        else:
            self.strategy_factory = None
            self.behavior_factory = None
            logger.warning("Enhanced Factory Pattern not available, using legacy mode")
        
        # Legacy factory statistics
        self.creation_stats = {
            "legacy_agents_created": 0,
            "strategy_agents_created": 0,
            "total_agents_created": 0
        }
    
    def create_agent(
        self, 
        agent_type: Union[str, AgentBehaviorType],
        config: Optional[Dict[str, Any]] = None
    ) -> Union[IAgent, CapibaraAgent]:
        """
        Create an agent of the specified type.
        
        Args:
            agent_type: Type of agent to create (string or AgentBehaviorType)
            config: Configuration for the agent
            
        Returns:
            Agent instance (IAgent if enhanced, CapibaraAgent if legacy)
        """
        
        final_config = self._merge_config(config)
        
        # Try enhanced factory first
        if ENHANCED_FACTORY_AVAILABLE and self.strategy_factory:
            try:
                # Convert string to AgentBehaviorType if needed
                if isinstance(agent_type, str):
                    if agent_type in ["reasoning", "planning", "execution", "research", "coding", "communication", "monitoring"]:
                        agent_type = AgentBehaviorType(agent_type)
                    else:
                        # Map legacy types to new types
                        type_mapping = {
                            "capibara": AgentBehaviorType.REASONING,
                            "auto": AgentBehaviorType.EXECUTION,
                            "ultra": AgentBehaviorType.COMMUNICATION
                        }
                        agent_type = type_mapping.get(agent_type, AgentBehaviorType.REASONING)
                
                # Create using strategy factory
                agent = self.strategy_factory.create_agent(agent_type, final_config)
                self.creation_stats["strategy_agents_created"] += 1
                self.creation_stats["total_agents_created"] += 1
                
                logger.info(f"Created strategy-based agent: {agent.agent_id} of type {agent_type}")
                return agent
                
            except Exception as e:
                logger.warning(f"Strategy factory failed, falling back to legacy: {e}")
        
        # Fallback to legacy creation
        return self._create_legacy_agent(agent_type, final_config)
    
    def create_agent_from_spec(self, spec: Dict[str, Any]) -> Union[IAgent, CapibaraAgent]:
        """
        Create agent from specification.
        
        Args:
            spec: Agent specification dictionary
            
        Returns:
            Agent instance
        """
        
        # Try enhanced factory first
        if ENHANCED_FACTORY_AVAILABLE and self.strategy_factory:
            try:
                agent = self.strategy_factory.create_agent_from_spec(spec)
                self.creation_stats["strategy_agents_created"] += 1
                self.creation_stats["total_agents_created"] += 1
                return agent
            except Exception as e:
                logger.warning(f"Strategy factory spec creation failed: {e}")
        
        # Fallback to legacy spec creation
        return create_agent_from_spec(spec)
    
    def create_agent_from_template(self, template_name: str, config: Optional[Dict[str, Any]] = None) -> Union[IAgent, CapibaraAgent]:
        """
        Create agent from predefined template.
        
        Args:
            template_name: Name of the template
            config: Additional configuration
            
        Returns:
            Agent instance
        """
        
        if ENHANCED_FACTORY_AVAILABLE and self.strategy_factory:
            try:
                agent = self.strategy_factory.create_agent_from_template(template_name, config)
                self.creation_stats["strategy_agents_created"] += 1
                self.creation_stats["total_agents_created"] += 1
                return agent
            except Exception as e:
                logger.warning(f"Template creation failed: {e}")
        
        # Fallback: create based on template mapping
        template_mapping = {
            "reasoning_specialist": "reasoning",
            "coding_developer": "coding",
            "research_analyst": "research",
            "general_assistant": "capibara"
        }
        
        agent_type = template_mapping.get(template_name, "capibara")
        return self.create_agent(agent_type, config)
    
    def get_supported_types(self) -> List[Union[str, AgentBehaviorType]]:
        """Get supported agent types."""
        
        types = []
        
        # Enhanced types
        if ENHANCED_FACTORY_AVAILABLE and self.strategy_factory:
            types.extend(self.strategy_factory.get_supported_types())
        
        # Legacy types
        types.extend(["capibara", "auto", "ultra"])
        
        return types
    
    def can_create(self, agent_type: Union[str, AgentBehaviorType]) -> bool:
        """Check if agent type can be created."""
        
        # Check enhanced factory
        if ENHANCED_FACTORY_AVAILABLE and self.strategy_factory:
            if isinstance(agent_type, AgentBehaviorType):
                return self.strategy_factory.can_create(agent_type)
            elif agent_type in ["reasoning", "planning", "execution", "research", "coding", "communication", "monitoring"]:
                return True
        
        # Check legacy types
        return agent_type in ["capibara", "auto", "ultra"]
    
    def get_available_templates(self) -> Dict[str, str]:
        """Get available agent templates."""
        
        templates = {}
        
        # Enhanced templates
        if ENHANCED_FACTORY_AVAILABLE and self.strategy_factory:
            templates.update(self.strategy_factory.get_available_templates())
        
        # Legacy templates
        templates.update({
            "capibara_basic": "Standard CapibaraAgent with TPU optimization",
            "capibara_auto": "Automatic agent with self-configuration",
            "capibara_ultra": "Ultra agent with advanced orchestration"
        })
        
        return templates
    
    def _merge_config(self, user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge user configuration with factory defaults."""
        
        final_config = self.config.copy()
        
        if user_config:
            final_config.update(user_config)
        
        return final_config
    
    def _create_legacy_agent(self, agent_type: Union[str, AgentBehaviorType], config: Dict[str, Any]) -> CapibaraAgent:
        """Create agent using legacy method."""
        
        # Convert AgentBehaviorType to string if needed
        if hasattr(agent_type, 'value'):
            agent_type = agent_type.value
        
        # Use existing legacy creation logic
        if agent_type in ["capibara", "reasoning", "planning", "execution"]:
            return create_capibara_agent(
                name=config.get("name", f"agent_{agent_type}"),
                llm_config=config.get("llm", {"type": "ollama", "model": "capibara-base"}),
                tools=config.get("tools", []),
                vectordb_config=config.get("vectordb", {"type": "qdrant"})
            )
        else:
            # Generic agent creation
            return create_capibara_agent(
                name=config.get("name", "capibara_agent"),
                llm_config=config.get("llm", {"type": "ollama", "model": "capibara-base"}),
                tools=config.get("tools", []),
                vectordb_config=config.get("vectordb", {"type": "qdrant"})
            )
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics."""
        
        stats = self.creation_stats.copy()
        
        # Add enhanced factory stats if available
        if ENHANCED_FACTORY_AVAILABLE and self.strategy_factory:
            enhanced_stats = self.strategy_factory.get_factory_stats()
            stats["enhanced_factory_stats"] = enhanced_stats
        
        # Add behavior factory stats if available  
        if ENHANCED_FACTORY_AVAILABLE and self.behavior_factory:
            behavior_stats = self.behavior_factory.get_factory_stats()
            stats["behavior_factory_stats"] = behavior_stats
        
        stats["enhanced_factory_available"] = ENHANCED_FACTORY_AVAILABLE
        
        return stats

def get_available_agents():
    """Get list of available agent types and configurations."""
    
    agents = {
        # Legacy agents
        "capibara": {
            "description": "Standard CapibaraAgent with TPU optimization",
            "features": ["reasoning", "tool_usage", "memory", "tpu_acceleration"],
            "default_config": {
                "llm": {"type": "default", "model": "capibara-base"},
                "tools": [],
                "vectordb": {"type": "memory", "dimension": 768}
            }
        },
        "auto": {
            "description": "Automatic agent with self-configuration",
            "features": ["auto_config", "adaptive_behavior", "self_optimization"],
            "default_config": {
                "mode": "auto",
                "adaptation_rate": 0.1
            }
        },
        "ultra": {
            "description": "Ultra agent with advanced orchestration",
            "features": ["multi_agent", "orchestration", "parallel_execution"],
            "default_config": {
                "orchestrator": True,
                "max_agents": 10
            }
        }
    }
    
    # Add enhanced agents if available
    if ENHANCED_FACTORY_AVAILABLE:
        agents.update({
            "reasoning": {
                "description": "Specialized reasoning agent with logical analysis",
                "features": ["logical_reasoning", "pattern_recognition", "causal_analysis"],
                "default_config": {
                    "reasoning_depth": 3,
                    "use_formal_logic": False
                }
            },
            "planning": {
                "description": "Strategic planning agent with task decomposition",
                "features": ["task_decomposition", "strategy_formulation", "resource_allocation"],
                "default_config": {
                    "planning_horizon": 5,
                    "use_contingency_planning": True
                }
            },
            "execution": {
                "description": "Reliable execution agent with progress monitoring",
                "features": ["action_execution", "progress_monitoring", "error_handling"],
                "default_config": {
                    "max_retries": 3,
                    "monitor_progress": True
                }
            },
            "research": {
                "description": "Research agent with information gathering capabilities",
                "features": ["information_gathering", "source_validation", "data_analysis"],
                "default_config": {
                    "max_sources": 10,
                    "quality_threshold": 0.7
                }
            },
            "coding": {
                "description": "Programming agent with code generation and debugging",
                "features": ["code_generation", "code_debugging", "testing_framework"],
                "default_config": {
                    "languages": ["python", "javascript", "typescript"],
                    "include_tests": True
                }
            },
            "communication": {
                "description": "Communication agent for inter-agent coordination",
                "features": ["inter_agent_communication", "message_routing", "conflict_resolution"],
                "default_config": {
                    "enable_broadcasting": True,
                    "max_message_history": 1000
                }
            },
            "monitoring": {
                "description": "Monitoring agent for system health and performance",
                "features": ["performance_monitoring", "health_checking", "anomaly_detection"],
                "default_config": {
                    "monitoring_interval": 10,
                    "alert_thresholds": {
                        "response_time_ms": 1000,
                        "error_rate": 0.05
                    }
                }
            }
        })
    
    return agents
