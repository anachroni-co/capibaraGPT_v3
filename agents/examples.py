"""
Factory and Strategy Pattern Examples - CapibaraGPT v3024
=========================================================

Ejemplos completos demostrando el uso de los patrones Factory y Strategy
en el system de agentes de CapibaraGPT:

1. Factory Pattern Examples:
   - Creación basic of agentes
   - Uso de templates predefinidos
   - Creación desde especificaciones
   - Fábricas de comportamientos

2. Strategy Pattern Examples:
   - Cambio dinámico de comportamientos
   - Combinación de múltiples strategys
   - Orquestación con diferentes strategys
   - Adaptación basada en contexto

3. Advanced Examples:
   - Sistema de agentes colaborativo
   - Pipeline de procesamiento con múltiples strategys
   - Sistema de monitoreo adaptativo
"""

import time
import logging
from typing import Dict, Any, List, Optional

# Safe imports
try:
    from ..interfaces.iagent import (
        AgentBehaviorType, AgentContext, AgentResult, IAgent
    )
    from .factories import StrategyBasedAgentFactory, BehaviorFactory
    from .capibara_agent_factory import CapibaraAgentFactory
    from .behaviors import ReasoningBehavior, PlanningBehavior, ExecutionBehavior
    from .advanced_behaviors import ResearchBehavior, CodingBehavior
    from .communication_behaviors import CommunicationBehavior, MonitoringBehavior
    from .orchestration_strategies import IntelligentOrchestrationStrategy
    PATTERNS_AVAILABLE = True
except ImportError:
    PATTERNS_AVAILABLE = False
    logger.warning("️ Pattern implementations not available. Running in demo mode.")

logger = logging.getLogger(__name__)


def demonstrate_factory_patterns():
    """
    Demostrar el uso del patrón Factory en diferentes escenarios.
    """
    
    logger.info(" FACTORY PATTERN DEMONSTRATIONS")
    logger.info("=" * 60)
    
    if not PATTERNS_AVAILABLE:
        logger.warning(" Pattern implementations not available")
        return
    
    # ========================================================================
    # Example 1: Basic Agent Factory Usage
    # ========================================================================
    
    logger.info("\n1️⃣ Basic Agent Factory Usage")
    logger.info("-" * 40)
    
    try:
        # Create factory
        factory = StrategyBasedAgentFactory()
        
        # Create different types of agents
        reasoning_agent = factory.create_agent(AgentBehaviorType.REASONING)
        coding_agent = factory.create_agent(AgentBehaviorType.CODING)
        research_agent = factory.create_agent(AgentBehaviorType.RESEARCH)
        
        logger.info(f" Created reasoning agent: {reasoning_agent.agent_id}")
        logger.info(f" Created coding agent: {coding_agent.agent_id}")
        logger.info(f" Created research agent: {research_agent.agent_id}")
        
        # Show agent capabilities
        logger.info(f"\n Agent Capabilities:")
        logger.info(f"   Reasoning: {[cap.value for cap in reasoning_agent.capabilities]}")
        logger.info(f"   Coding: {[cap.value for cap in coding_agent.capabilities]}")
        logger.info(f"   Research: {[cap.value for cap in research_agent.capabilities]}")
        
    except Exception as e:
        logger.error(f" Basic factory example failed: {e}")
    
    # ========================================================================
    # Example 2: Template-Based Agent Creation
    # ========================================================================
    
    logger.info("\n2️⃣ Template-Based Agent Creation")
    logger.info("-" * 40)
    
    try:
        factory = StrategyBasedAgentFactory()
        
        # Get available templates
        templates = factory.get_available_templates()
        logger.info(f" Available templates: {list(templates.keys())}")
        
        # Create agents from templates
        specialist = factory.create_agent_from_template("reasoning_specialist")
        developer = factory.create_agent_from_template("coding_developer")
        assistant = factory.create_agent_from_template("general_assistant")
        
        logger.info(f" Created specialist: {specialist.agent_id}")
        logger.info(f" Created developer: {developer.agent_id}")
        logger.info(f" Created assistant: {assistant.agent_id}")
        
        # Show template configurations
        logger.info(f"\n️ Template Features:")
        for name, description in templates.items():
            logger.info(f"   {name}: {description}")
        
    except Exception as e:
        logger.error(f" Template example failed: {e}")
    
    # ========================================================================
    # Example 3: Specification-Based Creation
    # ========================================================================
    
    logger.info("\n3️⃣ Specification-Based Creation")
    logger.info("-" * 40)
    
    try:
        factory = StrategyBasedAgentFactory()
        
        # Create agent from detailed specification
        custom_spec = {
            "name": "custom_research_agent",
            "type": "research",
            "behaviors": ["research", "reasoning", "communication"],
            "config": {
                "max_sources": 15,
                "quality_threshold": 0.8,
                "reasoning_depth": 4,
                "enable_broadcasting": True
            }
        }
        
        custom_agent = factory.create_agent_from_spec(custom_spec)
        
        logger.info(f" Created custom agent: {custom_agent.agent_id}")
        logger.info(f"   Type: {custom_agent.agent_type}")
        logger.info(f"   Capabilities: {len(custom_agent.capabilities)} total")
        
        # Test the custom agent
        test_context = AgentContext(
            task_id="test_research",
            task_description="Research the latest trends in AI agent architectures",
            requirements={"depth": "comprehensive", "sources": "academic"}
        )
        
        result = custom_agent.execute(test_context)
        logger.info(f"   Test execution: {result.status} ({result.execution_time_ms:.1f}ms)")
        
    except Exception as e:
        logger.error(f" Specification example failed: {e}")
    
    # ========================================================================
    # Example 4: Behavior Factory Usage
    # ========================================================================
    
    logger.info("\n4️⃣ Behavior Factory Usage")
    logger.info("-" * 40)
    
    try:
        behavior_factory = BehaviorFactory()
        
        # Create different behaviors with custom configurations
        reasoning_behavior = behavior_factory.create_behavior(
            AgentBehaviorType.REASONING,
            {"reasoning_depth": 5, "use_formal_logic": True}
        )
        
        coding_behavior = behavior_factory.create_behavior(
            AgentBehaviorType.CODING,
            {"languages": ["python", "rust", "go"], "include_docs": True}
        )
        
        monitoring_behavior = behavior_factory.create_behavior(
            AgentBehaviorType.MONITORING,
            {"monitoring_interval": 5, "alert_thresholds": {"response_time_ms": 500}}
        )
        
        logger.info(f" Created reasoning behavior: {reasoning_behavior.behavior_type}")
        logger.info(f" Created coding behavior: {coding_behavior.behavior_type}")
        logger.info(f" Created monitoring behavior: {monitoring_behavior.behavior_type}")
        
        # Show behavior factory statistics
        stats = behavior_factory.get_factory_stats()
        logger.info(f"\n Factory Stats:")
        logger.info(f"   Behaviors created: {stats['behaviors_created']}")
        logger.info(f"   Registry size: {stats['registry_size']}")
        logger.info(f"   Available behaviors: {stats['available_behaviors']}")
        
    except Exception as e:
        logger.error(f" Behavior factory example failed: {e}")
    
    # ========================================================================
    # Example 5: Enhanced CapibaraAgentFactory
    # ========================================================================
    
    logger.info("\n5️⃣ Enhanced CapibaraAgentFactory")
    logger.info("-" * 40)
    
    try:
        # Create enhanced factory (with backward compatibility)
        capibara_factory = CapibaraAgentFactory()
        
        # Create both legacy and new style agents
        legacy_agent = capibara_factory.create_agent("capibara")
        modern_agent = capibara_factory.create_agent("reasoning")
        
        logger.info(f" Created legacy agent: {getattr(legacy_agent, 'name', 'legacy_capibara')}")
        logger.info(f" Created modern agent: {modern_agent.agent_id}")
        
        # Show supported types
        supported_types = capibara_factory.get_supported_types()
        logger.info(f"\n Supported Types: {len(supported_types)} total")
        for agent_type in supported_types[:5]:  # Show first 5
            logger.info(f"   - {agent_type}")
        
        # Show factory statistics
        factory_stats = capibara_factory.get_factory_stats()
        logger.info(f"\n Factory Statistics:")
        logger.info(f"   Total agents created: {factory_stats['total_agents_created']}")
        logger.info(f"   Strategy agents: {factory_stats['strategy_agents_created']}")
        logger.info(f"   Legacy agents: {factory_stats['legacy_agents_created']}")
        
    except Exception as e:
        logger.error(f" Enhanced factory example failed: {e}")


def demonstrate_strategy_patterns():
    """
    Demostrar el uso del patrón Strategy en diferentes escenarios.
    """
    
    logger.info("\n STRATEGY PATTERN DEMONSTRATIONS")
    logger.info("=" * 60)
    
    if not PATTERNS_AVAILABLE:
        logger.warning(" Pattern implementations not available")
        return
    
    # ========================================================================
    # Example 1: Dynamic Behavior Switching
    # ========================================================================
    
    logger.info("\n1️⃣ Dynamic Behavior Switching")
    logger.info("-" * 40)
    
    try:
        factory = StrategyBasedAgentFactory()
        
        # Create agent with multiple behaviors
        multi_agent = factory.create_agent_from_spec({
            "name": "adaptive_agent",
            "type": "reasoning",
            "behaviors": ["reasoning", "planning", "execution", "communication"],
            "config": {"reasoning_depth": 3}
        })
        
        logger.info(f" Created adaptive agent: {multi_agent.agent_id}")
        logger.info(f"   Available behaviors: {len(multi_agent._secondary_behaviors) + 1}")
        
        # Test different types of tasks to see behavior switching
        test_tasks = [
            ("Analyze the performance bottlenecks in this system", "reasoning"),
            ("Create a project plan for implementing a new feature", "planning"),
            ("Execute the deployment of the application", "execution"),
            ("Coordinate with other agents for task completion", "communication")
        ]
        
        for task_desc, expected_behavior in test_tasks:
            context = AgentContext(
                task_id=f"test_{expected_behavior}",
                task_description=task_desc,
                requirements={}
            )
            
            result = multi_agent.execute(context)
            logger.info(f"   Task: {expected_behavior} -> {result.status} ({result.execution_time_ms:.1f}ms)")
        
        # Show behavior switching statistics
        agent_stats = multi_agent.get_status()
        logger.info(f"\n Agent Statistics:")
        logger.info(f"   Behavior switches: {agent_stats['metrics']['behavior_switches']}")
        logger.info(f"   Current behavior: {agent_stats['current_behavior']}")
        
    except Exception as e:
        logger.error(f" Dynamic switching example failed: {e}")
    
    # ========================================================================
    # Example 2: Behavior Composition and Combination
    # ========================================================================
    
    logger.info("\n2️⃣ Behavior Composition and Combination")
    logger.info("-" * 40)
    
    try:
        behavior_factory = BehaviorFactory()
        
        # Create complementary behaviors
        research_behavior = behavior_factory.create_behavior(
            AgentBehaviorType.RESEARCH,
            {"max_sources": 8, "quality_threshold": 0.75}
        )
        
        reasoning_behavior = behavior_factory.create_behavior(
            AgentBehaviorType.REASONING,
            {"reasoning_depth": 4}
        )
        
        communication_behavior = behavior_factory.create_behavior(
            AgentBehaviorType.COMMUNICATION,
            {"enable_broadcasting": True}
        )
        
        logger.info(" Created complementary behaviors:")
        logger.info(f"   Research: {research_behavior.behavior_type}")
        logger.info(f"   Reasoning: {reasoning_behavior.behavior_type}")
        logger.info(f"   Communication: {communication_behavior.behavior_type}")
        
        # Create agent that combines these behaviors
        agent_factory = StrategyBasedAgentFactory()
        combined_agent = agent_factory.create_agent(AgentBehaviorType.RESEARCH)
        
        # Add additional behaviors
        combined_agent.add_behavior(reasoning_behavior)
        combined_agent.add_behavior(communication_behavior)
        
        # Test complex task requiring multiple behaviors
        complex_context = AgentContext(
            task_id="complex_research",
            task_description="Research AI safety measures, analyze the findings, and coordinate with other agents to create a comprehensive report",
            requirements={"collaboration": True, "depth": "comprehensive"}
        )
        
        result = combined_agent.execute(complex_context)
        logger.info(f"\n Complex Task Result:")
        logger.info(f"   Status: {result.status}")
        logger.info(f"   Execution time: {result.execution_time_ms:.1f}ms")
        logger.info(f"   Confidence: {result.confidence:.2f}")
        
    except Exception as e:
        logger.error(f" Behavior composition example failed: {e}")
    
    # ========================================================================
    # Example 3: Context-Aware Strategy Selection
    # ========================================================================
    
    logger.info("\n3️⃣ Context-Aware Strategy Selection")
    logger.info("-" * 40)
    
    try:
        factory = StrategyBasedAgentFactory()
        smart_agent = factory.create_agent_from_template("general_assistant")
        
        # Test different contexts to see adaptive behavior selection
        contexts = [
            {
                "description": "Debug this Python code that's throwing exceptions",
                "requirements": {"urgency": "high", "language": "python"},
                "expected": "execution/coding"
            },
            {
                "description": "Find research papers about quantum computing applications",
                "requirements": {"sources": "academic", "quality": "high"},
                "expected": "research"
            },
            {
                "description": "Think through the logical implications of this decision",
                "requirements": {"depth": "analytical", "method": "formal"},
                "expected": "reasoning"
            },
            {
                "description": "Organize a timeline for the product launch",
                "requirements": {"timeline": "6_months", "resources": "limited"},
                "expected": "planning"
            }
        ]
        
        logger.info(" Testing Context-Aware Selection:")
        
        for i, ctx in enumerate(contexts, 1):
            context = AgentContext(
                task_id=f"adaptive_{i}",
                task_description=ctx["description"],
                requirements=ctx["requirements"]
            )
            
            # Check if agent can handle the task
            can_handle = smart_agent.can_handle(ctx["description"], ctx["requirements"])
            
            if can_handle:
                result = smart_agent.execute(context)
                logger.info(f"   {i}. {ctx['expected']} -> {result.status} ({'' if result.status == 'success' else ''})")
            else:
                logger.error(f"   {i}. {ctx['expected']} -> Cannot handle")
        
    except Exception as e:
        logger.error(f" Context-aware selection example failed: {e}")
    
    # ========================================================================
    # Example 4: Orchestration Strategy Patterns
    # ========================================================================
    
    logger.info("\n4️⃣ Orchestration Strategy Patterns")
    logger.info("-" * 40)
    
    try:
        # Create multiple agents for orchestration
        factory = StrategyBasedAgentFactory()
        agents = [
            factory.create_agent(AgentBehaviorType.REASONING),
            factory.create_agent(AgentBehaviorType.PLANNING),
            factory.create_agent(AgentBehaviorType.EXECUTION),
            factory.create_agent(AgentBehaviorType.RESEARCH)
        ]
        
        logger.info(f" Created agent team: {len(agents)} agents")
        
        # Create intelligent orchestration strategy
        orchestration_strategy = IntelligentOrchestrationStrategy()
        
        # Plan complex task execution
        complex_task = "Develop a comprehensive AI safety framework with implementation guidelines"
        requirements = {
            "complexity": "high",
            "collaboration": True,
            "timeline": "extended"
        }
        
        execution_plan = orchestration_strategy.plan_execution(
            complex_task, requirements, agents
        )
        
        logger.info(f"\n Orchestration Plan:")
        logger.info(f"   Strategy: {orchestration_strategy.strategy_name}")
        logger.info(f"   Estimated duration: {execution_plan.get('estimated_duration', 0):.1f}s")
        logger.info(f"   Agent utilization: {execution_plan.get('agent_utilization', 0):.2%}")
        logger.info(f"   Confidence: {execution_plan.get('confidence', 0):.2f}")
        
        # Execute the orchestrated plan
        coordination_result = orchestration_strategy.coordinate_execution(
            execution_plan, agents
        )
        
        logger.info(f"\n Coordination Result:")
        logger.info(f"   Status: {coordination_result.get('status', 'unknown')}")
        logger.info(f"   Phases completed: {coordination_result.get('phases_completed', 0)}")
        logger.info(f"   Agents used: {coordination_result.get('agents_used', 0)}")
        
    except Exception as e:
        logger.error(f" Orchestration strategy example failed: {e}")


def demonstrate_advanced_patterns():
    """
    Demostrar patrones avanzados combinando Factory y Strategy.
    """
    
    logger.info("\n ADVANCED PATTERN COMBINATIONS")
    logger.info("=" * 60)
    
    if not PATTERNS_AVAILABLE:
        logger.warning(" Pattern implementations not available")
        return
    
    # ========================================================================
    # Example 1: Collaborative Agent System
    # ========================================================================
    
    logger.info("\n1️⃣ Collaborative Agent System")
    logger.info("-" * 40)
    
    try:
        # Create specialized agent team using Factory pattern
        factory = StrategyBasedAgentFactory()
        
        team = {
            "project_manager": factory.create_agent_from_template("reasoning_specialist", {
                "name": "project_manager",
                "planning_horizon": 10
            }),
            "researcher": factory.create_agent_from_template("research_analyst", {
                "name": "researcher",
                "max_sources": 12
            }),
            "developer": factory.create_agent_from_template("coding_developer", {
                "name": "developer",
                "languages": ["python", "javascript", "rust"]
            }),
            "coordinator": factory.create_agent(AgentBehaviorType.COMMUNICATION, {
                "name": "coordinator",
                "enable_broadcasting": True
            })
        }
        
        logger.info(f" Created collaborative team: {len(team)} specialized agents")
        
        # Simulate collaborative project
        project_task = "Create an AI-powered code review system"
        
        # Each agent contributes according to their specialty
        contributions = {}
        
        for role, agent in team.items():
            if role == "project_manager":
                task_desc = f"Plan the development of: {project_task}"
            elif role == "researcher":
                task_desc = f"Research best practices for: {project_task}"
            elif role == "developer":
                task_desc = f"Implement core components for: {project_task}"
            elif role == "coordinator":
                task_desc = f"Coordinate team communication for: {project_task}"
            
            context = AgentContext(
                task_id=f"{role}_contribution",
                task_description=task_desc,
                requirements={"collaboration": True}
            )
            
            result = agent.execute(context)
            contributions[role] = {
                "status": result.status,
                "execution_time": result.execution_time_ms,
                "confidence": result.confidence
            }
        
        logger.info(f"\n Team Contributions:")
        for role, contrib in contributions.items():
            status_icon = "" if contrib["status"] == "success" else ""
            logger.info(f"   {role}: {status_icon} {contrib['status']} ({contrib['execution_time']:.1f}ms)")
        
        # Calculate team performance
        successful = sum(1 for c in contributions.values() if c["status"] == "success")
        avg_confidence = sum(c["confidence"] for c in contributions.values()) / len(contributions)
        
        logger.info(f"\n Team Performance:")
        logger.info(f"   Success rate: {successful}/{len(contributions)} ({successful/len(contributions):.1%})")
        logger.info(f"   Average confidence: {avg_confidence:.2f}")
        
    except Exception as e:
        logger.error(f" Collaborative system example failed: {e}")
    
    # ========================================================================
    # Example 2: Adaptive Processing Pipeline
    # ========================================================================
    
    logger.info("\n2️⃣ Adaptive Processing Pipeline")
    logger.info("-" * 40)
    
    try:
        # Create pipeline with different strategies for different stages
        behavior_factory = BehaviorFactory()
        
        pipeline_stages = [
            ("input_analysis", behavior_factory.create_behavior(
                AgentBehaviorType.REASONING, {"reasoning_depth": 2}
            )),
            ("data_research", behavior_factory.create_behavior(
                AgentBehaviorType.RESEARCH, {"max_sources": 5, "quality_threshold": 0.8}
            )),
            ("solution_planning", behavior_factory.create_behavior(
                AgentBehaviorType.PLANNING, {"planning_horizon": 3}
            )),
            ("implementation", behavior_factory.create_behavior(
                AgentBehaviorType.EXECUTION, {"max_retries": 2}
            )),
            ("quality_monitoring", behavior_factory.create_behavior(
                AgentBehaviorType.MONITORING, {"monitoring_interval": 1}
            ))
        ]
        
        logger.info(f" Created adaptive pipeline: {len(pipeline_stages)} stages")
        
        # Create agent that can use different strategies
        agent_factory = StrategyBasedAgentFactory()
        pipeline_agent = agent_factory.create_agent(AgentBehaviorType.EXECUTION)
        
        # Add all pipeline behaviors
        for stage_name, behavior in pipeline_stages:
            pipeline_agent.add_behavior(behavior)
        
        # Process through pipeline
        pipeline_input = "Create a recommendation system for an e-commerce platform"
        pipeline_results = {}
        
        for stage_name, behavior in pipeline_stages:
            context = AgentContext(
                task_id=f"pipeline_{stage_name}",
                task_description=f"{stage_name.replace('_', ' ').title()}: {pipeline_input}",
                requirements={"pipeline_stage": stage_name}
            )
            
            # Switch to appropriate behavior
            pipeline_agent._current_behavior = behavior
            result = pipeline_agent.execute(context)
            
            pipeline_results[stage_name] = {
                "status": result.status,
                "time": result.execution_time_ms,
                "behavior": behavior.behavior_type.value
            }
        
        logger.info(f"\n Pipeline Execution:")
        total_time = 0
        for stage, result in pipeline_results.items():
            status_icon = "" if result["status"] == "success" else ""
            logger.info(f"   {stage}: {status_icon} {result['behavior']} ({result['time']:.1f}ms)")
            total_time += result["time"]
        
        logger.info(f"\n️ Pipeline Performance:")
        logger.info(f"   Total execution time: {total_time:.1f}ms")
        logger.info(f"   Average stage time: {total_time/len(pipeline_stages):.1f}ms")
        
    except Exception as e:
        logger.error(f" Adaptive pipeline example failed: {e}")
    
    # ========================================================================
    # Example 3: Self-Optimizing System
    # ========================================================================
    
    logger.info("\n3️⃣ Self-Optimizing System")
    logger.info("-" * 40)
    
    try:
        # Create system that adapts its strategies based on performance
        factory = StrategyBasedAgentFactory()
        
        # Create monitoring agent
        monitor_agent = factory.create_agent(AgentBehaviorType.MONITORING, {
            "monitoring_interval": 1,
            "alert_thresholds": {
                "response_time_ms": 200,
                "success_rate": 0.8
            }
        })
        
        # Create adaptive worker agent
        worker_agent = factory.create_agent_from_spec({
            "name": "adaptive_worker",
            "type": "execution",
            "behaviors": ["execution", "reasoning", "planning"],
            "config": {"max_retries": 1}
        })
        
        logger.info(" Created self-optimizing system")
        logger.info(f"   Monitor: {monitor_agent.agent_id}")
        logger.info(f"   Worker: {worker_agent.agent_id}")
        
        # Simulate workload with performance tracking
        workload_tasks = [
            "Process data batch 1",
            "Process data batch 2", 
            "Process data batch 3",
            "Process complex analysis",
            "Process data batch 4"
        ]
        
        performance_history = []
        
        for i, task in enumerate(workload_tasks, 1):
            context = AgentContext(
                task_id=f"workload_{i}",
                task_description=task,
                requirements={}
            )
            
            # Execute task
            start_time = time.time()
            result = worker_agent.execute(context)
            execution_time = (time.time() - start_time) * 1000
            
            # Record performance
            performance = {
                "task": i,
                "success": result.status == "success",
                "time": execution_time,
                "confidence": result.confidence
            }
            performance_history.append(performance)
            
            # Monitor performance (simulate monitoring)
            monitor_context = AgentContext(
                task_id=f"monitor_{i}",
                task_description=f"Monitor performance of task {i}",
                requirements={"performance_data": performance}
            )
            
            monitor_result = monitor_agent.execute(monitor_context)
            
            # Simulate adaptation based on performance
            if execution_time > 200:  # Threshold
                logger.info(f"   Task {i}: {result.status} ({execution_time:.1f}ms) - Adapting strategy")
                # In real system, would adjust behavior configuration
            else:
                logger.info(f"   Task {i}: {result.status} ({execution_time:.1f}ms) - Optimal")
        
        # Calculate system performance
        success_rate = sum(1 for p in performance_history if p["success"]) / len(performance_history)
        avg_time = sum(p["time"] for p in performance_history) / len(performance_history)
        avg_confidence = sum(p["confidence"] for p in performance_history) / len(performance_history)
        
        logger.info(f"\n System Performance:")
        logger.info(f"   Success rate: {success_rate:.1%}")
        logger.info(f"   Average time: {avg_time:.1f}ms")
        logger.info(f"   Average confidence: {avg_confidence:.2f}")
        
        # Show adaptation recommendations
        if avg_time > 150:
            logger.info("    Recommendation: Optimize execution strategy")
        if success_rate < 0.9:
            logger.error("    Recommendation: Increase error handling")
        if avg_confidence < 0.8:
            logger.info("    Recommendation: Improve reasoning depth")
        
    except Exception as e:
        logger.error(f" Self-optimizing system example failed: {e}")


def run_all_examples():
    """
    Ejecutar todos los ejemplos de patrones Factory y Strategy.
    """
    
    logger.info(" CAPIBARA AGENT PATTERNS DEMONSTRATION")
    logger.info("=" * 70)
    logger.info("Demostrando el uso de patrones Factory y Strategy en el system de agentes")
    logger.info()
    
    try:
        # Run Factory Pattern examples
        demonstrate_factory_patterns()
        
        # Run Strategy Pattern examples  
        demonstrate_strategy_patterns()
        
        # Run Advanced Pattern combinations
        demonstrate_advanced_patterns()
        
        logger.info("\n" + "=" * 70)
        logger.info(" ALL PATTERN DEMONSTRATIONS COMPLETED")
        logger.info("=" * 70)
        
        # Summary
        if PATTERNS_AVAILABLE:
            logger.info("\n SUMMARY:")
            logger.info(" Factory Pattern: Flexible agent creation with multiple strategies")
            logger.info(" Strategy Pattern: Dynamic behavior switching and composition")
            logger.info(" Combined Patterns: Advanced collaborative and adaptive systems")
            logger.info("\n The Factory and Strategy patterns provide:")
            logger.info("   - Flexible agent creation and configuration")
            logger.info("   - Dynamic behavior adaptation based on context")
            logger.info("   - Scalable multi-agent coordination")
            logger.info("   - Maintainable and extensible architecture")
        else:
            logger.warning("\n️ Pattern implementations not available in this environment")
            logger.info("   Install the required dependencies to see full demonstrations")
        
    except Exception as e:
        logger.error(f"\n Demonstration failed: {e}")
        logger.error(f"Pattern demonstration error: {e}")


# ============================================================================
# Utility Functions for Examples
# ============================================================================

def create_example_context(task_description: str, **kwargs) -> 'AgentContext':
    """Create example context for testing."""
    if PATTERNS_AVAILABLE:
        return AgentContext(
            task_id=f"example_{int(time.time() * 1000)}",
            task_description=task_description,
            requirements=kwargs
        )
    else:
        # Fallback for demo mode
        return type('AgentContext', (), {
            'task_id': f"example_{int(time.time() * 1000)}",
            'task_description': task_description,
            'requirements': kwargs
        })()


def print_agent_info(agent, title: str = "Agent Info"):
    """Print detailed agent information."""
    logger.info(f"\n{title}:")
    
    if hasattr(agent, 'agent_id'):
        logger.info(f"   ID: {agent.agent_id}")
        logger.info(f"   Type: {agent.agent_type}")
        logger.info(f"   Capabilities: {len(agent.capabilities)} total")
        
        if hasattr(agent, 'get_status'):
            status = agent.get_status()
            logger.info(f"   Status: {status.get('status', 'unknown')}")
            
            if 'metrics' in status:
                metrics = status['metrics']
                logger.info(f"   Executions: {metrics.get('tasks_executed', 0)}")
                logger.info(f"   Success rate: {metrics.get('successful_executions', 0)}/{metrics.get('tasks_executed', 1)}")
    else:
        logger.info(f"   Legacy agent: {getattr(agent, 'name', 'unknown')}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Configure logging for examples
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run all pattern demonstrations
    run_all_examples()