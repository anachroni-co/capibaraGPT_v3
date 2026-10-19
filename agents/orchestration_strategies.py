"""
Orchestration Strategies - Strategy Pattern for Agent Coordination - CapibaraGPT v3024
=======================================================================================

Implementation of Strategy pattern for different orchestration approaches:
- IntelligentOrchestrationStrategy: AI-based intelligent coordination
- HierarchicalOrchestrationStrategy: Top-down hierarchical management
- CollaborativeOrchestrationStrategy: Peer-to-peer cooperation
- SpecializedOrchestrationStrategy: Task-specific routing
- AdaptiveOrchestrationStrategy: Dynamic adaptation
- UltraHybridOrchestrationStrategy: Ultra-advanced hybrid strategy
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

# Safe imports
try:
    from ..interfaces.iagent import (
        IOrchestrationStrategy, IAgent, AgentBehaviorType, 
        AgentContext, AgentResult, AgentCapability
    )
except ImportError:
    # Fallback imports
    from abc import ABC, abstractmethod
    from enum import Enum
    from dataclasses import dataclass
    from typing import Dict, Any, List, Optional
    
    class AgentBehaviorType(str, Enum):
        REASONING = "reasoning"
        PLANNING = "planning"
        EXECUTION = "execution"
        RESEARCH = "research"
        CODING = "coding"
        COMMUNICATION = "communication"
        MONITORING = "monitoring"
    
    class IOrchestrationStrategy(ABC):
        @abstractmethod
        def plan_execution(self, task_description, requirements, available_agents): pass
        @abstractmethod
        def coordinate_execution(self, execution_plan, agents): pass
        @abstractmethod
        def handle_agent_failure(self, failed_agent, context, available_agents): pass
    
    class IAgent(ABC):
        @property
        @abstractmethod
        def agent_id(self): pass
        @property  
        @abstractmethod
        def agent_type(self): pass
    
    @dataclass
    class AgentContext:
        task_id: str
        task_description: str
        requirements: Dict[str, Any]

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures for Orchestration
# ============================================================================

@dataclass
class TaskDecomposition:
    """Task decomposition for orchestration."""
    task_id: str
    subtasks: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    priority_order: List[str]
    estimated_duration: Dict[str, float]
    required_agents: Dict[str, AgentBehaviorType]


@dataclass
class ExecutionPlan:
    """Execution plan for agent coordination."""
    plan_id: str
    strategy_name: str
    task_decomposition: TaskDecomposition
    agent_assignments: Dict[str, str]  # subtask_id -> agent_id
    execution_phases: List[Dict[str, Any]]
    coordination_protocol: str
    success_criteria: List[str]
    fallback_plans: List[Dict[str, Any]]


@dataclass
class CoordinationEvent:
    """Coordination event during execution."""
    event_id: str
    event_type: str  # start, progress, completion, failure, coordination
    timestamp: float
    participants: List[str]
    data: Dict[str, Any]
    success: bool = True


# ============================================================================
# Base Orchestration Strategy
# ============================================================================

class BaseOrchestrationStrategy(IOrchestrationStrategy):
    """Base class for all orchestration strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "plans_created": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "agent_utilization": 0.0
        }
        
        # Common configuration
        self.max_concurrent_tasks = self.config.get("max_concurrent_tasks", 5)
        self.coordination_timeout = self.config.get("coordination_timeout", 60)
        self.enable_fallback_plans = self.config.get("enable_fallback_plans", True)

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Name of the strategy."""
        pass

    @property
    @abstractmethod
    def supported_agent_types(self) -> List[AgentBehaviorType]:
        """Supported agent types."""
        pass
    
    def configure_strategy(self, config: Dict[str, Any]) -> None:
        """Configure strategy."""
        self.config.update(config)
        logger.info(f"Configured {self.strategy_name} with {len(config)} parameters")
    
    def _decompose_task(self, task_description: str, requirements: Dict[str, Any]) -> TaskDecomposition:
        """Decompose task into manageable subtasks."""

        # Basic task analysis
        task_complexity = self._assess_task_complexity(task_description, requirements)
        task_type = self._classify_task_type(task_description)

        # Generate subtasks based on type
        subtasks = self._generate_subtasks(task_description, task_type, task_complexity)

        # Identify dependencies
        dependencies = self._identify_dependencies(subtasks)

        # Establish priorities
        priority_order = self._prioritize_subtasks(subtasks, dependencies)

        # Estimate durations
        estimated_duration = self._estimate_durations(subtasks, task_complexity)

        # Assign required agent types
        required_agents = self._assign_required_agent_types(subtasks)
        
        return TaskDecomposition(
            task_id=f"task_{int(time.time() * 1000)}",
            subtasks=subtasks,
            dependencies=dependencies,
            priority_order=priority_order,
            estimated_duration=estimated_duration,
            required_agents=required_agents
        )
    
    def _assess_task_complexity(self, task_description: str, requirements: Dict[str, Any]) -> str:
        """Evaluate task complexity."""
        factors = []

        # Description length
        if len(task_description.split()) > 30:
            factors.append("verbose_description")

        # Specific requirements
        if len(requirements) > 5:
            factors.append("many_requirements")

        # Complexity keywords
        complex_keywords = ["optimize", "integrate", "analyze", "comprehensive", "advanced"]
        if any(keyword in task_description.lower() for keyword in complex_keywords):
            factors.append("complex_keywords")

        # Determine level
        if len(factors) >= 3:
            return "high"
        elif len(factors) >= 1:
            return "medium"
        else:
            return "low"

    def _classify_task_type(self, task_description: str) -> str:
        """Classify task type."""
        task_lower = task_description.lower()
        
        if any(keyword in task_lower for keyword in ["code", "program", "implement", "develop"]):
            return "coding"
        elif any(keyword in task_lower for keyword in ["research", "analyze", "study", "investigate"]):
            return "research"
        elif any(keyword in task_lower for keyword in ["plan", "strategy", "design", "organize"]):
            return "planning"
        elif any(keyword in task_lower for keyword in ["coordinate", "manage", "communicate"]):
            return "coordination"
        elif any(keyword in task_lower for keyword in ["monitor", "track", "observe", "measure"]):
            return "monitoring"
        else:
            return "general"
    
    def _generate_subtasks(self, task_description: str, task_type: str, complexity: str) -> List[Dict[str, Any]]:
        """Generate subtasks based on type and complexity."""
        subtasks = []
        
        if task_type == "coding":
            subtasks = [
                {"id": "code_1", "name": "Analyze requirements", "type": "analysis"},
                {"id": "code_2", "name": "Design architecture", "type": "design"},
                {"id": "code_3", "name": "Implement core functionality", "type": "implementation"},
                {"id": "code_4", "name": "Add error handling", "type": "enhancement"},
                {"id": "code_5", "name": "Test implementation", "type": "testing"},
                {"id": "code_6", "name": "Optimize performance", "type": "optimization"}
            ]
        elif task_type == "research":
            subtasks = [
                {"id": "research_1", "name": "Define research scope", "type": "planning"},
                {"id": "research_2", "name": "Gather information", "type": "collection"},
                {"id": "research_3", "name": "Validate sources", "type": "validation"},
                {"id": "research_4", "name": "Analyze findings", "type": "analysis"},
                {"id": "research_5", "name": "Synthesize results", "type": "synthesis"}
            ]
        elif task_type == "planning":
            subtasks = [
                {"id": "plan_1", "name": "Analyze current state", "type": "analysis"},
                {"id": "plan_2", "name": "Define objectives", "type": "goal_setting"},
                {"id": "plan_3", "name": "Identify resources", "type": "resource_planning"},
                {"id": "plan_4", "name": "Create timeline", "type": "scheduling"},
                {"id": "plan_5", "name": "Define success metrics", "type": "metrics"}
            ]
        else:
            # Generic subtasks
            subtasks = [
                {"id": "generic_1", "name": "Understand task", "type": "analysis"},
                {"id": "generic_2", "name": "Plan approach", "type": "planning"},
                {"id": "generic_3", "name": "Execute work", "type": "execution"},
                {"id": "generic_4", "name": "Validate results", "type": "validation"}
            ]

        # Adjust based on complexity
        if complexity == "low":
            subtasks = subtasks[:3]  # Simplify
        elif complexity == "high":
            # Add additional subtasks
            subtasks.extend([
                {"id": f"{task_type}_extra_1", "name": "Additional validation", "type": "validation"},
                {"id": f"{task_type}_extra_2", "name": "Documentation", "type": "documentation"}
            ])
        
        return subtasks
    
    def _identify_dependencies(self, subtasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Identify dependencies between subtasks."""
        dependencies = {}

        for i, subtask in enumerate(subtasks):
            subtask_id = subtask["id"]
            subtask_type = subtask.get("type", "unknown")
            deps = []

            # Basic dependency rules
            if subtask_type == "implementation" and i > 0:
                # Implementation depends on design/analysis
                for prev_task in subtasks[:i]:
                    if prev_task.get("type") in ["analysis", "design", "planning"]:
                        deps.append(prev_task["id"])

            elif subtask_type == "testing" and i > 0:
                # Testing depends on implementation
                for prev_task in subtasks[:i]:
                    if prev_task.get("type") == "implementation":
                        deps.append(prev_task["id"])

            elif subtask_type == "optimization":
                # Optimization depends on implementation and testing
                for prev_task in subtasks[:i]:
                    if prev_task.get("type") in ["implementation", "testing"]:
                        deps.append(prev_task["id"])

            elif subtask_type == "synthesis":
                # Synthesis depends on analysis
                for prev_task in subtasks[:i]:
                    if prev_task.get("type") == "analysis":
                        deps.append(prev_task["id"])

            dependencies[subtask_id] = deps
        
        return dependencies
    
    def _prioritize_subtasks(self, subtasks: List[Dict[str, Any]], dependencies: Dict[str, List[str]]) -> List[str]:
        """Establish priority order of subtasks."""

        # Simple topological sorting
        priority_order = []
        remaining_tasks = {task["id"]: task for task in subtasks}

        while remaining_tasks:
            # Find tasks without pending dependencies
            ready_tasks = []
            for task_id, task in remaining_tasks.items():
                deps = dependencies.get(task_id, [])
                if all(dep_id in priority_order for dep_id in deps):
                    ready_tasks.append(task_id)

            if not ready_tasks:
                # Break cycles by taking any remaining task
                ready_tasks = [list(remaining_tasks.keys())[0]]

            # Add ready tasks by type priority
            type_priority = {
                "analysis": 1,
                "planning": 2,
                "design": 3,
                "implementation": 4,
                "testing": 5,
                "optimization": 6,
                "validation": 7,
                "synthesis": 8,
                "documentation": 9
            }
            
            ready_tasks.sort(key=lambda tid: type_priority.get(
                remaining_tasks[tid].get("type", "unknown"), 5
            ))

            # Add to priority order
            for task_id in ready_tasks:
                priority_order.append(task_id)
                del remaining_tasks[task_id]
        
        return priority_order
    
    def _estimate_durations(self, subtasks: List[Dict[str, Any]], complexity: str) -> Dict[str, float]:
        """Estimate subtask duration in seconds."""

        base_durations = {
            "analysis": 120,      # 2 minutes
            "planning": 180,      # 3 minutes
            "design": 240,        # 4 minutes
            "implementation": 300, # 5 minutes
            "testing": 180,       # 3 minutes
            "optimization": 240,  # 4 minutes
            "validation": 120,    # 2 minutes
            "synthesis": 180,     # 3 minutes
            "documentation": 120  # 2 minutes
        }

        # Complexity factors
        complexity_multipliers = {
            "low": 0.7,
            "medium": 1.0,
            "high": 1.5
        }
        
        multiplier = complexity_multipliers.get(complexity, 1.0)
        
        estimated_duration = {}
        for subtask in subtasks:
            task_type = subtask.get("type", "unknown")
            base_time = base_durations.get(task_type, 180)
            estimated_duration[subtask["id"]] = base_time * multiplier
        
        return estimated_duration
    
    def _assign_required_agent_types(self, subtasks: List[Dict[str, Any]]) -> Dict[str, AgentBehaviorType]:
        """Assign required agent types for each subtask."""
        
        type_mapping = {
            "analysis": AgentBehaviorType.REASONING,
            "planning": AgentBehaviorType.PLANNING,
            "design": AgentBehaviorType.REASONING,
            "implementation": AgentBehaviorType.EXECUTION,
            "testing": AgentBehaviorType.EXECUTION,
            "optimization": AgentBehaviorType.REASONING,
            "validation": AgentBehaviorType.EXECUTION,
            "synthesis": AgentBehaviorType.REASONING,
            "documentation": AgentBehaviorType.EXECUTION,
            "collection": AgentBehaviorType.RESEARCH,
            "coordination": AgentBehaviorType.COMMUNICATION,
            "monitoring": AgentBehaviorType.MONITORING
        }
        
        required_agents = {}
        for subtask in subtasks:
            task_type = subtask.get("type", "unknown")
            agent_type = type_mapping.get(task_type, AgentBehaviorType.EXECUTION)
            required_agents[subtask["id"]] = agent_type
        
        return required_agents
    
    def _select_agents_for_plan(self, task_decomposition: TaskDecomposition, available_agents: List[IAgent]) -> Dict[str, str]:
        """Select agents for the execution plan."""

        agent_assignments = {}
        used_agents = set()

        # Group agents by type
        agents_by_type = defaultdict(list)
        for agent in available_agents:
            agents_by_type[agent.agent_type].append(agent)

        # Assign agents according to subtask priority
        for subtask_id in task_decomposition.priority_order:
            required_type = task_decomposition.required_agents.get(subtask_id)

            # Find available agent of required type
            suitable_agents = agents_by_type.get(required_type, [])
            available_suitable = [a for a in suitable_agents if a.agent_id not in used_agents]

            if available_suitable:
                selected_agent = available_suitable[0]
                agent_assignments[subtask_id] = selected_agent.agent_id
                used_agents.add(selected_agent.agent_id)
            else:
                # Find available agent of any type
                for agent_type, agents in agents_by_type.items():
                    available_any = [a for a in agents if a.agent_id not in used_agents]
                    if available_any:
                        selected_agent = available_any[0]
                        agent_assignments[subtask_id] = selected_agent.agent_id
                        used_agents.add(selected_agent.agent_id)
                        break
        
        return agent_assignments
    
    def _create_execution_phases(self, task_decomposition: TaskDecomposition, agent_assignments: Dict[str, str]) -> List[Dict[str, Any]]:
        """Create execution phases based on dependencies."""
        
        phases = []
        processed_tasks = set()
        phase_number = 1
        
        while len(processed_tasks) < len(task_decomposition.subtasks):
            # Find tasks that can be executed in this phase
            ready_tasks = []

            for subtask_id in task_decomposition.priority_order:
                if subtask_id in processed_tasks:
                    continue

                # Verify all dependencies are processed
                dependencies = task_decomposition.dependencies.get(subtask_id, [])
                if all(dep in processed_tasks for dep in dependencies):
                    ready_tasks.append(subtask_id)

            if not ready_tasks:
                # Break deadlock
                remaining = [sid for sid in task_decomposition.priority_order if sid not in processed_tasks]
                if remaining:
                    ready_tasks = [remaining[0]]
            
            # Create phase
            phase_tasks = []
            for task_id in ready_tasks:
                task_info = next(t for t in task_decomposition.subtasks if t["id"] == task_id)
                phase_tasks.append({
                    "subtask_id": task_id,
                    "subtask_name": task_info["name"],
                    "assigned_agent": agent_assignments.get(task_id),
                    "estimated_duration": task_decomposition.estimated_duration.get(task_id, 180)
                })
                processed_tasks.add(task_id)
            
            phases.append({
                "phase_number": phase_number,
                "phase_name": f"Phase {phase_number}",
                "tasks": phase_tasks,
                "can_parallel": len(phase_tasks) > 1,
                "estimated_duration": max(t["estimated_duration"] for t in phase_tasks) if phase_tasks else 0
            })
            
            phase_number += 1
        
        return phases
    
    def _update_performance_metrics(self, execution_result: Dict[str, Any]) -> None:
        """Update performance metrics."""

        self.performance_metrics["plans_created"] += 1

        if execution_result.get("status") == "success":
            self.performance_metrics["successful_executions"] += 1
        else:
            self.performance_metrics["failed_executions"] += 1

        # Update average execution time
        execution_time = execution_result.get("execution_time", 0)
        current_avg = self.performance_metrics["average_execution_time"]
        total_plans = self.performance_metrics["plans_created"]
        
        self.performance_metrics["average_execution_time"] = (
            (current_avg * (total_plans - 1) + execution_time) / total_plans
        )
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get strategy metrics."""
        return {
            "strategy_name": self.strategy_name,
            "performance_metrics": self.performance_metrics.copy(),
            "executions_in_history": len(self.execution_history),
            "success_rate": (
                self.performance_metrics["successful_executions"] / 
                max(1, self.performance_metrics["plans_created"])
            )
        }


# ============================================================================
# Intelligent Orchestration Strategy
# ============================================================================

class IntelligentOrchestrationStrategy(BaseOrchestrationStrategy):
    """AI-based intelligent orchestration strategy."""
    
    @property
    def strategy_name(self) -> str:
        return "intelligent"
    
    @property
    def supported_agent_types(self) -> List[AgentBehaviorType]:
        return list(AgentBehaviorType)
    
    def plan_execution(
        self, 
        task_description: str,
        requirements: Dict[str, Any],
        available_agents: List[IAgent]
    ) -> Dict[str, Any]:
        """Plan execution using adaptive intelligence."""

        # Decompose task intelligently
        task_decomposition = self._decompose_task(task_description, requirements)

        # Intelligent analysis of available agents
        agent_analysis = self._analyze_available_agents(available_agents)

        # Intelligent agent assignment
        agent_assignments = self._intelligent_agent_assignment(task_decomposition, available_agents, agent_analysis)

        # Create optimized execution phases
        execution_phases = self._create_optimized_phases(task_decomposition, agent_assignments)

        # Generate coordination protocol
        coordination_protocol = self._generate_coordination_protocol(task_decomposition, agent_assignments)
        
        # Create contingency plans
        fallback_plans = []
        if self.enable_fallback_plans:
            fallback_plans = self._create_intelligent_fallback_plans(task_decomposition, available_agents)
        
        execution_plan = ExecutionPlan(
            plan_id=f"intelligent_{int(time.time() * 1000)}",
            strategy_name=self.strategy_name,
            task_decomposition=task_decomposition,
            agent_assignments=agent_assignments,
            execution_phases=execution_phases,
            coordination_protocol=coordination_protocol,
            success_criteria=self._define_success_criteria(task_decomposition),
            fallback_plans=fallback_plans
        )
        
        return {
            "execution_plan": execution_plan,
            "estimated_duration": sum(phase["estimated_duration"] for phase in execution_phases),
            "agent_utilization": len(agent_assignments) / len(available_agents),
            "confidence": self._calculate_plan_confidence(execution_plan, available_agents)
        }
    
    def coordinate_execution(self, execution_plan: Dict[str, Any], agents: List[IAgent]) -> Dict[str, Any]:
        """Coordinate execution with intelligent monitoring."""
        
        plan = execution_plan.get("execution_plan")
        if not plan:
            return {"status": "error", "message": "No execution plan provided"}
        
        coordination_events = []
        execution_results = {}
        start_time = time.time()
        
        try:
            # Execute phases sequentially
            for phase in plan.execution_phases:
                phase_result = self._execute_phase_intelligently(phase, agents)
                execution_results[f"phase_{phase['phase_number']}"] = phase_result
                
                # Record coordination event
                coordination_events.append(CoordinationEvent(
                    event_id=f"phase_{phase['phase_number']}_completed",
                    event_type="phase_completion",
                    timestamp=time.time(),
                    participants=[task["assigned_agent"] for task in phase["tasks"] if task["assigned_agent"]],
                    data=phase_result,
                    success=phase_result.get("status") == "success"
                ))
                
                # Check if the phase failed
                if phase_result.get("status") != "success":
                    logger.warning(f"Phase {phase['phase_number']} failed, applying recovery strategy")
                    recovery_result = self._apply_recovery_strategy(phase, agents)
                    if recovery_result.get("status") != "success":
                        break
            
            # Evaluate overall success
            overall_success = all(
                result.get("status") == "success" 
                for result in execution_results.values()
            )
            
            execution_time = time.time() - start_time
            
            result = {
                "status": "success" if overall_success else "partial_failure",
                "execution_results": execution_results,
                "coordination_events": [vars(event) for event in coordination_events],
                "execution_time": execution_time,
                "agents_used": len(set(
                    task["assigned_agent"] 
                    for phase in plan.execution_phases 
                    for task in phase["tasks"] 
                    if task["assigned_agent"]
                )),
                "phases_completed": len(execution_results)
            }
            
            self._update_performance_metrics(result)
            return result
            
        except Exception as e:
            logger.error(f"Intelligent coordination failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time,
                "coordination_events": [vars(event) for event in coordination_events]
            }
    
    def handle_agent_failure(
        self, 
        failed_agent: IAgent,
        context: AgentContext,
        available_agents: List[IAgent]
    ) -> Dict[str, Any]:
        """Handle agent failure with intelligent strategies."""

        # Analyze failure cause
        failure_analysis = self._analyze_agent_failure(failed_agent, context)

        # Find replacement agent
        replacement_agent = self._find_replacement_agent(failed_agent, available_agents, failure_analysis)
        
        # Create recovery strategy
        recovery_strategy = self._create_recovery_strategy(failed_agent, replacement_agent, context, failure_analysis)
        
        return {
            "failure_handled": True,
            "failure_analysis": failure_analysis,
            "replacement_agent": replacement_agent.agent_id if replacement_agent else None,
            "recovery_strategy": recovery_strategy,
            "estimated_recovery_time": recovery_strategy.get("estimated_time", 60)
        }
    
    def _analyze_available_agents(self, agents: List[IAgent]) -> Dict[str, Any]:
        """Intelligent analysis of available agents."""

        analysis = {
            "total_agents": len(agents),
            "agents_by_type": defaultdict(list),
            "capability_coverage": defaultdict(int),
            "agent_scores": {}
        }

        for agent in agents:
            agent_type = agent.agent_type
            analysis["agents_by_type"][agent_type].append(agent.agent_id)

            # Analyze capabilities
            capabilities = getattr(agent, 'capabilities', [])
            for capability in capabilities:
                analysis["capability_coverage"][capability] += 1

            # Calculate agent score (simplified)
            agent_score = self._calculate_agent_score(agent)
            analysis["agent_scores"][agent.agent_id] = agent_score
        
        return analysis
    
    def _calculate_agent_score(self, agent: IAgent) -> float:
        """Calculate agent quality score."""

        base_score = 0.5

        # Bonus for specialized type
        type_bonuses = {
            AgentBehaviorType.REASONING: 0.2,
            AgentBehaviorType.CODING: 0.15,
            AgentBehaviorType.RESEARCH: 0.15,
            AgentBehaviorType.PLANNING: 0.1
        }
        
        base_score += type_bonuses.get(agent.agent_type, 0.05)

        # Bonus for number of capabilities
        capabilities = getattr(agent, 'capabilities', [])
        base_score += len(capabilities) * 0.02

        # Bonus for status (if available)
        status = getattr(agent, 'get_status', lambda: {})()
        if status.get("status") == "ready":
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def _intelligent_agent_assignment(
        self, 
        task_decomposition: TaskDecomposition,
        available_agents: List[IAgent],
        agent_analysis: Dict[str, Any]
    ) -> Dict[str, str]:
        """Intelligent agent assignment based on analysis."""

        assignments = {}
        agent_workload = defaultdict(int)

        # Sort subtasks by priority and complexity
        sorted_subtasks = self._sort_subtasks_by_priority(task_decomposition)
        
        for subtask_id in sorted_subtasks:
            required_type = task_decomposition.required_agents.get(subtask_id)
            
            # Find best candidates
            candidates = self._find_best_agent_candidates(
                required_type, available_agents, agent_analysis, agent_workload
            )
            
            if candidates:
                # Select the best candidate
                best_agent = candidates[0]
                assignments[subtask_id] = best_agent.agent_id
                agent_workload[best_agent.agent_id] += 1
                
                logger.debug(f"Assigned {subtask_id} to {best_agent.agent_id} (type: {best_agent.agent_type})")
        
        return assignments
    
    def _sort_subtasks_by_priority(self, task_decomposition: TaskDecomposition) -> List[str]:
        """Sort subtasks by intelligent priority."""

        # Priority factors
        priority_scores = {}

        for subtask_id in task_decomposition.priority_order:
            score = 0

            # Bonus for dependencies (tasks with more dependents are more prioritized)
            dependents = sum(
                1 for deps in task_decomposition.dependencies.values()
                if subtask_id in deps
            )
            score += dependents * 10

            # Bonus for estimated duration (longer tasks first)
            duration = task_decomposition.estimated_duration.get(subtask_id, 0)
            score += duration / 60  # Convert to minutes

            priority_scores[subtask_id] = score

        # Sort by score (higher first)
        return sorted(task_decomposition.priority_order, key=lambda x: priority_scores.get(x, 0), reverse=True)
    
    def _find_best_agent_candidates(
        self, 
        required_type: AgentBehaviorType,
        available_agents: List[IAgent],
        agent_analysis: Dict[str, Any],
        current_workload: Dict[str, int]
    ) -> List[IAgent]:
        """Find the best candidates for a task."""

        candidates = []

        # Filter by preferred type
        preferred_agents = [a for a in available_agents if a.agent_type == required_type]

        # If none of preferred type, use any agent
        if not preferred_agents:
            preferred_agents = available_agents

        # Calculate score for each candidate
        scored_candidates = []
        for agent in preferred_agents:
            score = agent_analysis["agent_scores"].get(agent.agent_id, 0.5)

            # Penalize for current workload
            workload_penalty = current_workload.get(agent.agent_id, 0) * 0.1
            score -= workload_penalty

            # Bonus for exact type
            if agent.agent_type == required_type:
                score += 0.2

            scored_candidates.append((agent, score))

        # Sort by score (higher first)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [agent for agent, score in scored_candidates]
    
    def _create_optimized_phases(
        self, 
        task_decomposition: TaskDecomposition,
        agent_assignments: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Create optimized execution phases."""
        
        phases = self._create_execution_phases(task_decomposition, agent_assignments)

        # Optimize phases for intelligent parallelization
        optimized_phases = []

        for phase in phases:
            # Analyze if tasks can be executed in parallel
            parallel_groups = self._analyze_parallel_potential(phase["tasks"], task_decomposition)

            if len(parallel_groups) > 1:
                # Split into parallel subphases
                for i, group in enumerate(parallel_groups):
                    optimized_phases.append({
                        "phase_number": f"{phase['phase_number']}.{i+1}",
                        "phase_name": f"{phase['phase_name']} - Group {i+1}",
                        "tasks": group,
                        "can_parallel": True,
                        "estimated_duration": max(t["estimated_duration"] for t in group),
                        "parallel_group": i
                    })
            else:
                optimized_phases.append(phase)
        
        return optimized_phases
    
    def _analyze_parallel_potential(
        self, 
        tasks: List[Dict[str, Any]],
        task_decomposition: TaskDecomposition
    ) -> List[List[Dict[str, Any]]]:
        """Analyze task parallelization potential."""
        
        # Simplified: group tasks that have no dependencies between them
        groups = []
        remaining_tasks = tasks.copy()

        while remaining_tasks:
            current_group = []
            used_tasks = []

            for task in remaining_tasks:
                task_id = task["subtask_id"]

                # Check if can be added to current group
                can_add = True
                for group_task in current_group:
                    group_task_id = group_task["subtask_id"]

                    # Check mutual dependencies
                    if (task_id in task_decomposition.dependencies.get(group_task_id, []) or
                        group_task_id in task_decomposition.dependencies.get(task_id, [])):
                        can_add = False
                        break
                
                if can_add:
                    current_group.append(task)
                    used_tasks.append(task)

            # Remove used tasks
            for task in used_tasks:
                remaining_tasks.remove(task)

            if current_group:
                groups.append(current_group)
            else:
                # Avoid infinite loop
                if remaining_tasks:
                    groups.append([remaining_tasks.pop(0)])
        
        return groups
    
    def _generate_coordination_protocol(
        self, 
        task_decomposition: TaskDecomposition,
        agent_assignments: Dict[str, str]
    ) -> str:
        """Generate intelligent coordination protocol."""
        
        total_agents = len(set(agent_assignments.values()))
        total_tasks = len(task_decomposition.subtasks)
        
        if total_agents == 1:
            return "sequential_single_agent"
        elif total_tasks <= 3:
            return "simple_coordination"
        elif total_agents <= 3:
            return "small_team_coordination"
        else:
            return "complex_multi_agent_coordination"
    
    def _create_intelligent_fallback_plans(
        self, 
        task_decomposition: TaskDecomposition,
        available_agents: List[IAgent]
    ) -> List[Dict[str, Any]]:
        """Create intelligent contingency plans."""
        
        fallback_plans = []

        # Plan 1: Reduce parallelism
        fallback_plans.append({
            "plan_id": "reduce_parallelism",
            "trigger": "agent_unavailable",
            "description": "Reduce parallel execution to use fewer agents",
            "modifications": ["serialize_parallel_tasks", "extend_timeline"]
        })

        # Plan 2: Simplify tasks
        fallback_plans.append({
            "plan_id": "simplify_tasks",
            "trigger": "execution_timeout",
            "description": "Simplify complex subtasks to ensure completion",
            "modifications": ["remove_optimization_tasks", "reduce_quality_requirements"]
        })

        # Plan 3: Backup agent
        if len(available_agents) > len(set(task_decomposition.required_agents.values())):
            fallback_plans.append({
                "plan_id": "backup_agent",
                "trigger": "agent_failure",
                "description": "Use backup agent for failed tasks",
                "modifications": ["reassign_to_backup", "retry_failed_tasks"]
            })
        
        return fallback_plans
    
    def _define_success_criteria(self, task_decomposition: TaskDecomposition) -> List[str]:
        """Define intelligent success criteria."""

        criteria = [
            "all_subtasks_completed",
            "no_critical_failures",
            "execution_within_timeout"
        ]

        # Additional criteria based on task types
        task_types = set(task.get("type", "unknown") for task in task_decomposition.subtasks)
        
        if "implementation" in task_types:
            criteria.append("code_functionality_verified")
        
        if "testing" in task_types:
            criteria.append("tests_passed")
        
        if "analysis" in task_types:
            criteria.append("analysis_quality_threshold_met")
        
        return criteria
    
    def _calculate_plan_confidence(self, execution_plan: ExecutionPlan, available_agents: List[IAgent]) -> float:
        """Calculate confidence in the execution plan."""

        confidence = 0.5  # Base

        # Factor for agent coverage
        required_agents = len(execution_plan.agent_assignments)
        available_count = len(available_agents)

        if available_count >= required_agents:
            confidence += 0.2

        # Factor for coordination complexity
        coordination_complexity = len(execution_plan.execution_phases)
        if coordination_complexity <= 3:
            confidence += 0.1
        elif coordination_complexity > 6:
            confidence -= 0.1

        # Factor for contingency plans
        if execution_plan.fallback_plans:
            confidence += 0.1

        # Factor for success history
        success_rate = (
            self.performance_metrics["successful_executions"] / 
            max(1, self.performance_metrics["plans_created"])
        )
        confidence += success_rate * 0.2
        
        return min(1.0, max(0.0, confidence))


# Export all orchestration strategies
__all__ = [
    "BaseOrchestrationStrategy",
    "IntelligentOrchestrationStrategy",
    "TaskDecomposition",
    "ExecutionPlan",
    "CoordinationEvent"
]