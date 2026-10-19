"""
Communication and Monitoring Behaviors - CapibaraGPT v3024
==========================================================

Specialized behaviors for communication and monitoring:
- CommunicationBehavior: Advanced inter-agent communication
- MonitoringBehavior: System monitoring and metrics
- LearningBehavior: Adaptive learning
"""

import time
import json
import logging
import tracemalloc
from typing import Dict, Any, List, Optional, Tuple, Deque

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:
    psutil = None
    _HAS_PSUTIL = False
from collections import defaultdict, deque
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Safe imports
try:
    from ..interfaces.iagent import (
        IAgentBehavior, AgentBehaviorType, AgentCapability,
        AgentContext, AgentResult, IAgent, IAgentCommunication,
        IAgentMonitoring, IAgentLearning
    )
    from .behaviors import BaseBehavior
except ImportError:
    # Fallback imports
    from abc import ABC, abstractmethod
    from enum import Enum
    from dataclasses import dataclass, field
    
    class AgentBehaviorType(str, Enum):
        COMMUNICATION = "communication"
        MONITORING = "monitoring"
        LEARNING = "learning"
    
    class AgentCapability(str, Enum):
        INTER_AGENT_COMMUNICATION = "inter_agent_communication"
        PERFORMANCE_MONITORING = "performance_monitoring"
        ADAPTIVE_LEARNING = "adaptive_learning"
    
    @dataclass
    class AgentContext:
        task_id: str
        task_description: str
        requirements: Dict[str, Any]
    
    @dataclass
    class AgentResult:
        agent_id: str
        status: str
        result: Any
        execution_time_ms: float
        confidence: float = 0.0
        metadata: Dict[str, Any] = field(default_factory=dict)
        error: Optional[str] = None
    
    class BaseBehavior:
        def __init__(self, config=None):
            self.config = config or {}
            self.execution_count = 0
            self.success_count = 0
        
        def _update_metrics(self, time, success):
            self.execution_count += 1
            if success:
                self.success_count += 1
    
    class IAgentCommunication(ABC):
        @abstractmethod
        def send_message(self, recipient, message): pass
    
    class IAgentMonitoring(ABC):
        @abstractmethod
        def get_performance_metrics(self): pass
    
    class IAgentLearning(ABC):
        @abstractmethod
        def learn_from_experience(self, experience): pass

logger = logging.getLogger(__name__)


# ============================================================================
# Message and Communication Data Structures
# ============================================================================

@dataclass
class Message:
    """Message structure for inter-agent communication."""
    id: str
    sender: str
    recipient: str
    message_type: str
    content: Dict[str, Any]
    timestamp: float
    priority: str = "normal"  # low, normal, high, critical
    requires_response: bool = False
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommunicationEvent:
    """Communication event for tracking."""
    event_id: str
    event_type: str  # message_sent, message_received, broadcast, etc.
    participants: List[str]
    timestamp: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Performance metric."""
    metric_name: str
    value: float
    timestamp: float
    agent_id: str
    metric_type: str = "gauge"  # gauge, counter, histogram
    tags: Dict[str, str] = field(default_factory=dict)


# ============================================================================
# Communication Behavior - Strategy for Inter-Agent Communication
# ============================================================================

class CommunicationBehavior(BaseBehavior, IAgentCommunication):
    """Specialized behavior for inter-agent communication."""
    
    @property
    def behavior_type(self) -> AgentBehaviorType:
        return AgentBehaviorType.COMMUNICATION
    
    @property
    def required_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.INTER_AGENT_COMMUNICATION,
            AgentCapability.MESSAGE_ROUTING,
            AgentCapability.PROTOCOL_MANAGEMENT,
            AgentCapability.CONFLICT_RESOLUTION
        ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_message_history = self.config.get("max_message_history", 1000)
        self.default_timeout = self.config.get("default_timeout", 30)
        self.enable_broadcasting = self.config.get("enable_broadcasting", True)
        self.enable_encryption = self.config.get("enable_encryption", False)
        
        # Communication state
        self.message_history: Deque[Message] = deque(maxlen=self.max_message_history)
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.message_queue: Dict[str, List[Message]] = defaultdict(list)
        self.communication_metrics: Dict[str, Any] = {
            "messages_sent": 0,
            "messages_received": 0,
            "broadcasts_sent": 0,
            "failed_deliveries": 0,
            "average_response_time": 0.0
        }
    
    def execute_behavior(self, context: AgentContext, agent: IAgent) -> AgentResult:
        """Execute communication behavior."""
        start_time = time.time()
        
        try:
            # Register agent if not registered
            self._register_agent(agent)
            
            # Process context-based communication
            communication_result = self._handle_communication_task(context, agent)
            
            execution_time = (time.time() - start_time) * 1000
            self._update_metrics(execution_time, communication_result["success"])
            
            return AgentResult(
                agent_id=agent.agent_id,
                status="success" if communication_result["success"] else "failed",
                result=communication_result,
                execution_time_ms=execution_time,
                confidence=communication_result.get("confidence", 0.9),
                metadata={
                    "messages_processed": communication_result.get("messages_processed", 0),
                    "communication_events": communication_result.get("events", [])
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._update_metrics(execution_time, False)
            logger.error(f"Communication behavior failed: {e}")
            
            return AgentResult(
                agent_id=agent.agent_id,
                status="failed",
                result=None,
                execution_time_ms=execution_time,
                error=str(e)
            )
    
    def _handle_communication_task(self, context: AgentContext, agent: IAgent) -> Dict[str, Any]:
        """Handle specific communication task."""
        
        task_description = context.task_description.lower()
        events = []
        messages_processed = 0

        # Determine required communication type
        if "coordinate" in task_description or "collaborate" in task_description:
            result = self._coordinate_agents(context, agent)
            events.append({"type": "coordination", "result": result})
            
        elif "broadcast" in task_description or "announce" in task_description:
            result = self._handle_broadcast(context, agent)
            events.append({"type": "broadcast", "result": result})
            messages_processed += 1
            
        elif "message" in task_description or "communicate" in task_description:
            result = self._handle_direct_communication(context, agent)
            events.append({"type": "direct_message", "result": result})
            messages_processed += 1
            
        elif "resolve" in task_description or "conflict" in task_description:
            result = self._resolve_conflicts(context, agent)
            events.append({"type": "conflict_resolution", "result": result})
            
        else:
            # General communication
            result = self._handle_general_communication(context, agent)
            events.append({"type": "general", "result": result})
        
        # Process pending messages
        pending_messages = self._process_pending_messages(agent.agent_id)
        messages_processed += len(pending_messages)
        
        return {
            "success": True,
            "events": events,
            "messages_processed": messages_processed,
            "pending_messages": pending_messages,
            "confidence": 0.9,
            "communication_quality": self._assess_communication_quality()
        }
    
    def _register_agent(self, agent: IAgent) -> None:
        """Register agent in communication system."""
        agent_id = agent.agent_id
        
        if agent_id not in self.agent_registry:
            self.agent_registry[agent_id] = {
                "agent_type": agent.agent_type,
                "capabilities": agent.capabilities,
                "status": "active",
                "last_seen": time.time(),
                "message_count": 0
            }
            logger.info(f"Registered agent {agent_id} for communication")
    
    def _coordinate_agents(self, context: AgentContext, agent: IAgent) -> Dict[str, Any]:
        """Coordinate multiple agents for collaboration."""

        # Identify agents available for coordination
        available_agents = [aid for aid, info in self.agent_registry.items()
                          if info["status"] == "active" and aid != agent.agent_id]
        
        if not available_agents:
            return {
                "status": "no_agents_available",
                "coordinated_agents": [],
                "coordination_plan": None
            }
        
        # Create coordination plan
        coordination_plan = self._create_coordination_plan(context, available_agents)

        # Send coordination messages
        coordination_messages = []
        for target_agent in available_agents[:3]:  # Limit to 3 agents
            message = Message(
                id=f"coord_{int(time.time() * 1000)}",
                sender=agent.agent_id,
                recipient=target_agent,
                message_type="coordination_request",
                content={
                    "task": context.task_description,
                    "coordination_plan": coordination_plan,
                    "role": coordination_plan.get("roles", {}).get(target_agent, "collaborator")
                },
                timestamp=time.time(),
                requires_response=True,
                conversation_id=context.task_id
            )
            
            self._send_message_internal(message)
            coordination_messages.append(message.id)
        
        return {
            "status": "coordination_initiated",
            "coordinated_agents": available_agents[:3],
            "coordination_plan": coordination_plan,
            "messages_sent": coordination_messages,
            "expected_responses": len(coordination_messages)
        }
    
    def _create_coordination_plan(self, context: AgentContext, agents: List[str]) -> Dict[str, Any]:
        """Create coordination plan for agents."""
        
        # Basic task analysis
        task_complexity = "high" if len(context.task_description.split()) > 20 else "medium"

        # Assign roles based on agent types (simplified)
        roles = {}
        for i, agent_id in enumerate(agents):
            if i == 0:
                roles[agent_id] = "lead_executor"
            elif i == 1:
                roles[agent_id] = "supporter"
            else:
                roles[agent_id] = "monitor"
        
        return {
            "coordination_strategy": "collaborative",
            "task_complexity": task_complexity,
            "roles": roles,
            "communication_protocol": "request_response",
            "sync_intervals": 30,  # seconds
            "success_criteria": ["task_completion", "all_agents_responsive"]
        }
    
    def _handle_broadcast(self, context: AgentContext, agent: IAgent) -> Dict[str, Any]:
        """Handle broadcasting to multiple agents."""
        
        if not self.enable_broadcasting:
            return {"status": "broadcasting_disabled", "recipients": []}
        
        # Create broadcast message
        broadcast_content = {
            "announcement": context.task_description,
            "timestamp": time.time(),
            "priority": context.requirements.get("priority", "normal"),
            "requires_acknowledgment": context.requirements.get("require_ack", False)
        }
        
        # Send to all active agents
        recipients = [aid for aid, info in self.agent_registry.items() 
                     if info["status"] == "active" and aid != agent.agent_id]
        
        broadcast_messages = []
        for recipient in recipients:
            message = Message(
                id=f"broadcast_{int(time.time() * 1000)}_{recipient}",
                sender=agent.agent_id,
                recipient=recipient,
                message_type="broadcast",
                content=broadcast_content,
                timestamp=time.time(),
                priority=broadcast_content["priority"]
            )
            
            self._send_message_internal(message)
            broadcast_messages.append(message.id)
        
        self.communication_metrics["broadcasts_sent"] += 1
        
        return {
            "status": "broadcast_sent",
            "recipients": recipients,
            "message_ids": broadcast_messages,
            "broadcast_content": broadcast_content
        }
    
    def _handle_direct_communication(self, context: AgentContext, agent: IAgent) -> Dict[str, Any]:
        """Handle direct communication between agents."""
        
        # Extract recipient from context (simplified)
        requirements = context.requirements
        recipient = requirements.get("recipient")

        if not recipient:
            # Select recipient automatically
            available_agents = [aid for aid in self.agent_registry.keys() if aid != agent.agent_id]
            recipient = available_agents[0] if available_agents else None
        
        if not recipient:
            return {"status": "no_recipient_available", "message_sent": False}
        
        # Create direct message
        message = Message(
            id=f"direct_{int(time.time() * 1000)}",
            sender=agent.agent_id,
            recipient=recipient,
            message_type="direct_message",
            content={
                "message": context.task_description,
                "context": requirements
            },
            timestamp=time.time(),
            requires_response=requirements.get("requires_response", False)
        )
        
        success = self._send_message_internal(message)
        
        return {
            "status": "message_sent" if success else "message_failed",
            "recipient": recipient,
            "message_id": message.id,
            "message_sent": success
        }
    
    def _resolve_conflicts(self, context: AgentContext, agent: IAgent) -> Dict[str, Any]:
        """Resolve conflicts between agents."""

        # Identify agents in conflict (simplified)
        conflicted_agents = context.requirements.get("conflicted_agents", [])
        
        if not conflicted_agents:
            # Search for agents with recent conflicting activity
            conflicted_agents = self._detect_conflicts()
        
        resolution_strategy = self._determine_resolution_strategy(conflicted_agents)

        # Apply resolution strategy
        resolution_result = self._apply_resolution_strategy(
            resolution_strategy, conflicted_agents, agent
        )
        
        return {
            "status": "conflict_resolution_attempted",
            "conflicted_agents": conflicted_agents,
            "resolution_strategy": resolution_strategy,
            "resolution_result": resolution_result,
            "success": resolution_result.get("resolved", False)
        }
    
    def _detect_conflicts(self) -> List[str]:
        """Detect conflicts between agents (simplified implementation)."""
        # In a real implementation, this would analyze communication patterns,
        # response times, coordination failures, etc.
        return []

    def _determine_resolution_strategy(self, conflicted_agents: List[str]) -> str:
        """Determine conflict resolution strategy."""
        if len(conflicted_agents) <= 2:
            return "mediation"
        else:
            return "consensus_building"
    
    def _apply_resolution_strategy(self, strategy: str, agents: List[str], mediator: IAgent) -> Dict[str, Any]:
        """Apply resolution strategy."""
        if strategy == "mediation":
            return self._mediate_conflict(agents, mediator)
        elif strategy == "consensus_building":
            return self._build_consensus(agents, mediator)
        else:
            return {"resolved": False, "reason": "unknown_strategy"}
    
    def _mediate_conflict(self, agents: List[str], mediator: IAgent) -> Dict[str, Any]:
        """Mediate conflict between agents."""
        # Simplified implementation
        return {
            "resolved": True,
            "method": "mediation",
            "participants": agents,
            "mediator": mediator.agent_id
        }
    
    def _build_consensus(self, agents: List[str], facilitator: IAgent) -> Dict[str, Any]:
        """Build consensus among multiple agents."""
        # Simplified implementation
        return {
            "resolved": True,
            "method": "consensus",
            "participants": agents,
            "facilitator": facilitator.agent_id
        }
    
    def _handle_general_communication(self, context: AgentContext, agent: IAgent) -> Dict[str, Any]:
        """Handle general communication."""
        return {
            "status": "general_communication_handled",
            "agent": agent.agent_id,
            "context_processed": True
        }
    
    def send_message(self, recipient: str, message: Dict[str, Any]) -> bool:
        """Implementation of IAgentCommunication.send_message."""
        msg = Message(
            id=f"api_{int(time.time() * 1000)}",
            sender="api_caller",
            recipient=recipient,
            message_type="api_message",
            content=message,
            timestamp=time.time()
        )
        return self._send_message_internal(msg)
    
    def receive_message(self) -> Optional[Dict[str, Any]]:
        """Implementation of IAgentCommunication.receive_message."""
        # Simplified - real implementation would use agent-specific queues
        if self.message_history:
            latest_message = self.message_history[-1]
            return {
                "id": latest_message.id,
                "sender": latest_message.sender,
                "content": latest_message.content,
                "timestamp": latest_message.timestamp
            }
        return None
    
    def broadcast_message(self, message: Dict[str, Any]) -> List[str]:
        """Implementation of IAgentCommunication.broadcast_message."""
        recipients = list(self.agent_registry.keys())
        
        for recipient in recipients:
            self.send_message(recipient, message)
        
        return recipients
    
    def _send_message_internal(self, message: Message) -> bool:
        """Send message internally."""
        try:
            # Validate recipient
            if message.recipient not in self.agent_registry:
                logger.warning(f"Recipient {message.recipient} not registered")
                self.communication_metrics["failed_deliveries"] += 1
                return False
            
            # Add to history
            self.message_history.append(message)

            # Add to recipient queue
            self.message_queue[message.recipient].append(message)

            # Update metrics
            self.communication_metrics["messages_sent"] += 1
            
            # Log message
            logger.info(f"Message sent: {message.sender} -> {message.recipient} ({message.message_type})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.communication_metrics["failed_deliveries"] += 1
            return False
    
    def _process_pending_messages(self, agent_id: str) -> List[Dict[str, Any]]:
        """Process pending messages for an agent."""
        pending = self.message_queue.get(agent_id, [])
        processed_messages = []
        
        for message in pending:
            processed_messages.append({
                "id": message.id,
                "sender": message.sender,
                "type": message.message_type,
                "content": message.content,
                "timestamp": message.timestamp
            })
        
        # Clear processed queue
        self.message_queue[agent_id] = []
        self.communication_metrics["messages_received"] += len(processed_messages)
        
        return processed_messages
    
    def _assess_communication_quality(self) -> str:
        """Evaluate communication quality."""
        metrics = self.communication_metrics
        
        success_rate = 1.0 - (metrics["failed_deliveries"] / max(1, metrics["messages_sent"]))
        
        if success_rate > 0.95:
            return "excellent"
        elif success_rate > 0.85:
            return "good"
        elif success_rate > 0.70:
            return "fair"
        else:
            return "poor"


# ============================================================================
# Monitoring Behavior - Strategy for System Monitoring
# ============================================================================

class MonitoringBehavior(BaseBehavior, IAgentMonitoring):
    """Specialized behavior for system monitoring and metrics."""
    
    @property
    def behavior_type(self) -> AgentBehaviorType:
        return AgentBehaviorType.MONITORING
    
    @property
    def required_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.PERFORMANCE_MONITORING,
            AgentCapability.HEALTH_CHECKING,
            AgentCapability.ANOMALY_DETECTION,
            AgentCapability.ALERT_GENERATION
        ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.monitoring_interval = self.config.get("monitoring_interval", 10)  # seconds
        self.metric_retention = self.config.get("metric_retention", 3600)  # seconds
        self.alert_thresholds = self.config.get("alert_thresholds", {
            "response_time_ms": 1000,
            "error_rate": 0.05,
            "memory_usage": 0.8
        })
        
        # Monitoring state
        self.metrics_history: Deque[PerformanceMetric] = deque(maxlen=1000)
        self.agent_health_status: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: List[Dict[str, Any]] = []
        self.monitoring_stats = {
            "total_metrics_collected": 0,
            "alerts_generated": 0,
            "anomalies_detected": 0,
            "monitoring_uptime": time.time()
        }
        self._process = psutil.Process() if _HAS_PSUTIL else None
        self._tracemalloc_started = False
    
    def execute_behavior(self, context: AgentContext, agent: IAgent) -> AgentResult:
        """Execute monitoring behavior."""
        start_time = time.time()
        
        try:
            monitoring_result = self._perform_monitoring(context, agent)
            
            execution_time = (time.time() - start_time) * 1000
            self._update_metrics(execution_time, monitoring_result["success"])
            
            return AgentResult(
                agent_id=agent.agent_id,
                status="success" if monitoring_result["success"] else "failed",
                result=monitoring_result,
                execution_time_ms=execution_time,
                confidence=monitoring_result.get("confidence", 0.95),
                metadata={
                    "metrics_collected": monitoring_result.get("metrics_collected", 0),
                    "alerts_generated": monitoring_result.get("alerts_generated", 0)
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._update_metrics(execution_time, False)
            logger.error(f"Monitoring behavior failed: {e}")
            
            return AgentResult(
                agent_id=agent.agent_id,
                status="failed",
                result=None,
                execution_time_ms=execution_time,
                error=str(e)
            )
    
    def _perform_monitoring(self, context: AgentContext, agent: IAgent) -> Dict[str, Any]:
        """Perform complete system monitoring."""
        
        # Collect performance metrics
        performance_metrics = self._collect_performance_metrics(agent)

        # Verify agent health
        health_status = self._check_agent_health(agent)

        # Detect anomalies
        anomalies = self._detect_anomalies()

        # Generate alerts if necessary
        alerts = self._generate_alerts(performance_metrics, health_status, anomalies)
        
        # Update statistics
        self.monitoring_stats["total_metrics_collected"] += len(performance_metrics)
        self.monitoring_stats["alerts_generated"] += len(alerts)
        self.monitoring_stats["anomalies_detected"] += len(anomalies)
        
        return {
            "success": True,
            "performance_metrics": performance_metrics,
            "health_status": health_status,
            "anomalies": anomalies,
            "alerts": alerts,
            "metrics_collected": len(performance_metrics),
            "alerts_generated": len(alerts),
            "confidence": 0.95,
            "monitoring_quality": self._assess_monitoring_quality()
        }

    def _get_process_metrics(self) -> Dict[str, Any]:
        """Get real process CPU/memory metrics."""
        if _HAS_PSUTIL and self._process:
            try:
                cpu_percent = self._process.cpu_percent(interval=None)
                memory_percent = self._process.memory_percent()
                rss_bytes = self._process.memory_info().rss
                return {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "rss_bytes": rss_bytes
                }
            except Exception:
                pass

        if not self._tracemalloc_started:
            tracemalloc.start()
            self._tracemalloc_started = True
        current, peak = tracemalloc.get_traced_memory()
        return {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "rss_bytes": current,
            "peak_bytes": peak
        }
    
    def _collect_performance_metrics(self, agent: IAgent) -> List[Dict[str, Any]]:
        """Collect performance metrics."""
        metrics = []
        current_time = time.time()
        
        # Current agent metrics
        agent_metrics = self._collect_agent_metrics(agent, current_time)
        metrics.extend(agent_metrics)
        
        # General system metrics
        system_metrics = self._collect_system_metrics(current_time)
        metrics.extend(system_metrics)
        
        # Store metrics in history
        for metric_data in metrics:
            metric = PerformanceMetric(
                metric_name=metric_data["name"],
                value=metric_data["value"],
                timestamp=current_time,
                agent_id=agent.agent_id,
                metric_type=metric_data.get("type", "gauge"),
                tags=metric_data.get("tags", {})
            )
            self.metrics_history.append(metric)
        
        return metrics
    
    def _collect_agent_metrics(self, agent: IAgent, timestamp: float) -> List[Dict[str, Any]]:
        """Collect specific agent metrics."""
        metrics = []
        base_performance = getattr(agent, "success_rate", None)
        if base_performance is None:
            base_performance = getattr(self, 'success_count', 0) / max(1, getattr(self, 'execution_count', 1))

        response_time = getattr(agent, "last_execution_time_ms", 0.0)
        task_count = getattr(agent, "tasks_completed", 0) + getattr(agent, "tasks_failed", 0)

        proc = self._get_process_metrics()
        memory_percent = proc.get("memory_percent", 0.0)

        metrics.extend([
            {
                "name": "agent_response_time_ms",
                "value": response_time,
                "type": "gauge",
                "tags": {"agent_type": str(agent.agent_type)}
            },
            {
                "name": "agent_success_rate",
                "value": base_performance,
                "type": "gauge",
                "tags": {"agent_type": str(agent.agent_type)}
            },
            {
                "name": "agent_task_count",
                "value": task_count,
                "type": "counter",
                "tags": {"agent_type": str(agent.agent_type)}
            },
            {
                "name": "agent_memory_usage",
                "value": memory_percent / 100.0 if memory_percent > 1 else memory_percent,
                "type": "gauge",
                "tags": {"agent_type": str(agent.agent_type)}
            }
        ])
        
        return metrics
    
    def _collect_system_metrics(self, timestamp: float) -> List[Dict[str, Any]]:
        """Collect overall system metrics."""
        metrics = []
        uptime = time.time() - self.monitoring_stats["monitoring_uptime"]
        cpu_usage = None
        memory_usage = None
        if _HAS_PSUTIL:
            try:
                cpu_usage = psutil.cpu_percent(interval=None)
                memory_usage = psutil.virtual_memory().percent
            except Exception:
                cpu_usage = None
                memory_usage = None

        metrics.extend([
            {
                "name": "system_uptime_seconds",
                "value": uptime,
                "type": "gauge",
                "tags": {"component": "monitoring"}
            },
            {
                "name": "total_agents_active",
                "value": len(self.agent_health_status),
                "type": "gauge",
                "tags": {"component": "system"}
            },
            {
                "name": "total_metrics_collected",
                "value": self.monitoring_stats["total_metrics_collected"],
                "type": "counter",
                "tags": {"component": "monitoring"}
            },
        ])

        if cpu_usage is not None:
            metrics.append({
                "name": "system_cpu_usage",
                "value": cpu_usage / 100.0,
                "type": "gauge",
                "tags": {"component": "system"}
            })
        if memory_usage is not None:
            metrics.append({
                "name": "system_memory_usage",
                "value": memory_usage / 100.0,
                "type": "gauge",
                "tags": {"component": "system"}
            })
        
        return metrics
    
    def _check_agent_health(self, agent: IAgent) -> Dict[str, Any]:
        """Verify agent health."""
        agent_id = agent.agent_id
        current_time = time.time()
        total_tasks = getattr(agent, "tasks_completed", 0) + getattr(agent, "tasks_failed", 0)
        error_rate = getattr(agent, "tasks_failed", 0) / max(1, total_tasks)
        response_time = getattr(agent, "last_execution_time_ms", 0.0)

        # Get current health state
        health_data = {
            "agent_id": agent_id,
            "status": "healthy",
            "last_check": current_time,
            "response_time_ms": response_time,
            "error_count": getattr(agent, "tasks_failed", 0),
            "warnings": [],
            "performance_score": max(0.0, 1.0 - error_rate)
        }
        
        # Specific verifications
        checks = self._perform_health_checks(agent)
        health_data.update(checks)

        # Update health registry
        self.agent_health_status[agent_id] = health_data
        
        return health_data
    
    def _perform_health_checks(self, agent: IAgent) -> Dict[str, Any]:
        """Perform specific health checks."""
        total_tasks = getattr(agent, "tasks_completed", 0) + getattr(agent, "tasks_failed", 0)
        error_rate = getattr(agent, "tasks_failed", 0) / max(1, total_tasks)
        response_time = getattr(agent, "last_execution_time_ms", 0.0)

        checks = {
            "connectivity": "ok",
            "resource_usage": "normal",
            "response_quality": "good",
            "error_rate": error_rate,
            "response_time_ms": response_time,
        }
        
        warnings = []

        # Verify error rate
        if checks["error_rate"] > self.alert_thresholds.get("error_rate", 0.05):
            warnings.append("High error rate detected")
            checks["status"] = "warning"
        
        # Verify response time
        response_time_ms = checks.get("response_time_ms", 0.0)
        if response_time_ms > self.alert_thresholds.get("response_time_ms", 1000):
            warnings.append("High response time detected")
            checks["status"] = "warning"
        
        checks["warnings"] = warnings
        
        return checks
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics."""
        anomalies = []
        
        if len(self.metrics_history) < 10:
            return anomalies  # Need sufficient data

        # Analyze recent metrics
        recent_metrics = list(self.metrics_history)[-10:]

        # Detect anomalies by metric type
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric.metric_name].append(metric.value)
        
        for metric_name, values in metric_groups.items():
            anomaly = self._detect_metric_anomaly(metric_name, values)
            if anomaly:
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_metric_anomaly(self, metric_name: str, values: List[float]) -> Optional[Dict[str, Any]]:
        """Detect anomaly in a specific metric."""
        if len(values) < 5:
            return None
        
        # Calculate basic statistics
        avg_value = sum(values) / len(values)
        latest_value = values[-1]
        
        # Detect significant deviations (simplified implementation)
        threshold_multiplier = 2.0
        
        if abs(latest_value - avg_value) > avg_value * threshold_multiplier:
            return {
                "metric_name": metric_name,
                "anomaly_type": "significant_deviation",
                "current_value": latest_value,
                "expected_range": [avg_value * 0.8, avg_value * 1.2],
                "severity": "medium",
                "timestamp": time.time()
            }
        
        return None
    
    def _generate_alerts(self, metrics: List[Dict[str, Any]], health: Dict[str, Any], 
                        anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate alerts based on metrics, health, and anomalies."""
        alerts = []
        current_time = time.time()
        
        # Alerts based on metrics
        for metric in metrics:
            alert = self._check_metric_alert(metric, current_time)
            if alert:
                alerts.append(alert)
        
        # Alerts based on health
        if health.get("status") != "healthy":
            alerts.append({
                "alert_id": f"health_{int(current_time)}",
                "type": "health_warning",
                "severity": "medium",
                "message": f"Agent health issue: {health.get('warnings', [])}",
                "timestamp": current_time,
                "source": "health_check"
            })
        
        # Alerts based on anomalies
        for anomaly in anomalies:
            if anomaly.get("severity") in ["high", "critical"]:
                alerts.append({
                    "alert_id": f"anomaly_{int(current_time)}",
                    "type": "anomaly_detected",
                    "severity": anomaly["severity"],
                    "message": f"Anomaly in {anomaly['metric_name']}: {anomaly['anomaly_type']}",
                    "timestamp": current_time,
                    "source": "anomaly_detection"
                })
        
        # Store active alerts
        self.active_alerts.extend(alerts)
        
        return alerts
    
    def _check_metric_alert(self, metric: Dict[str, Any], timestamp: float) -> Optional[Dict[str, Any]]:
        """Verify if a metric requires an alert."""
        metric_name = metric["name"]
        value = metric["value"]

        # Verify specific thresholds
        if metric_name == "agent_response_time_ms":
            threshold = self.alert_thresholds.get("response_time_ms", 1000)
            if value > threshold:
                return {
                    "alert_id": f"response_time_{int(timestamp)}",
                    "type": "performance_degradation",
                    "severity": "medium",
                    "message": f"High response time: {value}ms (threshold: {threshold}ms)",
                    "timestamp": timestamp,
                    "source": "metric_threshold"
                }
        
        elif metric_name == "agent_memory_usage":
            threshold = self.alert_thresholds.get("memory_usage", 0.8)
            if value > threshold:
                return {
                    "alert_id": f"memory_{int(timestamp)}",
                    "type": "resource_exhaustion",
                    "severity": "high",
                    "message": f"High memory usage: {value:.2%} (threshold: {threshold:.2%})",
                    "timestamp": timestamp,
                    "source": "metric_threshold"
                }
        
        return None
    
    def _assess_monitoring_quality(self) -> str:
        """Evaluate monitoring quality."""
        total_metrics = self.monitoring_stats["total_metrics_collected"]
        total_alerts = self.monitoring_stats["alerts_generated"]
        
        if total_metrics > 100 and total_alerts < total_metrics * 0.1:
            return "excellent"
        elif total_metrics > 50:
            return "good"
        elif total_metrics > 10:
            return "fair"
        else:
            return "basic"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Implementation of IAgentMonitoring.get_performance_metrics."""
        recent_metrics = list(self.metrics_history)[-20:]  # Last 20 metrics
        
        return {
            "total_metrics": len(self.metrics_history),
            "recent_metrics": [
                {
                    "name": m.metric_name,
                    "value": m.value,
                    "timestamp": m.timestamp,
                    "agent_id": m.agent_id
                }
                for m in recent_metrics
            ],
            "monitoring_stats": self.monitoring_stats,
            "active_alerts_count": len(self.active_alerts)
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Implementation of IAgentMonitoring.get_health_status."""
        return {
            "overall_status": "healthy" if not self.active_alerts else "warning",
            "agents_monitored": len(self.agent_health_status),
            "active_alerts": len(self.active_alerts),
            "monitoring_uptime": time.time() - self.monitoring_stats["monitoring_uptime"],
            "agent_health_details": self.agent_health_status
        }
    
    def log_activity(self, activity: Dict[str, Any]) -> None:
        """Implementation of IAgentMonitoring.log_activity."""
        logger.info(f"Activity logged: {activity}")


# Export all communication and monitoring behaviors
__all__ = [
    "Message",
    "CommunicationEvent", 
    "PerformanceMetric",
    "CommunicationBehavior",
    "MonitoringBehavior"
]
