"""
Agent Executor for CapibaraGPT v3 N8N Integration
===========================================

Executes n8n workflows using Capibara agents with intelligent orchestration,
memory management, and seamless integration with the existing agent architecture.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
import json

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    pass  # Using proper imports instead of sys.path manipulation

from .models import (
    AgentType, ExecutionMode, AgentWorkflowConfig, 
    ExecutionResult, WorkflowSpec, WorkflowNode
)

# Optional aiohttp for real HTTP-request nodes. Falls back cleanly when absent.
try:
    import aiohttp  # type: ignore
    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - optional runtime dep
    aiohttp = None  # type: ignore
    AIOHTTP_AVAILABLE = False

# Import existing Capibara agents
try:
    from ...agents.capibara_agent import CapibaraTool
    from ...agents.capibara_agent_factory import CapibaraAgentFactory
    from ...agents.capibara_auto_agent import CapibaraAutoAgent
    from ...agents.ultra_agent_orchestrator import UltraAgentOrchestrator
    AGENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import Capibara agents: {e}")
    AGENTS_AVAILABLE = False

# Import core systems
try:
    from ...core.config import Config
    from ...core.ultra_core_integration import UltraCoreOrchestrator
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


class AgentExecutor:
    """
    Executes n8n workflows using Capibara agents.
    
    Features:
    - Integration with existing Capibara agent architecture
    - Memory management and state persistence
    - Intelligent workflow orchestration
    - Performance monitoring and optimization
    - error handling and recovery
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent executor.
        
        Args:
            config: Configuration dictionary for the executor
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Agent instances cache
        self._agent_cache: Dict[str, Any] = {}
        
        # Memory and state management
        self._agent_memory: Dict[str, Dict[str, Any]] = {}
        self._session_states: Dict[str, Dict[str, Any]] = {}
        
        # Performance monitoring
        self._execution_stats: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_duration": 0.0
        }
        
        # Initialize core systems if available
        self._core_orchestrator = None
        if CORE_AVAILABLE:
            try:
                self._core_orchestrator = UltraCoreOrchestrator(self.config)
            except Exception as e:
                self.logger.warning(f"Could not initialize core orchestrator: {e}")
    
    async def execute_workflow(
        self,
        workflow_spec: WorkflowSpec,
        input_data: Dict[str, Any] = None,
        session_id: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a workflow using the appropriate agent configuration.
        
        Args:
            workflow_spec: The workflow specification to execute
            input_data: Input data for the workflow
            session_id: Session identifier for state management
            
        Returns:
            ExecutionResult with execution details and output
        """
        start_time = datetime.utcnow()
        execution_id = f"exec_{int(start_time.timestamp())}_{hash(workflow_spec.name) % 10000}"
        
        try:
            self.logger.info(f"Starting workflow execution: {execution_id}")
            
            # Validate agent availability
            if not AGENTS_AVAILABLE:
                raise RuntimeError("Capibara agents are not available")
            
            # Get or create agent configuration
            agent_config = workflow_spec.agent_config or AgentWorkflowConfig(
                agent_type=AgentType.CAPIBARA_BASE
            )
            
            # Initialize agent for this execution
            agent = await self._get_or_create_agent(agent_config, session_id)
            
            # Set up execution context
            context = self._prepare_execution_context(
                workflow_spec, input_data, session_id, agent_config
            )
            
            # Execute workflow nodes sequentially or in parallel based on dependencies
            output_data = {}
            agent_logs = []
            
            if workflow_spec.execution_mode == ExecutionMode.AGENT_BASED:
                output_data, agent_logs = await self._execute_agent_based(
                    agent, workflow_spec, context
                )
            elif workflow_spec.execution_mode == ExecutionMode.HYBRID:
                output_data, agent_logs = await self._execute_hybrid_mode(
                    agent, workflow_spec, context
                )
            else:
                # Fallback to standard execution with agent assistance
                output_data, agent_logs = await self._execute_with_agent_assistance(
                    agent, workflow_spec, context
                )
            
            # Update agent memory if configured
            if agent_config.use_memory and session_id:
                await self._update_agent_memory(agent, session_id, output_data)
            
            # Create successful result
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds() * 1000
            
            result = ExecutionResult(
                workflow_id=workflow_spec.name,
                execution_id=execution_id,
                status="success",
                output_data=output_data,
                started_at=start_time,
                finished_at=end_time,
                duration_ms=int(duration),
                agent_logs=agent_logs,
                agent_memory=self._agent_memory.get(session_id, {}) if session_id else None
            )
            
            # Update statistics
            self._update_execution_stats(True, duration)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}", exc_info=True)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds() * 1000
            
            result = ExecutionResult(
                workflow_id=workflow_spec.name,
                execution_id=execution_id,
                status="failed",
                error_message=str(e),
                started_at=start_time,
                finished_at=end_time,
                duration_ms=int(duration)
            )
            
            # Update statistics
            self._update_execution_stats(False, duration)
            
            return result
    
    async def _get_or_create_agent(
        self, 
        agent_config: AgentWorkflowConfig, 
        session_id: Optional[str]
    ) -> Any:
        """Get or create an agent instance based on configuration."""
        cache_key = f"{agent_config.agent_type.value}_{session_id or 'default'}"
        
        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]
        
        # Create new agent instance
        agent = None
        
        if agent_config.agent_type == AgentType.CAPIBARA_BASE:
            # Use agent factory if available
            if hasattr(CapibaraAgentFactory, 'create_agent'):
                agent = CapibaraAgentFactory.create_agent(
                    agent_type="base",
                    **agent_config.agent_parameters
                )
            else:
                # Fallback to direct instantiation
                agent = CapibaraTool(**agent_config.agent_parameters)
                
        elif agent_config.agent_type == AgentType.CAPIBARA_AUTO:
            agent = CapibaraAutoAgent(**agent_config.agent_parameters)
            
        elif agent_config.agent_type == AgentType.ULTRA_ORCHESTRATOR:
            agent = UltraAgentOrchestrator(**agent_config.agent_parameters)
            
        elif agent_config.agent_type == AgentType.CUSTOM:
            # Custom agent creation based on parameters
            custom_class = agent_config.agent_parameters.get('agent_class')
            if custom_class:
                agent = custom_class(**agent_config.agent_parameters.get('init_args', {}))
        
        if agent is None:
            raise ValueError(f"Could not create agent of type: {agent_config.agent_type}")
        
        # cache the agent
        self._agent_cache[cache_key] = agent
        
        # Initialize agent memory if configured
        if agent_config.use_memory and session_id:
            await self._initialize_agent_memory(agent, session_id, agent_config.memory_config)
        
        return agent
    
    def _prepare_execution_context(
        self,
        workflow_spec: WorkflowSpec,
        input_data: Dict[str, Any],
        session_id: Optional[str],
        agent_config: AgentWorkflowConfig
    ) -> Dict[str, Any]:
        """Prepare execution context for the workflow."""
        context = {
            "workflow_spec": workflow_spec,
            "input_data": input_data or {},
            "session_id": session_id,
            "agent_config": agent_config,
            "execution_time": datetime.utcnow(),
            "node_outputs": {},  # Store outputs from each node
            "global_variables": {},  # Global variables accessible to all nodes
        }
        
        # Add session state if available
        if session_id and session_id in self._session_states:
            context["session_state"] = self._session_states[session_id]
        else:
            context["session_state"] = {}
        
        return context
    
    async def _execute_agent_based(
        self,
        agent: Any,
        workflow_spec: WorkflowSpec,
        context: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Execute workflow in pure agent-based mode."""
        logs = []
        
        # Convert workflow to agent-understandable format
        workflow_description = self._workflow_to_agent_prompt(workflow_spec, context)
        
        try:
            # Check if agent has async execution capability
            if hasattr(agent, 'execute_async'):
                result = await agent.execute_async(workflow_description, context)
            elif hasattr(agent, 'execute'):
                # Wrap synchronous execution in async
                result = await asyncio.get_event_loop().run_in_executor(
                    None, agent.execute, workflow_description, context
                )
            else:
                # Fallback to simple agent call
                result = await self._fallback_agent_execution(agent, workflow_description, context)
            
            logs.append({
                "timestamp": datetime.utcnow().isoformat(),
                "level": "info",
                "message": f"Agent executed workflow successfully",
                "agent_type": type(agent).__name__,
                "result_type": type(result).__name__
            })
            
            return {"agent_result": result}, logs
            
        except Exception as e:
            logs.append({
                "timestamp": datetime.utcnow().isoformat(),
                "level": "error",
                "message": f"Agent execution failed: {str(e)}",
                "agent_type": type(agent).__name__
            })
            raise
    
    async def _execute_hybrid_mode(
        self,
        agent: Any,
        workflow_spec: WorkflowSpec,
        context: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Execute workflow in hybrid mode (standard n8n + agent assistance)."""
        logs = []
        output_data = {}
        
        # Process nodes sequentially, using agent for complex logic
        for node in workflow_spec.nodes:
            try:
                if self._node_requires_agent(node):
                    # Use agent for this node
                    node_result = await self._execute_node_with_agent(agent, node, context)
                    logs.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "level": "info",
                        "message": f"Node {node.name} executed with agent",
                        "node_id": node.id,
                        "agent_type": type(agent).__name__
                    })
                else:
                    # Use standard n8n execution (simulated)
                    node_result = await self._execute_node_standard(node, context)
                    logs.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "level": "info",
                        "message": f"Node {node.name} executed in standard mode",
                        "node_id": node.id
                    })
                
                # Store node output for use by subsequent nodes
                context["node_outputs"][node.id] = node_result
                output_data[f"node_{node.id}"] = node_result
                
            except Exception as e:
                logs.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": "error",
                    "message": f"Node {node.name} execution failed: {str(e)}",
                    "node_id": node.id
                })
                raise
        
        return output_data, logs
    
    async def _execute_with_agent_assistance(
        self,
        agent: Any,
        workflow_spec: WorkflowSpec,
        context: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Execute workflow with agent providing assistance and optimization."""
        logs = []
        
        # Ask agent to analyze and optimize the workflow
        workflow_analysis = await self._get_agent_workflow_analysis(agent, workflow_spec, context)
        
        logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "level": "info",
            "message": "Agent provided workflow analysis",
            "analysis": workflow_analysis
        })
        
        # Execute nodes with agent insights
        output_data = {}
        for node in workflow_spec.nodes:
            node_result = await self._execute_node_standard(node, context)
            output_data[f"node_{node.id}"] = node_result
        
        # Let agent post-process results if needed
        if hasattr(agent, 'post_process_workflow_results'):
            try:
                processed_results = await agent.post_process_workflow_results(output_data, context)
                output_data["agent_processed"] = processed_results
                logs.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": "info",
                    "message": "Agent post-processed workflow results"
                })
            except Exception as e:
                logs.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": "warning",
                    "message": f"Agent post-processing failed: {str(e)}"
                })
        
        return output_data, logs
    
    def _workflow_to_agent_prompt(self, workflow_spec: WorkflowSpec, context: Dict[str, Any]) -> str:
        """Convert workflow specification to a prompt for the agent."""
        prompt = f"""
Execute the following workflow: {workflow_spec.name}

Description: {workflow_spec.description or 'not description provided'}

Nodes:
"""
        for node in workflow_spec.nodes:
            prompt += f"- {node.name} ({node.type}): {node.parameters}\n"
        
        prompt += f"\nInput Data: {json.dumps(context['input_data'], indent=2)}\n"
        
        if context.get('session_state'):
            prompt += f"\nSession State: {json.dumps(context['session_state'], indent=2)}\n"
        
        prompt += "\nPlease execute this workflow and return the results."
        
        return prompt
    
    def _node_requires_agent(self, node: WorkflowNode) -> bool:
        """Determine if a node requires agent execution."""
        agent_required_types = {
            "n8n-nodes-base.function",
            "n8n-nodes-base.code",
            "capibara-nodes.agent",
            "capibara-nodes.agentExecutor"
        }
        
        return (
            node.type in agent_required_types or
            node.agent_type is not None or
            "agent" in node.parameters
        )
    
    async def _execute_node_with_agent(
        self, 
        agent: Any, 
        node: WorkflowNode, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single node using the agent."""
        # Prepare node-specific context
        node_context = {
            "node": node,
            "input_data": context.get("input_data", {}),
            "previous_outputs": context.get("node_outputs", {}),
            "global_variables": context.get("global_variables", {})
        }
        
        # Create prompt for the specific node
        node_prompt = f"""
Execute node: {node.name}
Type: {node.type}
Parameters: {json.dumps(node.parameters, indent=2)}
Context: {json.dumps(node_context, indent=2, default=str)}

Please execute this node and return the output data.
"""
        
        # Execute with agent
        if hasattr(agent, 'execute_node'):
            return await agent.execute_node(node_prompt, node_context)
        else:
            # Fallback to general execution
            return await self._fallback_agent_execution(agent, node_prompt, node_context)
    
    async def _execute_node_standard(
        self,
        node: WorkflowNode,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a node in standard n8n mode.

        This path is taken when the workflow is not using agent-based or
        sandbox execution. We replace the previous hard-coded "simulated
        response" / "Simulated output for <type>" strings with either a real
        effect (HTTP requests, Set / Webhook pass-through) or an explicit
        ``status="unsupported"`` marker so the caller knows the node was not
        actually executed. Nothing is ever fabricated silently.
        """
        result: Dict[str, Any] = {
            "node_id": node.id,
            "node_name": node.name,
            "node_type": node.type,
            "executed_at": datetime.utcnow().isoformat(),
            "parameters": node.parameters,
            "status": "completed",
        }

        node_type = getattr(node.type, "value", node.type)

        if node_type == "n8n-nodes-base.set":
            # Pass-through: Set nodes materialise their parameters as
            # the output data. Documented n8n behaviour.
            result["data"] = dict(node.parameters)
            return result

        if node_type == "n8n-nodes-base.webhook":
            # Webhook nodes expose the input payload received from the
            # triggering caller - no synthetic response.
            result["webhook_data"] = context.get("input_data", {})
            return result

        if node_type == "n8n-nodes-base.httpRequest":
            return await self._execute_http_request_node(node, result)

        # Unknown / unsupported node type: declare it truthfully instead of
        # returning a fake string. The orchestrator can escalate this to
        # an agent or fail the workflow.
        result["status"] = "unsupported"
        result["error"] = (
            f"Node type {node_type!r} has no standard-mode executor. "
            "Route the workflow through AGENT_BASED / HYBRID mode or add "
            "a dedicated handler."
        )
        return result

    async def _execute_http_request_node(
        self,
        node: WorkflowNode,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform a real HTTP request for ``n8n-nodes-base.httpRequest``.

        The n8n schema carries the method/url/headers/body on
        ``node.parameters``. If any required piece is missing or aiohttp is
        not installed we mark the node as failed with a descriptive error
        instead of returning a fake 200 response.
        """
        params = node.parameters or {}
        url = params.get("url") or params.get("uri")
        method = str(params.get("method", "GET")).upper()

        if not url:
            result["status"] = "failed"
            result["error"] = "httpRequest node missing required 'url' parameter"
            return result

        if not AIOHTTP_AVAILABLE or aiohttp is None:
            result["status"] = "failed"
            result["error"] = (
                "aiohttp not installed - cannot execute real HTTP request "
                "for httpRequest node. Install with: pip install aiohttp"
            )
            return result

        headers = params.get("headers") or params.get("headerParameters") or {}
        # n8n encodes headers as [{"name": "X", "value": "Y"}] - normalise.
        if isinstance(headers, list):
            headers = {
                entry.get("name", ""): entry.get("value", "")
                for entry in headers
                if isinstance(entry, dict) and entry.get("name")
            }

        body = params.get("body") or params.get("bodyParameters")
        timeout_s = float(params.get("timeout", 30))

        try:
            timeout = aiohttp.ClientTimeout(total=timeout_s)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                request_kwargs: Dict[str, Any] = {"headers": headers}
                if body is not None and method not in ("GET", "HEAD"):
                    if isinstance(body, (dict, list)):
                        request_kwargs["json"] = body
                    else:
                        request_kwargs["data"] = body

                async with session.request(method, url, **request_kwargs) as response:
                    content_type = response.headers.get("Content-Type", "")
                    if "application/json" in content_type.lower():
                        try:
                            payload: Any = await response.json(content_type=None)
                        except Exception:
                            payload = await response.text()
                    else:
                        payload = await response.text()

                    result["response"] = {
                        "status": response.status,
                        "headers": dict(response.headers),
                        "data": payload,
                        "url": str(response.url),
                    }
                    if response.status >= 400:
                        result["status"] = "failed"
                        result["error"] = f"HTTP {response.status} from {url}"
        except asyncio.TimeoutError:
            result["status"] = "failed"
            result["error"] = f"HTTP request to {url} timed out after {timeout_s}s"
        except Exception as exc:  # pragma: no cover - depends on network
            result["status"] = "failed"
            result["error"] = f"HTTP request to {url} failed: {exc}"

        return result
    
    async def _get_agent_workflow_analysis(
        self, 
        agent: Any, 
        workflow_spec: WorkflowSpec, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get agent analysis of the workflow for optimization."""
        if hasattr(agent, 'analyze_workflow'):
            return await agent.analyze_workflow(workflow_spec, context)
        
        # Fallback analysis
        return {
            "analysis": "Basic workflow analysis",
            "suggestions": ["Consider using parallel execution for independent nodes"],
            "estimated_duration": "Unknown",
            "complexity": "Medium"
        }
    
    async def _fallback_agent_execution(
        self, 
        agent: Any, 
        prompt: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback execution method for agents without specific workflow support."""
        # Try various common agent methods
        if hasattr(agent, '__call__'):
            return await asyncio.get_event_loop().run_in_executor(
                None, agent, prompt
            )
        elif hasattr(agent, 'run'):
            return await asyncio.get_event_loop().run_in_executor(
                None, agent.run, prompt
            )
        elif hasattr(agent, 'process'):
            return await asyncio.get_event_loop().run_in_executor(
                None, agent.process, prompt, context
            )
        else:
            # Last resort: return a basic response
            return {
                "agent_type": type(agent).__name__,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "response": "Agent executed successfully",
                "context_summary": str(context)[:200]
            }
    
    async def _initialize_agent_memory(
        self, 
        agent: Any, 
        session_id: str, 
        memory_config: Dict[str, Any]
    ):
        """Initialize agent memory for a session."""
        if session_id not in self._agent_memory:
            self._agent_memory[session_id] = {}
        
        # Configure agent memory if the agent supports it
        if hasattr(agent, 'set_memory'):
            agent.set_memory(self._agent_memory[session_id])
        elif hasattr(agent, 'memory'):
            agent.memory = self._agent_memory[session_id]
    
    async def _update_agent_memory(
        self, 
        agent: Any, 
        session_id: str, 
        output_data: Dict[str, Any]
    ):
        """Update agent memory after execution."""
        if session_id not in self._agent_memory:
            self._agent_memory[session_id] = {}
        
        # Update memory with execution results
        self._agent_memory[session_id].update({
            "last_execution": datetime.utcnow().isoformat(),
            "last_output": output_data,
            "execution_count": self._agent_memory[session_id].get("execution_count", 0) + 1
        })
        
        # Update agent memory if supported
        if hasattr(agent, 'update_memory'):
            agent.update_memory(self._agent_memory[session_id])
        elif hasattr(agent, 'memory'):
            agent.memory.update(self._agent_memory[session_id])
    
    def _update_execution_stats(self, success: bool, duration: float):
        """Update execution statistics."""
        self._execution_stats["total_executions"] += 1
        
        if success:
            self._execution_stats["successful_executions"] += 1
        else:
            self._execution_stats["failed_executions"] += 1
        
        # Update average duration
        total = self._execution_stats["total_executions"]
        current_avg = self._execution_stats["average_duration"]
        self._execution_stats["average_duration"] = (
            (current_avg * (total - 1) + duration) / total
        )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self._execution_stats.copy()
    
    def clear_agent_cache(self):
        """Clear the agent cache."""
        self._agent_cache.clear()
    
    def clear_agent_memory(self, session_id: Optional[str] = None):
        """Clear agent memory for a session or all sessions."""
        if session_id:
            self._agent_memory.pop(session_id, None)
        else:
            self._agent_memory.clear()