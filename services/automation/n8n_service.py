import sys
"""
Main N8N Automation Service for Capibara6

# This module provides functionality for n8n_service.
"""

import os

import logging
from typing import Any, Dict, List, Optional

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.append(project_root)

def main():
    # Main function for this module.
    logger.info("Module n8n_service.py starting")
    return True

# FastAPI imports for REST API
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. Install with: pip install fastapi uvicorn")

# Pydantic for request/response models
try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


class CapibaraN8nAutomationService:
    """
    Main automation service that provides text-to-workflow conversion and execution.
    
    Features:
    - Natural language to n8n workflow conversion
    - Agent-based intelligent execution
    - E2b sandbox for secure code execution
    - REST API endpoints for integration
    - Session management and workflow persistence
    - Performance monitoring and analytics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the automation service.
        
        Args:
            config: Service configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.workflow_builder = WorkflowBuilder(self.config.get("workflow_builder", {}))
        self.agent_executor = AgentExecutor(self.config.get("agent_executor", {}))
        self.e2b_manager = E2bSandboxManager(self.config.get("e2b_manager", {}))
        
        # N8N connection configuration
        self.n8n_config = self.config.get("n8n", {
            "base_url": "http://localhost:5678",
            "api_key": None,
            "webhook_url": "http://localhost:5678/webhook"
        })
        
        # Session management
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._workflow_cache: Dict[str, WorkflowSpec] = {}
        self._execution_history: List[ExecutionResult] = []
        
        # Performance metrics
        self._service_stats = {
            "workflows_created": 0,
            "workflows_executed": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_creation_time": 0.0,
            "average_execution_time": 0.0,
            "agent_executions": 0,
            "sandbox_executions": 0
        }
        
        # HTTP client for n8n API
        self._http_session: Optional[aiohttp.ClientSession] = None
        
        # FastAPI app if available
        self.app: Optional[FastAPI] = None
        if FASTAPI_AVAILABLE:
            self._setup_fastapi()
    
    async def startup(self):
        """Initialize the service and start background tasks."""
        self.logger.info("Starting Capibara N8N Automation Service...")
        
        # Initialize HTTP session
        self._http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=self._get_n8n_headers()
        )
        
        # Test n8n connection
        try:
            await self._test_n8n_connection()
            self.logger.info("N8N connection established successfully")
        except Exception as e:
            self.logger.warning(f"Could not connect to n8n: {e}")
        
        # Start background cleanup tasks
        asyncio.create_task(self._cleanup_expired_sessions())
        asyncio.create_task(self._cleanup_expired_sandboxes())
        
        self.logger.info("Capibara N8N Automation Service started successfully")
    
    async def shutdown(self):
        """Cleanup and shutdown the service."""
        self.logger.info("Shutting down Capibara N8N Automation Service...")
        
        # Close HTTP session
        if self._http_session:
            await self._http_session.close()
        
        # Cleanup all active sessions
        for session_id in list(self._active_sessions.keys()):
            await self.cleanup_session(session_id)
        
        self.logger.info("Capibara N8N Automation Service shut down")
    
    async def create_automation(
        self,
        request: AutomationRequest,
        session_id: Optional[str] = None,
        execute_immediately: bool = False
    ) -> Dict[str, Any]:
        """
        Create an automation workflow from a natural language request.
        
        Args:
            request: Automation request with description and preferences
            session_id: Session identifier for tracking
            execute_immediately: Whether to execute the workflow immediately
            
        Returns:
            Dictionary with workflow specification and optional execution result
        """
        start_time = datetime.utcnow()
        session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        
        try:
            self.logger.info(f"Creating automation for session {session_id}: {request.description[:100]}...")
            
            # Build workflow from natural language
            workflow_spec = await self.workflow_builder.build_workflow(request)
            
            # cache the workflow
            workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
            self._workflow_cache[workflow_id] = workflow_spec
            
            # Track session
            if session_id not in self._active_sessions:
                self._active_sessions[session_id] = {
                    "created_at": datetime.utcnow(),
                    "workflows": [],
                    "executions": []
                }
            
            self._active_sessions[session_id]["workflows"].append({
                "workflow_id": workflow_id,
                "created_at": datetime.utcnow(),
                "spec": workflow_spec
            })
            
            # Create workflow in n8n if possible
            n8n_workflow_id = None
            try:
                n8n_workflow_id = await self._create_n8n_workflow(workflow_spec)
                self.logger.info(f"Workflow created in n8n: {n8n_workflow_id}")
            except Exception as e:
                self.logger.warning(f"Could not create workflow in n8n: {e}")
            
            # Update statistics
            creation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_creation_stats(creation_time)
            
            result = {
                "workflow_id": workflow_id,
                "session_id": session_id,
                "workflow_spec": workflow_spec.dict(),
                "n8n_workflow_id": n8n_workflow_id,
                "creation_time_ms": int(creation_time),
                "created_at": start_time.isoformat()
            }
            
            # Execute immediately if requested
            if execute_immediately:
                execution_result = await self.execute_workflow(
                    workflow_id=workflow_id,
                    input_data=request.context,
                    session_id=session_id
                )
                result["execution_result"] = execution_result.dict()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create automation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to create automation: {str(e)}")
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any] = None,
        session_id: Optional[str] = None,
        execution_mode: Optional[ExecutionMode] = None
    ) -> ExecutionResult:
        """
        Execute a workflow by ID.
        
        Args:
            workflow_id: Workflow identifier
            input_data: Input data for execution
            session_id: Session identifier
            execution_mode: Override execution mode
            
        Returns:
            Execution result with details and output
        """
        try:
            # Get workflow specification
            if workflow_id not in self._workflow_cache:
                raise ValueError(f"Workflow not found: {workflow_id}")
            
            workflow_spec = self._workflow_cache[workflow_id]
            
            # Override execution mode if specified
            if execution_mode:
                workflow_spec.execution_mode = execution_mode
            
            self.logger.info(f"Executing workflow {workflow_id} in {workflow_spec.execution_mode} mode")
            
            # Execute based on mode
            if workflow_spec.execution_mode == ExecutionMode.AGENT_BASED:
                result = await self._execute_with_agents(workflow_spec, input_data, session_id)
            elif workflow_spec.execution_mode == ExecutionMode.E2B_SANDBOX:
                result = await self._execute_with_sandbox(workflow_spec, input_data, session_id)
            elif workflow_spec.execution_mode == ExecutionMode.HYBRID:
                result = await self._execute_hybrid(workflow_spec, input_data, session_id)
            else:
                # Standard n8n execution
                result = await self._execute_standard_n8n(workflow_spec, input_data, session_id)
            
            # Track execution in session
            if session_id and session_id in self._active_sessions:
                self._active_sessions[session_id]["executions"].append({
                    "execution_id": result.execution_id,
                    "workflow_id": workflow_id,
                    "executed_at": result.started_at,
                    "status": result.status
                })
            
            # Add to execution history
            self._execution_history.append(result)
            if len(self._execution_history) > 1000:  # Keep last 1000 executions
                self._execution_history = self._execution_history[-1000:]
            
            # Update statistics
            self._update_execution_stats(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}", exc_info=True)
            
            # Create failed result
            failed_result = ExecutionResult(
                workflow_id=workflow_id,
                execution_id=f"failed_{uuid.uuid4().hex[:8]}",
                status="failed",
                error_message=str(e),
                started_at=datetime.utcnow(),
                finished_at=datetime.utcnow(),
                duration_ms=0
            )
            
            self._update_execution_stats(failed_result)
            return failed_result
    
    async def execute_from_text(
        self,
        description: str,
        context: Dict[str, Any] = None,
        session_id: Optional[str] = None,
        execution_mode: ExecutionMode = ExecutionMode.STANDARD
    ) -> Dict[str, Any]:
        """
        Create and execute a workflow from natural language in one step.
        
        Args:
            description: Natural language description
            context: Additional context
            session_id: Session identifier
            execution_mode: Execution mode preference
            
        Returns:
            Combined creation and execution result
        """
        # Create automation request
        request = AutomationRequest(
            description=description,
            context=context,
            execution_mode=execution_mode,
            session_id=session_id
        )
        
        # Create and execute
        result = await self.create_automation(
            request=request,
            session_id=session_id,
            execute_immediately=True
        )
        
        return result
    
    async def get_workflow_templates(self) -> List[Dict[str, Any]]:
        """Get available workflow templates."""
        templates = []
        
        # Get templates from workflow builder
        if hasattr(self.workflow_builder, '_templates'):
            for template_id, template in self.workflow_builder._templates.items():
                templates.append({
                    "id": template.id,
                    "name": template.name,
                    "description": template.description,
                    "category": template.category,
                    "supports_agents": template.supports_agents,
                    "supports_e2b": template.supports_e2b,
                    "example_requests": template.example_requests or []
                })
        
        return templates
    
    async def create_from_template(
        self,
        template_id: str,
        customization: Dict[str, Any] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates a workflow from a template."""
        try:
            # Build workflow from template
            workflow_spec = await self.workflow_builder.build_from_template(
                template_id=template_id,
                customization=customization
            )
            
            # cache and track the workflow
            workflow_id = f"template_{template_id}_{uuid.uuid4().hex[:8]}"
            self._workflow_cache[workflow_id] = workflow_spec
            
            session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
            
            if session_id not in self._active_sessions:
                self._active_sessions[session_id] = {
                    "created_at": datetime.utcnow(),
                    "workflows": [],
                    "executions": []
                }
            
            self._active_sessions[session_id]["workflows"].append({
                "workflow_id": workflow_id,
                "created_at": datetime.utcnow(),
                "spec": workflow_spec,
                "template_id": template_id
            })
            
            return {
                "workflow_id": workflow_id,
                "session_id": session_id,
                "template_id": template_id,
                "workflow_spec": workflow_spec.dict(),
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create workflow from template: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session."""
        if session_id not in self._active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = self._active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "created_at": session["created_at"].isoformat(),
            "workflows_count": len(session["workflows"]),
            "executions_count": len(session["executions"]),
            "workflows": [
                {
                    "workflow_id": w["workflow_id"],
                    "name": w["spec"].name,
                    "created_at": w["created_at"].isoformat(),
                    "execution_mode": w["spec"].execution_mode,
                    "nodes_count": len(w["spec"].nodes)
                }
                for w in session["workflows"]
            ],
            "recent_executions": [
                {
                    "execution_id": e["execution_id"],
                    "workflow_id": e["workflow_id"],
                    "executed_at": e["executed_at"].isoformat(),
                    "status": e["status"]
                }
                for e in session["executions"][-10:]  # Last 10 executions
            ]
        }
    
    async def cleanup_session(self, session_id: str) -> Dict[str, Any]:
        """Clean up a session and associated resources."""
        if session_id not in self._active_sessions:
            return {"message": "Session not found"}
        
        session = self._active_sessions[session_id]
        
        # Clean up E2b sandboxes for this session
        sandbox_cleanup_count = await self.e2b_manager.cleanup_session(session_id)
        
        # Clear agent memory for this session
        self.agent_executor.clear_agent_memory(session_id)
        
        # Remove workflows from cache
        workflows_removed = 0
        for workflow_info in session["workflows"]:
            workflow_id = workflow_info["workflow_id"]
            if workflow_id in self._workflow_cache:
                del self._workflow_cache[workflow_id]
                workflows_removed += 1
        
        # Remove session
        del self._active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "workflows_removed": workflows_removed,
            "sandboxes_cleaned": sandbox_cleanup_count,
            "message": "Session cleaned up successfully"
        }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service performance statistics."""
        return {
            **self._service_stats,
            "active_sessions": len(self._active_sessions),
            "cached_workflows": len(self._workflow_cache),
            "execution_history_size": len(self._execution_history),
            "agent_stats": self.agent_executor.get_execution_stats(),
            "e2b_stats": self.e2b_manager.get_execution_stats(),
            "active_sandboxes": len(self.e2b_manager.get_active_sandboxes())
        }
    
    # Private methods
    
    async def _execute_with_agents(
        self,
        workflow_spec: WorkflowSpec,
        input_data: Dict[str, Any],
        session_id: Optional[str]
    ) -> ExecutionResult:
        """Execute workflow using Capibara agents."""
        self._service_stats["agent_executions"] += 1
        return await self.agent_executor.execute_workflow(
            workflow_spec=workflow_spec,
            input_data=input_data,
            session_id=session_id
        )
    
    async def _execute_with_sandbox(
        self,
        workflow_spec: WorkflowSpec,
        input_data: Dict[str, Any],
        session_id: Optional[str]
    ) -> ExecutionResult:
        """Execute workflow using E2b sandbox."""
        self._service_stats["sandbox_executions"] += 1
        
        # Create sandbox for this execution
        sandbox_id = await self.e2b_manager.create_sandbox(
            template="python3",
            session_id=session_id
        )
        
        try:
            # Execute workflow nodes in sandbox
            results = {}
            for node in workflow_spec.nodes:
                node_result = await self.e2b_manager.execute_workflow_node(
                    sandbox_id=sandbox_id,
                    node_spec=node.dict(),
                    input_data=input_data,
                    use_agent=False
                )
                results[node.id] = node_result
            
            return ExecutionResult(
                workflow_id=workflow_spec.name,
                execution_id=f"sandbox_{uuid.uuid4().hex[:8]}",
                status="success",
                output_data=results,
                started_at=datetime.utcnow(),
                finished_at=datetime.utcnow(),
                duration_ms=0,
                sandbox_logs=[f"Executed in sandbox {sandbox_id}"]
            )
            
        finally:
            # Clean up sandbox
            await self.e2b_manager.cleanup_sandbox(sandbox_id)
    
    async def _execute_hybrid(
        self,
        workflow_spec: WorkflowSpec,
        input_data: Dict[str, Any],
        session_id: Optional[str]
    ) -> ExecutionResult:
        """Execute workflow in hybrid mode (agents + sandbox)."""
        # Use agent executor which can handle hybrid mode
        return await self._execute_with_agents(workflow_spec, input_data, session_id)
    
    async def _execute_standard_n8n(
        self,
        workflow_spec: WorkflowSpec,
        input_data: Dict[str, Any],
        session_id: Optional[str]
    ) -> ExecutionResult:
        """Execute workflow using standard n8n API."""
        try:
            # This would execute via n8n API
            # For now, simulate execution
            
            execution_id = f"n8n_{uuid.uuid4().hex[:8]}"
            start_time = datetime.utcnow()
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds() * 1000
            
            return ExecutionResult(
                workflow_id=workflow_spec.name,
                execution_id=execution_id,
                status="success",
                output_data={"message": "Executed via n8n", "nodes_executed": len(workflow_spec.nodes)},
                started_at=start_time,
                finished_at=end_time,
                duration_ms=int(duration)
            )
            
        except Exception as e:
            return ExecutionResult(
                workflow_id=workflow_spec.name,
                execution_id=f"failed_{uuid.uuid4().hex[:8]}",
                status="failed",
                error_message=str(e),
                started_at=datetime.utcnow(),
                finished_at=datetime.utcnow(),
                duration_ms=0
            )
    
    async def _create_n8n_workflow(self, workflow_spec: WorkflowSpec) -> Optional[str]:
        """Creates a workflow in n8n via API."""
        if not self._http_session:
            return None
        
        try:
            # Convert workflow spec to n8n format
            n8n_workflow = self._convert_to_n8n_format(workflow_spec)
            
            # Create workflow via n8n API
            async with self._http_session.post(
                f"{self.n8n_config['base_url']}/api/v1/workflows",
                json=n8n_workflow
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    return result.get("id")
                else:
                    self.logger.warning(f"Failed to create n8n workflow: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.warning(f"Error creating n8n workflow: {e}")
            return None
    
    def _convert_to_n8n_format(self, workflow_spec: WorkflowSpec) -> Dict[str, Any]:
        """Convert workflow specification to n8n format."""
        # This is a simplified conversion
        # In practice, you'd need to handle all n8n-specific formatting
        
        nodes = []
        for node in workflow_spec.nodes:
            n8n_node = {
                "id": node.id,
                "name": node.name,
                "type": node.type,
                "position": [node.position["x"], node.position["y"]],
                "parameters": node.parameters
            }
            nodes.append(n8n_node)
        
        connections = {}
        for conn in workflow_spec.connections:
            source_key = conn.source_node
            if source_key not in connections:
                connections[source_key] = {"main": []}
            
            connections[source_key]["main"].append([{
                "node": conn.target_node,
                "type": "main",
                "index": conn.target_index
            }])
        
        return {
            "name": workflow_spec.name,
            "nodes": nodes,
            "connections": connections,
            "active": True,
            "settings": workflow_spec.settings
        }
    
    def _get_n8n_headers(self) -> Dict[str, str]:
        """Get headers for n8n API requests."""
        headers = {"Content-Type": "application/json"}
        
        if self.n8n_config.get("api_key"):
            headers["X-N8N-API-KEY"] = self.n8n_config["api_key"]
        
        return headers
    
    async def _test_n8n_connection(self):
        """Test connection to n8n instance."""
        if not self._http_session:
            return
        
        async with self._http_session.get(
            f"{self.n8n_config['base_url']}/healthz"
        ) as response:
            if response.status != 200:
                raise ConnectionError(f"N8N health check failed: {response.status}")
    
    def _update_creation_stats(self, creation_time: float):
        """Update workflow creation statistics."""
        self._service_stats["workflows_created"] += 1
        
        # Update average creation time
        total = self._service_stats["workflows_created"]
        current_avg = self._service_stats["average_creation_time"]
        self._service_stats["average_creation_time"] = (
            (current_avg * (total - 1) + creation_time) / total
        )
    
    def _update_execution_stats(self, result: ExecutionResult):
        """Update execution statistics."""
        self._service_stats["workflows_executed"] += 1
        
        if result.status == "success":
            self._service_stats["successful_executions"] += 1
        else:
            self._service_stats["failed_executions"] += 1
        
        # Update average execution time
        if result.duration_ms:
            total = self._service_stats["workflows_executed"]
            current_avg = self._service_stats["average_execution_time"]
            self._service_stats["average_execution_time"] = (
                (current_avg * (total - 1) + result.duration_ms) / total
            )
    
    async def _cleanup_expired_sessions(self):
        """Background task to cleanup expired sessions."""
        while True:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                expired_sessions = []
                
                for session_id, session in self._active_sessions.items():
                    if session["created_at"] < cutoff_time:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    await self.cleanup_session(session_id)
                    self.logger.info(f"Cleaned up expired session: {session_id}")
                
                # Wait 1 hour before next cleanup
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"Session cleanup error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _cleanup_expired_sandboxes(self):
        """Background task to cleanup expired sandboxes."""
        while True:
            try:
                cleanup_count = await self.e2b_manager.cleanup_expired_sandboxes(max_age_hours=6)
                if cleanup_count > 0:
                    self.logger.info(f"Cleaned up {cleanup_count} expired sandboxes")
                
                # Wait 30 minutes before next cleanup
                await asyncio.sleep(1800)
                
            except Exception as e:
                self.logger.error(f"Sandbox cleanup error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def _setup_fastapi(self):
        """Setup FastAPI application with routes."""
        if not FASTAPI_AVAILABLE:
            return
        
        self.app = FastAPI(
            title="Capibara6 N8N Automation API",
            description="AI-powered workflow automation with n8n integration",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Add routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        if not self.app:
            return
        
        @self.app.post("/automation/create")
        async def create_automation_endpoint(request: AutomationRequest):
            """Creates an automation workflow from natural language."""
            return await self.create_automation(request)
        
        @self.app.post("/automation/execute-text")
        async def execute_text_endpoint(
            description: str,
            context: Dict[str, Any] = None,
            session_id: str = None,
            execution_mode: ExecutionMode = ExecutionMode.STANDARD
        ):
            """Creates and execute workflow from text in one step."""
            return await self.execute_from_text(description, context, session_id, execution_mode)
        
        @self.app.post("/automation/execute/{workflow_id}")
        async def execute_workflow_endpoint(
            workflow_id: str,
            input_data: Dict[str, Any] = None,
            session_id: str = None,
            execution_mode: ExecutionMode = None
        ):
            """Execute a workflow by ID."""
            result = await self.execute_workflow(workflow_id, input_data, session_id, execution_mode)
            return result.dict()
        
        @self.app.get("/automation/templates")
        async def get_templates_endpoint():
            """Get available workflow templates."""
            return await self.get_workflow_templates()
        
        @self.app.post("/automation/template/{template_id}")
        async def create_from_template_endpoint(
            template_id: str,
            customization: Dict[str, Any] = None,
            session_id: str = None
        ):
            """Creates workflow from template."""
            return await self.create_from_template(template_id, customization, session_id)
        
        @self.app.get("/session/{session_id}")
        async def get_session_endpoint(session_id: str):
            """Get session information."""
            return await self.get_session_info(session_id)
        
        @self.app.delete("/session/{session_id}")
        async def cleanup_session_endpoint(session_id: str):
            """Clean up a session."""
            return await self.cleanup_session(session_id)
        
        @self.app.get("/stats")
        async def get_stats_endpoint():
            """Get service statistics."""
            return self.get_service_stats()
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        
        # Startup and shutdown events
        @self.app.on_event("startup")
        async def startup_event():
            await self.startup()
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            await self.shutdown()
    
    def run_server(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the FastAPI server."""
        if not FASTAPI_AVAILABLE or not self.app:
            raise RuntimeError("FastAPI is not available")
        
        uvicorn.run(self.app, host=host, port=port, **kwargs)


# Factory function for easy instantiation
def create_automation_service(config: Optional[Dict[str, Any]] = None) -> CapibaraN8nAutomationService:
    """
    Factory function to create a configured automation service.
    
    Args:
        config: Service configuration
        
    Returns:
        Configured automation service instance
    """
    return CapibaraN8nAutomationService(config)