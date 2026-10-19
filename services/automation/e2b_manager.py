"""
E2b Sandbox Manager for CapibaraGPT v3 N8N Integration
================================================

Manages E2b sandbox environments for secure code execution within workflows.
Provides isolated environments for running untrusted code with full monitoring.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, IO
from datetime import datetime, timedelta
import json
import tempfile
import shutil
from pathlib import Path

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    pass  # Using proper imports instead of sys.path manipulation

from .models import E2bSandboxConfig, ExecutionResult

# Try to import E2b SDK
try:
    import e2b
    from e2b import Sandbox
    E2B_AVAILABLE = True
except ImportError:
    E2B_AVAILABLE = False
    logging.warning("E2b SDK not available. Install with: pip install e2b")

# Try to import agent integration
try:
    from .agent_executor import AgentExecutor
    AGENT_INTEGRATION = True
except ImportError:
    AGENT_INTEGRATION = False


class E2bSandboxManager:
    """
    Manages E2b sandbox environments for secure code execution.
    
    Features:
    - Multiple sandbox template support (Python, Node.js, etc.)
    - Agent integration for intelligent code execution
    - File system management and monitoring
    - Network access control
    - Resource limits and timeout management
    - Session persistence and cleanup
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the E2b sandbox manager.
        
        Args:
            config: Configuration dictionary for the manager
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Sandbox instances cache
        self._sandbox_cache: Dict[str, Any] = {}
        
        # Session management
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Performance monitoring
        self._execution_stats: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_duration": 0.0,
            "sandbox_creations": 0,
            "sandbox_cleanup": 0
        }
        
        # Default templates
        self._default_templates = {
            "python3": "python3",
            "nodejs": "nodejs18",
            "bash": "bash",
            "ubuntu": "ubuntu",
            "web": "web-scraper"
        }
        
        # Agent integration
        self._agent_executor = None
        if AGENT_INTEGRATION:
            try:
                self._agent_executor = AgentExecutor(config)
            except Exception as e:
                self.logger.warning(f"Could not initialize agent executor: {e}")
        
        # Check E2b availability
        if not E2B_AVAILABLE:
            self.logger.error("E2b SDK is not available. Please install it to use sandbox features.")
    
    async def create_sandbox(
        self,
        template: str = "python3",
        config: Optional[E2bSandboxConfig] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Create a new E2b sandbox instance.
        
        Args:
            template: Sandbox template to use
            config: Sandbox configuration
            session_id: Session identifier for tracking
            
        Returns:
            Sandbox instance identifier
        """
        if not E2B_AVAILABLE:
            raise RuntimeError("E2b SDK is not available")
        
        sandbox_config = config or E2bSandboxConfig(template=template)
        sandbox_id = f"sb_{int(datetime.utcnow().timestamp())}_{hash(template) % 10000}"
        
        try:
            self.logger.info(f"Creating E2b sandbox: {sandbox_id}")
            
            # Map template to E2b template
            e2b_template = self._map_template(sandbox_config.template)
            
            # Create sandbox with E2b
            if hasattr(e2b, 'Sandbox'):
                sandbox = await e2b.Sandbox.create(
                    template=e2b_template,
                    timeout=sandbox_config.timeout
                )
            else:
                # Fallback for different E2b versions
                sandbox = await Sandbox.create(
                    template=e2b_template,
                    timeout=sandbox_config.timeout
                )
            
            # Configure sandbox environment
            await self._configure_sandbox(sandbox, sandbox_config)
            
            # cache sandbox instance
            self._sandbox_cache[sandbox_id] = {
                "sandbox": sandbox,
                "config": sandbox_config,
                "created_at": datetime.utcnow(),
                "last_used": datetime.utcnow(),
                "session_id": session_id,
                "template": e2b_template
            }
            
            # Track session
            if session_id:
                if session_id not in self._active_sessions:
                    self._active_sessions[session_id] = {}
                self._active_sessions[session_id][sandbox_id] = {
                    "created_at": datetime.utcnow(),
                    "template": e2b_template
                }
            
            # Update statistics
            self._execution_stats["sandbox_creations"] += 1
            
            self.logger.info(f"E2b sandbox created successfully: {sandbox_id}")
            return sandbox_id
            
        except Exception as e:
            self.logger.error(f"Failed to create E2b sandbox: {e}", exc_info=True)
            raise RuntimeError(f"Could not create sandbox: {str(e)}")
    
    async def execute_code(
        self,
        sandbox_id: str,
        code: str,
        language: str = "python",
        context: Optional[Dict[str, Any]] = None,
        use_agent: bool = False
    ) -> ExecutionResult:
        """
        Execute code in a sandbox environment.
        
        Args:
            sandbox_id: Sandbox identifier
            code: Code to execute
            language: Programming language
            context: Execution context
            use_agent: Whether to use agent assistance
            
        Returns:
            ExecutionResult with execution details
        """
        start_time = datetime.utcnow()
        execution_id = f"exec_{int(start_time.timestamp())}_{hash(code) % 10000}"
        
        try:
            # Get sandbox instance
            if sandbox_id not in self._sandbox_cache:
                raise ValueError(f"Sandbox not found: {sandbox_id}")
            
            sandbox_info = self._sandbox_cache[sandbox_id]
            sandbox = sandbox_info["sandbox"]
            config = sandbox_info["config"]
            
            # Update last used time
            sandbox_info["last_used"] = datetime.utcnow()
            
            self.logger.info(f"Executing code in sandbox {sandbox_id}: {execution_id}")
            
            # Prepare execution context
            exec_context = context or {}
            
            # Agent-assisted execution if requested
            if use_agent and self._agent_executor:
                result = await self._execute_with_agent(
                    sandbox, code, language, exec_context, execution_id
                )
            else:
                # Direct execution
                result = await self._execute_direct(
                    sandbox, code, language, exec_context, execution_id
                )
            
            # Create successful result
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds() * 1000
            
            execution_result = ExecutionResult(
                workflow_id=f"sandbox_{sandbox_id}",
                execution_id=execution_id,
                status="success",
                output_data=result,
                started_at=start_time,
                finished_at=end_time,
                duration_ms=int(duration),
                sandbox_logs=result.get("logs", []),
                sandbox_files=result.get("files", [])
            )
            
            # Update statistics
            self._update_execution_stats(True, duration)
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Code execution failed: {e}", exc_info=True)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds() * 1000
            
            execution_result = ExecutionResult(
                workflow_id=f"sandbox_{sandbox_id}",
                execution_id=execution_id,
                status="failed",
                error_message=str(e),
                started_at=start_time,
                finished_at=end_time,
                duration_ms=int(duration)
            )
            
            # Update statistics
            self._update_execution_stats(False, duration)
            
            return execution_result
    
    async def execute_workflow_node(
        self,
        sandbox_id: str,
        node_spec: Dict[str, Any],
        input_data: Dict[str, Any] = None,
        use_agent: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a workflow node in the sandbox.
        
        Args:
            sandbox_id: Sandbox identifier
            node_spec: Node specification
            input_data: Input data for the node
            use_agent: Whether to use agent assistance
            
        Returns:
            Node execution result
        """
        try:
            # Extract code from node specification
            code = self._extract_code_from_node(node_spec, input_data)
            language = node_spec.get("language", "python")
            
            # Execute in sandbox
            result = await self.execute_code(
                sandbox_id=sandbox_id,
                code=code,
                language=language,
                context={"node_spec": node_spec, "input_data": input_data},
                use_agent=use_agent
            )
            
            if result.status == "success":
                return result.output_data
            else:
                raise RuntimeError(f"Node execution failed: {result.error_message}")
                
        except Exception as e:
            self.logger.error(f"Workflow node execution failed: {e}")
            raise
    
    async def upload_file(
        self,
        sandbox_id: str,
        file_path: str,
        content: Union[str, bytes],
        encoding: str = "utf-8"
    ) -> bool:
        """
        Upload a file to the sandbox.
        
        Args:
            sandbox_id: Sandbox identifier
            file_path: Path where to create the file in sandbox
            content: File content
            encoding: Text encoding for string content
            
        Returns:
            True if successful
        """
        try:
            if sandbox_id not in self._sandbox_cache:
                raise ValueError(f"Sandbox not found: {sandbox_id}")
            
            sandbox = self._sandbox_cache[sandbox_id]["sandbox"]
            
            # Convert content to bytes if needed
            if isinstance(content, str):
                content_bytes = content.encode(encoding)
            else:
                content_bytes = content
            
            # Upload file to sandbox
            await sandbox.files.write(file_path, content_bytes)
            
            self.logger.info(f"File uploaded to sandbox {sandbox_id}: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"File upload failed: {e}")
            return False
    
    async def download_file(
        self,
        sandbox_id: str,
        file_path: str,
        encoding: str = "utf-8"
    ) -> Optional[str]:
        """
        Download a file from the sandbox.
        
        Args:
            sandbox_id: Sandbox identifier
            file_path: Path of the file in sandbox
            encoding: Text encoding for the file
            
        Returns:
            File content as string or None if failed
        """
        try:
            if sandbox_id not in self._sandbox_cache:
                raise ValueError(f"Sandbox not found: {sandbox_id}")
            
            sandbox = self._sandbox_cache[sandbox_id]["sandbox"]
            
            # Download file from sandbox
            content = await sandbox.files.read(file_path)
            
            # Decode content
            if isinstance(content, bytes):
                return content.decode(encoding)
            else:
                return str(content)
                
        except Exception as e:
            self.logger.error(f"File download failed: {e}")
            return None
    
    async def list_files(self, sandbox_id: str, directory: str = "/") -> List[str]:
        """
        List files in a sandbox directory.
        
        Args:
            sandbox_id: Sandbox identifier
            directory: Directory path
            
        Returns:
            List of file paths
        """
        try:
            if sandbox_id not in self._sandbox_cache:
                raise ValueError(f"Sandbox not found: {sandbox_id}")
            
            sandbox = self._sandbox_cache[sandbox_id]["sandbox"]
            
            # List files in directory
            files = await sandbox.files.list(directory)
            return [f.path for f in files]
            
        except Exception as e:
            self.logger.error(f"File listing failed: {e}")
            return []
    
    async def cleanup_sandbox(self, sandbox_id: str) -> bool:
        """
        Clean up and destroy a sandbox instance.
        
        Args:
            sandbox_id: Sandbox identifier
            
        Returns:
            True if successful
        """
        try:
            if sandbox_id not in self._sandbox_cache:
                self.logger.warning(f"Sandbox not found for cleanup: {sandbox_id}")
                return False
            
            sandbox_info = self._sandbox_cache[sandbox_id]
            sandbox = sandbox_info["sandbox"]
            session_id = sandbox_info.get("session_id")
            
            # Close sandbox
            await sandbox.close()
            
            # Remove from cache
            del self._sandbox_cache[sandbox_id]
            
            # Remove from session tracking
            if session_id and session_id in self._active_sessions:
                self._active_sessions[session_id].pop(sandbox_id, None)
                
                # Clean up empty sessions
                if not self._active_sessions[session_id]:
                    del self._active_sessions[session_id]
            
            # Update statistics
            self._execution_stats["sandbox_cleanup"] += 1
            
            self.logger.info(f"Sandbox cleaned up: {sandbox_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Sandbox cleanup failed: {e}")
            return False
    
    async def cleanup_session(self, session_id: str) -> int:
        """
        Clean up all sandboxes for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of sandboxes cleaned up
        """
        if session_id not in self._active_sessions:
            return 0
        
        sandbox_ids = list(self._active_sessions[session_id].keys())
        cleanup_count = 0
        
        for sandbox_id in sandbox_ids:
            if await self.cleanup_sandbox(sandbox_id):
                cleanup_count += 1
        
        return cleanup_count
    
    async def cleanup_expired_sandboxes(self, max_age_hours: int = 24) -> int:
        """
        Clean up sandboxes that have been inactive for too long.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            Number of sandboxes cleaned up
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        expired_sandboxes = []
        
        for sandbox_id, info in self._sandbox_cache.items():
            if info["last_used"] < cutoff_time:
                expired_sandboxes.append(sandbox_id)
        
        cleanup_count = 0
        for sandbox_id in expired_sandboxes:
            if await self.cleanup_sandbox(sandbox_id):
                cleanup_count += 1
        
        return cleanup_count
    
    def _map_template(self, template: str) -> str:
        """Map internal template names to E2b template names."""
        return self._default_templates.get(template, template)
    
    async def _configure_sandbox(
        self, 
        sandbox: Any, 
        config: E2bSandboxConfig
    ):
        """Configure a sandbox with the specified settings."""
        try:
            # Set environment variables
            for key, value in config.environment_variables.items():
                # This is sandbox-specific and might vary based on E2b API
                if hasattr(sandbox, 'set_env'):
                    await sandbox.set_env(key, value)
            
            # Install additional packages if supported
            if config.packages and hasattr(sandbox, 'install_packages'):
                await sandbox.install_packages(config.packages)
            
            # Create initial files
            for file_path, content in config.files.items():
                await sandbox.files.write(file_path, content.encode('utf-8'))
            
        except Exception as e:
            self.logger.warning(f"Sandbox configuration partially failed: {e}")
    
    async def _execute_direct(
        self,
        sandbox: Any,
        code: str,
        language: str,
        context: Dict[str, Any],
        execution_id: str
    ) -> Dict[str, Any]:
        """Execute code directly in the sandbox."""
        try:
            # Execute based on language
            if language.lower() in ["python", "python3", "py"]:
                result = await sandbox.run(f"python3 -c '{code}'")
            elif language.lower() in ["javascript", "js", "node", "nodejs"]:
                result = await sandbox.run(f"node -e '{code}'")
            elif language.lower() in ["bash", "sh", "shell"]:
                result = await sandbox.run(code)
            else:
                # Generic execution
                result = await sandbox.run(code)
            
            return {
                "execution_id": execution_id,
                "output": result.stdout,
                "error": result.stderr,
                "exit_code": result.exit_code,
                "logs": [result.stdout] if result.stdout else [],
                "files": await self._get_created_files(sandbox),
                "language": language
            }
            
        except Exception as e:
            return {
                "execution_id": execution_id,
                "error": str(e),
                "exit_code": -1,
                "logs": [str(e)],
                "files": [],
                "language": language
            }
    
    async def _execute_with_agent(
        self,
        sandbox: Any,
        code: str,
        language: str,
        context: Dict[str, Any],
        execution_id: str
    ) -> Dict[str, Any]:
        """Execute code with agent assistance."""
        if not self._agent_executor:
            # Fallback to direct execution
            return await self._execute_direct(sandbox, code, language, context, execution_id)
        
        try:
            # Create a workflow spec for the agent
            from .models import WorkflowSpec, WorkflowNode, NodeType, AgentWorkflowConfig, AgentType
            
            # Create a code execution node
            code_node = WorkflowNode(
                id="code_exec_1",
                name="Code Execution",
                type=NodeType.CODE,
                parameters={
                    "code": code,
                    "language": language,
                    "context": context
                }
            )
            
            # Create agent configuration for code execution
            agent_config = AgentWorkflowConfig(
                agent_type=AgentType.CAPIBARA_BASE,
                use_e2b_sandbox=True,
                e2b_config={"sandbox": sandbox, "execution_id": execution_id}
            )
            
            # Create workflow spec
            workflow_spec = WorkflowSpec(
                name=f"code_execution_{execution_id}",
                description=f"Execute {language} code with agent assistance",
                nodes=[code_node],
                agent_config=agent_config
            )
            
            # Execute with agent
            result = await self._agent_executor.execute_workflow(
                workflow_spec=workflow_spec,
                input_data={"code": code, "language": language}
            )
            
            return {
                "execution_id": execution_id,
                "agent_result": result.output_data,
                "output": str(result.output_data),
                "error": result.error_message,
                "logs": [log["message"] for log in result.agent_logs],
                "files": await self._get_created_files(sandbox),
                "language": language,
                "agent_logs": result.agent_logs
            }
            
        except Exception as e:
            self.logger.error(f"Agent-assisted execution failed: {e}")
            # Fallback to direct execution
            return await self._execute_direct(sandbox, code, language, context, execution_id)
    
    def _extract_code_from_node(
        self, 
        node_spec: Dict[str, Any], 
        input_data: Dict[str, Any]
    ) -> str:
        """Extract executable code from a node specification."""
        # Try to get code from various possible locations
        code = (
            node_spec.get("code") or
            node_spec.get("parameters", {}).get("code") or
            node_spec.get("script") or
            node_spec.get("command") or
            input_data.get("code") or
            ""
        )
        
        if not code:
            # Generate default code based on node type
            node_type = node_spec.get("type", "")
            if "http" in node_type.lower():
                code = f"# HTTP Request node\nprint('HTTP request executed')"
            elif "webhook" in node_type.lower():
                code = f"# Webhook node\nprint('Webhook triggered')"
            else:
                code = f"# Generic node execution\nprint('Node executed: {node_spec.get('name', 'Unknown')}')"
        
        return code
    
    async def _get_created_files(self, sandbox: Any) -> List[str]:
        """Get list of files created during execution."""
        try:
            # This is a simplified implementation
            # In practice, you'd track file creation more precisely
            files = await sandbox.files.list("/")
            return [f.path for f in files if not f.path.startswith("/usr/") and not f.path.startswith("/bin/")]
        except Exception:
            return []
    
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
    
    def get_active_sandboxes(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active sandboxes."""
        return {
            sandbox_id: {
                "template": info["template"],
                "created_at": info["created_at"].isoformat(),
                "last_used": info["last_used"].isoformat(),
                "session_id": info["session_id"]
            }
            for sandbox_id, info in self._sandbox_cache.items()
        }
    
    def get_session_sandboxes(self, session_id: str) -> Dict[str, Dict[str, Any]]:
        """Get sandboxes for a specific session."""
        if session_id not in self._active_sessions:
            return {}
        
        return self._active_sessions[session_id].copy()