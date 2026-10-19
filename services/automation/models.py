"""
Data models for CapibaraGPT v3 N8N Automation Service
================================================

Pydantic models for workflow specification, execution results, 
agent configuration, and E2b sandbox integration.
"""

from typing import Dict, Any, List, Optional, Union, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass


class NodeType(str, Enum):
    """Supported n8n node types for workflow construction."""
    HTTP_REQUEST = "n8n-nodes-base.httpRequest"
    WEBHOOK = "n8n-nodes-base.webhook"
    SET = "n8n-nodes-base.set"
    IF = "n8n-nodes-base.if"
    FUNCTION = "n8n-nodes-base.function"
    CODE = "n8n-nodes-base.code"
    EMAIL = "n8n-nodes-base.emailSend"
    SLACK = "n8n-nodes-base.slack"
    TELEGRAM = "n8n-nodes-base.telegram"
    GOOGLE_SHEETS = "n8n-nodes-base.googleSheets"
    MYSQL = "n8n-nodes-base.mySql"
    POSTGRES = "n8n-nodes-base.postgres"
    MONGODB = "n8n-nodes-base.mongoDb"
    REDIS = "n8n-nodes-base.redis"
    SCHEDULE = "n8n-nodes-base.scheduleTrigger"
    MANUAL = "n8n-nodes-base.manualTrigger"
    
    # Agent-specific nodes
    CAPIBARA_AGENT = "capibara-nodes.agent"
    E2B_SANDBOX = "capibara-nodes.e2bSandbox"
    AGENT_EXECUTOR = "capibara-nodes.agentExecutor"


class ExecutionMode(str, Enum):
    """Workflow execution modes."""
    STANDARD = "standard"
    AGENT_BASED = "agent_based"
    E2B_SANDBOX = "e2b_sandbox"
    HYBRID = "hybrid"


class AgentType(str, Enum):
    """Types of Capibara agents available for workflow execution."""
    CAPIBARA_BASE = "capibara_base"
    CAPIBARA_AUTO = "capibara_auto"
    ULTRA_ORCHESTRATOR = "ultra_orchestrator"
    CUSTOM = "custom"


class WorkflowNode(BaseModel):
    """Represents a single node in an n8n workflow."""
    id: str = Field(..., description="Unique node identifier")
    name: str = Field(..., description="Human-readable node name")
    type: NodeType = Field(..., description="N8N node type")
    position: Dict[str, float] = Field(default_factory=dict, description="Node position in UI")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Node configuration parameters")
    
    # Agent-specific fields
    agent_type: Optional[AgentType] = Field(None, description="Type of agent if this is an agent node")
    agent_config: Optional[Dict[str, Any]] = Field(None, description="Agent-specific configuration")
    
    # E2b sandbox fields
    sandbox_template: Optional[str] = Field(None, description="E2b sandbox template")
    sandbox_config: Optional[Dict[str, Any]] = Field(None, description="E2b sandbox configuration")
    
    @validator('position', pre=True, always=True)
    def set_default_position(cls, v):
        if not v:
            return {"x": 0, "y": 0}
        return v


class WorkflowConnection(BaseModel):
    """Represents a connection between two workflow nodes."""
    source_node: str = Field(..., description="Source node ID")
    target_node: str = Field(..., description="Target node ID")
    source_index: int = Field(0, description="Source output index")
    target_index: int = Field(0, description="Target input index")
    connection_type: str = Field("main", description="Type of connection")


class AgentWorkflowConfig(BaseModel):
    """Configuration for agent-based workflow execution."""
    agent_type: AgentType = Field(..., description="Type of agent to use")
    agent_parameters: Dict[str, Any] = Field(default_factory=dict, description="Agent initialization parameters")
    execution_mode: ExecutionMode = Field(ExecutionMode.AGENT_BASED, description="How to execute the workflow")
    
    # Memory and state management
    use_memory: bool = Field(True, description="Whether to use agent memory")
    memory_config: Dict[str, Any] = Field(default_factory=dict, description="Memory configuration")
    
    # E2b integration
    use_e2b_sandbox: bool = Field(False, description="Whether to use E2b sandbox for code execution")
    e2b_template: Optional[str] = Field(None, description="E2b sandbox template")
    e2b_config: Dict[str, Any] = Field(default_factory=dict, description="E2b sandbox configuration")
    
    # Performance settings
    max_execution_time: int = Field(300, description="Maximum execution time in seconds")
    max_memory_usage: int = Field(1024, description="Maximum memory usage in MB")


class AutomationRequest(BaseModel):
    """Request to create an automation workflow from natural language."""
    description: str = Field(..., description="Natural language description of the automation")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context information")
    
    # Execution preferences
    execution_mode: ExecutionMode = Field(ExecutionMode.STANDARD, description="Preferred execution mode")
    agent_config: Optional[AgentWorkflowConfig] = Field(None, description="Agent configuration if using agents")
    
    # Template and customization
    template_id: Optional[str] = Field(None, description="Base template to customize")
    custom_nodes: List[Dict[str, Any]] = Field(default_factory=list, description="Custom node definitions")
    
    # Security and sandbox settings
    require_sandbox: bool = Field(False, description="Whether to require sandbox execution")
    security_level: Literal["low", "medium", "high"] = Field("medium", description="Security level")
    
    # Metadata
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation continuity")


class WorkflowSpec(BaseModel):
    """Complete specification of an n8n workflow."""
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    nodes: List[WorkflowNode] = Field(..., description="List of workflow nodes")
    connections: List[WorkflowConnection] = Field(default_factory=list, description="Node connections")
    
    # Execution configuration
    execution_mode: ExecutionMode = Field(ExecutionMode.STANDARD, description="How to execute this workflow")
    agent_config: Optional[AgentWorkflowConfig] = Field(None, description="Agent configuration")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Workflow tags")
    version: str = Field("1.0.0", description="Workflow version")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    # Settings
    settings: Dict[str, Any] = Field(default_factory=dict, description="Workflow settings")
    
    @validator('nodes')
    def validate_nodes(cls, v):
        if not v:
            raise ValueError("Workflow must have at least one node")
        return v


class ExecutionResult(BaseModel):
    """Result of workflow execution."""
    workflow_id: str = Field(..., description="Executed workflow ID")
    execution_id: str = Field(..., description="Unique execution identifier")
    status: Literal["success", "failed", "running", "cancelled"] = Field(..., description="Execution status")
    
    # Results
    output_data: Dict[str, Any] = Field(default_factory=dict, description="Execution output data")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Timing
    started_at: datetime = Field(..., description="Execution start time")
    finished_at: Optional[datetime] = Field(None, description="Execution end time")
    duration_ms: Optional[int] = Field(None, description="Execution duration in milliseconds")
    
    # Agent-specific results
    agent_logs: List[Dict[str, Any]] = Field(default_factory=list, description="Agent execution logs")
    agent_memory: Optional[Dict[str, Any]] = Field(None, description="Agent memory state after execution")
    
    # E2b sandbox results
    sandbox_logs: List[str] = Field(default_factory=list, description="Sandbox execution logs")
    sandbox_files: List[str] = Field(default_factory=list, description="Files created in sandbox")
    
    # Performance metrics
    memory_used_mb: Optional[float] = Field(None, description="Memory usage in MB")
    cpu_time_ms: Optional[float] = Field(None, description="CPU time in milliseconds")


class E2bSandboxConfig(BaseModel):
    """Configuration for E2b sandbox environment."""
    template: str = Field("python3", description="Sandbox template")
    timeout: int = Field(300, description="Execution timeout in seconds")
    memory_limit: int = Field(1024, description="Memory limit in MB")
    
    # Environment setup
    environment_variables: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    packages: List[str] = Field(default_factory=list, description="Additional packages to install")
    files: Dict[str, str] = Field(default_factory=dict, description="Files to create in sandbox")
    
    # Security settings
    network_access: bool = Field(False, description="Whether to allow network access")
    file_system_access: bool = Field(True, description="Whether to allow file system access")
    
    # Integration with agents
    agent_integration: bool = Field(False, description="Whether to integrate with Capibara agents")
    agent_communication_channel: Optional[str] = Field(None, description="Channel for agent communication")


@dataclass
class WorkflowTemplate:
    """Template for common workflow patterns."""
    id: str
    name: str
    description: str
    category: str
    nodes: List[Dict[str, Any]]
    connections: List[Dict[str, Any]]
    
    # Agent and E2b support
    supports_agents: bool = False
    supports_e2b: bool = False
    default_agent_type: Optional[AgentType] = None
    default_e2b_template: Optional[str] = None
    
    # Example usage
    example_requests: List[str] = None
    setup_instructions: Optional[str] = None