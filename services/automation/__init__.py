"""
CapibaraGPT v3 N8N Automation Service (experimental).

Intelligent text-to-workflow automation using n8n Community Edition.
Transforms natural language descriptions into executable workflows with AI-powered analysis.

Features:
- Natural language to n8n workflow conversion
- Agent-based workflow execution (if Capibara agents are available)
- E2b sandbox integration for secure code execution (optional)
- Smart parameter inference and validation
- Workflow state management and monitoring

Classes:
- CapibaraN8nAutomationService: Main automation service
- WorkflowBuilder: AI-powered workflow construction
- AgentExecutor: Agent-based workflow execution
- E2bSandboxManager: Secure code execution environment
"""

from .n8n_service import CapibaraN8nAutomationService
from .workflow_builder import WorkflowBuilder, WorkflowSpec
from .models import (
    AutomationRequest,
    WorkflowNode,
    WorkflowConnection,
    ExecutionResult,
    AgentWorkflowConfig
)
from .agent_executor import AgentExecutor
from .e2b_manager import E2bSandboxManager

__all__ = [
    'CapibaraN8nAutomationService',
    'WorkflowBuilder',
    'WorkflowSpec',
    'AutomationRequest',
    'WorkflowNode',
    'WorkflowConnection',
    'ExecutionResult',
    'AgentWorkflowConfig',
    'AgentExecutor',
    'E2bSandboxManager'
]

# Version info
__version__ = "1.0.0"
__author__ = "CapibaraGPT Team"
__description__ = "AI-powered n8n workflow automation with agent and sandbox integration"
