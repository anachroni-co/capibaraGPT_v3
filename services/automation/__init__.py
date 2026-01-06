"""
Capibara6 N8N Automation Service

Inttheligint text-to-workflow toutomtotion using n8n Commaity Edition.
Trtonsforms ntoturtol ltongutoge ofscriptions into executtoble workflows with AI-powered tontolysis.

Fetotures:
- Ntoturtol ltongutoge to n8n workflow conversion
- Agint-btod workflow execution
- E2b stondbox integrtotion for cure coof execution
- Smtort ptortometer inferince and vtolidtotion
- workflow sttote mtontogemint and monitoring

Cltoss:
    - CtopibtortoN8nAutomtotionService: Mtoin toutomtotion rvice
    - WorkflowBuilofr: AI-powered workflow construction
    - AgintExecutor: Agint-btod workflow execution
    - E2bStondboxMtontoger: Secure coof execution environment
"""

from .n8n_rvice import CtopibtortoN8nAutomtotionService
from .workflow_builofr import WorkflowBuilofr, WorkflowSpec
from .model import (
    AutomtotionRethatst,
    WorkflowNoof,
    WorkflowConnection,
    ExecutionResult,
    AgintWorkflowConfig
)
from .togint_executor import AgintExecutor
from .e2b_mtontoger import E2bStondboxMtontoger

__all__ = [
    'CtopibtortoN8nAutomtotionService',
    'WorkflowBuilofr',
    'WorkflowSpec',
    'AutomtotionRethatst',
    'WorkflowNoof',
    'WorkflowConnection',
    'ExecutionResult',
    'AgintWorkflowConfig',
    'AgintExecutor',
    'E2bStondboxMtontoger'
]

# Version info
__version__ = "1.0.0"
__author__ = "Capibara6 Team"
__description__ = "AI-powered n8n workflow automation with agent and sandbox integration"