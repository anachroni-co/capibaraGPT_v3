"""
AI-Powered Workflow Builder for Capibara6 N8N Integration

# This module provides functionality for workflow_builder.
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

from .models import (
    WorkflowSpec, WorkflowNode, WorkflowConnection, NodeType,
    AutomationRequest, ExecutionMode, AgentType, AgentWorkflowConfig,
    WorkflowTemplate, E2bSandboxConfig
)

# Import Capibara AI components
try:
    from ...agents.capibara_agent import CapibaraTool
    from ...agents.ultra_agent_orchestrator import UltraAgentOrchestrator
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logging.warning("Capibara AI components not available")


class WorkflowBuilder:
    """
    AI-powered workflow builder that converts natural language to n8n workflows.
    
    Features:
    - Natural language understanding for workflow creation
    - Intelligent node type selection and parameter inference
    - Automatic connection generation between nodes
    - Template-based workflow generation
    - Agent and E2b integration support
    - Workflow validation and optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the workflow builder.
        
        Args:
            config: Configuration dictionary for the builder
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # AI models for workflow generation
        self._ai_agent = None
        if AI_AVAILABLE:
            try:
                self._ai_agent = UltraAgentOrchestrator(self.config)
            except Exception as e:
                self.logger.warning(f"Could not initialize AI agent: {e}")
                try:
                    self._ai_agent = CapibaraTool()
                except Exception as e2:
                    self.logger.warning(f"Could not initialize fallback AI agent: {e2}")
        
        # Built-in templates
        self._templates = self._initialize_templates()
        
        # Node type patterns for natural language matching
        self._node_patterns = self._initialize_node_patterns()
        
        # Common workflow patterns
        self._workflow_patterns = self._initialize_workflow_patterns()
    
    async def build_workflow(
        self, 
        request: AutomationRequest
    ) -> WorkflowSpec:
        """
        Build a workflow from a natural language request.
        
        Args:
            request: Automation request with description and preferences
            
        Returns:
            Complete workflow specification
        """
        try:
            self.logger.info(f"Building workflow from request: {request.description[:100]}...")
            
            # Analyze the request using AI
            analysis = await self._analyze_request(request)
            
            # Generate workflow structure
            workflow_structure = await self._generate_workflow_structure(analysis, request)
            
            # Create nodes based on analysis
            nodes = await self._create_nodes(workflow_structure, request)
            
            # Generate connections between nodes
            connections = await self._create_connections(nodes, workflow_structure)
            
            # Configure execution mode and agent settings
            execution_config = await self._configure_execution(request, analysis)
            
            # Create workflow specification
            workflow_spec = WorkflowSpec(
                name=analysis.get("workflow_name", "Generated Workflow"),
                description=analysis.get("workflow_description", request.description),
                nodes=nodes,
                connections=connections,
                execution_mode=execution_config["mode"],
                agent_config=execution_config.get("agent_config"),
                tags=analysis.get("tags", []),
                settings=analysis.get("settings", {})
            )
            
            # Validate and optimize workflow
            validated_workflow = await self._validate_and_optimize(workflow_spec, request)
            
            self.logger.info(f"Workflow built successfully: {validated_workflow.name}")
            return validated_workflow
            
        except Exception as e:
            self.logger.error(f"Workflow building failed: {e}", exc_info=True)
            # Return a basic fallback workflow
            return self._create_fallback_workflow(request)
    
    async def build_from_template(
        self, 
        template_id: str, 
        customization: Dict[str, Any] = None
    ) -> WorkflowSpec:
        """
        Build a workflow from a predefined template.
        
        Args:
            template_id: Template identifier
            customization: Customization parameters
            
        Returns:
            Workflow specification based on template
        """
        if template_id not in self._templates:
            raise ValueError(f"Template not found: {template_id}")
        
        template = self._templates[template_id]
        customization = customization or {}
        
        # Create nodes from template
        nodes = []
        for i, node_template in enumerate(template.nodes):
            node = WorkflowNode(
                id=f"node_{i+1}",
                name=node_template.get("name", f"Node {i+1}"),
                type=NodeType(node_template["type"]),
                position={"x": i * 200, "y": 100},
                parameters=node_template.get("parameters", {})
            )
            
            # Apply customizations
            if node.name in customization:
                custom_params = customization[node.name]
                node.parameters.update(custom_params)
            
            nodes.append(node)
        
        # Create connections from template
        connections = []
        for conn_template in template.connections:
            connection = WorkflowConnection(
                source_node=f"node_{conn_template['source'] + 1}",
                target_node=f"node_{conn_template['target'] + 1}",
                source_index=conn_template.get("source_index", 0),
                target_index=conn_template.get("target_index", 0)
            )
            connections.append(connection)
        
        # Configure execution for template
        execution_mode = ExecutionMode.STANDARD
        agent_config = None
        
        if template.supports_agents:
            execution_mode = ExecutionMode.AGENT_BASED
            agent_config = AgentWorkflowConfig(
                agent_type=template.default_agent_type or AgentType.CAPIBARA_BASE,
                use_e2b_sandbox=template.supports_e2b,
                e2b_template=template.default_e2b_template
            )
        elif template.supports_e2b:
            execution_mode = ExecutionMode.E2B_SANDBOX
        
        return WorkflowSpec(
            name=template.name,
            description=template.description,
            nodes=nodes,
            connections=connections,
            execution_mode=execution_mode,
            agent_config=agent_config,
            tags=[template.category],
            settings=customization.get("settings", {})
        )
    
    async def _analyze_request(self, request: AutomationRequest) -> Dict[str, Any]:
        """Analyze the automation request using AI."""
        if not self._ai_agent:
            return self._analyze_request_fallback(request)
        
        try:
            # Create analysis prompt
            analysis_prompt = f"""
        except Exception: pass
Analyze the following automation request and extract key information:

Request: "{request.description}"
Context: {json.dumps(request.context or {}, indent=2)}

Please provide analysis in the following format:
{{
    "workflow_name": "Descriptive workflow name",
    "workflow_description": "Clear description of what the workflow does",
    "main_actions": ["action1", "action2", "action3"],
    "required_nodes": ["node_type1", "node_type2"],
    "data_flow": "Description of how data flows through the workflow",
    "triggers": ["trigger_type"],
    "integrations": ["service1", "service2"],
    "complexity": "low|medium|high",
    "estimated_nodes": 3,
    "requires_code": true/false,
    "requires_agents": true/false,
    "requires_sandbox": true/false,
    "tags": ["tag1", "tag2"],
    "settings": {{"key": "value"}}
}}
"""
            
            # Get AI analysis
            if hasattr(self._ai_agent, 'analyze_workflow_request'):
                analysis = await self._ai_agent.analyze_workflow_request(analysis_prompt, request)
            elif hasattr(self._ai_agent, 'process'):
                analysis_result = await self._ai_agent.process(analysis_prompt, {"request": request})
                analysis = self._parse_ai_analysis(analysis_result)
            else:
                # Fallback execution
                analysis = self._analyze_request_fallback(request)
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"AI analysis failed, using fallback: {e}")
            return self._analyze_request_fallback(request)
    
    def _analyze_request_fallback(self, request: AutomationRequest) -> Dict[str, Any]:
        """Fallback analysis using pattern matching."""
        description = request.description.lower()
        
        # Extract workflow name
        workflow_name = "Custom Automation"
        if "when" in description or "if" in description:
            workflow_name = "Conditional Automation"
        elif "schedule" in description or "daily" in description or "hourly" in description:
            workflow_name = "Scheduled Automation"
        elif "webhook" in description or "api" in description:
            workflow_name = "API Automation"
        
        # Identify main actions
        actions = []
        if any(word in description for word in ["send", "email", "notify"]):
            actions.append("send_notification")
        if any(word in description for word in ["fetch", "get", "retrieve", "download"]):
            actions.append("fetch_data")
        if any(word in description for word in ["save", "store", "write", "upload"]):
            actions.append("store_data")
        if any(word in description for word in ["process", "transform", "convert"]):
            actions.append("process_data")
        
        # Identify required nodes
        required_nodes = ["n8n-nodes-base.manualTrigger"]  # Default trigger
        
        if any(word in description for word in ["webhook", "api", "http"]):
            required_nodes.extend(["n8n-nodes-base.webhook", "n8n-nodes-base.httpRequest"])
        if any(word in description for word in ["email", "mail"]):
            required_nodes.append("n8n-nodes-base.emailSend")
        if any(word in description for word in ["slack"]):
            required_nodes.append("n8n-nodes-base.slack")
        if any(word in description for word in ["database", "sql", "mysql", "postgres"]):
            required_nodes.extend(["n8n-nodes-base.mySql", "n8n-nodes-base.postgres"])
        if any(word in description for word in ["schedule", "timer", "daily", "hourly"]):
            required_nodes.append("n8n-nodes-base.scheduleTrigger")
        if any(word in description for word in ["code", "script", "function"]):
            required_nodes.append("n8n-nodes-base.function")
        
        # Determine complexity
        complexity = "low"
        if len(actions) > 3 or "code" in description:
            complexity = "medium"
        if any(word in description for word in ["complex", "advanced", "multiple", "integrate"]):
            complexity = "high"
        
        # Check for special requirements
        requires_code = any(word in description for word in ["code", "script", "function", "calculate", "transform"])
        requires_agents = any(word in description for word in ["ai", "intelligent", "smart", "analyze"])
        requires_sandbox = requires_code and any(word in description for word in ["secure", "isolated", "safe"])
        
        return {
            "workflow_name": workflow_name,
            "workflow_description": request.description,
            "main_actions": actions,
            "required_nodes": required_nodes,
            "data_flow": "Sequential processing of automation steps",
            "triggers": ["manual"],
            "integrations": self._extract_integrations(description),
            "complexity": complexity,
            "estimated_nodes": len(required_nodes),
            "requires_code": requires_code,
            "requires_agents": requires_agents,
            "requires_sandbox": requires_sandbox,
            "tags": self._extract_tags(description),
            "settings": {}
        }
    
    def _extract_integrations(self, description: str) -> List[str]:
        """Extract mentioned integrations from description."""
        integrations = []
        integration_keywords = {
            "slack": "slack",
            "email": "email",
            "gmail": "gmail",
            "sheets": "google_sheets",
            "google": "google",
            "database": "database",
            "mysql": "mysql",
            "postgres": "postgresql",
            "mongodb": "mongodb",
            "redis": "redis",
            "webhook": "webhook",
            "api": "api",
            "http": "http"
        }
        
        for keyword, integration in integration_keywords.items():
            if keyword in description:
                integrations.append(integration)
        
        return list(set(integrations))
    
    def _extract_tags(self, description: str) -> List[str]:
        """Extract relevant tags from description."""
        tags = []
        tag_keywords = {
            "notification": ["notify", "alert", "send", "email"],
            "data": ["data", "fetch", "store", "process"],
            "scheduled": ["schedule", "daily", "hourly", "timer"],
            "api": ["api", "webhook", "http", "rest"],
            "automation": ["automate", "automatic", "trigger"],
            "integration": ["integrate", "connect", "sync"]
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in description for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    def _parse_ai_analysis(self, ai_result: Any) -> Dict[str, Any]:
        """Parse AI analysis result into structured format."""
        try:
            if isinstance(ai_result, dict):
                return ai_result
            elif isinstance(ai_result, str):
                # Try to extract JSON from string
                json_match = re.search(r'\{.*\}', ai_result, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            
            # If parsing fails, return basic structure
            return {
                "workflow_name": "AI Generated Workflow",
                "workflow_description": str(ai_result)[:200],
                "main_actions": ["process"],
                "required_nodes": ["n8n-nodes-base.manualTrigger"],
                "complexity": "medium",
                "estimated_nodes": 2,
                "requires_code": False,
                "requires_agents": True,
                "requires_sandbox": False,
                "tags": ["ai_generated"],
                "settings": {}
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse AI analysis: {e}")
            return self._analyze_request_fallback(AutomationRequest(description=str(ai_result)))
    
    async def _generate_workflow_structure(
        self, 
        analysis: Dict[str, Any], 
        request: AutomationRequest
    ) -> Dict[str, Any]:
        """Generates the overall workflow structure."""
        structure = {
            "trigger_nodes": [],
            "processing_nodes": [],
            "action_nodes": [],
            "conditional_nodes": [],
            "code_nodes": []
        }
        
        # Add trigger nodes
        if "webhook" in analysis.get("triggers", []):
            structure["trigger_nodes"].append({"type": "n8n-nodes-base.webhook", "name": "Webhook Trigger"})
        elif "schedule" in analysis.get("triggers", []):
            structure["trigger_nodes"].append({"type": "n8n-nodes-base.scheduleTrigger", "name": "Schedule Trigger"})
        else:
            structure["trigger_nodes"].append({"type": "n8n-nodes-base.manualTrigger", "name": "Manual Trigger"})
        
        # Add processing nodes based on actions
        for action in analysis.get("main_actions", []):
            if action == "fetch_data":
                structure["processing_nodes"].append({"type": "n8n-nodes-base.httpRequest", "name": "Fetch Data"})
            elif action == "process_data":
                if analysis.get("requires_code"):
                    structure["code_nodes"].append({"type": "n8n-nodes-base.function", "name": "Process Data"})
                else:
                    structure["processing_nodes"].append({"type": "n8n-nodes-base.set", "name": "Set Data"})
            elif action == "store_data":
                # Choose storage based on integrations
                if "database" in analysis.get("integrations", []):
                    structure["action_nodes"].append({"type": "n8n-nodes-base.mySql", "name": "Store in Database"})
                else:
                    structure["processing_nodes"].append({"type": "n8n-nodes-base.set", "name": "Store Data"})
            elif action == "send_notification":
                if "slack" in analysis.get("integrations", []):
                    structure["action_nodes"].append({"type": "n8n-nodes-base.slack", "name": "Send Slack Message"})
                elif "email" in analysis.get("integrations", []):
                    structure["action_nodes"].append({"type": "n8n-nodes-base.emailSend", "name": "Send Email"})
        
        # Add conditional logic if needed
        if analysis.get("complexity") != "low" or "if" in request.description.lower():
            structure["conditional_nodes"].append({"type": "n8n-nodes-base.if", "name": "Conditional Logic"})
        
        # Add agent or E2b nodes if required
        if analysis.get("requires_agents"):
            structure["processing_nodes"].append({"type": "capibara-nodes.agent", "name": "AI Agent"})
        
        if analysis.get("requires_sandbox"):
            structure["code_nodes"].append({"type": "capibara-nodes.e2bSandbox", "name": "Secure Code Execution"})
        
        return structure
    
    async def _create_nodes(
        self, 
        structure: Dict[str, Any], 
        request: AutomationRequest
    ) -> List[WorkflowNode]:
        """Creates workflow nodes from structure."""
        nodes = []
        node_id_counter = 1
        
        # Create all nodes from structure
        all_node_groups = [
            structure["trigger_nodes"],
            structure["processing_nodes"],
            structure["conditional_nodes"],
            structure["code_nodes"],
            structure["action_nodes"]
        ]
        
        y_position = 100
        for node_group in all_node_groups:
            x_position = 100
            for node_spec in node_group:
                node = WorkflowNode(
                    id=f"node_{node_id_counter}",
                    name=node_spec["name"],
                    type=NodeType(node_spec["type"]),
                    position={"x": x_position, "y": y_position},
                    parameters=await self._generate_node_parameters(node_spec, request)
                )
                
                # Add agent-specific configuration
                if node_spec["type"] == "capibara-nodes.agent":
                    node.agent_type = AgentType.CAPIBARA_BASE
                    node.agent_config = {"use_context": True}
                
                # Add E2b-specific configuration
                if node_spec["type"] == "capibara-nodes.e2bSandbox":
                    node.sandbox_template = "python3"
                    node.sandbox_config = {"timeout": 300, "memory_limit": 1024}
                
                nodes.append(node)
                node_id_counter += 1
                x_position += 200
            
            y_position += 150
        
        return nodes
    
    async def _generate_node_parameters(
        self, 
        node_spec: Dict[str, Any], 
        request: AutomationRequest
    ) -> Dict[str, Any]:
        """Generates parameters for a specific node."""
        node_type = node_spec["type"]
        parameters = {}
        
        if node_type == "n8n-nodes-base.webhook":
            parameters = {
                "path": "automation-webhook",
                "httpMethod": "POST",
                "responseMode": "onReceived"
            }
        elif node_type == "n8n-nodes-base.scheduleTrigger":
            parameters = {
                "rule": {"interval": [{"field": "hours", "value": 1}]},
                "triggerAtStartup": False
            }
        elif node_type == "n8n-nodes-base.httpRequest":
            parameters = {
                "method": "GET",
                "url": "https://api.example.com/data",
                "options": {}
            }
        elif node_type == "n8n-nodes-base.function":
            parameters = {
                "functionCode": self._generate_function_code(request)
            }
        elif node_type == "n8n-nodes-base.set":
            parameters = {
                "values": {
                    "string": [{"name": "processed", "value": "true"}]
                }
            }
        elif node_type == "n8n-nodes-base.if":
            parameters = {
                "conditions": {
                    "string": [{"value1": "{{ $json.data }}", "operation": "isNotEmpty"}]
                }
            }
        elif node_type == "n8n-nodes-base.emailSend":
            parameters = {
                "fromEmail": "automation@example.com",
                "toEmail": "user@example.com",
                "subject": "Automation Notification",
                "text": "Your automation has completed successfully."
            }
        elif node_type == "n8n-nodes-base.slack":
            parameters = {
                "channel": "#general",
                "text": "Automation notification: {{ $json.message }}"
            }
        elif node_type == "capibara-nodes.agent":
            parameters = {
                "agent_type": "capibara_base",
                "prompt": f"Process the following automation request: {request.description}",
                "use_memory": True
            }
        elif node_type == "capibara-nodes.e2bSandbox":
            parameters = {
                "template": "python3",
                "code": self._generate_sandbox_code(request),
                "timeout": 300
            }
        
        return parameters
    
    def _generate_function_code(self, request: AutomationRequest) -> str:
        """Generates JavaScript function code for the request."""
        return """
// Auto-generated function code
const inputData = $input.all();

// Process the data
const processedData = inputData.map(item => {
    // Add your processing logic here
    return {
        ...item.json,
        processed: true,
        timestamp: new Date().toISOString()
    };
});

return processedData;
"""
    
    def _generate_sandbox_code(self, request: AutomationRequest) -> str:
        """Generates Python code for E2b sandbox execution."""
        return f"""
# Auto-generated Python code for: {request.description}
import json
import sys

def main():
    # Main function for this module.
    logger.info("Module workflow_builder.py starting")
    return True

if __name__ == "__main__":
    main()
"""
