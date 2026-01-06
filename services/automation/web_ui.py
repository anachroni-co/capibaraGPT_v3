import sys
"""
Web UI for Capibara6 N8N Automation Service

# This module provides functionality for web_ui.
"""

import os

import logging
from typing import Any, Dict, List, Optional

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.append(project_root)

try:
    from fastapi import FastAPI, Request, Form, HTTPException
    from fastapi.templating import Jinja2Templates
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, RedirectResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from .n8n_service import CapibaraN8nAutomationService
from .models import AutomationRequest, ExecutionMode, AgentType


class CapibaraAutomationWebUI:
    """
    Web UI for the Capibara6 N8N Automation Service.
    
    Provides a simple, user-friendly interface for creating and managing
    automation workflows without technical knowledge.
    """
    
    def __init__(self, automation_service: CapibaraN8nAutomationService):
        """
        Initialize the web UI.
        
        Args:
            automation_service: The automation service instance
        """
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI is required for the web UI")
        
        self.automation_service = automation_service
        self.app = FastAPI(
            title="Capibara6 Automation Studio",
            description="Create workflows with natural language",
            version="1.0.0"
        )
        
        # Setup templates and static files
        self.templates_dir = Path(__file__).parent / "templates"
        self.static_dir = Path(__file__).parent / "static"
        
        # Create directories if they don't exist
        self.templates_dir.mkdir(exist_ok=True)
        self.static_dir.mkdir(exist_ok=True)
        
        self.templates = Jinja2Templates(directory=str(self.templates_dir))
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory=str(self.static_dir)), name="static")
        
        # Setup routes
        self._setup_routes()
        
        # Create default templates if they don't exist
        self._create_default_templates()
    
    def _setup_routes(self):
        """Setup web UI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            """Home page with workflow creation form."""
            return self.templates.TemplateResponse("home.html", {
                "request": request,
                "title": "Capibara6 Automation Studio"
            })
        
        @self.app.post("/create", response_class=HTMLResponse)
        async def create_workflow(
            request: Request,
            description: str = Form(...),
            execution_mode: str = Form("standard"),
            use_agents: bool = Form(False),
            use_sandbox: bool = Form(False),
            execute_now: bool = Form(False)
        ):
            """Creates a new workflow from the form."""
            try:
                # Determine execution mode
                if use_agents and use_sandbox:
                    mode = ExecutionMode.HYBRID
                elif use_agents:
                    mode = ExecutionMode.AGENT_BASED
                elif use_sandbox:
                    mode = ExecutionMode.E2B_SANDBOX
                else:
                    mode = ExecutionMode.STANDARD
                
                # Create automation request
                automation_request = AutomationRequest(
                    description=description,
                    execution_mode=mode,
                    require_sandbox=use_sandbox
                )
                
                # Create workflow
                result = await self.automation_service.create_automation(
                    request=automation_request,
                    execute_immediately=execute_now
                )
                
                return self.templates.TemplateResponse("workflow_created.html", {
                    "request": request,
                    "title": "Workflow Created",
                    "workflow_id": result["workflow_id"],
                    "workflow_name": result["workflow_spec"]["name"],
                    "description": description,
                    "execution_mode": mode,
                    "executed": execute_now,
                    "execution_result": result.get("execution_result"),
                    "nodes_count": len(result["workflow_spec"]["nodes"])
                })
                
            except Exception as e:
                return self.templates.TemplateResponse("error.html", {
                    "request": request,
                    "title": "Error",
                    "error_message": str(e)
                })
        
        @self.app.get("/workflows", response_class=HTMLResponse)
        async def list_workflows(request: Request, session_id: Optional[str] = None):
            """List workflows for a session."""
            try:
                if session_id:
                    session_info = await self.automation_service.get_session_info(session_id)
                    workflows = session_info["workflows"]
                    executions = session_info["recent_executions"]
                else:
                    workflows = []
                    executions = []
                
                return self.templates.TemplateResponse("workflows.html", {
                    "request": request,
                    "title": "My Workflows",
                    "session_id": session_id,
                    "workflows": workflows,
                    "executions": executions
                })
                
            except Exception as e:
                return self.templates.TemplateResponse("error.html", {
                    "request": request,
                    "title": "Error",
                    "error_message": str(e)
                })
        
        @self.app.get("/templates", response_class=HTMLResponse)
        async def workflow_templates(request: Request):
            """Show available workflow templates."""
            try:
                templates = await self.automation_service.get_workflow_templates()
                
                return self.templates.TemplateResponse("templates.html", {
                    "request": request,
                    "title": "Workflow Templates",
                    "templates": templates
                })
                
            except Exception as e:
                return self.templates.TemplateResponse("error.html", {
                    "request": request,
                    "title": "Error",
                    "error_message": str(e)
                })
        
        @self.app.post("/templates/{template_id}", response_class=HTMLResponse)
        async def use_template(
            request: Request,
            template_id: str,
            customization: str = Form("")
        ):
            """Creates workflow from template."""
            try:
                # Parse customization (simple key=value format)
                custom_params = {}
                if customization:
                    for line in customization.split('\n'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            custom_params[key.strip()] = value.strip()
                
                result = await self.automation_service.create_from_template(
                    template_id=template_id,
                    customization=custom_params
                )
                
                return self.templates.TemplateResponse("workflow_created.html", {
                    "request": request,
                    "title": "Workflow Created from Template",
                    "workflow_id": result["workflow_id"],
                    "workflow_name": result["workflow_spec"]["name"],
                    "description": result["workflow_spec"]["description"],
                    "template_id": template_id,
                    "nodes_count": len(result["workflow_spec"]["nodes"])
                })
                
            except Exception as e:
                return self.templates.TemplateResponse("error.html", {
                    "request": request,
                    "title": "Error",
                    "error_message": str(e)
                })
        
        @self.app.post("/execute/{workflow_id}")
        async def execute_workflow(workflow_id: str):
            """Execute a workflow and redirect to results."""
            try:
                result = await self.automation_service.execute_workflow(workflow_id)
                return RedirectResponse(
                    url=f"/results/{result.execution_id}",
                    status_code=303
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/results/{execution_id}", response_class=HTMLResponse)
        async def execution_results(request: Request, execution_id: str):
            """Show execution results."""
            # This would require storing execution results
            # For now, show a placeholder
            return self.templates.TemplateResponse("results.html", {
                "request": request,
                "title": "Execution Results",
                "execution_id": execution_id,
                "status": "completed",
                "message": "Execution completed successfully"
            })
        
        @self.app.get("/stats", response_class=HTMLResponse)
        async def service_stats(request: Request):
            """Show service statistics."""
            try:
                stats = self.automation_service.get_service_stats()
                
                return self.templates.TemplateResponse("stats.html", {
                    "request": request,
                    "title": "Service Statistics",
                    "stats": stats
                })
                
            except Exception as e:
                return self.templates.TemplateResponse("error.html", {
                    "request": request,
                    "title": "Error",
                    "error_message": str(e)
                })
    
    def _create_default_templates(self):
        """Creates default HTML templates."""
        
        # Base template
        base_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - Capibara6 Automation</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .capibara-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .workflow-card { transition: transform 0.2s; }
        .workflow-card:hover { transform: translateY(-2px); }
        .stats-card { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    </style>
</head>
<body class="bg-light">
    <nav class="navbar navbar-expand-lg navbar-dark capibara-header">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="fas fa-robot"></i> Capibara6 Automation</a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/"><i class="fas fa-home"></i> Home</a>
                <a class="nav-link" href="/templates"><i class="fas fa-th-large"></i> Templates</a>
                <a class="nav-link" href="/workflows"><i class="fas fa-list"></i> Workflows</a>
                <a class="nav-link" href="/stats"><i class="fas fa-chart-bar"></i> Stats</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        {% block content %}{% endblock %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>'''
        
        # Home template
        home_template = '''{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-lg-8 mx-auto">
        <div class="card shadow">
            <div class="card-header bg-primary text-white">
                <h3 class="mb-0"><i class="fas fa-magic"></i> Create Automation Workflow</h3>
            </div>
            <div class="card-body">
                <form action="/create" method="post">
                    <div class="mb-3">
                        <label for="description" class="form-label">Describe your automation:</label>
                        <textarea class="form-control" id="description" name="description" rows="4" 
                                placeholder="Example: Send me an email when someone submits the contact form" required></textarea>
                    </div>
                    
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="use_agents" name="use_agents">
                                <label class="form-check-label" for="use_agents">
                                    <i class="fas fa-brain"></i> Use AI Agents
                                </label>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="use_sandbox" name="use_sandbox">
                                <label class="form-check-label" for="use_sandbox">
                                    <i class="fas fa-shield-alt"></i> Secure Sandbox
                                </label>
                            </div>
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="execute_now" name="execute_now">
                            <label class="form-check-label" for="execute_now">
                                <i class="fas fa-play"></i> Execute immediately after creation
                            </label>
                        </div>
                    </div>
                    
                    <button type="submit" class="btn btn-primary btn-lg w-100">
                        <i class="fas fa-cog"></i> Create Workflow
                    </button>
                </form>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-4">
                <div class="card text-center">
                    <div class="card-body">
                        <i class="fas fa-comments fa-3x text-primary mb-3"></i>
                        <h5>Natural Language</h5>
                        <p>Describe workflows in plain English</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card text-center">
                    <div class="card-body">
                        <i class="fas fa-robot fa-3x text-success mb-3"></i>
                        <h5>AI-Powered</h5>
                        <p>Intelligent automation with Capibara agents</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card text-center">
                    <div class="card-body">
                        <i class="fas fa-shield-alt fa-3x text-warning mb-3"></i>
                        <h5>Secure</h5>
                        <p>Safe code execution with E2b sandboxes</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
        
        # Create template files
        templates = {
            "base.html": base_template,
            "home.html": home_template,
            "workflow_created.html": '''{% extends "base.html" %}
{% block content %}
<div class="alert alert-success">
    <h4><i class="fas fa-check-circle"></i> Workflow Created Successfully!</h4>
    <p><strong>ID:</strong> {{ workflow_id }}</p>
    <p><strong>Name:</strong> {{ workflow_name }}</p>
    <p><strong>Nodes:</strong> {{ nodes_count }}</p>
    {% if executed %}
    <p><strong>Status:</strong> {{ execution_result.status if execution_result else "Unknown" }}</p>
    {% endif %}
</div>
<a href="/" class="btn btn-primary">Create Another</a>
{% endblock %}''',
            "error.html": '''{% extends "base.html" %}
{% block content %}
<div class="alert alert-danger">
    <h4><i class="fas fa-exclamation-triangle"></i> error</h4>
    <p>{{ error_message }}</p>
</div>
<a href="/" class="btn btn-primary">Go Home</a>
{% endblock %}''',
            "workflows.html": '''{% extends "base.html" %}
{% block content %}
<h2>My Workflows</h2>
{% for workflow in workflows %}
<div class="card mb-3">
    <div class="card-body">
        <h5>{{ workflow.name }}</h5>
        <p>Created: {{ workflow.created_at }}</p>
        <button class="btn btn-primary btn-sm" onclick="executeWorkflow('{{ workflow.workflow_id }}')">Execute</button>
    </div>
</div>
{% endfor %}
{% endblock %}''',
            "templates.html": '''{% extends "base.html" %}
{% block content %}
<h2>Workflow Templates</h2>
{% for template in templates %}
<div class="card mb-3">
    <div class="card-body">
        <h5>{{ template.name }}</h5>
        <p>{{ template.description }}</p>
        <form action="/templates/{{ template.id }}" method="post" class="d-inline">
            <button type="submit" class="btn btn-primary">Use Template</button>
        </form>
    </div>
</div>
{% endfor %}
{% endblock %}''',
            "stats.html": '''{% extends "base.html" %}
{% block content %}
<h2>Service Statistics</h2>
<div class="row">
    <div class="col-md-3">
        <div class="card stats-card">
            <div class="card-body text-center">
                <h3>{{ stats.workflows_created }}</h3>
                <p>Workflows Created</p>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card stats-card">
            <div class="card-body text-center">
                <h3>{{ stats.workflows_executed }}</h3>
                <p>Executions</p>
            </div>
        </div>
    </div>
</div>
{% endblock %}''',
            "results.html": '''{% extends "base.html" %}
{% block content %}
<h2>Execution Results</h2>
<div class="card">
    <div class="card-body">
        <p><strong>Execution ID:</strong> {{ execution_id }}</p>
        <p><strong>Status:</strong> <span class="badge bg-success">{{ status }}</span></p>
        <p>{{ message }}</p>
    </div>
</div>
{% endblock %}'''
        }
        
        for filename, content in templates.items():
            template_path = self.templates_dir / filename
            if not template_path.exists():
                template_path.write_text(content)
    
    async def start(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the web UI server."""
        await self.automation_service.startup()
        uvicorn_config = uvicorn.Config(self.app, host=host, port=port)
        server = uvicorn.Server(uvicorn_config)
        await server.serve()


async def run_web_ui():
    """Run the web UI standalone."""
    from .n8n_service import create_automation_service
    
    # Create automation service
    automation_service = create_automation_service()
    
    # Create web UI
    web_ui = CapibaraAutomationWebUI(automation_service)
    
    # Start the server
    await web_ui.start(port=8080)

def main():
    # Main function for this module.
    logger.info("Module web_ui.py starting")
    return True

if __name__ == "__main__":
    main()
