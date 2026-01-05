#!/usr/bin/env python3
"""
Servidor simulado de Acontext para desarrollo
Este servidor proporciona endpoints de Acontext sin necesidad de la infraestructura completa
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Acontext Mock Server", version="1.0.0")

# Almacenamiento en memoria para simular la persistencia
sessions: Dict[str, Dict[str, Any]] = {}
spaces: Dict[str, Dict[str, Any]] = {}
messages: Dict[str, list] = {}  # session_id -> list of messages

@app.get("/health")
async def health():
    """Endpoint de salud"""
    return {"status": "healthy", "service": "acontext-mock-server", "timestamp": datetime.now().isoformat()}

@app.post("/api/v1/project/{project_id}/session")
async def create_session(request: Request, project_id: str):
    """Crear una nueva sesión"""
    body = await request.json() if await request.body() else {}
    space_id = body.get('space_id')
    
    session_id = str(uuid4())
    session = {
        "id": session_id,
        "project_id": project_id,
        "space_id": space_id,
        "created_at": datetime.now().isoformat()
    }
    
    sessions[session_id] = session
    messages[session_id] = []  # Inicializar lista de mensajes
    
    logger.info(f"📝 Created Acontext session: {session_id}")
    return {"id": session_id}

@app.post("/api/v1/session/{session_id}/messages")
async def send_message_to_session(request: Request, session_id: str):
    """Enviar un mensaje a una sesión"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    body = await request.json()
    message = body.get('blob', {})
    message['timestamp'] = datetime.now().isoformat()
    
    messages[session_id].append(message)
    
    logger.info(f"💬 Stored message for session {session_id}, total messages: {len(messages[session_id])}")
    return {"status": "ok", "message_id": str(uuid4())}

@app.post("/api/v1/project/{project_id}/session/{session_id}/flush")
async def flush_session(project_id: str, session_id: str):
    """Simular flush de sesión"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    logger.info(f"🔄 Flushed session {session_id}")
    return {"status": "ok"}

@app.post("/api/v1/project/{project_id}/space")
async def create_space(request: Request, project_id: str):
    """Crear un nuevo espacio"""
    body = await request.json()
    name = body.get('name', 'unnamed-space')
    
    space_id = str(uuid4())
    space = {
        "id": space_id,
        "name": name,
        "project_id": project_id,
        "created_at": datetime.now().isoformat()
    }
    
    spaces[space_id] = space
    
    logger.info(f"📁 Created Acontext space: {space_id}")
    return {"id": space_id}

@app.get("/api/v1/project/{project_id}/space/{space_id}/experience_search")
async def search_space(project_id: str, space_id: str, query: str, mode: str = "fast"):
    """Buscar en un espacio (funcionalidad simulada)"""
    if space_id not in spaces:
        raise HTTPException(status_code=404, detail=f"Space {space_id} not found")
    
    # Simular búsqueda de experiencias relevantes
    # En una implementación real, esto buscaría en vectores o contenido
    logger.info(f"🔍 Searched space {space_id} with query: '{query}' using mode: {mode}")
    
    # Devolver resultados simulados si hay coincidencias básicas
    simulated_results = []
    
    # Ejemplo de coincidencia simple para simular experiencia relevante
    if "code" in query.lower() or "programming" in query.lower():
        simulated_results = [
            {
                "block_id": str(uuid4()),
                "title": "Programming Best Practices",
                "type": "sop",
                "props": {
                    "use_when": "when writing code",
                    "tool_sops": [
                        {"tool_name": "think", "action": "analyze the requirements first"},
                        {"tool_name": "code", "action": "implement the solution step by step"}
                    ]
                },
                "distance": 0.6
            }
        ]
    elif "search" in query.lower() or "research" in query.lower():
        simulated_results = [
            {
                "block_id": str(uuid4()),
                "title": "Research Methodology",
                "type": "sop",
                "props": {
                    "use_when": "when conducting research",
                    "tool_sops": [
                        {"tool_name": "search", "action": "find reliable sources"},
                        {"tool_name": "analyze", "action": "evaluate the credibility of sources"}
                    ]
                },
                "distance": 0.5
            }
        ]
    
    logger.info(f"🎯 Found {len(simulated_results)} simulated results for query: '{query}'")
    return {
        "cited_blocks": simulated_results,
        "final_answer": None
    }

if __name__ == "__main__":
    port = int(os.getenv("ACONTEXT_MOCK_PORT", "8029"))
    logger.info(f"🚀 Starting Acontext Mock Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")