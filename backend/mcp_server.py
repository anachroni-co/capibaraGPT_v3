#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servidor MCP para capibara6
Integración con el backend Flask existente
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import queue
import time

from mcp_connector import Capibara6MCPConnector

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPServer:
    """
    Servidor MCP que integra capibara6 con el protocolo MCP
    """
    
    def __init__(self, host: str = "localhost", port: int = 3000):
        self.host = host
        self.port = port
        self.connector = Capibara6MCPConnector()
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.running = False
        
    async def start_server(self):
        """Iniciar servidor MCP"""
        logger.info(f"Iniciando servidor MCP en {self.host}:{self.port}")
        self.running = True
        
        # Simular servidor MCP (en implementación real usaría stdio o HTTP)
        while self.running:
            try:
                # Procesar solicitudes de la cola
                if not self.request_queue.empty():
                    request_data = self.request_queue.get_nowait()
                    response = await self.connector.handle_request(request_data)
                    self.response_queue.put(response)
                
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error en servidor MCP: {e}")
                break
    
    def stop_server(self):
        """Detener servidor MCP"""
        self.running = False
        logger.info("Servidor MCP detenido")
    
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Procesar solicitud MCP de forma síncrona"""
        # Agregar a la cola de solicitudes
        self.request_queue.put(request_data)
        
        # Esperar respuesta
        timeout = 10  # segundos
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not self.response_queue.empty():
                return self.response_queue.get_nowait()
            time.sleep(0.1)
        
        return {
            "jsonrpc": "2.0",
            "id": request_data.get("id"),
            "error": {
                "code": -32603,
                "message": "Timeout waiting for response"
            }
        }

# Instancia global del servidor MCP
mcp_server = MCPServer()

# Iniciar servidor MCP en hilo separado
def start_mcp_server():
    """Iniciar servidor MCP en hilo separado"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(mcp_server.start_server())

mcp_thread = threading.Thread(target=start_mcp_server, daemon=True)
mcp_thread.start()

# Integración con Flask
app = Flask(__name__)
CORS(app)

@app.route('/api/mcp/initialize', methods=['POST'])
def mcp_initialize():
    """Endpoint para inicializar conexión MCP"""
    try:
        request_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": request.get_json() or {}
        }
        
        response = mcp_server.process_request(request_data)
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error en inicialización MCP: {e}")
        return jsonify({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }), 500

@app.route('/api/mcp/tools/list', methods=['GET', 'POST'])
def mcp_tools_list():
    """Endpoint para listar herramientas MCP"""
    try:
        request_data = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": request.get_json() or {}
        }
        
        response = mcp_server.process_request(request_data)
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error listando herramientas MCP: {e}")
        return jsonify({
            "jsonrpc": "2.0",
            "id": 2,
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }), 500

@app.route('/api/mcp/tools/call', methods=['POST'])
def mcp_tools_call():
    """Endpoint para ejecutar herramienta MCP"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Datos requeridos"}), 400
        
        request_data = {
            "jsonrpc": "2.0",
            "id": data.get("id", 3),
            "method": "tools/call",
            "params": data
        }
        
        response = mcp_server.process_request(request_data)
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error ejecutando herramienta MCP: {e}")
        return jsonify({
            "jsonrpc": "2.0",
            "id": request.get_json().get("id", 3) if request.get_json() else 3,
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }), 500

@app.route('/api/mcp/resources/list', methods=['GET', 'POST'])
def mcp_resources_list():
    """Endpoint para listar recursos MCP"""
    try:
        request_data = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/list",
            "params": request.get_json() or {}
        }
        
        response = mcp_server.process_request(request_data)
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error listando recursos MCP: {e}")
        return jsonify({
            "jsonrpc": "2.0",
            "id": 4,
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }), 500

@app.route('/api/mcp/resources/read', methods=['POST'])
def mcp_resources_read():
    """Endpoint para leer recurso MCP"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Datos requeridos"}), 400
        
        request_data = {
            "jsonrpc": "2.0",
            "id": data.get("id", 5),
            "method": "resources/read",
            "params": data
        }
        
        response = mcp_server.process_request(request_data)
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error leyendo recurso MCP: {e}")
        return jsonify({
            "jsonrpc": "2.0",
            "id": request.get_json().get("id", 5) if request.get_json() else 5,
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }), 500

@app.route('/api/mcp/prompts/list', methods=['GET', 'POST'])
def mcp_prompts_list():
    """Endpoint para listar prompts MCP"""
    try:
        request_data = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "prompts/list",
            "params": request.get_json() or {}
        }
        
        response = mcp_server.process_request(request_data)
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error listando prompts MCP: {e}")
        return jsonify({
            "jsonrpc": "2.0",
            "id": 6,
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }), 500

@app.route('/api/mcp/prompts/get', methods=['POST'])
def mcp_prompts_get():
    """Endpoint para obtener prompt MCP"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Datos requeridos"}), 400
        
        request_data = {
            "jsonrpc": "2.0",
            "id": data.get("id", 7),
            "method": "prompts/get",
            "params": data
        }
        
        response = mcp_server.process_request(request_data)
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error obteniendo prompt MCP: {e}")
        return jsonify({
            "jsonrpc": "2.0",
            "id": request.get_json().get("id", 7) if request.get_json() else 7,
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }), 500

@app.route('/api/mcp/status', methods=['GET'])
def mcp_status():
    """Endpoint para verificar estado del servidor MCP"""
    return jsonify({
        "status": "running",
        "connector": "capibara6-mcp-connector",
        "version": "1.0.0",
        "capabilities": mcp_server.connector.capabilities,
        "timestamp": time.time()
    })

@app.route('/api/mcp/test', methods=['POST'])
def mcp_test():
    """Endpoint para probar funcionalidad MCP"""
    try:
        data = request.get_json() or {}
        test_type = data.get("test_type", "full")
        
        results = {}
        
        if test_type in ["full", "tools"]:
            # Test de herramientas
            tools_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            }
            tools_response = mcp_server.process_request(tools_request)
            results["tools"] = tools_response
        
        if test_type in ["full", "resources"]:
            # Test de recursos
            resources_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "resources/list",
                "params": {}
            }
            resources_response = mcp_server.process_request(resources_request)
            results["resources"] = resources_response
        
        if test_type in ["full", "prompts"]:
            # Test de prompts
            prompts_request = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "prompts/list",
                "params": {}
            }
            prompts_response = mcp_server.process_request(prompts_request)
            results["prompts"] = prompts_response
        
        return jsonify({
            "status": "success",
            "test_type": test_type,
            "results": results,
            "timestamp": time.time()
        })
    
    except Exception as e:
        logger.error(f"Error en test MCP: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }), 500

# Página de documentación MCP
@app.route('/mcp', methods=['GET'])
def mcp_documentation():
    """Página de documentación del conector MCP"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>capibara6 MCP Connector</title>
        <meta charset="UTF-8">
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                line-height: 1.6; 
                color: #333; 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
                background: #f5f5f5;
            }
            .header { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 40px; 
                border-radius: 10px; 
                text-align: center; 
                margin-bottom: 30px;
            }
            .header h1 { margin: 0; font-size: 36px; }
            .header p { margin: 10px 0 0 0; font-size: 18px; opacity: 0.9; }
            .section { 
                background: white; 
                padding: 30px; 
                border-radius: 10px; 
                margin-bottom: 20px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .section h2 { color: #667eea; margin-top: 0; }
            .endpoint { 
                background: #f8f9fa; 
                padding: 15px; 
                border-radius: 5px; 
                margin: 10px 0; 
                border-left: 4px solid #667eea;
            }
            .method { 
                font-weight: bold; 
                color: #28a745; 
                font-family: monospace; 
            }
            .url { 
                font-family: monospace; 
                background: #e9ecef; 
                padding: 2px 6px; 
                border-radius: 3px;
            }
            .code { 
                background: #2d3748; 
                color: #e2e8f0; 
                padding: 20px; 
                border-radius: 5px; 
                overflow-x: auto; 
                font-family: 'Courier New', monospace;
            }
            .feature { 
                display: inline-block; 
                background: #667eea; 
                color: white; 
                padding: 5px 15px; 
                border-radius: 20px; 
                margin: 5px; 
                font-size: 14px;
            }
            .status { 
                display: inline-block; 
                background: #28a745; 
                color: white; 
                padding: 5px 15px; 
                border-radius: 20px; 
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🦫 capibara6 MCP Connector</h1>
            <p>Conector Model Context Protocol para IA híbrida Transformer-Mamba</p>
            <div class="status">🟢 Servidor Activo</div>
        </div>
        
        <div class="section">
            <h2>📋 Descripción General</h2>
            <p>El conector MCP de capibara6 permite integrar el sistema de IA híbrido con aplicaciones que soporten el Model Context Protocol. Proporciona acceso a herramientas, recursos y prompts del modelo a través de una API estandarizada.</p>
            
            <h3>Características Principales:</h3>
            <div class="feature">Arquitectura Híbrida 70/30</div>
            <div class="feature">Google TPU v5e/v6e</div>
            <div class="feature">Google ARM Axion</div>
            <div class="feature">10M+ Tokens Contexto</div>
            <div class="feature">Compliance UE Total</div>
            <div class="feature">Multimodal</div>
            <div class="feature">Chain-of-Thought</div>
        </div>
        
        <div class="section">
            <h2>🔧 Endpoints Disponibles</h2>
            
            <div class="endpoint">
                <div class="method">GET</div>
                <div class="url">/api/mcp/status</div>
                <p>Verificar estado del servidor MCP</p>
            </div>
            
            <div class="endpoint">
                <div class="method">POST</div>
                <div class="url">/api/mcp/initialize</div>
                <p>Inicializar conexión MCP</p>
            </div>
            
            <div class="endpoint">
                <div class="method">GET/POST</div>
                <div class="url">/api/mcp/tools/list</div>
                <p>Listar herramientas disponibles</p>
            </div>
            
            <div class="endpoint">
                <div class="method">POST</div>
                <div class="url">/api/mcp/tools/call</div>
                <p>Ejecutar herramienta específica</p>
            </div>
            
            <div class="endpoint">
                <div class="method">GET/POST</div>
                <div class="url">/api/mcp/resources/list</div>
                <p>Listar recursos disponibles</p>
            </div>
            
            <div class="endpoint">
                <div class="method">POST</div>
                <div class="url">/api/mcp/resources/read</div>
                <p>Leer recurso específico</p>
            </div>
            
            <div class="endpoint">
                <div class="method">GET/POST</div>
                <div class="url">/api/mcp/prompts/list</div>
                <p>Listar prompts disponibles</p>
            </div>
            
            <div class="endpoint">
                <div class="method">POST</div>
                <div class="url">/api/mcp/prompts/get</div>
                <p>Obtener prompt específico</p>
            </div>
            
            <div class="endpoint">
                <div class="method">POST</div>
                <div class="url">/api/mcp/test</div>
                <p>Probar funcionalidad MCP</p>
            </div>
        </div>
        
        <div class="section">
            <h2>🛠️ Herramientas Disponibles</h2>
            <ul>
                <li><strong>analyze_document</strong> - Análisis de documentos extensos (10M+ tokens)</li>
                <li><strong>codebase_analysis</strong> - Análisis completo de bases de código</li>
                <li><strong>multimodal_processing</strong> - Procesamiento de texto, imagen, video y audio</li>
                <li><strong>compliance_check</strong> - Verificación GDPR, AI Act UE, CCPA</li>
                <li><strong>reasoning_chain</strong> - Chain-of-Thought reasoning hasta 12 pasos</li>
                <li><strong>performance_optimization</strong> - Optimización para TPU y ARM</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>📚 Ejemplo de Uso</h2>
            <div class="code">
# Ejemplo de llamada a herramienta
curl -X POST http://localhost:5000/api/mcp/tools/call \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "analyze_document",
    "arguments": {
      "document": "Contenido del documento...",
      "analysis_type": "compliance",
      "language": "es"
    }
  }'
            </div>
        </div>
        
        <div class="section">
            <h2>🔗 Recursos Adicionales</h2>
            <ul>
                <li><a href="https://modelcontextprotocol.io">Documentación oficial MCP</a></li>
                <li><a href="https://capibara6.com">Sitio web capibara6</a></li>
                <li><a href="https://github.com/anachroni-co/capibara6">Repositorio GitHub</a></li>
                <li><a href="https://www.anachroni.co">Anachroni s.coop</a></li>
            </ul>
        </div>
        
        <div class="section">
            <h2>📞 Soporte</h2>
            <p>Para soporte técnico o consultas sobre el conector MCP de capibara6:</p>
            <p>📧 Email: <a href="mailto:info@anachroni.co">info@anachroni.co</a></p>
            <p>🌐 Web: <a href="https://www.anachroni.co">www.anachroni.co</a></p>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    import os
    
    # Puerto para Railway (usa variable de entorno PORT)
    port = int(os.getenv('PORT', 5000))
    
    logger.info(f"Iniciando servidor Flask con MCP en puerto {port}")
    app.run(host='0.0.0.0', port=port, debug=False)