#!/usr/bin/env python3
"""
Servidor proxy local para resolver problemas de CORS entre frontend y backend Capibara6
"""

from flask import Flask, request, jsonify, Response
import requests
import json

app = Flask(__name__)

# URL del backend real de Capibara6
BACKEND_URL = 'http://34.12.166.76:5000'

@app.route('/api/mcp/tools/call-proxy', methods=['POST'])
def mcp_tools_call_proxy():
    """Endpoint proxy para herramientas MCP que resuelve problemas CORS"""
    try:
        # Obtener datos de la solicitud original
        data = request.get_json()
        
        # Construir URL del endpoint real
        target_url = f"{BACKEND_URL}/api/mcp/tools/call"
        
        # Hacer la solicitud al backend real
        response = requests.post(
            target_url,
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        
        # Devolver la respuesta del backend real
        return Response(
            response.content,
            status=response.status_code,
            mimetype='application/json'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/proxy', methods=['POST'])
def generic_proxy():
    """Endpoint proxy genérico"""
    try:
        data = request.get_json()
        target_url = data.get('target_url')
        method = data.get('method', 'GET').upper()
        headers = data.get('headers', {})
        body = data.get('body', {})
        
        if not target_url:
            return jsonify({'error': 'Target URL is required'}), 400
        
        # Hacer la solicitud al destino real
        if method == 'GET':
            response = requests.get(target_url, headers=headers)
        elif method == 'POST':
            response = requests.post(target_url, json=body, headers=headers)
        elif method == 'PUT':
            response = requests.put(target_url, json=body, headers=headers)
        elif method == 'DELETE':
            response = requests.delete(target_url, headers=headers)
        else:
            return jsonify({'error': f'Method {method} not supported'}), 400
        
        # Devolver la respuesta
        return Response(
            response.content,
            status=response.status_code,
            mimetype='application/json'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mcp-proxy', methods=['POST'])
def mcp_proxy():
    """Endpoint específico para proxy MCP"""
    try:
        data = request.get_json()
        target = data.get('target', '')
        method = data.get('method', 'GET')
        body = data.get('body', {})
        
        # Si no hay target, usar el backend base
        if not target:
            target = f"{BACKEND_URL}/api/mcp"
        
        # Hacer la solicitud al destino MCP real
        if method.upper() == 'GET':
            response = requests.get(target)
        elif method.upper() == 'POST':
            response = requests.post(target, json=body)
        
        return Response(
            response.content,
            status=response.status_code,
            mimetype='application/json'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check del proxy
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'proxy_ok',
        'backend_target': BACKEND_URL,
        'timestamp': json.dumps(str(__import__('datetime').datetime.now()))
    })

if __name__ == '__main__':
    print("🚀 Servidor Proxy Local para Capibara6 iniciado")
    print(f"📡 Backend objetivo: {BACKEND_URL}")
    print("🔌 Endpoints disponibles:")
    print("   POST /api/mcp/tools/call-proxy")
    print("   POST /api/proxy")
    print("   POST /api/mcp-proxy")
    print("   GET  /health")
    print("🔧 Escuchando en puerto 8001 (localhost:8001)")
    
    app.run(host='localhost', port=8001, debug=False)