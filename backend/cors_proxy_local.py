#!/usr/bin/env python3
"""
Proxy CORS local para evitar problemas de Cross-Origin al conectar
con el servidor backend remoto de Capibara6
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import json

app = Flask(__name__)
CORS(app)

# URL del backend remoto (bounty2:5001)
BACKEND_URL = 'http://34.12.166.76:5001'

# Proxy para cualquier ruta que comience con /api/
# Proxy para rutas de health check y otros endpoints comunes
@app.route('/<endpoint>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy_general(endpoint):
    """Proxy para endpoints generales (como 'health')"""
    if request.method == 'OPTIONS':
        # Responder al preflight request
        response = Response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')
        return response
    
    # Sólo permitir ciertos endpoints generales
    allowed_endpoints = ['health', 'status', 'models', 'version']
    if endpoint not in allowed_endpoints:
        return jsonify({'error': 'Endpoint no permitido'}), 404

    # Determinar el método HTTP original
    method = request.method
    # Mapear la ruta proxy a la ruta real del backend - importante: la mayoría de endpoints están bajo /api/
    backend_path = f"/api/{endpoint}"
    target_url = f"{BACKEND_URL}{backend_path}"
    
    # Obtener los datos originales
    content_type = request.headers.get('Content-Type')
    
    # Reenviar la solicitud al backend remoto
    try:
        headers = {key: value for key, value in request.headers if key.lower() not in ['host', 'origin', 'referer']}
        headers['Host'] = '34.12.166.76:5001'
        
        if method == 'GET':
            response = requests.get(target_url, headers=headers, params=request.args)
        elif method == 'POST':
            if content_type and 'application/json' in content_type:
                data = request.get_json()
                response = requests.post(target_url, headers=headers, json=data)
            else:
                response = requests.post(target_url, headers=headers, data=request.get_data())
        elif method == 'PUT':
            if content_type and 'application/json' in content_type:
                data = request.get_json()
                response = requests.put(target_url, headers=headers, json=data)
            else:
                response = requests.put(target_url, headers=headers, data=request.get_data())
        elif method == 'DELETE':
            response = requests.delete(target_url, headers=headers)
        else:
            return jsonify({'error': f'Método {method} no soportado'}), 405

        # Devolver la respuesta del backend remoto
        return Response(
            response.content,
            status=response.status_code,
            headers=dict(response.headers),
            mimetype=response.headers.get('content-type', 'application/json')
        )
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error al conectar con el backend remoto: {str(e)}'}), 502

@app.route('/api/<path:subpath>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy_api(subpath):
    """Proxy para rutas del API"""
    if request.method == 'OPTIONS':
        # Responder al preflight request
        response = Response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')
        return response

    # Determinar el método HTTP original
    method = request.method
    # Mapear la ruta proxy a la ruta real del backend
    backend_path = f"/api/{subpath}"
    target_url = f"{BACKEND_URL}{backend_path}"
    
    # Obtener los datos originales
    content_type = request.headers.get('Content-Type')
    
    # Reenviar la solicitud al backend remoto
    try:
        headers = {key: value for key, value in request.headers if key.lower() not in ['host', 'origin', 'referer']}
        headers['Host'] = '34.12.166.76:5001'
        
        if method == 'GET':
            response = requests.get(target_url, headers=headers, params=request.args)
        elif method == 'POST':
            # Obtener directamente los datos de la solicitud sin intentar parsear JSON primero
            raw_data = request.get_data()
            if raw_data:
                # Agregar Content-Type si no está presente
                if 'Content-Type' not in headers:
                    headers['Content-Type'] = content_type or 'application/json'
                response = requests.post(target_url, headers=headers, data=raw_data)
            else:
                response = requests.post(target_url, headers=headers)

        elif method == 'PUT':
            if content_type and 'application/json' in content_type:
                data = request.get_json()
                response = requests.put(target_url, headers=headers, json=data)
            else:
                response = requests.put(target_url, headers=headers, data=request.get_data())
        elif method == 'DELETE':
            response = requests.delete(target_url, headers=headers)
        else:
            return jsonify({'error': f'Método {method} no soportado'}), 405

        # Devolver la respuesta del backend remoto
        return Response(
            response.content,
            status=response.status_code,
            headers=dict(response.headers),
            mimetype=response.headers.get('content-type', 'application/json')
        )
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error al conectar con el backend remoto: {str(e)}'}), 502

@app.route('/api/chat', methods=['POST'])
def proxy_chat():
    """Proxy específico para el endpoint de chat"""
    try:
        # Reenviar petición al backend remoto
        headers = {key: value for key, value in request.headers if key.lower() not in ['host', 'origin', 'referer']}
        headers['Host'] = '34.12.166.76:5001'
        
        # Forzar la obtención del JSON para manejar correctamente solicitudes con charset
        try:
            data = request.get_json(force=True)
            response = requests.post(
                f"{BACKEND_URL}/api/chat", 
                headers=headers, 
                json=data
            )
        except Exception as e:
            # Si falla el parsing JSON, usar el cuerpo crudo
            raw_data = request.get_data()
            if raw_data:
                headers['Content-Type'] = 'application/json'
                response = requests.post(
                    f"{BACKEND_URL}/api/chat", 
                    headers=headers, 
                    data=raw_data
                )
            else:
                return jsonify({'error': f'Error parsing JSON: {str(e)}'}), 400
        
        return Response(
            response.content,
            status=response.status_code,
            headers=dict(response.headers),
            mimetype=response.headers.get('content-type', 'application/json')
        )
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error al conectar con el backend remoto: {str(e)}'}), 502

@app.route('/api/health', methods=['GET'])
def proxy_health():
    """Proxy específico para health check"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/health")
        return Response(
            response.content,
            status=response.status_code,
            headers=dict(response.headers),
            mimetype=response.headers.get('content-type', 'application/json')
        )
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error al conectar con el backend remoto: {str(e)}'}), 502

@app.route('/', methods=['GET'])
def health_check():
    """Endpoint para verificar que el proxy está corriendo"""
    return jsonify({
        'status': 'ok',
        'service': 'capibara6-cors-proxy',
        'backend_target': BACKEND_URL,
        'timestamp': __import__('datetime').datetime.now().isoformat()
    })

if __name__ == '__main__':
    print('🚀 Iniciando Proxy CORS local para Capibara6...')
    print(f'🎯 Backend remoto: {BACKEND_URL}')
    print('🌐 Puerto local: 8001')
    print('🔗 Uso: http://localhost:8001/api/proxy -> http://34.12.166.76:5001/api/chat')
    app.run(host='0.0.0.0', port=8001, debug=False)