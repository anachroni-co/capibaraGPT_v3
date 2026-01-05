#!/usr/bin/env python3
"""
Proxy CORS local simplificado y robusto para evitar problemas de Cross-Origin 
al conectar con el servidor backend remoto de Capibara6
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import json

app = Flask(__name__)
CORS(app)

# URL del backend remoto (bounty2:5001)
BACKEND_URL = 'http://34.12.166.76:5001'

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def proxy_chat():
    """Proxy específico para el endpoint de chat - versión robusta"""
    # flask-cors maneja automáticamente las peticiones OPTIONS (preflight)
    # No necesitamos manejarlas manualmente
    
    try:
        # Obtener el cuerpo de la solicitud como bytes
        raw_data = request.get_data()
        
        # Obtener todos los headers relevantes, manteniendo cuidadosamente solo los necesarios
        headers = {}
        for key, value in request.headers:
            # Mantener headers importantes para el procesamiento
            if key.lower() not in ['host', 'origin', 'referer', 'content-length', 'connection']:
                headers[key] = value
        
        # Asegurarse de que Host esté configurado correctamente
        headers['Host'] = '34.12.166.76:5001'
        
        # Añadir header de conexión si no está presente
        if 'Connection' not in headers:
            headers['Connection'] = 'close'
        
        # Hacer la solicitud al backend remoto
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            headers=headers,
            data=raw_data
        )
        
        # Devolver la respuesta del backend remoto
        response_headers = dict(response.headers)
        
        # Eliminar headers conflictivos y CORS del backend (el proxy maneja CORS)
        headers_to_remove = [
            'Transfer-Encoding', 
            'Access-Control-Allow-Origin',
            'Access-Control-Allow-Methods',
            'Access-Control-Allow-Headers',
            'Access-Control-Allow-Credentials',
            'Access-Control-Max-Age'
        ]
        for header in headers_to_remove:
            response_headers.pop(header, None)
        
        # flask-cors del proxy añadirá automáticamente los headers CORS correctos
        # No necesitamos añadirlos manualmente aquí
            
        return Response(
            response.content,
            status=response.status_code,
            headers=response_headers
        )
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error de conexión con el backend remoto: {str(e)}'}), 502
    except Exception as e:
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def proxy_health():
    """Proxy para health check - flask-cors maneja automáticamente OPTIONS"""
    # Intentar múltiples endpoints del backend
    endpoints_to_try = [
        f"{BACKEND_URL}/api/health",
        f"{BACKEND_URL}/health",
        f"{BACKEND_URL}/"
    ]
    
    for endpoint_url in endpoints_to_try:
        try:
            response = requests.get(endpoint_url, timeout=5)
            
            # Eliminar headers CORS del backend (el proxy maneja CORS)
            response_headers = dict(response.headers)
            cors_headers = [
                'Access-Control-Allow-Origin',
                'Access-Control-Allow-Methods',
                'Access-Control-Allow-Headers',
                'Access-Control-Allow-Credentials',
                'Access-Control-Max-Age'
            ]
            for header in cors_headers:
                response_headers.pop(header, None)
            
            # flask-cors del proxy añadirá automáticamente los headers CORS correctos
            proxy_response = Response(
                response.content,
                status=response.status_code,
                headers=response_headers
            )
            return proxy_response
        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout conectando a {endpoint_url}")
            continue
        except requests.exceptions.ConnectionError as e:
            print(f"⚠️ Error de conexión a {endpoint_url}: {e}")
            continue
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error en {endpoint_url}: {e}")
            continue
    
    # Si todos los endpoints fallan, devolver error (flask-cors añadirá headers CORS automáticamente)
    error_response = jsonify({
        'error': 'No se pudo conectar con el backend remoto',
        'backend_url': BACKEND_URL,
        'message': 'El servidor backend no está respondiendo. Verifica que esté corriendo en el puerto correcto.',
        'tried_endpoints': endpoints_to_try
    })
    # flask-cors maneja los headers CORS automáticamente, no necesitamos añadirlos manualmente
    return error_response, 502

@app.route('/<path:subpath>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy_all(subpath):
    """Proxy general para todas las demás rutas - flask-cors maneja automáticamente OPTIONS"""
    try:
        # Determinar el método HTTP
        method = request.method
        
        # Obtener headers relevantes
        headers = {}
        for key, value in request.headers:
            if key.lower() not in ['host', 'origin', 'referer', 'content-length']:
                headers[key] = value
        
        # Asegurarse de que Host esté configurado correctamente
        headers['Host'] = '34.12.166.76:5001'
        
        # Obtener el cuerpo de la solicitud si existe
        raw_data = request.get_data()
        
        # Determinar la URL de destino según el subpath
        if subpath.startswith('api/'):
            target_url = f"{BACKEND_URL}/{subpath}"
        else:
            target_url = f"{BACKEND_URL}/api/{subpath}"
        
        # Hacer la solicitud al backend remoto
        if method == 'GET':
            response = requests.get(target_url, headers=headers, params=request.args)
        elif method == 'POST':
            response = requests.post(target_url, headers=headers, data=raw_data)
        elif method == 'PUT':
            response = requests.put(target_url, headers=headers, data=raw_data)
        elif method == 'DELETE':
            response = requests.delete(target_url, headers=headers)
        else:
            return jsonify({'error': f'Método {method} no soportado'}), 405
        
        # Devolver la respuesta del backend remoto
        response_headers = dict(response.headers)
        
        # Eliminar headers CORS del backend (el proxy maneja CORS)
        cors_headers = [
            'Access-Control-Allow-Origin',
            'Access-Control-Allow-Methods',
            'Access-Control-Allow-Headers',
            'Access-Control-Allow-Credentials',
            'Access-Control-Max-Age'
        ]
        for header in cors_headers:
            response_headers.pop(header, None)
        
        # flask-cors del proxy añadirá automáticamente los headers CORS correctos
        return Response(
            response.content,
            status=response.status_code,
            headers=response_headers
        )
        
    except requests.exceptions.RequestException as e:
        # flask-cors maneja los headers CORS automáticamente
        error_response = jsonify({'error': f'Error de conexión con el backend remoto: {str(e)}'})
        return error_response, 502
    except Exception as e:
        # flask-cors maneja los headers CORS automáticamente
        error_response = jsonify({'error': f'Error interno: {str(e)}'})
        return error_response, 500

@app.route('/', methods=['GET'])
def health_check():
    """Endpoint para verificar que el proxy está corriendo"""
    return jsonify({
        'status': 'ok',
        'service': 'capibara6-cors-proxy-simplified',
        'backend_target': BACKEND_URL,
        'timestamp': __import__('datetime').datetime.now().isoformat()
    })

if __name__ == '__main__':
    print('🚀 Iniciando Proxy CORS local simplificado para Capibara6...')
    print(f'🎯 Backend remoto: {BACKEND_URL}')
    print('🌐 Puerto local: 8001')
    print('🔗 Endpoints: /api/chat, /api/health, y otros /api/*')
    app.run(host='0.0.0.0', port=8001, debug=False)