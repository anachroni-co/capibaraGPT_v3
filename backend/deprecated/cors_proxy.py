#!/usr/bin/env python3
"""
Proxy CORS para conectar con GPT-OSS-20B
Soluciona el problema de CORS permitiendo peticiones desde localhost:8000
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app, origins=['http://localhost:8000', 'http://127.0.0.1:8000'])

# URL de la VM GPT-OSS-20B
GPTOSS_API_URL = 'http://34.175.215.109/api/chat'
GPTOSS_HEALTH_URL = 'http://34.175.215.109/health'

@app.route('/health', methods=['GET'])
def health():
    """Health check del proxy"""
    try:
        # Verificar conexión con la VM
        response = requests.get(GPTOSS_HEALTH_URL, timeout=5)
        vm_status = response.json() if response.ok else {'error': 'VM no disponible'}
        
        return jsonify({
            'status': 'ok',
            'proxy': 'CORS Proxy funcionando',
            'vm_status': vm_status,
            'timestamp': datetime.now().isoformat()
        })
    except requests.exceptions.RequestException as e:
        return jsonify({
            'status': 'error',
            'proxy': 'CORS Proxy funcionando',
            'vm_status': {'error': f'No se puede conectar con la VM: {str(e)}'},
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat_proxy():
    """Proxy para las peticiones de chat"""
    
    # Manejar preflight OPTIONS
    if request.method == 'OPTIONS':
        response = Response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    try:
        # Obtener datos de la petición
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se recibieron datos JSON'}), 400
        
        print(f"📨 Proxy recibió petición: {data.get('message', 'Sin mensaje')[:50]}...")
        
        # Reenviar petición a la VM
        response = requests.post(
            GPTOSS_API_URL,
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=300  # 5 minutos
        )
        
        if response.ok:
            result = response.json()
            print(f"✅ Proxy respondió correctamente: {result.get('response', 'Sin respuesta')[:50]}...")
            return jsonify(result)
        else:
            print(f"❌ Error de la VM: {response.status_code}")
            return jsonify({
                'error': f'Error de la VM: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except requests.exceptions.Timeout:
        print("⏰ Timeout en la petición a la VM")
        return jsonify({'error': 'Timeout: La VM tardó demasiado en responder'}), 504
    except requests.exceptions.RequestException as e:
        print(f"❌ Error de conexión con la VM: {e}")
        return jsonify({'error': f'Error de conexión: {str(e)}'}), 502
    except Exception as e:
        print(f"❌ Error interno del proxy: {e}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/api/models', methods=['GET'])
def models():
    """Información de modelos disponibles"""
    return jsonify({
        'models': [{
            'id': 'gpt-oss-20b',
            'name': 'GPT-OSS-20B',
            'description': 'Modelo de 20B parámetros ejecutándose en Google Cloud VM'
        }]
    })

if __name__ == '__main__':
    print('🚀 Iniciando CORS Proxy para GPT-OSS-20B...')
    print(f'📡 VM URL: {GPTOSS_API_URL}')
    print('🌐 Puerto: 5000')
    print('🔧 CORS habilitado para localhost:8000')
    
    # Verificar conexión con la VM al inicio
    try:
        response = requests.get(GPTOSS_HEALTH_URL, timeout=5)
        if response.ok:
            print('✅ Conexión con VM GPT-OSS-20B: OK')
        else:
            print('⚠️ Advertencia: VM no responde correctamente')
    except requests.exceptions.RequestException as e:
        print(f'⚠️ Advertencia: No se puede conectar con la VM: {e}')
    
    app.run(host='0.0.0.0', port=5000, debug=False)
