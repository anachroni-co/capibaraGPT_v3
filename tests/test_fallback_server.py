#!/usr/bin/env python3
"""
Versión actualizada del servidor con fallback para pruebas
"""

from flask import Flask, request, jsonify, Response
import requests
import json
import os
from datetime import datetime

app = Flask(__name__)

# Endpoint para conectar frontend a servidor de modelos en VM models-europe
@app.route('/api/chat', methods=['POST'])
def proxy_to_models_europe():
    """Proxy endpoint para conectar frontend a servidor de modelos"""
    try:
        data = request.get_json()

        payload = {
            "model": data.get("model", "aya_expanse_multilingual"),
            "messages": [{"role": "user", "content": data.get("message", "")}],
            "temperature": data.get("temperature", 0.7),
            "max_tokens": data.get("max_tokens", 200)
        }

        # Intentar conectar al servidor remoto
        response = requests.post(
            "http://34.175.48.2:8082/v1/chat/completions",  # ESTA VM
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30  # Reducido para pruebas
        )

        return jsonify(response.json()), response.status_code
    except requests.exceptions.Timeout:
        # En caso de timeout, devolver una respuesta simulada
        return jsonify({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"Respuesta simulada para: {data.get('message', 'mensaje predeterminado') if 'data' in locals() else 'mensaje no disponible'}"
                }
            }],
            "model": data.get("model", "aya_expanse_multilingual") if 'data' in locals() else "aya_expanse_multilingual",
            "status": "simulated_response"
        }), 200
    except requests.exceptions.ConnectionError:
        # En caso de error de conexión, devolver una respuesta simulada
        return jsonify({
            "choices": [{
                "message": {
                    "role": "assistant", 
                    "content": f"Respuesta simulada para: {data.get('message', 'mensaje predeterminado') if 'data' in locals() else 'mensaje no disponible'}"
                }
            }],
            "model": data.get("model", "aya_expanse_multilingual") if 'data' in locals() else "aya_expanse_multilingual",
            "status": "simulated_response_due_to_connection_error"
        }), 200
    except Exception as e:
        return {"error": f"Error connecting to models VM: {str(e)}"}, 500

# Health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'chat-proxy-service',
        'models_endpoint': 'http://34.175.48.2:8082/v1/chat/completions'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9001))  # Puerto diferente para evitar conflictos
    app.run(host='0.0.0.0', port=port, debug=False)