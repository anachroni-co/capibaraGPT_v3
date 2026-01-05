# consensus_server.py
# Servidor de consenso para el proyecto Capibara6

from flask import Flask, request, jsonify, Response
import requests
import json
import logging
from models_config import MODEL_CONFIGS, TIMEOUT
from toon_utils.format_manager import FormatManager

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/api/consensus/query', methods=['POST'])
def consensus_query():
    try:
        # Determinar formato de entrada
        content_type = request.headers.get('Content-Type', 'application/json').lower()
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        
        if 'application/toon' in content_type or 'text/plain' in content_type:
            input_data = FormatManager.decode(request.get_data(as_text=True), 'toon')
        else:
            input_data = request.get_json()
        
        user_prompt = input_data.get('prompt', '')
        
        # Recopilar respuestas de múltiples modelos
        responses = {}
        errors = {}
        
        for model_name, config in MODEL_CONFIGS.items():
            try:
                # Determinar el mejor formato para enviar al modelo
                _, request_format = FormatManager.encode(input_data)
                
                if request_format == 'toon':
                    model_request_data = FormatManager.encode(input_data)[0]
                    headers = {'Content-Type': 'text/plain'}
                else:
                    model_request_data = input_data
                    headers = {'Content-Type': 'application/json'}
                
                response = requests.post(
                    config['endpoint'],
                    json={'prompt': user_prompt, **input_data},
                    timeout=TIMEOUT/1000  # Convertir de ms a segundos
                )
                
                if response.status_code == 200:
                    responses[model_name] = response.json()
                else:
                    errors[model_name] = f"Status {response.status_code}: {response.text}"
            except Exception as e:
                errors[model_name] = str(e)
                logger.error(f"Error al consultar modelo {model_name}: {e}")
        
        # Preparar el resultado
        result = {
            'status': 'success',
            'responses': responses,
            'errors': errors,
            'consensus_applied': True,
            'primary_choice': get_primary_choice(responses)
        }
        
        # Determinar formato de salida
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManager.encode(result, 'toon')
            return Response(content, mimetype='text/plain')
        else:
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error en servidor de consenso: {e}")
        error_response = {'error': 'Error en el servidor de consenso'}
        
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManager.encode(error_response, 'toon')
            return Response(content, mimetype='text/plain', status=500)
        else:
            return jsonify(error_response), 500

def get_primary_choice(responses):
    """Lógica simple para determinar la mejor respuesta"""
    if not responses:
        return None
    
    # Para simplificar, retornamos la primera respuesta exitosa
    # En una implementación real, se aplicarían criterios más sofisticados
    for model_name, response in responses.items():
        if 'response' in response or 'content' in response:
            return {'model': model_name, 'response': response}
    
    return None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'consensus_server',
        'models_available': list(MODEL_CONFIGS.keys()),
        'toon_support': True
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5003))  # Cambiado de 5002 a 5003 para evitar conflicto
    app.run(host='0.0.0.0', port=port, debug=False)