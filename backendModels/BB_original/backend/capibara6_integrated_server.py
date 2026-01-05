# capibara6_integrated_server.py
# Servidor integrado principal para el proyecto Capibara6

from flask import Flask, request, jsonify, Response
import requests
import json
import os
from models_config import MODEL_CONFIGS, DEFAULT_MODEL, TIMEOUT
from toon_utils.format_manager import FormatManager
import logging

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Proxy para GPT-OSS-20B
@app.route('/api/chat', methods=['POST'])
def proxy_gpt_oss_20b():
    try:
        # Determinar formato de entrada
        content_type = request.headers.get('Content-Type', 'application/json').lower()
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        
        if 'application/toon' in content_type or 'text/plain' in content_type:
            input_data = FormatManager.decode(request.get_data(as_text=True), 'toon')
        else:
            input_data = request.get_json()
        
        model_config = MODEL_CONFIGS.get('gpt_oss_20b')
        
        if not model_config:
            error_response = {'error': 'Modelo GPT-OSS-20B no configurado'}
            
            if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
                content, format_type = FormatManager.encode(error_response, 'toon')
                return Response(content, mimetype='text/plain', status=404)
            else:
                return jsonify(error_response), 404
        
        # Reenviar la solicitud al servidor remoto
        response = requests.post(
            model_config['endpoint'],
            json=input_data,
            timeout=TIMEOUT/1000  # Convertir de ms a segundos
        )
        
        # Determinar el formato de la respuesta del modelo
        if response.headers.get('Content-Type', '').startswith('application/json'):
            model_response = response.json()
        else:
            model_response = response.text  # Si no es JSON, manejar como texto
        
        # Determinar formato de salida para el cliente
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManager.encode(model_response, 'toon')
            return Response(
                content,
                status=response.status_code,
                mimetype='text/plain'
            )
        else:
            # Devolver directamente la respuesta del modelo
            return Response(
                response.content,
                status=response.status_code,
                content_type='application/json'
            )
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al conectar con GPT-OSS-20B: {e}")
        error_response = {'error': 'Error al conectar con el modelo GPT-OSS-20B'}
        
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManager.encode(error_response, 'toon')
            return Response(content, mimetype='text/plain', status=500)
        else:
            return jsonify(error_response), 500
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        error_response = {'error': 'Error interno del servidor'}
        
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManager.encode(error_response, 'toon')
            return Response(content, mimetype='text/plain', status=500)
        else:
            return jsonify(error_response), 500

# Smart MCP integrado
@app.route('/api/mcp/analyze', methods=['POST'])
def smart_mcp_analyze():
    try:
        # Determinar formato de entrada
        content_type = request.headers.get('Content-Type', 'application/json').lower()
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        
        if 'application/toon' in content_type or 'text/plain' in content_type:
            input_data = FormatManager.decode(request.get_data(as_text=True), 'toon')
        else:
            input_data = request.get_json()
        
        # Análisis inteligente del contexto (misma lógica que en smart_mcp_server)
        context = input_data.get('context', '')
        query = input_data.get('query', '')
        
        analysis_result = {
            'status': 'completed',
            'context_analysis': _analyze_context(context),
            'query_understanding': _understand_query(query),
            'recommended_action': _determine_action(context, query),
            'context_relevance': _calculate_relevance(context, query),
            'token_optimization_used': True,  # Indicar soporte de TOON
            'source': 'integrated_server'
        }
        
        # Determinar formato de salida
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManager.encode(analysis_result, 'toon')
            return Response(content, mimetype='text/plain')
        else:
            return jsonify(analysis_result)
            
    except Exception as e:
        logger.error(f"Error en MCP integrado: {e}")
        error_response = {'error': 'Error en el análisis MCP integrado'}
        
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManager.encode(error_response, 'toon')
            return Response(content, mimetype='text/plain', status=500)
        else:
            return jsonify(error_response), 500

def _analyze_context(context):
    """Analiza el contexto proporcionado (función auxiliar)"""
    if isinstance(context, str):
        return {
            'length': len(context),
            'word_count': len(context.split()),
            'has_personal_info': 'nombre' in context.lower() or 'usuario' in context.lower(),
            'context_type': 'text'
        }
    return {'type': type(context).__name__}

def _understand_query(query):
    """Entiende la intención de la consulta (función auxiliar)"""
    query_lower = query.lower() if isinstance(query, str) else ''
    
    intent_analysis = {
        'is_question': '?' in query or any(word in query_lower for word in ['qué', 'cuál', 'cómo', 'por qué', 'when', 'what', 'how', 'why']),
        'is_command': any(word in query_lower for word in ['haz', 'crea', 'genera', 'hablemos', 'do', 'create', 'generate', 'let\'s']),
        'complexity': 'high' if len(query) > 100 else 'medium' if len(query) > 50 else 'low'
    }
    
    return intent_analysis

def _determine_action(context, query):
    """Determina la acción recomendada basada en contexto y consulta (función auxiliar)"""
    return {
        'next_step': 'process_query',
        'requires_context_extension': len(str(context)) < 100,
        'model_preference': 'context_aware'
    }

def _calculate_relevance(context, query):
    """Calcula la relevancia entre contexto y consulta (función auxiliar)"""
    if not context or not query:
        return 0.0
    
    context_words = set(str(context).lower().split())
    query_words = set(str(query).lower().split())
    
    if not context_words or not query_words:
        return 0.0
    
    common_words = context_words.intersection(query_words)
    relevance_score = len(common_words) / len(query_words) if query_words else 0.0
    
    return min(relevance_score, 1.0)  # Asegurar que esté entre 0 y 1

# TTS básico integrado
@app.route('/api/tts/speak', methods=['POST'])
def basic_tts():
    try:
        # Determinar formato de entrada
        content_type = request.headers.get('Content-Type', 'application/json').lower()
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        
        if 'application/toon' in content_type or 'text/plain' in content_type:
            input_data = FormatManager.decode(request.get_data(as_text=True), 'toon')
        else:
            input_data = request.get_json()
        
        text = input_data.get('text', '')
        
        # Simulación de respuesta TTS
        result = {
            'status': 'success',
            'message': f'Texto procesado para TTS: {text[:50]}...',
            'token_optimization_used': True
        }
        
        # Determinar formato de salida
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManager.encode(result, 'toon')
            return Response(content, mimetype='text/plain')
        else:
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error en TTS: {e}")
        error_response = {'error': 'Error en el servicio TTS'}
        
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManager.encode(error_response, 'toon')
            return Response(content, mimetype='text/plain', status=500)
        else:
            return jsonify(error_response), 500

# Health check funcional
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'capibara6_integrated_server',
        'models': list(MODEL_CONFIGS.keys()),
        'toon_support': True
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)