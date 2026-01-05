# smart_mcp_server.py
# Servidor MCP inteligente standalone para el proyecto Capibara6

from flask import Flask, request, jsonify, Response
import json
import logging
import os
from toon_utils.format_manager import FormatManager

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # Análisis inteligente del contexto
        context = input_data.get('context', '')
        query = input_data.get('query', '')
        
        # Lógica avanzada de MCP (Model Context Propagation)
        analysis_result = {
            'status': 'completed',
            'context_analysis': analyze_context(context),
            'query_understanding': understand_query(query),
            'recommended_action': determine_action(context, query),
            'context_relevance': calculate_relevance(context, query),
            'token_optimization_used': True  # Indicar que este servidor soporta TOON
        }
        
        # Determinar formato de salida
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManager.encode(analysis_result, 'toon')
            return Response(content, mimetype='text/plain')
        else:
            return jsonify(analysis_result)
            
    except Exception as e:
        logger.error(f"Error en Smart MCP: {e}")
        error_response = {'error': 'Error en el servicio Smart MCP'}
        
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManager.encode(error_response, 'toon')
            return Response(content, mimetype='text/plain', status=500)
        else:
            return jsonify(error_response), 500

def analyze_context(context):
    """Analiza el contexto proporcionado"""
    if isinstance(context, str):
        return {
            'length': len(context),
            'word_count': len(context.split()),
            'has_personal_info': 'nombre' in context.lower() or 'usuario' in context.lower(),
            'context_type': 'text'
        }
    return {'type': type(context).__name__}

def understand_query(query):
    """Entiende la intención de la consulta"""
    query_lower = query.lower() if isinstance(query, str) else ''
    
    intent_analysis = {
        'is_question': '?' in query or any(word in query_lower for word in ['qué', 'cuál', 'cómo', 'por qué', 'when', 'what', 'how', 'why']),
        'is_command': any(word in query_lower for word in ['haz', 'crea', 'genera', 'hablemos', 'do', 'create', 'generate', 'let\'s']),
        'complexity': 'high' if len(query) > 100 else 'medium' if len(query) > 50 else 'low'
    }
    
    return intent_analysis

def determine_action(context, query):
    """Determina la acción recomendada basada en contexto y consulta"""
    return {
        'next_step': 'process_query',
        'requires_context_extension': len(str(context)) < 100,
        'model_preference': 'context_aware'
    }

def calculate_relevance(context, query):
    """Calcula la relevancia entre contexto y consulta"""
    if not context or not query:
        return 0.0
    
    context_words = set(str(context).lower().split())
    query_words = set(str(query).lower().split())
    
    if not context_words or not query_words:
        return 0.0
    
    common_words = context_words.intersection(query_words)
    relevance_score = len(common_words) / len(query_words) if query_words else 0.0
    
    return min(relevance_score, 1.0)  # Asegurar que esté entre 0 y 1

@app.route('/api/mcp/extend', methods=['POST'])
def extend_context():
    try:
        # Determinar formato de entrada
        content_type = request.headers.get('Content-Type', 'application/json').lower()
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        
        if 'application/toon' in content_type or 'text/plain' in content_type:
            input_data = FormatManager.decode(request.get_data(as_text=True), 'toon')
        else:
            input_data = request.get_json()
        
        extended_context = {
            'original_context': input_data.get('context'),
            'additional_info': input_data.get('additional_info', []),
            'extended_by': 'smart_mcp',
            'timestamp': 'generated_at_server',
            'token_optimization_used': True
        }
        
        # Determinar formato de salida
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManager.encode(extended_context, 'toon')
            return Response(content, mimetype='text/plain')
        else:
            return jsonify(extended_context)
            
    except Exception as e:
        logger.error(f"Error extendiendo contexto: {e}")
        error_response = {'error': 'Error extendiendo el contexto'}
        
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManager.encode(error_response, 'toon')
            return Response(content, mimetype='text/plain', status=500)
        else:
            return jsonify(error_response), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'smart_mcp_server',
        'capabilities': ['context_analysis', 'query_understanding', 'action_determination'],
        'toon_support': True
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5010))  # Puerto 5010 como se menciona en el análisis
    app.run(host='0.0.0.0', port=port, debug=False)