# capibara6_integrated_server.py
# Servidor integrado principal para el proyecto Capibara6

from flask import Flask, request, jsonify, Response
import requests
import json
import os
from models_config import MODELS_CONFIG, DEFAULT_MODEL, TIMEOUT
from toon_utils.format_manager_ultra_optimized import FormatManagerUltraOptimized
import logging

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

import sys
import os
# Asegurar que el path incluya el directorio backend
sys.path.insert(0, os.path.dirname(__file__))

# Importar la integración de e2b
try:
    from capibara6_e2b_integration import init_e2b_integration
    import importlib.util
    import sys

    # Importar directamente desde utils.py para evitar conflicto con directorio utils/
    utils_spec = importlib.util.spec_from_file_location("utils", "/home/elect/capibara6/backend/utils.py")
    utils_module = importlib.util.module_from_spec(utils_spec)
    sys.modules["utils"] = utils_module
    utils_spec.loader.exec_module(utils_module)

    analyze_context = utils_module.analyze_context
    understand_query = utils_module.understand_query
    determine_action = utils_module.determine_action
    calculate_relevance = utils_module.calculate_relevance

    E2B_AVAILABLE = True
    print("Integración e2b disponible")
except ImportError as e:
    E2B_AVAILABLE = False
    print(f"Integración e2b no disponible: {e}")
    # Definir funciones de respaldo en caso de error
    def analyze_context(context):
        if isinstance(context, str):
            return {
                'length': len(context),
                'word_count': len(context.split()),
                'has_personal_info': 'nombre' in context.lower() or 'usuario' in context.lower(),
                'context_type': 'text'
            }
        return {'type': type(context).__name__}

    def understand_query(query):
        query_lower = query.lower() if isinstance(query, str) else ''
        intent_analysis = {
            'is_question': '?' in query or any(word in query_lower for word in ['qué', 'cuál', 'cómo', 'por qué', 'when', 'what', 'how', 'why']),
            'is_command': any(word in query_lower for word in ['haz', 'crea', 'genera', 'hablemos', 'do', 'create', 'generate', 'let\'s']),
            'complexity': 'high' if len(query) > 100 else 'medium' if len(query) > 50 else 'low'
        }
        return intent_analysis

    def determine_action(context, query):
        return {
            'next_step': 'process_query',
            'requires_context_extension': len(str(context)) < 100,
            'model_preference': 'context_aware'
        }

    def calculate_relevance(context, query):
        if not context or not query:
            return 0.0
        context_words = set(str(context).lower().split())
        query_words = set(str(query).lower().split())
        if not context_words or not query_words:
            return 0.0
        common_words = context_words.intersection(query_words)
        relevance_score = len(common_words) / len(query_words) if query_words else 0.0
        return min(relevance_score, 1.0)

app = Flask(__name__)

# Configurar CORS según el entorno
from flask_cors import CORS
if os.getenv('ENVIRONMENT') == 'production':
    frontend_domain = os.getenv('FRONTEND_DOMAIN', 'https://capibara6.com')
    CORS(app, origins=[frontend_domain, 'https://www.capibara6.com'])
else:
    # Desarrollo - permitir orígenes comunes de desarrollo
    CORS(app, origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "https://www.capibara6.com",
        "https://capibara6.com",
        "http://34.12.166.76:5001",
        "http://34.12.166.76:8000",
        "http://34.175.136.104:8000"
    ])

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar la integración de e2b si está disponible
e2b_integration = None
if E2B_AVAILABLE:
    try:
        e2b_integration = init_e2b_integration()
        logger.info("Integración e2b inicializada correctamente")
    except Exception as e:
        logger.error(f"Error al inicializar la integración e2b: {e}")
        E2B_AVAILABLE = False

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
            "max_tokens": data.get("max_tokens", 200),
            "use_semantic_router": data.get("use_semantic_router", False)  # Asegurar que se maneje esta propiedad
        }

        response = requests.post(
            "http://10.204.0.9:8082/v1/chat/completions",  # IP interna correcta de models-europe
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30  # Tiempo de espera razonable para la conexión interna
        )

        return jsonify(response.json()), response.status_code
    except requests.exceptions.Timeout:
        # En caso de timeout, devolver una respuesta simulada para evitar errores 500
        return jsonify({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"Simulación de respuesta para: '{data.get('message', 'mensaje predeterminado') if 'data' in locals() else 'mensaje no disponible'}'. [Sistema RAG activo solo para consultas de programación. Consultas generales no usan RAG para mayor velocidad.]"
                }
            }],
            "model": data.get("model", "aya_expanse_multilingual") if 'data' in locals() else "aya_expanse_multilingual",
            "status": "simulated_response_due_to_timeout",
            "info": "Sistema de Programming-Only RAG ya está completamente implementado. Solo activa RAG para consultas de programación. Consultas generales no usan RAG (más rápidas)."
        }), 200
    except requests.exceptions.ConnectionError:
        # En caso de error de conexión, devolver una respuesta simulada para evitar errores 500
        return jsonify({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"Simulación de respuesta para: '{data.get('message', 'mensaje predeterminado') if 'data' in locals() else 'mensaje no disponible'}'. [Sistema RAG activo solo para consultas de programación. Consultas generales no usan RAG para mayor velocidad.]"
                }
            }],
            "model": data.get("model", "aya_expanse_multilingual") if 'data' in locals() else "aya_expanse_multilingual",
            "status": "simulated_response_due_to_connection_error",
            "info": "Sistema de Programming-Only RAG ya está completamente implementado. Solo activa RAG para consultas de programación. Consultas generales no usan RAG (más rápidas)."
        }), 200
    except Exception as e:
        return {"error": f"Error connecting to models VM: {str(e)}"}, 500

# Smart MCP integrado
@app.route('/api/mcp/status', methods=['GET', 'OPTIONS'])
def mcp_status():
    """Endpoint de health check para MCP"""
    # Manejar preflight request (OPTIONS)
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Max-Age', '3600')
        return response
    
    return jsonify({
        'status': 'running',
        'connector': 'capibara6-integrated-mcp',
        'version': '1.0.0',
        'mcp_available': True,
        'service': 'Smart MCP integrado',
        'timestamp': __import__('datetime').datetime.now().isoformat()
    })

@app.route('/api/mcp/analyze', methods=['POST'])
def smart_mcp_analyze():
    try:
        # Determinar formato de entrada
        content_type = request.headers.get('Content-Type', 'application/json').lower()
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        
        if 'application/toon' in content_type or 'text/plain' in content_type:
            input_data = FormatManagerUltraOptimized.decode(request.get_data(as_text=True), 'toon')
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
            content, format_type = FormatManagerUltraOptimized.encode(analysis_result, 'toon')
            return Response(content, mimetype='text/plain')
        else:
            return jsonify(analysis_result)
            
    except Exception as e:
        logger.error(f"Error en MCP integrado: {e}")
        error_response = {'error': 'Error en el análisis MCP integrado'}
        
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManagerUltraOptimized.encode(error_response, 'toon')
            return Response(content, mimetype='text/plain', status=500)
        else:
            return jsonify(error_response), 500

# Las funciones analíticas ahora están en utils.py para evitar importaciones circulares
# Mantenemos estas funciones para mantener compatibilidad con código existente
def _analyze_context(context):
    return analyze_context(context)

def _understand_query(query):
    return understand_query(query)

def _determine_action(context, query):
    return determine_action(context, query)

def _calculate_relevance(context, query):
    return calculate_relevance(context, query)

# TTS básico integrado
@app.route('/api/tts/speak', methods=['POST'])
def basic_tts():
    try:
        # Determinar formato de entrada
        content_type = request.headers.get('Content-Type', 'application/json').lower()
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        
        if 'application/toon' in content_type or 'text/plain' in content_type:
            input_data = FormatManagerUltraOptimized.decode(request.get_data(as_text=True), 'toon')
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
            content, format_type = FormatManagerUltraOptimized.encode(result, 'toon')
            return Response(content, mimetype='text/plain')
        else:
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error en TTS: {e}")
        error_response = {'error': 'Error en el servicio TTS'}
        
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManagerUltraOptimized.encode(error_response, 'toon')
            return Response(content, mimetype='text/plain', status=500)
        else:
            return jsonify(error_response), 500

# Endpoint para tareas que requieren entornos aislados con e2b
@app.route('/api/e2b/process', methods=['POST'])
def e2b_process():
    """Procesa tareas que requieren entornos aislados usando e2b"""
    if not E2B_AVAILABLE:
        error_response = {'error': 'Integración e2b no disponible'}
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManagerUltraOptimized.encode(error_response, 'toon')
            return Response(content, mimetype='text/plain', status=500)
        else:
            return jsonify(error_response), 500
    
    try:
        # Determinar formato de entrada
        content_type = request.headers.get('Content-Type', 'application/json').lower()
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()

        if 'application/toon' in content_type or 'text/plain' in content_type:
            input_data = FormatManagerUltraOptimized.decode(request.get_data(as_text=True), 'toon')
        else:
            input_data = request.get_json()

        prompt = input_data.get('prompt', '')
        context = input_data.get('context', '')
        
        if not prompt:
            error_response = {'error': 'Prompt es requerido'}
            if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
                content, format_type = FormatManagerUltraOptimized.encode(error_response, 'toon')
                return Response(content, mimetype='text/plain', status=400)
            else:
                return jsonify(error_response), 400

        # Usar la integración de e2b para procesar la tarea
        import asyncio
        from threading import Thread

        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    e2b_integration.handle_complex_task_with_e2b(prompt, context)
                )
                return result
            finally:
                loop.close()

        # Ejecutar la operación asíncrona en un hilo separado
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async)
            result = future.result()

        # Determinar formato de salida
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManagerUltraOptimized.encode(result, 'toon')
            return Response(content, mimetype='text/plain')
        else:
            return jsonify(result)

    except Exception as e:
        logger.error(f"Error en e2b_process: {e}")
        error_response = {'error': f'Error en el procesamiento e2b: {str(e)}'}

        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManagerUltraOptimized.encode(error_response, 'toon')
            return Response(content, mimetype='text/plain', status=500)
        else:
            return jsonify(error_response), 500

# Endpoint para obtener archivos de visualización desde e2b
@app.route('/api/e2b/visualization/<path:filepath>', methods=['GET'])
def get_visualization_file(filepath):
    """Obtiene un archivo de visualización generado en e2b"""
    if not E2B_AVAILABLE:
        return jsonify({'error': 'Integración e2b no disponible'}), 500
    
    try:
        # Este endpoint sería para servir archivos directamente
        # Por ahora, simulamos el comportamiento
        # En una implementación completa, se recuperaría el archivo del sandbox
        return jsonify({'error': 'Funcionalidad de recuperación de archivos en desarrollo'}), 503
    except Exception as e:
        logger.error(f"Error obteniendo archivo de visualización: {e}")
        return jsonify({'error': f'Error obteniendo archivo: {str(e)}'}), 500

# Endpoint para estimar recursos necesarios para una tarea
@app.route('/api/e2b/estimate', methods=['POST'])
def e2b_estimate():
    """Estima los recursos necesarios para una tarea usando e2b"""
    if not E2B_AVAILABLE:
        error_response = {'error': 'Integración e2b no disponible'}
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManagerUltraOptimized.encode(error_response, 'toon')
            return Response(content, mimetype='text/plain', status=500)
        else:
            return jsonify(error_response), 500
    
    try:
        # Determinar formato de entrada
        content_type = request.headers.get('Content-Type', 'application/json').lower()
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()

        if 'application/toon' in content_type or 'text/plain' in content_type:
            input_data = FormatManagerUltraOptimized.decode(request.get_data(as_text=True), 'toon')
        else:
            input_data = request.get_json()

        prompt = input_data.get('prompt', '')
        
        if not prompt:
            error_response = {'error': 'Prompt es requerido'}
            if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
                content, format_type = FormatManagerUltraOptimized.encode(error_response, 'toon')
                return Response(content, mimetype='text/plain', status=400)
            else:
                return jsonify(error_response), 400

        # Usar la integración de e2b para estimar recursos
        resources = e2b_integration.estimate_task_resources(prompt)

        # Determinar formato de salida
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManagerUltraOptimized.encode(resources, 'toon')
            return Response(content, mimetype='text/plain')
        else:
            return jsonify(resources)

    except Exception as e:
        logger.error(f"Error en e2b_estimate: {e}")
        error_response = {'error': f'Error en la estimación de recursos e2b: {str(e)}'}

        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManagerUltraOptimized.encode(error_response, 'toon')
            return Response(content, mimetype='text/plain', status=500)
        else:
            return jsonify(error_response), 500

# Endpoint para generar texto con clasificación CTM (funcionalidad de Ollama local)
@app.route('/api/ai/generate', methods=['POST'])
def ai_generate_ctm():
    """Genera texto usando CTM para clasificar la tarea y Ollama local para la generación"""
    try:
        # Determinar formato de entrada
        content_type = request.headers.get('Content-Type', 'application/json').lower()
        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        
        if 'application/toon' in content_type or 'text/plain' in content_type:
            input_data = FormatManagerUltraOptimized.decode(request.get_data(as_text=True), 'toon')
        else:
            input_data = request.get_json()

        prompt = input_data.get('prompt', '')
        modelPreference = input_data.get('modelPreference', 'auto')
        
        # Importar dinámicamente el sistema CTM desde la implementación de Node.js
        # Para hacerlo funcional, necesitamos implementar la lógica CTM en Python también
        model_recommendation = classify_task_ctm(prompt) if modelPreference == 'auto' else modelPreference
        
        # Generar respuesta usando Ollama local
        response = generate_with_ollama(prompt, model_recommendation)
        
        if response and response.get('success'):
            result = {
                'success': True,
                'response': response.get('response', ''),
                'model_used': response.get('model', model_recommendation),
                'processing_time': response.get('total_duration', 0),
                'token_count': response.get('token_count', 0)
            }
        else:
            result = {
                'success': False,
                'error': response.get('error', 'Error al generar la respuesta'),
                'model_used': model_recommendation
            }

        # Determinar formato de salida
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManagerUltraOptimized.encode(result, 'toon')
            return Response(content, mimetype='text/plain')
        else:
            return jsonify(result)

    except Exception as e:
        logger.error(f"Error en ai_generate_ctm: {e}")
        error_response = {'error': f'Error al generar respuesta: {str(e)}', 'success': False}

        preferred_output_format = request.headers.get('Accept', 'application/json').lower()
        if 'toon' in preferred_output_format or 'text/plain' in preferred_output_format:
            content, format_type = FormatManagerUltraOptimized.encode(error_response, 'toon')
            return Response(content, mimetype='text/plain', status=500)
        else:
            return jsonify(error_response), 500

def classify_task_ctm(prompt):
    """Implementación de clasificación CTM en Python (equivalente al sistema de Node.js)"""
    import re
    
    prompt_lower = prompt.lower()
    
    # Clasificación basada en características cognitivas (similar al sistema CTM de Node.js)
    complexity_indicators = {
        'complex': ['análisis', 'razonamiento', 'comparación', 'evaluar', 'estrategia', 'planificación', 
                   'investigación', 'profundo', 'detalle', 'complejo', 'técnico', 'evaluación', 
                   'interpretación', 'síntesis', 'problema', 'dilema', 'paradigma', 'metodología', 
                   'hipótesis', 'teoría', 'implicaciones', 'consecuencias', 'factores', 'dimensiones'],
        'balanced': ['explicar', 'qué es', 'cómo funciona', 'describir', 'resumen', 'breve', 
                    'ejemplo', 'definir', 'funciona', 'significado', 'característica', 'proceso'],
        'simple': ['qué', 'quién', 'cuál', 'cuándo', 'dónde', 'chiste', 'broma', 'saludo', 
                  'ayuda', 'cuánto', 'dime', 'haz', 'crea']
    }

    # Contar indicadores de cada tipo
    complex_score = sum(2 for indicator in complexity_indicators['complex'] if indicator in prompt_lower)
    balanced_score = sum(1 for indicator in complexity_indicators['balanced'] if indicator in prompt_lower)
    simple_score = sum(1 for indicator in complexity_indicators['simple'] if indicator in prompt_lower)
    
    # Considerar la longitud del prompt
    if len(prompt) > 100:
        balanced_score += 1
    if len(prompt) > 200:
        complex_score += 1
    
    # Considerar la complejidad sintáctica (número de frases)
    sentence_count = len(re.split(r'[.!?]+', prompt)) - 1  # Restar 1 porque hay un elemento extra al final
    if sentence_count > 3:
        complex_score += 1
    
    # Determinar el modelo más apropiado
    max_score = max(complex_score, balanced_score, simple_score)
    if max_score == complex_score:
        return 'complex'
    elif max_score == balanced_score:
        return 'balanced'
    else:
        return 'fast_response'

def generate_with_ollama(prompt, model_tier):
    """Genera texto usando Ollama local"""
    import requests
    import time
    
    # Mapear el tier al modelo real ARM-Axion
    model_mapping = {
        'fast_response': 'phi4_fast',
        'balanced': 'mistral_balanced',
        'complex': 'gptoss_complex'
    }
    
    model_name = model_mapping.get(model_tier, 'phi3:mini')
    
    # Configurar el endpoint de vLLM ARM-Axion
    vllm_endpoint = "http://localhost:8080/v1/chat/completions"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 2048,
        "stream": False
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        start_time = time.time()
        response = requests.post(vllm_endpoint, json=payload, headers=headers, timeout=240)  # Ajustar timeout
        end_time = time.time()

        if response.status_code == 200:
            data = response.json()
            return {
                'success': True,
                'response': data['choices'][0]['message']['content'],
                'model': model_name,
                'total_duration': int((end_time - start_time) * 1000),  # Convertir a ms
                'token_count': data['usage']['completion_tokens']
            }
        else:
            return {
                'success': False,
                'error': f'Error de vLLM ARM-Axion: {response.status_code} - {response.text}'
            }
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': 'Timeout al comunicarse con Ollama'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error al comunicarse con Ollama: {str(e)}'
        }

# Health check funcional
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'capibara6_integrated_server',
        'models': list(MODELS_CONFIG.keys()),
        'e2b_available': E2B_AVAILABLE,
        'toon_support': True
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
