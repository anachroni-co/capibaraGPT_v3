#!/usr/bin/env python3

import os
import sys

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

# Asegurar que el path incluya el directorio backend
sys.path.insert(0, '/home/elect/capibara6/backend')

print("E2B_API_KEY desde entorno:", os.getenv("E2B_API_KEY", "NO ESTABLECIDA"))

# Importar la integración de e2b
try:
    from capibara6_e2b_integration import init_e2b_integration
    import importlib.util

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

# Inicializar la integración de e2b si está disponible
e2b_integration = None
if E2B_AVAILABLE:
    try:
        e2b_integration = init_e2b_integration()
        print("Integración e2b inicializada correctamente")
    except Exception as e:
        print(f"Error al inicializar la integración e2b: {e}")
        import traceback
        traceback.print_exc()
        E2B_AVAILABLE = False

print(f"Estado final de E2B_AVAILABLE: {E2B_AVAILABLE}")
print(f"e2b_integration objeto: {e2b_integration is not None}")