"""
Funciones de utilidad compartidas entre diferentes componentes de Capibara6
"""

def analyze_context(context):
    """Analiza el contexto proporcionado (función auxiliar)"""
    if isinstance(context, str):
        return {
            'length': len(context),
            'word_count': len(context.split()),
            'has_personal_info': 'nombre' in context.lower() or 'usuario' in context.lower(),
            'context_type': 'text'
        }
    return {'type': type(context).__name__}

def understand_query(query):
    """Entiende la intención de la consulta (función auxiliar)"""
    query_lower = query.lower() if isinstance(query, str) else ''

    intent_analysis = {
        'is_question': '?' in query or any(word in query_lower for word in ['qué', 'cuál', 'cómo', 'por qué', 'when', 'what', 'how', 'why']),
        'is_command': any(word in query_lower for word in ['haz', 'crea', 'genera', 'hablemos', 'do', 'create', 'generate', 'let\'s']),
        'complexity': 'high' if len(query) > 100 else 'medium' if len(query) > 50 else 'low'
    }

    return intent_analysis

def determine_action(context, query):
    """Determina la acción recomendada basada en contexto y consulta (función auxiliar)"""
    return {
        'next_step': 'process_query',
        'requires_context_extension': len(str(context)) < 100,
        'model_preference': 'context_aware'
    }

def calculate_relevance(context, query):
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