"""
Smart MCP Server - Versión Completa y Funcional
Con contexto verificado sobre Capibara6
Puerto: 5010
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import re

app = Flask(__name__)
CORS(app, origins='*')

# Base de conocimiento verificada
KNOWLEDGE_BASE = {
    "identity": {
        "name": "Capibara6",
        "creator": "Anachroni s.coop",
        "status": "Producción",
        "type": "Modelo de lenguaje basado en Gemma 3-12B",
        "hardware": "Google Cloud VM en europe-southwest1-b",
        "website": "https://capibara6.com",
        "email": "info@anachroni.co"
    },
    "current_info": {
        "date": "11 de octubre de 2025",
        "day": "sábado"
    }
}

# Patrones para detectar cuándo agregar contexto
CONTEXT_TRIGGERS = {
    "identity": {
        "patterns": [
            r'\b(quién|quien|que|qué)\s+(eres|soy|es)\b',
            r'\b(cómo|como)\s+(te\s+llamas|se\s+llama)\b',
            r'\b(tu|tú)\s+(nombre|identidad)\b',
            r'\bcapibara\b',
            r'\bcreo|creador|desarrollador\b',
            r'\bquién\s+te\s+(creó|creo|hizo|desarrollo)\b',
            r'\b(tu|tú)\s+nombre\b'
        ],
        "context": lambda: f"""[INFORMACIÓN VERIFICADA]
Tu nombre es: {KNOWLEDGE_BASE['identity']['name']}
Estado: {KNOWLEDGE_BASE['identity']['status']}
Creado por: {KNOWLEDGE_BASE['identity']['creator']}
Tipo: {KNOWLEDGE_BASE['identity']['type']}
Contacto: {KNOWLEDGE_BASE['identity']['email']}
Web: {KNOWLEDGE_BASE['identity']['website']}
"""
    },
    "date": {
        "patterns": [
            r'\b(qué|que)\s+(día|fecha)\b',
            r'\b(hoy|ahora|actual)\b.*\b(día|fecha)\b',
            r'\bcuándo\s+estamos\b',
            r'\bfecha\s+actual\b',
            r'\bqué\s+día\s+es\b'
        ],
        "context": lambda: f"""[FECHA ACTUAL VERIFICADA]
Hoy es {KNOWLEDGE_BASE['current_info']['day']}, {KNOWLEDGE_BASE['current_info']['date']}
"""
    }
}

def detect_context_needs(query):
    """Detecta qué contextos son relevantes para la consulta"""
    query_lower = query.lower()
    contexts = []
    
    for context_type, config in CONTEXT_TRIGGERS.items():
        for pattern in config['patterns']:
            try:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    context_result = config['context']()
                    if context_result:
                        contexts.append(context_result)
                        print(f"✓ Contexto agregado: {context_type}")
                    break  # Solo un contexto por tipo
            except Exception as e:
                print(f"⚠️ Error en pattern {context_type}: {e}")
                continue
    
    return contexts

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'smart-mcp-capibara6',
        'version': '2.0',
        'approach': 'selective-rag',
        'contexts_available': len(CONTEXT_TRIGGERS),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_query():
    """Analiza si la consulta necesita contexto adicional"""
    
    # Manejar preflight
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json(force=True)
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({'error': 'Query is required'}), 400
        
        print(f"\n📝 Query: {user_query[:100]}")
        
        # Detectar contextos relevantes
        relevant_contexts = detect_context_needs(user_query)
        
        # Solo agregar contexto si es realmente relevante
        if relevant_contexts:
            augmented_prompt = "\n".join(relevant_contexts) + f"\n\nPregunta del usuario: {user_query}"
            
            print(f"✅ {len(relevant_contexts)} contexto(s) agregado(s)")
            
            return jsonify({
                'needs_context': True,
                'original_query': user_query,
                'augmented_prompt': augmented_prompt,
                'contexts_added': len(relevant_contexts),
                'lightweight': True
            })
        else:
            print(f"✓ Sin contexto necesario")
            
            return jsonify({
                'needs_context': False,
                'original_query': user_query,
                'augmented_prompt': user_query,
                'contexts_added': 0,
                'lightweight': True
            })
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Smart MCP Server v2.0 - Completo")
    print("=" * 60)
    print("📦 Contextos disponibles:")
    print("   - Identidad de Capibara6")
    print("   - Fecha actual")
    print("📊 Enfoque: Selective RAG")
    print("🎯 Puerto: 5010")
    print("=" * 60)
    print("")
    
    app.run(
        host='0.0.0.0',
        port=5010,
        debug=False
    )

