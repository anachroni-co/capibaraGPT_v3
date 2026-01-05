"""
Smart MCP Server - Versión mejorada basada en estándares reales
Selectivo, ligero y efectivo para Capibara6
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import re  # Para patterns de contexto

app = Flask(__name__)
CORS(app, origins=[
    'http://localhost:5500',
    'http://127.0.0.1:5500',
    'http://172.22.128.1:5500',  # IP de red local (Live Server)
    'http://localhost:8000',
    'http://127.0.0.1:8000',
    'http://172.22.128.1:8000',   # IP de red local (Python server)
    'https://capibara6.vercel.app',  # Vercel production
    'https://*.vercel.app',  # Vercel previews
    'http://34.175.104.187:8080',  # VM IP (si se accede directamente)
    '*'  # Permitir todos los orígenes (ajustar en producción)
])

# Base de conocimiento verificada
KNOWLEDGE_BASE = {
    "identity": {
        "name": "Capibara6 Consensus",
        "creator": "Anachroni s.coop",
        "status": "Beta (en pruebas)",
        "type": "Modelo de lenguaje basado en Gemma 3-12B",
        "hardware": "Google Cloud TPU v5e-64 en europe-southwest1-b",
        "website": "http://www.anachroni.co",
        "email": "info@anachroni.co"
    },
    "current_info": {
        "date": "9 de octubre de 2025",
        "day": "jueves"
    }
}

# Detectores inteligentes de cuándo usar contexto
CONTEXT_TRIGGERS = {
    "identity": {
        "patterns": [
            r'\b(quién|quien|que)\s+(eres|soy|es)\b',
            r'\b(cómo|como)\s+(te\s+llamas|se\s+llama)\b',
            r'\b(tu|tú)\s+(nombre|identidad)\b',
            r'\bcapibara\b',
            r'\bcreo|creador|desarrollador\b',
            r'\bquién\s+te\s+(creó|creo)\b'
        ],
        "context": lambda: f"""[INFO VERIFICADA]
Nombre: {KNOWLEDGE_BASE['identity']['name']}
Estado: {KNOWLEDGE_BASE['identity']['status']}
Creador: {KNOWLEDGE_BASE['identity']['creator']}
Web: {KNOWLEDGE_BASE['identity']['website']}
Contacto: {KNOWLEDGE_BASE['identity']['email']}
Tipo: {KNOWLEDGE_BASE['identity']['type']}
Hardware: {KNOWLEDGE_BASE['identity']['hardware']}
"""
    },
    "date": {
        "patterns": [
            r'\b(qué|que)\s+(día|fecha)\b',
            r'\b(hoy|ahora|actual)\b.*\b(día|fecha)\b',
            r'\bcuándo\s+estamos\b',
            r'\bfecha\s+actual\b'
        ],
        "context": lambda: f"""[FECHA ACTUAL]
Hoy es {KNOWLEDGE_BASE['current_info']['day']}, {KNOWLEDGE_BASE['current_info']['date']}
"""
    },
    "calculation": {
        "patterns": [
            r'\b(calcula|calcular|suma|resta|multiplica|divide)\b',
            r'\d+\s*[\+\-\*×÷/]\s*\d+',
            r'\bcuánto\s+es\b.*\d+'
        ],
        "context": lambda query: calculate_if_possible(query)
    }
}

def calculate_if_possible(query):
    """Intenta resolver cálculos matemáticos básicos"""
    # Buscar operaciones matemáticas
    math_pattern = r'(\d+(?:\.\d+)?)\s*([\+\-\*×÷/])\s*(\d+(?:\.\d+)?)'
    match = re.search(math_pattern, query)
    
    if match:
        num1, op, num2 = match.groups()
        num1, num2 = float(num1), float(num2)
        
        operations = {
            '+': num1 + num2,
            '-': num1 - num2,
            '*': num1 * num2,
            '×': num1 * num2,
            '/': num1 / num2 if num2 != 0 else None,
            '÷': num1 / num2 if num2 != 0 else None
        }
        
        result = operations.get(op)
        if result is not None:
            # Formatear resultado (sin decimales si es entero)
            result_str = f"{int(result)}" if result.is_integer() else f"{result:.2f}"
            return f"""[CÁLCULO]
{num1} {op} {num2} = {result_str}
"""
    
    return None

def detect_context_needs(query):
    """Detecta qué contextos son relevantes para la consulta"""
    query_lower = query.lower()
    contexts = []
    
    for context_type, config in CONTEXT_TRIGGERS.items():
        for pattern in config['patterns']:
            if re.search(pattern, query_lower):
                context_func = config['context']
                if context_type == 'calculation':
                    context_result = context_func(query_lower)
                else:
                    context_result = context_func()
                
                if context_result:
                    contexts.append(context_result)
                break  # Solo un contexto por tipo
    
    return contexts

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'smart-mcp-capibara6',
        'version': '2.0',
        'approach': 'selective-rag',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_query():
    """Analiza si la consulta necesita contexto adicional"""
    try:
        data = request.json
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Detectar contextos relevantes
        relevant_contexts = detect_context_needs(user_query)
        
        # Solo agregar contexto si es realmente relevante
        if relevant_contexts:
            augmented_prompt = "\n".join(relevant_contexts) + f"\n\nPregunta: {user_query}"
            return jsonify({
                'needs_context': True,
                'original_query': user_query,
                'augmented_prompt': augmented_prompt,
                'contexts_added': len(relevant_contexts),
                'lightweight': True
            })
        else:
            return jsonify({
                'needs_context': False,
                'original_query': user_query,
                'augmented_prompt': user_query,
                'contexts_added': 0,
                'lightweight': True
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update-date', methods=['POST'])
def update_date():
    """Actualiza la fecha actual (para mantenimiento)"""
    try:
        data = request.json
        KNOWLEDGE_BASE['current_info']['date'] = data.get('date', KNOWLEDGE_BASE['current_info']['date'])
        KNOWLEDGE_BASE['current_info']['day'] = data.get('day', KNOWLEDGE_BASE['current_info']['day'])
        
        return jsonify({
            'success': True,
            'current_info': KNOWLEDGE_BASE['current_info']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Smart MCP Server v2.0 iniciando...")
    print("📊 Enfoque: Selective RAG (Retrieval-Augmented Generation)")
    print("✅ Contexto SOLO cuando es necesario")
    print("🎯 Ligero y efectivo para Capibara6")
    app.run(host='0.0.0.0', port=5010, debug=True)

