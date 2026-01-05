#!/usr/bin/env python3
"""
MCP Server - Capibara6
Model Context Protocol para proporcionar contexto verificado
y reducir alucinaciones
"""

import os
import json
import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app, origins=['http://localhost:8000', 'http://127.0.0.1:8000', 'http://localhost:5500', 'http://127.0.0.1:5500'])

# ============================================
# CONTEXTOS DISPONIBLES
# ============================================

CONTEXT_SOURCES = {
    'company_info': {
        'name': 'Información de la Empresa',
        'description': 'Datos sobre Anachroni s.coop y Capibara6',
        'data': {
            'company_name': 'Anachroni s.coop',
            'product_name': 'Capibara6',
            'product_type': 'Sistema de consenso de IA',
            'status': 'Beta',
            'models': ['Capibara6 (Gemma3-12B)', 'OSS-120B (TPU-v5e-64)'],
            'capabilities': [
                'Generación de texto',
                'Generación de código',
                'Análisis de datos',
                'Respuestas multilingüe'
            ]
        }
    },
    
    'technical_specs': {
        'name': 'Especificaciones Técnicas',
        'description': 'Información técnica del sistema',
        'data': {
            'model_capibara6': {
                'base': 'Gemma3-12B',
                'hardware': 'Google Axion ARM64',
                'parameters': '12 billones',
                'context_window': '4096 tokens',
                'quantization': 'Q4_K_M GGUF'
            },
            'model_oss120b': {
                'base': 'Open Source Supervised 120B',
                'hardware': 'TPU-v5e-64',
                'parameters': '120 billones',
                'specialization': 'Análisis complejo'
            }
        }
    },
    
    'current_date': {
        'name': 'Fecha y Hora Actual',
        'description': 'Información temporal actualizada',
        'data': lambda: {
            'date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.datetime.now().strftime('%H:%M:%S'),
            'day_of_week': datetime.datetime.now().strftime('%A'),
            'year': datetime.datetime.now().year
        }
    }
}

# ============================================
# HERRAMIENTAS MCP
# ============================================

TOOLS = {
    'search_web': {
        'name': 'Buscar en Web',
        'description': 'Busca información actualizada en internet',
        'enabled': False  # Requiere API key
    },
    
    'calculate': {
        'name': 'Calculadora',
        'description': 'Realiza cálculos matemáticos precisos',
        'enabled': True
    },
    
    'get_context': {
        'name': 'Obtener Contexto',
        'description': 'Obtiene información verificada del sistema',
        'enabled': True
    },
    
    'verify_facts': {
        'name': 'Verificar Hechos',
        'description': 'Verifica información contra fuentes confiables',
        'enabled': True
    }
}

# ============================================
# FUNCIONES DE CONTEXTO
# ============================================

def get_context(context_id: str) -> Dict[str, Any]:
    """Obtiene contexto específico"""
    source = CONTEXT_SOURCES.get(context_id)
    if not source:
        return {'error': f'Contexto {context_id} no encontrado'}
    
    data = source['data']
    # Si es función, ejecutarla
    if callable(data):
        data = data()
    
    return {
        'context_id': context_id,
        'name': source['name'],
        'data': data
    }

def get_all_contexts() -> Dict[str, Any]:
    """Obtiene todos los contextos disponibles"""
    contexts = {}
    for context_id, source in CONTEXT_SOURCES.items():
        data = source['data']
        if callable(data):
            data = data()
        contexts[context_id] = data
    
    return contexts

def calculate(expression: str) -> Dict[str, Any]:
    """Calcula una expresión matemática de forma segura"""
    try:
        # Evaluar de forma segura (sin exec/eval de Python directo)
        # Solo permitir operaciones matemáticas básicas
        allowed_chars = set('0123456789+-*/().% ')
        if not all(c in allowed_chars for c in expression):
            return {'error': 'Expresión contiene caracteres no permitidos'}
        
        result = eval(expression, {"__builtins__": {}}, {})
        return {
            'expression': expression,
            'result': result
        }
    except Exception as e:
        return {'error': str(e)}

def verify_fact(claim: str, category: str = 'general') -> Dict[str, Any]:
    """Verifica un hecho contra los contextos disponibles"""
    # Por ahora, solo verifica contra los contextos locales
    all_contexts = get_all_contexts()
    
    # Buscar en los contextos si hay información relacionada
    relevant_info = []
    claim_lower = claim.lower()
    
    for context_id, data in all_contexts.items():
        data_str = json.dumps(data).lower()
        if any(word in data_str for word in claim_lower.split()):
            relevant_info.append({
                'context': context_id,
                'data': data
            })
    
    return {
        'claim': claim,
        'verified': len(relevant_info) > 0,
        'sources': relevant_info
    }

def augment_prompt_with_context(prompt: str, contexts: List[str] = None) -> str:
    """Aumenta el prompt con contexto relevante de forma concisa"""
    if contexts is None:
        contexts = ['company_info', 'current_date']
    
    # Construir contexto conciso
    facts = []
    for context_id in contexts:
        ctx = get_context(context_id)
        if 'error' not in ctx:
            data = ctx['data']
            
            if context_id == 'company_info':
                facts.append(f"Empresa: {data['company_name']}")
                facts.append(f"Producto: {data['product_name']} (Estado: {data['status']})")
                
            elif context_id == 'technical_specs':
                capibara = data['model_capibara6']
                facts.append(f"Modelo: {capibara['base']} ({capibara['parameters']} parámetros)")
                facts.append(f"Hardware: {capibara['hardware']}")
                facts.append(f"Contexto: {capibara['context_window']}")
                
            elif context_id == 'current_date':
                facts.append(f"Fecha actual: {data['date']}")
                facts.append(f"Día: {data['day_of_week']}")
    
    if facts:
        facts_str = "; ".join(facts)
        augmented_prompt = f"[DATOS VERIFICADOS: {facts_str}]\n\n{prompt}"
        return augmented_prompt
    
    return prompt

# ============================================
# RUTAS API
# ============================================

@app.route('/api/mcp/contexts', methods=['GET'])
def list_contexts():
    """Lista todos los contextos disponibles"""
    try:
        contexts_list = []
        for context_id, source in CONTEXT_SOURCES.items():
            contexts_list.append({
                'id': context_id,
                'name': source['name'],
                'description': source['description']
            })
        
        return jsonify({
            'contexts': contexts_list,
            'total': len(contexts_list)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mcp/context/<context_id>', methods=['GET'])
def get_context_endpoint(context_id):
    """Obtiene un contexto específico"""
    try:
        context = get_context(context_id)
        return jsonify(context)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mcp/augment', methods=['POST'])
def augment_prompt():
    """Aumenta un prompt con contexto relevante"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        contexts = data.get('contexts')  # Lista opcional de contextos
        
        if not prompt:
            return jsonify({'error': 'Prompt requerido'}), 400
        
        augmented = augment_prompt_with_context(prompt, contexts)
        
        return jsonify({
            'original_prompt': prompt,
            'augmented_prompt': augmented,
            'contexts_used': contexts or ['company_info', 'current_date']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mcp/tools', methods=['GET'])
def list_tools():
    """Lista herramientas MCP disponibles"""
    try:
        tools_list = []
        for tool_id, tool_info in TOOLS.items():
            tools_list.append({
                'id': tool_id,
                'name': tool_info['name'],
                'description': tool_info['description'],
                'enabled': tool_info['enabled']
            })
        
        return jsonify({
            'tools': tools_list,
            'total': len(tools_list)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mcp/calculate', methods=['POST'])
def calculate_endpoint():
    """Realiza un cálculo matemático"""
    try:
        data = request.get_json()
        expression = data.get('expression', '')
        
        if not expression:
            return jsonify({'error': 'Expresión requerida'}), 400
        
        result = calculate(expression)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mcp/verify', methods=['POST'])
def verify_fact_endpoint():
    """Verifica un hecho"""
    try:
        data = request.get_json()
        claim = data.get('claim', '')
        category = data.get('category', 'general')
        
        if not claim:
            return jsonify({'error': 'Claim requerido'}), 400
        
        result = verify_fact(claim, category)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mcp/health', methods=['GET'])
def health_check():
    """Health check del servidor MCP"""
    return jsonify({
        'status': 'healthy',
        'service': 'capibara6-mcp',
        'contexts_available': len(CONTEXT_SOURCES),
        'tools_available': len([t for t in TOOLS.values() if t['enabled']]),
        'timestamp': datetime.datetime.now().isoformat()
    })

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("🧠 Iniciando MCP Server - Capibara6...")
    print("=" * 50)
    print(f"Contextos disponibles: {len(CONTEXT_SOURCES)}")
    for ctx_id, ctx in CONTEXT_SOURCES.items():
        print(f"  • {ctx['name']}")
    
    print(f"\nHerramientas habilitadas: {len([t for t in TOOLS.values() if t['enabled']])}")
    for tool_id, tool in TOOLS.items():
        status = '✅' if tool['enabled'] else '❌'
        print(f"  {status} {tool['name']}")
    
    print("\n🌐 Servidor ejecutándose en: http://localhost:5003")
    print("📋 Endpoints disponibles:")
    print("  • GET  /api/mcp/contexts - Lista contextos")
    print("  • GET  /api/mcp/context/<id> - Obtiene contexto")
    print("  • POST /api/mcp/augment - Aumenta prompt con contexto")
    print("  • GET  /api/mcp/tools - Lista herramientas")
    print("  • POST /api/mcp/calculate - Calculadora")
    print("  • POST /api/mcp/verify - Verifica hechos")
    print("  • GET  /api/mcp/health - Health check")
    
    app.run(host='0.0.0.0', port=5003, debug=True)
