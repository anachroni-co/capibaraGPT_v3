#!/usr/bin/env python3
"""
Configuración de Modelos - Capibara6 Consensus
Soporte para múltiples modelos con diferentes configuraciones
"""

import os
from typing import Dict, List, Any

# ============================================
# CONFIGURACIÓN DE MODELOS
# ============================================

MODELS_CONFIG = {
    'capibara6': {
        'name': 'Capibara6',
        'base_model': 'Gemma3-12B',
        'server_url': 'http://34.175.104.187:8080/completion',  # IP actualizada
        'type': 'llama_cpp',
        'hardware': 'GPU',
        'status': 'active',
        'priority': 1,
        'prompt_template': {
            'system': 'Responde en el mismo idioma de la pregunta. Si piden código, usa bloques markdown: ```lenguaje',
            'user': '{prompt}',
            'assistant': '',
            'stop_tokens': ['<end_of_turn>', '<|im_end|>', '\n```', 'html<!DOCTYPE', 'html<', 'php<', 'js<', '{-', '<audio', '<video']
        },
        'parameters': {
            'n_predict': 100,
            'temperature': 0.6,
            'top_p': 0.85,
            'repeat_penalty': 1.3,
            'stream': True
        }
    },
    
    'oss-120b': {
        'name': 'OSS-120B',
        'base_model': 'Open Source Supervised 120B',
        'server_url': 'http://tpu-server:8080/completion',  # Cambiar por la URL real del TPU
        'type': 'tpu_inference',
        'hardware': 'TPU-v5e-64',
        'status': 'active',
        'priority': 2,
        'prompt_template': {
            'system': 'Eres un asistente de IA avanzado. Responde de manera precisa y estructurada.',
            'user': 'Usuario: {prompt}\nAsistente:',
            'assistant': '',
            'stop_tokens': ['Usuario:', 'Asistente:', '\n\n', '<|endoftext|>', '<|im_end|>']
        },
        'parameters': {
            'max_tokens': 150,
            'temperature': 0.7,
            'top_p': 0.9,
            'frequency_penalty': 0.1,
            'presence_penalty': 0.1,
            'stream': True
        }
    }
}

# ============================================
# PLANTILLAS DE PROMPTS POR CATEGORÍA
# ============================================

PROMPT_TEMPLATES = {
    'general': {
        'name': 'General',
        'description': 'Conversación general y preguntas abiertas',
        'system_prompt': 'Eres un asistente útil y preciso. Responde de manera clara y concisa.',
        'models': ['capibara6', 'oss-120b']
    },
    
    'coding': {
        'name': 'Programación',
        'description': 'Ayuda con código, debugging y desarrollo',
        'system_prompt': 'Eres un experto programador. Proporciona código limpio, bien documentado y con ejemplos.',
        'models': ['capibara6', 'oss-120b'],
        'additional_instructions': 'Siempre usa bloques de código markdown con el lenguaje especificado.'
    },
    
    'analysis': {
        'name': 'Análisis',
        'description': 'Análisis de datos, investigación y pensamiento crítico',
        'system_prompt': 'Eres un analista experto. Proporciona análisis estructurado, evidencia y conclusiones claras.',
        'models': ['oss-120b'],  # OSS-120B es mejor para análisis complejos
        'additional_instructions': 'Estructura tu respuesta con: 1) Resumen, 2) Análisis detallado, 3) Conclusiones.'
    },
    
    'creative': {
        'name': 'Creativo',
        'description': 'Escritura creativa, storytelling y contenido',
        'system_prompt': 'Eres un escritor creativo y original. Crea contenido atractivo y bien estructurado.',
        'models': ['capibara6', 'oss-120b'],
        'additional_instructions': 'Usa un tono apropiado para el contexto y mantén la coherencia narrativa.'
    },
    
    'technical': {
        'name': 'Técnico',
        'description': 'Documentación técnica, arquitectura y sistemas',
        'system_prompt': 'Eres un arquitecto de software experto. Proporciona documentación técnica precisa y detallada.',
        'models': ['oss-120b'],  # OSS-120B para documentación compleja
        'additional_instructions': 'Incluye diagramas en formato Mermaid cuando sea apropiado.'
    }
}

# ============================================
# CONFIGURACIÓN DE CONSENSO
# ============================================

CONSENSUS_CONFIG = {
    'enabled': True,
    'min_models': 2,
    'max_models': 3,
    'voting_method': 'weighted',  # 'simple', 'weighted', 'confidence'
    'model_weights': {
        'capibara6': 0.6,  # Peso mayor para respuestas rápidas
        'oss-120b': 0.4    # Peso menor pero mayor calidad
    },
    'fallback_model': 'capibara6',  # Modelo de respaldo si falla el consenso
    'timeout': 30  # Segundos para esperar respuestas
}

# ============================================
# FUNCIONES DE UTILIDAD
# ============================================

def get_active_models() -> List[str]:
    """Obtiene la lista de modelos activos"""
    return [model_id for model_id, config in MODELS_CONFIG.items() 
            if config['status'] == 'active']

def get_model_config(model_id: str) -> Dict[str, Any]:
    """Obtiene la configuración de un modelo específico"""
    return MODELS_CONFIG.get(model_id, {})

def get_prompt_template(template_id: str) -> Dict[str, Any]:
    """Obtiene una plantilla de prompt específica"""
    return PROMPT_TEMPLATES.get(template_id, {})

def get_available_templates() -> List[str]:
    """Obtiene la lista de plantillas disponibles"""
    return list(PROMPT_TEMPLATES.keys())

def get_models_for_template(template_id: str) -> List[str]:
    """Obtiene los modelos recomendados para una plantilla"""
    template = get_prompt_template(template_id)
    return template.get('models', [])

def format_prompt(model_id: str, template_id: str, user_prompt: str) -> str:
    """Formatea un prompt usando la plantilla y configuración del modelo"""
    model_config = get_model_config(model_id)
    template = get_prompt_template(template_id)
    
    if not model_config or not template:
        return user_prompt
    
    # Obtener el template del modelo
    model_template = model_config.get('prompt_template', {})
    system_prompt = template.get('system_prompt', model_template.get('system', ''))
    
    # Formatear según el tipo de modelo
    if model_id == 'capibara6':
        # Formato Gemma3
        return f"<bos><start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
    elif model_id == 'oss-120b':
        # Formato OSS-120B
        user_template = model_template.get('user', 'Usuario: {prompt}\nAsistente:')
        return f"{system_prompt}\n\n{user_template.format(prompt=user_prompt)}"
    
    return user_prompt

# ============================================
# CONFIGURACIÓN DE DESARROLLO
# ============================================

def get_development_config():
    """Configuración para desarrollo local"""
    return {
        'models': {
            'capibara6': {
                **MODELS_CONFIG['capibara6'],
                'server_url': 'http://localhost:8080/completion'
            },
            'oss-120b': {
                **MODELS_CONFIG['oss-120b'],
                'server_url': 'http://localhost:8081/completion'
            }
        },
        'consensus': CONSENSUS_CONFIG
    }

def get_production_config():
    """Configuración para producción"""
    return {
        'models': MODELS_CONFIG,
        'consensus': CONSENSUS_CONFIG
    }

# ============================================
# INFORMACIÓN DEL SISTEMA
# ============================================

def get_system_info():
    """Obtiene información del sistema de modelos"""
    active_models = get_active_models()
    return {
        'total_models': len(MODELS_CONFIG),
        'active_models': len(active_models),
        'models_list': active_models,
        'consensus_enabled': CONSENSUS_CONFIG['enabled'],
        'available_templates': get_available_templates(),
        'hardware_info': {
            model_id: config['hardware'] 
            for model_id, config in MODELS_CONFIG.items() 
            if config['status'] == 'active'
        }
    }

if __name__ == '__main__':
    print("🤖 Configuración de Modelos Capibara6")
    print("=" * 50)
    
    info = get_system_info()
    print(f"Modelos activos: {info['active_models']}/{info['total_models']}")
    print(f"Consenso habilitado: {info['consensus_enabled']}")
    print(f"Plantillas disponibles: {len(info['available_templates'])}")
    
    print("\n📋 Modelos configurados:")
    for model_id in info['models_list']:
        config = get_model_config(model_id)
        print(f"  • {config['name']} ({config['hardware']}) - {config['status']}")
    
    print("\n🎯 Plantillas disponibles:")
    for template_id in info['available_templates']:
        template = get_prompt_template(template_id)
        print(f"  • {template['name']}: {template['description']}")
