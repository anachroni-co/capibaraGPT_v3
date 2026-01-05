#!/usr/bin/env python3
"""
Configuración de Modelos - Capibara6 Consensus
Soporte para múltiples modelos con diferentes configuraciones ARM-Axion
"""

import os
from typing import Dict, List, Any

# Constantes requeridas por otros módulos
DEFAULT_MODEL = 'phi4_fast'  # Modelo predeterminado para respuestas rápidas
TIMEOUT = 120000  # Timeout en milisegundos (2 minutos)

# ============================================
# CONFIGURACIÓN DE MODELOS ARM-Axion
# ============================================

MODELS_CONFIG = {
    'gptoss_complex': {
        'name': 'GPT-OSS-20B (ARM-Axion)',
        'base_model': 'GPT-OSS-20B',
        'server_url': 'http://localhost:8082/v1',
        'type': 'vllm',
        'hardware': 'ARM-Axion',
        'status': 'active',
        'priority': 1,
        'prompt_template': {
            'system': 'Eres un asistente experto en programación y análisis técnico.',
            'user': '{prompt}',
            'assistant': '',
            'stop_tokens': ['<|end|>', '']
        },
        'parameters': {
            'n_predict': 200,
            'temperature': 0.7,
            'top_p': 0.9,
            'repeat_penalty': 1.2,
            'stream': True
        }
    },
    'phi4_fast': {
        'name': 'Phi-4 Mini (ARM-Axion)',
        'base_model': 'Microsoft Phi-4 Mini (14B)',
        'server_url': 'http://localhost:8082/v1',
        'type': 'vllm',
        'hardware': 'ARM-Axion',
        'status': 'active',
        'priority': 5,
        'prompt_template': {
            'system': 'You are a helpful AI assistant. Respond concisely and accurately.',
            'user': '{prompt}',
            'assistant': '',
            'stop_tokens': ['<|end|>', '']
        },
        'parameters': {
            'n_predict': 120,
            'temperature': 0.5,
            'top_p': 0.85,
            'repeat_penalty': 1.2,
            'stream': True
        }
    },
    'qwen_coder': {
        'name': 'Qwen2.5-Coder 1.5B (ARM-Axion)',
        'base_model': 'Qwen/Qwen2.5-Coder-1.5B-Instruct',
        'server_url': 'http://localhost:8082/v1',
        'type': 'vllm',
        'hardware': 'ARM-Axion',
        'status': 'active',
        'priority': 3,
        'prompt_template': {
            'system': 'You are an expert code assistant. Provide accurate, efficient and well-documented code solutions.',
            'user': '{prompt}',
            'assistant': '',
            'stop_tokens': ['<|end|>', '']
        },
        'parameters': {
            'n_predict': 200,
            'temperature': 0.3,
            'top_p': 0.9,
            'repeat_penalty': 1.1,
            'stream': True
        }
    },
    'mistral_balanced': {
        'name': 'Mistral 7B (ARM-Axion)',
        'base_model': 'Mistral-7B-Instruct-v0.2',
        'server_url': 'http://localhost:8082/v1',
        'type': 'vllm',
        'hardware': 'ARM-Axion',
        'status': 'active',
        'priority': 4,
        'prompt_template': {
            'system': 'You are a creative and multilingual AI assistant. Provide detailed and engaging responses.',
            'user': '[INST] {prompt} [/INST]',
            'assistant': '',
            'stop_tokens': ['</s>', '[/INST]', '']
        },
        'parameters': {
            'n_predict': 250,
            'temperature': 0.7,
            'top_p': 0.95,
            'repeat_penalty': 1.1,
            'stream': True
        }
    },
    'gemma3_multimodal': {
        'name': 'Gemma3-27B (ARM-Axion)',
        'base_model': 'Gemma-3-27B-it-awq',
        'server_url': 'http://localhost:8082/v1',
        'type': 'vllm',
        'hardware': 'ARM-Axion',
        'status': 'active',
        'priority': 2,
        'prompt_template': {
            'system': 'You are a multimodal expert assistant. Provide detailed and engaging responses.',
            'user': '<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n',
            'assistant': '',
            'stop_tokens': ['<|end|>', '<end_of_turn>', '']
        },
        'parameters': {
            'n_predict': 300,
            'temperature': 0.7,
            'top_p': 0.9,
            'repeat_penalty': 1.1,
            'stream': True
        }
    }
}

# Resto del archivo
PROMPT_TEMPLATES = {
    'general': {
        'name': 'General',
        'description': 'Conversación general y preguntas abiertas',
        'system_prompt': 'Eres un asistente útil y preciso. Responde de manera clara y concisa.',
        'models': ['gptoss_complex', 'qwen_coder']
    },
    'coding': {
        'name': 'Programación',
        'description': 'Ayuda con código, debugging y desarrollo',
        'system_prompt': 'Eres un experto programador. Proporciona código limpio, bien documentado y con ejemplos.',
        'models': ['qwen_coder', 'phi4_fast'],
        'additional_instructions': 'Siempre usa bloques de código markdown con el lenguaje especificado.'
    },
    'analysis': {
        'name': 'Análisis',
        'description': 'Análisis de datos, investigación y pensamiento crítico',
        'system_prompt': 'Eres un analista experto. Proporciona análisis estructurado, evidencia y conclusiones claras.',
        'models': ['gptoss_complex', 'gemma3_multimodal'],
        'additional_instructions': 'Estructura tu respuesta con: 1) Resumen, 2) Análisis detallado, 3) Conclusiones.'
    },
    'creative': {
        'name': 'Creativo',
        'description': 'Escritura creativa, storytelling y contenido',
        'system_prompt': 'Eres un escritor creativo y original. Crea contenido atractivo y bien estructurado.',
        'models': ['mistral_balanced', 'gemma3_multimodal'],
        'additional_instructions': 'Usa un tono apropiado para el contexto y mantén la coherencia narrativa.'
    },
    'technical': {
        'name': 'Técnico',
        'description': 'Documentación técnica, arquitectura y sistemas',
        'system_prompt': 'Eres un arquitecto de software experto. Proporciona documentación técnica precisa y detallada.',
        'models': ['qwen_coder', 'gptoss_complex'],
        'additional_instructions': 'Incluye diagramas en formato Mermaid cuando sea apropiado.'
    }
}

CONSENSUS_CONFIG = {
    'enabled': True,
    'min_models': 2,
    'max_models': 3,
    'voting_method': 'weighted',
    'model_weights': {
        'phi4_fast': 0.7,
        'qwen_coder': 0.8,
        'gptoss_complex': 0.9,
        'mistral_balanced': 0.6,
        'gemma3_multimodal': 0.85
    },
    'fallback_model': 'phi4_fast',
    'timeout': 30
}

def get_active_models() -> List[str]:
    return [model_id for model_id, config in MODELS_CONFIG.items() if config['status'] == 'active']

def get_model_config(model_id: str) -> Dict[str, Any]:
    return MODELS_CONFIG.get(model_id, {})

def get_prompt_template(template_id: str) -> Dict[str, Any]:
    return PROMPT_TEMPLATES.get(template_id, {})

def get_available_templates() -> List[str]:
    return list(PROMPT_TEMPLATES.keys())

def get_models_for_template(template_id: str) -> List[str]:
    template = get_prompt_template(template_id)
    return template.get('models', [])

def format_prompt(model_id: str, template_id: str, user_prompt: str) -> str:
    model_config = get_model_config(model_id)
    template = get_prompt_template(template_id)
    
    if not model_config or not template:
        return user_prompt
    
    model_template = model_config.get('prompt_template', {})
    system_prompt = template.get('system_prompt', model_template.get('system', ''))
    
    if model_id in ['gemma3_multimodal']:
        return f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
    elif model_id in ['qwen_coder']:
        return f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
    elif model_id in ['mistral_balanced']:
        return f"[INST] {system_prompt} {user_prompt} [/INST]"
    elif model_id in ['phi4_fast']:
        return f"<|system|>{system_prompt}<|end|><|user|>{user_prompt}<|end|><|assistant|>"
    else:
        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

def get_development_config():
    return {
        'models': MODELS_CONFIG,
        'consensus': CONSENSUS_CONFIG
    }

def get_production_config():
    return {
        'models': MODELS_CONFIG,
        'consensus': CONSENSUS_CONFIG
    }

def get_system_info():
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