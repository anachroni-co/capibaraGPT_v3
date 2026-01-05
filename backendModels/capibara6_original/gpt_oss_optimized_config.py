#!/usr/bin/env python3
"""
Configuración Optimizada para GPT-OSS-20B
Parámetros y prompts mejorados para obtener respuestas más coherentes
"""

# ============================================
# CONFIGURACIÓN DE PARÁMETROS OPTIMIZADOS
# ============================================

GPT_OSS_OPTIMIZED_PARAMS = {
    # Parámetros principales
    "n_predict": 200,  # Respuestas completas pero no excesivas
    "temperature": 0.8,  # Más creatividad y diversidad
    "top_p": 0.9,  # Mejor diversidad de respuestas
    "repeat_penalty": 1.1,  # Evita repeticiones sin ser excesivo
    "top_k": 40,  # Añadido para mejor calidad
    "tfs_z": 1.0,  # Tail free sampling
    "typical_p": 1.0,  # Typical sampling
    
    # Tokens de parada optimizados
    "stop": [
        "Usuario:",
        "Capibara6:",
        "\n\n",
        "<|endoftext|>",
        "</s>",
        "<|end|>",
        "<end_of_turn>",
        "<|im_end|>"
    ],
    
    # Configuración de streaming
    "stream": True,
    "stream_options": {
        "include_stop_str": True,
        "include_stop_token": True
    }
}

# ============================================
# PROMPTS DEL SISTEMA OPTIMIZADOS
# ============================================

SYSTEM_PROMPTS = {
    "default": """Eres Capibara6, un asistente de IA especializado en tecnología, programación e inteligencia artificial desarrollado por Anachroni s.coop.

INSTRUCCIONES CRÍTICAS:
- Responde SIEMPRE en español
- Sé específico y detallado en tus respuestas (mínimo 50 palabras)
- Evita respuestas genéricas como "soy un modelo de IA"
- Proporciona información útil y práctica
- Mantén un tono profesional pero amigable
- Si no sabes algo, admítelo honestamente
- Incluye ejemplos cuando sea apropiado

Tu personalidad es profesional pero cercana, y siempre intentas ayudar de la mejor manera posible.""",

    "technical": """Eres Capibara6, un experto en tecnología y programación.

INSTRUCCIONES TÉCNICAS:
- Responde SIEMPRE en español
- Proporciona código limpio y bien documentado
- Explica conceptos técnicos de manera clara
- Incluye ejemplos prácticos
- Menciona mejores prácticas cuando sea relevante
- Si hay múltiples soluciones, explica las ventajas de cada una""",

    "creative": """Eres Capibara6, un asistente creativo y original.

INSTRUCCIONES CREATIVAS:
- Responde SIEMPRE en español
- Sé creativo pero mantén la coherencia
- Usa un lenguaje rico y expresivo
- Estructura tu respuesta de manera atractiva
- Incluye detalles que hagan la respuesta más interesante
- Mantén un tono amigable y cercano""",

    "concise": """Eres Capibara6, un asistente directo y eficiente.

INSTRUCCIONES CONCISAS:
- Responde SIEMPRE en español
- Sé directo y al punto
- Evita información innecesaria
- Proporciona respuestas claras y útiles
- Mantén un tono profesional
- Máximo 100 palabras por respuesta"""
}

# ============================================
# FUNCIONES DE UTILIDAD
# ============================================

def get_optimized_payload(prompt, template="default", custom_params=None):
    """
    Crear payload optimizado para GPT-OSS-20B
    
    Args:
        prompt: El prompt del usuario
        template: Template del sistema a usar
        custom_params: Parámetros personalizados (opcional)
    
    Returns:
        dict: Payload optimizado para la API
    """
    system_prompt = SYSTEM_PROMPTS.get(template, SYSTEM_PROMPTS["default"])
    
    # Crear prompt completo
    full_prompt = f"{system_prompt}\n\nUsuario: {prompt}\n\nCapibara6:"
    
    # Combinar parámetros base con personalizados
    params = GPT_OSS_OPTIMIZED_PARAMS.copy()
    if custom_params:
        params.update(custom_params)
    
    # Crear payload
    payload = {
        "prompt": full_prompt,
        **params
    }
    
    return payload

def get_context_aware_payload(prompt, context=None, template="default"):
    """
    Crear payload con contexto adicional
    
    Args:
        prompt: El prompt del usuario
        context: Contexto adicional (opcional)
        template: Template del sistema a usar
    
    Returns:
        dict: Payload con contexto
    """
    system_prompt = SYSTEM_PROMPTS.get(template, SYSTEM_PROMPTS["default"])
    
    # Añadir contexto si está disponible
    if context:
        context_section = f"\n\nCONTEXTO ADICIONAL:\n{context}"
        full_prompt = f"{system_prompt}{context_section}\n\nUsuario: {prompt}\n\nCapibara6:"
    else:
        full_prompt = f"{system_prompt}\n\nUsuario: {prompt}\n\nCapibara6:"
    
    # Crear payload
    payload = {
        "prompt": full_prompt,
        **GPT_OSS_OPTIMIZED_PARAMS
    }
    
    return payload

# ============================================
# CONFIGURACIÓN DE TEMPLATES POR CATEGORÍA
# ============================================

TEMPLATE_CATEGORIES = {
    "programming": {
        "template": "technical",
        "additional_params": {
            "temperature": 0.7,  # Más conservador para código
            "n_predict": 300  # Más tokens para código
        }
    },
    "creative_writing": {
        "template": "creative",
        "additional_params": {
            "temperature": 0.9,  # Más creativo
            "n_predict": 250
        }
    },
    "quick_questions": {
        "template": "concise",
        "additional_params": {
            "temperature": 0.6,  # Más directo
            "n_predict": 100
        }
    },
    "general": {
        "template": "default",
        "additional_params": {}
    }
}

def get_category_payload(prompt, category="general", context=None):
    """
    Obtener payload optimizado por categoría
    
    Args:
        prompt: El prompt del usuario
        category: Categoría de la consulta
        context: Contexto adicional (opcional)
    
    Returns:
        dict: Payload optimizado para la categoría
    """
    category_config = TEMPLATE_CATEGORIES.get(category, TEMPLATE_CATEGORIES["general"])
    
    # Obtener parámetros base
    base_params = GPT_OSS_OPTIMIZED_PARAMS.copy()
    base_params.update(category_config["additional_params"])
    
    # Crear payload con contexto
    if context:
        return get_context_aware_payload(prompt, context, category_config["template"])
    else:
        return get_optimized_payload(prompt, category_config["template"], base_params)

# ============================================
# CONFIGURACIÓN DE CALIDAD
# ============================================

QUALITY_SETTINGS = {
    "high_quality": {
        "temperature": 0.7,
        "top_p": 0.85,
        "repeat_penalty": 1.15,
        "n_predict": 250
    },
    "balanced": {
        "temperature": 0.8,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "n_predict": 200
    },
    "creative": {
        "temperature": 0.9,
        "top_p": 0.95,
        "repeat_penalty": 1.05,
        "n_predict": 300
    }
}

def get_quality_payload(prompt, quality="balanced", template="default"):
    """
    Obtener payload con configuración de calidad específica
    
    Args:
        prompt: El prompt del usuario
        quality: Nivel de calidad ("high_quality", "balanced", "creative")
        template: Template del sistema a usar
    
    Returns:
        dict: Payload con configuración de calidad
    """
    quality_params = QUALITY_SETTINGS.get(quality, QUALITY_SETTINGS["balanced"])
    return get_optimized_payload(prompt, template, quality_params)
