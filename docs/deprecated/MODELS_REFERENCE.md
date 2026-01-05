# Modelos de Capibara6

## Modelos Disponibles

### phi4:mini (modelo rápido)
- **Nombre**: Phi-4 Mini
- **Parámetros**: 14B (mejora de phi3:mini con 3.8B)
- **Uso**: Tareas simples, respuestas rápidas, saludos
- **Endpoint**: `http://34.12.166.76:8001/v1`
- **Tipo**: vLLM OpenAI-compatible
- **Hardware**: GPU
- **Características**: Responde a tareas simples y directas

### qwen2.5-coder-1.5b (modelo experto en código)
- **Nombre**: Qwen2.5-Coder 1.5B
- **Parámetros**: 1.5B
- **Uso**: Programación, desarrollo, debugging, tareas técnicas
- **Endpoint**: `http://34.12.166.76:8002/v1` 
- **Tipo**: vLLM OpenAI-compatible
- **Hardware**: GPU
- **Características**: Experto en código, análisis técnico, desarrollo

### gpt-oss-20b (modelo complejo)
- **Nombre**: GPT-OSS-20B
- **Parámetros**: 20B
- **Uso**: Análisis profundos, razonamiento complejo, planificación
- **Endpoint**: `http://34.12.166.76:8000/v1`
- **Tipo**: vLLM OpenAI-compatible
- **Hardware**: GPU
- **Características**: Complejidad alta, análisis profundos

### mixtral (modelo general)
- **Nombre**: Mixtral 8x7B
- **Parámetros**: 8x7B (sparse mixture of experts)
- **Uso**: Creatividad, escritura, tareas generales
- **Endpoint**: `http://34.12.166.76:8003/v1`
- **Tipo**: vLLM OpenAI-compatible
- **Hardware**: GPU
- **Características**: Multilingüe, creatividad, contenido general

## Configuración del Sistema

### model_config.json
```json
{
  "models": {
    "fast_response": {
      "name": "phi4:mini",
      "description": "Modelo más rápido para respuestas simples",
      "max_tokens": 512,
      "timeout": 8000,
      "use_case": [
        "preguntas simples", 
        "respuestas rápidas", 
        "chistes", 
        "saludos", 
        "respuestas directas"
      ]
    },
    "balanced": {
      "name": "qwen2.5-coder-1.5b",
      "description": "Modelo experto en código y tareas técnicas",
      "max_tokens": 1024,
      "timeout": 20000,
      "use_case": [
        "explicaciones",
        "análisis intermedio", 
        "redacción",
        "resumen corto",
        "programación",
        "análisis técnico"
      ]
    },
    "complex": {
      "name": "gpt-oss-20b",
      "description": "Modelo más potente para tareas complejas",
      "max_tokens": 2048,
      "timeout": 240000,
      "use_case": [
        "análisis profundo",
        "razonamiento complejo",
        "planificación",
        "análisis técnico"
      ]
    }
  }
}
```

### Sistema de Consenso
```json
{
  "fallback_strategy": {
    "enabled": true,
    "order": ["fast_response", "balanced", "complex"]
  },
  "load_balancing": {
    "enable_preloading": true,
    "models_to_keep_loaded": ["phi4:mini", "qwen2.5-coder-1.5b"],
    "max_loaded_models": 2,
    "cache_enabled": true,
    "cache_ttl_seconds": 3600
  },
  "api_settings": {
    "vllm_endpoint": "http://34.12.166.76:8000/v1",
    "default_model": "phi4:mini",
    "max_concurrent_requests": 4,
    "streaming_enabled": true
  }
}
```

## Peso de Modelos en Consenso
- `phi4`: 0.7
- `qwen2.5-coder`: 0.8
- `gpt-oss-20b`: 0.9
- `mixtral`: 0.6

## Enrutamiento Automático
El sistema detecta automáticamente el tipo de pregunta y selecciona el modelo más adecuado:
- **Preguntas simples**: phi4:mini
- **Programación/código**: qwen2.5-coder-1.5b
- **Análisis complejo**: gpt-oss-20b
- **Contenido creativo**: mixtral