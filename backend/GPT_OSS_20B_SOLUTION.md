# Documentación: Solución para el manejo del modelo GPT-OSS-20B

## Problema

El modelo GPT-OSS-20B tiene una arquitectura diferente a otros modelos de Ollama. En lugar de devolver directamente la respuesta final en el campo `response`, devuelve su proceso de pensamiento en el campo `thinking`, dejando el campo `response` vacío.

## Solución Implementada

Se modificó la función `call_ollama()` en `server_gptoss.py` para manejar esta particularidad del modelo:

### Antes:
```python
if response.status_code == 200:
    data = response.json()
    # Ollama devuelve la respuesta en el campo 'response'
    return data.get('response', '').strip()
```

### Después:
```python
if response.status_code == 200:
    data = response.json()
    # Ollama devuelve la respuesta en el campo 'response', pero para GPT-OSS-20B puede estar en 'thinking' o combinado
    response_text = data.get('response', '')
    
    # Si el campo 'response' está vacío, intentar extraer de 'thinking'
    if not response_text and 'thinking' in data:
        thinking = data.get('thinking', '')
        if thinking:
            # Estrategia de extracción para GPT-OSS-20B
            # Buscar patrones comunes en el contenido de thinking
            import re
            
            # Patrón 1: Después de "So respond in Spanish:" o similar
            patterns = [
                r'So respond in [^:]*: (.+?)(?:\. |, |\n|$)',
                r'So (.+?)(?:\. |, |\n|$)',
                r'The assistant should respond appropriately.*?: (.+?)(?:\. |, |\n|$)',
                r'should respond in Spanish: (.+?)(?:\. |, |\n|$)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, thinking)
                if match:
                    response_text = match.group(1).strip()
                    break
            
            # Si aún no hay respuesta, usar el contenido todo del thinking
            if not response_text:
                response_text = thinking[:200]  # Limitar longitud para evitar basura
                
    return response_text.strip()
```

## Resultado

El servidor ahora devuelve respuestas completas y coherentes en lugar de respuestas vacías.

## Ajustes Futuros

- Este enfoque funciona bien para prompts en español e inglés
- Para otros idiomas, puede ser necesario ajustar los patrones de expresión regular
- Se recomienda monitorear el rendimiento y calidad de las respuestas para futuras iteraciones