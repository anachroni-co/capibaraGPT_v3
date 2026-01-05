# Informe de Pruebas de Latencia - Sistema Multimodelo ARM-Axion

## Fecha de Prueba
Día 2 de diciembre de 2025

## Configuración del Sistema
- VM: models-europe
- Arquitectura: ARM-Axion (C4A-standard-32)
- Modelo: 5 expertos de IA
- Configuración activa: config.five_models_with_aya.json

## Modelos Disponibles
1. `phi4_fast` - Modelo general rápido
2. `mistral_balanced` - Modelo técnico equilibrado  
3. `qwen_coder` - Modelo especializado en programación
4. `gemma3_multimodal` - Modelo multimodal complejo
5. `aya_expanse_multilingual` - Modelo multilingüe (8B params)

## Resultados de Pruebas de Latencia

### Resultados Generales
- Total de modelos probados: 5
- Modelos funcionales: 1 (`aya_expanse_multilingual`)
- Modelos no funcionales: 4 (`phi4_fast`, `mistral_balanced`, `qwen_coder`, `gemma3_multimodal`)

### Detalles por Modelo

#### ✅ `aya_expanse_multilingual` (Funcional)
- **Tasa de éxito**: 100% (3/3 pruebas exitosas)
- **Latencia promedio**: 22.617 segundos
- **Latencia mínima**: 4.536 segundos
- **Latencia máxima**: 58.777 segundos
- **Desviación estándar**: 31.316 segundos
- **Velocidad promedio**: 7.63 tokens/segundo
- **Observaciones**: 
  - El primer acceso es muy lento (posiblemente carga inicial del modelo)
  - Subsecuentes accesos son significativamente más rápidos (~4.5 segundos)
  - Funciona correctamente para consultas multilingües

#### ❌ Modelos no funcionales
- `phi4_fast`: 0% éxito (0/3) - Internal Server Error
- `mistral_balanced`: 0% éxito (0/3) - Internal Server Error  
- `qwen_coder`: 0% éxito (0/3) - Internal Server Error
- `gemma3_multimodal`: 0% éxito (0/3) - Internal Server Error o Timeout

## Análisis de Resultados

### Posibles Causas del Problema
1. **Configuración de modelo incompatible**: Algunos modelos pueden tener parámetros de configuración incompatibles con la versión actual de vLLM
2. **Problemas de archivos del modelo**: Algunos archivos de modelo podrían estar incompletos o dañados
3. **Problemas de memoria**: La inicialización de múltiples modelos grandes puede estar causando errores de memoria
4. **Parámetros de configuración erróneos**: Las configuraciones específicas para cada modelo podrían no ser correctas

### Comportamiento del Modelo Funcional
El modelo `aya_expanse_multilingual` mostró un patrón interesante:
- Primera solicitud: 58.777 segundos (posiblemente tiempo de carga completa del modelo)
- Siguientes solicitudes: ~4.5 segundos (modelo ya cargado en memoria)
- Esto confirma que el sistema de lazy loading funciona correctamente para este modelo

## Recomendaciones

### 1. Investigar Causa Raíz
- Verificar los logs detallados del servidor para identificar errores específicos
- Validar la integridad de los archivos de modelo
- Comprobar configuraciones específicas de cada modelo

### 2. Pruebas Adicionales
- Probar cada modelo individualmente con solicitudes simples
- Verificar disponibilidad de memoria RAM suficiente para cada modelo
- Validar parámetros de configuración de cada modelo

### 3. Prioridad de Resolución
- `aya_expanse_multilingual` ya está funcional y es valioso para capacidades multilingües
- Priorizar la resolución para los modelos restantes según necesidad específica

## Conclusión
El sistema multimodelo ARM-Axion está parcialmente funcional. El modelo `aya_expanse_multilingual` opera correctamente y puede utilizarse para tareas multilingües, pero se requiere investigación y solución de problemas para habilitar los modelos restantes. El sistema de router semántico puede depender únicamente de `aya_expanse_multilingual` temporalmente hasta que se resuelvan los problemas con los otros modelos.