# Análisis de Configuraciones de Modelos ARM-Axion

## Estado Actual de las Configuraciones

### Configuración Activa
- **Archivo principal**: `config.json` (enlace simbólico a `config.five_models_with_aya.json`)
- **Modelos definidos**:
  1. `phi4_fast` - Modelo rápido general
  2. `mistral_balanced` - Modelo técnico equilibrado
  3. `qwen_coder` - Modelo de programación
  4. `gemma3_multimodal` - Modelo multimodal complejo
  5. `aya_expanse_multilingual` - Modelo multilingüe (8B params)

### Problemas Identificados

#### 1. Funcionalidad Parcial del Sistema
**Observación**: De los 5 modelos definidos, solo `aya_expanse_multilingual` está funcionando correctamente.
**Evidencia**: Pruebas de latencia mostraron:
- `aya_expanse_multilingual`: 100% éxito (3/3 pruebas)
- Otros 4 modelos: 0% éxito (0/3 pruebas) con "Internal Server Error"

#### 2. Posibles Causas
1. **Configuración incompatible**: Parámetros específicos de algunos modelos podrían no ser compatibles con la versión actual de vLLM
2. **Falta de recursos**: Los modelos grandes podrían requerir más memoria de la disponible
3. **Problemas de archivos**: Algunos archivos de modelo podrían estar incompletos
4. **Parámetros de cuantización**: Configuraciones de AWQ/Q4 podrían no ser compatibles

#### 3. Comportamiento del Modelo Funcional
- `aya_expanse_multilingual` carga correctamente (lazy loading funciona)
- Tiempo de primera carga alto (~58s) seguido de tiempos más rápidos (~4.5s)
- Responde correctamente a consultas multilingües

## Comparación de Configuraciones

### Configuración Funcional (`config.five_models_with_aya.json`)
```json
{
  "experts": [
    {
      "expert_id": "aya_expanse_multilingual",
      "model_path": "/home/elect/models/aya-expanse-8b",
      "domain": "multilingual_expert",
      "quantization": "awq",  // o null
      "dtype": "float16",    // o bfloat16
      ...
    }
  ]
}
```

### Parámetros Clave por Modelo
- `phi4_fast`: quantization "awq", dtype "float16"
- `mistral_balanced`: quantization "awq", dtype "float16" 
- `qwen_coder`: quantization "awq", dtype "float16"
- `gemma3_multimodal`: quantization null, dtype "bfloat16"
- `aya_expanse_multilingual`: quantization "awq", dtype "float16" (funcional)

## Recomendaciones para Resolución

### 1. Verificación de Configuración
- Validar los parámetros de configuración para cada modelo no funcional
- Comparar configuración exitosa de `aya_expanse_multilingual` con otros modelos

### 2. Pruebas Aisladas
- Probar cada modelo individualmente para identificar el problema específico
- Verificar integridad de archivos de modelo en `/home/elect/models/`

### 3. Recursos del Sistema
- Verificar disponibilidad de memoria RAM para cada modelo
- Validar configuración de lazy loading para todos los modelos

### 4. Actualización de Configuraciones
- Crear una configuración de trabajo con solo modelos funcionales
- Gradualmente añadir modelos no funcionales una vez resueltos los problemas

## Archivos de Configuración Relacionados

### Archivos Críticos
- `/home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.five_models_with_aya.json` - Configuración actual
- `/home/elect/capibara6/backend/models_config.py` - Configuración de backend
- `/home/elect/capibara6/vm-bounty2/config/models_config.py` - Configuración de VM

### Archivos de Modelo Individuales
- `/home/elect/models/aya-expanse-8b/config.json` - Configuración específica
- `/home/elect/models/phi-4-mini/config.json` - Configuración específica
- `/home/elect/models/mistral-7b-instruct-v0.2/config.json` - Configuración específica
- `/home/elect/models/qwen2.5-coder-1.5b/config.json` - Configuración específica
- `/home/elect/models/gemma-3-27b-it/config.json` - Configuración específica

## Conclusión

El sistema multimodelo ARM-Axion está correctamente configurado en términos de archivos de configuración, pero presenta problemas de funcionalidad con 4 de los 5 modelos definidos. El modelo `aya_expanse_multilingual` está funcionando correctamente y puede servir como referencia para resolver problemas con los modelos restantes.