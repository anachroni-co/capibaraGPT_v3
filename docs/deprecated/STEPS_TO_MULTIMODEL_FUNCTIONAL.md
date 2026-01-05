# Plan de Acción: Sistema Multimodelo ARM-Axion Funcional

## Informe de Estado Actual

El sistema multimodelo ARM-Axion en la VM `models-europe` está configurado pero requiere los siguientes ajustes para operar correctamente con los 5 modelos, incluyendo `aya_expanse_multilingual`.

## Pasos Recomendados

### 1. Configuración Inicial (Importante)

**Objetivo**: Asegurar que el archivo de configuración principal apunte a la configuración de 5 modelos

**Acción**:
```bash
cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration/
ln -sf config.five_models_with_aya.json config.json
```

**Verificación**:
```bash
# Comprobar enlace simbólico
ls -la config.json
# Debe mostrar: config.json -> config.five_models_with_aya.json
```

### 2. Iniciar el Servidor Correctamente

**Acción**:
```bash
cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration/
python3 inference_server.py --host 0.0.0.0 --port 8080
```

### 3. Verificar Disponibilidad de Modelos

**Acción**:
```bash
# Verificar que los 5 modelos estén disponibles
curl http://localhost:8080/experts | jq '.experts | length'
# Debería retornar: 5

# Verificar detalles de los modelos
curl http://localhost:8080/experts | jq '.experts[].expert_id'
```

**Resultado esperado**: Debe mostrar los 5 modelos:
- `phi4_fast`
- `mistral_balanced` 
- `qwen_coder`
- `gemma3_multimodal`
- `aya_expanse_multilingual`

### 4. Probar Funcionalidad por Modelo

**Acción**:
```bash
# Probar aya_expanse_multilingual (modelo multilingüe)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aya_expanse_multilingual",
    "messages": [{"role": "user", "content": "¿Cómo se dice 'hello' en español y francés?"}],
    "temperature": 0.1,
    "max_tokens": 50
  }'
```

**Resultado esperado**: Debe retornar una respuesta multilingüe apropiada.

### 5. Probar Router Semántico

**Acción**:
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Traduce 'good morning' al italiano"}],
    "temperature": 0.1,
    "max_tokens": 50
  }'
```

**Resultado esperado**: Debe retornar una respuesta, posiblemente manejada por el modelo más apropiado.

### 6. Verificar Estadísticas del Sistema

**Acción**:
```bash
curl http://localhost:8080/stats | jq '.router_stats.num_experts'
# Debería retornar: 5

curl http://localhost:8080/stats | jq '.lazy_loading.total_experts'
# Debería retornar: 5
```

## Posibles Problemas y Soluciones

### Problema: Modelos no responden
**Causa**: Lazy loading puede requerir tiempo para cargar el primer modelo
**Solución**: Las primeras solicitudes a un modelo no cargado pueden ser lentas

### Problema: Internal Server Error
**Causa**: Error en la inicialización del modelo
**Solución**: Verificar que los archivos de modelo existan y tengan permisos correctos

### Problema: 4 modelos en lugar de 5
**Causa**: Archivo de configuración incorrecto
**Solución**: Revisar y corregir el enlace simbólico `config.json`

## Recomendaciones del Sistema

### 1. Mantenimiento del Sistema
- Verificar que `config.json` siempre apunte a la configuración de 5 modelos
- Monitorear uso de memoria durante operación con todos los modelos
- Asegurar suficiente espacio en disco para todos los modelos

### 2. Operación del Sistema
- El modelo `aya_expanse_multilingual` es especialmente útil para consultas multilingües
- El lazy loading optimiza uso de memoria manteniendo modelos no usados descargados
- El router semántico puede mejorar la calidad de respuesta dirigiendo consultas al modelo más apropiado

### 3. Uso por Agentes
- Los agentes deben poder especificar el modelo directamente si se requiere especialidad
- Para consultas generales, el routing automático es efectivo
- El sistema puede manejar solicitudes concurrentes de múltiples agentes gracias a vLLM

## Resultado Esperado

Después de completar estos pasos, el sistema:
- ✅ Tendrá los 5 modelos disponibles, incluyendo `aya_expanse_multilingual`
- ✅ Podrá enrutar consultas automáticamente al modelo más apropiado
- ✅ Será accesible tanto con especificación de modelo como con routing automático
- ✅ Utilizará Lazy Loading para eficiencia de recursos

## Próximos Pasos

1. Implementar el cambio de configuración
2. Iniciar el servidor con la configuración correcta
3. Verificar disponibilidad de todos los modelos
4. Probar cada modelo individualmente
5. Validar el funcionamiento del router semántico
6. Documentar el estado final del sistema