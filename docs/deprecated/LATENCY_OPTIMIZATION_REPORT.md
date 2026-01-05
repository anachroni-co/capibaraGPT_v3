# ANÁLISIS DE RENDIMIENTO Y OPTIMIZACIONES - SISTEMA MULTIMODELO ARM-Axion

## RESUMEN EJECUTIVO

Tras el análisis exhaustivo del sistema multimodelo actual en la VM models-europe (vLLM con optimizaciones ARM-Axion), se identifican áreas de mejora significativas para reducir la latencia. El sistema actual muestra tiempos de respuesta entre 0.5s-0.7s para TTFT (Time To First Token) con un throughput de 14.16 tokens/segundo en pruebas iniciales.

## CONFIGURACIÓN ACTUAL

### Parámetros de servidor:
- **Modelos activos**: 5 modelos ARM-Axion optimizados
- **Puerto**: 8082 (OpenAI-compatible API)
- **Optimizaciones**: 
  - NEON kernels activados
  - ARM Compute Library (ACL) integrada
  - Flash Attention (solo para gemma3_multimodal)
  - Chunked prefill con max_num_batched_tokens = 32768 (phi4), 16384 (mistral/qwen), 8192 (gemma3/aya)

### Configuración de cada modelo:
- **phi4_fast**: max_num_seqs=1024, max_num_batched_tokens=32768, dtype=float16
- **mistral_balanced**: max_num_seqs=512, max_num_batched_tokens=16384, dtype=float16  
- **qwen_coder**: max_num_seqs=512, max_num_batched_tokens=16384, dtype=float16
- **gemma3_multimodal**: max_num_seqs=128, max_num_batched_tokens=8192, dtype=bfloat16
- **aya_expanse_multilingual**: max_num_seqs=256, max_num_batched_tokens=8192, dtype=bfloat16

## ANÁLISIS DE RENDIMIENTO ACTUAL

### Métricas observadas:
- **TTFT**: ~0.706 segundos (medido con curl)
- **Tokens/segundo**: 14.16 (bajo para un sistema optimizado)
- **Tiempo total de respuesta**: 0.7069 segundos para 10 tokens

### Observaciones:
- El modelo está respondiendo con texto incoherente, lo que puede indicar problemas de inferencia
- La tasa de tokens/segundo es baja considerando las optimizaciones ARM-Axion
- La configuración actual tiene parámetros que pueden no estar optimizados para baja latencia

## ÁREAS DE MEJORA IDENTIFICADAS

### 1. OPTIMIZACIONES DE PROCESAMIENTO POR LOTES (BATCHING)

**Problema actual**: 
- max_num_seqs varía entre modelos (1024, 512, 256, 128)
- max_num_batched_tokens está configurado alto (hasta 32768) lo que puede aumentar latencia en solicitudes individuales

**Recomendación**:
```
# Para baja latencia, configurar valores más bajos
"phi4_fast": {
  "max_num_seqs": 256,  # Reducido de 1024
  "max_num_batched_tokens": 4096,  # Reducido de 32768
  "enable_chunked_prefill": false  # Deshabilitar para baja latencia
}
```

### 2. OPTIMIZACIONES DE MEMORIA Y CACHÉ

**Problema actual**:
- enable_prefix_caching = false en todos los modelos excepto gemma3 y aya
- KV cache dtype = "auto" sin optimización específica
- No se utiliza CPU offloading a pesar de tener suficiente RAM

**Recomendación**:
```
"performance_tuning": {
  "enable_prefix_caching": true,  # Habilitar para modelos grandes
  "kv_cache_dtype": "fp8",        # Si disponible, reduce uso de memoria
  "cpu_offload_gb": 4,            # Offload para mantener modelos activos
  "max_context_len_to_capture": 16384,  # Optimizar para contexto promedio
}
```

### 3. CONFIGURACIÓN DE MODELO ESPECÍFICA

**Problema actual**:
- dtype bfloat16 en modelos grandes (mayor precisión pero más lento)
- No se usa cuantización AWQ (deshabilitado temporalmente)
- enforce_eager activado en algunos modelos (más lento pero más estable)

**Recomendación**:
```
"gemma3_multimodal": {
  "dtype": "float16",        # Cambiar de bfloat16 a float16
  "quantization": "awq",     # Re-enable AWQ para velocidad
  "enforce_eager": false,    # Permitir optimizaciones de PyTorch
}
```

### 4. PARÁMETROS DE CHUNKED PREFILL Y SCHEDULING

**Problema actual**:
- enable_chunked_prefill = true en todos los modelos
- Para solicitudes pequeñas/cortas, esto puede aumentar latencia
- No se usa v2 block manager en todos los modelos que lo soportan

**Recomendación**:
```
"chunked_prefill_config": {
  "enabled": false,                    # Deshabilitar para baja latencia
  "chunk_size": 150,                   # Si se mantiene, usar valor óptimo
  "min_chunk_size": 100               # Configurar tamaño óptimo
},
"scheduler_config": {
  "use_v2_block_manager": true,       # Habilitar en todos los modelos
  "num_scheduler_steps": 4,           # Reducir de 8 a 4 para baja latencia
}
```

### 5. PRELOADING Y CACHÉ INTELIGENTE

**Problema actual**:
- Lazy loading activado con "auto_unload_after_s": 300
- Warmup pool size = 2, lo que puede causar cold start

**Recomendación**:
```
"lazy_loading": {
  "enabled": false,                    # Deshabilitar para baja latencia
  "max_loaded_experts": 5,             # Cargar todos los expertos
  "warmup_pool_size": 5,               # Pre-cargar todos los modelos
  "memory_threshold": 0.6              # Configurar para mantenimiento activo
}
```

## IMPLEMENTACIÓN RECOMENDADA

### 1. Archivo de configuración optimizado (config.low_latency.json):

```json
{
  "experts": [
    {
      "expert_id": "phi4_fast",
      "model_path": "/home/elect/models/phi-4-mini",
      "domain": "general",
      "priority": 5,
      "quantization": "awq",          # Re-enable para velocidad
      "tensor_parallel_size": 1,
      "gpu_memory_utilization": 0.7,  # Reducido para permitir más cachés
      "max_num_seqs": 256,            # Reducido para baja latencia
      "enable_neon": true,
      "enable_chunked_prefill": false, # Deshabilitado para baja latencia
      "max_num_batched_tokens": 4096,  # Reducido para TTFT más rápido
      "max_model_len": 4096,
      "enforce_eager": false,         # Permitir optimizaciones
      "kv_cache_dtype": "auto",
      "use_v2_block_manager": true,
      "enable_prefix_caching": true,   # Habilitado para velocidad
      "device": "cpu",
      "dtype": "float16",
      "use_captured_graph": true,     # Habilitado para inferencia rápida
      "swap_space": 4,
      "cpu_offload_gb": 2,
      "num_scheduler_steps": 4        # Reducido para baja latencia
    },
    // ... configuraciones similares para otros modelos
  ],
  "lazy_loading": {
    "enabled": false,                 # Todos los modelos precargados
    "warmup_pool_size": 5,
    "max_loaded_experts": 5,
    "memory_threshold": 0.6,
    "auto_unload_after_s": null       # No auto-unload
  },
  "performance_tuning": {
    "swap_space": 16,                 # Mayor espacio de swap
    "cpu_offload_gb": 4,
    "enforce_eager": false,           # Permitir optimizaciones
    "max_context_len_to_capture": 8192,
    "enable_prefix_caching": true,
    "use_v2_block_manager": true,
    "num_scheduler_steps": 4,         # Prioridad a latencia sobre throughput
    "enable_chunked_prefill": false,
    "max_num_cached_tokens": 16384    # Aumentado para caché de prefijos
  }
}
```

### 2. Optimizaciones del sistema operativo

```bash
# Ajustes de sistema para baja latencia
echo 'net.core.somaxconn = 65535' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_max_syn_backlog = 65535' >> /etc/sysctl.conf
echo 'vm.swappiness = 10' >> /etc/sysctl.conf  # Minimizar swapping

# Ajustes de CPU para prioridad de inferencia
echo 'net.core.netdev_max_backlog = 5000' >> /etc/sysctl.conf

# Configurar scheduler de CPU
echo 'deadline' > /sys/devices/system/cpu/cpu*/scheduler
```

### 3. Monitorización de rendimiento

- Implementar métricas de TTFT (Time To First Token) y TBT (Time Between Tokens)
- Añadir logging de tiempos de inferencia por modelo
- Implementar alertas para degradación de rendimiento

## RESULTADOS ESPERADOS

Con las optimizaciones implementadas:

- **Reducción de latencia TTFT**: De ~0.7s a ~0.2-0.3s
- **Aumento de tokens/segundo**: De 14.16 a >25 tokens/segundo
- **Mayor estabilidad**: Reducción de respuestas incoherentes
- **Mejor experiencia de usuario**: Respuestas más rápidas y consistentes

## CONCLUSIONES

El sistema actual tiene una base sólida con optimizaciones ARM-Axion, pero requiere ajustes específicos para baja latencia. La configuración actual prioriza el throughput sobre la latencia, lo cual puede no ser óptimo para aplicaciones interactivas. Las principales áreas de mejora están en el batching, la gestión de memoria y la precarga de modelos.

La implementación de las recomendaciones anteriores debería reducir la latencia en un 50-60% mientras se mantiene la calidad de la inferencia.