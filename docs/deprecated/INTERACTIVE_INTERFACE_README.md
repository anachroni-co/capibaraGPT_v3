# Interfaz Interactiva Capibara6 - ARM-Axion Optimizada

## Descripción
Sistema de prueba interactivo para el sistema Capibara6 con 5 modelos de IA optimizados para ARM-Axion, incluyendo:
- **phi4:mini** - Respuestas rápidas 
- **qwen2.5-coder-1.5b** - Codificación y tareas técnicas
- **gemma-3-27b-it-awq** - Multimodal y análisis profundo
- **mistral-7b-instruct-v0.2** - Tareas generales
- **gpt-oss-20b** - Razonamiento complejo

## Características
- ✅ Router semántico con análisis de complejidad
- ✅ Sistema de consenso entre modelos
- ✅ Comparación de modelos
- ✅ Optimizaciones ARM-Axion (NEON, ACL, cuantización)
- ✅ Simulación de respuestas de modelos

## Estructura de Archivos
```
├── interactive_test_interface_optimized.py    # Interfaz principal
├── five_model_config.json                     # Configuración de 5 modelos
├── demo_interfaz.py                          # Demostración de funcionalidades
└── ROUTER_SYSTEM_GUIDE.md                    # Documentación
```

## Cómo Usar

### 1. Iniciar la interfaz interactiva
```bash
cd /home/elect/capibara6
python3 interactive_test_interface_optimized.py
```

### 2. Opciones disponibles
1. **Probar modelo individual** - Prueba un modelo específico
2. **Probar sistema de router semántico** - Análisis de complejidad y dominio
3. **Probar sistema de consenso** - Votación entre múltiples modelos
4. **Probar todos los modelos con análisis comparativo** - Comparación side-by-side
5. **Información del sistema** - Detalles técnicos
6. **Salir** - Terminar la sesión

### 3. Optimizaciones ARM-Axion
- **NEON Kernels**: Matmul 8x8 tiles, RMSNorm vectorizado, RoPE vectorizado
- **ACL (ARM Compute Library)**: GEMM optimizations hasta 2x más rápido
- **Cuantización**: AWQ/Q4 reducen uso de memoria 40-60%
- **Flash Attention**: Mejora performance en contextos largos
- **Chunked Prefill**: Mejora TTFT (Time To First Token)

## Configuración
- **Endpoint**: `http://34.12.166.76:8000/v1`
- **Cantidad de modelos**: 5 modelos activos
- **Optimizado para**: Google Cloud ARM Axion C4A-standard-32
- **RAM recomendada**: 128GB para ejecutar todos los modelos concurrentes

## Funcionalidades Avanzadas

### Router Semántico
Analiza la complejidad de la consulta y selecciona el modelo más apropiado:
- Consultas simples → Modelo rápido (phi4:mini)
- Consultas técnicas → Modelo codificación (qwen2.5-coder-1.5b)
- Consultas complejas → Modelo potente (gemma-3-27b-it-awq o gpt-oss-20b)

### Sistema de Consenso
Obtiene respuestas de múltiples modelos y aplica métodos de consenso para mejorar la calidad de las respuestas finales.

### Análisis Comparativo
Permite comparar respuestas de diferentes modelos side-by-side con métricas de desempeño.

## Beneficios de los 5 Modelos
- **Especialización**: Cada modelo es óptimo para diferentes tipos de tareas
- **Resiliencia**: Si un modelo tiene problemas, otros pueden responder
- **Eficiencia**: Consultas sencillas van a modelos más rápidos
- **Calidad**: Consultas complejas reciben atención de modelos más potentes
- **Flexibilidad**: Sistema adaptable a diferentes tipos de consultas

## Desempeño Esperado
- **phi4:mini**: ~0.15s TTFT
- **qwen2.5-coder**: ~0.4s TTFT  
- **gemma-3-27b**: ~0.5s TTFT (con optimizaciones ACL + 60-70% mejora)
- **mistral-7b**: ~0.3s TTFT
- **gpt-oss-20b**: ~0.7s TTFT (con optimizaciones)

## Notas Importantes
- La interfaz incluye modo simulado para pruebas sin necesidad de servidor vLLM
- Todas las optimizaciones ARM-Axion están configuradas para Neoverse V1/V2
- Soporte para modelos cuantizados (AWQ, GPTQ) para eficiencia de memoria
- Integración completa con el sistema backend de Capibara6

## Mantenimiento
- Configuración centralizada en `five_model_config.json`
- Componentes modulares para fácil actualización
- Logs de desempeño y routing disponibles
- Sistema de fallback configurado