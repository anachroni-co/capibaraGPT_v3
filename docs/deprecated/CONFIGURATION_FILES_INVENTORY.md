# Archivos de Configuración del Sistema Multimodelo ARM-Axion

Este documento lista todos los archivos de configuración importantes relacionados con el sistema multimodelo ARM-Axion en el proyecto Capibara6.

## 1. Archivos de Configuración del Sistema Multimodelo (vLLM Integration)

Ubicados en: `/home/elect/capibara6/arm-axion-optimizations/vllm_integration/`

### Archivos de Configuración Principal
- `config.five_models_with_aya.json` - Configuración actual con 5 modelos incluyendo `aya_expanse_multilingual`
- `config.json` (enlace simbólico) - Apunta a `config.five_models_with_aya.json`
- `config.production.json` (enlace simbólico) - Apunta a `config.five_models_all_working.json`

### Archivos de Configuración Alternativos
- `config.five_models_optimized.json` - Configuración optimizada de 5 modelos
- `config.five_models_optimized_with_aya.json` - Versión optimizada con `aya_expanse`
- `config.five_models_all_working.json` - Configuración con todos los modelos que funcionan
- `config.five_models_awq_resolved.json` - Configuración con problemas de AWQ resueltos
- `config.five_models_corrected.json` - Configuración corregida
- `config.five_models_fixed.json` - Configuración fija
- `config.five_models_original_awq.json` - Configuración original con AWQ
- `config.five_real_models.json` - Configuración con 5 modelos reales
- `config.aya_expanse_optimized.json` - Configuración específica para `aya_expanse`

### Configuraciones con 4 Modelos
- `config.four_models_no_gptoss.json` - 4 modelos sin GPT-OSS
- `config.four_models_gemma3_included.json` - 4 modelos con Gemma3
- `config.four_models_gemma3_optimized.json` - Versión optimizada con Gemma3
- `config.four_models_hardware_optimized.json` - Optimización de hardware

### Otras Configuraciones Especializadas
- `config.example.json` - Archivo de ejemplo
- `config_mistral_only.json` - Solo modelo Mistral
- `config.two_models_bf16.json` - 2 modelos en bfloat16
- `config.no_quantization.json` - Sin cuantización
- `config.production.optimized.json` - Optimizado para producción
- `config.low_latency_batching.json` - Batching para baja latencia
- `config.optimized_kv_cache.json` - Optimizado para caché KV
- `config.ultra_low_latency_kv_cache.json` - Ultra baja latencia con caché KV

## 2. Archivos de Configuración del Proyecto

### Configuración Principal del Proyecto
- `capibara6_config.json` - Configuración general del proyecto
- `model_config.json` - Configuración general de modelos
- `five_model_config.json` - Configuración específica para 5 modelos
- `vm_coordination_config.json` - Configuración de coordinación entre VMs
- `e2b_config.json` - Configuración del sistema E2B

## 3. Archivos de Configuración de Backend

Ubicados en: `/home/elect/capibara6/backend/`
- `models_config.py` - Configuración de modelos en Python
- `models_config_updated.py` - Versión actualizada de la configuración
- `semantic_model_router.py` - Configuración del router semántico

## 4. Archivos de Configuración de VMs

Ubicados en: `/home/elect/capibara6/vm-bounty2/config/`
- `models_config.py` - Configuración de modelos para VM bounty2
- `models_config_updated.py` - Versión actualizada para VM bounty2

## 5. Archivos de Configuración de Modelos Individuales

En cada directorio de modelo en `/home/elect/models/[modelo]/`:
- `config.json` - Configuración específica del modelo
- `tokenizer_config.json` - Configuración del tokenizer
- `generation_config.json` - Configuración de generación
- `model.safetensors.index.json` - Índice de archivos del modelo

## 6. Archivos de Configuración del Sistema

Otros archivos encontrados:
- `vllm-source-modified/.buildkite/performance-benchmarks/tests/serving-tests-cpu-snc2.json` - Configuración de benchmarks
- `vllm-source-modified/.buildkite/performance-benchmarks/tests/serving-tests-cpu-snc3.json` - Configuración de benchmarks
- `monitoring/grafana-dashboard-config.json` - Configuración de monitoreo
- `vm_verification_results.json` - Resultados de verificación de VM

## Importancia de los Archivos

### Críticos para el Funcionamiento
- `config.five_models_with_aya.json` - Contiene la definición de los 5 modelos incluyendo `aya_expanse_multilingual`
- `config.json` - Archivo de configuración principal que lee el servidor

### Importantes para la Operación
- `five_model_config.json` - Configuración del sistema de 5 modelos
- `models_config.py` (backend y vm-bounty2) - Configuración de modelos en Python

### Para Mantenimiento
- Todos los archivos `config.*.json` - Diferentes configuraciones para distintos escenarios
- Archivos de modelo individuales - Configuración específica de cada modelo