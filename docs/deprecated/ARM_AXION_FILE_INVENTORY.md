# Archivos Importantes del Sistema ARM Axion

## Directorio Principal (/home/elect/capibara6/)

### Documentación Importante
- `CURRENT_SYSTEM_STATUS.md` - Estado actual del sistema
- `SYSTEM_STATUS_SUMMARY.md` - Resumen del sistema ARM Axion
- `ANALYSIS_MODELS_ARM_AXION.md` - Análisis detallado de modelos
- `README.md` - Documentación principal del proyecto

### Configuraciones y Scripts
- `e2b_config.json` - Configuración de E2B
- `model_config.json` - Configuración de modelos
- `five_model_config.json` - Configuración de 5 modelos (anterior)
- `requirements.txt` - Dependencias del sistema

## Directorio de Optimización ARM (/home/elect/capibara6/arm-axion-optimizations/vllm_integration/)

### Archivos de Configuración (MANTENER)
- `config.four_models.gemma3_optimized.json` - ✅ ACTIVO: Configuración actual con 4 modelos incluyendo Gemma3
- `config.four_models.gemma3_included.json` - Configuración con Gemma3 (backup)
- `config.example.json` - Ejemplo de configuración base
- `config.production.optimized.json` - Configuración de producción optimizada
- `config.production.json.backup.20251124_122450` - Backup de seguridad

### Archivos de Servidor (MANTENER)
- `multi_model_server.py` - ✅ SERVIDOR ACTIVO: Código del servidor multi-modelo con todas las correcciones
- `README.md` - Documentación del sistema vLLM

### Archivos de Soporte (MANTENER)
- `vllm_axion_backend.py` - Backend de vLLM optimizado para ARM
- `semantic_router.py` - Router semántico con optimizaciones
- `livemind_orchestrator.py` - Orquestador principal
- `embedding_cache.py` - Sistema de caché de embeddings

## Archivos Temporales (Pueden limpiarse periódicamente)
- Archivos `server_*.log` - Logs de servidores temporales (pueden eliminarse)
- Archivos de backup antiguos que no sean los principales

## Archivos Obsoletos (Pueden eliminarse si se está seguro)
- `config.json` (symlink) - Usualmente apunta a configuración de prueba
- `config_mistral_only.json` - Configuración de prueba con solo mistral
- `config.production.json` - Symlink que apunta al original

## Importante
- No eliminar `config.four_models.gemma3_optimized.json` - Es la configuración activa del sistema
- No eliminar `multi_model_server.py` - Contiene todas las correcciones importantes
- Mantener los archivos de documentación para referencia futura

## Servidor Actual en Uso
El sistema está operando actualmente con:
- `config.four_models.gemma3_optimized.json`
- Puerto 8080
- 4 modelos activos (phi4, mistral, qwen, gemma3)