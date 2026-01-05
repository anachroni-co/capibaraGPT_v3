# Sistema Multimodelo ARM-Axion con Router Semántico

## Descripción General

Sistema multimodelo de código abierto basado en vLLM optimizado para arquitectura ARM-Axion de Google Cloud, con router semántico inteligente y Lazy Loading para eficiencia de recursos.

## Modelos Disponibles

El sistema está configurado con **5 modelos de IA especializados**:

1. **`phi4_fast`** (general) - Modelo rápido para consultas simples
2. **`mistral_balanced`** (technical) - Modelo equilibrado para tareas técnicas
3. **`qwen_coder`** (coding) - Modelo especializado en programación
4. **`gemma3_multimodal`** (multimodal_expert) - Modelo potente para análisis complejo
5. **`aya_expanse_multilingual`** (multilingual_expert) - **Modelo multilingüe (8B params) para 23 idiomas**

## Configuración del Sistema

### Archivo de Configuración Principal

El sistema utiliza un archivo de configuración principal: `config.json` como enlace simbólico al archivo de configuración actual. Para asegurar que todos los 5 modelos estén disponibles:

```bash
cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration/
ln -sf config.five_models_with_aya.json config.json
```

### Ruta de Modelos

Los modelos están ubicados en:
- `/home/elect/models/phi-4-mini`
- `/home/elect/models/mistral-7b-instruct-v0.2`
- `/home/elect/models/qwen2.5-coder-1.5b`
- `/home/elect/models/gemma-3-27b-it`
- `/home/elect/models/aya-expanse-8b`

## Configuración para Agentes

### Importante - Arquitectura Distribuida

**En VM models-europe (esta VM)** - Solo servicios de IA:
- ✅ `multi_model_server.py` en puerto 8082 (servidor de modelos con router semántico)
- ✅ 5 modelos de IA con optimizaciones ARM-Axion
- ❌ NO iniciar: MCP, TTS, servidores backend (corren en la VM `services`)

**En VM services** - Servicios de backend y coordinación:
- ✅ `capibara6_integrated_server.py` (backend principal)
- ✅ `mcp_server.py` (Model Context Protocol en puerto 5003)
- ✅ `kyutai_tts_server.py` (Text-to-Speech en puerto 5002)
- ✅ `smart_mcp_server.py` (alternativa en puerto 5010)

### Paso 1: Verificar Configuración

Antes de iniciar el servidor, asegúrate de que el archivo de configuración apunta al archivo correcto:

```bash
# Verificar el enlace simbólico actual
ls -la /home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.json

# Debe apuntar a config.five_models_with_aya.json
```

### Paso 2: Iniciar el Servidor

```bash
cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration/
python3 inference_server.py --host 0.0.0.0 --port 8080
```

### Paso 3: Verificar Disponibilidad

```bash
# Verificar estado del servidor
curl http://localhost:8080/health

# Listar expertos disponibles
curl http://localhost:8080/experts

# Verificar que estén los 5 expertos
curl http://localhost:8080/stats
```

## Endpoints Disponibles

- `GET /health` - Estado del sistema
- `GET /experts` - Lista de expertos disponibles
- `GET /stats` - Estadísticas del sistema
- `POST /v1/chat/completions` - API OpenAI-compatible para chat
- `POST /v1/completions` - API OpenAI-compatible para completions

## Uso de Modelos Específicos

Para utilizar un modelo específico (como `aya_expanse_multilingual`):

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aya_expanse_multilingual",
    "messages": [{"role": "user", "content": "¿Cómo se dice 'hello' en francés?"}],
    "temperature": 0.1,
    "max_tokens": 50
  }'
```

## Router Semántico

El sistema incluye un router semántico que puede distribuir automáticamente consultas al modelo más apropiado. Para usar el routing automático, simplemente no especifiques el modelo en la solicitud:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Traduce 'hello world' al español"}],
    "temperature": 0.1,
    "max_tokens": 50
  }'
```

## Optimizaciones ARM-Axion

- **Kernels NEON** - Aceleración de operaciones vectoriales
- **ARM Compute Library** - Optimización de operaciones matriciales
- **Cuantización AWQ** - Reducción de uso de memoria
- **Lazy Loading** - Carga bajo demanda para eficiencia

## Solución de Problemas

### Si no se ven todos los 5 modelos:
1. Verifica que el enlace simbólico apunta al archivo correcto
2. Asegúrate de que los archivos de modelo existen en las rutas especificadas

### Si algunos modelos no responden:
1. El primer acceso a un modelo puede tardar más debido al lazy loading
2. Verifica que el sistema tenga suficiente memoria RAM disponible

### Si hay errores en solicitudes:
1. Asegúrate de especificar un modelo en particular o usar el routing automático
2. Verifica que los parámetros de solicitud sean válidos

## Scripts de Inicio

### Para iniciar con la configuración de 5 modelos:
```bash
# Asegura que config.json apunte a la configuración correcta
cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration/
ln -sf config.five_models_with_aya.json config.json

# Inicia el servidor
python3 inference_server.py --host 0.0.0.0 --port 8080
```

## Importante: Modelo Aya Expanse

El modelo `aya_expanse_multilingual` es especialmente valioso para:

- **Soporte multilingüe**: 23 idiomas diferentes
- **Traducción y localización**: Capacidades de traducción de calidad
- **Razonamiento complejo**: Tareas que requieren comprensión profunda
- **Contenido culturalmente sensible**: Adaptado para diferentes contextos culturales

## Pruebas del Sistema

### Verificar todos los modelos:
```bash
curl http://localhost:8080/experts | jq '.experts | length'
# Debe devolver 5
```

### Probar modelo específico:
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aya_expanse_multilingual",
    "messages": [{"role": "user", "content": "¿Qué idiomas hablas?"}],
    "temperature": 0.1,
    "max_tokens": 30
  }'
```

## Notas de Desarrollo

- El modelo `aya_expanse_multilingual` reemplaza al previsto `gpt-oss-20b`
- Ofrece mejor eficiencia con 8B params vs modelos más grandes
- Arquitectura moderna de Cohere para razonamiento multilingüe
- Integrado completamente con el sistema de router semántico