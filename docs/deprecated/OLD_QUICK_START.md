# Guía Rápida - Capibara6 ARM Axion

## Para Agentes y Desarrolladores

Esta es una guía rápida para entender el sistema actual y ponerse en marcha rápidamente.

## ⚠️ Información Importante para Agentes - Leer Primero

**Arquitectura Distribuida - VM models-europe:**
- Esta VM (`models-europe`) SOLO debe ejecutar servicios de IA/modelos
- NO iniciar servicios MCP, TTS o backend en esta VM
- Servicios como `mcp_server.py`, `kyutai_tts_server.py`, `capibara6_integrated_server.py` corren en la VM `services`
- Esta VM ejecuta: `multi_model_server.py` en el puerto 8082 con 5 modelos de IA

## ⚡ Estado Actual (2025-12-02)

**Sistema**: ✅ Operativo
**Servidor**: Puerto 8082
**Modelos**: 5 modelos disponibles (con lazy loading)
**VM**: models-europe (ARM Axion C4A-standard-32)

## 🚀 Inicio Rápido en 30 Segundos

```bash
# 1. Verificar que el servidor está corriendo
curl http://localhost:8082/health

# 2. Si no está corriendo, iniciarlo
cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration
python3 multi_model_server.py --host 0.0.0.0 --port 8082 --config config.json

# 3. Verificar modelos disponibles
curl http://localhost:8082/v1/models | jq '.data[].id'
```

## 📚 Documentación Esencial

**LEE ESTOS ARCHIVOS EN ESTE ORDEN:**

1. **`README.md`** (este directorio)
   - Estado actual completo del sistema
   - Arquitectura y componentes
   - Ejemplos de uso

2. **`arm-axion-optimizations/vllm_integration/README.md`**
   - Documentación técnica del servidor
   - API completa
   - Troubleshooting

3. **`PRODUCTION_ARCHITECTURE.md`**
   - Arquitectura distribuida entre VMs
   - Comunicación entre servicios

## 🤖 Modelos Disponibles

1. **phi4_fast** → Respuestas rápidas y simples
2. **mistral_balanced** → Tareas técnicas intermedias
3. **qwen_coder** → Especializado en código
4. **gemma3_multimodal** → Análisis complejo, imágenes
5. **aya_expanse_multilingual** → 23 idiomas, multilingüe

**Nota**: Los modelos usan **lazy loading**. Primera carga tarda 20-60 segundos.

## 🔍 Verificaciones Rápidas

```bash
# ¿Está el servidor corriendo?
ps aux | grep multi_model_server

# ¿Puerto 8082 está escuchando?
ss -tlnp | grep 8082

# ¿Cuánta memoria hay disponible?
free -h

# Ver logs del servidor
tail -50 /tmp/multi_model_server.log

# Verificar configuración
cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration
cat config.json | jq '.experts | length'
# Debe devolver: 5
```

## 🧪 Prueba Rápida

```bash
# Test simple
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi4_fast",
    "messages": [{"role": "user", "content": "Hola"}],
    "max_tokens": 20
  }'

# Test con router automático (sin especificar modelo)
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Escribe una función Python"}],
    "max_tokens": 100
  }'
```

## 📁 Estructura de Archivos

```
/home/elect/capibara6/
├── README.md                         ← Lee esto primero
├── QUICK_START.md                    ← Este archivo
├── PRODUCTION_ARCHITECTURE.md        ← Arquitectura
│
├── arm-axion-optimizations/
│   └── vllm_integration/
│       ├── README.md                 ← Docs técnicas
│       ├── multi_model_server.py     ← Servidor principal
│       └── config.json               ← Configuración (symlink)
│
├── backend/                          ← Backend services
├── /home/elect/models/               ← Modelos (5 modelos)
└── docs/deprecated/                  ← Docs antiguas (NO USAR)
```

## ⚠️ Documentos Obsoletos

**NO USES** los documentos en `docs/deprecated/`. Son históricos y están desactualizados.

Si un documento no está en esta lista, probablemente está obsoleto:
- ✅ README.md
- ✅ QUICK_START.md (este archivo)
- ✅ PRODUCTION_ARCHITECTURE.md
- ✅ README_MODELS_SETUP.md
- ✅ AYA_EXPANSE_MODEL_CONFIRMATION.md
- ✅ arm-axion-optimizations/vllm_integration/README.md

## 🛠️ Comandos Comunes

### Iniciar Servidor

```bash
cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration
python3 multi_model_server.py --host 0.0.0.0 --port 8082 --config config.json
```

### Servicios por VM (Importante para Agentes)

**En VM models-europe (esta VM)** - Solo servicios de IA:
- ✅ `multi_model_server.py` en puerto 8082 (servidor de modelos con router semántico)
- ✅ 5 modelos de IA con optimizaciones ARM-Axion
- ❌ NO iniciar: MCP, TTS, servidores backend (corren en otras VMs)

**En VM services** - Servicios de backend y coordinación:
- ✅ `capibara6_integrated_server.py` (backend principal)
- ✅ `mcp_server.py` (Model Context Protocol en puerto 5003)
- ✅ `kyutai_tts_server.py` (Text-to-Speech en puerto 5002)
- ✅ `smart_mcp_server.py` (alternativa en puerto 5010)
- ❌ NO iniciar: Servidor de modelos vLLM (corre en models-europe)

**Comunicación entre VMs**:
- services → models-europe: `http://34.175.48.2:8082` (API de modelos)
- frontend → services: `http://34.175.255.139:5000/api/chat` (endpoint principal)

### Detener Servidor

```bash
# Encontrar PID
ps aux | grep multi_model_server | grep -v grep

# Matar proceso (reemplazar <PID> con el número)
kill <PID>
```

### Cambiar Configuración

```bash
cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration

# Ver configuración actual
ls -la config.json

# Cambiar a otra configuración
ln -sf config.five_models_optimized_with_aya.json config.json

# Reiniciar servidor para aplicar cambios
```

### Ver Estadísticas

```bash
# Health check
curl http://localhost:8082/health

# Modelos disponibles
curl http://localhost:8082/v1/models

# Estadísticas detalladas
curl http://localhost:8082/stats | jq
```

## 🔧 Resolución Rápida de Problemas

### "Servidor no responde"

```bash
# 1. Verificar que está corriendo
ps aux | grep multi_model_server

# 2. Si no está, iniciarlo
cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration
python3 multi_model_server.py --host 0.0.0.0 --port 8082 --config config.json
```

### "Modelo tarda mucho"

- **Normal**: Primera carga (lazy loading) tarda 20-60 segundos
- **Espera**: Deja que termine de cargar
- **Siguiente vez**: Será instantáneo

### "Error de memoria"

```bash
# Ver memoria disponible
free -h

# Reducir modelos cargados simultáneamente
# Editar config.json y reducir max_loaded_experts de 5 a 3
```

## 📞 ¿Necesitas Más Ayuda?

1. **Primero**: Lee `README.md` completo
2. **Luego**: Lee `arm-axion-optimizations/vllm_integration/README.md`
3. **Logs**: Revisa `/tmp/multi_model_server.log`
4. **Arquitectura**: Lee `PRODUCTION_ARCHITECTURE.md`

## 🎯 Objetivo del Sistema

Sistema de IA conversacional con múltiples modelos especializados que:
- Usa router semántico para seleccionar el mejor modelo automáticamente
- Optimizado para ARM Axion con kernels NEON
- API compatible con OpenAI
- Lazy loading para eficiencia de memoria

---

**Última actualización**: 2025-12-02
**Servidor**: Puerto 8082
**Estado**: ✅ Operativo
