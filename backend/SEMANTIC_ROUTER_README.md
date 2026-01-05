# Semantic Router - Capibara6

Selección automática e inteligente de modelos AI basada en análisis semántico de las consultas del usuario.

## 🎯 ¿Qué es?

El **Semantic Router** analiza el significado semántico de cada consulta y selecciona automáticamente el modelo de IA más adecuado para responder. Esto optimiza la calidad de las respuestas y el uso de recursos.

### Ventajas

- ⚡ **Ultra-rápido**: Decisión instantánea sin latencia de LLM
- 🎯 **Preciso**: Usa embeddings semánticos, no reglas simples
- 💰 **Sin costos**: Funciona 100% local con FastEmbed
- 🧠 **Inteligente**: Entiende el significado, no solo palabras clave
- 📊 **Transparente**: El frontend sabe qué modelo se usó

---

## 📦 Instalación

### 1. Instalar semantic-router

```bash
cd backend
pip install "semantic-router[local]"
```

Esto instala:
- `semantic-router` - La librería principal
- `fastembed` - Encoder local (sin API keys)

### 2. Verificar instalación

```bash
python -c "import semantic_router; print('✅ Instalado correctamente')"
```

---

## 🚀 Uso

### Arrancar servidor con Semantic Router

```bash
cd backend
python capibara6_integrated_server.py
```

Al iniciar, verás:

```
============================================================
🚀 Iniciando Servidor Integrado Capibara6...
============================================================
📡 VM GPT-OSS-20B: http://34.175.215.109:8080/completion
🧠 Smart MCP: Activo
🎵 Coqui TTS: Activo
🎯 Semantic Router: ✅ Activo
🤖 Models Config: ✅ Activo
🌐 Puerto: 5001

📋 Semantic Router configurado:
   • Rutas: 7 (programming, creative_writing, quick_facts...)
   • Modelos: 8
============================================================
```

---

## 🧪 Testing

### 1. Test completo

Prueba todas las categorías:

```bash
cd backend
python test_semantic_router.py
```

Output:
```
🧪 Test Suite - Semantic Router Capibara6
...
📊 Estadísticas Globales
📝 Total de queries probadas: 35
🎯 Queries con ruta específica: 32
⚠️  Queries con fallback: 3

🗺️  Distribución por Rutas:
   • programming          8 queries (22.9%)
   • creative_writing     5 queries (14.3%)
   • quick_facts          5 queries (14.3%)
...
```

### 2. Modo interactivo

```bash
python test_semantic_router.py --interactive
```

```
Query > cómo programar en Python
→ Query: "cómo programar en Python"
   ├─ Ruta detectada: programming
   ├─ Modelo: gpt-oss-20b
   ├─ Confianza: 90%
   ├─ Fallback: No
   └─ Razón: Query clasificada como 'programming' → usando gpt-oss-20b

Query > quit
```

### 3. Test de una categoría

```bash
python test_semantic_router.py --category "Programming"
```

### 4. Test de una query

```bash
python test_semantic_router.py --query "escribe un cuento sobre el espacio"
```

---

## 🗺️ Rutas y Modelos

El router clasifica queries en las siguientes categorías:

| Ruta | Modelo Asignado | Ejemplos |
|------|----------------|----------|
| **programming** | `gpt-oss-20b` | "cómo programar en Python", "debug este código" |
| **creative_writing** | `mixtral` | "escribe un cuento", "crea un poema" |
| **quick_facts** | `phi` | "qué es Python", "quién descubrió América" |
| **analysis** | `gpt-oss-20b` | "analiza las diferencias entre...", "compara..." |
| **conversation** | `phi` | "hola", "háblame de ti" |
| **math** | `gpt-oss-20b` | "resuelve 25 + 37", "calcula el área" |
| **translation** | `mixtral` | "traduce esto al inglés" |
| **default** | `gpt-oss-20b` | Queries que no matchean ninguna ruta |

---

## 🔧 Configuración

### Modificar rutas

Edita `backend/semantic_model_router.py`:

```python
Route(
    name="mi_nueva_ruta",
    utterances=[
        "ejemplo 1",
        "ejemplo 2",
        "ejemplo 3"
    ]
)
```

### Cambiar asignación de modelos

```python
self.model_mapping = {
    "programming": "gpt-oss-20b",  # Cambiar a otro modelo
    "creative_writing": "mixtral",
    # ...
}
```

### Agregar nuevos modelos

Edita `backend/models_config.py`:

```python
'mi-modelo': {
    'name': 'Mi Modelo',
    'base_model': 'Base Model Name',
    'server_url': 'http://ip:puerto/completion',
    'type': 'llama_cpp',
    'hardware': 'GPU',
    'status': 'active',
    'priority': 1,
    'prompt_template': { ... },
    'parameters': { ... }
}
```

---

## 🌐 API Endpoints

### 1. POST `/api/chat` - Chat con selección automática

**Request:**
```json
{
  "message": "cómo programar en Python",
  "use_semantic_router": true
}
```

**Response:**
```json
{
  "response": "Python es un lenguaje...",
  "model": "gpt-oss-20b",
  "tokens": 150,
  "routing_info": {
    "model_id": "gpt-oss-20b",
    "route_name": "programming",
    "confidence": 0.9,
    "reasoning": "Query clasificada como 'programming' → usando gpt-oss-20b",
    "fallback": false
  }
}
```

### 2. GET `/api/router/info` - Info del router

```bash
curl http://localhost:5001/api/router/info
```

**Response:**
```json
{
  "enabled": true,
  "routes": ["programming", "creative_writing", ...],
  "model_mapping": { "programming": "gpt-oss-20b", ... },
  "encoder": "FastEmbed (local)",
  "status": "active",
  "models_configured": 8
}
```

### 3. POST `/api/router/test` - Probar routing

```bash
curl -X POST http://localhost:5001/api/router/test \
  -H "Content-Type: application/json" \
  -d '{"query": "escribe un cuento"}'
```

**Response:**
```json
{
  "query": "escribe un cuento",
  "decision": {
    "model_id": "mixtral",
    "route_name": "creative_writing",
    "confidence": 0.9,
    "reasoning": "Query clasificada como 'creative_writing' → usando mixtral",
    "fallback": false
  }
}
```

### 4. GET `/api/router/routes` - Ver todas las rutas

```bash
curl http://localhost:5001/api/router/routes
```

### 5. GET `/api/router/models` - Ver todos los modelos

```bash
curl http://localhost:5001/api/router/models
```

---

## 🎨 Integración Frontend

### Mostrar modelo usado en UI

```javascript
async function sendMessage(message) {
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message,
            use_semantic_router: true
        })
    });

    const data = await response.json();

    // Mostrar información de routing
    if (data.routing_info) {
        console.log(`🎯 Modelo: ${data.model}`);
        console.log(`📍 Ruta: ${data.routing_info.route_name}`);
        console.log(`💭 Confianza: ${(data.routing_info.confidence * 100).toFixed(0)}%`);

        // Opcional: mostrar badge en UI
        showModelBadge(data.model, data.routing_info.route_name);
    }

    return data.response;
}
```

### Deshabilitar routing temporalmente

```javascript
fetch('/api/chat', {
    method: 'POST',
    body: JSON.stringify({
        message: "test",
        use_semantic_router: false  // Usar modelo por defecto
    })
});
```

---

## 📊 Modelos Configurados (Backend BB)

Actualmente hay **3 modelos activos** en el backend BB:

| ID | Nombre | Parámetros | Hardware | Puerto | Uso |
|----|--------|------------|----------|--------|-----|
| `gpt-oss-20b` | GPT-OSS-20B | 20B | GPU | 8080 | Programación/Matemáticas/Análisis/Default |
| `phi` | Phi-3 Mini | 3.8B | GPU | 8081 | Facts rápidos/Conversación |
| `mixtral` | Mixtral 8x7B | ~47B | GPU | 8082 | Creatividad/Traducción |

---

## 🐛 Troubleshooting

### Error: "Semantic Router no disponible"

**Solución:**
```bash
pip install "semantic-router[local]"
```

### Error: "No module named 'fastembed'"

**Solución:**
```bash
pip install fastembed
```

### Router siempre usa modelo por defecto

**Posibles causas:**
1. Las queries no matchean ninguna ruta
2. Los ejemplos (`utterances`) necesitan mejorarse

**Solución:**
```bash
# Probar query específica
python test_semantic_router.py --query "tu consulta aquí"

# Agregar más ejemplos en semantic_model_router.py
```

### Modelos no disponibles

Si algún modelo no está corriendo en su puerto:
1. El router seleccionará el modelo igual
2. La petición fallará con error 502/504
3. Verifica que los modelos estén corriendo:

```bash
# Verificar puertos
lsof -i :8080
lsof -i :8081
lsof -i :8082
```

---

## 🔄 Actualizar

```bash
cd backend
pip install --upgrade semantic-router
```

---

## 📚 Referencias

- [Semantic Router GitHub](https://github.com/gmarko/semantic-router)
- [FastEmbed Documentation](https://qdrant.github.io/fastembed/)
- Configuración de modelos: `backend/models_config.py`
- Router implementation: `backend/semantic_model_router.py`

---

## 🆘 Soporte

Si tienes problemas:

1. Revisa logs del servidor: `python capibara6_integrated_server.py`
2. Prueba con: `python test_semantic_router.py --interactive`
3. Verifica instalación: `pip list | grep semantic-router`
4. Consulta documentación: `backend/SEMANTIC_ROUTER_README.md`

---

**Última actualización**: Noviembre 2025
**Versión**: 1.0.0
