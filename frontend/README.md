# Frontend - Aplicación Web Capibara6

Aplicación web frontend para interactuar con el sistema Capibara6.

## 📋 Características

- **Chat interactivo** con GPT-OSS-20B
- **Búsqueda RAG** (Milvus + Nebula Graph)
- **TTS** (Text-to-Speech) integrado
- **OAuth** (GitHub, Google)
- **Sistema de evaluación** de respuestas
- **Plantillas** de prompts
- **Monitorización** de entropía
- **Dark/Light mode**

## 🚀 Inicio Rápido

### Desarrollo Local

```bash
# Servir archivos estáticos
cd public
python3 -m http.server 8080

# Abrir en navegador
# http://localhost:8080/chat.html
```

### Con Servidor HTTP

```bash
# Usar cualquier servidor HTTP
# Nginx, Apache, etc.

# Ejemplo con nginx:
cp deployment/nginx.conf /etc/nginx/sites-available/capibara6
ln -s /etc/nginx/sites-available/capibara6 /etc/nginx/sites-enabled/
systemctl reload nginx
```

## 📁 Estructura

```
frontend/
├── public/               # Archivos estáticos (HTML)
│   ├── index.html        # Landing page
│   ├── chat.html         # Aplicación de chat
│   ├── login.html        # Página de login
│   └── assets/           # Imágenes, fuentes, etc.
├── src/                  # Código fuente JavaScript
│   ├── config.js         # Configuración principal
│   ├── chat-app.js       # Aplicación principal de chat
│   ├── script.js         # Scripts globales
│   ├── clients/          # Clientes de API
│   │   ├── milvus-client.js    # Cliente Milvus
│   │   ├── nebula-client.js    # Cliente Nebula Graph
│   │   ├── rag-client.js       # Cliente RAG unificado
│   │   └── api-client.js       # Cliente API genérico
│   ├── components/       # Componentes UI
│   │   ├── chatbot.js          # Componente chatbot
│   │   ├── chatbot_gptoss.js   # Chatbot GPT-OSS-20B
│   │   ├── rating-system.js    # Sistema de evaluación
│   │   ├── template-profiles.js # Plantillas
│   │   └── neural-animation.js # Animaciones
│   ├── integrations/     # Integraciones con servicios
│   │   ├── smart-mcp-integration.js  # MCP v2.0
│   │   ├── mcp-integration.js        # MCP v1.0 (legacy)
│   │   ├── consensus-integration.js  # Consensus
│   │   ├── consensus-ui.js           # UI Consensus
│   │   ├── entropy-monitor.js        # Monitor de entropía
│   │   ├── entropy-auto-inject.js    # Auto-inject entropía
│   │   └── tts-integration.js        # Text-to-Speech
│   └── utils/            # Utilidades
│       ├── translations.js
│       └── helpers.js
├── styles/               # CSS
│   ├── main.css
│   └── chat.css
├── deployment/           # Configs de deploy
│   ├── nginx.conf        # Configuración nginx
│   ├── Dockerfile        # Dockerfile para producción
│   └── deploy.sh         # Script de deploy
└── tests/                # Tests frontend
```

## ⚙️ Configuración

### config.js

Archivo principal de configuración en `src/config.js`:

```javascript
// Detecta si estás en localhost o producción
const isLocalhost = window.location.hostname === 'localhost';

const CHATBOT_CONFIG = {
    // Backend principal (VM bounty2)
    BACKEND_URL: isLocalhost
        ? 'http://localhost:5001'
        : 'http://34.12.166.76:5001',

    // Servicios
    SERVICES: {
        MCP: {
            enabled: false,
            url: isLocalhost
                ? 'http://localhost:5003'
                : 'http://34.175.136.104:5003'
        },
        TTS: {
            enabled: true,
            url: isLocalhost
                ? 'http://localhost:5002'
                : 'http://34.175.136.104:5002'
        },
        RAG3_BRIDGE: {
            enabled: true,
            url: isLocalhost
                ? 'http://localhost:8000'
                : 'http://10.154.0.2:8000'
        },
        // ... más servicios
    }
};
```

### Variables de Entorno

Para producción, configurar en `deployment/.env`:

```bash
# URLs de VMs
VM_MODELS_URL=http://34.12.166.76
VM_SERVICES_URL=http://34.175.136.104
VM_RAG_URL=http://10.154.0.2

# OAuth
GITHUB_CLIENT_ID=your_id
GOOGLE_CLIENT_ID=your_id

# Features
ENABLE_TTS=true
ENABLE_MCP=false
ENABLE_RAG=true
```

## 🔧 Componentes Principales

### Chat App

Aplicación principal de chat (`src/chat-app.js`):

```javascript
// Enviar mensaje
async function sendMessage(message) {
    const response = await fetch(
        `${CHATBOT_CONFIG.BACKEND_URL}/api/v1/query`,
        {
            method: 'POST',
            headers: CHATBOT_CONFIG.HEADERS,
            body: JSON.stringify({ message })
        }
    );
    const data = await response.json();
    return data.response;
}
```

### RAG Client

Cliente unificado RAG (`src/clients/rag-client.js`):

```javascript
const ragClient = new RAGClient({
    hybridWeight: 0.7,      // 70% vector, 30% grafo
    enrichContext: true,    // Enriquecer con grafo
    useTOON: true          // Optimización TOON
});

// Búsqueda híbrida
const results = await ragClient.search("¿Qué es Python?");
console.log(results.context);  // Contexto formateado
console.log(results.stats);    // Estadísticas (tokens ahorrados, etc.)
```

### Milvus Client

Cliente para búsqueda vectorial (`src/clients/milvus-client.js`):

```javascript
const milvusClient = new MilvusClient();

// Búsqueda por texto
const results = await milvusClient.searchByText("machine learning", {
    top_k: 10
});

// Búsqueda híbrida con filtros
const filteredResults = await milvusClient.hybridSearch(
    "deep learning",
    { category: "AI", timestamp: { $gte: "2025-01-01" } },
    { top_k: 5 }
);
```

### Nebula Client

Cliente para queries de grafo (`src/clients/nebula-client.js`):

```javascript
const nebulaClient = new NebulaClient();

// Query nGQL directo
const results = await nebulaClient.query(`
    MATCH (v:entity)-[r:relates_to]->(connected)
    WHERE v.name == "Python"
    RETURN v, r, connected LIMIT 10
`);

// Buscar vértices
const vertices = await nebulaClient.findVertices('entity', {
    type: 'programming_language'
});

// Camino más corto
const path = await nebulaClient.findShortestPath('node1', 'node2');
```

### TTS Integration

Text-to-Speech (`src/integrations/tts-integration.js`):

```javascript
// Sintetizar texto a voz
const audio = await synthesizeText("Hola, soy Capibara6", {
    voice: 'default',
    speed: 1.0
});

// Reproducir
audio.play();
```

### Smart MCP

Model Context Protocol v2.0 (`src/integrations/smart-mcp-integration.js`):

```javascript
const smartMCP = new SmartMCPClient();

// Analizar si query necesita contexto
const result = await smartMCP.analyze("¿Qué es Python?");

// Query simple → No agrega contexto
if (!result.needsContext) {
    console.log("Query ligero, sin contexto adicional");
}

// Query complejo → Agrega contexto
if (result.needsContext) {
    console.log("Query complejo, contexto agregado");
}
```

## 📊 Características Avanzadas

### Sistema de Evaluación

Permite evaluar respuestas del LLM:

```javascript
// Evaluar respuesta
ratingSystem.rate(messageId, {
    accuracy: 5,
    relevance: 4,
    helpfulness: 5,
    comment: "Excelente respuesta"
});
```

### Plantillas de Prompts

Plantillas predefinidas para queries comunes:

```javascript
// Usar plantilla
const prompt = templateProfiles.apply("code_review", {
    language: "Python",
    code: "def hello(): print('hi')"
});
```

### Monitorización de Entropía

Detecta degradación de respuestas:

```javascript
// Calcular entropía
const entropy = entropyMonitor.calculate(response);

if (entropy < threshold) {
    console.log("⚠️ Entropía baja detectada");
}
```

## 🎨 Personalización

### Estilos

Modificar `styles/chat.css`:

```css
/* Tema oscuro */
.dark-theme {
    --bg-color: #1a1a1a;
    --text-color: #ffffff;
    --accent-color: #00d4aa;
}

/* Tema claro */
.light-theme {
    --bg-color: #ffffff;
    --text-color: #000000;
    --accent-color: #0066cc;
}
```

### Logo y Branding

Reemplazar archivos en `public/assets/`:
- `logo.png` - Logo principal
- `favicon.ico` - Icono del sitio
- `banner.jpg` - Banner de landing page

## 🐳 Deployment

### Nginx

```bash
# Copiar configuración
sudo cp deployment/nginx.conf /etc/nginx/sites-available/capibara6

# Activar sitio
sudo ln -s /etc/nginx/sites-available/capibara6 /etc/nginx/sites-enabled/

# Copiar archivos
sudo cp -r public/* /var/www/capibara6/
sudo cp -r src /var/www/capibara6/
sudo cp -r styles /var/www/capibara6/

# Recargar nginx
sudo systemctl reload nginx
```

### Docker

```bash
# Build imagen
docker build -f deployment/Dockerfile -t capibara6-frontend .

# Run contenedor
docker run -p 80:80 capibara6-frontend

# Con docker-compose
cd deployment
docker-compose up -d
```

### Script de Deploy

```bash
# Usar script de deploy automático
./deployment/deploy.sh production

# O desarrollo
./deployment/deploy.sh development
```

## 🔍 Debugging

### DevTools Console

Verificar configuración:

```javascript
// Ver configuración cargada
console.log(CHATBOT_CONFIG);

// Ver servicios habilitados
console.log(CHATBOT_CONFIG.SERVICES);

// Test conexión backend
fetch(`${CHATBOT_CONFIG.BACKEND_URL}/health`)
    .then(r => r.json())
    .then(console.log);

// Test cliente RAG
const rag = new RAGClient();
rag.search("test").then(console.log);
```

### Network Tab

Verificar requests:
- Backend debe ser puerto 5001 (NO 8001)
- TTS debe ser puerto 5002
- MCP debe ser puerto 5003
- RAG Bridge debe ser puerto 8000

### Errores Comunes

**Error 404 en puerto 8001**:
- Caché del navegador con archivos antiguos
- Solución: Hard refresh (Ctrl + Shift + R)
- Ver: [ACTUALIZAR_SERVIDOR_WEB.md](../docs/ACTUALIZAR_SERVIDOR_WEB.md)

**Backend no responde**:
```javascript
// Verificar URL correcta
console.log(CHATBOT_CONFIG.BACKEND_URL);
// Debe ser: http://localhost:5001 o http://34.12.166.76:5001
```

**CORS errors**:
- Backend debe tener CORS habilitado
- Headers correctos en `config.js`

## 📚 Documentación Relacionada

- [Configuración de VMs](../docs/INFRASTRUCTURE_FINDINGS.md)
- [Sistema RAG](../docs/IMPROVEMENTS_VM_RAG3.md)
- [Troubleshooting](../docs/SOLUCIÓN_ERRORES_404.md)

## 🚀 Mejoras Futuras

- [ ] Migrar a framework moderno (React, Vue, Svelte)
- [ ] Implementar lazy loading de componentes
- [ ] Agregar Service Worker para PWA
- [ ] Mejorar accesibilidad (ARIA labels)
- [ ] Implementar tests E2E
- [ ] Optimizar bundle size
- [ ] Agregar i18n completo (múltiples idiomas)

## 🧪 Tests

```bash
# Tests unitarios
npm run test

# Tests E2E
npm run test:e2e

# Linting
npm run lint

# Build
npm run build
```

---

**Mantenedor**: Capibara6 Team
**Última actualización**: 2025-11-14
**URL Demo**: http://34.12.166.76 (si está desplegado)
