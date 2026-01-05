# Plan de Reorganización del Proyecto - Estructura por VMs

## 🎯 Objetivo

Reorganizar el proyecto en carpetas específicas por VM para mejorar la claridad y mantenibilidad:

1. **vm-bounty2/** - Servicios de modelos de IA (Ollama, GPT-OSS-20B)
2. **vm-services/** - Servicios auxiliares (TTS, MCP, N8N) en VM gpt-oss-20b
3. **vm-rag3/** - Sistema RAG completo (Milvus, Nebula, Bridge API)
4. **frontend/** - Aplicación web frontend y deploy

## 📂 Estructura Propuesta

```
capibara6/
├── vm-bounty2/              # VM de Modelos de IA (34.12.166.76)
│   ├── README.md            # Documentación de la VM
│   ├── servers/             # Servidores principales
│   │   ├── server_gptoss.py (puerto 5001)
│   │   ├── auth_server.py (puerto 5004)
│   │   └── consensus_server.py (puerto 5005)
│   ├── config/              # Configuraciones
│   │   ├── models_config.py
│   │   ├── gpt_oss_optimized_config.py
│   │   └── production_config.py
│   ├── scripts/             # Scripts de inicio y gestión
│   │   ├── start_gptoss_server.py
│   │   ├── start_system.py
│   │   └── check-services.sh
│   ├── core/                # Lógica de negocio
│   │   ├── router/          # Router semántico
│   │   ├── consensus/       # Consensus multi-modelo
│   │   └── execution/       # E2B execution
│   ├── deployment/          # Docker, K8s
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── k8s/
│   └── tests/               # Tests específicos
│
├── vm-services/             # VM de Servicios (34.175.136.104)
│   ├── README.md            # Documentación de la VM
│   ├── tts/                 # Text-to-Speech
│   │   ├── kyutai_tts_server.py (puerto 5002)
│   │   ├── config.py
│   │   └── api/
│   ├── mcp/                 # Model Context Protocol
│   │   ├── mcp_server.py (puerto 5003)
│   │   ├── smart_mcp_server.py
│   │   ├── config.py
│   │   └── api/
│   ├── n8n/                 # Workflow automation
│   │   ├── config/
│   │   └── workflows/
│   ├── scripts/             # Scripts de gestión
│   │   ├── start-all-services.sh
│   │   ├── stop-all-services.sh
│   │   └── check-services.sh
│   └── deployment/          # Docker, K8s
│       ├── Dockerfile.tts
│       ├── Dockerfile.mcp
│       └── docker-compose.yml
│
├── vm-rag3/                 # VM RAG (10.154.0.2)
│   ├── README.md            # Documentación de la VM
│   ├── api/                 # Bridge API
│   │   ├── main.py (puerto 8000)
│   │   ├── routes/
│   │   │   ├── milvus.py
│   │   │   ├── nebula.py
│   │   │   └── rag.py
│   │   └── config/
│   ├── databases/           # Configuraciones de bases de datos
│   │   ├── milvus/
│   │   │   ├── config/
│   │   │   └── schemas/
│   │   ├── nebula/
│   │   │   ├── config/
│   │   │   └── schemas/
│   │   ├── postgres/
│   │   ├── timescaledb/
│   │   └── redis/
│   ├── monitoring/          # Sistema de monitoreo
│   │   ├── prometheus/
│   │   │   ├── prometheus.yml
│   │   │   └── alerts/
│   │   ├── grafana/
│   │   │   ├── dashboards/
│   │   │   └── datasources/
│   │   └── jaeger/
│   ├── scripts/             # Scripts de administración
│   │   ├── start-all-services.sh
│   │   ├── stop-all-services.sh
│   │   ├── check-services.sh
│   │   ├── backup-databases.sh
│   │   └── diagnostics/
│   ├── deployment/          # Docker compose
│   │   ├── docker-compose.yml
│   │   ├── docker-compose.monitoring.yml
│   │   └── .env.example
│   └── docs/                # Documentación específica
│       ├── SETUP.md
│       ├── MONITORING.md
│       └── TROUBLESHOOTING.md
│
├── frontend/                # Aplicación Web Frontend
│   ├── README.md
│   ├── public/              # Archivos estáticos
│   │   ├── index.html
│   │   ├── chat.html
│   │   ├── login.html
│   │   └── assets/
│   ├── src/                 # Código fuente JavaScript
│   │   ├── config.js
│   │   ├── chat-app.js
│   │   ├── clients/         # Clientes de API
│   │   │   ├── milvus-client.js
│   │   │   ├── nebula-client.js
│   │   │   ├── rag-client.js
│   │   │   └── api-client.js
│   │   ├── components/      # Componentes UI
│   │   │   ├── chatbot.js
│   │   │   ├── rating-system.js
│   │   │   ├── template-profiles.js
│   │   │   └── tts-integration.js
│   │   ├── integrations/    # Integraciones
│   │   │   ├── smart-mcp-integration.js
│   │   │   ├── consensus-integration.js
│   │   │   └── entropy-monitor.js
│   │   └── utils/
│   ├── styles/              # CSS
│   │   ├── main.css
│   │   └── chat.css
│   ├── deployment/          # Deploy configs
│   │   ├── nginx.conf
│   │   ├── Dockerfile
│   │   └── deploy.sh
│   └── tests/               # Tests frontend
│
├── shared/                  # Código compartido entre VMs
│   ├── utils/
│   ├── types/
│   └── constants/
│
├── docs/                    # Documentación general
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   ├── API.md
│   └── IMPROVEMENTS_VM_RAG3.md
│
├── scripts/                 # Scripts globales
│   ├── deploy-all.sh
│   ├── backup-all.sh
│   └── health-check-all.sh
│
├── .github/                 # GitHub Actions
│   └── workflows/
│
└── archived/                # Archivos antiguos (mantener)
```

## 🔄 Mapeo de Archivos Actuales → Nuevos

### VM Bounty2 (Modelos)

```
backend/server_gptoss.py                → vm-bounty2/servers/server_gptoss.py
backend/auth_server.py                  → vm-bounty2/servers/auth_server.py
backend/consensus_server.py             → vm-bounty2/servers/consensus_server.py
backend/models_config.py                → vm-bounty2/config/models_config.py
backend/gpt_oss_optimized_config.py     → vm-bounty2/config/gpt_oss_optimized_config.py
backend/production_config.py            → vm-bounty2/config/production_config.py
backend/start_gptoss_server.py          → vm-bounty2/scripts/start_gptoss_server.py
backend/start_system.py                 → vm-bounty2/scripts/start_system.py
backend/core/router/                    → vm-bounty2/core/router/
backend/core/consensus/                 → vm-bounty2/core/consensus/
backend/execution/                      → vm-bounty2/core/execution/
backend/deployment/                     → vm-bounty2/deployment/
api/consensus/                          → vm-bounty2/api/consensus/
k8s/                                    → vm-bounty2/deployment/k8s/
```

### VM Services (TTS, MCP, N8N)

```
backend/kyutai_tts_server.py            → vm-services/tts/kyutai_tts_server.py
backend/coqui_tts_server.py             → vm-services/tts/coqui_tts_server.py
backend/mcp_server.py                   → vm-services/mcp/mcp_server.py
backend/smart_mcp_server.py             → vm-services/mcp/smart_mcp_server.py
api/tts/                                → vm-services/tts/api/
api/mcp/                                → vm-services/mcp/api/
```

### VM RAG3 (RAG System)

```
backend/main.py                         → vm-rag3/api/main.py
monitoring/                             → vm-rag3/monitoring/
backend/scripts/check-services.sh       → vm-rag3/scripts/check-services.sh
backend/scripts/start-all-services.sh   → vm-rag3/scripts/start-all-services.sh
backend/scripts/stop-all-services.sh    → vm-rag3/scripts/stop-all-services.sh
```

### Frontend

```
web/                                    → frontend/public/ (HTML)
web/*.js                                → frontend/src/ (JS organizados)
web/config.js                           → frontend/src/config.js
web/chat-app.js                         → frontend/src/chat-app.js
web/milvus-client.js                    → frontend/src/clients/milvus-client.js
web/nebula-client.js                    → frontend/src/clients/nebula-client.js
web/rag-client.js                       → frontend/src/clients/rag-client.js
web/smart-mcp-integration.js            → frontend/src/integrations/smart-mcp-integration.js
web/rating-system.js                    → frontend/src/components/rating-system.js
web/template-profiles.js                → frontend/src/components/template-profiles.js
web/*.css                               → frontend/styles/
```

### Documentación

```
IMPROVEMENTS_VM_RAG3.md                 → docs/IMPROVEMENTS_VM_RAG3.md
SOLUCIÓN_ERRORES_404.md                 → docs/SOLUCIÓN_ERRORES_404.md
ACTUALIZAR_SERVIDOR_WEB.md              → docs/ACTUALIZAR_SERVIDOR_WEB.md
INFRASTRUCTURE_FINDINGS.md              → docs/INFRASTRUCTURE_FINDINGS.md
VM_RAG3_COMPLETE_ANALYSIS.md            → docs/VM_RAG3_COMPLETE_ANALYSIS.md
BACKEND_CONSOLIDATION_PLAN.md           → docs/BACKEND_CONSOLIDATION_PLAN.md
```

## 📝 Archivos a Crear

### VM Bounty2
- `vm-bounty2/README.md` - Documentación de la VM
- `vm-bounty2/deployment/docker-compose.yml` - Docker compose
- `vm-bounty2/deployment/Dockerfile` - Dockerfile
- `vm-bounty2/.env.example` - Variables de entorno

### VM Services
- `vm-services/README.md` - Documentación de la VM
- `vm-services/deployment/docker-compose.yml` - Docker compose
- `vm-services/deployment/Dockerfile.tts` - Dockerfile TTS
- `vm-services/deployment/Dockerfile.mcp` - Dockerfile MCP
- `vm-services/.env.example` - Variables de entorno

### VM RAG3
- `vm-rag3/README.md` - Documentación de la VM
- `vm-rag3/deployment/docker-compose.yml` - Docker compose completo
- `vm-rag3/deployment/.env.example` - Variables de entorno
- `vm-rag3/docs/SETUP.md` - Guía de setup
- `vm-rag3/docs/MONITORING.md` - Guía de monitoreo
- `vm-rag3/scripts/backup-databases.sh` - Backup script

### Frontend
- `frontend/README.md` - Documentación del frontend
- `frontend/deployment/nginx.conf` - Configuración nginx
- `frontend/deployment/Dockerfile` - Dockerfile
- `frontend/deployment/deploy.sh` - Script de deploy

### Global
- `docs/ARCHITECTURE.md` - Arquitectura general del sistema
- `scripts/deploy-all.sh` - Deploy de todas las VMs
- `scripts/health-check-all.sh` - Health check global

## ⚠️ Consideraciones Importantes

### Imports y Referencias

Después de mover archivos, actualizar imports en:
- Python: `from backend.core.router` → `from vm_bounty2.core.router`
- JavaScript: rutas relativas en HTML
- Configuraciones: paths en docker-compose, nginx.conf

### Deployment

- Cada VM tendrá su propio `docker-compose.yml`
- Scripts de deploy independientes por VM
- Variables de entorno separadas (`.env` por VM)

### Testing

- Tests organizados junto al código de cada VM
- Tests de integración en directorio raíz

### Documentación

- README.md en cada carpeta de VM
- Documentación técnica centralizada en `docs/`
- Guías específicas en cada VM

## 🚀 Plan de Ejecución

### Fase 1: Crear Estructura de Directorios
1. Crear carpetas principales: vm-bounty2, vm-services, vm-rag3, frontend
2. Crear subcarpetas según estructura propuesta

### Fase 2: Mover Archivos VM Bounty2
1. Mover servidores principales
2. Mover configuraciones
3. Mover core logic (router, consensus, execution)
4. Crear README.md

### Fase 3: Mover Archivos VM Services
1. Mover TTS servers
2. Mover MCP servers
3. Mover API endpoints
4. Crear README.md

### Fase 4: Mover Archivos VM RAG3
1. Mover bridge API (main.py)
2. Mover configuraciones de monitoring
3. Mover scripts de administración
4. Crear README.md

### Fase 5: Reorganizar Frontend
1. Separar HTML (public) de JS (src)
2. Organizar clientes de API
3. Organizar componentes UI
4. Organizar integraciones
5. Crear README.md

### Fase 6: Actualizar Referencias
1. Actualizar imports en Python
2. Actualizar rutas en HTML
3. Actualizar paths en configs
4. Actualizar docker-compose

### Fase 7: Documentación
1. Crear README.md para cada VM
2. Crear ARCHITECTURE.md general
3. Mover documentación existente a docs/
4. Crear guías de deploy

### Fase 8: Testing
1. Verificar que todos los servicios arrancan
2. Verificar que frontend se conecta correctamente
3. Actualizar tests existentes

## 📊 Beneficios Esperados

### Organización
- ✅ Separación clara por VM
- ✅ Fácil identificar qué código va en qué servidor
- ✅ Estructura escalable

### Desarrollo
- ✅ Desarrollo independiente por VM
- ✅ Deploy independiente
- ✅ Testing más fácil

### Mantenimiento
- ✅ Más fácil encontrar archivos
- ✅ Documentación específica por VM
- ✅ Debugging más rápido

### Deploy
- ✅ Docker compose por VM
- ✅ CI/CD independiente
- ✅ Rollback más fácil

---

**Estado**: Plan creado, pendiente de aprobación
**Próximo paso**: Ejecutar Fase 1 - Crear estructura de directorios
