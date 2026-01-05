# Capibara6 - Arquitectura del Sistema

**Versión**: 2.0
**Fecha**: 2025-11-14
**Estado**: Reorganizado por VMs

## 📋 Visión General

Capibara6 es un sistema de IA distribuido en 3 VMs de Google Cloud, diseñado para proporcionar:
- Interacción con múltiples modelos de IA (GPT-OSS-20B, Ollama)
- Sistema RAG (Retrieval-Augmented Generation) con Milvus y Nebula Graph
- Servicios auxiliares (TTS, MCP, N8N)
- Monitorización completa con Prometheus, Grafana y Jaeger

## 🏗️ Arquitectura de 3 Capas

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                │
│                   (Aplicación Web)                              │
│   - Chat interactivo                                            │
│   - Búsqueda RAG                                                │
│   - TTS integrado                                               │
│   - OAuth (GitHub, Google)                                      │
└────────────┬────────────────────────────────────────────────────┘
             │
             │ HTTPS/REST
             │
┌────────────┴────────────────────────────────────────────────────┐
│                    CAPA DE SERVICIOS                            │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐ │
│  │  VM Bounty2     │  │  VM Services    │  │    VM RAG3     │ │
│  │  (Modelos IA)   │  │  (Servicios)    │  │    (RAG)       │ │
│  │                 │  │                 │  │                │ │
│  │  - GPT-OSS-20B  │  │  - TTS (5002)   │  │  - Milvus      │ │
│  │  - Auth (5004)  │  │  - MCP (5003)   │  │  - Nebula      │ │
│  │  - Consensus    │  │  - N8N (5678)   │  │  - Bridge API  │ │
│  │    (5005)       │  │                 │  │    (8000)      │ │
│  └─────────────────┘  └─────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
             │
             │
┌────────────┴────────────────────────────────────────────────────┐
│                  CAPA DE DATOS                                  │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │   Milvus     │  │ Nebula Graph │  │   PostgreSQL        │  │
│  │  (Vectores)  │  │   (Grafo)    │  │   TimescaleDB       │  │
│  │              │  │              │  │   Redis             │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🖥️ VMs y Distribución

### VM Bounty2 - Modelos de IA
**IP**: 34.12.166.76
**Directorio**: `vm-bounty2/`

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| Backend Principal | 5001 | GPT-OSS-20B API |
| Auth Server | 5004 | OAuth (GitHub, Google) |
| Consensus Server | 5005 | Multi-modelo consensus |

**Responsabilidades**:
- Inferencia de modelos (GPT-OSS-20B, phi3, llama2)
- Router semántico para selección de modelos
- E2B sandboxes para ejecución de código
- Autenticación OAuth
- Consensus multi-modelo

### VM Services - Servicios Auxiliares
**IP**: 34.175.136.104
**Directorio**: `vm-services/`

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| TTS (Kyutai) | 5002 | Text-to-Speech |
| MCP | 5003 | Model Context Protocol |
| N8N | 5678 | Workflow Automation |

**Responsabilidades**:
- Síntesis de voz (TTS)
- Contexto inteligente (MCP)
- Automatización de workflows (N8N)

### VM RAG3 - Sistema RAG
**IP**: 10.154.0.2 (interna)
**Directorio**: `vm-rag3/`

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| Bridge API | 8000 | capibara6-api Gateway |
| Milvus | 19530 | Vector database |
| Nebula Graph | 9669 | Knowledge graph |
| PostgreSQL | 5432 | Base de datos relacional |
| TimescaleDB | 5433 | Time-series database |
| Redis | 6379 | Cache y message broker |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Dashboards |
| Jaeger | 16686 | Distributed tracing |

**Responsabilidades**:
- Búsqueda vectorial (Milvus)
- Knowledge graph (Nebula Graph)
- Búsqueda híbrida RAG
- Monitorización del sistema completo
- Métricas y alertas

## 🔄 Flujo de Datos

### 1. Query Simple

```
Usuario → Frontend → Backend (5001) → Modelo → Respuesta
```

### 2. Query con RAG

```
Usuario → Frontend
    ↓
    Backend (5001)
    ↓
    Bridge API (8000)
    ↓
    ┌─────────┬──────────────┐
    ↓         ↓              ↓
  Milvus  Nebula Graph  PostgreSQL
    ↓         ↓              ↓
    Contexto combinado
    ↓
    Modelo + Contexto
    ↓
    Respuesta enriquecida
```

### 3. Query con Consensus

```
Usuario → Frontend → Consensus Server (5005)
    ↓
    ┌────────┬─────────┬──────────┐
    ↓        ↓         ↓          ↓
  phi3    llama2   gpt-oss   (otros)
    ↓        ↓         ↓          ↓
    Respuestas combinadas y rankeadas
    ↓
    Mejor respuesta
```

### 4. Query con TTS

```
Usuario → Frontend → Backend (5001) → Respuesta
                                          ↓
                                      TTS Server (5002)
                                          ↓
                                      Audio WAV
                                          ↓
                                      Frontend reproduce
```

## 🧩 Componentes Principales

### Router Semántico

Selecciona automáticamente el modelo apropiado:

```
Query → Embeddings (all-MiniLM-L6-v2)
  ↓
Análisis de complejidad
  ↓
  ┌──────────────┬────────────────┬──────────────┐
  ↓              ↓                ↓              ↓
Simple      Moderado        Complejo      Muy Complejo
  ↓              ↓                ↓              ↓
phi3:mini     llama2        gpt-oss-20b    consensus
```

**Criterios**:
- Longitud del query
- Complejidad semántica
- Palabras clave técnicas
- Confidence score

### Sistema RAG

Pipeline de búsqueda híbrida:

```
Query
  ↓
1. Búsqueda Vectorial (Milvus)
   → Top 10 resultados por similitud semántica
  ↓
2. Enriquecimiento con Grafo (Nebula)
   → Para cada resultado, obtener nodos relacionados
  ↓
3. Ranking Híbrido
   → Score = (0.7 × vector_score) + (0.3 × graph_score) + recency_bonus
  ↓
4. Formateo (TOON si > 5 fuentes)
   → Optimización de tokens (30-60% ahorro)
  ↓
5. Contexto final
   → Enviado al LLM
```

### E2B Sandboxes

Ejecución segura de código:

```
Código Python
  ↓
Validación
  ↓
Selección de Template
  ├─ default (timeout: 5min, mem: 512MB)
  ├─ data_analysis (timeout: 10min, mem: 1GB)
  ├─ visualization (timeout: 10min, mem: 1GB)
  └─ machine_learning (timeout: 15min, mem: 2GB)
  ↓
Ejecución en Sandbox Aislado
  ↓
Resultado (stdout, stderr, archivos generados)
```

### Consensus Multi-Modelo

Combina respuestas de múltiples modelos:

```
Query
  ↓
Broadcast a todos los modelos
  ↓
  ┌─────────┬──────────┬───────────────┐
  ↓         ↓          ↓               ↓
phi3     llama2   gpt-oss-20b     gemma:7b
  ↓         ↓          ↓               ↓
Respuestas paralelas
  ↓
Análisis de Consenso
  ├─ Similitud semántica
  ├─ Calidad de respuesta
  ├─ Completitud
  └─ Coherencia
  ↓
Ranking y selección
  ↓
Mejor respuesta (o combinación)
```

## 📊 Monitorización

### Métricas Recolectadas

**Sistema**:
- CPU, memoria, disco, network
- Latencia de requests (p50, p95, p99)
- Tasa de errores (4xx, 5xx)
- Throughput (requests/s)

**RAG**:
- Búsquedas vectoriales/s en Milvus
- Queries/s en Nebula Graph
- Cache hit rate de embeddings
- Tamaño de colecciones/spaces

**Modelos**:
- Distribución de modelos seleccionados
- Complejidad promedio de queries
- Confidence scores
- Tiempo de inferencia

**E2B**:
- Sandboxes activos
- Tiempo de ejecución
- Tasa de éxito/fallo
- Timeout rate

### Alertas

**Críticas** (🔴):
- Latencia p99 > 5 segundos
- Tasa de errores > 50/s
- CPU > 95% por 2 minutos
- Memoria > 95%
- Milvus/Nebula/PostgreSQL DOWN
- Cluster Nebula unhealthy
- Workers RQ < 2 activos

**Warnings** (⚠️):
- Latencia p95 > 2 segundos
- CPU > 80% por 5 minutos
- Memoria > 85%
- Disco > 80%
- Cache hit rate < 30%
- Cola RQ > 100 tareas

## 🔐 Seguridad

### Autenticación

```
Usuario → OAuth Provider (GitHub/Google)
  ↓
Authorization Code
  ↓
Auth Server (5004)
  ↓
Access Token + Refresh Token
  ↓
Frontend (almacenado en localStorage)
  ↓
Requests con Authorization: Bearer <token>
  ↓
Backend valida token
```

### Aislamiento

- **E2B Sandboxes**: Ejecución de código en contenedores aislados
- **Docker**: Todos los servicios en contenedores separados
- **Network Policies**: Firewall rules entre VMs
- **Secrets**: Variables sensibles en .env (no commiteadas)

## 📁 Estructura del Proyecto

```
capibara6/
├── vm-bounty2/          # Modelos de IA
│   ├── servers/
│   ├── config/
│   ├── core/
│   ├── scripts/
│   └── deployment/
├── vm-services/         # Servicios auxiliares
│   ├── tts/
│   ├── mcp/
│   ├── n8n/
│   └── deployment/
├── vm-rag3/             # Sistema RAG
│   ├── api/
│   ├── databases/
│   ├── monitoring/
│   ├── scripts/
│   └── docs/
├── frontend/            # Aplicación web
│   ├── public/
│   ├── src/
│   ├── styles/
│   └── deployment/
├── docs/                # Documentación
├── scripts/             # Scripts globales
└── shared/              # Código compartido
```

## 🚀 Deployment

### Development

```bash
# Iniciar VM Bounty2
cd vm-bounty2
python3 scripts/start_system.py

# Iniciar VM Services
cd vm-services
./scripts/start-all-services.sh

# Iniciar VM RAG3
cd vm-rag3
./scripts/start-all-services.sh

# Iniciar Frontend
cd frontend/public
python3 -m http.server 8080
```

### Production

```bash
# Deploy todas las VMs
./scripts/deploy-all.sh production

# O individual
cd vm-bounty2/deployment && docker-compose up -d
cd vm-services/deployment && docker-compose up -d
cd vm-rag3/deployment && docker-compose up -d
```

## 📈 Escalabilidad

### Horizontal Scaling

- **Milvus**: Cluster mode con múltiples nodos
- **Nebula Graph**: Ya en cluster (3 metad, 3 storaged, 3 graphd)
- **Backend**: Load balancer con múltiples instancias
- **Redis**: Redis Cluster o Sentinel

### Vertical Scaling

- **Modelos**: GPUs más potentes (A100, H100)
- **Databases**: Más RAM y SSD NVMe
- **Workers**: Más CPU cores

## 🔄 CI/CD

### Pipeline

```
Git Push → GitHub Actions
  ↓
1. Linting (flake8, eslint)
  ↓
2. Tests Unitarios
  ↓
3. Tests de Integración
  ↓
4. Build Docker Images
  ↓
5. Push to Container Registry
  ↓
6. Deploy to VMs
  ├─ VM Bounty2
  ├─ VM Services
  └─ VM RAG3
  ↓
7. Health Checks
  ↓
8. Notificación (Slack/Email)
```

## 📚 Documentación Relacionada

- [Plan de Reorganización](PLAN_REORGANIZACION.md)
- [VM Bounty2 README](../vm-bounty2/README.md)
- [VM Services README](../vm-services/README.md)
- [VM RAG3 README](../vm-rag3/README.md)
- [Frontend README](../frontend/README.md)
- [Mejoras VM RAG3](IMPROVEMENTS_VM_RAG3.md)
- [Infrastructure Findings](INFRASTRUCTURE_FINDINGS.md)

## 🎯 Roadmap

### Q1 2025
- [ ] Migrar frontend a React/Vue
- [ ] Implementar auto-scaling
- [ ] Agregar más modelos (Mixtral, Qwen)
- [ ] Mejorar cache de embeddings

### Q2 2025
- [ ] Multi-tenancy
- [ ] API pública con rate limiting
- [ ] Mobile app (React Native)
- [ ] Fine-tuning de modelos propios

### Q3 2025
- [ ] Kubernetes migration
- [ ] Multi-region deployment
- [ ] Advanced analytics
- [ ] Enterprise features

---

**Mantenedor**: Capibara6 Team
**Última actualización**: 2025-11-14
**Versión**: 2.0 (Reorganizado por VMs)
