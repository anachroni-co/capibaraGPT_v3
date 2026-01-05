// Configuración del chatbot capibara6 con GPT-OSS-20B

// Detectar si estamos en localhost o en producción
const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

// URLs de las VMs de Google Cloud
const VM_MODELS = 'http://34.12.166.76';      // VM de modelos (bounty2)
const VM_SERVICES = 'http://34.175.136.104';   // VM de servicios (TTS, MCP, N8N)
const VM_RAG = 'http://10.154.0.2';            // VM rag3 - Sistema RAG completo (IP interna)

const CHATBOT_CONFIG = {
    // URL del backend - cambiar según entorno
    BACKEND_URL: isLocalhost ? 'http://localhost:5001' : VM_MODELS + ':5001',

    // Endpoints
    ENDPOINTS: {
        CHAT: '/api/v1/query',  // Usando el endpoint principal
        CHAT_STREAM: '/api/v1/chat/stream',  // Si está disponible
        SAVE_CONVERSATION: '/api/v1/conversations/save',  // Endpoint actualizado
        HEALTH: '/health',  // Endpoint de salud del backend
        MODELS: '/api/v1/models',
        TTS_SPEAK: '/api/tts/speak',      // Para síntesis de voz
        TTS_VOICES: '/api/tts/voices',    // Para listar voces
        MCP_CONTEXT: '/api/v1/mcp/context',  // Para contexto inteligente
        MCP_STATUS: '/api/v1/mcp/status',  // Endpoint de estado MCP
        E2B_EXECUTE: '/api/v1/e2b/execute'   // Endpoint actualizado para E2B
    },

    // Configuración del modelo GPT-OSS-20B
    MODEL_CONFIG: {
        max_tokens: 100,
        temperature: 0.7,
        model_name: 'gpt-oss-20b',
        timeout: 300000 // 5 minutos como recomienda la documentación
    },

    // Configuración de timeouts
    TIMEOUTS: {
        REQUEST: 30000,      // 30 segundos
        CONNECTION: 5000,    // 5 segundos
        RESPONSE: 120000,    // 2 minutos
        MCP_HEALTH: 5000     // 5 segundos para health check MCP
    },

    // Headers para todas las peticiones
    HEADERS: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Requested-With': 'XMLHttpRequest'
    },

    // Servicios opcionales (configurar si deben usarse o no)
    SERVICES: {
        MCP_ENABLED: false,  // Deshabilitado por defecto - requiere MCP server en puerto 5003
        TTS_ENABLED: true,   // TTS disponible en VM_SERVICES:5002
        E2B_ENABLED: true,   // E2B integrado en backend
        N8N_ENABLED: false,  // N8N requiere VPN/túnel a VM_SERVICES:5678

        // Configuración detallada por servicio
        MCP: {
            enabled: false,
            url: isLocalhost ? 'http://localhost:5003' : VM_SERVICES + ':5003',
            endpoints: {
                AUGMENT: '/api/mcp/augment',
                CONTEXTS: '/api/mcp/contexts',
                HEALTH: '/api/mcp/health',
                CALCULATE: '/api/mcp/calculate',
                VERIFY: '/api/mcp/verify'
            },
            timeout: 5000,
            note: 'MCP principal - Context & RAG'
        },

        TTS: {
            enabled: true,
            url: isLocalhost ? 'http://localhost:5002' : VM_SERVICES + ':5002',
            endpoints: {
                SPEAK: '/tts',
                VOICES: '/voices',
                CLONE: '/clone',
                HEALTH: '/health',
                PRELOAD: '/preload'
            },
            timeout: 10000,
            note: 'Kyutai TTS - Text to Speech'
        },

        AUTH: {
            enabled: true,
            url: isLocalhost ? 'http://localhost:5004' : VM_MODELS + ':5004',
            endpoints: {
                GITHUB: '/auth/github',
                GOOGLE: '/auth/google',
                VERIFY: '/auth/verify',
                LOGOUT: '/auth/logout',
                CALLBACK_GITHUB: '/auth/callback/github',
                CALLBACK_GOOGLE: '/auth/callback/google',
                HEALTH: '/health'
            },
            timeout: 10000,
            note: 'OAuth Authentication - GitHub & Google'
        },

        CONSENSUS: {
            enabled: false,
            url: isLocalhost ? 'http://localhost:5005' : VM_MODELS + ':5005',
            endpoints: {
                QUERY: '/api/consensus/query',
                MODELS: '/api/consensus/models',
                TEMPLATES: '/api/consensus/templates',
                CONFIG: '/api/consensus/config',
                HEALTH: '/api/consensus/health'
            },
            timeout: 30000,
            note: 'Consensus multi-modelo - Combina respuestas de varios modelos'
        },

        SMART_MCP: {
            enabled: false,
            url: isLocalhost ? 'http://localhost:5010' : VM_SERVICES + ':5010',
            endpoints: {
                HEALTH: '/health',
                ANALYZE: '/analyze',
                UPDATE_DATE: '/update-date'
            },
            timeout: 5000,
            note: 'MCP alternativo - RAG selectivo simplificado'
        },

        E2B: {
            enabled: true,
            note: 'E2B integrado en backend principal (puerto 5001)'
        },

        // ========== VM rag3 - Sistema RAG Completo ==========

        RAG3_BRIDGE: {
            enabled: true,
            url: isLocalhost ? 'http://localhost:8000' : VM_RAG + ':8000',
            endpoints: {
                HEALTH: '/health',
                QUERY: '/api/v1/query',
                RAG_SEARCH: '/api/v1/rag/search',
                MILVUS_SEARCH: '/api/v1/milvus/search',
                MILVUS_INSERT: '/api/v1/milvus/insert',
                NEBULA_QUERY: '/api/v1/nebula/query',
                NEBULA_INSERT: '/api/v1/nebula/insert',
                CONTEXT_AUGMENT: '/api/v1/context/augment',
                ANALYTICS: '/api/v1/analytics'
            },
            timeout: 30000,
            note: 'capibara6-api - Bridge principal para RAG (Milvus + Nebula Graph)',
            features: {
                vector_search: true,
                graph_queries: true,
                async_processing: true,
                rq_workers: 3
            }
        },

        MILVUS: {
            enabled: true,
            url: isLocalhost ? 'http://localhost:19530' : VM_RAG + ':19530',
            version: 'v2.3.10',
            config: {
                collection_name: 'capibara6_vectors',
                dimension: 384,  // all-MiniLM-L6-v2 embeddings
                index_type: 'IVF_FLAT',
                metric_type: 'L2',
                nlist: 1024
            },
            search_params: {
                top_k: 10,
                nprobe: 10,
                offset: 0
            },
            timeout: 10000,
            note: 'Milvus Vector Database - Búsqueda semántica y embeddings',
            use_via_bridge: true  // Acceder a través de capibara6-api
        },

        NEBULA_GRAPH: {
            enabled: true,
            url: isLocalhost ? 'http://localhost:9669' : VM_RAG + ':9669',
            version: 'v3.1.0',
            config: {
                space_name: 'capibara6_graph',
                charset: 'utf8',
                collate: 'utf8_bin',
                vid_type: 'FIXED_STRING(32)'
            },
            studio_url: isLocalhost ? 'http://localhost:7001' : VM_RAG + ':7001',
            cluster: {
                metad_nodes: 3,
                storaged_nodes: 3,
                graphd_nodes: 3
            },
            timeout: 15000,
            note: 'Nebula Graph - Knowledge graph para relaciones complejas',
            use_via_bridge: true  // Acceder a través de capibara6-api
        },

        // Bases de datos en VM rag3
        RAG3_POSTGRES: {
            enabled: true,
            host: isLocalhost ? 'localhost' : '10.154.0.2',
            port: 5432,
            database: 'capibara6',
            note: 'PostgreSQL - Base de datos relacional principal'
        },

        RAG3_TIMESCALE: {
            enabled: true,
            host: isLocalhost ? 'localhost' : '10.154.0.2',
            port: 5433,
            database: 'capibara6_timeseries',
            note: 'TimescaleDB - Time-series data (métricas, logs)'
        },

        RAG3_REDIS: {
            enabled: true,
            host: isLocalhost ? 'localhost' : '10.154.0.2',
            port: 6379,
            db: 0,
            note: 'Redis - Cache y queue para RQ workers'
        },

        // Monitoring Stack en VM rag3
        MONITORING: {
            GRAFANA: {
                enabled: true,
                url: isLocalhost ? 'http://localhost:3000' : VM_RAG + ':3000',
                note: 'Grafana - Dashboards de monitoreo'
            },
            PROMETHEUS: {
                enabled: true,
                url: isLocalhost ? 'http://localhost:9090' : VM_RAG + ':9090',
                note: 'Prometheus - Recolección de métricas'
            },
            JAEGER: {
                enabled: true,
                url: isLocalhost ? 'http://localhost:16686' : VM_RAG + ':16686',
                note: 'Jaeger - Distributed tracing'
            }
        }
    },

    // URLs de VMs para servicios externos
    VMS: {
        MODELS: VM_MODELS,          // 34.12.166.76 - VM de modelos (bounty2)
        SERVICES: VM_SERVICES,      // 34.175.136.104 - VM de servicios
        RAG: VM_RAG                 // 10.154.0.2 - VM rag3 (IP interna)
    },

    // Configuración de optimización
    OPTIMIZATION: {
        // Router semántico
        ROUTER: {
            complexity_threshold: 0.7,
            confidence_threshold: 0.6,
            use_embeddings_cache: true,
            cache_ttl: 3600  // 1 hora
        },

        // TOON para optimización de tokens
        TOON: {
            enabled: true,
            auto_detect: true,
            min_sources: 5,  // Activar TOON con 5+ fuentes
            expected_savings: '30-60%'
        },

        // E2B Templates
        E2B_TEMPLATES: {
            default: {
                timeout: 300,
                memory_mb: 512,
                cpu_percent: 50
            },
            data_analysis: {
                timeout: 600,
                memory_mb: 1024,
                cpu_percent: 75,
                packages: ['pandas', 'numpy', 'scipy']
            },
            visualization: {
                timeout: 600,
                memory_mb: 1024,
                cpu_percent: 75,
                packages: ['pandas', 'matplotlib', 'seaborn', 'plotly']
            },
            machine_learning: {
                timeout: 900,
                memory_mb: 2048,
                cpu_percent: 100,
                packages: ['pandas', 'numpy', 'scikit-learn', 'tensorflow']
            }
        },

        // Cache de embeddings
        EMBEDDINGS_CACHE: {
            enabled: true,
            max_size: 1000,
            ttl: 3600,  // 1 hora
            algorithm: 'LRU'  // Least Recently Used
        }
    }
};