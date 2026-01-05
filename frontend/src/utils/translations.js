// Traducciones completas capibara6
const translations = {
    es: {
        // Navegación
        'nav.features': 'Características',
        'nav.architecture': 'Arquitectura',
        'nav.datasets': 'Datasets',
        
        // Hero
        'hero.badge': 'Sistema de IA Conversacional Avanzado',
        'hero.title': 'capibara6',
        'hero.subtitle': 'Arquitectura Híbrida Transformer-Mamba',
        'hero.description': 'Sistema avanzado de IA conversacional con arquitectura híbrida (70% Transformer + 30% Mamba), optimizaciones Google TPU v5e/v6e-64 y Google ARM Axion. La mayor ventana de contexto del mercado. Compliance total para empresas y administraciones públicas.',
        'hero.cta.start': 'Comenzar Ahora',
        'hero.cta.docs': 'Ver Documentación',
        'hero.stats.hybrid': 'Transformer/Mamba',
        'hero.stats.tpu': 'Google TPU',
        'hero.stats.context': 'Contexto Líder',
        'hero.stats.compliance': 'Compliance EU',
        
        // Features
        'features.title': 'Características Principales',
        'features.subtitle': 'Tecnología de vanguardia con arquitectura enterprise-grade',
        
        'feature.moe.title': 'Mixture of Experts (MoE)',
        'feature.moe.desc': '32 expertos especializados con enrutamiento dinámico para dominios como matemáticas, ciencias, código y creatividad.',
        'feature.moe.item1': 'Especialización automática por dominio',
        'feature.moe.item2': 'Balanceamiento de carga inteligente',
        'feature.moe.item3': 'Expert routing adaptativo (96.3% precisión)',
        
        'feature.cot.title': 'Chain-of-Thought Reasoning',
        'feature.cot.desc': 'Razonamiento paso a paso con hasta 12 pasos, meta-cognición avanzada y auto-reflexión para máxima calidad.',
        'feature.cot.item1': 'Razonamiento estructurado verificable',
        'feature.cot.item2': 'Ajuste de confianza automático',
        'feature.cot.item3': 'Process reward models integrados',
        
        'feature.multimodal.title': 'Capacidades Multimodales',
        'feature.multimodal.desc': 'Procesamiento de texto, imágenes y video con encoders especializados y fusión por atención multimodal.',
        'feature.multimodal.item1': 'Vision encoder (224x224, patches 16x16)',
        'feature.multimodal.item2': 'Video encoder (64 frames, 30 FPS)',
        'feature.multimodal.item3': 'Text-to-Speech con contexto emocional',
        
        'feature.tpu.title': 'Google TPU v5e/v6e-64',
        'feature.tpu.desc': 'Kernels optimizados para Google TPU v5e-64 y v6e-64 de última generación con XLA compilation y mixed precision.',
        'feature.tpu.item1': '4,500+ tokens/sec en TPU v6e-64',
        'feature.tpu.item2': 'Flash attention y kernel fusion',
        'feature.tpu.item3': 'Eficiencia energética superior',
        
        'feature.arm.title': 'Google ARM Axion',
        'feature.arm.desc': 'Inferencia optimizada para procesadores Google ARM Axion con NEON, SVE2 vectorization y cuantización avanzada.',
        'feature.arm.item1': '2,100+ tokens/sec (cuantizado 8-bit)',
        'feature.arm.item2': 'Arquitectura ARM de Google Cloud',
        'feature.arm.item3': 'Eficiencia energética excepcional',
        
        'feature.context.title': 'Ventana de Contexto Líder',
        'feature.context.desc': 'Mayor capacidad de contexto del mercado con más de 10M tokens, superando a cualquier competidor actual.',
        'feature.context.item1': '10M+ tokens de contexto real',
        'feature.context.item2': 'Arquitectura híbrida optimizada',
        'feature.context.item3': 'Gestión eficiente de memoria',
        
        'feature.age.title': 'Adaptación por Edad',
        'feature.age.desc': 'Sistema inteligente que adapta contenido, complejidad y tono según la edad del usuario (3-18 años).',
        'feature.age.item1': 'Ajuste automático de vocabulario',
        'feature.age.item2': 'Filtrado de contenido por edad',
        'feature.age.item3': 'Estándares educativos integrados',
        
        'feature.compliance.title': 'Compliance Total UE',
        'feature.compliance.desc': 'Cumplimiento exhaustivo de normativas europeas de privacidad, seguridad, ética y uso legal para sector público y privado.',
        'feature.compliance.item1': 'GDPR, CCPA, AI Act compliance',
        'feature.compliance.item2': 'Certificado para administraciones públicas',
        'feature.compliance.item3': 'Auditorías de seguridad y ética',
        
        'feature.monitoring.title': 'Monitorización Enterprise',
        'feature.monitoring.desc': 'Dashboard completo con métricas TPU, análisis predictivo y alertas automáticas con escalación.',
        'feature.monitoring.item1': 'Métricas en tiempo real (TFLOPS, memoria)',
        'feature.monitoring.item2': 'Exportación Grafana/Prometheus',
        'feature.monitoring.item3': 'Auto-optimización basada en métricas',
        
        // Architecture
        'arch.title': 'Arquitectura del Sistema',
        'arch.subtitle': 'Diseño modular enterprise-grade',
        
        // Quick Start
        'quickstart.title': 'Inicio Rápido',
        'quickstart.subtitle': 'Configura y ejecuta Capibara6 en minutos',
        
        // Scripts
        'scripts.title': 'Scripts Principales',
        'scripts.subtitle': 'Herramientas completas para gestión y operación',
        
        // Config
        'config.title': 'Configuración Flexible',
        'config.subtitle': 'Sistema de configuración basado en YAML',
        
        // Monitoring
        'monitoring.title': 'Monitorización Avanzada',
        'monitoring.subtitle': 'Visibilidad completa del sistema en tiempo real',
        
        // Troubleshooting
        'trouble.title': 'Resolución de Problemas',
        'trouble.subtitle': 'Soluciones a problemas comunes',
        
        // Documentation
        'docs.title': 'Documentación Unificada',
        'docs.subtitle': 'Guías completas y referencias',
        
        // Performance
        'perf.title': 'Rendimiento Enterprise-Grade',
        'perf.subtitle': 'Benchmarks en hardware de producción',
        'perf.comparison.title': 'Comparativa con Modelos Líderes',
        'perf.comp.model': 'Modelo',
        'perf.comp.context': 'Contexto',
        'perf.comp.speed': 'Velocidad',
        'perf.comp.latency': 'Latencia',
        'perf.comp.architecture': 'Arquitectura',
        'perf.comp.multimodal': 'Multimodal',
        
        // CTA
        'cta.title': '¿Listo para comenzar con capibara6?',
        'cta.subtitle': 'Únete a la revolución de IA conversacional con Mixture of Experts y Chain-of-Thought reasoning',
        'cta.button.start': 'Comenzar Ahora',
        'cta.button.github': 'Ver en GitHub',
        
        // Footer
        'footer.description': 'Sistema avanzado de IA conversacional con Mixture of Experts, Chain-of-Thought y capacidades multimodales.',
        'footer.company': 'Anachroni s.coop',
        'footer.country': 'España',
        'footer.product': 'Producto',
        'footer.resources': 'Recursos',
        'footer.community': 'Comunidad',
        'footer.copyright': '© 2025 <strong>Anachroni s.coop</strong> - capibara6.com | Licencia Apache 2.0',
        
        // Chatbot
        'chat.title': 'Asistente capibara6',
        'chat.status': 'En línea',
        'chat.welcome': '¡Hola! Soy el asistente de capibara6. ¿En qué puedo ayudarte?',
        'chat.placeholder': 'Escribe tu pregunta...',
        
        // Chat Page
        'chat.new': 'Nueva Conversación',
        'chat.today': 'Hoy',
        'chat.previous': 'Anteriores',
        'chat.empty.title': '¿En qué puedo ayudarte hoy?',
        'chat.empty.subtitle': 'Soy capibara6, tu asistente de IA avanzado con arquitectura híbrida Transformer-Mamba',
        'chat.suggestion1.title': 'Arquitectura Híbrida',
        'chat.suggestion1.text': 'Explícame cómo funciona',
        'chat.suggestion2.title': 'Google TPU',
        'chat.suggestion2.text': 'Ventajas para entrenamiento',
        'chat.suggestion3.title': 'Programación',
        'chat.suggestion3.text': 'Ayuda con código Python',
        'chat.suggestion4.title': 'Optimización',
        'chat.suggestion4.text': 'Mejorar rendimiento web',
        'chat.share': 'Compartir',
        'chat.settings': 'Configuración',
        'chat.attach': 'Adjuntar archivo',
        'chat.input.placeholder': 'Escribe tu mensaje aquí...',
        'chat.input.hint': 'Capibara6 puede cometer errores. Considera verificar información importante.',
        
        // Settings
        'settings.title': 'Configuración',
        'settings.model': 'Modelo',
        'settings.temperature': 'Temperatura (Creatividad)',
        'settings.language': 'Idioma',
        
        // Quick Start Steps
        'step.1.title': 'Requisitos Previos',
        'step.2.title': 'Configuración',
        'step.3.title': 'Despliegue',
        'step.4.title': 'Entrenamiento',
        
        // Scripts
        'script.master.badge': 'Principal',
        'script.master.desc': 'Interfaz unificada para deploy, train, maintenance, status y setup. Punto de entrada principal del sistema.',
        'script.config.badge': 'Config',
        'script.config.desc': 'Gestión de configuración: init, generate, validate, show y perfiles personalizados.',
        'script.deploy.badge': 'Deploy',
        'script.deploy.desc': 'Despliegue en workers: venv, dependencias, JAX TPU, Cython y pruebas automáticas.',
        'script.sync.badge': 'Sync',
        'script.sync.desc': 'Sincronización de proyecto en todos los workers de forma eficiente y consistente.',
        'script.train.badge': 'Train',
        'script.train.desc': 'Arranque distribuido y monitor básico para procesos de entrenamiento.',
        'script.monitor.badge': 'Monitor',
        'script.monitor.desc': 'Métricas avanzadas: Cython/Mamba/Quant, latencia, memoria y utilización de TPU.',
        'script.cleanup.badge': 'Maint',
        'script.cleanup.desc': 'Limpieza de procesos, logs, cache, checkpoints y mantenimiento del sistema.',
        'script.verify.badge': 'Utils',
        'script.verify.desc': 'Verificación de scripts y mejoras avanzadas para asegurar integridad del sistema.',
        
        // Config section
        'config.feature1': 'Perfiles para desarrollo, staging y producción',
        'config.feature2': 'Validación automática de configuración',
        'config.feature3': 'Generación de .env desde YAML',
        'config.feature4': 'Hot-reload de configuración en desarrollo',
        
        // Monitoring cards
        'monitor.perf.title': '📈 Rendimiento',
        'monitor.perf.desc': 'Monitorea latencia, throughput y utilización de recursos.',
        'monitor.features.title': '🔬 Características',
        'monitor.features.desc': 'Estado de Cython, Mamba SSM, cuantización y kernels.',
        'monitor.report.title': '📊 Reportes',
        'monitor.report.desc': 'Genera reportes completos con métricas y recomendaciones.',
        
        // Troubleshooting
        'trouble.tpu.title': '🔴 TPU no accesible',
        'trouble.tpu.symptom': '<strong>Síntoma:</strong> Error al conectar con TPU',
        'trouble.tpu.solution': '<strong>Solución:</strong>',
        'trouble.jax.title': '⚠️ JAX sin TPU',
        'trouble.jax.symptom': '<strong>Síntoma:</strong> JAX no detecta TPU',
        'trouble.jax.solution': '<strong>Solución:</strong>',
        'trouble.memory.title': '💾 Problemas de Memoria',
        'trouble.memory.symptom': '<strong>Síntoma:</strong> OOM durante entrenamiento',
        'trouble.memory.solution': '<strong>Solución:</strong>',
        'trouble.slow.title': '🐌 Rendimiento Lento',
        'trouble.slow.symptom': '<strong>Síntoma:</strong> Entrenamiento lento',
        'trouble.slow.solution': '<strong>Solución:</strong>',
        
        // Documentation cards
        'doc.meta.title': 'Meta-Consensus y Mamba',
        'doc.meta.file': 'fusion_meta_consensus_mamba.md',
        'doc.operations.title': 'Operación y Scripts',
        'doc.operations.file': 'fusion_operacion_scripts.md',
        'doc.api.title': 'Referencia API',
        'doc.api.desc': 'Documentación completa de la librería',
        'doc.examples.title': 'Ejemplos y Tutoriales',
        'doc.examples.desc': 'Casos de uso y ejemplos prácticos',
        
        // Architecture Layers
        'arch.layer1.title': '🌐 Capa de Entrada Multimodal',
        'arch.layer1.desc': 'Encoders especializados para texto, imagen y video',
        'arch.layer2.title': '🔍 Capa de Recuperación (RAG 2.0)',
        'arch.layer2.desc': 'Contexto de 1M tokens con hybrid search',
        'arch.layer3.title': '🧠 Arquitectura Híbrida',
        'arch.layer3.desc': '70% Transformer + 30% Mamba SSM optimizado',
        'arch.layer4.title': '🔗 Capa de Razonamiento (CoT)',
        'arch.layer4.desc': 'Chain-of-Thought con hasta 12 pasos',
        'arch.layer5.title': '⚡ Capa de Computación',
        'arch.layer5.desc': 'Google TPU v5e/v6e-64 y Google ARM Axion',
        'arch.layer6.title': '🔒 Capa de Compliance',
        'arch.layer6.desc': 'Normativas UE para sector público y privado',
        
        // Performance Labels
        'perf.label.throughput': 'Throughput',
        'perf.label.latency': 'Latencia P95',
        'perf.label.memory': 'Memoria HBM',
        'perf.label.memoryarm': 'Memoria',
        'perf.label.efficiency': 'Eficiencia',
        'perf.label.power': 'Consumo',
        'perf.label.transformer': 'Transformer',
        'perf.label.mamba': 'Mamba SSM',
        'perf.label.context': 'Contexto',
        'perf.label.precision': 'Precisión',
        
        // Buttons
        'button.copy': 'Copiar',
        
        // Datasets
        'datasets.title': 'Datasets Especializados',
        'datasets.subtitle': 'Colección curada de datasets de alta calidad para entrenamiento avanzado',
        
        'dataset.academic.title': 'Datasets Académicos',
        'dataset.academic.purpose': 'Datasets especializados en investigación académica',
        'dataset.academic.item1': 'Datasets institucionales de universidades',
        'dataset.academic.item2': 'Datasets de Wikipedia académica',
        'dataset.academic.item3': 'Código académico y papers',
        'dataset.academic.item4': 'Metadatos de investigación',
        
        'dataset.multimodal.title': 'Datasets Multimodales',
        'dataset.multimodal.purpose': 'Datasets que combinan texto, audio y otros formatos',
        'dataset.multimodal.item1': 'Datasets de audio emocional',
        'dataset.multimodal.item2': 'Análisis de sentimientos multimodal',
        'dataset.multimodal.item3': 'Datasets de conversación',
        
        'dataset.engineering.title': 'Datasets de Ingeniería',
        'dataset.engineering.purpose': 'Datasets especializados en ingeniería y diseño',
        'dataset.engineering.item1': 'Datasets de electrónica',
        'dataset.engineering.item2': 'Datasets de FPGA',
        'dataset.engineering.item3': 'Diseños de circuitos',
        'dataset.engineering.item4': 'Documentación técnica',
        
        'dataset.physics.title': 'Datasets de Física',
        'dataset.physics.purpose': 'Datasets especializados en física teórica y aplicada',
        'dataset.physics.item1': 'Datasets de física cuántica',
        'dataset.physics.item2': 'Simulaciones físicas',
        'dataset.physics.item3': 'Datasets de mecánica clásica',
        'dataset.physics.item4': 'Datasets de física de partículas',
        
        'dataset.robotics.title': 'Datasets de Robótica',
        'dataset.robotics.purpose': 'Datasets para robótica avanzada',
        'dataset.robotics.item1': 'Datasets de control robótico',
        'dataset.robotics.item2': 'Datasets de percepción',
        'dataset.robotics.item3': 'Datasets de planificación de movimiento',
        'dataset.robotics.item4': 'Datasets de interacción humano-robot',
        
        'dataset.mathematics.title': 'Datasets de Matemáticas',
        'dataset.mathematics.purpose': 'Datasets especializados en matemáticas puras y aplicadas',
        'dataset.mathematics.item1': 'Datasets de álgebra',
        'dataset.mathematics.item2': 'Datasets de cálculo',
        'dataset.mathematics.item3': 'Datasets de estadística',
        'dataset.mathematics.item4': 'Datasets de optimización',
        
        'dataset.systems.title': 'Datasets de Sistemas',
        'dataset.systems.purpose': 'Datasets de sistemas operativos y computación',
        'dataset.systems.item1': 'Datasets de Linux kernel',
        'dataset.systems.item2': 'Logs de sistemas',
        'dataset.systems.item3': 'Datasets de administración de sistemas',
        'dataset.systems.item4': 'Datasets de seguridad',
        
        'dataset.spanish.title': 'Comunidad Española',
        'dataset.spanish.purpose': 'Datasets específicos para la comunidad hispanohablante',
        'dataset.spanish.item1': 'Datasets de NLP en español',
        'dataset.spanish.item2': 'Datasets de literatura española',
        'dataset.spanish.item3': 'Datasets de medios en español',
        
        // Component Status
        'status.title': 'Estado de Componentes',
        'status.subtitle': 'Sistema completamente operativo y optimizado',
        'status.components.title': 'Componentes del Sistema',
        'status.table.component': 'Componente',
        'status.table.version': 'Versión/Capacidad',
        'status.operational': 'Operativo',
        'status.configured': 'Configurado',
        'status.compiled': 'Compilados',
        'status.integrated': 'Integrado',
        'status.active': 'activo',
        'status.complete': 'completo',
        'status.ready': 'Listos',
        'status.samples': 'muestras',
        'status.system.complete': 'Sistema completo',
        'status.agent.system': 'Sistema Agentes',
        
        // Technical Capabilities
        'status.cap.performance': 'Performance',
        'status.cap.perf1': 'aceleración con Cython kernels',
        'status.cap.perf2': 'reducción de memoria con cuantización INT8',
        'status.cap.perf3': 'mejora teórica combinada',
        'status.cap.scalability': 'Escalabilidad',
        'status.cap.scale1': 'Entrenamiento distribuido multi-worker',
        'status.cap.scale2': 'Consenso federado Byzantine fault-tolerant',
        'status.cap.scale3': 'Soporte TPU/ARM/CUDA',
        'status.cap.modularity': 'Modularidad',
        'status.cap.mod1': 'Sistema de configuración TOML completo',
        'status.cap.mod2': 'Factory pattern para agentes',
        'status.cap.mod3': 'Strategy pattern para orquestación',
        'status.cap.mod4': 'Adapter pattern para hardware',
        
        // Footer links
        'footer.guides': 'Guías de Usuario',
        'footer.api': 'API Reference',
        'footer.usecases': 'Casos de Uso',
        'footer.benchmarks': 'Benchmarks',
        'footer.github': 'GitHub',
        'footer.linkedin': 'LinkedIn',
        'footer.discord': 'Discord',
        'footer.twitter': 'Twitter',
        'footer.privacy': 'Privacidad',
        'footer.terms': 'Términos',
        'footer.license': 'Licencia'
    },
    en: {
        // Navigation
        'nav.features': 'Features',
        'nav.architecture': 'Architecture',
        'nav.datasets': 'Datasets',
        
        // Hero
        'hero.badge': 'Advanced Conversational AI System',
        'hero.title': 'capibara6',
        'hero.subtitle': 'Hybrid Transformer-Mamba Architecture',
        'hero.description': 'Advanced conversational AI system with hybrid architecture (70% Transformer + 30% Mamba), Google TPU v5e/v6e-64 and Google ARM Axion optimizations. Largest context window in the market. Full compliance for enterprises and public administrations.',
        'hero.cta.start': 'Get Started',
        'hero.cta.docs': 'View Documentation',
        'hero.stats.hybrid': 'Transformer/Mamba',
        'hero.stats.tpu': 'Google TPU',
        'hero.stats.context': 'Leading Context',
        'hero.stats.compliance': 'EU Compliance',
        
        // Features
        'features.title': 'Key Features',
        'features.subtitle': 'Cutting-edge technology with enterprise-grade architecture',
        
        'feature.moe.title': 'Mixture of Experts (MoE)',
        'feature.moe.desc': '32 specialized experts with dynamic routing for domains like mathematics, science, code and creativity.',
        'feature.moe.item1': 'Automatic domain specialization',
        'feature.moe.item2': 'Intelligent load balancing',
        'feature.moe.item3': 'Adaptive expert routing (96.3% accuracy)',
        
        'feature.cot.title': 'Chain-of-Thought Reasoning',
        'feature.cot.desc': 'Step-by-step reasoning with up to 12 steps, advanced meta-cognition and self-reflection for maximum quality.',
        'feature.cot.item1': 'Verifiable structured reasoning',
        'feature.cot.item2': 'Automatic confidence adjustment',
        'feature.cot.item3': 'Integrated process reward models',
        
        'feature.multimodal.title': 'Multimodal Capabilities',
        'feature.multimodal.desc': 'Text, image and video processing with specialized encoders and multimodal attention fusion.',
        'feature.multimodal.item1': 'Vision encoder (224x224, 16x16 patches)',
        'feature.multimodal.item2': 'Video encoder (64 frames, 30 FPS)',
        'feature.multimodal.item3': 'Text-to-Speech with emotional context',
        
        'feature.tpu.title': 'Google TPU v5e/v6e-64',
        'feature.tpu.desc': 'Optimized kernels for latest generation Google TPU v5e-64 and v6e-64 with XLA compilation and mixed precision.',
        'feature.tpu.item1': '4,500+ tokens/sec on TPU v6e-64',
        'feature.tpu.item2': 'Flash attention and kernel fusion',
        'feature.tpu.item3': 'Superior energy efficiency',
        
        'feature.arm.title': 'Google ARM Axion',
        'feature.arm.desc': 'Optimized inference for Google ARM Axion processors with NEON, SVE2 vectorization and advanced quantization.',
        'feature.arm.item1': '2,100+ tokens/sec (8-bit quantized)',
        'feature.arm.item2': 'Google Cloud ARM architecture',
        'feature.arm.item3': 'Exceptional energy efficiency',
        
        'feature.context.title': 'Leading Context Window',
        'feature.context.desc': 'Largest context capacity in the market with over 10M tokens, surpassing any current competitor.',
        'feature.context.item1': '10M+ real context tokens',
        'feature.context.item2': 'Optimized hybrid architecture',
        'feature.context.item3': 'Efficient memory management',
        
        'feature.age.title': 'Age Adaptation',
        'feature.age.desc': 'Intelligent system that adapts content, complexity and tone according to user age (3-18 years).',
        'feature.age.item1': 'Automatic vocabulary adjustment',
        'feature.age.item2': 'Age-based content filtering',
        'feature.age.item3': 'Integrated educational standards',
        
        'feature.compliance.title': 'Full EU Compliance',
        'feature.compliance.desc': 'Comprehensive compliance with European privacy, security, ethics and legal use regulations for public and private sector.',
        'feature.compliance.item1': 'GDPR, CCPA, AI Act compliance',
        'feature.compliance.item2': 'Certified for public administrations',
        'feature.compliance.item3': 'Security and ethics audits',
        
        'feature.monitoring.title': 'Enterprise Monitoring',
        'feature.monitoring.desc': 'Complete dashboard with TPU metrics, predictive analysis and automatic alerts with escalation.',
        'feature.monitoring.item1': 'Real-time metrics (TFLOPS, memory)',
        'feature.monitoring.item2': 'Grafana/Prometheus export',
        'feature.monitoring.item3': 'Metrics-based auto-optimization',
        
        // Architecture
        'arch.title': 'System Architecture',
        'arch.subtitle': 'Enterprise-grade modular design',
        
        // Quick Start
        'quickstart.title': 'Quick Start',
        'quickstart.subtitle': 'Set up and run Capibara6 in minutes',
        
        // Scripts
        'scripts.title': 'Main Scripts',
        'scripts.subtitle': 'Complete tools for management and operations',
        
        // Config
        'config.title': 'Flexible Configuration',
        'config.subtitle': 'YAML-based configuration system',
        
        // Monitoring
        'monitoring.title': 'Advanced Monitoring',
        'monitoring.subtitle': 'Full system visibility in real-time',
        
        // Troubleshooting
        'trouble.title': 'Troubleshooting',
        'trouble.subtitle': 'Solutions to common problems',
        
        // Documentation
        'docs.title': 'Unified Documentation',
        'docs.subtitle': 'Complete guides and references',
        
        // Performance
        'perf.title': 'Enterprise-Grade Performance',
        'perf.subtitle': 'Benchmarks on production hardware',
        'perf.comparison.title': 'Comparison with Leading Models',
        'perf.comp.model': 'Model',
        'perf.comp.context': 'Context',
        'perf.comp.speed': 'Speed',
        'perf.comp.latency': 'Latency',
        'perf.comp.architecture': 'Architecture',
        'perf.comp.multimodal': 'Multimodal',
        
        // CTA
        'cta.title': 'Ready to start with capibara6?',
        'cta.subtitle': 'Join the conversational AI revolution with Mixture of Experts and Chain-of-Thought reasoning',
        'cta.button.start': 'Get Started',
        'cta.button.github': 'View on GitHub',
        
        // Footer
        'footer.description': 'Advanced conversational AI system with Mixture of Experts, Chain-of-Thought and multimodal capabilities.',
        'footer.company': 'Anachroni s.coop',
        'footer.country': 'Spain',
        'footer.product': 'Product',
        'footer.resources': 'Resources',
        'footer.community': 'Community',
        'footer.copyright': '© 2025 <strong>Anachroni s.coop</strong> - capibara6.com | Apache 2.0 License',
        
        // Chatbot
        'chat.title': 'capibara6 Assistant',
        'chat.status': 'Online',
        'chat.welcome': 'Hello! I\'m the capibara6 assistant. How can I help you?',
        'chat.placeholder': 'Type your question...',
        
        // Chat Page
        'chat.new': 'New Conversation',
        'chat.today': 'Today',
        'chat.previous': 'Previous',
        'chat.empty.title': 'How can I help you today?',
        'chat.empty.subtitle': 'I\'m capibara6, your advanced AI assistant with hybrid Transformer-Mamba architecture',
        'chat.suggestion1.title': 'Hybrid Architecture',
        'chat.suggestion1.text': 'Explain how it works',
        'chat.suggestion2.title': 'Google TPU',
        'chat.suggestion2.text': 'Training advantages',
        'chat.suggestion3.title': 'Programming',
        'chat.suggestion3.text': 'Help with Python code',
        'chat.suggestion4.title': 'Optimization',
        'chat.suggestion4.text': 'Improve web performance',
        'chat.share': 'Share',
        'chat.settings': 'Settings',
        'chat.attach': 'Attach file',
        'chat.input.placeholder': 'Type your message here...',
        'chat.input.hint': 'Capibara6 can make mistakes. Consider verifying important information.',
        
        // Settings
        'settings.title': 'Settings',
        'settings.model': 'Model',
        'settings.temperature': 'Temperature (Creativity)',
        'settings.language': 'Language',
        
        // Quick Start Steps
        'step.1.title': 'Prerequisites',
        'step.2.title': 'Configuration',
        'step.3.title': 'Deployment',
        'step.4.title': 'Training',
        
        // Scripts
        'script.master.badge': 'Main',
        'script.master.desc': 'Unified interface for deploy, train, maintenance, status and setup. Main system entry point.',
        'script.config.badge': 'Config',
        'script.config.desc': 'Configuration management: init, generate, validate, show and custom profiles.',
        'script.deploy.badge': 'Deploy',
        'script.deploy.desc': 'Worker deployment: venv, dependencies, JAX TPU, Cython and automated tests.',
        'script.sync.badge': 'Sync',
        'script.sync.desc': 'Efficient and consistent project synchronization across all workers.',
        'script.train.badge': 'Train',
        'script.train.desc': 'Distributed startup and basic monitoring for training processes.',
        'script.monitor.badge': 'Monitor',
        'script.monitor.desc': 'Advanced metrics: Cython/Mamba/Quant, latency, memory and TPU utilization.',
        'script.cleanup.badge': 'Maint',
        'script.cleanup.desc': 'Process, logs, cache, checkpoints cleanup and system maintenance.',
        'script.verify.badge': 'Utils',
        'script.verify.desc': 'Script verification and advanced improvements to ensure system integrity.',
        
        // Config section
        'config.feature1': 'Profiles for development, staging and production',
        'config.feature2': 'Automatic configuration validation',
        'config.feature3': '.env generation from YAML',
        'config.feature4': 'Configuration hot-reload in development',
        
        // Monitoring cards
        'monitor.perf.title': '📈 Performance',
        'monitor.perf.desc': 'Monitor latency, throughput and resource utilization.',
        'monitor.features.title': '🔬 Features',
        'monitor.features.desc': 'Status of Cython, Mamba SSM, quantization and kernels.',
        'monitor.report.title': '📊 Reports',
        'monitor.report.desc': 'Generate complete reports with metrics and recommendations.',
        
        // Troubleshooting
        'trouble.tpu.title': '🔴 TPU not accessible',
        'trouble.tpu.symptom': '<strong>Symptom:</strong> Error connecting to TPU',
        'trouble.tpu.solution': '<strong>Solution:</strong>',
        'trouble.jax.title': '⚠️ JAX without TPU',
        'trouble.jax.symptom': '<strong>Symptom:</strong> JAX doesn\'t detect TPU',
        'trouble.jax.solution': '<strong>Solution:</strong>',
        'trouble.memory.title': '💾 Memory Problems',
        'trouble.memory.symptom': '<strong>Symptom:</strong> OOM during training',
        'trouble.memory.solution': '<strong>Solution:</strong>',
        'trouble.slow.title': '🐌 Slow Performance',
        'trouble.slow.symptom': '<strong>Symptom:</strong> Slow training',
        'trouble.slow.solution': '<strong>Solution:</strong>',
        
        // Documentation cards
        'doc.meta.title': 'Meta-Consensus and Mamba',
        'doc.meta.file': 'fusion_meta_consensus_mamba.md',
        'doc.operations.title': 'Operations and Scripts',
        'doc.operations.file': 'fusion_operacion_scripts.md',
        'doc.api.title': 'API Reference',
        'doc.api.desc': 'Complete library documentation',
        'doc.examples.title': 'Examples and Tutorials',
        'doc.examples.desc': 'Use cases and practical examples',
        
        // Architecture Layers
        'arch.layer1.title': '🌐 Multimodal Input Layer',
        'arch.layer1.desc': 'Specialized encoders for text, image and video',
        'arch.layer2.title': '🔍 Retrieval Layer (RAG 2.0)',
        'arch.layer2.desc': '1M tokens context with hybrid search',
        'arch.layer3.title': '🧠 Hybrid Architecture',
        'arch.layer3.desc': '70% Transformer + 30% Mamba SSM optimized',
        'arch.layer4.title': '🔗 Reasoning Layer (CoT)',
        'arch.layer4.desc': 'Chain-of-Thought with up to 12 steps',
        'arch.layer5.title': '⚡ Computation Layer',
        'arch.layer5.desc': 'Google TPU v5e/v6e-64 and Google ARM Axion',
        'arch.layer6.title': '🔒 Compliance Layer',
        'arch.layer6.desc': 'EU regulations for public and private sector',
        
        // Performance Labels
        'perf.label.throughput': 'Throughput',
        'perf.label.latency': 'Latency P95',
        'perf.label.memory': 'HBM Memory',
        'perf.label.memoryarm': 'Memory',
        'perf.label.efficiency': 'Efficiency',
        'perf.label.power': 'Power',
        'perf.label.transformer': 'Transformer',
        'perf.label.mamba': 'Mamba SSM',
        'perf.label.context': 'Context',
        'perf.label.precision': 'Accuracy',
        
        // Buttons
        'button.copy': 'Copy',
        
        // Datasets
        'datasets.title': 'Specialized Datasets',
        'datasets.subtitle': 'Curated collection of high-quality datasets for advanced training',
        
        'dataset.academic.title': 'Academic Datasets',
        'dataset.academic.purpose': 'Specialized datasets for academic research',
        'dataset.academic.item1': 'University institutional datasets',
        'dataset.academic.item2': 'Academic Wikipedia datasets',
        'dataset.academic.item3': 'Academic code and papers',
        'dataset.academic.item4': 'Research metadata',
        
        'dataset.multimodal.title': 'Multimodal Datasets',
        'dataset.multimodal.purpose': 'Datasets combining text, audio and other formats',
        'dataset.multimodal.item1': 'Emotional audio datasets',
        'dataset.multimodal.item2': 'Multimodal sentiment analysis',
        'dataset.multimodal.item3': 'Conversation datasets',
        
        'dataset.engineering.title': 'Engineering Datasets',
        'dataset.engineering.purpose': 'Specialized datasets for engineering and design',
        'dataset.engineering.item1': 'Electronics datasets',
        'dataset.engineering.item2': 'FPGA datasets',
        'dataset.engineering.item3': 'Circuit designs',
        'dataset.engineering.item4': 'Technical documentation',
        
        'dataset.physics.title': 'Physics Datasets',
        'dataset.physics.purpose': 'Specialized datasets for theoretical and applied physics',
        'dataset.physics.item1': 'Quantum physics datasets',
        'dataset.physics.item2': 'Physical simulations',
        'dataset.physics.item3': 'Classical mechanics datasets',
        'dataset.physics.item4': 'Particle physics datasets',
        
        'dataset.robotics.title': 'Robotics Datasets',
        'dataset.robotics.purpose': 'Datasets for advanced robotics',
        'dataset.robotics.item1': 'Robotic control datasets',
        'dataset.robotics.item2': 'Perception datasets',
        'dataset.robotics.item3': 'Motion planning datasets',
        'dataset.robotics.item4': 'Human-robot interaction datasets',
        
        'dataset.mathematics.title': 'Mathematics Datasets',
        'dataset.mathematics.purpose': 'Specialized datasets for pure and applied mathematics',
        'dataset.mathematics.item1': 'Algebra datasets',
        'dataset.mathematics.item2': 'Calculus datasets',
        'dataset.mathematics.item3': 'Statistics datasets',
        'dataset.mathematics.item4': 'Optimization datasets',
        
        'dataset.systems.title': 'Systems Datasets',
        'dataset.systems.purpose': 'Operating systems and computing datasets',
        'dataset.systems.item1': 'Linux kernel datasets',
        'dataset.systems.item2': 'System logs',
        'dataset.systems.item3': 'System administration datasets',
        'dataset.systems.item4': 'Security datasets',
        
        'dataset.spanish.title': 'Spanish Community',
        'dataset.spanish.purpose': 'Specific datasets for the Spanish-speaking community',
        'dataset.spanish.item1': 'Spanish NLP datasets',
        'dataset.spanish.item2': 'Spanish literature datasets',
        'dataset.spanish.item3': 'Spanish media datasets',
        
        // Component Status
        'status.title': 'Component Status',
        'status.subtitle': 'Fully operational and optimized system',
        'status.components.title': 'System Components',
        'status.table.component': 'Component',
        'status.table.version': 'Version/Capacity',
        'status.operational': 'Operational',
        'status.configured': 'Configured',
        'status.compiled': 'Compiled',
        'status.integrated': 'Integrated',
        'status.active': 'active',
        'status.complete': 'complete',
        'status.ready': 'Ready',
        'status.samples': 'samples',
        'status.system.complete': 'Complete system',
        'status.agent.system': 'Agent System',
        
        // Technical Capabilities
        'status.cap.performance': 'Performance',
        'status.cap.perf1': 'acceleration with Cython kernels',
        'status.cap.perf2': 'memory reduction with INT8 quantization',
        'status.cap.perf3': 'combined theoretical improvement',
        'status.cap.scalability': 'Scalability',
        'status.cap.scale1': 'Multi-worker distributed training',
        'status.cap.scale2': 'Byzantine fault-tolerant federated consensus',
        'status.cap.scale3': 'TPU/ARM/CUDA support',
        'status.cap.modularity': 'Modularity',
        'status.cap.mod1': 'Complete TOML configuration system',
        'status.cap.mod2': 'Factory pattern for agents',
        'status.cap.mod3': 'Strategy pattern for orchestration',
        'status.cap.mod4': 'Adapter pattern for hardware',
        
        // Footer links
        'footer.guides': 'User Guides',
        'footer.api': 'API Reference',
        'footer.usecases': 'Use Cases',
        'footer.benchmarks': 'Benchmarks',
        'footer.github': 'GitHub',
        'footer.linkedin': 'LinkedIn',
        'footer.discord': 'Discord',
        'footer.twitter': 'Twitter',
        'footer.privacy': 'Privacy',
        'footer.terms': 'Terms',
        'footer.license': 'License'
    }
};

// Exportar para uso en script.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = translations;
}

