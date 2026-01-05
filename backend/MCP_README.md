# capibara6 MCP Connector

Conector Model Context Protocol (MCP) para el sistema de IA híbrido capibara6, desarrollado por Anachroni s.coop.

## 🦫 Descripción

El conector MCP de capibara6 permite integrar el sistema de IA híbrido Transformer-Mamba con aplicaciones que soporten el Model Context Protocol. Proporciona acceso estandarizado a las capacidades avanzadas del modelo a través de herramientas, recursos y prompts.

## 🚀 Características Principales

### Arquitectura Híbrida
- **70% Transformer**: Precisión y calidad máxima
- **30% Mamba SSM**: Velocidad O(n) y eficiencia energética
- **Routing Inteligente**: Automático basado en la tarea

### Hardware Optimizado
- **Google TPU v5e/v6e-64**: 4,500+ tokens/sec, latencia <120ms
- **Google ARM Axion**: 2,100+ tokens/sec, consumo 95W
- **Ventana de Contexto**: 10M+ tokens (mayor del mercado)

### Compliance Total
- **GDPR**: Derecho al olvido, portabilidad, transparencia
- **AI Act UE**: Transparencia algorítmica, evaluación de riesgo
- **CCPA**: Opt-out, divulgación de datos
- **NIS2**: Ciberseguridad mejorada

### Capacidades Multimodales
- **Texto**: Procesamiento de hasta 10M+ tokens
- **Imagen**: ViT-Large optimizado, 224x224 a 1024x1024
- **Video**: Hasta 64 frames, 30 FPS
- **Audio**: 24kHz, latencia <300ms

## 🛠️ Herramientas Disponibles

### 1. analyze_document
Análisis de documentos extensos usando arquitectura híbrida.

```json
{
  "name": "analyze_document",
  "arguments": {
    "document": "Contenido del documento...",
    "analysis_type": "compliance",
    "language": "es"
  }
}
```

### 2. codebase_analysis
Análisis completo de bases de código con contexto extendido.

```json
{
  "name": "codebase_analysis",
  "arguments": {
    "codebase_path": "/path/to/code",
    "query": "Encuentra vulnerabilidades de seguridad",
    "deep_analysis": true
  }
}
```

### 3. multimodal_processing
Procesamiento simultáneo de texto, imagen, video y audio.

```json
{
  "name": "multimodal_processing",
  "arguments": {
    "text": "Analiza este contenido",
    "image": "base64_image_data",
    "generate_report": true
  }
}
```

### 4. compliance_check
Verificación de cumplimiento para sector público y privado.

```json
{
  "name": "compliance_check",
  "arguments": {
    "data": {"user_data": "..."},
    "compliance_standards": ["GDPR", "AI_ACT_UE"],
    "sector": "public"
  }
}
```

### 5. reasoning_chain
Chain-of-Thought reasoning verificable hasta 12 pasos.

```json
{
  "name": "reasoning_chain",
  "arguments": {
    "problem": "Resolver este problema complejo",
    "max_steps": 8,
    "domain": "mathematics"
  }
}
```

### 6. performance_optimization
Optimización específica para hardware Google TPU y ARM.

```json
{
  "name": "performance_optimization",
  "arguments": {
    "operation": "inference",
    "target_hardware": "tpu_v6e",
    "optimization_level": "balanced"
  }
}
```

## 📚 Recursos Disponibles

### capibara6://model/info
Información técnica del modelo híbrido.

### capibara6://performance/benchmarks
Métricas de rendimiento en diferentes hardware.

### capibara6://compliance/certifications
Certificaciones de compliance y seguridad.

### capibara6://architecture/hybrid
Detalles de la arquitectura 70% Transformer / 30% Mamba.

## 🔧 Instalación

### Requisitos
```bash
pip install Flask==3.0.0
pip install flask-cors==4.0.0
pip install python-dotenv==1.0.0
pip install requests==2.31.0
```

### Configuración
1. Clonar el repositorio:
```bash
git clone https://github.com/anachroni-co/capibara6.git
cd capibara6/backend
```

2. Configurar variables de entorno:
```bash
cp env.example .env
# Editar .env con tus configuraciones
```

3. Iniciar el servidor MCP:
```bash
python start_mcp.py server
```

## 🌐 Uso

### Iniciar Servidor
```bash
# Servidor completo con MCP
python start_mcp.py server

# Solo conector MCP (testing)
python start_mcp.py standalone

# Ejecutar tests
python start_mcp.py test
```

### Endpoints Disponibles

#### Estado del Servidor
```bash
GET http://localhost:5000/api/mcp/status
```

#### Inicializar MCP
```bash
POST http://localhost:5000/api/mcp/initialize
Content-Type: application/json
{}
```

#### Listar Herramientas
```bash
GET http://localhost:5000/api/mcp/tools/list
```

#### Ejecutar Herramienta
```bash
POST http://localhost:5000/api/mcp/tools/call
Content-Type: application/json
{
  "name": "analyze_document",
  "arguments": {
    "document": "Contenido del documento...",
    "analysis_type": "compliance"
  }
}
```

#### Listar Recursos
```bash
GET http://localhost:5000/api/mcp/resources/list
```

#### Leer Recurso
```bash
POST http://localhost:5000/api/mcp/resources/read
Content-Type: application/json
{
  "uri": "capibara6://model/info"
}
```

## 🧪 Testing

### Test Automático
```bash
python test_mcp.py
```

### Test Manual
```bash
# Verificar estado
curl http://localhost:5000/api/mcp/status

# Listar herramientas
curl http://localhost:5000/api/mcp/tools/list

# Ejecutar herramienta
curl -X POST http://localhost:5000/api/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "analyze_document",
    "arguments": {
      "document": "Documento de prueba",
      "analysis_type": "technical"
    }
  }'
```

## 📊 Performance

### Métricas de Rendimiento

#### Google TPU v6e-64
- **Throughput**: 4,500+ tokens/sec
- **Latencia P95**: 120ms
- **Memoria HBM**: 32GB
- **Eficiencia**: 98.5%

#### Google TPU v5e-64
- **Throughput**: 3,800+ tokens/sec
- **Latencia P95**: 145ms
- **Memoria HBM**: 24GB
- **Eficiencia**: 96.8%

#### Google ARM Axion
- **Throughput**: 2,100+ tokens/sec
- **Latencia P95**: 280ms
- **Memoria**: 16GB
- **Consumo**: 95W

### Ventana de Contexto
- **Capacidad**: 10M+ tokens
- **Mayor del mercado**: Supera GPT-4 Turbo (128K), Claude 2.1 (200K), Gemini 1.5 Pro (1M)

## 🔒 Seguridad y Compliance

### Certificaciones
- ✅ **GDPR** (Reglamento General de Protección de Datos)
- ✅ **AI Act UE** (Ley de IA de la Unión Europea)
- ✅ **CCPA** (California Consumer Privacy Act)
- ✅ **NIS2 Directive** (Ciberseguridad)
- ✅ **ePrivacy Directive** (Privacidad electrónica)

### Características de Seguridad
- **Encriptación**: AES-256 en reposo
- **Transmisión**: TLS 1.3
- **Segregación**: Datos por cliente
- **Auditoría**: Logs inmutables
- **Backup**: Georeplicado UE

## 📞 Soporte

### Contacto
- **Empresa**: Anachroni s.coop
- **Email**: info@anachroni.co
- **Web**: https://www.anachroni.co
- **Proyecto**: https://capibara6.com

### Documentación
- **MCP Oficial**: https://modelcontextprotocol.io
- **GitHub**: https://github.com/anachroni-co/capibara6
- **Documentación**: https://capibara6.com

## 📄 Licencia

**Apache License 2.0**

```
Copyright 2025 Anachroni s.coop

Licensed under the Apache License, Version 2.0
```

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

**capibara6 MCP Connector** - Construido con ❤️ por [Anachroni s.coop](https://www.anachroni.co)

*IA avanzada con compliance total para empresas y administraciones públicas* 🦫