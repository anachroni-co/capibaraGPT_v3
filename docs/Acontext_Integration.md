# Integración Acontext para Capibara6

## Descripción General

Acontext es una plataforma de datos de contexto para aplicaciones de IA con capacidad de autoaprendizaje. En Capibara6, hemos integrado Acontext para proporcionar **persitencia de contexto adaptativo** y **aprendizaje de experiencias** entre conversaciones.

## Objetivos

- ✅ **Persistencia de contexto**: Almacenar conversaciones para recuperarlas en futuras interacciones
- ✅ **Aprendizaje de experiencias**: Identificar patrones y mejores prácticas de conversaciones anteriores
- ✅ **Automejora**: Permitir que el sistema mejore con cada interacción
- ✅ **Búsqueda de conocimiento**: Recuperar información relevante de sesiones anteriores

## Arquitectura

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │◄──►│  Gateway Server  │◄──►│  Acontext Mock  │
│   (Chat UI)     │    │   (Puerto 8001)  │    │   (Puerto 8029) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                        │
         │              ┌──────────────────┐              │
         └─────────────►│  Acontext Proxy  │◄─────────────┘
                        │ (API Endpoints)  │
                        └──────────────────┘
```

### Componentes

1. **Servidor Acontext Mock** (`acontext_mock_server.py`)
   - API simulada con funcionalidades básicas de Acontext
   - Almacenamiento en memoria para desarrollo
   - Simula búsquedas de experiencias relevantes

2. **Gateway Server** (`backend/gateway_server.py`)
   - Proxy para todos los endpoints de Acontext
   - Gestión de sesiones y almacenamiento automático
   - Integración con el flujo de chat existente

3. **Frontend Integration** (`frontend/src/chat-app.js`)
   - Indicador visual de estado de Acontext
   - Creación automática de sesiones
   - Almacenamiento de mensajes en Acontext
   - Interfaz de usuario para gestionar contextos

## Funcionalidades

### 1. Gestión de Sesiones

- Creación automática de sesiones Acontext para cada conversación
- Almacenamiento de mensajes de usuario y asistente
- Vinculación de sesiones a espacios de conocimiento

### 2. Almacenamiento de Contexto

- Cada mensaje se envía automáticamente a Acontext
- Separación por sesiones para mantener contexto
- Persistencia entre recargas de la página

### 3. Búsqueda de Experiencias

- Búsqueda contextual en conversaciones anteriores
- Recomendaciones basadas en consultas similares
- Aprendizaje de patrones de interacción

### 4. Espacios de Conocimiento

- Organización de experiencias por temas
- Creación de espacios personalizados
- Búsqueda dentro de espacios específicos

## Endpoints de API

### Gateway Server Proxy
- `GET /api/acontext/status` - Estado del sistema
- `POST /api/acontext/session/create` - Crear sesión
- `POST /api/acontext/session/{id}/messages` - Almacenar mensaje
- `POST /api/acontext/space/create` - Crear espacio
- `GET /api/acontext/{path:path}` - Proxy general para otros endpoints

### Frontend Integration
- Función `checkAcontextStatus()` - Verificar disponibilidad
- Función `createAcontextSession()` - Crear sesión nueva
- Función `sendToAcontext()` - Enviar mensaje a Acontext
- Función `initAcontext()` - Inicializar integración

## Configuración

### Variables de Entorno

```
# En backend/.env
ACONTEXT_ENABLED=true
ACONTEXT_BASE_URL=http://localhost:8029/api/v1
ACONTEXT_API_KEY=sk-ac-your-root-api-bearer-token
ACONTEXT_PROJECT_ID=capibara6-project
ACONTEXT_SPACE_ID=capibara6-space
```

### Frontend Configuration

```javascript
// En frontend/src/config.js
ACONTEXT: {
    BASE_URL: isLocalhost ? 'http://localhost:8001/api' : 'https://www.capibara6.com/api',
    STATUS: '/acontext/status',
    SESSION_CREATE: '/acontext/session/create',
    SPACE_CREATE: '/acontext/space/create',
    SEARCH: '/acontext/search',
    HEALTH: '/acontext/health'
}
```

## Instalación y Despliegue

### Desarrollo

1. Iniciar el servidor Acontext Mock:
   ```bash
   python acontext_mock_server.py
   ```

2. Iniciar el Gateway Server:
   ```bash
   cd backend
   uvicorn gateway_server:app --host 0.0.0.0 --port 8001
   ```

3. El frontend se conectará automáticamente a ambos servicios

### Producción

Para producción, se recomienda:

1. Desplegar Acontext con PostgreSQL y Redis completos en lugar del mock
2. Actualizar la configuración para usar los endpoints de producción
3. Configurar balanceadores de carga y alta disponibilidad

## Seguridad

- Validación de tokens de autenticación
- Rate limiting para prevenir abusos
- Validación de entradas de usuario
- Protección contra inyecciones de código

## Consideraciones para Desarrollo Futuro

1. **Reemplazo del Mock**: Para producción, sustituir el servidor mock por una instalación completa de Acontext con PostgreSQL y Redis

2. **Mejora de Búsqueda**: Implementar búsqueda semántica real en lugar de simulación

3. **Autenticación**: Añadir soporte para tokens de usuario y control de acceso

4. **Escalabilidad**: Implementar sistema de colas para alta concurrencia

## Estilos CSS

El indicador de Acontext en el sidebar usa estilos gradientes para facilitar la identificación:

```css
.acontext-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    margin: 8px 16px;
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    border-radius: 8px;
    color: white;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.3s ease;
    border: 1px solid rgba(255, 255, 255, 0.2);
}
```

## Monitoreo

- Logging de todas las operaciones de Acontext
- Verificación de estado automática
- Métricas de uso y rendimiento
- Sistema de alertas para fallos

## Pruebas

La integración ha sido probada con:

- Pruebas unitarias de módulos de integración
- Pruebas de extremo a extremo del flujo de mensajes
- Pruebas de rendimiento bajo carga simulada
- Compatibilidad con diferentes navegadores