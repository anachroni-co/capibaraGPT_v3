# Integración de n8n con capibara6-consensus

## 📋 Resumen

n8n ha sido instalado correctamente como parte del sistema **capibara6-consensus** en esta VM de Google Cloud ARM (Axion).

## 🎯 Información del Sistema

### Servicios Activos
- **llama-server.service** - GPT-OSS-20B (puerto 8080)
- **flask-app.service** - API Flask con endpoints MCP (puerto 5000)
- **coqui-tts.service** - Servidor TTS
- **n8n.service** - Automatización de workflows (puerto 5678)
- **nginx** - Reverse proxy

### Arquitectura capibara6-consensus
- Modelos de IA: gpt-oss-20b, mixtral, phi3-mini
- Enrutador de consenso entre modelos
- Servidor MCP (Model Context Protocol)
- Sistema TTS (Text-to-Speech)
- **NUEVO**: n8n para automatización de workflows

## 🔧 Detalles de Instalación

### Ubicación y Configuración
- **Binario n8n**: `/usr/bin/n8n`
- **Versión**: 1.118.2
- **Datos de usuario**: `/home/elect/.n8n/`
- **Logs**: `/home/elect/.n8n/logs/`
- **Servicio systemd**: `/etc/systemd/system/n8n.service`

### Acceso
- **URL interna**: `http://localhost:5678`
- **URL via nginx**: `http://<IP-VM>/n8n/`
- **WebSocket**: Habilitado para comunicación en tiempo real

### Node.js
- **Versión**: v20.19.5
- **npm**: v10.8.2

## 🚀 Uso Básico

### Gestión del Servicio
```bash
# Ver estado
sudo systemctl status n8n.service

# Iniciar/Detener/Reiniciar
sudo systemctl start n8n.service
sudo systemctl stop n8n.service
sudo systemctl restart n8n.service

# Ver logs
sudo journalctl -u n8n.service -f
```

### Primer Acceso
1. Accede a `http://<IP-VM>/n8n/`
2. Crea tu cuenta de administrador en el primer acceso
3. Configura las credenciales necesarias para las integraciones

## 🔗 Integración con capibara6-consensus

### Endpoints Disponibles para Workflows

#### 1. API Flask (server.py)
- **Base URL**: `http://localhost:5000/api/`
- **Health Check**: `GET /api/health`
- **MCP Status**: `GET /api/mcp/status`
- **MCP Tools List**: `GET /api/mcp/tools/list`
- **MCP Tools Call**: `POST /api/mcp/tools/call`
- **Guardar conversación**: `POST /api/save-conversation`
- **Guardar lead**: `POST /api/save-lead`

#### 2. llama-server (GPT-OSS-20B)
- **Base URL**: `http://localhost:8080`
- **Completions**: Consulta la API de llama.cpp para inferencia

#### 3. Coqui TTS
- **Puerto**: Consultar configuración del servicio coqui-tts

### Ejemplos de Workflows

#### Workflow 1: Automatización de Procesamiento de Leads
1. **Trigger**: Webhook recibe nuevo lead
2. **Acción 1**: Guardar en `/api/save-lead`
3. **Acción 2**: Enviar a GPT-OSS-20B para análisis
4. **Acción 3**: Clasificar y enrutar según consenso
5. **Acción 4**: Enviar notificación

#### Workflow 2: Consenso Multi-Modelo
1. **Trigger**: Nueva consulta de usuario
2. **Acción 1**: Enviar a gpt-oss-20b (llama-server)
3. **Acción 2**: Enviar a mixtral (si disponible)
4. **Acción 3**: Enviar a phi3-mini (si disponible)
5. **Acción 4**: Aplicar lógica de consenso
6. **Acción 5**: Retornar respuesta unificada

#### Workflow 3: Pipeline TTS
1. **Trigger**: Webhook con texto
2. **Acción 1**: Preprocesar texto
3. **Acción 2**: Llamar a Coqui TTS
4. **Acción 3**: Guardar audio generado
5. **Acción 4**: Notificar completado

#### Workflow 4: Monitoreo de Sistema
1. **Trigger**: Cron cada 5 minutos
2. **Acción 1**: Verificar `/api/health`
3. **Acción 2**: Verificar `/api/mcp/status`
4. **Acción 3**: Verificar estado de servicios
5. **Acción 4**: Alertar si hay problemas

## 🔌 Nodos de n8n Útiles

### Para Integración
- **HTTP Request**: Para llamar a los endpoints de Flask y llama-server
- **Webhook**: Para recibir eventos externos
- **Function**: Para lógica de enrutamiento y consenso
- **Split In Batches**: Para procesamiento por lotes
- **Merge**: Para combinar respuestas de múltiples modelos
- **Switch**: Para enrutamiento condicional
- **Set**: Para transformar datos

### Para Desarrollo
- **Execute Command**: Para ejecutar scripts del sistema
- **Read/Write Binary File**: Para manejar archivos de audio TTS
- **Code**: Para lógica JavaScript/Python personalizada

## 🛡️ Seguridad

### Recomendaciones
1. **Autenticación**: Configurar autenticación robusta en n8n
2. **HTTPS**: Considerar añadir certificado SSL a nginx
3. **Firewall**: Restringir acceso al puerto 5678 directo
4. **Secrets**: Usar el sistema de credenciales de n8n para APIs

### Variables de Entorno Configuradas
```bash
N8N_HOST=0.0.0.0
N8N_PORT=5678
N8N_PROTOCOL=http
NODE_ENV=production
N8N_USER_FOLDER=/home/elect/.n8n
N8N_LOG_LEVEL=info
```

## 📊 Monitoreo

### Logs
```bash
# n8n logs
sudo journalctl -u n8n.service -f

# n8n file logs
tail -f /home/elect/.n8n/logs/*.log

# nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

### Métricas
- CPU y Memoria del proceso n8n
- Tasa de ejecución de workflows
- Tasa de errores en workflows

## 🔄 Backup

### Archivos Importantes
```bash
# Workflows y credenciales
/home/elect/.n8n/

# Configuración del servicio
/etc/systemd/system/n8n.service

# Configuración nginx
/etc/nginx/sites-available/llama-api
```

### Comando de Backup
```bash
tar -czf n8n-backup-$(date +%Y%m%d).tar.gz /home/elect/.n8n/
```

## 📚 Recursos

- **n8n Documentación**: https://docs.n8n.io
- **n8n Community**: https://community.n8n.io
- **Repositorio n8n**: https://github.com/n8n-io/n8n
- **MCP Protocol**: https://modelcontextprotocol.io

## 🤝 Soporte

Para soporte del sistema capibara6-consensus:
- Email: info@anachroni.co
- Web: https://www.anachroni.co

---

**Fecha de instalación**: 2025-11-09
**Versión n8n**: 1.118.2
**Node.js**: v20.19.5
**Sistema**: Debian 12 (ARM64/aarch64)
**Plataforma**: Google Cloud ARM Axion
