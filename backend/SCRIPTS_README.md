# Scripts de Gestión de Servicios - Capibara6

## 📋 Descripción

Este directorio contiene scripts bash para gestionar el ciclo de vida de todos los servicios backend de Capibara6.

## 🚀 Scripts Disponibles

### 1. `start-all-services.sh` - Iniciar Servicios Principales

Inicia los servicios **CORE** requeridos para el funcionamiento básico del sistema.

**Servicios iniciados:**
- ✅ Backend Principal (server_gptoss.py) - Puerto 5001
- ✅ TTS Server (kyutai_tts_server.py) - Puerto 5002
- ✅ Auth Server (auth_server.py) - Puerto 5004

**Uso:**
```bash
cd backend
./start-all-services.sh
```

**Características:**
- ✓ Verifica puertos antes de iniciar
- ✓ Crea logs en `backend/logs/`
- ✓ Guarda PIDs en `.service_pids`
- ✓ Verifica que cada servicio inició correctamente
- ✓ Salida con colores para mejor visibilidad

### 2. `start-optional-services.sh` - Iniciar Servicios Opcionales

Inicia servicios adicionales que mejoran la funcionalidad pero NO son requeridos.

**Servicios disponibles:**
- ⚙️ MCP Server (puerto 5003) - Context & RAG
- ⚙️ Consensus Server (puerto 5005) - Multi-modelo
- ⚙️ Smart MCP Server (puerto 5010) - RAG selectivo alternativo
- ⚙️ FastAPI Server (puerto 8000) - API alternativa

**Uso:**
```bash
cd backend
./start-optional-services.sh
```

**Nota:** El script preguntará interactivamente qué servicios opcionales iniciar.

### 3. `stop-all-services.sh` - Detener Todos los Servicios

Detiene **todos** los servicios (principales y opcionales) de forma segura.

**Uso:**
```bash
cd backend
./stop-all-services.sh
```

**Características:**
- ✓ Intenta detención graciosa (SIGTERM)
- ✓ Forzado automático si es necesario (SIGKILL después de 5 segundos)
- ✓ Limpia archivo de PIDs
- ✓ Verifica procesos residuales

### 4. `check-services.sh` - Verificar Estado

Verifica el estado completo de todos los servicios y dependencias.

**Uso:**
```bash
cd backend
./check-services.sh
```

**Verifica:**
- ✅ Servicios principales (requeridos)
- ⚙️ Servicios opcionales
- 🗄️ Bases de datos Docker (PostgreSQL, TimescaleDB, Redis)
- 🌐 Servicios remotos en VMs (TTS, MCP, N8N)
- 📊 Health endpoints de cada servicio
- 🔍 Procesos Python activos

**Salida de ejemplo:**
```
🔍 Capibara6 - Verificando Estado de Servicios...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SERVICIOS PRINCIPALES (REQUERIDOS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 Backend Principal (server_gptoss.py) (Puerto 5001)
  ✓ Puerto 5001: ACTIVO
  → Probando health endpoint: http://localhost:5001/api/health
  ✓ Health check: OK
  ✅ Backend Principal: FUNCIONANDO CORRECTAMENTE

...

📊 RESUMEN
✅ Todos los servicios requeridos están activos (3/3)
✅ El sistema está OPERATIVO
```

## 📁 Estructura de Archivos

```
backend/
├── start-all-services.sh          # Iniciar servicios principales
├── start-optional-services.sh     # Iniciar servicios opcionales
├── stop-all-services.sh           # Detener todos los servicios
├── check-services.sh              # Verificar estado de servicios
├── SCRIPTS_README.md              # Esta documentación
├── .service_pids                  # PIDs de servicios principales (auto-generado)
├── .optional_service_pids         # PIDs de servicios opcionales (auto-generado)
└── logs/                          # Logs de cada servicio
    ├── server_gptoss.log
    ├── kyutai_tts.log
    ├── auth_server.log
    ├── mcp_server.log
    ├── consensus_server.log
    └── ...
```

## 🔧 Flujo de Trabajo Típico

### Desarrollo Local

```bash
# 1. Iniciar servicios principales
cd backend
./start-all-services.sh

# 2. Verificar que todo está corriendo
./check-services.sh

# 3. (Opcional) Iniciar servicios adicionales si los necesitas
./start-optional-services.sh

# 4. Trabajar en tu desarrollo...

# 5. Verificar estado cuando sea necesario
./check-services.sh

# 6. Al terminar, detener todo
./stop-all-services.sh
```

### Testing

```bash
# 1. Iniciar solo lo necesario
./start-all-services.sh

# 2. Ejecutar tests
pytest tests/

# 3. Detener servicios
./stop-all-services.sh
```

### Debugging

```bash
# 1. Verificar qué está corriendo
./check-services.sh

# 2. Ver logs de un servicio específico
tail -f logs/server_gptoss.log

# 3. Ver logs de todos los servicios
tail -f logs/*.log

# 4. Reiniciar un servicio específico
./stop-all-services.sh
./start-all-services.sh
```

## 🗺️ Mapa de Puertos

| Puerto | Servicio | Tipo | Estado |
|--------|----------|------|--------|
| 5001 | Backend Principal (server_gptoss.py) | Principal | ✅ Requerido |
| 5002 | TTS Server (kyutai_tts_server.py) | Principal | ✅ Requerido |
| 5003 | MCP Server (mcp_server.py) | Opcional | ⚙️ Opcional |
| 5004 | Auth Server (auth_server.py) | Principal | ✅ Requerido |
| 5005 | Consensus Server (consensus_server.py) | Opcional | ⚙️ Opcional |
| 5010 | Smart MCP Server (smart_mcp_server.py) | Opcional | ⚙️ Alternativo |
| 8000 | FastAPI Server (main.py) | Opcional | ⚙️ Alternativo |
| 5432 | PostgreSQL | Database | 🐳 Docker |
| 5433 | TimescaleDB | Database | 🐳 Docker |
| 6379 | Redis | Cache | 🐳 Docker |

## 📊 Health Endpoints

Cada servicio expone un endpoint de health check:

```bash
# Backend Principal
curl http://localhost:5001/api/health

# TTS Server
curl http://localhost:5002/health

# MCP Server
curl http://localhost:5003/api/mcp/health

# Auth Server
curl http://localhost:5004/health

# Consensus Server
curl http://localhost:5005/api/consensus/health

# Smart MCP Server
curl http://localhost:5010/health

# FastAPI Server
curl http://localhost:8000/health
```

## ⚠️ Troubleshooting

### Problema: Puerto ya en uso

**Síntoma:**
```
⚠️  Puerto 5001 ya está en uso. Omitiendo Backend Principal
```

**Solución:**
```bash
# Opción 1: Detener todos los servicios
./stop-all-services.sh

# Opción 2: Identificar y detener el proceso específico
lsof -ti :5001 | xargs kill

# Opción 3: Forzar detención de todos los procesos Python relacionados
pkill -f "server_gptoss.py"
```

### Problema: Servicio no inicia

**Síntoma:**
```
❌ Backend Principal falló al iniciar. Ver logs/server_gptoss.log
```

**Solución:**
```bash
# 1. Ver el log del servicio
cat logs/server_gptoss.log

# 2. Verificar dependencias
pip install -r requirements.txt

# 3. Verificar que el archivo existe
ls -la server_gptoss.py

# 4. Intentar iniciar manualmente para ver errores
python3 server_gptoss.py
```

### Problema: Health check falla

**Síntoma:**
```
✓ Puerto 5001: ACTIVO
✗ Health check: FALLÓ
⚠️  Backend Principal: Puerto activo pero health check falló
```

**Solución:**
```bash
# 1. Verificar el endpoint manualmente
curl -v http://localhost:5001/api/health

# 2. Ver logs del servicio
tail -f logs/server_gptoss.log

# 3. El servicio puede estar iniciándose, esperar y reintentar
sleep 5
./check-services.sh
```

### Problema: Procesos residuales

**Síntoma:**
```
⚠️  Advertencia: Algunos procesos aún están corriendo
```

**Solución:**
```bash
# Forzar detención de todos los procesos
pkill -9 -f "server_gptoss.py"
pkill -9 -f "kyutai_tts_server.py"
pkill -9 -f "mcp_server.py"
pkill -9 -f "auth_server.py"
pkill -9 -f "consensus_server.py"
pkill -9 -f "smart_mcp_server.py"

# O usar el script de detención y luego verificar
./stop-all-services.sh
ps aux | grep python
```

## 🔒 Requisitos

### Software Necesario

- Bash 4.0+
- Python 3.8+
- `lsof` (para verificar puertos)
- `curl` (para health checks)
- `pgrep` y `pkill` (para gestión de procesos)

### Permisos

Los scripts deben tener permisos de ejecución:
```bash
chmod +x *.sh
```

### Dependencias Python

Asegúrate de tener instaladas las dependencias:
```bash
pip install -r requirements.txt
```

## 📝 Notas Adicionales

### Logs

- Los logs se crean automáticamente en `backend/logs/`
- Cada servicio tiene su propio archivo de log
- Los logs persisten entre reinicios
- Para limpiar logs: `rm logs/*.log`

### PIDs

- Los PIDs se guardan en archivos ocultos (`.service_pids`, `.optional_service_pids`)
- Se limpian automáticamente al detener servicios
- No es necesario gestionarlos manualmente

### Docker

Para iniciar las bases de datos Docker:
```bash
cd ..  # Volver al root del proyecto
docker-compose up -d
```

Para detenerlas:
```bash
docker-compose down
```

### Servicios Remotos (VMs)

Los servicios en VMs externas (TTS, MCP, N8N) se verifican pero NO se gestionan con estos scripts. Para gestionarlos, conectarse a la VM correspondiente.

## 🔗 Referencias

- **Configuración completa:** `../web/config.js`
- **Arquitectura:** `../ARCHITECTURE_QUICK_REF.md`
- **Plan de consolidación:** `../BACKEND_CONSOLIDATION_PLAN.md`
- **Hallazgos de infraestructura:** `../INFRASTRUCTURE_FINDINGS.md`
- **Setup de servicios:** `../SERVICES_SETUP.md`

## 🆘 Soporte

Si encuentras problemas con los scripts:

1. Verifica los logs en `backend/logs/`
2. Ejecuta `./check-services.sh` para diagnóstico completo
3. Revisa `../INFRASTRUCTURE_FINDINGS.md` para entender la arquitectura
4. Consulta la sección de Troubleshooting arriba

---

**Última actualización:** 2025-11-13
**Versión:** 1.0
**Autor:** Consolidación backend Fase 4
