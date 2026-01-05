# Guía de Despliegue n8n - Capibara6

## 📋 Resumen

Esta guía cubre el despliegue de n8n en el ecosistema Capibara6 usando Docker Compose o Kubernetes.

---

## 🚀 Opción 1: Despliegue con Docker Compose

### Prerrequisitos

- Docker >= 20.10
- Docker Compose >= 2.0
- 2GB RAM libre
- 5GB espacio en disco

### Pasos de Despliegue

#### 1. Configurar Variables de Entorno

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Generar encryption key para n8n
openssl rand -hex 32

# Editar .env y añadir:
nano .env
```

Añadir las siguientes variables:

```bash
# n8n Configuration
N8N_ENCRYPTION_KEY=<tu_encryption_key_generada>
N8N_WEBHOOK_URL=http://localhost:5678/
N8N_EXECUTIONS_MODE=regular

# Opcional: Basic Auth
# N8N_BASIC_AUTH_ACTIVE=true
# N8N_BASIC_AUTH_USER=admin
# N8N_BASIC_AUTH_PASSWORD=tu_password_seguro
```

#### 2. Crear Directorio de Datos

```bash
mkdir -p backend/data/n8n
chmod 755 backend/data/n8n
```

#### 3. Levantar Servicios

```bash
# Levantar solo n8n
docker-compose up -d n8n

# O levantar todo el stack
docker-compose up -d
```

#### 4. Verificar Estado

```bash
# Ver logs
docker-compose logs -f n8n

# Verificar salud del contenedor
docker-compose ps n8n

# Health check
curl http://localhost:5678/healthz
```

#### 5. Acceder a n8n

- **URL local**: http://localhost:5678
- **URL con nginx**: http://localhost/n8n/

En el primer acceso, se te pedirá crear una cuenta de administrador.

---

## ☸️ Opción 2: Despliegue con Kubernetes

### Prerrequisitos

- Cluster Kubernetes >= 1.20
- kubectl configurado
- 2GB RAM por pod
- 5GB almacenamiento persistente
- Ingress Controller (nginx)
- Cert-manager (para SSL/TLS)

### Pasos de Despliegue

#### 1. Crear Namespace

```bash
kubectl create namespace capibara6
```

#### 2. Configurar Secrets

```bash
# Generar encryption key
N8N_KEY=$(openssl rand -hex 32)

# Codificar en base64
N8N_KEY_B64=$(echo -n "$N8N_KEY" | base64)

# Editar secrets.yaml
nano k8s/secrets.yaml
```

Reemplazar `<base64_encoded_n8n_encryption_key>` con el valor de `$N8N_KEY_B64`.

#### 3. Aplicar Configuraciones

```bash
# Orden de aplicación:
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

#### 4. Verificar Despliegue

```bash
# Ver pods
kubectl get pods -n capibara6 -l app=capibara6-n8n

# Ver logs
kubectl logs -n capibara6 -l app=capibara6-n8n -f

# Ver servicios
kubectl get svc -n capibara6 n8n-service

# Ver ingress
kubectl get ingress -n capibara6
```

#### 5. Configurar DNS

Añadir registro DNS apuntando a tu Ingress Controller:

```
n8n.capibara6.com  ->  <INGRESS_IP>
```

#### 6. Verificar Certificado SSL

```bash
# Ver certificado
kubectl get certificate -n capibara6

# Ver eventos de cert-manager
kubectl describe certificate capibara6-n8n-tls -n capibara6
```

#### 7. Acceder a n8n

- **URL producción**: https://n8n.capibara6.com

---

## 🔧 Configuración Post-Despliegue

### 1. Crear Cuenta de Administrador

En el primer acceso:
1. Ir a la URL de n8n
2. Crear cuenta de owner
3. Configurar email y contraseña

### 2. Configurar Credenciales

Para integración con Capibara6:

1. **HTTP Request Node**
   - Name: `Capibara6 API`
   - URL: `http://capibara6-api:8000/api/` (Docker) o `http://capibara6-api-service:8000/api/` (K8s)

2. **Webhook Node**
   - Production URL: `https://n8n.capibara6.com/webhook/`
   - Test URL: `http://localhost:5678/webhook-test/`

### 3. Verificar Integración con Redis

```bash
# Docker
docker-compose exec n8n n8n info

# Kubernetes
kubectl exec -n capibara6 deployment/capibara6-n8n -- n8n info
```

Debe mostrar:
- Database: SQLite o PostgreSQL
- Queue: Redis (si EXECUTIONS_MODE=queue)

---

## 📊 Monitoreo

### Métricas de n8n

n8n expone métricas en formato Prometheus:

```
http://localhost:5678/metrics  (Docker)
http://n8n-service:5678/metrics  (K8s)
```

### Integración con Prometheus

Las métricas de n8n ya están configuradas en:
- `backend/deployment/prometheus.yml` (Docker)
- Prometheus ServiceMonitor (K8s - crear manualmente)

Ejemplo de ServiceMonitor para K8s:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: n8n-metrics
  namespace: capibara6
spec:
  selector:
    matchLabels:
      app: capibara6-n8n
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

### Dashboard Grafana

Importar dashboard de n8n:
- ID: 19817 (n8n Community Dashboard)
- URL: https://grafana.com/grafana/dashboards/19817

---

## 🔐 Seguridad

### 1. Habilitar Basic Auth (Producción)

**Docker:**
```bash
# En .env
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=password_muy_seguro_aqui

# Reiniciar
docker-compose restart n8n
```

**Kubernetes:**
```bash
# Crear secret
kubectl create secret generic n8n-basic-auth \
  --from-literal=user=admin \
  --from-literal=password=password_muy_seguro_aqui \
  -n capibara6

# Actualizar deployment para usar el secret
```

### 2. Restricción de IPs (Kubernetes)

Añadir a ingress.yaml:

```yaml
annotations:
  nginx.ingress.kubernetes.io/whitelist-source-range: "1.2.3.4/32,5.6.7.8/32"
```

### 3. Rate Limiting

Ya configurado en:
- `backend/deployment/nginx.conf` (Docker)
- `k8s/ingress.yaml` (Kubernetes)

---

## 💾 Backup y Restore

### Docker Compose

**Backup:**
```bash
# Backup de workflows y datos
docker-compose exec n8n n8n export:workflow --all --output=/data/n8n/backup-$(date +%Y%m%d).json

# Backup del volumen completo
docker run --rm -v capibara6_n8n_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/n8n-backup-$(date +%Y%m%d).tar.gz /data
```

**Restore:**
```bash
# Restaurar workflows
docker-compose exec n8n n8n import:workflow --input=/data/n8n/backup.json

# Restaurar volumen
docker run --rm -v capibara6_n8n_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/n8n-backup-YYYYMMDD.tar.gz -C /
```

### Kubernetes

**Backup:**
```bash
# Export workflows
kubectl exec -n capibara6 deployment/capibara6-n8n -- \
  n8n export:workflow --all --output=/data/n8n/backup.json

# Copiar del pod
kubectl cp capibara6/capibara6-n8n-xxx:/data/n8n/backup.json ./n8n-backup-$(date +%Y%m%d).json

# Backup de PVC (requiere velero u otra herramienta)
```

**Restore:**
```bash
# Copiar al pod
kubectl cp ./n8n-backup.json capibara6/capibara6-n8n-xxx:/data/n8n/backup.json

# Import workflows
kubectl exec -n capibara6 deployment/capibara6-n8n -- \
  n8n import:workflow --input=/data/n8n/backup.json
```

---

## 🔄 Migrar de SQLite a PostgreSQL

Cuando tengas PostgreSQL en otra VM:

### 1. Dump de SQLite

```bash
# Docker
docker-compose exec n8n n8n export:workflow --all --output=/home/node/.n8n/backup.json
docker-compose exec n8n n8n export:credentials --all --output=/home/node/.n8n/credentials.json
```

### 2. Actualizar Configuración

**Docker (.env):**
```bash
# Comentar SQLite, descomentar PostgreSQL
# DB_TYPE=sqlite
DB_TYPE=postgresdb
DB_POSTGRESDB_HOST=postgres-vm-ip
DB_POSTGRESDB_PORT=5432
DB_POSTGRESDB_DATABASE=n8n
DB_POSTGRESDB_USER=n8n
DB_POSTGRESDB_PASSWORD=tu_password
```

**Kubernetes (deployment.yaml):**
```yaml
env:
- name: DB_TYPE
  value: "postgresdb"
- name: DB_POSTGRESDB_HOST
  value: "postgres-vm-ip"
# ... resto de variables
```

### 3. Reiniciar y Restaurar

```bash
# Docker
docker-compose restart n8n
docker-compose exec n8n n8n import:workflow --input=/home/node/.n8n/backup.json

# Kubernetes
kubectl rollout restart deployment/capibara6-n8n -n capibara6
```

---

## 🐛 Troubleshooting

### n8n no inicia

```bash
# Ver logs detallados
docker-compose logs -f n8n  # Docker
kubectl logs -f deployment/capibara6-n8n -n capibara6  # K8s

# Problemas comunes:
# 1. N8N_ENCRYPTION_KEY no configurada
# 2. Puerto 5678 ocupado
# 3. Falta de permisos en volúmenes
```

### WebSocket no funciona

Verificar configuración de proxy:
- Nginx debe tener `Upgrade` y `Connection` headers
- Ingress debe tener anotaciones de WebSocket

### Workflows no se ejecutan

```bash
# Verificar conexión a Redis
# Docker
docker-compose exec n8n redis-cli -h redis ping

# Kubernetes
kubectl exec -n capibara6 deployment/capibara6-n8n -- \
  redis-cli -h redis-service ping
```

### Errores de permisos

```bash
# Docker
sudo chown -R 1000:1000 backend/data/n8n

# Kubernetes - verificar securityContext en deployment
```

---

## 📚 Recursos

- **n8n Documentación**: https://docs.n8n.io
- **n8n Community**: https://community.n8n.io
- **Workflows de ejemplo**: Ver `/docs/n8n/workflows/`
- **API Capibara6**: Ver `/docs/API.md`

---

## ✅ Checklist de Producción

- [ ] Encryption key configurada y guardada en lugar seguro
- [ ] Basic Auth o SSO habilitado
- [ ] HTTPS/TLS configurado (Kubernetes)
- [ ] Backups automatizados configurados
- [ ] Monitoreo en Prometheus/Grafana
- [ ] Rate limiting verificado
- [ ] Logs centralizados
- [ ] Alertas configuradas para workflows fallidos
- [ ] DNS configurado (n8n.capibara6.com)
- [ ] Documentación de workflows creados

---

**Última actualización**: 2025-11-10
**Versión n8n**: latest (compatible con 1.0+)
**Mantenedor**: Anachroni s.coop
