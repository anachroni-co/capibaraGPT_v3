# Sistema de Monitoreo Capibara6

## 📊 Visión General

Este directorio contiene la configuración completa del sistema de monitoreo para Capibara6, incluyendo:

- **Grafana** - Dashboards y visualizaciones
- **Prometheus** - Recolección de métricas
- **Jaeger** - Distributed tracing
- **Alertas** - Notificaciones automáticas

## 🏗️ Arquitectura de Monitoreo

```
┌─────────────────────────────────────────────────────────────┐
│                    Servicios Capibara6                      │
│                                                             │
│  Backend (5001)  TTS (5002)  MCP (5003)  Auth (5004)       │
│  Milvus (19530)  Nebula (9669)  Bridge API (8000)          │
└────────────────────┬────────────────────────────────────────┘
                     │ Métricas
                     ▼
              ┌──────────────┐
              │  Prometheus  │ ← Recolector de métricas
              │   (9090)     │
              └──────┬───────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
   ┌─────────┐  ┌────────┐  ┌─────────┐
   │ Grafana │  │ Jaeger │  │ Alertas │
   │ (3000)  │  │(16686) │  │  Email  │
   └─────────┘  └────────┘  └─────────┘
```

## 🎯 Métricas Principales

### 1. Sistema General
- **Requests/s** - Tasa de requests HTTP
- **Latencia** - p50, p95, p99
- **Tasa de errores** - 4xx, 5xx
- **CPU, Memoria, Disco** - Recursos del sistema

### 2. Sistema RAG
- **Milvus** - Búsquedas vectoriales, tamaño de colección, latencia
- **Nebula Graph** - Consultas de grafo, vértices, aristas, latencia
- **Bridge API** - Throughput, cache hit rate

### 3. Router Semántico
- **Distribución de modelos** - Qué modelos se usan más
- **Complejidad de queries** - Distribución de complejidad
- **Cache hit rate** - Eficiencia del cache

### 4. E2B Sandboxes
- **Sandboxes activos** - Contador en tiempo real
- **Tiempo de ejecución** - Distribución de tiempos
- **Tasa de éxito/fallo** - Confiabilidad

### 5. RQ Workers
- **Cola de tareas** - Longitud de queue
- **Workers activos** - 3/3 esperados
- **Throughput** - Jobs completados/s
- **Tasa de fallos** - Jobs fallidos

### 6. Optimización TOON
- **Ahorro de tokens** - Porcentaje y cantidad
- **Activación automática** - Cuándo se usa TOON
- **Tamaño promedio** - Contexto antes/después

## 🚀 Instalación y Configuración

### 1. Importar Dashboard en Grafana

```bash
# Acceder a Grafana
open http://rag3:3000

# Credenciales por defecto
Usuario: admin
Password: admin

# Importar dashboard
1. Dashboard → Import
2. Upload JSON file: grafana-dashboard-config.json
3. Seleccionar datasource: Prometheus
4. Click "Import"
```

### 2. Configurar Alertas en Prometheus

```bash
# En VM rag3
cd /path/to/prometheus

# Copiar archivo de alertas
cp monitoring/prometheus-alerts.yml /etc/prometheus/rules/

# Editar prometheus.yml
rule_files:
  - '/etc/prometheus/rules/prometheus-alerts.yml'

# Recargar configuración
curl -X POST http://localhost:9090/-/reload
# O reiniciar Prometheus
docker restart capibara6-prometheus
```

### 3. Verificar Estado

```bash
# Verificar Prometheus
curl http://rag3:9090/-/healthy
curl http://rag3:9090/api/v1/targets

# Verificar Grafana
curl http://rag3:3000/api/health

# Verificar Jaeger
curl http://rag3:16686/api/services
```

## 📈 Dashboards Disponibles

### 1. Dashboard Principal "Capibara6 - Sistema Completo"

**Secciones:**

**Visión General**
- Requests por segundo
- Latencia (p95, p99)
- Tasa de errores
- Estado general del sistema

**Sistema RAG**
- Milvus: búsquedas vectoriales, colección
- Nebula Graph: consultas, vértices, aristas

**Router & E2B**
- Distribución de modelos
- Sandboxes activos
- Tiempo de ejecución
- Success rate

**Workers & Cache**
- Cola RQ
- Workers activos
- Cache hit rate

**Recursos**
- CPU, Memoria, Disco I/O
- Network traffic

## 🚨 Alertas Configuradas

### Críticas (🔴)
- Latencia > 5s
- Tasa de errores > 50/s
- CPU > 95%
- Memoria > 95%
- Milvus/Nebula DOWN
- PostgreSQL/Redis DOWN
- Cluster Nebula unhealthy
- Workers RQ < 2

### Warnings (⚠️)
- Latencia > 2s
- Tasa de errores > 10/s
- CPU > 80%
- Memoria > 85%
- Disco > 80%
- Sandboxes E2B cerca del límite (4/5)
- Cola RQ > 100 tareas
- Cache hit rate < 30%

### Informativas (ℹ️)
- Queries muy complejas
- Colección Milvus creciendo rápido
- Ejecuciones E2B largas
- Ahorro TOON bajo

## 📊 Ejemplos de Uso

### Ver Métricas en Prometheus

```bash
# Latencia p99
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))

# Tasa de errores
rate(http_requests_total{status=~"5.."}[5m])

# Milvus búsquedas/s
rate(milvus_search_requests_total[5m])

# Workers RQ activos
rq_workers_active

# Cache hit rate
rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))
```

### Consultar Alertas Activas

```bash
# API Prometheus
curl http://rag3:9090/api/v1/alerts

# Verificar reglas
curl http://rag3:9090/api/v1/rules
```

### Jaeger - Buscar Traces

```bash
# UI Web
open http://rag3:16686

# API - Buscar traces
curl 'http://rag3:16686/api/traces?service=capibara6-api&limit=20'

# Traces lentos (> 1s)
curl 'http://rag3:16686/api/traces?service=capibara6-api&minDuration=1s'
```

## 🔧 Configuración Avanzada

### Ajustar Intervalos de Scrape

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'capibara6'
    scrape_interval: 15s  # Cada 15 segundos
    scrape_timeout: 10s
    static_configs:
      - targets: ['localhost:8000', 'localhost:5001']
```

### Retención de Datos

```yaml
# prometheus.yml
storage:
  tsdb:
    retention.time: 15d  # Mantener 15 días
    retention.size: 50GB # O hasta 50GB
```

### Alertmanager - Notificaciones

```yaml
# alertmanager.yml
route:
  receiver: 'email'
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h

receivers:
  - name: 'email'
    email_configs:
      - to: 'alerts@example.com'
        from: 'prometheus@capibara6.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'user@gmail.com'
        auth_password: 'app-password'
```

## 📱 Notificaciones

### Slack Integration

```yaml
receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#alerts'
        title: 'Capibara6 Alert'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

### PagerDuty Integration

```yaml
receivers:
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
        severity: '{{ .CommonLabels.severity }}'
```

## 🎯 Mejores Prácticas

### 1. Revisar Dashboards Regularmente
- **Diariamente**: Métricas principales, alertas activas
- **Semanalmente**: Tendencias, patrones de uso
- **Mensualmente**: Optimizaciones, ajustes

### 2. Umbrales de Alertas
- Ajustar según baseline real del sistema
- Evitar alert fatigue con demasiadas alertas
- Priorizar correctamente (critical vs warning)

### 3. Retención de Datos
- Prometheus: 15-30 días (métricas detalladas)
- Grafana: Indefinido (dashboards)
- Jaeger: 7 días (traces)

### 4. Backup
- Exportar dashboards regularmente
- Versionar configuraciones en git
- Backup de datos de Prometheus

## 🔍 Troubleshooting

### Prometheus no recolecta métricas

```bash
# Verificar targets
curl http://rag3:9090/api/v1/targets

# Ver errores en logs
docker logs capibara6-prometheus

# Verificar conectividad
curl http://rag3:8000/metrics
```

### Grafana no muestra datos

```bash
# Verificar datasource
Settings → Data Sources → Prometheus → Test

# Verificar queries
Explore → Run query manualmente

# Ver logs
docker logs capibara6-grafana
```

### Alertas no se disparan

```bash
# Verificar reglas cargadas
curl http://rag3:9090/api/v1/rules

# Verificar sintaxis
promtool check rules prometheus-alerts.yml

# Ver estado de alertas
curl http://rag3:9090/api/v1/alerts
```

## 📚 Referencias

- **Grafana**: http://rag3:3000
- **Prometheus**: http://rag3:9090
- **Jaeger**: http://rag3:16686
- **Prometheus Docs**: https://prometheus.io/docs
- **Grafana Docs**: https://grafana.com/docs
- **Jaeger Docs**: https://www.jaegertracing.io/docs

## 🎓 Tutoriales

### Crear Dashboard Personalizado

1. Dashboard → New → Add Visualization
2. Seleccionar datasource: Prometheus
3. Escribir query PromQL
4. Configurar panel (tipo, título, etc.)
5. Save dashboard

### Agregar Nueva Alerta

1. Editar `prometheus-alerts.yml`
2. Agregar nueva regla bajo el group apropiado
3. Validar: `promtool check rules prometheus-alerts.yml`
4. Recargar Prometheus: `curl -X POST http://rag3:9090/-/reload`
5. Verificar: `curl http://rag3:9090/api/v1/rules`

---

**Última actualización:** 2025-11-13
**Versión:** 1.0.0
**Mantenido por:** Equipo Capibara6
