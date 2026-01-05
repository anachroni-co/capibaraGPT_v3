# Docker Manager - Capibara6

Script de gestión de contenedores Docker para el proyecto Capibara6.

## Ubicación

`/home/elect/docker_manager.py`

## Características

- ✅ Ver estado de todos los contenedores organizados por categoría
- ✅ Iniciar/Detener/Reiniciar todos los contenedores de forma ordenada
- ✅ Reconstruir servicios específicos
- ✅ Ver logs de servicios
- ✅ Verificar salud de todos los servicios
- ✅ Limpiar recursos no utilizados
- ✅ Colores y formato amigable en terminal

## Uso

### Ver estado de todos los contenedores

```bash
python3 /home/elect/docker_manager.py status
```

### Iniciar todos los contenedores

```bash
python3 /home/elect/docker_manager.py start
```

El script inicia los contenedores en el orden correcto:
1. Bases de datos (PostgreSQL, TimescaleDB, Redis, Milvus)
2. Nebula Graph (metad, storaged, graphd, studio)
3. Monitoreo (Prometheus, Grafana, Jaeger)
4. Aplicación (API, Workers, Nginx, n8n)

### Detener todos los contenedores

```bash
python3 /home/elect/docker_manager.py stop
```

Detiene los contenedores en orden inverso para evitar errores.

### Reiniciar todos los contenedores

```bash
python3 /home/elect/docker_manager.py restart
```

Ejecuta stop + start de forma automática.

### Reconstruir un servicio específico

```bash
python3 /home/elect/docker_manager.py rebuild capibara6-api
```

Esto:
1. Detiene el contenedor
2. Elimina el contenedor
3. Reconstruye la imagen
4. Inicia el contenedor nuevamente

### Ver logs de un servicio

```bash
python3 /home/elect/docker_manager.py logs capibara6-api
```

Muestra las últimas 50 líneas de logs por defecto.

### Verificar salud de servicios

```bash
python3 /home/elect/docker_manager.py health
```

Muestra el estado de healthcheck de todos los servicios.

### Limpiar recursos no utilizados

```bash
python3 /home/elect/docker_manager.py clean
```

Elimina:
- Contenedores detenidos
- Imágenes sin usar
- Volúmenes sin usar

### Ayuda

```bash
python3 /home/elect/docker_manager.py help
```

## Grupos de Servicios

El script organiza los contenedores en grupos:

### Databases
- capibara6-postgres
- capibara6-timescaledb
- capibara6-redis
- milvus-standalone, milvus-etcd, milvus-minio

### Nebula Graph
- nebula-docker-compose-metad0-1, metad1-1, metad2-1
- nebula-docker-compose-storaged0-1, storaged1-1, storaged2-1
- nebula-docker-compose-graphd-1, graphd1-1, graphd2-1
- nebula-docker-compose-studio-1

### Application
- capibara6-api
- capibara6-worker-1, worker-2, worker-3
- capibara6-nginx
- capibara6-n8n

### Monitoring
- capibara6-prometheus
- capibara6-grafana
- capibara6-jaeger

## Ejemplos

```bash
# Ver estado actual
python3 /home/elect/docker_manager.py status

# Reiniciar todo rápidamente
python3 /home/elect/docker_manager.py restart

# Reconstruir el API después de cambios en código
python3 /home/elect/docker_manager.py rebuild capibara6-api

# Ver logs de un worker
python3 /home/elect/docker_manager.py logs capibara6-worker-1

# Verificar que todos los servicios estén healthy
python3 /home/elect/docker_manager.py health

# Limpiar espacio en disco
python3 /home/elect/docker_manager.py clean
```

## Notas

- El script usa colores ANSI para mejor visualización
- Los tiempos de espera están optimizados para evitar errores de inicio
- Los servicios se inician en orden de dependencias
- El script es seguro para usar en producción

## Troubleshooting

### Error: "No such file or directory: 'docker-compose'"

El script usa `docker compose` (versión nueva). Si tienes la versión antigua, modifica el script.

### Los contenedores no inician correctamente

Verifica los logs:
```bash
python3 /home/elect/docker_manager.py logs <nombre-servicio>
```

### Un servicio se queda en "starting"

Espera el tiempo de start-period del healthcheck (60 segundos típicamente).

## Autor

Creado para el proyecto Capibara6 - Anachroni s.coop
