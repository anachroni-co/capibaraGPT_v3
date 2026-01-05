# Containerized RAG Stack

This directory contains the dockerized infrastructure for the RAG stack (PostgreSQL, SQLite, Redis, ChromaDB, Milvus, and Nebula Graph).

## Quick Steps

1. Copy `.env.example` as `.env` and adjust credentials.
2. Review scripts in `scripts/` before first startup.
3. Start PostgreSQL and Redis and verify logs.
4. Activate remaining services.
5. Run initialization scripts:
   - PostgreSQL: `scripts/init_postgres.sh`
   - Milvus: `scripts/init_milvus.sh`
   - Nebula: `scripts/init_nebula.sh`

## Key Directories

- `docker-compose.yml`: Definition of all services.
- `.env.example`: Environment variables template.
- `scripts/`: Bootstrap scripts for each engine.
- `data/`: Generated automatically for persistence (mounts local volumes).

## Connections within Docker Network

| Service | Connection |
|---------|------------|
| PostgreSQL | `postgres:5432` |
| Redis | `redis:6379` |
| ChromaDB | `chromadb:8000` |
| Milvus | `milvus:19530` |
| Nebula Graph | `nebula:9669` |
| SQLite | Files under `data/sqlite/` |

## Services Overview

### PostgreSQL
- Relational database for structured data
- Stores user data, conversations, metadata

### Redis
- In-memory cache
- Session management
- Rate limiting

### ChromaDB
- Vector database
- Embedding storage
- Similarity search

### Milvus
- Scalable vector database
- High-performance similarity search
- Production-ready vector storage

### Nebula Graph
- Graph database
- Knowledge graph storage
- Relationship queries

## Suggested Next Steps

- Build ingestion pipelines that insert into SQL, vectors, and graph.
- Add monitoring (Prometheus/Grafana) and alerts.
- Establish backups for PostgreSQL, MinIO, and Nebula.

## Usage

### Start All Services

```bash
docker-compose up -d
```

### Check Service Status

```bash
docker-compose ps
```

### View Logs

```bash
docker-compose logs -f [service_name]
```

### Stop All Services

```bash
docker-compose down
```

### Reset Data

```bash
docker-compose down -v
rm -rf data/
```

## Configuration

### Environment Variables

Copy and modify the template:

```bash
cp .env.example .env
```

Key variables:
- `POSTGRES_PASSWORD`: PostgreSQL password
- `REDIS_PASSWORD`: Redis password (optional)
- `MILVUS_HOST`: Milvus connection host

### Resource Allocation

Adjust in `docker-compose.yml`:

```yaml
services:
  milvus:
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

## Troubleshooting

### Service won't start
```bash
# Check logs
docker-compose logs [service_name]

# Verify ports are free
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis
```

### Data persistence issues
```bash
# Verify volume mounts
docker volume ls
docker volume inspect [volume_name]
```

### Connection refused
```bash
# Verify service is healthy
docker-compose ps
docker-compose exec [service] healthcheck
```
