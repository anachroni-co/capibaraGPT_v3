# Pila RAG Contenerizada

Este directorio contiene la infraestructura dockerizada de la pila RAG (PostgreSQL, SQLite, Redis, ChromaDB, Milvus y Nebula Graph).

## Pasos rÃ¡pidos

1. Copia  como  y ajusta credenciales.
2. Revisa los scripts en  antes del primer arranque.
3. Levanta PostgreSQL y Redis () y verifica logs.
4. Activa el resto de servicios ().
5. Ejecuta los scripts de inicializaciÃ³n:
   - PostgreSQL: 
   - Milvus: 
   - Nebula: 

## Directorios clave

- : definiciÃ³n de todos los servicios.
- : plantilla de variables de entorno.
- : scripts de bootstrap para cada motor.
- : se generarÃ¡ automÃ¡ticamente para persistencia (monta volÃºmenes locales).

## Conexiones dentro de la red Docker

- PostgreSQL: 
- Redis: 
- ChromaDB: 
- Milvus: 
- Nebula Graph: 
- SQLite: archivos bajo 

## PrÃ³ximos pasos sugeridos

- Construir pipelines de ingestiÃ³n que inserten en SQL, vectores y grafo.
- AÃ±adir monitoreo (Prometheus/Grafana) y alertas.
- Establecer backups para PostgreSQL, MinIO y Nebula.
