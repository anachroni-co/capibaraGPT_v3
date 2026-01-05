#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main API - API REST principal para Capibara6.
"""

import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Variables de entorno
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# Rate limiting (simulado)
rate_limit_storage = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación."""
    # Startup
    logger.info("🚀 Iniciando Capibara6 API...")
    
    try:
        # Inicializar componentes principales
        await initialize_components()
        logger.info("✅ Componentes inicializados correctamente")
    except Exception as e:
        logger.error(f"❌ Error inicializando componentes: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Cerrando Capibara6 API...")
    await cleanup_components()

async def initialize_components():
    """Inicializa los componentes principales del sistema."""
    try:
        # Importar y inicializar componentes
        from core.router import Router
        from ace.integration import ACEIntegration
        from execution.e2b_integration import E2BIntegration
        from scalability.aggressive_caching import AggressiveCache
        from scalability.dynamic_batching import DynamicBatcher
        
        # Inicializar componentes
        app.state.router = Router()
        app.state.ace_integration = ACEIntegration()
        app.state.e2b_integration = E2BIntegration()
        app.state.cache = AggressiveCache()
        app.state.batcher = DynamicBatcher()
        
        # Iniciar procesamiento de batches
        await app.state.batcher.start_processing()
        
        logger.info("Componentes del sistema inicializados")
        
    except ImportError as e:
        logger.warning(f"Algunos componentes no están disponibles: {e}")
        # Inicializar componentes básicos
        app.state.router = None
        app.state.ace_integration = None
        app.state.e2b_integration = None
        app.state.cache = None
        app.state.batcher = None

async def cleanup_components():
    """Limpia los componentes al cerrar la aplicación."""
    try:
        if hasattr(app.state, 'batcher') and app.state.batcher:
            await app.state.batcher.stop_processing()
        
        if hasattr(app.state, 'cache') and app.state.cache:
            app.state.cache.shutdown()
        
        logger.info("Componentes limpiados correctamente")
    except Exception as e:
        logger.error(f"Error limpiando componentes: {e}")

# Crear aplicación FastAPI
app = FastAPI(
    title="Capibara6 API",
    description="Advanced AI Agent System with Intelligent Routing, ACE, E2B, and Scalability",
    version="1.0.0",
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ENVIRONMENT == "development" else ["https://capibara6.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if ENVIRONMENT == "development" else ["capibara6.com", "*.capibara6.com"]
)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Middleware de rate limiting."""
    client_ip = request.client.host
    current_time = time.time()
    
    # Limpiar entradas antiguas (más de 1 minuto)
    rate_limit_storage = {k: v for k, v in rate_limit_storage.items() 
                         if current_time - v['last_request'] < 60}
    
    # Verificar límite (100 requests por minuto por IP)
    if client_ip in rate_limit_storage:
        if rate_limit_storage[client_ip]['count'] >= 100:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "retry_after": 60}
            )
        rate_limit_storage[client_ip]['count'] += 1
    else:
        rate_limit_storage[client_ip] = {'count': 1, 'last_request': current_time}
    
    response = await call_next(request)
    return response

# Dependencias
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Obtiene el usuario actual (simulado)."""
    # En un entorno real, esto validaría el token JWT
    if not credentials or credentials.credentials != "valid_token":
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    return {
        "user_id": "user_123",
        "username": "test_user",
        "permissions": ["read", "write", "execute"]
    }

# Endpoints de salud
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": ENVIRONMENT
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Health check detallado."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "components": {}
    }
    
    # Verificar componentes
    components = [
        ("router", app.state.router),
        ("ace_integration", app.state.ace_integration),
        ("e2b_integration", app.state.e2b_integration),
        ("cache", app.state.cache),
        ("batcher", app.state.batcher)
    ]
    
    for name, component in components:
        if component is not None:
            health_status["components"][name] = "healthy"
        else:
            health_status["components"][name] = "unavailable"
    
    return health_status

# Endpoints de API
@app.post(f"{API_PREFIX}/query")
async def process_query(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Procesa una query usando el sistema completo."""
    try:
        query = request.get("query", "")
        context = request.get("context", {})
        options = request.get("options", {})
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        start_time = time.time()
        
        # 1. Routing
        routing_result = None
        if app.state.router:
            routing_result = app.state.router.route_query(query, context)
        
        # 2. ACE Integration
        ace_result = None
        if app.state.ace_integration:
            ace_result = app.state.ace_integration.process_query(query)
        
        # 3. E2B Execution (si hay código)
        e2b_result = None
        if app.state.e2b_integration and "code" in query.lower():
            e2b_result = app.state.e2b_integration.execute_query(query)
        
        # 4. Cache result
        cache_key = f"query_{hash(query)}"
        if app.state.cache:
            app.state.cache.set(cache_key, {
                "query": query,
                "routing_result": routing_result,
                "ace_result": ace_result,
                "e2b_result": e2b_result
            })
        
        processing_time = time.time() - start_time
        
        # Background task para métricas
        background_tasks.add_task(log_query_metrics, {
            "query": query,
            "processing_time": processing_time,
            "user_id": current_user["user_id"],
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "query": query,
            "routing_result": routing_result,
            "ace_result": ace_result,
            "e2b_result": e2b_result,
            "processing_time_ms": processing_time * 1000,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error procesando query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{API_PREFIX}/models")
async def get_models(current_user: dict = Depends(get_current_user)):
    """Obtiene información de los modelos disponibles."""
    return {
        "models": [
            {
                "id": "capibara6-20b",
                "name": "Capibara6 20B",
                "description": "Modelo de 20B parámetros para tareas de complejidad media",
                "max_tokens": 8000,
                "capabilities": ["text_generation", "code_generation", "reasoning"]
            },
            {
                "id": "capibara6-120b",
                "name": "Capibara6 120B",
                "description": "Modelo de 120B parámetros para tareas complejas",
                "max_tokens": 32000,
                "capabilities": ["text_generation", "code_generation", "reasoning", "analysis"]
            }
        ]
    }

@app.get(f"{API_PREFIX}/metrics")
async def get_metrics(current_user: dict = Depends(get_current_user)):
    """Obtiene métricas del sistema."""
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
            "environment": ENVIRONMENT,
            "version": "1.0.0"
        },
        "components": {}
    }
    
    # Métricas de componentes
    if app.state.cache:
        cache_stats = app.state.cache.get_cache_stats()
        metrics["components"]["cache"] = cache_stats
    
    if app.state.batcher:
        batch_metrics = app.state.batcher.get_batch_metrics()
        metrics["components"]["batcher"] = batch_metrics
    
    return metrics

@app.post(f"{API_PREFIX}/batch")
async def process_batch(
    request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Procesa múltiples queries en batch."""
    try:
        queries = request.get("queries", [])
        if not queries:
            raise HTTPException(status_code=400, detail="Queries list is required")
        
        if len(queries) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 queries per batch")
        
        results = []
        
        for query_data in queries:
            query = query_data.get("query", "")
            priority = query_data.get("priority", "medium")
            
            if app.state.batcher:
                request_id = await app.state.batcher.submit_request(
                    content=query,
                    priority=priority
                )
                results.append({
                    "query": query,
                    "request_id": request_id,
                    "status": "queued"
                })
            else:
                results.append({
                    "query": query,
                    "request_id": None,
                    "status": "error",
                    "error": "Batcher not available"
                })
        
        return {
            "batch_id": f"batch_{int(time.time() * 1000)}",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error procesando batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{API_PREFIX}/cache/stats")
async def get_cache_stats(current_user: dict = Depends(get_current_user)):
    """Obtiene estadísticas del caché."""
    if not app.state.cache:
        raise HTTPException(status_code=503, detail="Cache not available")
    
    stats = app.state.cache.get_cache_stats()
    return stats

@app.delete(f"{API_PREFIX}/cache")
async def clear_cache(current_user: dict = Depends(get_current_user)):
    """Limpia el caché."""
    if not app.state.cache:
        raise HTTPException(status_code=503, detail="Cache not available")
    
    app.state.cache.clear()
    return {"message": "Cache cleared successfully"}

# Background tasks
async def log_query_metrics(metrics_data: Dict[str, Any]):
    """Registra métricas de query en background."""
    try:
        # En un entorno real, esto enviaría las métricas a un sistema de monitoreo
        logger.info(f"Query metrics: {metrics_data}")
    except Exception as e:
        logger.error(f"Error logging query metrics: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Manejador de excepciones HTTP."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Manejador de excepciones generales."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# Inicializar tiempo de inicio
app.state.start_time = time.time()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=ENVIRONMENT == "development",
        workers=1 if ENVIRONMENT == "development" else 4
    )
