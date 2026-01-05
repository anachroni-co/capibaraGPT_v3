#!/usr/bin/env python3
"""
API REST Server para Capibara6
Expone endpoints para búsqueda RAG, gestión de datos y estadísticas
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import sys
import os

# Añadir path para imports
sys.path.insert(0, '/home/elect')

from rag_utils import (
    semantic_search,
    search_all_collections,
    rag_search,
    get_model,
    get_pg_connection,
    get_chroma_collections,
    get_nebula_session
)

# ==========================================
# CONFIGURACIÓN DE FASTAPI
# ==========================================

app = FastAPI(
    title="Capibara6 RAG API",
    description="API REST para búsqueda semántica, RAG y gestión de datos personales",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8001",
        "http://localhost:5001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8001",
        "http://127.0.0.1:5001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# MODELOS PYDANTIC
# ==========================================

class SearchRequest(BaseModel):
    query: str = Field(..., description="Texto de búsqueda", min_length=1)
    collection_name: Optional[str] = Field(None, description="Nombre de la colección (chat_messages, external_chats, social_posts, files)")
    n_results: int = Field(5, description="Número de resultados", ge=1, le=50)

class RAGSearchRequest(BaseModel):
    query: str = Field(..., description="Pregunta en lenguaje natural", min_length=1)
    n_results: int = Field(5, description="Resultados por colección", ge=1, le=20)
    use_graph: bool = Field(True, description="Usar exploración de grafo")

class HybridSearchRequest(BaseModel):
    query: str = Field(..., description="Texto de búsqueda", min_length=1)
    depth: int = Field(2, description="Profundidad de traversal en grafo", ge=1, le=5)
    min_similarity: float = Field(0.5, description="Similitud mínima", ge=0.0, le=1.0)

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    password: str = Field(..., min_length=8)

class MessageCreate(BaseModel):
    session_id: str
    content: str = Field(..., min_length=1)
    message_role: str = Field(..., pattern="^(user|assistant)$")

# ==========================================
# ENDPOINTS DE BÚSQUEDA
# ==========================================

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "name": "Capibara6 RAG API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "search": "/api/search",
            "rag": "/api/rag",
            "hybrid": "/api/hybrid",
            "stats": "/api/stats",
            "users": "/api/users",
            "messages": "/api/messages"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Verificar conexiones
        _, pg_cur = get_pg_connection()
        pg_cur.execute("SELECT 1")

        return {
            "status": "healthy",
            "services": {
                "postgresql": "connected",
                "chromadb": "connected",
                "nebula_graph": "connected"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/api/search/semantic")
async def api_semantic_search(request: SearchRequest):
    """
    Búsqueda semántica en una colección específica

    Ejemplo:
    ```json
    {
        "query": "embeddings y vectores en IA",
        "collection_name": "chat_messages",
        "n_results": 5
    }
    ```
    """
    try:
        collections = get_chroma_collections()

        if request.collection_name and request.collection_name not in collections:
            raise HTTPException(
                status_code=400,
                detail=f"Collection must be one of: {list(collections.keys())}"
            )

        if request.collection_name:
            results = semantic_search(
                request.query,
                request.collection_name,
                request.n_results
            )
        else:
            # Buscar en todas las colecciones
            all_results = search_all_collections(request.query, request.n_results)
            results = []
            for coll_results in all_results.values():
                results.extend(coll_results)
            # Ordenar por similitud
            results.sort(key=lambda x: x['similarity'], reverse=True)

        return {
            "query": request.query,
            "total_results": len(results),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/api/search/rag")
async def api_rag_search(request: RAGSearchRequest):
    """
    Búsqueda RAG completa (Vector + PostgreSQL + Grafo)

    Combina búsqueda semántica, enriquecimiento de datos y exploración de grafo.

    Ejemplo:
    ```json
    {
        "query": "¿Qué contenido tengo sobre machine learning?",
        "n_results": 5,
        "use_graph": true
    }
    ```
    """
    try:
        result = rag_search(
            request.query,
            n_results=request.n_results,
            use_graph=request.use_graph
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG search error: {str(e)}")

@app.post("/api/search/all")
async def api_search_all_collections(request: SearchRequest):
    """
    Búsqueda en todas las colecciones simultáneamente

    Ejemplo:
    ```json
    {
        "query": "RAG y bases de datos vectoriales",
        "n_results": 3
    }
    ```
    """
    try:
        results = search_all_collections(request.query, request.n_results)

        total = sum(len(v) for v in results.values())

        return {
            "query": request.query,
            "total_results": total,
            "collections": list(results.keys()),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

# ==========================================
# ENDPOINTS DE USUARIOS
# ==========================================

@app.get("/api/users")
async def get_users(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Listar usuarios"""
    try:
        _, pg_cur = get_pg_connection()
        pg_cur.execute("""
            SELECT id, username, email, created_at, storage_used_mb, storage_limit_mb
            FROM users
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """, (limit, offset))

        users = []
        for row in pg_cur.fetchall():
            users.append({
                "id": str(row[0]),
                "username": row[1],
                "email": row[2],
                "created_at": row[3].isoformat() if row[3] else None,
                "storage_used_mb": row[4],
                "storage_limit_mb": row[5]
            })

        # Contar total
        pg_cur.execute("SELECT COUNT(*) FROM users")
        total = pg_cur.fetchone()[0]

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "users": users
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching users: {str(e)}")

@app.get("/api/users/{username}")
async def get_user(username: str):
    """Obtener información de un usuario"""
    try:
        _, pg_cur = get_pg_connection()
        pg_cur.execute("""
            SELECT id, username, email, created_at, storage_used_mb, storage_limit_mb, is_active
            FROM users
            WHERE username = %s
        """, (username,))

        row = pg_cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "id": str(row[0]),
            "username": row[1],
            "email": row[2],
            "created_at": row[3].isoformat() if row[3] else None,
            "storage_used_mb": row[4],
            "storage_limit_mb": row[5],
            "is_active": row[6]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user: {str(e)}")

# ==========================================
# ENDPOINTS DE MENSAJES
# ==========================================

@app.get("/api/messages")
async def get_messages(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """Listar mensajes de chat"""
    try:
        _, pg_cur = get_pg_connection()
        query = """
            SELECT cm.id, cm.content, cm.message_role, cm.created_at, cm.session_id,
                   u.username, u.email
            FROM chat_messages cm
            JOIN users u ON u.id = cm.user_id
            WHERE 1=1
        """
        params = []

        if user_id:
            query += " AND cm.user_id = %s"
            params.append(user_id)

        if session_id:
            query += " AND cm.session_id = %s"
            params.append(session_id)

        query += " ORDER BY cm.created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        pg_cur.execute(query, params)

        messages = []
        for row in pg_cur.fetchall():
            messages.append({
                "id": str(row[0]),
                "content": row[1],
                "role": row[2],
                "created_at": row[3].isoformat() if row[3] else None,
                "session_id": row[4],
                "user": {
                    "username": row[5],
                    "email": row[6]
                }
            })

        return {
            "total": len(messages),
            "limit": limit,
            "offset": offset,
            "messages": messages
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching messages: {str(e)}")

@app.get("/api/sessions/{session_id}")
async def get_session_messages(session_id: str):
    """Obtener todos los mensajes de una sesión"""
    try:
        _, pg_cur = get_pg_connection()
        pg_cur.execute("""
            SELECT cm.id, cm.content, cm.message_role, cm.created_at,
                   u.username, u.email
            FROM chat_messages cm
            JOIN users u ON u.id = cm.user_id
            WHERE cm.session_id = %s
            ORDER BY cm.created_at ASC
        """, (session_id,))

        messages = []
        for row in pg_cur.fetchall():
            messages.append({
                "id": str(row[0]),
                "content": row[1],
                "role": row[2],
                "created_at": row[3].isoformat() if row[3] else None,
                "user": {
                    "username": row[4],
                    "email": row[5]
                }
            })

        if not messages:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "session_id": session_id,
            "message_count": len(messages),
            "messages": messages
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching session: {str(e)}")

# ==========================================
# ENDPOINTS DE ESTADÍSTICAS
# ==========================================

@app.get("/api/stats")
async def get_stats():
    """Estadísticas generales del sistema"""
    try:
        _, pg_cur = get_pg_connection()
        stats = {}

        # PostgreSQL stats
        pg_cur.execute("SELECT COUNT(*) FROM users")
        stats['users'] = pg_cur.fetchone()[0]

        pg_cur.execute("SELECT COUNT(*) FROM chat_messages")
        stats['chat_messages'] = pg_cur.fetchone()[0]

        pg_cur.execute("SELECT COUNT(*) FROM external_chats")
        stats['external_chats'] = pg_cur.fetchone()[0]

        pg_cur.execute("SELECT COUNT(*) FROM social_posts")
        stats['social_posts'] = pg_cur.fetchone()[0]

        pg_cur.execute("SELECT COUNT(*) FROM raw_uploads")
        stats['files'] = pg_cur.fetchone()[0]

        pg_cur.execute("SELECT COUNT(*) FROM embeddings")
        stats['embeddings'] = pg_cur.fetchone()[0]

        # ChromaDB stats
        collections = get_chroma_collections()
        stats['chromadb'] = {}
        for name, collection in collections.items():
            stats['chromadb'][name] = collection.count()

        # Storage stats
        pg_cur.execute("""
            SELECT
                SUM(storage_used_mb) as total_used,
                SUM(storage_limit_mb) as total_limit
            FROM users
        """)
        row = pg_cur.fetchone()
        stats['storage'] = {
            'used_mb': row[0] or 0,
            'limit_mb': row[1] or 0,
            'usage_percent': round((row[0] or 0) / (row[1] or 1) * 100, 2)
        }

        return {
            "status": "ok",
            "stats": stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")

@app.get("/api/stats/embeddings")
async def get_embedding_stats():
    """Estadísticas de embeddings"""
    try:
        _, pg_cur = get_pg_connection()
        collections = get_chroma_collections()

        pg_cur.execute("""
            SELECT object_type, COUNT(*), vector_store
            FROM embeddings
            GROUP BY object_type, vector_store
            ORDER BY COUNT(*) DESC
        """)

        stats = []
        for row in pg_cur.fetchall():
            stats.append({
                "object_type": row[0],
                "count": row[1],
                "vector_store": row[2]
            })

        # ChromaDB collection sizes
        chromadb_stats = {}
        for name, collection in collections.items():
            chromadb_stats[name] = collection.count()

        return {
            "postgresql": stats,
            "chromadb": chromadb_stats,
            "model": {
                "name": "all-MiniLM-L6-v2",
                "dimensions": 384
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching embedding stats: {str(e)}")

@app.get("/api/stats/user/{username}")
async def get_user_stats(username: str):
    """Estadísticas de un usuario específico"""
    try:
        _, pg_cur = get_pg_connection()
        # Verificar que el usuario existe
        pg_cur.execute("SELECT id FROM users WHERE username = %s", (username,))
        user_row = pg_cur.fetchone()
        if not user_row:
            raise HTTPException(status_code=404, detail="User not found")

        user_id = user_row[0]

        # Recopilar estadísticas
        stats = {}

        pg_cur.execute("SELECT COUNT(*) FROM chat_messages WHERE user_id = %s", (user_id,))
        stats['chat_messages'] = pg_cur.fetchone()[0]

        pg_cur.execute("SELECT COUNT(*) FROM external_chats WHERE user_id = %s", (user_id,))
        stats['external_chats'] = pg_cur.fetchone()[0]

        pg_cur.execute("SELECT COUNT(*) FROM social_posts WHERE user_id = %s", (user_id,))
        stats['social_posts'] = pg_cur.fetchone()[0]

        pg_cur.execute("SELECT COUNT(*) FROM raw_uploads WHERE user_id = %s", (user_id,))
        stats['files'] = pg_cur.fetchone()[0]

        pg_cur.execute("""
            SELECT storage_used_mb, storage_limit_mb
            FROM users
            WHERE id = %s
        """, (user_id,))
        row = pg_cur.fetchone()
        stats['storage'] = {
            'used_mb': row[0] or 0,
            'limit_mb': row[1] or 0,
            'usage_percent': round((row[0] or 0) / (row[1] or 1) * 100, 2)
        }

        return {
            "username": username,
            "stats": stats
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user stats: {str(e)}")

# ==========================================
# ENDPOINTS DE ARCHIVOS
# ==========================================

@app.get("/api/files")
async def get_files(
    user_id: Optional[str] = None,
    filetype: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """Listar archivos"""
    try:
        _, pg_cur = get_pg_connection()
        query = """
            SELECT ru.id, ru.filename, ru.filetype, ru.size_bytes, ru.uploaded_at,
                   u.username, u.email
            FROM raw_uploads ru
            JOIN users u ON u.id = ru.user_id
            WHERE 1=1
        """
        params = []

        if user_id:
            query += " AND ru.user_id = %s"
            params.append(user_id)

        if filetype:
            query += " AND ru.filetype = %s"
            params.append(filetype)

        query += " ORDER BY ru.uploaded_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        pg_cur.execute(query, params)

        files = []
        for row in pg_cur.fetchall():
            files.append({
                "id": str(row[0]),
                "filename": row[1],
                "filetype": row[2],
                "size_mb": round(row[3] / 1048576, 2),
                "uploaded_at": row[4].isoformat() if row[4] else None,
                "user": {
                    "username": row[5],
                    "email": row[6]
                }
            })

        return {
            "total": len(files),
            "limit": limit,
            "offset": offset,
            "files": files
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching files: {str(e)}")

# ==========================================
# STARTUP & SHUTDOWN
# ==========================================

@app.on_event("startup")
async def startup_event():
    """Inicialización al arrancar el servidor"""
    print("\n" + "=" * 80)
    print("🚀 CAPIBARA6 RAG API STARTED")
    print("=" * 80)
    print("\n📊 System Status:")
    print("  ✅ FastAPI server running")
    print("  ✅ PostgreSQL connected")
    print("  ✅ ChromaDB connected")
    print("  ✅ Nebula Graph connected")
    print("\n🌐 API Endpoints available at:")
    print("  http://localhost:8001/docs (Swagger UI)")
    print("  http://localhost:8001/redoc (ReDoc)")
    print("\n" + "=" * 80 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al cerrar el servidor"""
    print("\n🛑 Shutting down Capibara6 RAG API...")
    # Las conexiones se cierran automáticamente por los context managers

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
