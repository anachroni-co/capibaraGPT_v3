-- Script de inicializaciÃ³n para PostgreSQL
-- Ajusta los nombres de esquema/tablas segÃºn tus necesidades antes del primer arranque.

CREATE SCHEMA IF NOT EXISTS rag_core;

CREATE TABLE IF NOT EXISTS rag_core.documents (
    id UUID PRIMARY KEY,
    source VARCHAR(255) NOT NULL,
    original_uri TEXT,
    title TEXT,
    content TEXT,
    metadata JSONB DEFAULT {}::JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rag_core.chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES rag_core.documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding_hash CHAR(64),
    metadata JSONB DEFAULT {}::JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rag_core.queries (
    id UUID PRIMARY KEY,
    query_text TEXT NOT NULL,
    response TEXT,
    metadata JSONB DEFAULT {}::JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON rag_core.chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hash ON rag_core.chunks(embedding_hash);

GRANT ALL PRIVILEGES ON SCHEMA rag_core TO rag_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA rag_core TO rag_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA rag_core TO rag_user;

