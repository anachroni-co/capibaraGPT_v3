#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector Store - Wrapper para FAISS/Chroma con funcionalidades avanzadas.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import pickle
import os
from pathlib import Path
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS no disponible, usando implementación básica")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB no disponible")


class Document:
    """Representa un documento en el vector store."""
    
    def __init__(self, content: str, metadata: Dict[str, Any] = None, doc_id: str = None):
        """
        Inicializa un documento.
        
        Args:
            content: Contenido del documento
            metadata: Metadata del documento
            doc_id: ID único del documento
        """
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id or self._generate_id()
        self.created_at = datetime.now()
    
    def _generate_id(self) -> str:
        """Genera ID único basado en contenido."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"doc_{content_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte documento a diccionario."""
        return {
            'doc_id': self.doc_id,
            'content': self.content,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Crea documento desde diccionario."""
        doc = cls(
            content=data['content'],
            metadata=data.get('metadata', {}),
            doc_id=data.get('doc_id')
        )
        if 'created_at' in data:
            doc.created_at = datetime.fromisoformat(data['created_at'])
        return doc


class VectorStore:
    """
    Wrapper unificado para FAISS y ChromaDB con funcionalidades avanzadas.
    """
    
    def __init__(self, store_type: str = "faiss", 
                 index_path: str = "backend/data/vector_store",
                 embedding_dim: int = 384):
        """
        Inicializa el VectorStore.
        
        Args:
            store_type: Tipo de store ("faiss" o "chromadb")
            index_path: Ruta para almacenar índices
            embedding_dim: Dimensión de los embeddings
        """
        self.store_type = store_type.lower()
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        
        # Inicializar store
        self.index = None
        self.documents = {}
        self.metadata_index = {}
        
        self._initialize_store()
        
        logger.info(f"VectorStore inicializado: {self.store_type}, "
                   f"dimensión: {embedding_dim}")
    
    def _initialize_store(self):
        """Inicializa el store según el tipo."""
        try:
            if self.store_type == "faiss" and FAISS_AVAILABLE:
                self._initialize_faiss()
            elif self.store_type == "chromadb" and CHROMADB_AVAILABLE:
                self._initialize_chromadb()
            else:
                # Fallback: usar implementación básica
                self._initialize_basic()
                
        except Exception as e:
            logger.error(f"Error inicializando store: {e}")
            self._initialize_basic()
    
    def _initialize_faiss(self):
        """Inicializa FAISS."""
        try:
            # Crear índice FAISS
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (cosine similarity)
            
            # Cargar índice existente si existe
            faiss_file = self.index_path / "faiss_index.bin"
            if faiss_file.exists():
                self.index = faiss.read_index(str(faiss_file))
                logger.info(f"Índice FAISS cargado: {self.index.ntotal} vectores")
            
            # Cargar documentos y metadata
            self._load_documents()
            
        except Exception as e:
            logger.error(f"Error inicializando FAISS: {e}")
            raise
    
    def _initialize_chromadb(self):
        """Inicializa ChromaDB."""
        try:
            # Configurar ChromaDB
            settings = Settings(
                persist_directory=str(self.index_path / "chromadb"),
                anonymized_telemetry=False
            )
            
            self.chroma_client = chromadb.Client(settings)
            self.collection = self.chroma_client.get_or_create_collection(
                name="capibara6_documents",
                metadata={"description": "Documentos para Capibara6 RAG"}
            )
            
            logger.info(f"ChromaDB inicializado: {self.collection.count()} documentos")
            
        except Exception as e:
            logger.error(f"Error inicializando ChromaDB: {e}")
            raise
    
    def _initialize_basic(self):
        """Inicializa implementación básica (sin FAISS/ChromaDB)."""
        try:
            self.index = None
            self.embeddings = []
            self.document_ids = []
            
            # Cargar datos básicos
            self._load_documents()
            
            logger.info("Implementación básica inicializada")
            
        except Exception as e:
            logger.error(f"Error inicializando implementación básica: {e}")
            raise
    
    def _load_documents(self):
        """Carga documentos desde disco."""
        try:
            docs_file = self.index_path / "documents.pkl"
            if docs_file.exists():
                with open(docs_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', {})
                    self.metadata_index = data.get('metadata_index', {})
                
                logger.info(f"Documentos cargados: {len(self.documents)}")
            
        except Exception as e:
            logger.error(f"Error cargando documentos: {e}")
            self.documents = {}
            self.metadata_index = {}
    
    def _save_documents(self):
        """Guarda documentos en disco."""
        try:
            docs_file = self.index_path / "documents.pkl"
            data = {
                'documents': self.documents,
                'metadata_index': self.metadata_index,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(docs_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug("Documentos guardados")
            
        except Exception as e:
            logger.error(f"Error guardando documentos: {e}")
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """
        Agrega documentos al vector store.
        
        Args:
            documents: Lista de documentos
            embeddings: Embeddings de los documentos
        """
        try:
            if len(documents) != len(embeddings):
                raise ValueError("Número de documentos y embeddings no coincide")
            
            if self.store_type == "faiss" and FAISS_AVAILABLE:
                self._add_to_faiss(documents, embeddings)
            elif self.store_type == "chromadb" and CHROMADB_AVAILABLE:
                self._add_to_chromadb(documents, embeddings)
            else:
                self._add_to_basic(documents, embeddings)
            
            # Actualizar documentos y metadata
            for doc in documents:
                self.documents[doc.doc_id] = doc
                self._update_metadata_index(doc)
            
            # Guardar
            self._save_documents()
            
            logger.info(f"Agregados {len(documents)} documentos al vector store")
            
        except Exception as e:
            logger.error(f"Error agregando documentos: {e}")
            raise
    
    def _add_to_faiss(self, documents: List[Document], embeddings: np.ndarray):
        """Agrega documentos a FAISS."""
        try:
            # Normalizar embeddings para cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Agregar al índice
            self.index.add(embeddings.astype('float32'))
            
            # Guardar índice
            faiss_file = self.index_path / "faiss_index.bin"
            faiss.write_index(self.index, str(faiss_file))
            
        except Exception as e:
            logger.error(f"Error agregando a FAISS: {e}")
            raise
    
    def _add_to_chromadb(self, documents: List[Document], embeddings: np.ndarray):
        """Agrega documentos a ChromaDB."""
        try:
            # Preparar datos para ChromaDB
            ids = [doc.doc_id for doc in documents]
            contents = [doc.content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Agregar a colección
            self.collection.add(
                ids=ids,
                documents=contents,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )
            
        except Exception as e:
            logger.error(f"Error agregando a ChromaDB: {e}")
            raise
    
    def _add_to_basic(self, documents: List[Document], embeddings: np.ndarray):
        """Agrega documentos a implementación básica."""
        try:
            for doc, embedding in zip(documents, embeddings):
                self.document_ids.append(doc.doc_id)
                self.embeddings.append(embedding)
            
        except Exception as e:
            logger.error(f"Error agregando a implementación básica: {e}")
            raise
    
    def _update_metadata_index(self, document: Document):
        """Actualiza índice de metadata."""
        try:
            for key, value in document.metadata.items():
                if key not in self.metadata_index:
                    self.metadata_index[key] = {}
                
                if value not in self.metadata_index[key]:
                    self.metadata_index[key][value] = []
                
                self.metadata_index[key][value].append(document.doc_id)
                
        except Exception as e:
            logger.error(f"Error actualizando índice de metadata: {e}")
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5, 
                         filter_metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Busca documentos similares.
        
        Args:
            query_embedding: Embedding de la query
            k: Número de resultados
            filter_metadata: Filtros de metadata
            
        Returns:
            Lista de documentos similares
        """
        try:
            if self.store_type == "faiss" and FAISS_AVAILABLE:
                return self._search_faiss(query_embedding, k, filter_metadata)
            elif self.store_type == "chromadb" and CHROMADB_AVAILABLE:
                return self._search_chromadb(query_embedding, k, filter_metadata)
            else:
                return self._search_basic(query_embedding, k, filter_metadata)
                
        except Exception as e:
            logger.error(f"Error en búsqueda de similitud: {e}")
            return []
    
    def _search_faiss(self, query_embedding: np.ndarray, k: int, 
                     filter_metadata: Dict[str, Any] = None) -> List[Document]:
        """Búsqueda en FAISS."""
        try:
            # Normalizar query embedding
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Buscar
            scores, indices = self.index.search(query_embedding, k)
            
            # Obtener documentos
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    doc_id = list(self.documents.keys())[idx]
                    doc = self.documents[doc_id]
                    
                    # Aplicar filtros de metadata
                    if self._matches_filter(doc, filter_metadata):
                        results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda FAISS: {e}")
            return []
    
    def _search_chromadb(self, query_embedding: np.ndarray, k: int,
                        filter_metadata: Dict[str, Any] = None) -> List[Document]:
        """Búsqueda en ChromaDB."""
        try:
            # Preparar filtros para ChromaDB
            where_clause = None
            if filter_metadata:
                where_clause = filter_metadata
            
            # Buscar
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=where_clause
            )
            
            # Convertir a documentos
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, doc_content in enumerate(results['documents'][0]):
                    doc_id = results['ids'][0][i]
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    
                    doc = Document(
                        content=doc_content,
                        metadata=metadata,
                        doc_id=doc_id
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error en búsqueda ChromaDB: {e}")
            return []
    
    def _search_basic(self, query_embedding: np.ndarray, k: int,
                     filter_metadata: Dict[str, Any] = None) -> List[Document]:
        """Búsqueda en implementación básica."""
        try:
            if not self.embeddings:
                return []
            
            # Calcular similitudes
            similarities = []
            for i, embedding in enumerate(self.embeddings):
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((similarity, i))
            
            # Ordenar por similitud
            similarities.sort(reverse=True)
            
            # Obtener top k
            results = []
            for similarity, idx in similarities[:k]:
                if idx < len(self.document_ids):
                    doc_id = self.document_ids[idx]
                    doc = self.documents[doc_id]
                    
                    # Aplicar filtros de metadata
                    if self._matches_filter(doc, filter_metadata):
                        results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda básica: {e}")
            return []
    
    def _matches_filter(self, document: Document, filter_metadata: Dict[str, Any]) -> bool:
        """Verifica si documento coincide con filtros."""
        if not filter_metadata:
            return True
        
        try:
            for key, value in filter_metadata.items():
                if key not in document.metadata:
                    return False
                
                if document.metadata[key] != value:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verificando filtros: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Obtiene documento por ID."""
        return self.documents.get(doc_id)
    
    def delete_document(self, doc_id: str) -> bool:
        """Elimina documento por ID."""
        try:
            if doc_id in self.documents:
                del self.documents[doc_id]
                
                # Limpiar metadata index
                for key in self.metadata_index:
                    for value in self.metadata_index[key]:
                        if doc_id in self.metadata_index[key][value]:
                            self.metadata_index[key][value].remove(doc_id)
                
                self._save_documents()
                logger.info(f"Documento {doc_id} eliminado")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error eliminando documento: {e}")
            return False
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any]) -> List[Document]:
        """Busca documentos por metadata."""
        try:
            results = []
            
            for doc_id, doc in self.documents.items():
                if self._matches_filter(doc, metadata_filter):
                    results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Error buscando por metadata: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del vector store."""
        try:
            stats = {
                'store_type': self.store_type,
                'total_documents': len(self.documents),
                'embedding_dimension': self.embedding_dim,
                'index_path': str(self.index_path)
            }
            
            if self.store_type == "faiss" and self.index:
                stats['faiss_vectors'] = self.index.ntotal
            elif self.store_type == "chromadb" and hasattr(self, 'collection'):
                stats['chromadb_documents'] = self.collection.count()
            elif self.store_type == "basic":
                stats['basic_vectors'] = len(self.embeddings)
            
            # Metadata stats
            stats['metadata_keys'] = list(self.metadata_index.keys())
            stats['metadata_entries'] = sum(
                len(values) for values in self.metadata_index.values()
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}
    
    def clear(self):
        """Limpia el vector store."""
        try:
            self.documents.clear()
            self.metadata_index.clear()
            
            if self.store_type == "faiss" and self.index:
                self.index.reset()
            elif self.store_type == "chromadb" and hasattr(self, 'collection'):
                # ChromaDB no tiene método clear directo
                pass
            elif self.store_type == "basic":
                self.embeddings.clear()
                self.document_ids.clear()
            
            self._save_documents()
            logger.info("Vector store limpiado")
            
        except Exception as e:
            logger.error(f"Error limpiando vector store: {e}")


# Funciones de conveniencia
def create_vector_store(store_type: str = "faiss", 
                       index_path: str = None,
                       embedding_dim: int = 384) -> VectorStore:
    """Crea una instancia de VectorStore."""
    if index_path is None:
        index_path = "backend/data/vector_store"
    return VectorStore(store_type, index_path, embedding_dim)


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.INFO)
    
    # Crear VectorStore
    store = create_vector_store("basic")  # Usar básico para test
    
    # Crear documentos de prueba
    docs = [
        Document("Python es un lenguaje de programación", {"category": "programming"}),
        Document("JavaScript se usa para desarrollo web", {"category": "programming"}),
        Document("SQL es para bases de datos", {"category": "database"})
    ]
    
    # Embeddings de prueba (simulados)
    embeddings = np.random.rand(len(docs), 384)
    
    # Agregar documentos
    store.add_documents(docs, embeddings)
    
    # Test búsqueda
    query_embedding = np.random.rand(384)
    results = store.similarity_search(query_embedding, k=2)
    
    print("=== Test VectorStore ===")
    print(f"Documentos encontrados: {len(results)}")
    for doc in results:
        print(f"- {doc.content[:50]}...")
    
    # Test estadísticas
    stats = store.get_stats()
    print(f"\nEstadísticas: {stats}")
