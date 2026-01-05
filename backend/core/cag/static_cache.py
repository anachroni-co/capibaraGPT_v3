#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StaticCache - Pre-loaded knowledge base con información estática.
"""

import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import pickle
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class StaticCache:
    """
    Pre-loaded knowledge base con información estática.
    Optimizado para recuperación rápida de conocimiento relevante.
    """
    
    def __init__(self, knowledge_dir: str = "backend/data/knowledge_base"):
        """
        Inicializa el StaticCache.
        
        Args:
            knowledge_dir: Directorio con archivos de conocimiento
        """
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        # Estructura de conocimiento
        self.knowledge = {}
        self.index = {}
        self.cache_file = self.knowledge_dir / "static_cache.pkl"
        
        # Cargar conocimiento
        self._load_knowledge()
        
        logger.info(f"StaticCache inicializado con {len(self.knowledge)} documentos")
    
    def _load_knowledge(self):
        """Carga conocimiento desde archivos y caché."""
        try:
            # Intentar cargar desde caché
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.knowledge = cached_data.get('knowledge', {})
                    self.index = cached_data.get('index', {})
                    logger.info(f"Conocimiento cargado desde caché: {len(self.knowledge)} documentos")
                    return
            
            # Cargar desde archivos
            self._load_from_files()
            
            # Construir índice
            self._build_index()
            
            # Guardar caché
            self._save_cache()
            
        except Exception as e:
            logger.error(f"Error cargando conocimiento: {e}")
            self.knowledge = {}
            self.index = {}
    
    def _load_from_files(self):
        """Carga conocimiento desde archivos en el directorio."""
        try:
            # Buscar archivos de conocimiento
            knowledge_files = list(self.knowledge_dir.glob("*.json"))
            
            if not knowledge_files:
                # Crear archivos de ejemplo si no existen
                self._create_example_knowledge()
                knowledge_files = list(self.knowledge_dir.glob("*.json"))
            
            for file_path in knowledge_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Procesar documento
                    doc_id = data.get('id', file_path.stem)
                    self.knowledge[doc_id] = {
                        'title': data.get('title', ''),
                        'content': data.get('content', ''),
                        'category': data.get('category', 'general'),
                        'tags': data.get('tags', []),
                        'source': str(file_path),
                        'created': data.get('created', datetime.now().isoformat()),
                        'updated': data.get('updated', datetime.now().isoformat())
                    }
                    
                except Exception as e:
                    logger.error(f"Error cargando archivo {file_path}: {e}")
            
            logger.info(f"Cargados {len(self.knowledge)} documentos desde archivos")
            
        except Exception as e:
            logger.error(f"Error cargando desde archivos: {e}")
    
    def _create_example_knowledge(self):
        """Crea archivos de conocimiento de ejemplo."""
        try:
            examples = [
                {
                    'id': 'python_basics',
                    'title': 'Conceptos Básicos de Python',
                    'content': 'Python es un lenguaje de programación interpretado, de alto nivel y de propósito general. Características principales: sintaxis simple, tipado dinámico, orientado a objetos, multiplataforma.',
                    'category': 'programming',
                    'tags': ['python', 'programming', 'basics', 'syntax']
                },
                {
                    'id': 'flask_intro',
                    'title': 'Introducción a Flask',
                    'content': 'Flask es un framework web ligero para Python. Permite crear aplicaciones web rápidamente con un mínimo de código. Características: microframework, flexible, extensible.',
                    'category': 'programming',
                    'tags': ['flask', 'python', 'web', 'framework']
                },
                {
                    'id': 'sql_basics',
                    'title': 'Fundamentos de SQL',
                    'content': 'SQL (Structured Query Language) es un lenguaje estándar para gestionar bases de datos relacionales. Operaciones básicas: SELECT, INSERT, UPDATE, DELETE.',
                    'category': 'database',
                    'tags': ['sql', 'database', 'queries', 'relational']
                },
                {
                    'id': 'api_design',
                    'title': 'Diseño de APIs REST',
                    'content': 'APIs REST siguen principios de arquitectura REST. Características: stateless, cacheable, uniform interface, client-server. Usa métodos HTTP: GET, POST, PUT, DELETE.',
                    'category': 'api',
                    'tags': ['api', 'rest', 'http', 'design']
                },
                {
                    'id': 'docker_basics',
                    'title': 'Conceptos de Docker',
                    'content': 'Docker es una plataforma de contenedores que permite empaquetar aplicaciones y sus dependencias. Componentes principales: imágenes, contenedores, Dockerfile, registries.',
                    'category': 'devops',
                    'tags': ['docker', 'containers', 'devops', 'deployment']
                }
            ]
            
            for example in examples:
                file_path = self.knowledge_dir / f"{example['id']}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(example, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Creados {len(examples)} archivos de conocimiento de ejemplo")
            
        except Exception as e:
            logger.error(f"Error creando conocimiento de ejemplo: {e}")
    
    def _build_index(self):
        """Construye índice para búsqueda rápida."""
        try:
            self.index = {
                'by_category': {},
                'by_tags': {},
                'by_title': {},
                'content_hashes': {}
            }
            
            for doc_id, doc in self.knowledge.items():
                # Índice por categoría
                category = doc['category']
                if category not in self.index['by_category']:
                    self.index['by_category'][category] = []
                self.index['by_category'][category].append(doc_id)
                
                # Índice por tags
                for tag in doc['tags']:
                    if tag not in self.index['by_tags']:
                        self.index['by_tags'][tag] = []
                    self.index['by_tags'][tag].append(doc_id)
                
                # Índice por título (palabras clave)
                title_words = doc['title'].lower().split()
                for word in title_words:
                    if len(word) > 2:  # Ignorar palabras muy cortas
                        if word not in self.index['by_title']:
                            self.index['by_title'][word] = []
                        self.index['by_title'][word].append(doc_id)
                
                # Hash del contenido para detección de duplicados
                content_hash = hashlib.md5(doc['content'].encode()).hexdigest()
                self.index['content_hashes'][content_hash] = doc_id
            
            logger.info("Índice construido exitosamente")
            
        except Exception as e:
            logger.error(f"Error construyendo índice: {e}")
    
    def _save_cache(self):
        """Guarda caché en disco."""
        try:
            cache_data = {
                'knowledge': self.knowledge,
                'index': self.index,
                'cached_at': datetime.now().isoformat()
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.debug("Caché guardado exitosamente")
            
        except Exception as e:
            logger.error(f"Error guardando caché: {e}")
    
    def retrieve(self, query: str, max_tokens: int = 2000, 
                category: str = None, tags: List[str] = None) -> str:
        """
        Recupera conocimiento relevante limitado por tokens.
        
        Args:
            query: Query de búsqueda
            max_tokens: Máximo número de tokens
            category: Categoría específica (opcional)
            tags: Tags específicos (opcional)
            
        Returns:
            Texto de contexto relevante
        """
        try:
            # Buscar documentos relevantes
            relevant_docs = self._search_relevant_docs(query, category, tags)
            
            # Ordenar por relevancia
            relevant_docs.sort(key=lambda x: x['relevance'], reverse=True)
            
            # Construir contexto respetando límite de tokens
            context_parts = []
            current_tokens = 0
            
            for doc in relevant_docs:
                doc_content = f"**{doc['title']}**\n{doc['content']}\n"
                doc_tokens = len(doc_content.split())
                
                if current_tokens + doc_tokens <= max_tokens:
                    context_parts.append(doc_content)
                    current_tokens += doc_tokens
                else:
                    # Truncar si es necesario
                    remaining_tokens = max_tokens - current_tokens
                    if remaining_tokens > 50:  # Solo si queda espacio significativo
                        truncated_content = ' '.join(doc_content.split()[:remaining_tokens])
                        context_parts.append(truncated_content + "...")
                    break
            
            context = "\n\n".join(context_parts)
            
            logger.debug(f"Contexto recuperado: {len(context_parts)} documentos, "
                        f"{current_tokens} tokens")
            
            return context
            
        except Exception as e:
            logger.error(f"Error recuperando conocimiento: {e}")
            return ""
    
    def _search_relevant_docs(self, query: str, category: str = None, 
                            tags: List[str] = None) -> List[Dict[str, Any]]:
        """Busca documentos relevantes para la query."""
        try:
            query_lower = query.lower()
            query_words = query_lower.split()
            
            relevant_docs = []
            
            for doc_id, doc in self.knowledge.items():
                # Filtros
                if category and doc['category'] != category:
                    continue
                
                if tags and not any(tag in doc['tags'] for tag in tags):
                    continue
                
                # Calcular relevancia
                relevance = self._calculate_relevance(query_words, doc)
                
                if relevance > 0:
                    relevant_docs.append({
                        'id': doc_id,
                        'title': doc['title'],
                        'content': doc['content'],
                        'category': doc['category'],
                        'relevance': relevance
                    })
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error buscando documentos relevantes: {e}")
            return []
    
    def _calculate_relevance(self, query_words: List[str], doc: Dict[str, Any]) -> float:
        """Calcula relevancia de un documento para la query."""
        try:
            relevance = 0.0
            
            # Buscar en título (peso alto)
            title_lower = doc['title'].lower()
            for word in query_words:
                if word in title_lower:
                    relevance += 2.0
            
            # Buscar en contenido (peso medio)
            content_lower = doc['content'].lower()
            for word in query_words:
                if word in content_lower:
                    relevance += 1.0
            
            # Buscar en tags (peso alto)
            for word in query_words:
                if word in [tag.lower() for tag in doc['tags']]:
                    relevance += 1.5
            
            # Normalizar por longitud del documento
            doc_length = len(doc['content'].split())
            if doc_length > 0:
                relevance = relevance / (doc_length / 100)  # Normalizar por 100 palabras
            
            return relevance
            
        except Exception as e:
            logger.error(f"Error calculando relevancia: {e}")
            return 0.0
    
    def add_document(self, doc_id: str, title: str, content: str, 
                    category: str = "general", tags: List[str] = None):
        """
        Agrega un nuevo documento al conocimiento.
        
        Args:
            doc_id: ID único del documento
            title: Título del documento
            content: Contenido del documento
            category: Categoría del documento
            tags: Tags del documento
        """
        try:
            if tags is None:
                tags = []
            
            # Verificar si ya existe
            if doc_id in self.knowledge:
                logger.warning(f"Documento {doc_id} ya existe, actualizando")
            
            # Agregar documento
            self.knowledge[doc_id] = {
                'title': title,
                'content': content,
                'category': category,
                'tags': tags,
                'source': 'manual',
                'created': datetime.now().isoformat(),
                'updated': datetime.now().isoformat()
            }
            
            # Reconstruir índice
            self._build_index()
            
            # Guardar caché
            self._save_cache()
            
            logger.info(f"Documento {doc_id} agregado exitosamente")
            
        except Exception as e:
            logger.error(f"Error agregando documento: {e}")
    
    def get_categories(self) -> List[str]:
        """Retorna lista de categorías disponibles."""
        return list(self.index['by_category'].keys())
    
    def get_tags(self) -> List[str]:
        """Retorna lista de tags disponibles."""
        return list(self.index['by_tags'].keys())
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retorna un documento específico."""
        return self.knowledge.get(doc_id)
    
    def search_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Busca documentos por categoría."""
        doc_ids = self.index['by_category'].get(category, [])
        return [self.knowledge[doc_id] for doc_id in doc_ids if doc_id in self.knowledge]
    
    def search_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Busca documentos por tag."""
        doc_ids = self.index['by_tags'].get(tag, [])
        return [self.knowledge[doc_id] for doc_id in doc_ids if doc_id in self.knowledge]
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del StaticCache."""
        return {
            'total_documents': len(self.knowledge),
            'categories': len(self.index['by_category']),
            'tags': len(self.index['by_tags']),
            'cache_file_size_mb': self.cache_file.stat().st_size / (1024 * 1024) 
                                if self.cache_file.exists() else 0,
            'knowledge_dir': str(self.knowledge_dir)
        }
    
    def refresh_cache(self):
        """Refresca el caché desde archivos."""
        try:
            self._load_from_files()
            self._build_index()
            self._save_cache()
            logger.info("Caché refrescado exitosamente")
        except Exception as e:
            logger.error(f"Error refrescando caché: {e}")


# Función de conveniencia
def create_static_cache(knowledge_dir: str = None) -> StaticCache:
    """Crea una instancia de StaticCache."""
    if knowledge_dir is None:
        knowledge_dir = "backend/data/knowledge_base"
    return StaticCache(knowledge_dir)


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.INFO)
    
    # Crear StaticCache
    cache = create_static_cache()
    
    # Test recuperación
    context = cache.retrieve("python programming", max_tokens=500)
    print("Contexto recuperado:")
    print(context)
    
    # Test estadísticas
    stats = cache.get_stats()
    print(f"\nEstadísticas: {stats}")
    
    # Test categorías
    categories = cache.get_categories()
    print(f"Categorías: {categories}")
