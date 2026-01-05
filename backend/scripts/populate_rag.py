#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para cargar 10K documentos en el sistema RAG.
"""

import logging
import sys
import os
from pathlib import Path
import json
import numpy as np

# Agregar el directorio backend al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rag.vector_store import VectorStore, Document
from core.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


def create_sample_documents(count: int = 10000) -> list:
    """Crea documentos de muestra para el RAG."""
    
    # Categorías y contenido de ejemplo
    categories = {
        'programming': [
            'Python es un lenguaje de programación interpretado de alto nivel',
            'JavaScript es el lenguaje de programación de la web',
            'Java es un lenguaje orientado a objetos multiplataforma',
            'C++ es una extensión del lenguaje C con programación orientada a objetos',
            'Go es un lenguaje de programación desarrollado por Google',
            'Rust es un lenguaje de sistemas que enfatiza la seguridad',
            'TypeScript es un superconjunto tipado de JavaScript',
            'Kotlin es un lenguaje moderno para desarrollo Android',
            'Swift es el lenguaje de programación de Apple',
            'PHP es un lenguaje de programación para desarrollo web'
        ],
        'web_development': [
            'HTML es el lenguaje de marcado estándar para páginas web',
            'CSS se usa para estilizar y diseñar páginas web',
            'React es una biblioteca de JavaScript para interfaces de usuario',
            'Vue.js es un framework progresivo para JavaScript',
            'Angular es un framework de desarrollo web de Google',
            'Node.js permite ejecutar JavaScript en el servidor',
            'Express.js es un framework web minimalista para Node.js',
            'Django es un framework web de alto nivel para Python',
            'Flask es un microframework web para Python',
            'Laravel es un framework web elegante para PHP'
        ],
        'database': [
            'SQL es el lenguaje estándar para bases de datos relacionales',
            'PostgreSQL es una base de datos relacional de código abierto',
            'MySQL es un sistema de gestión de bases de datos popular',
            'MongoDB es una base de datos NoSQL orientada a documentos',
            'Redis es una base de datos en memoria de estructura de datos',
            'Elasticsearch es un motor de búsqueda y análisis distribuido',
            'Cassandra es una base de datos NoSQL distribuida',
            'SQLite es una base de datos SQL embebida',
            'Oracle es un sistema de gestión de bases de datos empresarial',
            'SQL Server es la base de datos de Microsoft'
        ],
        'devops': [
            'Docker es una plataforma de contenedores para aplicaciones',
            'Kubernetes es un sistema de orquestación de contenedores',
            'Jenkins es una herramienta de integración continua',
            'Git es un sistema de control de versiones distribuido',
            'AWS es la plataforma de computación en la nube de Amazon',
            'Azure es la plataforma de nube de Microsoft',
            'GCP es la plataforma de nube de Google',
            'Terraform es una herramienta de infraestructura como código',
            'Ansible es una herramienta de automatización de TI',
            'Prometheus es un sistema de monitoreo y alertas'
        ],
        'machine_learning': [
            'Machine Learning es un subcampo de la inteligencia artificial',
            'TensorFlow es una biblioteca de código abierto para ML',
            'PyTorch es un framework de aprendizaje automático',
            'Scikit-learn es una biblioteca de ML para Python',
            'Pandas es una biblioteca de análisis de datos para Python',
            'NumPy es una biblioteca fundamental para computación científica',
            'Matplotlib es una biblioteca de visualización para Python',
            'Jupyter es un entorno de desarrollo interactivo',
            'Keras es una API de redes neuronales de alto nivel',
            'OpenCV es una biblioteca de visión por computadora'
        ],
        'security': [
            'Ciberseguridad es la práctica de proteger sistemas digitales',
            'Autenticación es el proceso de verificar la identidad',
            'Autorización determina qué recursos puede acceder un usuario',
            'Encriptación es el proceso de codificar información',
            'HTTPS es el protocolo seguro para comunicación web',
            'Firewall es un sistema de seguridad de red',
            'VPN es una red privada virtual para conexiones seguras',
            'OAuth es un protocolo de autorización estándar',
            'JWT es un estándar para tokens de acceso seguros',
            'Penetration testing es la práctica de probar seguridad'
        ]
    }
    
    documents = []
    doc_id = 1
    
    # Crear documentos balanceados por categoría
    docs_per_category = count // len(categories)
    
    for category, contents in categories.items():
        for i in range(docs_per_category):
            # Seleccionar contenido base
            base_content = contents[i % len(contents)]
            
            # Crear variaciones
            variations = [
                f"{base_content} y es ampliamente utilizado en la industria",
                f"En el desarrollo moderno, {base_content.lower()}",
                f"Una característica importante es que {base_content.lower()}",
                f"Los desarrolladores utilizan {base_content.lower()} para crear aplicaciones",
                f"La popularidad de {base_content.lower()} ha crecido significativamente"
            ]
            
            content = variations[i % len(variations)]
            
            # Agregar detalles específicos
            if i % 3 == 0:
                content += " Es especialmente útil para proyectos de gran escala."
            elif i % 3 == 1:
                content += " Ofrece excelente rendimiento y flexibilidad."
            else:
                content += " Tiene una gran comunidad de desarrolladores."
            
            # Crear documento
            doc = Document(
                content=content,
                metadata={
                    'category': category,
                    'subcategory': f"{category}_{i % 5}",
                    'difficulty': ['beginner', 'intermediate', 'advanced'][i % 3],
                    'tags': [category, f"tag_{i % 10}", f"topic_{i % 20}"],
                    'source': 'sample_data',
                    'created_year': 2020 + (i % 4)
                },
                doc_id=f"doc_{doc_id:06d}"
            )
            
            documents.append(doc)
            doc_id += 1
    
    # Agregar documentos adicionales si es necesario
    while len(documents) < count:
        category = list(categories.keys())[len(documents) % len(categories)]
        content = f"Documento adicional sobre {category} - ID {len(documents) + 1}"
        
        doc = Document(
            content=content,
            metadata={
                'category': category,
                'subcategory': 'additional',
                'difficulty': 'intermediate',
                'tags': [category, 'additional'],
                'source': 'sample_data'
            },
            doc_id=f"doc_{doc_id:06d}"
        )
        
        documents.append(doc)
        doc_id += 1
    
    return documents[:count]


def populate_rag(vector_store_path: str = "backend/data/vector_store", 
                document_count: int = 10000):
    """Pobla el sistema RAG con documentos de muestra."""
    
    try:
        logger.info(f"🚀 Iniciando población de RAG con {document_count} documentos")
        
        # Crear directorio si no existe
        Path(vector_store_path).mkdir(parents=True, exist_ok=True)
        
        # Crear documentos de muestra
        logger.info("📝 Creando documentos de muestra...")
        documents = create_sample_documents(document_count)
        logger.info(f"✅ Creados {len(documents)} documentos")
        
        # Crear modelo de embeddings
        logger.info("🧠 Inicializando modelo de embeddings...")
        embedding_model = EmbeddingModel()
        logger.info("✅ Modelo de embeddings listo")
        
        # Generar embeddings
        logger.info("🔄 Generando embeddings...")
        contents = [doc.content for doc in documents]
        embeddings = embedding_model.encode(contents, batch_size=32)
        logger.info(f"✅ Generados {len(embeddings)} embeddings")
        
        # Crear vector store
        logger.info("🗄️ Creando vector store...")
        vector_store = VectorStore("basic", vector_store_path)
        logger.info("✅ Vector store creado")
        
        # Agregar documentos al vector store
        logger.info("📚 Agregando documentos al vector store...")
        vector_store.add_documents(documents, embeddings)
        logger.info("✅ Documentos agregados al vector store")
        
        # Verificar estadísticas
        stats = vector_store.get_stats()
        logger.info(f"📊 Estadísticas del vector store: {stats}")
        
        # Guardar información de población
        population_info = {
            'document_count': len(documents),
            'embedding_dimension': len(embeddings[0]) if len(embeddings) > 0 else 0,
            'vector_store_path': vector_store_path,
            'categories': list(set(doc.metadata['category'] for doc in documents)),
            'created_at': str(np.datetime64('now'))
        }
        
        info_file = Path(vector_store_path) / "population_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(population_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Información guardada en {info_file}")
        
        logger.info("🎉 ¡Población de RAG completada exitosamente!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en población de RAG: {e}")
        return False


def main():
    """Función principal."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Poblar sistema RAG con documentos de muestra')
    parser.add_argument('--count', type=int, default=10000, 
                       help='Número de documentos a crear (default: 10000)')
    parser.add_argument('--path', type=str, default='backend/data/vector_store',
                       help='Ruta del vector store (default: backend/data/vector_store)')
    
    args = parser.parse_args()
    
    success = populate_rag(args.path, args.count)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
