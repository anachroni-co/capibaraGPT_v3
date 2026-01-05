#!/usr/bin/env python3
"""
Parche para actualizar el LiveMind Orchestrator con el detector RAG solo para programación

Este parche modifica el archivo livemind_orchestrator.py para que el RAG
solo se active para consultas de programación, no para cualquier tipo de conocimiento.
"""

import re
from pathlib import Path

def patch_livemind_orchestrator():
    """Aplicar parche para usar detector solo de programación"""
    
    # Ruta al archivo original
    orchestrator_file = Path("/home/elect/capibara6/arm-axion-optimizations/vllm_integration/livemind_orchestrator.py")
    
    if not orchestrator_file.exists():
        print(f"❌ Error: No se encontró el archivo {orchestrator_file}")
        return False
    
    # Leer el contenido original
    content = orchestrator_file.read_text(encoding='utf-8')
    
    print("🔍 Analizando el archivo livemind_orchestrator.py...")
    
    # Importar el nuevo detector al principio del archivo
    import_section = '''import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass
import asyncio
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm_integration.vllm_axion_backend import (
    AxionVLLMEngine,
    AxionMultiExpertVLLM,
    AxionVLLMConfig
)
from vllm_integration.semantic_router import (
    IncrementalSemanticRouter,
    FastDomainClassifier,
    RoutingPrediction
)
'''
    
    # Añadir la importación del nuevo detector
    new_import = '''from vllm_integration.semantic_router import (
    IncrementalSemanticRouter,
    FastDomainClassifier,
    RoutingPrediction
)
from vllm_integration.programming_rag_detector import (
    ProgrammingRAGDetector,
    ProgrammingRAGParallelFetcher as ProgrammingRAGFetcher,
    is_programming_query
)
'''
    
    # Actualizar la importación
    updated_content = content.replace(
        'from vllm_integration.semantic_router import (\n    IncrementalSemanticRouter,\n    FastDomainClassifier,\n    RoutingPrediction\n)', 
        'from vllm_integration.semantic_router import (\n    IncrementalSemanticRouter,\n    FastDomainClassifier,\n    RoutingPrediction\n)\nfrom vllm_integration.programming_rag_detector import (\n    ProgrammingRAGDetector,\n    ProgrammingRAGParallelFetcher as ProgrammingRAGFetcher,\n    is_programming_query\n)'
    )
    
    # Si la importación aún no existe, la agregaremos
    if 'from vllm_integration.programming_rag_detector' not in updated_content:
        updated_content = content.replace(
            'from vllm_integration.semantic_router import (\n    IncrementalSemanticRouter,\n    FastDomainClassifier,\n    RoutingPrediction\n)',
            'from vllm_integration.semantic_router import (\n    IncrementalSemanticRouter,\n    FastDomainClassifier,\n    RoutingPrediction\n)\nfrom vllm_integration.programming_rag_detector import (\n    ProgrammingRAGDetector,\n    ProgrammingRAGParallelFetcher as ProgrammingRAGFetcher,\n    is_programming_query\n)'
        )
    
    # Actualizar el constructor para usar el nuevo detector
    # Cambiar la inicialización del RAG parallel fetcher
    rag_init_pattern_old = r'self\.rag_fetcher = RAGParallelFetcher\(\s*\n\s*bridge_url=rag_bridge_url,\s*\n\s*collection_name=rag_collection,\s*\n\s*enable_rag=True\s*\n\s*\)'
    
    rag_init_pattern_new = '''self.rag_fetcher = ProgrammingRAGFetcher(
                bridge_url=rag_bridge_url,
                collection_name=rag_collection,
                enable_rag=True
            )'''
    
    updated_content = re.sub(
        rag_init_pattern_old.replace('(', '\\(').replace(')', '\\)').replace('\n', '\\n').replace('*', '\\*').replace('+', '\\+'),
        rag_init_pattern_new,
        updated_content,
        flags=re.MULTILINE | re.DOTALL
    )
    
    if "ProgrammingRAGFetcher" not in updated_content:
        # Si no se pudo hacer el reemplazo preciso, intentaremos uno más general
        updated_content = updated_content.replace(
            'RAGParallelFetcher(',
            'ProgrammingRAGFetcher('
        )
    
    # Actualizar también el parámetro enable_rag en la clase
    # Asegurarnos de que el detector solo se inicialice si enable_rag es True
    init_code_pattern = r'if enable_rag:\s*\n(\s+.+?\n)+'
    # Ya debería estar correcto si hicimos el reemplazo anterior
    
    # Actualizar la sección de generación donde se verifica el RAG
    # En el método generate, donde se llama a rag_fetcher.detect_and_fetch
    generate_section_old = '''# Phase 1: PARALLEL processing - routing AND RAG fetch
        # Start RAG fetch in parallel (if enabled)
        rag_task = None
        if self.rag_fetcher:
            rag_task = asyncio.create_task(
                self.rag_fetcher.detect_and_fetch(request.prompt, request.request_id)
            )'''
    
    generate_section_new = '''# Phase 1: PARALLEL processing - routing AND Programming RAG fetch
        # Start Programming RAG fetch in parallel (if enabled)
        rag_task = None
        if self.rag_fetcher:
            rag_task = asyncio.create_task(
                self.rag_fetcher.detect_and_fetch(request.prompt, request.request_id)
            )'''
    
    updated_content = updated_content.replace(generate_section_old, generate_section_new)
    
    # Actualizar la parte del código que procesa el resultado del RAG
    rag_handling_old = '''# Wait for RAG fetch to complete (if started)
        is_rag_query = False
        rag_context = None
        if rag_task:
            is_rag_query, rag_context = await rag_task
            if rag_context:
                # Inject context into prompt
                request.prompt = self.rag_fetcher.inject_context(request.prompt, rag_context)
                print(f"✅ [{request.request_id}] RAG context injected ({rag_context.tokens_count} tokens)")'''
    
    rag_handling_new = '''# Wait for Programming RAG fetch to complete (if started)
        is_programming_query = False
        rag_context = None
        if rag_task:
            is_programming_query, rag_context = await rag_task
            if rag_context:
                # Inject context into prompt
                request.prompt = self.rag_fetcher.inject_context(request.prompt, rag_context)
                print(f"💻 [{request.request_id}] Programming RAG context injected ({rag_context.tokens_count} tokens)")
            elif is_programming_query:
                print(f"💻 [{request.request_id}] Programming query detected but no RAG context available")'''
    
    updated_content = updated_content.replace(rag_handling_old, rag_handling_new)
    
    # Similar update en la sección de streaming
    streaming_rag_handling_old = '''# Wait for RAG fetch to complete (if started)
        is_rag_query = False
        rag_context = None
        if rag_task:
            is_rag_query, rag_context = await rag_task
            if rag_context:
                # Inject context into prompt
                request.prompt = self.rag_fetcher.inject_context(request.prompt, rag_context)
                print(f"✅ [{request.request_id}] RAG context injected ({rag_context.tokens_count} tokens)")'''
    
    streaming_rag_handling_new = '''# Wait for Programming RAG fetch to complete (if started)
        is_programming_query = False
        rag_context = None
        if rag_task:
            is_programming_query, rag_context = await rag_task
            if rag_context:
                # Inject context into prompt
                request.prompt = self.rag_fetcher.inject_context(request.prompt, rag_context)
                print(f"💻 [{request.request_id}] Programming RAG context injected ({rag_context.tokens_count} tokens)")
            elif is_programming_query:
                print(f"💻 [{request.request_id}] Programming query detected but no RAG context available")'''
    
    updated_content = updated_content.replace(streaming_rag_handling_old, streaming_rag_handling_new)
    
    # Escribir el contenido actualizado al archivo
    backup_path = str(orchestrator_file) + ".backup_before_programming_rag"
    print(f"💾 Creando copia de seguridad en: {backup_path}")
    Path(backup_path).write_text(content, encoding='utf-8')
    
    print(f"📝 Escribiendo actualización al archivo: {orchestrator_file}")
    orchestrator_file.write_text(updated_content, encoding='utf-8')
    
    print("\n✅ Parche aplicado exitosamente!")
    print("\n📋 Cambios realizados:")
    print("   • Añadida importación del detector de programación")
    print("   • Reemplazado RAGParallelFetcher con ProgrammingRAGFetcher")
    print("   • Actualizadas secciones de detección y manejo de RAG")
    print("   • Ahora el RAG solo se activará para consultas de programación")
    
    return True

def create_instruction_file():
    """Crear archivo con instrucciones para activar el parche"""
    
    instructions = '''# Activación del Sistema RAG Solo para Programación

## Descripción
El sistema ha sido actualizado para que el RAG (Retrieval Augmented Generation) 
solo se active para consultas relacionadas con programación, no para cualquier 
tipo de conocimiento general.

## Funcionamiento
- El detector ahora identifica explícitamente consultas de programación
- Solo se activa RAG para consultas que involucren:
  * Código en lenguajes de programación
  * Sintaxis y semántica de lenguajes
  * Algoritmos e implementaciones
  * Depuración y resolución de errores
  * Documentación de APIs y bibliotecas
  * Frameworks y herramientas de desarrollo

## Archivos Actualizados
- `livemind_orchestrator.py`: Actualizado para usar ProgrammingRAGFetcher
- `programming_rag_detector.py`: Nuevo detector específico para programación

## Validación
Para validar el funcionamiento, se puede probar con:

1. Consultas de programación (deben activar RAG):
   - "¿Cómo implemento un algoritmo de ordenamiento en Python?"
   - "Necesito ayuda con un error en mi código JavaScript"
   - "Muestra un ejemplo de conexión a base de datos en Java"

2. Consultas generales (NO deben activar RAG):
   - "¿Cuál es la capital de Francia?"
   - "Explícame la teoría de la relatividad"
   - "¿Cómo cocinar una tortilla española?"

## Beneficios
- Menor latencia para consultas no técnicas
- Uso más eficiente de recursos
- Mejor enfoque en casos de uso específicos de programación
'''
    
    instruction_file = Path("/home/elect/capibara6/ACTIVATE_PROGRAMMING_ONLY_RAG.md")
    instruction_file.write_text(instructions, encoding='utf-8')
    
    print(f"\n📄 Instrucciones guardadas en: {instruction_file}")

if __name__ == "__main__":
    print("🔧 Aplicando parche para RAG exclusivo para programación")
    print("=" * 60)
    
    success = patch_livemind_orchestrator()
    
    if success:
        create_instruction_file()
        print(f"\n🎉 ¡Éxito! El parche ha sido aplicado correctamente.")
        print("   El sistema RAG ahora solo se activará para consultas de programación.")
    else:
        print(f"\n❌ Falló la aplicación del parche.")