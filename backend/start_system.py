#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de inicio para el Sistema de Agentes IA Avanzado Capibara6.
"""

import logging
import sys
import os
from pathlib import Path
import time
import signal
from typing import Dict, Any

# Agregar el directorio backend al path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logging_config import setup_logging, get_logger
from core.router import RouterModel20B
from core.embeddings import EmbeddingModel
from core.thresholds import ThresholdManager
from core.cag.static_cache import StaticCache
from core.cag.dynamic_context import DynamicContext
from core.cag.awareness_gate import AwarenessGate
from core.cag.mini_cag import MiniCAG
from core.cag.full_cag import FullCAG
from core.rag.vector_store import VectorStore
from core.rag.mini_rag import MiniRAG
from core.rag.full_rag import FullRAG
from core.rag.guided_search import GuidedSearch

logger = logging.getLogger(__name__)


class Capibara6System:
    """Sistema principal de Capibara6."""
    
    def __init__(self):
        """Inicializa el sistema."""
        self.components = {}
        self.running = False
        
        # Setup logging
        self.logger = setup_logging("backend/logs", "INFO")
        self.logger.logger.info("🚀 Iniciando Sistema Capibara6")
    
    def initialize_components(self):
        """Inicializa todos los componentes del sistema."""
        try:
            self.logger.logger.info("🔧 Inicializando componentes...")
            
            # 1. Sistema de logging
            self.components['logging'] = self.logger
            self.logger.logger.info("✅ Sistema de logging inicializado")
            
            # 2. Modelo de embeddings
            self.logger.logger.info("🧠 Inicializando modelo de embeddings...")
            self.components['embeddings'] = EmbeddingModel()
            self.logger.logger.info("✅ Modelo de embeddings inicializado")
            
            # 3. Gestor de umbrales
            self.logger.logger.info("⚙️ Inicializando gestor de umbrales...")
            self.components['thresholds'] = ThresholdManager()
            self.logger.logger.info("✅ Gestor de umbrales inicializado")
            
            # 4. Router inteligente
            self.logger.logger.info("🛣️ Inicializando router inteligente...")
            self.components['router'] = RouterModel20B()
            self.logger.logger.info("✅ Router inteligente inicializado")
            
            # 5. Vector Store
            self.logger.logger.info("🗄️ Inicializando vector store...")
            self.components['vector_store'] = VectorStore("basic")
            self.logger.logger.info("✅ Vector store inicializado")
            
            # 6. Static Cache
            self.logger.logger.info("📚 Inicializando static cache...")
            self.components['static_cache'] = StaticCache()
            self.logger.logger.info("✅ Static cache inicializado")
            
            # 7. Dynamic Context
            self.logger.logger.info("🔄 Inicializando dynamic context...")
            self.components['dynamic_context'] = DynamicContext()
            self.logger.logger.info("✅ Dynamic context inicializado")
            
            # 8. Awareness Gate
            self.logger.logger.info("🎯 Inicializando awareness gate...")
            self.components['awareness_gate'] = AwarenessGate()
            self.logger.logger.info("✅ Awareness gate inicializado")
            
            # 9. MiniRAG
            self.logger.logger.info("🔍 Inicializando MiniRAG...")
            self.components['mini_rag'] = MiniRAG(
                self.components['vector_store'],
                self.components['embeddings']
            )
            self.logger.logger.info("✅ MiniRAG inicializado")
            
            # 10. FullRAG
            self.logger.logger.info("🔍 Inicializando FullRAG...")
            self.components['full_rag'] = FullRAG(
                self.components['vector_store'],
                self.components['embeddings'],
                self.components['mini_rag']
            )
            self.logger.logger.info("✅ FullRAG inicializado")
            
            # 11. Guided Search
            self.logger.logger.info("🎯 Inicializando guided search...")
            self.components['guided_search'] = GuidedSearch(
                self.components['mini_rag'],
                self.components['full_rag']
            )
            self.logger.logger.info("✅ Guided search inicializado")
            
            # 12. MiniCAG
            self.logger.logger.info("🧠 Inicializando MiniCAG...")
            self.components['mini_cag'] = MiniCAG(
                self.components['static_cache'],
                self.components['dynamic_context'],
                self.components['awareness_gate']
            )
            self.logger.logger.info("✅ MiniCAG inicializado")
            
            # 13. FullCAG
            self.logger.logger.info("🧠 Inicializando FullCAG...")
            self.components['full_cag'] = FullCAG(
                self.components['static_cache'],
                self.components['dynamic_context'],
                self.components['awareness_gate'],
                self.components['mini_rag'],
                self.components['full_rag']
            )
            self.logger.logger.info("✅ FullCAG inicializado")
            
            self.logger.logger.info("🎉 Todos los componentes inicializados exitosamente")
            return True
            
        except Exception as e:
            self.logger.logger.error(f"❌ Error inicializando componentes: {e}")
            return False
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Procesa una query usando el sistema completo."""
        try:
            if context is None:
                context = {}
            
            start_time = time.time()
            
            # 1. Decisión de routing
            router = self.components['router']
            should_escalate = router.should_escalate(query, context)
            
            # 2. Generación de contexto
            if should_escalate:
                # Usar FullCAG para queries complejas
                cag = self.components['full_cag']
                model_used = "120B"
            else:
                # Usar MiniCAG para queries simples
                cag = self.components['mini_cag']
                model_used = "20B"
            
            context_result = cag.generate_context(query, context)
            
            # 3. Búsqueda RAG si es necesario
            rag_results = []
            if context_result.get('sources_used') and 'rag' in context_result['sources_used']:
                guided_search = self.components['guided_search']
                rag_result = guided_search.search(query, context)
                rag_results = rag_result.get('results', [])
            
            # 4. Preparar respuesta
            total_latency = (time.time() - start_time) * 1000
            
            response = {
                'query': query,
                'model_used': model_used,
                'context': context_result.get('context', ''),
                'context_tokens': context_result.get('tokens_used', 0),
                'rag_results': len(rag_results),
                'total_latency_ms': total_latency,
                'components_used': {
                    'router': True,
                    'cag': True,
                    'rag': len(rag_results) > 0
                },
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Log de la decisión
            self.logger.log_routing_decision(
                query_hash=hash(query) % 1000000,  # Hash simple
                complexity=0.8 if should_escalate else 0.4,
                domain_conf=0.7,
                decision="escalate" if should_escalate else "use_20b",
                model_used=model_used,
                latency_ms=total_latency,
                success=True
            )
            
            return response
            
        except Exception as e:
            self.logger.logger.error(f"❌ Error procesando query: {e}")
            return {
                'query': query,
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Retorna el estado del sistema."""
        try:
            status = {
                'running': self.running,
                'components': {},
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Verificar estado de cada componente
            for name, component in self.components.items():
                if hasattr(component, 'get_stats'):
                    status['components'][name] = component.get_stats()
                elif hasattr(component, 'is_available'):
                    status['components'][name] = {'available': component.is_available()}
                else:
                    status['components'][name] = {'status': 'initialized'}
            
            return status
            
        except Exception as e:
            self.logger.logger.error(f"❌ Error obteniendo estado del sistema: {e}")
            return {'error': str(e)}
    
    def start(self):
        """Inicia el sistema."""
        try:
            self.logger.logger.info("🚀 Iniciando Sistema Capibara6...")
            
            # Inicializar componentes
            if not self.initialize_components():
                self.logger.logger.error("❌ Error inicializando componentes")
                return False
            
            self.running = True
            self.logger.logger.info("✅ Sistema Capibara6 iniciado exitosamente")
            
            # Mostrar estado inicial
            status = self.get_system_status()
            self.logger.logger.info(f"📊 Componentes activos: {len(status['components'])}")
            
            return True
            
        except Exception as e:
            self.logger.logger.error(f"❌ Error iniciando sistema: {e}")
            return False
    
    def stop(self):
        """Detiene el sistema."""
        try:
            self.logger.logger.info("🛑 Deteniendo Sistema Capibara6...")
            self.running = False
            
            # Limpiar recursos si es necesario
            for name, component in self.components.items():
                if hasattr(component, 'clear_cache'):
                    component.clear_cache()
                elif hasattr(component, 'reset_stats'):
                    component.reset_stats()
            
            self.logger.logger.info("✅ Sistema Capibara6 detenido")
            
        except Exception as e:
            self.logger.logger.error(f"❌ Error deteniendo sistema: {e}")


def signal_handler(signum, frame):
    """Maneja señales del sistema."""
    logger.info(f"🛑 Recibida señal {signum}, deteniendo sistema...")
    if 'system' in globals():
        system.stop()
    sys.exit(0)


def main():
    """Función principal."""
    global system
    
    # Configurar manejo de señales
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Crear e iniciar sistema
    system = Capibara6System()
    
    if not system.start():
        logger.error("❌ Error iniciando sistema")
        sys.exit(1)
    
    # Modo interactivo para testing
    logger.info("🎮 Sistema en modo interactivo. Escribe 'quit' para salir.")
    logger.info("📝 Ejemplo: ¿Qué es Python?")
    
    try:
        while system.running:
            try:
                query = input("\n> ").strip()
                
                if query.lower() in ['quit', 'exit', 'salir']:
                    break
                
                if not query:
                    continue
                
                # Procesar query
                response = system.process_query(query)
                
                # Mostrar respuesta
                print(f"\n🤖 Respuesta ({response.get('model_used', 'unknown')}):")
                print(f"📊 Latencia: {response.get('total_latency_ms', 0):.1f}ms")
                print(f"🧠 Contexto: {response.get('context_tokens', 0)} tokens")
                print(f"🔍 RAG: {response.get('rag_results', 0)} resultados")
                
                if 'error' in response:
                    print(f"❌ Error: {response['error']}")
                else:
                    print(f"📝 Contexto generado: {len(response.get('context', ''))} caracteres")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ Error en loop principal: {e}")
    
    except Exception as e:
        logger.error(f"❌ Error en main: {e}")
    
    finally:
        system.stop()


if __name__ == "__main__":
    main()
