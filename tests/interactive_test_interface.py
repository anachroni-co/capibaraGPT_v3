#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interfaz Interactiva de Capibara6 - Router, Consenso y Modelos
Permite probar los 5 modelos por separado, el sistema de consenso y el router semántico
"""

import json
import time
import sys
import os
from typing import Dict, List, Any
import numpy as np
from sentence_transformers import SentenceTransformer

# Asegurar que estamos en el directorio correcto
sys.path.insert(0, '/home/elect/capibara6/vm-bounty2/core')
sys.path.insert(0, '/home/elect/capibara6/vm-bounty2/config')
sys.path.insert(0, '/home/elect/capibara6/backend')

# Importar componentes existentes
try:
    from ollama_client import VLLMClient
    print("✅ VLLMClient importado correctamente")
except ImportError as e:
    print(f"⚠️  No se pudo importar VLLMClient: {e}")
    print("⚠️  Usando modo simulado para pruebas")

from vm_bounty2.servers.consensus_server import ModelConsensus
from vm_bounty2.core.router.router import RouterModel20B


class InteractiveCapibara6:
    def __init__(self):
        # Cargar configuración
        self.load_model_config()
        
        # Inicializar componentes
        self.router = RouterModel20B()
        self.consensus = ModelConsensus()
        
        # Inicializar cliente VLLM
        try:
            self.client = VLLMClient(self.config)
            self.vllm_available = True
        except:
            self.vllm_available = False
            print("⚠️  Cliente VLLM no disponible, usando modo simulado")
        
        # Mapeo de claves internas a nombres reales
        self.model_key_to_name = {
            'fast_response': 'phi4:mini',
            'balanced': 'qwen2.5-coder-1.5b', 
            'complex': 'gpt-oss-20b'  # Aunque está reemplazado por gemma3, lo mantenemos para mostrar rutas
        }
        
        # Modelos en el sistema real (actualizado)
        self.real_models = [
            'phi4:mini',
            'qwen2.5-coder-1.5b', 
            'gemma-3-27b-it-awq',
            'mistral-7b-instruct-v0.2',
            'gpt-oss-20b'  # Tu modelo adicional
        ]

    def load_model_config(self):
        """Cargar configuración del modelo"""
        try:
            with open('/home/elect/capibara6/model_config.json', 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print("⚠️  Archivo model_config.json no encontrado")
            # Configuración simulada
            self.config = {
                'models': {
                    'fast_response': {
                        'name': 'phi4:mini',
                        'description': 'Modelo rápido para respuestas simples',
                        'max_tokens': 512,
                        'timeout': 8000
                    },
                    'balanced': {
                        'name': 'qwen2.5-coder-1.5b',
                        'description': 'Modelo experto en código y tareas técnicas',
                        'max_tokens': 1024,
                        'timeout': 20000
                    },
                    'complex': {
                        'name': 'gpt-oss-20b',
                        'description': 'Modelo más potente para tareas complejas',
                        'max_tokens': 2048,
                        'timeout': 240000
                    }
                }
            }

    def print_header(self, title: str):
        """Imprimir encabezado con formato"""
        print("\n" + "=" * 80)
        print(f"{title:^80}")
        print("=" * 80)

    def print_menu(self):
        """Imprimir menú principal"""
        self.print_header("INTERFAZ INTERACTIVA CAPIBARA6")
        print("Sistema de prueba para 5 modelos, router semántico y consenso")
        print("\nOpciones disponibles:")
        print("1. Probar modelo individual")
        print("2. Probar sistema de router semántico")
        print("3. Probar sistema de consenso")
        print("4. Probar todos los modelos con análisis comparativo")
        print("5. Información del sistema")
        print("6. Salir")
        print("-" * 80)

    def get_available_models(self):
        """Obtener lista de modelos disponibles"""
        models = list(self.config['models'].keys())
        return models

    def test_individual_model(self):
        """Probar un modelo individual"""
        self.print_header("PROBAR MODELO INDIVIDUAL")
        
        models = self.get_available_models()
        print("Modelos disponibles:")
        for i, model_key in enumerate(models, 1):
            model_name = self.config['models'][model_key]['name']
            print(f"  {i}. {model_key} ({model_name})")
        
        try:
            choice = input(f"\nSelecciona modelo (1-{len(models)}): ").strip()
            model_idx = int(choice) - 1
            
            if 0 <= model_idx < len(models):
                selected_model = models[model_idx]
                model_name = self.config['models'][selected_model]['name']
                
                print(f"\nProbando modelo: {selected_model} ({model_name})")
                query = input("Ingresa tu consulta: ").strip()
                
                if not query:
                    print("Consulta vacía, regresando al menú...")
                    return
                
                print(f"\nEnviando consulta a {model_name}...")
                
                if self.vllm_available:
                    try:
                        start_time = time.time()
                        result = self.client.generate(query, selected_model)
                        end_time = time.time()
                        
                        print(f"\n✅ Respuesta de {model_name}:")
                        print(f"   Tiempo: {end_time - start_time:.2f}s")
                        print(f"   Longitud: {len(result)} caracteres")
                        print(f"   Respuesta:\n{result}")
                        
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    # Modo simulado
                    print(f"\n🤖 Simulando respuesta de {model_name}:")
                    print(f"   Consulta: {query}")
                    print(f"   [Respuesta simulada - El modelo {model_name} procesaría esta consulta]")
                    print("   [Tiempo simulado: 0.5s]")
            else:
                print("Opción inválida")
        except ValueError:
            print("Entrada inválida, debes ingresar un número")

    def test_semantic_router(self):
        """Probar el router semántico"""
        self.print_header("SISTEMA DE ROUTER SEMÁNTICO")
        
        query = input("Ingresa tu consulta para análisis de routing: ").strip()
        if not query:
            print("Consulta vacía, regresando al menú...")
            return
        
        print(f"\nAnalizando consulta con el router semántico: '{query[:50]}...'")
        
        # Usar el router real
        try:
            routing_result = self.router.route_query(query)
            
            print(f"\n📊 RESULTADO DEL ROUTER:")
            print(f"   ID de consulta: {routing_result['query_id']}")
            print(f"   Modelo recomendado: {routing_result['recommended_model']}")
            print(f"   Nivel del modelo: {routing_result['model_tier']}")
            print(f"   Puntuación de complejidad: {routing_result['complexity_score']:.2f}")
            print(f"   Confianza de dominio: {routing_result['domain_confidence']:.2f}")
            print(f"   Relacionado con código: {routing_result['code_related']}")
            print(f"   Template E2B: {routing_result['e2b_template_suggestion']}")
            print(f"   Razonamiento: {routing_result['reasoning']}")
            
        except Exception as e:
            print(f"❌ Error en el router: {e}")

    def test_consensus_system(self):
        """Probar el sistema de consenso"""
        self.print_header("SISTEMA DE CONSENSO")
        
        query = input("Ingresa tu consulta para consenso: ").strip()
        if not query:
            print("Consulta vacía, regresando al menú...")
            return
        
        print(f"\nProcesando consenso para: '{query}'")
        
        # Determinar qué modelos usar para consenso
        print("\n¿Qué modelos quieres incluir en el consenso?")
        models = self.get_available_models()
        for i, model_key in enumerate(models, 1):
            model_name = self.config['models'][model_key]['name']
            print(f"  {i}. {model_key} ({model_name}) - {'Sí' if i <= 3 else 'No'}")
        
        # Para pruebas, usaremos los primeros 3 modelos
        selected_models = models[:3]  # Usar los primeros 3 modelos
        print(f"\nUsando modelos para consenso: {[self.config['models'][m]['name'] for m in selected_models]}")
        
        try:
            import asyncio
            result = asyncio.run(self.consensus.get_consensus(query, models=selected_models))
            
            print(f"\n📊 RESULTADO DEL CONSENSO:")
            if 'error' in result:
                print(f"   ❌ Error: {result['error']}")
                if 'failed_models' in result:
                    print(f"   Modelos fallidos: {result['failed_models']}")
            else:
                print(f"   Consenso alcanzado: {'✅' if result.get('consensus', False) else '➡️  Simple'}")
                print(f"   Modelo usado: {result.get('model_used', 'Desconocido')}")
                print(f"   Tiempo: {result.get('duration', 0):.2f}s")
                print(f"   Modelos consultados: {result.get('models_queried', 0)}")
                print(f"   Modelos exitosos: {result.get('successful_models', 0)}")
                print(f"   Método: {result.get('consensus_method', 'N/A')}")
                print(f"   Respuesta:\n{result.get('response', 'No disponible')[:300]}...")
                if len(result.get('response', '')) > 300:
                    print(f"   ... (truncado de {len(result.get('response', ''))} caracteres)")
                    
        except Exception as e:
            print(f"❌ Error en consenso: {e}")

    def test_all_models_comparison(self):
        """Probar todos los modelos con análisis comparativo"""
        self.print_header("ANÁLISIS COMPARATIVO DE MODELOS")
        
        query = input("Ingresa tu consulta para comparar modelos: ").strip()
        if not query:
            print("Consulta vacía, regresando al menú...")
            return
        
        print(f"\nProbando '{query[:50]}...' en todos los modelos...")
        
        # Probar con el router primero
        try:
            routing_result = self.router.route_query(query)
            recommended_model = routing_result['recommended_model']
            complexity = routing_result['complexity_score']
            
            print(f"\n🎯 RECOMENDACIÓN DEL ROUTER:")
            print(f"   Modelo recomendado: {recommended_model}")
            print(f"   Complejidad: {complexity:.2f}")
            print(f"   Razonamiento: {routing_result['reasoning']}")
        except:
            print("⚠️  Router no disponible")
            recommended_model = "Desconocido"
        
        # Probar con todos los modelos disponibles
        models = self.get_available_models()
        results = {}
        
        print(f"\n🧪 PROBANDO {len(models)} MODELOS:")
        for i, model_key in enumerate(models):
            model_name = self.config['models'][model_key]['name']
            print(f"\n  {i+1}/{len(models)}. {model_name} ({model_key})...")
            
            if self.vllm_available:
                try:
                    start_time = time.time()
                    result = self.client.generate(query, model_key)
                    end_time = time.time()
                    
                    results[model_key] = {
                        'response': result,
                        'time': end_time - start_time,
                        'length': len(result)
                    }
                    print(f"     ✅ Respondió en {end_time - start_time:.2f}s ({len(result)} chars)")
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    results[model_key] = {
                        'response': f"Error: {e}",
                        'time': 0,
                        'length': 0
                    }
            else:
                # Modo simulado
                results[model_key] = {
                    'response': f"[Simulado] {model_name} procesaría: {query[:30]}...",
                    'time': 0.5,
                    'length': 50
                }
                print(f"     🤖 Simulado (no disponible)")
        
        # Mostrar resultados comparativos
        print(f"\n📊 RESULTADOS COMPARATIVOS:")
        print("-" * 80)
        print(f"{'Modelo':<20} {'Tiempo (s)':<12} {'Chars':<8} {'Relevancia':<12}")
        print("-" * 80)
        
        for model_key, result in results.items():
            model_name = self.config['models'][model_key]['name']
            time_str = f"{result['time']:.2f}" if result['time'] > 0 else "N/A"
            chars_str = str(result['length']) if result['length'] > 0 else "N/A"
            
            # Simular relevancia basada en longitud y tiempo
            if result['time'] > 0:
                relevance = min(100, max(0, int(100 * (result['length'] / max(result['time'], 0.1)))))
            else:
                relevance = 0
                
            relevance_str = f"{relevance}%"
            
            marker = "🎯" if model_name == recommended_model else "  "
            print(f"{marker} {model_name:<18} {time_str:<12} {chars_str:<8} {relevance_str:<12}")

    def system_info(self):
        """Mostrar información del sistema"""
        self.print_header("INFORMACIÓN DEL SISTEMA")
        
        print("SISTEMA CAPIBARA6 - ARM AXION OPTIMIZADO")
        print("-" * 50)
        print("✓ 5 MODELOS CONFIGURADOS:")
        print("   • phi4:mini          - Respuestas rápidas")
        print("   • qwen2.5-coder-1.5b - Codificación y tareas técnicas") 
        print("   • gemma-3-27b-it-awq  - Multimodal y contexto largo")
        print("   • mistral-7b-instruct - Tareas generales")
        print("   • gpt-oss-20b         - Razonamiento complejo")
        
        print("\n✓ SISTEMAS DISPONIBLES:")
        print("   • Router semántico    - Análisis de complejidad y dominio")
        print("   • Sistema de consenso - Votación entre modelos")
        print("   • ARM-Axion optimizado - NEON + ACL + cuantización")
        
        print("\n✓ OPTIMIZACIONES ARM-Axion:")
        print("   • Kernels NEON optimizados")
        print("   • ARM Compute Library (ACL) integrada")
        print("   • Cuantización AWQ/GPTQ")
        print("   • Flash Attention")
        print("   • Matmul 8x8 tiles con prefetching")
        print("   • RMSNorm vectorizado (4-5x más rápido)")
        
        print(f"\n✓ ESTADO DEL VLLM: {'✅ Disponible' if self.vllm_available else '⚠️  No disponible'}")
        
        # Mostrar modelos disponibles
        models = self.get_available_models()
        print(f"\n✓ MODELOS CONFIGURADOS ({len(models)}):")
        for model_key in models:
            model_name = self.config['models'][model_key]['name']
            description = self.config['models'][model_key]['description']
            print(f"   • {model_key} ({model_name}): {description}")

    def run(self):
        """Ejecutar el bucle principal de la interfaz"""
        while True:
            self.print_menu()
            choice = input("\nElige una opción (1-6): ").strip()
            
            if choice == "1":
                self.test_individual_model()
            elif choice == "2":
                self.test_semantic_router()
            elif choice == "3":
                self.test_consensus_system()
            elif choice == "4":
                self.test_all_models_comparison()
            elif choice == "5":
                self.system_info()
            elif choice == "6":
                print("\n👋 ¡Gracias por usar la Interfaz Interactiva Capibara6!")
                print("Sistema optimizado para ARM-Axion con 5 modelos")
                break
            else:
                print("❌ Opción inválida. Por favor elige del 1 al 6.")
            
            input("\nPresiona Enter para continuar...")


def main():
    print("🚀 Iniciando Interfaz Interactiva Capibara6...")
    print("Sistema de pruebas para 5 modelos con ARM-Axion optimizado")
    
    try:
        app = InteractiveCapibara6()
        app.run()
    except KeyboardInterrupt:
        print("\n\n👋 Interfaz interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error en la interfaz: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()