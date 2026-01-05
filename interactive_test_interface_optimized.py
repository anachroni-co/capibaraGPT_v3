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

# Asegurar que estamos en el directorio correcto
sys.path.insert(0, '.')

# Importar los componentes necesarios directamente
sys.path.insert(0, '/home/elect/capibara6/vm-bounty2/core')
sys.path.insert(0, '/home/elect/capibara6/vm-bounty2/config')
sys.path.insert(0, '/home/elect/capibara6/vm-bounty2/servers')


class SimpleRouter:
    """
    Router semántico simplificado para demostración
    """
    def __init__(self):
        # Dominios conocidos para cálculo de análisis
        self.known_domains = {
            'programming': [
                'python', 'javascript', 'java', 'c++', 'sql', 'html', 'css',
                'django', 'flask', 'react', 'node', 'api', 'database', 'algorithm',
                'function', 'code', 'codigo', 'program', 'script'
            ],
            'science': [
                'physics', 'chemistry', 'biology', 'mathematics', 'theory',
                'research', 'experiment', 'hypothesis', 'analysis', 'quantum',
                'neural network', 'deep learning', 'machine learning'
            ],
            'business': [
                'marketing', 'finance', 'strategy', 'management', 'sales',
                'revenue', 'profit', 'investment', 'market', 'business plan'
            ],
            'general': [
                'what', 'how', 'why', 'when', 'where', 'explain', 'describe',
                'help', 'question', 'answer', 'hello', 'hi', '2+2'
            ]
        }

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analiza la query y determina el modelo más apropiado"""
        query_lower = query.lower()
        
        # Detectar complejidad
        complexity = self._calculate_complexity(query)
        
        # Detectar dominio
        domain_scores = {}
        for domain, keywords in self.known_domains.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            domain_scores[domain] = score
        
        # Determinar dominio principal
        main_domain = max(domain_scores, key=domain_scores.get) if max(domain_scores.values()) > 0 else 'general'
        
        # Recomendar modelo basado en análisis
        if complexity > 0.7 or 'analysis' in query_lower or 'compare' in query_lower:
            recommended_model = 'complex'
            tier = 'complex'
            reasoning = f"Alta complejidad ({complexity:.2f}) o análisis profundo"
        elif complexity > 0.4 or 'code' in query_lower or 'python' in query_lower or 'javascript' in query_lower:
            recommended_model = 'balanced'
            tier = 'balanced'
            reasoning = f"Complejidad media ({complexity:.2f}) o contenido técnico"
        else:
            recommended_model = 'fast_response'
            tier = 'fast_response'
            reasoning = f"Baja complejidad ({complexity:.2f}) o consulta simple"
        
        return {
            'recommended_model': recommended_model,
            'model_tier': tier,
            'complexity_score': complexity,
            'main_domain': main_domain,
            'domain_scores': domain_scores,
            'code_related': 'code' in query_lower or 'function' in query_lower,
            'reasoning': reasoning
        }

    def _calculate_complexity(self, query: str) -> float:
        """Calcula la complejidad de la query (0.0 - 1.0)"""
        length_score = min(len(query) / 200, 1.0)  # Max 200 chars = 1.0
        
        # Palabras que indican complejidad
        complex_indicators = [
            'analyze', 'compare', 'evaluate', 'research', 'strategy',
            'algorithm', 'complex', 'detailed', 'comprehensive', 'deep',
            'multiple', 'several', 'various', 'different', 'relationship'
        ]
        
        indicator_count = sum(1 for indicator in complex_indicators if indicator in query.lower())
        indicator_score = min(indicator_count / 5, 1.0)  # Max 5 indicadores = 1.0
        
        # Combinar factores
        complexity = (length_score * 0.4) + (indicator_score * 0.6)
        return min(complexity, 1.0)


class SimpleConsensus:
    """
    Sistema de consenso simplificado para demostración
    """
    def __init__(self):
        self.models = ['phi4:mini', 'qwen2.5-coder-1.5b', 'gemma-3-27b-it-awq', 'mistral-7b-instruct-v0.2', 'gpt-oss-20b']
        
    def get_consensus(self, query: str, selected_models: List[str] = None) -> Dict[str, Any]:
        """Simula consenso entre modelos"""
        if selected_models is None:
            selected_models = self.models[:3]  # Usar primeros 3 modelos de ejemplo
        
        # Simular respuestas de diferentes modelos
        responses = {}
        for model in selected_models:
            # Simular tiempo de respuesta y calidad basada en el modelo
            if 'fast' in model.lower() or 'phi' in model.lower():
                response_time = 0.3
                quality = "Respuesta rápida y directa"
            elif 'code' in model.lower() or 'qwen' in model.lower():
                response_time = 0.5
                quality = "Explicación técnica detallada"
            elif 'gemma' in model.lower() or 'large' in model.lower():
                response_time = 0.8
                quality = "Análisis profundo y detallado"
            elif 'mistral' in model.lower():
                response_time = 0.4
                quality = "Explicación balanceada y clara"
            else:
                response_time = 0.6
                quality = "Análisis completo"
            
            responses[model] = {
                'response': f"[Simulado] {quality} para: {query[:20]}...",
                'time': response_time,
                'confidence': 0.8  # Simular confianza
            }
        
        # Para demostración, usar el modelo con mayor "confianza" (en este caso, el último)
        best_model = list(responses.keys())[-1]  # Último modelo como "ganador"
        
        return {
            'consensus': True,
            'consensus_method': 'weighted',
            'selected_model': best_model,
            'responses': responses,
            'models_queried': len(selected_models),
            'total_time': sum(r['time'] for r in responses.values()),
            'final_response': responses[best_model]['response']
        }


class InteractiveCapibara6:
    def __init__(self):
        # Cargar configuración
        self.load_model_config()
        
        # Inicializar componentes
        self.router = SimpleRouter()
        self.consensus = SimpleConsensus()
        
        # Modelos en el sistema
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
            with open('/home/elect/capibara6/five_model_config.json', 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Configuración simulada si no existe
            self.config = {
                'models': {
                    'fast_response': {
                        'name': 'phi4:mini',
                        'description': 'Modelo rápido para respuestas simples',
                        'max_tokens': 512,
                        'timeout': 8000,
                        'use_case': ["preguntas simples", "respuestas rápidas"]
                    },
                    'balanced': {
                        'name': 'qwen2.5-coder-1.5b',
                        'description': 'Modelo experto en código y tareas técnicas',
                        'max_tokens': 1024,
                        'timeout': 20000,
                        'use_case': ["explicaciones", "programación"]
                    },
                    'complex': {
                        'name': 'gpt-oss-20b',
                        'description': 'Modelo más potente para tareas complejas',
                        'max_tokens': 2048,
                        'timeout': 240000,
                        'use_case': ["análisis profundo", "razonamiento complejo"]
                    },
                    'multimodal': {
                        'name': 'gemma-3-27b-it-awq',
                        'description': 'Modelo multimodal para contexto largo',
                        'max_tokens': 32768,
                        'timeout': 60000,
                        'use_case': ["análisis multimodal", "contexto largo"]
                    },
                    'general': {
                        'name': 'mistral-7b-instruct-v0.2',
                        'description': 'Modelo general para tareas intermedias',
                        'max_tokens': 2048,
                        'timeout': 30000,
                        'use_case': ["explicaciones generales", "redacción"]
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
        print("ARM-Axion Optimizado con NEON + ACL + Cuantización")
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
            description = self.config['models'][model_key]['description']
            print(f"  {i}. {model_name}")
            print(f"     → {description}")
        
        try:
            choice = input(f"\nSelecciona modelo (1-{len(models)}): ").strip()
            model_idx = int(choice) - 1
            
            if 0 <= model_idx < len(models):
                selected_model = models[model_idx]
                model_name = self.config['models'][selected_model]['name']
                description = self.config['models'][selected_model]['description']
                
                print(f"\nProbando modelo: {model_name}")
                print(f"Descripción: {description}")
                
                query = input("\nIngresa tu consulta: ").strip()
                if not query:
                    print("Consulta vacía, regresando al menú...")
                    return
                
                print(f"\nSimulando consulta a {model_name}...")
                print(f"Consulta: {query}")
                
                # Simular respuesta del modelo basada en tipo de modelo
                if 'phi4' in model_name:
                    response = f"[phi4:mini] Simulación: Respondería rápidamente a '{query[:30]}...'"
                    time_taken = 0.2
                elif 'qwen' in model_name:
                    response = f"[qwen2.5-coder] Simulación: Explicaría técnicamente '{query[:30]}...'"
                    time_taken = 0.5
                elif 'gemma' in model_name:
                    response = f"[gemma-3-27b] Simulación: Haría un análisis profundo de '{query[:30]}...'"
                    time_taken = 0.8
                elif 'mistral' in model_name:
                    response = f"[mistral-7b] Simulación: Daría una respuesta balanceada sobre '{query[:30]}...'"
                    time_taken = 0.4
                else:  # gpt-oss-20b
                    response = f"[gpt-oss-20b] Simulación: Realizaría un análisis complejo de '{query[:30]}...'"
                    time_taken = 0.7
                
                print(f"\n✅ Simulación de respuesta:")
                print(f"   Tiempo estimado: {time_taken}s")
                print(f"   Longitud: ~{len(response)} caracteres")
                print(f"   Respuesta: {response}")
                
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
        
        # Usar el router
        try:
            routing_result = self.router.analyze_query(query)
            
            print(f"\n📊 RESULTADO DEL ROUTER:")
            print(f"   Modelo recomendado: {routing_result['recommended_model']}")
            print(f"   Nivel del modelo: {routing_result['model_tier']}")
            print(f"   Puntuación de complejidad: {routing_result['complexity_score']:.2f}")
            print(f"   Dominio principal: {routing_result['main_domain']}")
            print(f"   Relacionado con código: {routing_result['code_related']}")
            print(f"   Razonamiento: {routing_result['reasoning']}")
            
            # Mostrar puntuaciones por dominio
            print(f"\n   Puntuaciones por dominio:")
            for domain, score in routing_result['domain_scores'].items():
                if score > 0:
                    print(f"     • {domain}: {score}")
                    
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
            print(f"  {i}. {model_name}")
        
        # Preguntar qué modelos incluir (simulación)
        print(f"\nPara esta demo, usaremos los primeros 3 modelos:")
        selected_models = models[:3]  # Usar los primeros 3 modelos
        for model in selected_models:
            print(f"  • {self.config['models'][model]['name']}")
        
        try:
            result = self.consensus.get_consensus(query, selected_models)
            
            print(f"\n📊 RESULTADO DEL CONSENSO:")
            if 'error' in result:
                print(f"   ❌ Error: {result['error']}")
            else:
                print(f"   Consenso alcanzado: ✅")
                print(f"   Modelo seleccionado: {result['selected_model']}")
                print(f"   Total tiempo: {result['total_time']:.2f}s")
                print(f"   Modelos consultados: {result['models_queried']}")
                
                print(f"\n   RESPUESTAS INDIVIDUALES:")
                for model, data in result['responses'].items():
                    print(f"     • {model}: {data['response']} (t: {data['time']}s)")
                
                print(f"\n   RESPUESTA FINAL:")
                print(f"     {result['final_response']}")
                    
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
            routing_result = self.router.analyze_query(query)
            recommended_model = self.config['models'][routing_result['recommended_model']]['name']
            complexity = routing_result['complexity_score']
            
            print(f"\n🎯 RECOMENDACIÓN DEL ROUTER:")
            print(f"   Modelo recomendado: {recommended_model}")
            print(f"   Complejidad: {complexity:.2f}")
            print(f"   Dominio: {routing_result['main_domain']}")
        except:
            print("⚠️  Router no disponible")
            recommended_model = "Desconocido"
        
        # Simular pruebas con todos los modelos
        models = self.get_available_models()
        results = {}
        
        print(f"\n🧪 PROBANDO {len(models)} MODELOS:")
        for i, model_key in enumerate(models):
            model_name = self.config['models'][model_key]['name']
            print(f"\n  {i+1}/{len(models)}. {model_name}...")
            
            # Simular respuesta basada en tipo de modelo
            if 'phi4' in model_name:
                response = f"[Rápido] Respondería directamente a: {query[:20]}..."
                time_taken = 0.2
                length = len(response)
            elif 'qwen' in model_name:
                response = f"[Técnico] Explicaría detalladamente: {query[:20]}..."
                time_taken = 0.5
                length = len(response)
            elif 'gemma' in model_name:
                response = f"[Profundo] Analizaría profundamente: {query[:20]}..."
                time_taken = 0.8
                length = len(response)
            elif 'mistral' in model_name:
                response = f"[Balanceado] Explicaría claramente: {query[:20]}..."
                time_taken = 0.4
                length = len(response)
            else:  # gpt-oss-20b
                response = f"[Complejo] Razonaría extensamente: {query[:20]}..."
                time_taken = 0.7
                length = len(response)
            
            results[model_key] = {
                'response': response,
                'time': time_taken,
                'length': length
            }
            print(f"     ✅ Simulado: {time_taken}s ({length} chars)")
        
        # Mostrar resultados comparativos
        print(f"\n📊 RESULTADOS COMPARATIVOS:")
        print("-" * 80)
        print(f"{'Modelo':<20} {'Tiempo (s)':<12} {'Chars':<8} {'Tipo':<15}")
        print("-" * 80)
        
        for model_key, result in results.items():
            model_name = self.config['models'][model_key]['name']
            time_str = f"{result['time']:.2f}"
            chars_str = str(result['length'])
            
            # Determinar tipo de modelo
            if 'phi4' in model_name:
                model_type = "Rápido"
            elif 'qwen' in model_name:
                model_type = "Técnico"
            elif 'gemma' in model_name:
                model_type = "Profundo"
            elif 'mistral' in model_name:
                model_type = "Balanceado"
            else:
                model_type = "Complejo"
            
            marker = "🎯" if model_name == recommended_model else "  "
            print(f"{marker} {model_name[:18]:<18} {time_str:<12} {chars_str:<8} {model_type:<15}")

    def system_info(self):
        """Mostrar información del sistema"""
        self.print_header("INFORMACIÓN DEL SISTEMA")
        
        print("SISTEMA CAPIBARA6 - ARM AXION OPTIMIZADO")
        print("-" * 50)
        print("✅ 5 MODELOS CONFIGURADOS:")
        print("   • phi4:mini          - Respuestas rápidas (NEON optimized)")
        print("   • qwen2.5-coder-1.5b - Codificación y tareas técnicas (NEON + ACL)") 
        print("   • gemma-3-27b-it-awq  - Multimodal y contexto largo (NEON + ACL + Q4)")
        print("   • mistral-7b-instruct - Tareas generales (NEON optimized)")
        print("   • gpt-oss-20b         - Razonamiento complejo (NEON + ACL + Q4)")
        
        print("\n✅ SISTEMAS DISPONIBLES:")
        print("   • Router semántico    - Análisis de complejidad y dominio")
        print("   • Sistema de consenso - Votación entre modelos")
        print("   • ARM-Axion optimizado - NEON + ACL + cuantización")
        
        print("\n✅ OPTIMIZACIONES ARM-Axion:")
        print("   • Kernels NEON optimizados")
        print("   • ARM Compute Library (ACL) integrada")
        print("   • Cuantización AWQ/GPTQ (menos uso de memoria)")
        print("   • Flash Attention (mejor rendimiento en secuencias largas)")
        print("   • Matmul 8x8 tiles con prefetching (1.5x más rápido)")
        print("   • RMSNorm vectorizado (4-5x más rápido)")
        print("   • SwiGLU fusionado (1.35x más rápido)")
        print("   • Softmax con fast exp (1.4x más rápido)")
        
        print("\n✅ CONFIGURACIÓN:")
        print("   • Archivo de configuración: five_model_config.json")
        print("   • Router semántico: Simulador integrado")
        print("   • Consenso: Simulador de votación entre modelos")
        print("   • Endpoint: http://34.12.166.76:8000/v1")
        
        # Mostrar modelos disponibles
        models = self.get_available_models()
        print(f"\n✅ MODELOS CONFIGURADOS ({len(models)}):")
        for model_key in models:
            model_name = self.config['models'][model_key]['name']
            description = self.config['models'][model_key]['description']
            use_cases = ', '.join(self.config['models'][model_key]['use_case'][:2])
            print(f"   • {model_name}: {description}")
            print(f"     → Casos de uso: {use_cases}...")

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
                print("Sistema de 5 modelos con ARM-Axion optimizado")
                print("Incluye: phi4, qwen2.5, gemma3, mistral, gpt-oss-20b")
                break
            else:
                print("❌ Opción inválida. Por favor elige del 1 al 6.")
            
            input("\nPresiona Enter para continuar...")


def main():
    print("🚀 Iniciando Interfaz Interactiva Capibara6...")
    print("Sistema de pruebas para 5 modelos con ARM-Axion optimizado")
    print("✓ phi4:mini, qwen2.5-coder, gemma-3-27b, mistral-7b, gpt-oss-20b")
    print("✓ Router semántico, consenso, optimizaciones ARM")
    
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