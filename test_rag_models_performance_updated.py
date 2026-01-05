#!/usr/bin/env python3
"""
Script de prueba actualizado para evaluar el rendimiento del sistema RAG
con cada modelo individualmente, incluyendo traducción al español
"""

import requests
import time
import json
from typing import Dict, List, Tuple, Optional
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor


class RAGModelsTester:
    """Tester para evaluar el rendimiento de RAG con cada modelo"""
    
    def __init__(self, base_url: str = "http://localhost:8082"):
        self.base_url = base_url
        self.models = [
            'phi4_fast',
            'mistral_balanced', 
            'qwen_coder',
            'gemma3_multimodal',
            'aya_expanse_multilingual'
        ]
        
        # Consultas de prueba que probablemente activarían RAG
        self.rag_queries = [
            "What is the latest research on quantum computing?",
            "Explain the technical details of ARM architecture optimization",
            "Tell me about recent developments in machine learning"
        ]
        
        # Consultas que requieren traducción
        self.translation_queries = [
            "Please translate 'Hello world' to Spanish with explanation",
            "Explain quantum physics in Spanish", 
            "What are the benefits of renewable energy in Spanish"
        ]

    def get_model_info(self) -> Dict:
        """Obtener información sobre los modelos disponibles"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Error al obtener modelos: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            return {}

    def get_loaded_models(self) -> List[str]:
        """Obtener solo los modelos que están cargados"""
        try:
            response = requests.get(f"{self.base_url}/stats", timeout=10)
            if response.status_code == 200:
                stats = response.json()
                return stats.get("models_loaded", [])
            else:
                print(f"❌ Error al obtener stats: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error de conexión a stats: {e}")
            return []

    def test_model_rag(
        self, 
        model_id: str, 
        query: str, 
        translate: bool = True
    ) -> Dict:
        """Probar un modelo específico con RAG y opcionalmente traducción"""
        try:
            start_time = time.time()
            
            # Preparar el mensaje, opcionalmente incluyendo instrucción de traducción
            messages = [{"role": "user", "content": query}]
            
            if translate:
                # Añadir instrucción de traducción al español
                translation_msg = f"{query}\n\nPor favor responde en español."
                messages = [{"role": "user", "content": translation_msg}]
            
            payload = {
                "model": model_id,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 300,
                "stream": False
            }
            
            # Hacer la solicitud al servidor
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120
            )
            
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['choices'][0]['message']['content']
                
                return {
                    "success": True,
                    "model": model_id,
                    "query": query,
                    "response": generated_text,
                    "total_time": total_time,
                    "tokens_used": result.get('usage', {}).get('total_tokens', 0),
                    "translate": translate
                }
            else:
                return {
                    "success": False,
                    "model": model_id,
                    "query": query,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "total_time": total_time,
                    "translate": translate
                }
                
        except Exception as e:
            total_time = time.time() - start_time
            return {
                "success": False,
                "model": model_id,
                "query": query,
                "error": str(e),
                "total_time": total_time,
                "translate": translate
            }

    def run_comprehensive_test(self):
        """Correr pruebas completas con y sin traducción"""
        print("🚀 Iniciando pruebas de rendimiento RAG por modelo")
        print("="*80)
        
        # Obtener modelos disponibles
        models_info = self.get_model_info()
        if not models_info:
            print("❌ No se pudieron obtener los modelos disponibles")
            return
        
        available_models = [model['id'] for model in models_info.get('data', [])]
        print(f"✅ Modelos disponibles: {available_models}")
        
        # Obtener modelos cargados
        loaded_models = self.get_loaded_models()
        print(f"📦 Modelos cargados actualmente: {loaded_models}")
        
        # Filtrar modelos que están disponibles y cargarlos si es necesario
        all_models_to_test = available_models

        results = []
        
        # Probar cada modelo con cada tipo de consulta
        for model in all_models_to_test:
            print(f"\n🤖 Probando modelo: {model}")
            print("-" * 50)
            
            # Probar con consultas RAG (sin traducción)
            print("🔍 Consultas RAG (sin traducción):")
            for i, query in enumerate(self.rag_queries[:2]):  # Limitar a 2 por modelo
                print(f"  Query {i+1}: {query[:50]}...")
                result = self.test_model_rag(model, query, translate=False)
                results.append(result)
                
                if result['success']:
                    print(f"    ✅ {result['total_time']:.2f}s | Tokens: {result.get('tokens_used', 'N/A')}")
                    # Mostrar una muestra de la respuesta
                    print(f"    📝 Muestra: {result['response'][:100]}...")
                else:
                    print(f"    ❌ Error: {result['error']}")
            
            # Probar con consultas de traducción
            print("🌎 Consultas con traducción al español:")
            for i, query in enumerate(self.translation_queries[:2]):  # Limitar a 2 por modelo
                print(f"  Query {i+1}: {query[:50]}...")
                result = self.test_model_rag(model, query, translate=True)
                results.append(result)
                
                if result['success']:
                    print(f"    ✅ {result['total_time']:.2f}s | Tokens: {result.get('tokens_used', 'N/A')}")
                    # Mostrar una muestra de la respuesta
                    print(f"    📝 Muestra: {result['response'][:100]}...")
                else:
                    print(f"    ❌ Error: {result['error']}")
        
        # Mostrar resumen
        self.print_summary(results)
        return results

    def print_summary(self, results: List[Dict]):
        """Imprimir resumen de resultados"""
        print("\n" + "="*80)
        print("📊 RESUMEN DE RESULTADOS")
        print("="*80)
        
        # Agrupar por modelo
        model_results = {}
        for result in results:
            model = result['model']
            if model not in model_results:
                model_results[model] = []
            model_results[model].append(result)
        
        for model, model_tests in model_results.items():
            print(f"\n🤖 {model}:")
            print(f"  Total pruebas: {len(model_tests)}")
            
            successful_tests = [t for t in model_tests if t['success']]
            if successful_tests:
                avg_time = sum(t['total_time'] for t in successful_tests) / len(successful_tests)
                avg_tokens = sum(t.get('tokens_used', 0) for t in successful_tests) / len(successful_tests) if successful_tests else 0
                
                print(f"  Éxito: {len(successful_tests)}/{len(model_tests)}")
                print(f"  Tiempo promedio: {avg_time:.2f}s")
                print(f"  Tokens promedio: {avg_tokens:.1f}")
                
                # Separar por tipo de consulta
                rag_tests = [t for t in successful_tests if not t['translate']]
                if rag_tests:
                    avg_rag_time = sum(t['total_time'] for t in rag_tests) / len(rag_tests)
                    print(f"  - RAG (sin traducción): {avg_rag_time:.2f}s promedio")
                
                translation_tests = [t for t in successful_tests if t['translate']]
                if translation_tests:
                    avg_trans_time = sum(t['total_time'] for t in translation_tests) / len(translation_tests)
                    print(f"  - Traducción (al español): {avg_trans_time:.2f}s promedio")
                    
                    # Mostrar ejemplo de traducción exitosa si hay una
                    if len(translation_tests) > 0:
                        sample_response = translation_tests[0]['response']
                        print(f"  - Ejemplo de traducción:")
                        print(f"    '{sample_response[:200]}...'")
            else:
                print(f"  ❌ Todos los tests fallaron")
                
                # Para modelos que fallaron, mostrar el tipo de error más común
                if model_tests:
                    errors = [t.get('error', 'Unknown error') for t in model_tests]
                    # Encontrar el error más común
                    error_counts = {}
                    for error in errors:
                        error_counts[error] = error_counts.get(error, 0) + 1
                    
                    most_common_error = max(error_counts, key=error_counts.get)
                    print(f"  - Error más común: {most_common_error[:100]}...")
        
        # Estadísticas generales
        all_successful = [r for r in results if r['success']]
        if all_successful:
            total_avg_time = sum(r['total_time'] for r in all_successful) / len(all_successful)
            print(f"\n📈 ESTADÍSTICAS GLOBALES:")
            print(f"  Total pruebas: {len(results)}")
            print(f"  Pruebas exitosas: {len(all_successful)}")
            print(f"  Tasa de éxito: {len(all_successful)/len(results)*100:.1f}%")
            print(f"  Tiempo promedio total: {total_avg_time:.2f}s")
        else:
            print(f"\n❌ No se completó ninguna prueba exitosamente")
        
        print("\n" + "="*80)

    def save_results(self, results: List[Dict], filename: str = "rag_models_test_results.json"):
        """Guardar resultados en archivo JSON"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✅ Resultados guardados en {filename}")
        except Exception as e:
            print(f"❌ Error guardando resultados: {e}")


def main():
    """Función principal"""
    print("🧪 TEST DE RENDIMIENTO RAG POR MODELO")
    print("="*80)
    print("Este script probará:")
    print("- Cada modelo individualmente")
    print("- Consultas RAG (sin traducción)")
    print("- Consultas con traducción automática al español")
    print("- Medición de latencias y tokens")
    print("- Comparación de rendimiento entre modelos")
    print("="*80)
    
    # Crear tester
    tester = RAGModelsTester()
    
    # Correr pruebas
    results = tester.run_comprehensive_test()
    
    # Guardar resultados
    tester.save_results(results)
    
    print("\n✅ Pruebas completadas")


if __name__ == "__main__":
    main()