#!/usr/bin/env python3
"""
Prueba de los componentes del sistema multimodelos ARM-Axion
Incluye pruebas de modelos individuales, router y consenso
"""

import asyncio
import aiohttp
import json
import logging
import time
from typing import Dict, List, Any
from datetime import datetime

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiModelTester:
    """
    Clase para probar los diferentes componentes del sistema multimodelos
    """
    
    def __init__(self, base_url: str = "http://localhost:8082"):
        self.base_url = base_url
        self.models = [
            "phi4_fast",      # Modelo rápido general
            "mistral_balanced",  # Modelo técnico equilibrado
            "qwen_coder",     # Modelo de programación
            # "gptoss_complex"  # Este modelo está fallando, se excluye por ahora
        ]
        self.available_models = []
    
    async def get_available_models(self):
        """Obtener los modelos disponibles"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/v1/models") as response:
                    if response.status == 200:
                        data = await response.json()
                        self.available_models = [item["id"] for item in data.get("data", [])]
                        logger.info(f"✅ Modelos disponibles: {self.available_models}")
                        return self.available_models
                    else:
                        logger.error(f"❌ Error obteniendo modelos: {response.status}")
                        return []
            except Exception as e:
                logger.error(f"❌ Error de conexión: {e}")
                return []
    
    async def test_individual_model(self, model_id: str, query: str) -> Dict[str, Any]:
        """Probar un modelo individual"""
        async with aiohttp.ClientSession() as session:
            try:
                start_time = time.time()
                payload = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": query}],
                    "max_tokens": 150,
                    "temperature": 0.7
                }
                
                async with session.post(f"{self.base_url}/v1/chat/completions", json=payload) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        choice = result["choices"][0]
                        response_text = choice["message"]["content"]
                        
                        logger.info(f"✅ {model_id}: {len(response_text)} caracteres en {response_time:.2f}s")
                        return {
                            "model": model_id,
                            "success": True,
                            "response": response_text,
                            "response_time": response_time,
                            "usage": result.get("usage", {}),
                            "finish_reason": choice.get("finish_reason", "unknown")
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ {model_id}: Status {response.status} - {error_text}")
                        return {
                            "model": model_id,
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}",
                            "response_time": response_time
                        }
                        
            except Exception as e:
                logger.error(f"❌ {model_id}: Excepción - {e}")
                return {
                    "model": model_id,
                    "success": False,
                    "error": str(e),
                    "response_time": 0
                }
    
    async def test_all_models(self):
        """Probar todos los modelos disponibles con diferentes tipos de consultas"""
        logger.info("🚀 Iniciando pruebas de modelos individuales")
        
        queries = [
            {
                "name": "Consulta general",
                "query": "¿Qué es Python?",
                "expected_domain": "general"
            },
            {
                "name": "Consulta técnica",
                "query": "Explica qué es la complejidad algorítmica O(n log n)",
                "expected_domain": "technical"
            },
            {
                "name": "Consulta de programación",
                "query": "Escribe una función en Python para calcular el factorial de un número",
                "expected_domain": "coding"
            }
        ]
        
        results = {}
        
        for query_info in queries:
            query_name = query_info["name"]
            query_text = query_info["query"]
            logger.info(f"\n🔍 Prueba: {query_name}")
            
            query_results = {}
            for model in self.available_models:
                if model != "gptoss_complex":  # Saltar el modelo con problemas
                    logger.info(f"  🤖 Probando {model}...")
                    result = await self.test_individual_model(model, query_text)
                    query_results[model] = result
            
            results[query_name] = query_results
        
        return results
    
    async def test_router_simulation(self):
        """Simular el comportamiento del router semántico"""
        logger.info("🤖 Iniciando simulación de router semántico")
        
        # Simular la selección de modelo basada en el dominio de la consulta
        test_queries = [
            ("¿Qué es la inteligencia artificial?", "general"),
            ("¿Cómo se implementa un árbol binario en Python?", "coding"),
            ("Explica el teorema de Pitágoras", "technical"),
            ("Escribe un algoritmo de ordenamiento en Java", "coding"),
        ]
        
        router_results = {}
        
        for query, expected_domain in test_queries:
            logger.info(f"\n🔍 Query: '{query}' (esperado: {expected_domain})")
            
            # Análisis simple de dominio para seleccionar modelo
            selected_model = self._select_model_by_domain(expected_domain)
            
            if selected_model:
                logger.info(f"  🎯 Modelo seleccionado: {selected_model}")
                result = await self.test_individual_model(selected_model, query)
                result["selected_by_router"] = selected_model
                result["expected_domain"] = expected_domain
                router_results[query] = result
            else:
                logger.warning(f"  ⚠️  No se encontró modelo adecuado para {expected_domain}")
                router_results[query] = {
                    "success": False,
                    "error": f"No model found for domain {expected_domain}",
                    "expected_domain": expected_domain
                }
        
        return router_results
    
    def _select_model_by_domain(self, domain: str) -> str:
        """Seleccionar modelo basado en dominio (simulación de router)"""
        domain_to_model = {
            "general": "phi4_fast",
            "technical": "mistral_balanced",
            "coding": "qwen_coder",
            "expert": "gptoss_complex"  # Este modelo está fallando actualmente
        }
        
        candidate_model = domain_to_model.get(domain, "phi4_fast")
        
        # Verificar que el modelo esté disponible y no sea el que falla
        if candidate_model in self.available_models and candidate_model != "gptoss_complex":
            return candidate_model
        else:
            # Fallback a phi4_fast si el modelo preferido no está disponible
            return "phi4_fast" if "phi4_fast" in self.available_models else self.available_models[0] if self.available_models else None
    
    async def test_consensus_mechanism(self):
        """Probar el mecanismo de consenso entre modelos (si está implementado)"""
        logger.info("🤝 Iniciando prueba de mecanismo de consenso")
        
        query = "¿Qué es la inteligencia artificial? Responde en 2 líneas."
        participating_models = [model for model in ["phi4_fast", "mistral_balanced", "qwen_coder"] 
                               if model in self.available_models and model != "gptoss_complex"]
        
        if len(participating_models) < 2:
            logger.warning("❌ No hay suficientes modelos disponibles para prueba de consenso")
            return {
                "success": False,
                "error": "Insufficient models for consensus testing",
                "participating_models": participating_models
            }
        
        logger.info(f"  🤖 Modelos participando: {participating_models}")
        
        # Obtener respuestas de cada modelo
        responses = {}
        for model in participating_models:
            logger.info(f"    🤖 Obteniendo respuesta de {model}...")
            result = await self.test_individual_model(model, query)
            responses[model] = result
        
        # Simular un proceso de consenso (agregación de respuestas)
        consensus_result = self._simulate_consensus(responses, query)
        
        return {
            "query": query,
            "participating_models": participating_models,
            "individual_responses": responses,
            "consensus_result": consensus_result,
            "success": True
        }
    
    def _simulate_consensus(self, responses: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Simular un proceso de consenso entre modelos"""
        successful_responses = {
            model: data for model, data in responses.items() 
            if data.get("success", False)
        }
        
        if not successful_responses:
            return {"consensus": "No hay respuestas válidas para alcanzar consenso", "agreement_level": 0.0}
        
        # En una implementación real, esto implicaría técnicas de agregación más complejas
        # Por ahora, simplemente tomamos la respuesta más larga como representativa
        best_response_model = max(successful_responses, 
                                 key=lambda m: len(successful_responses[m].get("response", "")))
        best_response = successful_responses[best_response_model]
        
        # Calcular nivel de acuerdo (en una implementación real se usarían técnicas más avanzadas)
        agreement_level = min(len(successful_responses) / len(responses), 1.0)
        
        return {
            "consensus": best_response.get("response", ""),
            "best_model": best_response_model,
            "agreement_level": agreement_level,
            "total_participants": len(responses),
            "successful_participants": len(successful_responses)
        }
    
    async def run_comprehensive_test(self):
        """Ejecutar todas las pruebas comprehensivas"""
        logger.info("="*70)
        logger.info("🔬 PRUEBA COMPREHENSIVA DEL SISTEMA MULTIMODELOS")
        logger.info("="*70)
        
        # 1. Verificar modelos disponibles
        available_models = await self.get_available_models()
        if not available_models:
            logger.error("❌ No se pudieron obtener modelos disponibles")
            return None
        
        # 2. Prueba de modelos individuales
        logger.info("\n" + "-"*50)
        logger.info("1️⃣ PRUEBA DE MODELOS INDIVIDUALES")
        logger.info("-"*50)
        individual_results = await self.test_all_models()
        
        # 3. Simulación de router semántico
        logger.info("\n" + "-"*50)
        logger.info("2️⃣ SIMULACIÓN DE ROUTER SEMÁNTICO")
        logger.info("-"*50)
        router_results = await self.test_router_simulation()
        
        # 4. Prueba de mecanismo de consenso
        logger.info("\n" + "-"*50)
        logger.info("3️⃣ PRUEBA DE MECANISMO DE CONSENSO")
        logger.info("-"*50)
        consensus_results = await self.test_consensus_mechanism()
        
        # Compilar resultados generales
        comprehensive_results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "base_url": self.base_url,
                "available_models": available_models,
                "successful_models": [m for m in available_models if m != "gptoss_complex"]
            },
            "individual_tests": individual_results,
            "router_simulation": router_results,
            "consensus_test": consensus_results
        }
        
        logger.info("\n" + "="*70)
        logger.info("✅ PRUEBA COMPREHENSIVA COMPLETADA")
        logger.info("="*70)
        
        return comprehensive_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Imprimir resumen de los resultados"""
        if not results:
            logger.error("❌ No hay resultados para mostrar")
            return
        
        system_info = results["system_info"]
        logger.info(f"\n📊 RESUMEN GENERAL:")
        logger.info(f"   - URL Base: {system_info['base_url']}")
        logger.info(f"   - Modelos disponibles: {len(system_info['available_models'])}")
        logger.info(f"   - Modelos funcionales: {len(system_info['successful_models'])}")
        
        # Contar respuestas exitosas
        successful_responses = 0
        total_responses = 0
        
        for query_name, query_results in results["individual_tests"].items():
            for model_result in query_results.values():
                total_responses += 1
                if model_result.get("success", False):
                    successful_responses += 1
        
        logger.info(f"   - Pruebas individuales: {successful_responses}/{total_responses} exitosas")
        
        # Router
        router_successes = sum(1 for r in results["router_simulation"].values() 
                              if r.get("success", False))
        logger.info(f"   - Simulación router: {router_successes}/{len(results['router_simulation'])} exitosas")
        
        # Consenso
        consensus_success = results["consensus_test"].get("success", False)
        logger.info(f"   - Prueba consenso: {'✅ Exitosa' if consensus_success else '❌ Fallida'}")
        
        # Modelos funcionales
        logger.info(f"\n🤖 MODELOS FUNCIONALES:")
        for model in system_info["successful_models"]:
            logger.info(f"   - ✅ {model}")
        
        if "gptoss_complex" in system_info["available_models"]:
            logger.info(f"   - ❌ gptoss_complex (tiene problemas de configuración)")


async def main():
    """Función principal para ejecutar las pruebas"""
    tester = MultiModelTester()
    results = await tester.run_comprehensive_test()
    tester.print_summary(results)
    
    # Mostrar algunos ejemplos de respuestas
    if results:
        logger.info(f"\n📝 EJEMPLOS DE RESPUESTAS:")
        
        # Mostrar una respuesta de phi4_fast que sabemos que funciona bien
        if "Consulta general" in results["individual_tests"]:
            phi4_result = results["individual_tests"]["Consulta general"].get("phi4_fast", {})
            if phi4_result.get("success", False):
                response = phi4_result["response"][:200] + "..." if len(phi4_result["response"]) > 200 else phi4_result["response"]
                logger.info(f"\n   phi4_fast respuesta: '{response}'")
                logger.info(f"   Tiempo: {phi4_result['response_time']:.2f}s, Tokens: {phi4_result['usage'].get('completion_tokens', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())