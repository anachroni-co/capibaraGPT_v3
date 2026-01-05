#!/usr/bin/env python3
"""
Script de pruebas de latencia optimizado para servidor multi-modelo ARM Axion
Prueba cada modelo individualmente y genera un reporte detallado
Implementa medidas para evitar la saturación del servidor:
- Limita la concurrencia
- Añade retrasos entre pruebas
- Ajusta timeouts
- Implementa manejo de errores robusto
"""

import requests
import time
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import statistics
import random

# Configuración
SERVER_URL = "http://localhost:8082"
TEST_PROMPTS = {
    "simple": "Hola, ¿cómo estás?",
    "technical": "Explica qué es el algoritmo de ordenamiento quicksort",
    "coding": "Escribe una función Python para calcular números primos",
    "multilingual": "Translate 'Hello, how are you?' to Spanish, French, and German",
    "complex": "Analiza las ventajas y desventajas de usar arquitecturas de microservicios vs monolitos"
}

@dataclass
class LatencyResult:
    """Resultado de prueba de latencia"""
    model: str
    prompt_type: str
    success: bool
    ttft: float  # Time to first token (tiempo hasta recibir respuesta)
    total_time: float
    tokens_generated: int
    tokens_per_second: float
    error: str = ""
    response_preview: str = ""


class LatencyTester:
    """Tester de latencia para modelos con medidas anti-saturación"""

    def __init__(self, base_url: str, max_concurrent: int = 1, delay_between_tests: float = 2.0):
        self.base_url = base_url
        self.results: List[LatencyResult] = []
        self.max_concurrent = max_concurrent  # Limitar concurrencia
        self.delay_between_tests = delay_between_tests  # Retraso entre pruebas

    def get_models(self) -> List[Dict]:
        """Obtener lista de modelos disponibles"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=30)  # Timeout más corto
            response.raise_for_status()
            return response.json()["data"]
        except requests.exceptions.Timeout:
            print(f"❌ Timeout obteniendo modelos")
            return []
        except Exception as e:
            print(f"❌ Error obteniendo modelos: {e}")
            return []

    def test_model(self, model_id: str, prompt: str, prompt_type: str, max_tokens: int = 50) -> LatencyResult:
        """Probar latencia de un modelo específico con manejo de errores robusto"""
        print(f"  Testing {model_id} with {prompt_type} prompt...", end=" ", flush=True)

        start_time = time.time()

        try:
            # Ajuste de timeout a un valor más razonable
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=60  # Reducido de 120 a 60 segundos para evitar bloqueos largos
            )

            ttft = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                total_time = time.time() - start_time

                # Extraer información de la respuesta
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "")

                # Tokens generados
                usage = data.get("usage", {})
                tokens_generated = usage.get("completion_tokens", len(content.split()))

                # Performance metrics (con protección contra división por cero)
                generation_time = total_time  # Usamos total_time en lugar de performance.get()
                tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0

                print(f"✅ {total_time:.2f}s ({tokens_per_second:.2f} tok/s)")

                return LatencyResult(
                    model=model_id,
                    prompt_type=prompt_type,
                    success=True,
                    ttft=ttft,
                    total_time=total_time,
                    tokens_generated=tokens_generated,
                    tokens_per_second=tokens_per_second,
                    response_preview=content[:100] + "..." if len(content) > 100 else content
                )
            elif response.status_code == 429:  # Too Many Requests
                print(f"❌ Too Many Requests (429)")
                return LatencyResult(
                    model=model_id,
                    prompt_type=prompt_type,
                    success=False,
                    ttft=0,
                    total_time=time.time() - start_time,
                    tokens_generated=0,
                    tokens_per_second=0,
                    error=f"Too Many Requests (429): Server is overloaded"
                )
            else:
                print(f"❌ Status {response.status_code}")
                return LatencyResult(
                    model=model_id,
                    prompt_type=prompt_type,
                    success=False,
                    ttft=0,
                    total_time=time.time() - start_time,
                    tokens_generated=0,
                    tokens_per_second=0,
                    error=f"HTTP {response.status_code}: {response.text[:200]}"
                )

        except requests.exceptions.Timeout:
            print(f"⏱️ TIMEOUT (>60s)")
            return LatencyResult(
                model=model_id,
                prompt_type=prompt_type,
                success=False,
                ttft=0,
                total_time=time.time() - start_time,
                tokens_generated=0,
                tokens_per_second=0,
                error="Request timeout (>60s)"
            )
        except requests.exceptions.ConnectionError:
            print(f"🔌 Connection Error")
            return LatencyResult(
                model=model_id,
                prompt_type=prompt_type,
                success=False,
                ttft=0,
                total_time=time.time() - start_time,
                tokens_generated=0,
                tokens_per_second=0,
                error="Connection Error - Server may be down"
            )
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            return LatencyResult(
                model=model_id,
                prompt_type=prompt_type,
                success=False,
                ttft=0,
                total_time=time.time() - start_time,
                tokens_generated=0,
                tokens_per_second=0,
                error=str(e)
            )

    def run_all_tests(self, models: List[Dict], prompts: Dict[str, str], iterations: int = 1):
        """Ejecutar todas las pruebas con medidas anti-saturación"""
        print(f"\n{'='*70}")
        print(f"🧪 PRUEBAS DE LATENCIA OPTIMIZADAS - ARM AXION MULTI-MODEL SERVER")
        print(f"{'='*70}\n")
        print(f"Servidor: {self.base_url}")
        print(f"Modelos: {len(models)}")
        print(f"Prompts: {len(prompts)}")
        print(f"Iteraciones por prueba: {iterations}")
        print(f"Máx. concurrencia: {self.max_concurrent}")
        print(f"Retraso entre pruebas: {self.delay_between_tests}s")
        print(f"\n{'='*70}\n")

        for model in models:
            model_id = model["id"]
            print(f"\n🤖 Modelo: {model_id} ({model.get('domain', 'unknown')})")
            print(f"   Estado: {model.get('status', 'unknown')}")
            print(f"   {'-'*65}")

            for prompt_type, prompt_text in prompts.items():
                # Múltiples iteraciones para promediar (limitado a 1 por defecto para evitar sobrecarga)
                iteration_results = []
                for i in range(iterations):
                    # Aplicar retraso entre pruebas para evitar saturar el servidor
                    if i > 0:
                        print(f"   ⏳ Esperando {self.delay_between_tests}s entre iteraciones...")
                        time.sleep(self.delay_between_tests)
                    
                    result = self.test_model(model_id, prompt_text, prompt_type)
                    iteration_results.append(result)
                    self.results.append(result)
                    
                    # Si la prueba falló, considerar un retraso adicional para permitir recuperación del servidor
                    if not result.success:
                        print(f"   ⚠️  Prueba fallida, esperando adicionalmente {self.delay_between_tests}s...")
                        time.sleep(self.delay_between_tests)

                # Mostrar promedio si hay múltiples iteraciones
                if iterations > 1:
                    successful = [r for r in iteration_results if r.success]
                    if successful:
                        avg_time = statistics.mean([r.total_time for r in successful])
                        avg_tps = statistics.mean([r.tokens_per_second for r in successful])
                        print(f"     → Promedio: {avg_time:.2f}s, {avg_tps:.2f} tok/s")

            print()
            
            # Añadir un retraso entre modelos para evitar sobrecarga
            if len(models) > 1:
                print(f"   ⏳ Esperando {self.delay_between_tests}s entre modelos...")
                time.sleep(self.delay_between_tests)

        print(f"\n{'='*70}")
        print(f"✅ Pruebas completadas: {len(self.results)} tests")
        print(f"{'='*70}\n")

    def generate_report(self) -> Dict:
        """Generar reporte de resultados"""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "server": self.base_url,
            "total_tests": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "test_config": {
                "max_concurrent": self.max_concurrent,
                "delay_between_tests": self.delay_between_tests
            }
        }

        # Agrupar por modelo
        models_tested = set(r.model for r in self.results)
        for model_id in models_tested:
            model_results = [r for r in successful if r.model == model_id]

            if model_results:
                report["models"][model_id] = {
                    "tests": len(model_results),
                    "avg_latency": statistics.mean([r.total_time for r in model_results]),
                    "min_latency": min([r.total_time for r in model_results]),
                    "max_latency": max([r.total_time for r in model_results]),
                    "avg_tokens_per_second": statistics.mean([r.tokens_per_second for r in model_results]),
                    "total_tokens_generated": sum([r.tokens_generated for r in model_results])
                }
            else:
                # Incluir modelos con todas las pruebas fallidas
                failed_model_results = [r for r in self.results if r.model == model_id]
                report["models"][model_id] = {
                    "tests": len(failed_model_results),
                    "avg_latency": 0,
                    "min_latency": 0,
                    "max_latency": 0,
                    "avg_tokens_per_second": 0,
                    "total_tokens_generated": 0,
                    "status": "all_failed"
                }

        return report

    def print_summary(self):
        """Imprimir resumen de resultados"""
        report = self.generate_report()

        print(f"\n{'='*70}")
        print(f"📊 RESUMEN DE RESULTADOS")
        print(f"{'='*70}\n")

        print(f"Timestamp: {report['timestamp']}")
        print(f"Total tests: {report['total_tests']}")
        print(f"Successful: {report['successful']} ({report['successful']/report['total_tests']*100:.1f}%)")
        print(f"Failed: {report['failed']}")
        print(f"Configuración de pruebas: {report['test_config']}")

        print(f"\n{'─'*70}")
        print(f"📈 LATENCIAS POR MODELO")
        print(f"{'─'*70}\n")

        # Ordenar modelos por latencia promedio
        sorted_models = sorted(
            [(k, v) for k, v in report["models"].items() if v.get("avg_latency", 0) > 0],
            key=lambda x: x[1]["avg_latency"]
        )
        
        # Mostrar primero los modelos que tuvieron éxito
        for model_id, stats in sorted_models:
            print(f"🤖 {model_id}")
            print(f"   Latencia promedio: {stats['avg_latency']:.2f}s")
            print(f"   Rango: {stats['min_latency']:.2f}s - {stats['max_latency']:.2f}s")
            print(f"   Velocidad: {stats['avg_tokens_per_second']:.2f} tokens/segundo")
            print(f"   Tests: {stats['tests']}")
            print()
        
        # Mostrar modelos que fallaron
        failed_models = [(k, v) for k, v in report["models"].items() if v.get("status") == "all_failed"]
        if failed_models:
            print(f"❌ Modelos con pruebas fallidas:")
            for model_id, stats in failed_models:
                print(f"   🤖 {model_id}: {stats['tests']} pruebas fallidas")
            print()

        print(f"{'='*70}\n")

        return report


def main():
    """Función principal con configuración anti-saturación"""
    
    # Configuración anti-saturación
    tester = LatencyTester(
        base_url=SERVER_URL,
        max_concurrent=1,  # Procesar un modelo a la vez
        delay_between_tests=3.0  # Esperar 3 segundos entre pruebas
    )

    # Obtener modelos disponibles
    print("🔍 Obteniendo modelos disponibles...")
    models = tester.get_models()

    if not models:
        print("❌ No se pudieron obtener los modelos. Verifica que el servidor esté corriendo.")
        return

    print(f"✅ Encontrados {len(models)} modelos:")
    for model in models:
        print(f"   - {model['id']} ({model.get('domain', 'unknown')})")

    # Ejecutar pruebas con configuración más conservadora
    # iterations=1 para evitar sobrecarga
    tester.run_all_tests(models, TEST_PROMPTS, iterations=1)

    # Generar y mostrar resumen
    report = tester.print_summary()

    # Guardar reporte en JSON
    output_file = "latency_test_results_optimized.json"
    with open(output_file, "w") as f:
        json.dump({
            "report": report,
            "detailed_results": [asdict(r) for r in tester.results]
        }, f, indent=2)

    print(f"💾 Reporte detallado guardado en: {output_file}")


if __name__ == "__main__":
    main()