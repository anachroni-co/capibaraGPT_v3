#!/usr/bin/env python3
"""
CLI Interactiva para probar los 5 modelos ARM-Axion
Con soporte para pruebas individuales y modo consenso
"""
import json
import requests
import time
from typing import Dict, List, Optional
import sys

class MultiModelCLI:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.models = [
            {"id": "phi4-fast", "name": "Phi4-Fast (14B)", "desc": "Respuestas rápidas y simples"},
            {"id": "qwen25-coder", "name": "Qwen2.5-Coder (1.5B)", "desc": "Experto en código"},
            {"id": "mistral7b-balanced", "name": "Mistral7B (7B)", "desc": "Equilibrado técnico"},
            {"id": "gemma3-27b", "name": "Gemma3-27B (27B)", "desc": "Análisis complejo multimodal"},
            {"id": "gptoss-20b", "name": "GPT-OSS-20B (20B)", "desc": "Razonamiento complejo"}
        ]

    def print_header(self):
        print("\n" + "="*80)
        print("🚀 SISTEMA MULTI-MODELO ARM-AXION - CLI INTERACTIVA")
        print("="*80)
        print("5 Modelos especializados con vLLM + ARM NEON Optimizations\n")

    def test_connection(self) -> bool:
        """Verificar que el servidor está disponible"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                info = response.json()
                print(f"✅ Servidor: {info.get('name', 'Multi-Model Server')}")
                print(f"   Backend: {info.get('backend', 'unknown')}")
                print(f"   Plataforma: {info.get('platform', 'unknown')}")
                print(f"   Modelos disponibles: {info.get('models_available', 0)}")
                print(f"   Modelos cargados en memoria: {info.get('models_loaded', 0)}")
                return True
            else:
                print(f"❌ Error: Servidor respondió con código {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"❌ No se puede conectar a {self.base_url}")
            print("   Asegúrate de que el servidor está corriendo:")
            print("   ./start_vllm_arm_axion.sh")
            return False
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            return False

    def list_models(self) -> Optional[List[Dict]]:
        """Listar modelos disponibles en el servidor"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                print(f"\n📚 MODELOS DISPONIBLES ({len(models)}):\n")
                for i, model in enumerate(models, 1):
                    status = "🧠 LOADED" if model.get("status") == "loaded" else "💾 LAZY"
                    print(f"  {i}. [{status}] {model['id']}")
                    print(f"     {model.get('description', 'No description')}")
                    print()
                return models
            else:
                print(f"❌ Error obteniendo modelos: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def query_model(self, model_id: str, query: str, max_tokens: int = 150, temperature: float = 0.7, stream: bool = False) -> Optional[Dict]:
        """Consultar un modelo específico"""
        print(f"\n🔄 Consultando modelo: {model_id}")
        print(f"   Query: {query[:60]}{'...' if len(query) > 60 else ''}")

        try:
            url = f"{self.base_url}/v1/chat/completions"
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": query}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }

            start_time = time.time()

            if stream:
                # Streaming response
                response = requests.post(url, json=payload, timeout=120, stream=True)
                if response.status_code == 200:
                    print(f"\n💬 Respuesta ({model_id}):")
                    print("-" * 60)
                    full_text = ""
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith("data: "):
                                data_str = line_str[6:]
                                if data_str.strip() == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    if "choices" in chunk and len(chunk["choices"]) > 0:
                                        delta = chunk["choices"][0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            print(content, end="", flush=True)
                                            full_text += content
                                except json.JSONDecodeError:
                                    pass

                    elapsed = time.time() - start_time
                    print(f"\n{'-' * 60}")
                    print(f"⏱️  Tiempo: {elapsed:.2f}s")
                    return {"text": full_text, "time": elapsed, "tokens": len(full_text.split())}
                else:
                    print(f"❌ Error {response.status_code}: {response.text[:200]}")
                    return None
            else:
                # Non-streaming response
                response = requests.post(url, json=payload, timeout=120)
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        text = result["choices"][0]["message"]["content"]
                        tokens = len(text.split())

                        print(f"\n💬 Respuesta ({model_id}):")
                        print("-" * 60)
                        print(text)
                        print("-" * 60)
                        print(f"⏱️  Tiempo: {elapsed:.2f}s | 📊 Tokens: {tokens} | 🚀 Velocidad: {tokens/elapsed:.1f} t/s")

                        return {"text": text, "time": elapsed, "tokens": tokens}
                    else:
                        print(f"⚠️  Respuesta sin contenido válido")
                        return None
                else:
                    print(f"❌ Error {response.status_code}: {response.text[:200]}")
                    return None

        except requests.exceptions.Timeout:
            print(f"⏰ Timeout: El modelo tardó más de 120 segundos")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def test_single_model(self):
        """Probar un solo modelo"""
        print("\n" + "="*60)
        print("MODO: PRUEBA INDIVIDUAL")
        print("="*60)

        models = self.list_models()
        if not models:
            return

        print("\nSelecciona un modelo:")
        for i, model in enumerate(self.models, 1):
            print(f"  {i}. {model['name']}")

        try:
            choice = int(input(f"\nModelo (1-{len(self.models)}): "))
            if 1 <= choice <= len(self.models):
                model = self.models[choice - 1]
                query = input("\n💭 Tu pregunta: ").strip()

                if not query:
                    print("❌ Pregunta vacía")
                    return

                stream_choice = input("¿Usar streaming? (s/N): ").strip().lower()
                use_stream = stream_choice == 's'

                self.query_model(model["id"], query, stream=use_stream)
            else:
                print("❌ Opción inválida")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Entrada inválida o cancelado")

    def test_all_models(self):
        """Probar todos los modelos con la misma consulta"""
        print("\n" + "="*60)
        print("MODO: COMPARACIÓN DE TODOS LOS MODELOS")
        print("="*60)

        query = input("\n💭 Tu pregunta para todos los modelos: ").strip()
        if not query:
            print("❌ Pregunta vacía")
            return

        print(f"\n🧪 Probando consulta en {len(self.models)} modelos...\n")

        results = []
        for model in self.models:
            print(f"\n{'='*60}")
            result = self.query_model(model["id"], query, max_tokens=150)
            if result:
                results.append({
                    "model": model["name"],
                    "model_id": model["id"],
                    **result
                })
            time.sleep(0.5)  # Pequeña pausa entre consultas

        # Resumen comparativo
        if results:
            print(f"\n\n{'='*80}")
            print("📊 RESUMEN COMPARATIVO")
            print("="*80)
            print(f"{'Modelo':<25} {'Tiempo':<10} {'Tokens':<8} {'Velocidad (t/s)':<15}")
            print("-"*80)

            for r in sorted(results, key=lambda x: x['time']):
                speed = r['tokens'] / r['time'] if r['time'] > 0 else 0
                print(f"{r['model']:<25} {r['time']:>7.2f}s  {r['tokens']:>6}   {speed:>10.1f}")

            # Ganador por velocidad
            fastest = min(results, key=lambda x: x['time'])
            print(f"\n🏆 Más rápido: {fastest['model']} ({fastest['time']:.2f}s)")

    def consensus_mode(self):
        """Modo consenso: obtener respuestas de múltiples modelos y votar"""
        print("\n" + "="*60)
        print("MODO: CONSENSO MULTI-MODELO")
        print("="*60)
        print("Este modo consulta a varios modelos y presenta todas las respuestas")
        print("para que puedas evaluar cuál es mejor.\n")

        query = input("💭 Tu pregunta: ").strip()
        if not query:
            print("❌ Pregunta vacía")
            return

        print("\nSelecciona cuántos modelos usar (2-5):")
        print("  [Sugerencia: 3 modelos para balance entre calidad y velocidad]")

        try:
            num_models = int(input("\nNúmero de modelos (2-5): "))
            if num_models < 2 or num_models > 5:
                print("❌ Debe ser entre 2 y 5")
                return
        except ValueError:
            print("❌ Entrada inválida")
            return

        # Usar los modelos más potentes para consenso
        selected_models = self.models[-num_models:]  # Últimos N modelos (más grandes)

        print(f"\n🧠 Usando {num_models} modelos para consenso:")
        for m in selected_models:
            print(f"   - {m['name']}")

        print(f"\n🔄 Consultando {num_models} modelos...\n")

        responses = []
        for model in selected_models:
            print(f"\n{'='*60}")
            result = self.query_model(model["id"], query, max_tokens=200)
            if result:
                responses.append({
                    "model": model["name"],
                    **result
                })
            time.sleep(0.5)

        # Mostrar todas las respuestas para que el usuario elija
        if len(responses) >= 2:
            print(f"\n\n{'='*80}")
            print("🗳️  RESPUESTAS RECOPILADAS - VOTA LA MEJOR")
            print("="*80)

            for i, resp in enumerate(responses, 1):
                print(f"\n{'─'*80}")
                print(f"Opción {i}: {resp['model']} ({resp['time']:.2f}s, {resp['tokens']} tokens)")
                print(f"{'─'*80}")
                print(resp['text'])

            print(f"\n{'='*80}")
            print("💡 Basándote en las respuestas anteriores, elige la mejor")
            print("   o combina las ideas de varias respuestas.")
        else:
            print("\n❌ No se obtuvieron suficientes respuestas para consenso")

    def show_system_status(self):
        """Mostrar estado del sistema"""
        print("\n" + "="*60)
        print("🖥️  ESTADO DEL SISTEMA")
        print("="*60)

        try:
            # Info general
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                info = response.json()
                print(f"\n📋 Información General:")
                print(f"   Nombre: {info.get('name')}")
                print(f"   Backend: {info.get('backend')}")
                print(f"   Versión: {info.get('version')}")
                print(f"   Plataforma: {info.get('platform')}")
                print(f"   Modelos disponibles: {info.get('models_available')}")
                print(f"   Modelos en memoria: {info.get('models_loaded')}")

            # Health check
            health_response = requests.get(f"{self.base_url}/health", timeout=5)
            if health_response.status_code == 200:
                health = health_response.json()
                status = health.get('status', 'unknown')
                emoji = "✅" if status == "healthy" else "⚠️"
                print(f"\n{emoji} Estado de Salud: {status}")

            # Listar modelos
            self.list_models()

        except Exception as e:
            print(f"❌ Error obteniendo estado: {e}")

    def run(self):
        """Ejecutar CLI interactiva"""
        self.print_header()

        if not self.test_connection():
            print("\n❌ No se puede conectar al servidor. Inicialo con:")
            print("   cd /home/elect/capibara6 && ./start_vllm_arm_axion.sh")
            return

        while True:
            print("\n" + "─"*60)
            print("MENÚ PRINCIPAL:")
            print("  1. 🎯 Probar un modelo individual")
            print("  2. ⚖️  Comparar todos los modelos")
            print("  3. 🧠 Modo consenso (múltiples modelos)")
            print("  4. 🖥️  Estado del sistema")
            print("  5. 📚 Listar modelos disponibles")
            print("  6. 🚪 Salir")
            print("─"*60)

            try:
                choice = input("\nSelecciona opción (1-6): ").strip()

                if choice == "1":
                    self.test_single_model()
                elif choice == "2":
                    self.test_all_models()
                elif choice == "3":
                    self.consensus_mode()
                elif choice == "4":
                    self.show_system_status()
                elif choice == "5":
                    self.list_models()
                elif choice == "6":
                    print("\n👋 ¡Hasta luego!")
                    break
                else:
                    print("❌ Opción inválida")

            except KeyboardInterrupt:
                print("\n\n👋 Saliendo...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    cli = MultiModelCLI()
    cli.run()


if __name__ == "__main__":
    main()
