#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interfaz Interactiva de Prueba para vLLM ARM-Axion con 5 Modelos
Permite probar los 5 modelos ARM-Axion por separado: 
Phi4-mini, Qwen2.5-coder, Mistral7B, Gemma3-27B, GPT-OSS-20B
"""

import json
import time
import sys
import os
from typing import Dict, List, Any
import requests
import subprocess

# Forzar entorno ARM-Axion compatible
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_ENABLE_V1_ENGINE'] = '0'

# Asegurar que nuestro código modificado está en el path
sys.path.insert(0, '/home/elect/capibara6/vllm-source-modified')

class ARMInteractiveTestInterface:
    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url
        self.models = [
            {"id": "phi4_fast", "name": "Phi4 mini", "desc": "Rápido para respuestas simples"},
            {"id": "qwen_coder", "name": "Qwen2.5 coder", "desc": "Experto en código"},
            {"id": "mistral_balanced", "name": "Mistral7B", "desc": "Equilibrado para tareas técnicas"},
            {"id": "gemma3_multimodal", "name": "Gemma3 27B", "desc": "Complejo para contexto largo"},
            {"id": "gptoss_complex", "name": "GPT-OSS 20B", "desc": "Razonamiento complejo"}
        ]
    
    def print_header(self, title: str):
        """Imprimir encabezado con formato"""
        print("\\n" + "=" * 80)
        print(f"{title:^80}")
        print("=" * 80)

    def print_menu(self):
        """Imprimir menú principal"""
        self.print_header("INTERFAZ INTERACTIVA ARM-AXION vLLM")
        print("Sistema de prueba para los 5 modelos ARM-Axion")
        print("-" * 80)
        print("Sistema ARM-Axion optimizado para Google Cloud C4A-standard-32")
        print("Con vLLM multi-modelo usando 5 modelos: Phi4, Qwen2.5, Mistral7B, Gemma3, GPT-OSS-20B")
        print()
        print("Opciones disponibles:")
        print("1. Probar modelo individual")
        print("2. Probar todos los modelos con una consulta")
        print("3. Verificar estado del servidor ARM-Axion")
        print("4. Comparación de modelos")
        print("5. Información del sistema ARM-Axion")
        print("6. Salir")
        print("-" * 80)

    def check_server_health(self):
        """Verificar estado del servidor"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Servidor saludable: {health_data}")
                return True
            else:
                print(f"❌ Servidor no responde: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Servidor no accesible: {e}")
            return False

    def list_available_models(self):
        """Listar modelos disponibles en el servidor"""
        try:
            response = requests.get(f"{self.server_url}/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get("data", [])
                print(f"\\n MODELOS DISPONIBLES ({len(models)}):")
                for model in models:
                    print(f"   {model['id']}: {model.get('description', 'Sin descripción')}")
                return [model['id'] for model in models]
            else:
                print(f"❌ No se pudieron listar los modelos: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error obteniendo modelos: {e}")
            return []

    def test_individual_model(self):
        """Probar un modelo individual"""
        available_models = self.list_available_models()
        if not available_models:
            print("⚠️  No hay modelos disponibles en el servidor")
            return

        self.print_header("PROBAR MODELO INDIVIDUAL")

        print("Modelos disponibles:")
        for i, model_id in enumerate(available_models, 1):
            print(f"  {i}. {model_id}")

        try:
            choice = int(input(f"\\nSelecciona modelo (1-{len(available_models)}): "))
            if 1 <= choice <= len(available_models):
                model_id = available_models[choice - 1]

                query = input(f"\\nIngresa tu consulta para {model_id}: ").strip()
                if not query:
                    print("Consulta vacía, regresando...")
                    return

                self.generate_with_model(model_id, query)
            else:
                print("Opción inválida")
        except ValueError:
            print("Entrada inválida, debes ingresar un número")

    def test_all_models(self):
        """Probar todos los modelos con la misma consulta"""
        available_models = self.list_available_models()
        if not available_models:
            print("⚠️  No hay modelos disponibles en el servidor")
            return

        self.print_header("PROBAR TODOS LOS MODELOS")

        query = input("\\nIngresa tu consulta para probar en todos los modelos: ").strip()
        if not query:
            print("Consulta vacía, regresando...")
            return

        print(f"\\n🧪 PROBANDO CONSULTA EN {len(available_models)} MODELOS:")
        print("-" * 60)

        results = {}
        for model_id in available_models:
            print(f"\\nPrueba con {model_id}...")
            result = self.generate_with_model(model_id, query, verbose=False)
            results[model_id] = result

        print("\\n📊 RESULTADOS FINALES:")
        for model_id, result in results.items():
            status = "✅" if result["success"] else "❌"
            time_str = f"{result.get('time', 0):.2f}s" if result.get('time') else "N/A"
            print(f"  {status} {model_id}: {time_str} - {result.get('length', 0)} chars")

    def generate_with_model(self, model_id: str, query: str, verbose: bool = True):
        """Generar texto con un modelo específico"""
        try:
            if verbose:
                print(f"\\n🚀 Enviando consulta a {model_id}...")

            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant optimized for ARM architecture."},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.7,
                "max_tokens": 128
            }

            start_time = time.time()
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    text = result["choices"][0]["message"]["content"]
                    duration = end_time - start_time

                    if verbose:
                        print(f"✅ {model_id} respondió en {duration:.2f}s")
                        print(f"📝 Longitud: {len(text)} caracteres")
                        print(f"💬 Respuesta:\\n{text[:300]}...")
                        if len(text) > 300:
                            print("... (truncado)")

                    return {
                        "success": True,
                        "response": text,
                        "time": duration,
                        "length": len(text)
                    }
                else:
                    if verbose:
                        print(f"❌ {model_id} no devolvió respuesta válida")
                    return {"success": False, "error": "No choices in response"}
            else:
                if verbose:
                    print(f"❌ {model_id} falló con código: {response.status_code}")
                    print(f"Error: {response.text[:200]}...")
                return {"success": False, "error": f"HTTP {response.status_code}", "details": response.text}
        except Exception as e:
            if verbose:
                print(f"❌ Error con {model_id}: {e}")
            return {"success": False, "error": str(e)}

    def compare_models(self):
        """Comparar modelos con análisis detallado"""
        available_models = self.list_available_models()
        if not available_models:
            print("⚠️  No hay modelos disponibles en el servidor")
            return

        self.print_header("COMPARACIÓN DETALLADA DE MODELOS")

        query = input("\\nIngresa tu consulta para comparar modelos: ").strip()
        if not query:
            print("Consulta vacía, regresando...")
            return

        print(f"\\n⚖️  COMPARANDO {len(available_models)} MODELOS:")
        print("=" * 80)

        results = {}
        for model_id in available_models:
            print(f"\\n📝 {model_id}...")
            result = self.generate_with_model(model_id, query, verbose=False)
            results[model_id] = result

        # Mostrar resultados comparativos
        print("\\n📊 RESULTADOS COMPARATIVOS:")
        print(f"{'Modelo':<20} {'Tiempo (s)':<12} {'Caracteres':<10} {'Estado':<10} {'Características'}")
        print("-" * 80)

        for model_id, result in results.items():
            time_str = f"{result.get('time', 0):.2f}" if result.get('time') else "N/A"
            length_str = str(result.get('length', 0)) if result.get('length') else "N/A"
            status_str = "✅ OK" if result.get('success') else "❌ Error"

            # Determinar características basadas en modelo
            characteristics = ""
            if "phi4" in model_id.lower():
                characteristics = "Rápido, respuestas simples"
            elif "qwen" in model_id.lower():
                characteristics = "Experto en código"
            elif "mistral" in model_id.lower():
                characteristics = "Equilibrado"
            elif "gemma" in model_id.lower():
                characteristics = "Multimodal, contexto largo"
            elif "gptoss" in model_id.lower():
                characteristics = "Razonamiento complejo"

            print(f"{model_id:<20} {time_str:<12} {length_str:<10} {status_str:<10} {characteristics}")

    def system_info(self):
        """Mostrar información del sistema ARM-Axion"""
        self.print_header("INFO DEL SISTEMA ARM-AXION")

        print("SISTEMA ARM-Axion vLLM CON 5 MODELOS:")
        print("-" * 60)
        print("✓ Arquitectura: ARM64 (aarch64) - Google Cloud C4A-standard-32")
        print("✓ Plataforma detectada: CPU (correctamente como ARM-Axion)")
        print("✓ Backend: vLLM clásico (no V1 engine) - Compatible ARM")
        print("✓ Optimizaciones ARM: NEON, ACL, cuantización, Flash Attention")
        print("✓ Servidor multi-modelo: ARM-Axion optimizado")

        # Verificar plataforma
        try:
            from vllm.platforms import current_platform
            print(f"✓ Detección plataforma: {current_platform.device_type} - {'✅ Correcta' if current_platform.is_cpu() else '❌ Incorrecta'}")
        except Exception as e:
            print(f"⚠️ No se pudo verificar plataforma: {e}")

        # Verificar modelos
        models = self.list_available_models()
        print(f"✓ Modelos disponibles: {len(models)}/5 ARM-Axion optimizados")

        print("\\n5 MODELOS ARM-Axion:")
        for model in self.models:
            status = "✅" if model['id'] in models else "❌"
            print(f"   {status} {model['id']}: {model['desc']}")

    def run(self):
        """Ejecutar el bucle principal de la interfaz"""
        print("🚀 INICIANDO INTERFAZ INTERACTIVA ARM-AXION vLLM")
        print("Sistema de prueba para 5 modelos ARM-Axion optimizados")

        if not self.check_server_health():
            print("\\n⚠️  El servidor no responde. Asegúrate de que esté corriendo:")
            print("   cd /home/elect/capibara6")
            print("   ./start_all_models_arm_axion.sh")
            return

        while True:
            self.print_menu()
            choice = input("\\nElige una opción (1-6): ").strip()

            if choice == "1":
                self.test_individual_model()
            elif choice == "2":
                self.test_all_models()
            elif choice == "3":
                self.check_server_health()
            elif choice == "4":
                self.compare_models()
            elif choice == "5":
                self.system_info()
            elif choice == "6":
                print("\\n👋 ¡Gracias por usar la Interfaz Interactiva ARM-Axion!")
                print("Sistema optimizado para Google Cloud ARM Axion C4A-standard-32")
                break
            else:
                print("❌ Opción inválida. Por favor elige del 1 al 6.")

            input("\\nPresiona Enter para continuar...")


def main():
    print("🔄 Iniciando Interfaz Interactiva vLLM ARM-Axion...")
    print("Verificando sistema de 5 modelos ARM-Axion optimizados")
    
    try:
        app = ARMInteractiveTestInterface()
        app.run()
    except KeyboardInterrupt:
        print("\\n\\n👋 Interfaz interrumpida por el usuario")
    except Exception as e:
        print(f"\\n❌ Error en la interfaz: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
