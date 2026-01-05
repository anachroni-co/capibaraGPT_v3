#!/usr/bin/env python3
"""
Interfaz interactiva para probar los 5 modelos ARM-Axion
usando el servidor API REST que ya está funcionando
"""
import json
import requests
import time
from typing import Dict, List

class ARMModelTester:
    def __init__(self, base_url="http://localhost:8081"):
        self.base_url = base_url
        self.models = [
            {"id": "phi4-fast", "name": "Phi4-fast", "desc": "Rápido para respuestas simples"},
            {"id": "qwen25-coder", "name": "Qwen2.5-coder", "desc": "Experto en código"},
            {"id": "mistral7b-balanced", "name": "Mistral7B", "desc": "Equilibrado para tareas técnicas"},
            {"id": "gemma3-27b", "name": "Gemma3-27B", "desc": "Complejo para contexto largo"},
            {"id": "gptoss-20b", "name": "GPT-OSS-20B", "desc": "Razonamiento complejo"}
        ]
    
    def test_connection(self):
        """Verificar conexión con el servidor"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                info = response.json()
                print(f"✅ Servidor disponible: {info.get('name', 'Desconocido')}")
                print(f"   Backend: {info.get('backend', 'Desconocido')}")
                print(f"   Modelos disponibles: {info.get('models_available', 0)}")
                return True
            else:
                print(f"❌ Servidor respondió con código: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ No se puede conectar al servidor: {e}")
            return False
    
    def list_models(self):
        """Listar modelos disponibles"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                print(f"\n📚 MODELOS DISPONIBLES ({len(models)}):")
                for model in models:
                    status = "🧠" if model.get("status") == "loaded" else "💾"
                    print(f"  {status} {model['id']}: {model['description']}")
                return models
            else:
                print(f"❌ Error obteniendo modelos: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error obteniendo modelos: {e}")
            return []
    
    def test_model(self, model_id: str, query: str):
        """Probar un modelo específico con una consulta"""
        print(f"\n🚀 Probando {model_id}...")
        
        try:
            # Intentar usar el endpoint de completions de chat (si está disponible)
            url = f"{self.base_url}/v1/chat/completions"
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=60)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    text = result["choices"][0]["message"]["content"]
                    tokens = len(text.split()) if text else 0
                    print(f"✅ {model_id} respondió en {elapsed_time:.2f}s")
                    print(f"📊 Tokens generados: {tokens}")
                    print(f"💬 Respuesta:\n{text}")
                    return True
                else:
                    print(f"⚠️  {model_id} no devolvió resultados válidos")
                    return False
            else:
                print(f"❌ {model_id} falló con código {response.status_code}: {response.text[:100]}...")
                return False
                
        except requests.exceptions.ReadTimeout:
            print(f"⏰ {model_id} excedió tiempo de espera")
            return False
        except Exception as e:
            print(f"❌ Error con {model_id}: {e}")
            return False
    
    def run_interactive_test(self):
        """Ejecutar interfaz interactiva"""
        print("="*80)
        print("🧪 INTERFAZ INTERACTIVA - SISTEMA ARM-Axion CON 5 MODELOS")
        print("="*80)
        print("Sistema de pruebas para ARM-Axion con:")
        print("  - Phi4-fast (rápido)")
        print("  - Qwen2.5-coder (experto en código)")
        print("  - Mistral7B (equilibrado)")  
        print("  - Gemma3-27B (complejo)")
        print("  - GPT-OSS-20B (razonamiento)")
        
        # Verificar conexión
        if not self.test_connection():
            print("\n❌ No se puede conectar al servidor ARM-Axion")
            return
        
        # Listar modelos
        available_models = self.list_models()
        if not available_models:
            print("\n❌ No se encontraron modelos disponibles")
            return
        
        while True:
            print("\n" + "-"*60)
            print("Opciones:")
            print("  1. Probar un modelo específico")
            print("  2. Probar todos los modelos con una consulta")
            print("  3. Comparar todos los modelos")
            print("  4. Ver estado del sistema")
            print("  5. Salir")
            
            choice = input("\nSelecciona opción (1-5): ").strip()
            
            if choice == "1":
                # Probar modelo específico
                print(f"\nModelos disponibles:")
                for i, model in enumerate(available_models, 1):
                    print(f"  {i}. {model['id']}: {model['description']}")
                
                try:
                    idx = int(input(f"\nSelecciona modelo (1-{len(available_models)}): ")) - 1
                    if 0 <= idx < len(available_models):
                        model_id = available_models[idx]['id']
                        query = input("Ingresa tu consulta: ").strip()
                        if query:
                            self.test_model(model_id, query)
                        else:
                            print("Consulta vacía")
                    else:
                        print("Opción inválida")
                except ValueError:
                    print("Entrada inválida")
            
            elif choice == "2":
                # Probar todos los modelos con una consulta
                query = input("\nIngresa tu consulta para todos los modelos: ").strip()
                if not query:
                    print("Consulta vacía")
                    continue
                
                print(f"\n🧪 PROBANDO CONSULTA EN TODOS LOS MODELOS:")
                for model in available_models:
                    model_id = model['id']
                    self.test_model(model_id, query)
                    print("-" * 40)
            
            elif choice == "3":
                # Comparar todos los modelos
                query = input("\nIngresa consulta para comparar modelos: ").strip()
                if not query:
                    print("Consulta vacía")
                    continue
                
                print(f"\n⚖️  COMPARACIÓN ENTRE MODELOS:")
                results = []
                for model in available_models:
                    model_id = model['id']
                    try:
                        url = f"{self.base_url}/v1/chat/completions"
                        payload = {
                            "model": model_id,
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": query}
                            ],
                            "temperature": 0.7,
                            "max_tokens": 100
                        }
                        
                        start_time = time.time()
                        response = requests.post(url, json=payload, timeout=60)
                        elapsed_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            result = response.json()
                            if "choices" in result and len(result["choices"]) > 0:
                                text = result["choices"][0]["message"]["content"]
                                tokens = len(text.split())
                                results.append({
                                    "model": model_id,
                                    "response": text,
                                    "time": elapsed_time,
                                    "tokens": tokens
                                })
                                print(f"✅ {model_id}: {elapsed_time:.2f}s ({tokens} tokens)")
                            else:
                                print(f"❌ {model_id}: No valid response")
                        else:
                            print(f"❌ {model_id}: Error {response.status_code}")
                    except Exception as e:
                        print(f"❌ {model_id}: {e}")
                
                # Mostrar resultados comparativos
                print(f"\n📊 RESULTADOS COMPARATIVOS:")
                print(f"{'Modelo':<20} {'Tiempo':<10} {'Tokens':<8} {'Palabras/s':<12}")
                print("-" * 60)
                for result in results:
                    words_per_sec = result['tokens'] / result['time'] if result['time'] > 0 else 0
                    print(f"{result['model']:<20} {result['time']:.2f}s     {result['tokens']:<8} {words_per_sec:.1f}")
            
            elif choice == "4":
                # Ver estado del sistema
                response = requests.get(f"{self.base_url}/", timeout=10)
                if response.status_code == 200:
                    info = response.json()
                    print(f"\n🖥️  ESTADO DEL SISTEMA:")
                    print(f"   Nombre: {info.get('name')}")
                    print(f"   Backend: {info.get('backend')}")
                    print(f"   Plataforma: {info.get('platform')}")
                    print(f"   Modelos disponibles: {info.get('models_available')}")
                    print(f"   Modelos cargados: {info.get('models_loaded')}")
                    
                    # Estado de salud
                    health_response = requests.get(f"{self.base_url}/health", timeout=10)
                    if health_response.status_code == 200:
                        health = health_response.json()
                        print(f"   Salud: {health.get('status')}")
                        print(f"   Modelos cargados (health): {health.get('models_loaded')}")
                else:
                    print("❌ No se puede obtener estado del sistema")
            
            elif choice == "5":
                print("\n👋 ¡Gracias por usar la interfaz ARM-Axion!")
                break
            
            else:
                print("❌ Opción inválida")


def main():
    print("🚀 Iniciando Interfaz Interactiva ARM-Axion...")
    tester = ARMModelTester()
    tester.run_interactive_test()


if __name__ == "__main__":
    main()