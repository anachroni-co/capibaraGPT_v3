#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interacción Interactiva con el Servidor ARM-Axion
Permite probar los 5 modelos ARM-Axion: Phi4, Qwen2.5, Mistral7B, Gemma3-27B, GPT-OSS-20B
"""

import requests
import json
import time
import sys
from typing import Dict, List, Any, Optional

class ARMModelTester:
    """Cliente para interactuar con el servidor ARM-Axion"""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.session = requests.Session()
        self.available_models = []
        self.loaded_models = []
        self.load_available_models()
    
    def load_available_models(self):
        """Cargar la lista de modelos disponibles desde el servidor"""
        try:
            response = requests.get(f"{self.server_url}/experts", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.available_models = [exp['expert_id'] for exp in data['experts']]
                self.loaded_models = [exp['expert_id'] for exp in data['experts'] if exp['is_loaded']]
                print(f"✅ {len(self.available_models)} modelos disponibles")
                print(f"✅ {len(self.loaded_models)} modelos cargados: {self.loaded_models}")
            else:
                print(f"❌ Error al cargar modelos: {response.status_code}")
        except Exception as e:
            print(f"❌ Error de conexión al cargar modelos: {e}")
    
    def test_model(self, model_name: str, prompt: str, max_tokens: int = 100):
        """Probar un modelo específico"""
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            print(f"🚀 Enviando solicitud a {model_name}...")
            
            start_time = time.time()
            response = self.session.post(
                f"{self.server_url}/v1/chat/completions",
                json=data,
                headers=headers,
                timeout=120
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                print(f"✅ {model_name} respondió en {end_time - start_time:.2f}s")
                print(f"📄 Contenido ({len(content)} chars):")
                print(f"   {content[:500]}{'...' if len(content) > 500 else ''}")
                
                return content
            else:
                print(f"❌ {model_name} error {response.status_code}: {response.text}")
                return f"Error {response.status_code}: {response.text}"
                
        except Exception as e:
            print(f"❌ Error al comunicarse con {model_name}: {e}")
            return f"Error de comunicación: {e}"
    
    def interactive_chat(self):
        """Modo de chat interactivo con todos los modelos"""
        print("\\n💬 MODO INTERACTIVO CON MODELOS ARM-Axion")
        print("=" * 60)
        print("Modelos disponibles:")
        for i, model in enumerate(self.available_models, 1):
            status = "✅ CARGADO" if model in self.loaded_models else "⏰ NO CARGADO"
            print(f"  {i}. {model} - {status}")
        print("  0. Salir")
        print("-" * 60)
        
        while True:
            try:
                choice = input("\\nSelecciona modelo (0 para salir): ").strip()
                
                if choice == "0":
                    print("👋 Saliendo del modo interactivo")
                    break
                
                try:
                    model_idx = int(choice) - 1
                    if 0 <= model_idx < len(self.available_models):
                        selected_model = self.available_models[model_idx]
                        
                        print(f"\\n chatting con {selected_model}")
                        print("Escribe 'exit' para cambiar de modelo o 'quit' para salir")
                        
                        while True:
                            user_input = input(f"\\n{selected_model} > ").strip()
                            
                            if user_input.lower() in ['exit', 'quit', 'salir']:
                                break
                            
                            if user_input.lower() in ['quit', 'terminar']:
                                return
                            
                            if user_input:
                                self.test_model(selected_model, user_input)
                    else:
                        print("❌ Selección inválida")
                except ValueError:
                    print("❌ Ingresa un número válido")
                    
            except KeyboardInterrupt:
                print("\\n👋 Interrupción por usuario")
                break
    
    def batch_test_models(self):
        """Probar todos los modelos con la misma consulta"""
        print("\\n🧪 PRUEBA DE RENDIMIENTO - TODOS LOS MODELOS")
        print("=" * 60)
        
        query = input("Ingresa tu consulta para probar en todos los modelos: ").strip()
        if not query:
            print("Consulta vacía, regresando...")
            return
        
        print(f"\\nEnviando '{query[:50]}...' a todos los modelos...\\n")
        
        results = {}
        for model in self.available_models:
            print(f"\\n  → Probando {model}...")
            
            start_time = time.time()
            response = self.test_model(model, query, max_tokens=150)
            end_time = time.time()
            
            results[model] = {
                'response': response,
                'time': end_time - start_time,
                'length': len(response) if isinstance(response, str) else 0
            }
        
        print("\\n📊 RESULTADOS FINALES:")
        print("-" * 80)
        print(f"{'Modelo':<20} {'Tiempo (s)':<12} {'Caracteres':<10} {'Status'}")
        print("-" * 80)
        
        for model, result in results.items():
            status = "✅" if isinstance(result['response'], str) and len(result['response']) > 0 else "❌"
            print(f"{model:<20} {result['time']:<12.2f} {result['length']:<10} {status}")
    
    def model_comparison_test(self):
        """Comparar modelos con la misma consulta"""
        print("\\n⚖️  COMPARACIÓN ENTRE MODELOS")
        print("=" * 60)
        
        query = input("Ingresa tu consulta para comparar modelos: ").strip()
        if not query:
            print("Consulta vacía, regresando...")
            return
        
        print(f"\\nComparando respuesta a: '{query[:70]}...'\\n")
        
        responses = {}
        for model in self.loaded_models:  # Solo modelos cargados
            print(f"→ {model}...")
            start_time = time.time()
            response = self.test_model(model, query, max_tokens=200)
            end_time = time.time()
            
            responses[model] = {
                'response': response,
                'time': end_time - start_time,
                'length': len(response) if isinstance(response, str) else 0
            }
        
        print("\\n🔍 ANÁLISIS DE RESPUESTAS:")
        print("=" * 80)
        for model, data in responses.items():
            print(f"\\n🔹 {model}:")
            print(f"   Tiempo: {data['time']:.2f}s | Caracteres: {data['length']}")
            print(f"   Respuesta: {data['response'][:300]}{'...' if len(data['response']) > 300 else ''}")
            print("-" * 80)
    
    def system_health_check(self):
        """Verificar estado del sistema"""
        print("\\n🏥 VERIFICACIÓN DE SALUD DEL SISTEMA")
        print("=" * 60)
        
        try:
            # Verificar salud del servidor
            health_resp = requests.get(f"{self.server_url}/health", timeout=10)
            if health_resp.status_code == 200:
                health_data = health_resp.json()
                print(f"✅ Servidor: {health_data.get('status', 'unknown')}")
                print(f"✅ Orquestador: {'✅' if health_data.get('orchestrator_ready', False) else '❌'}")
                print(f"✅ Expertos: {'✅' if health_data.get('experts_ready', False) else '❌'}")
            else:
                print("❌ Servidor no responde")
        except Exception as e:
            print(f"❌ Error verificando salud: {e}")
        
        # Estado de expertos
        try:
            experts_resp = requests.get(f"{self.server_url}/experts", timeout=10)
            if experts_resp.status_code == 200:
                experts_data = experts_resp.json()
                print("\\n📦 ESTADO DE MODELOS:")
                for expert in experts_data['experts']:
                    status = "✅" if expert['is_loaded'] else "⏰"
                    print(f"  {status} {expert['expert_id']} - {expert['domain']} (prioridad: {expert['priority']})")
        except Exception as e:
            print(f"❌ Error verificando estado de expertos: {e}")
    
    def run_interactive_menu(self):
        """Menú interactivo principal"""
        while True:
            print("\\n" + "="*70)
            print("🤖 INTERACCIÓN CON SERVIDOR ARM-Axion - 5 MODELOS")
            print("="*70)
            print("1. Modo chat interactivo")
            print("2. Probar modelo individual")
            print("3. Prueba de rendimiento (todos los modelos)")
            print("4. Comparación entre modelos cargados")
            print("5. Verificar estado del sistema")
            print("6. Salir")
            print("-"*70)
            
            choice = input("Selecciona opción (1-6): ").strip()
            
            if choice == "1":
                self.interactive_chat()
            elif choice == "2":
                self.single_model_test()
            elif choice == "3":
                self.batch_test_models()
            elif choice == "4":
                self.model_comparison_test()
            elif choice == "5":
                self.system_health_check()
            elif choice == "6":
                print("\\n👋 ¡Gracias por usar la interfaz ARM-Axion!")
                break
            else:
                print("❌ Opción inválida")
    
    def single_model_test(self):
        """Probar un modelo individual"""
        print("\\n🔍 PRUEBA DE MODELO INDIVIDUAL")
        print("=" * 60)
        
        # Mostrar modelos disponibles
        print("Modelos disponibles:")
        for i, model in enumerate(self.available_models, 1):
            status = "✅ CARGADO" if model in self.loaded_models else "⏰ NO CARGADO"
            print(f"  {i}. {model} - {status}")
        
        try:
            choice = input(f"\\nSelecciona modelo (1-{len(self.available_models)}): ").strip()
            model_idx = int(choice) - 1
            
            if 0 <= model_idx < len(self.available_models):
                selected_model = self.available_models[model_idx]
                
                print(f"\\nProbando modelo: {selected_model}")
                query = input("Ingresa tu consulta: ").strip()
                
                if query:
                    self.test_model(selected_model, query)
            else:
                print("❌ Selección inválida")
        except ValueError:
            print("❌ Ingresa un número válido")


def main():
    print("🚀 INICIANDO INTERFAZ DE PRUEBA ARM-Axion")
    print("=" * 60)
    print("Sistema de prueba para 5 modelos ARM-Axion optimizados:")
    print("  - Phi4-Fast (respuesta rápida)")
    print("  - Qwen2.5-Coder (especializado en código)")  
    print("  - Mistral7B-Balanced (tareas técnicas)")
    print("  - Gemma3-27B-Multimodal (análisis complejo)")
    print("  - GPT-OSS-20B (razonamiento avanzado)")
    print("=" * 60)
    
    # Crear cliente
    tester = ARMModelTester()
    
    # Verificar conexión
    try:
        health = requests.get("http://localhost:8080/health", timeout=5)
        if health.status_code == 200:
            print("\\n✅ Servidor ARM-Axion detectado en http://localhost:8080")
        else:
            print("\\n⚠️  Servidor responde pero no está saludable")
    except Exception:
        print("\\n❌ No se puede conectar al servidor ARM-Axion en http://localhost:8080")
        print("   Verifica que el servidor esté corriendo en otro terminal")
        return 1
    
    # Iniciar menú interactivo
    try:
        tester.run_interactive_menu()
    except KeyboardInterrupt:
        print("\\n\\n👋 Sesión interrumpida por el usuario")
    except Exception as e:
        print(f"\\n❌ Error en la interfaz: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())