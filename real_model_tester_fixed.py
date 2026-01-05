#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interfaz Interactiva Real de Capibara6 - Router, Consenso y Modelos
Conecta directamente con los modelos reales a través de vLLM (versión corregida)
"""

import json
import time
import sys
import os
from typing import Dict, List, Any
import threading
import requests

# Asegurar que estamos en el directorio correcto
sys.path.insert(0, '/home/elect/capibara6/backend')

from ollama_client_fixed import VLLMClient

class RealModelTester:
    def __init__(self):
        # Cargar configuración
        self.load_model_config()
        
        # Crear cliente VLLM
        try:
            self.client = VLLMClient(self.config)
            self.vllm_available = True
            print("✅ Cliente VLLM (corregido) conectado correctamente")
        except Exception as e:
            print(f"❌ Error al conectar con VLLM: {e}")
            print("⚠️  Modo de pruebas limitado, endpoint no disponible")
            self.vllm_available = False
            self.client = None

    def load_model_config(self):
        """Cargar configuración del modelo"""
        try:
            # Usar el archivo recién creado con los 5 modelos
            with open('/home/elect/capibara6/five_model_config.json', 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Si no existe, intentar con el original
            with open('/home/elect/capibara6/model_config.json', 'r') as f:
                self.config = json.load(f)

    def get_available_models(self):
        """Obtener lista de modelos disponibles"""
        models = list(self.config['models'].keys())
        return models

    def test_individual_model(self, model_key: str, query: str):
        """Probar un modelo individual real"""
        if not self.vllm_available:
            print("❌ VLLM no disponible, no se puede probar modelo real")
            return None
            
        print(f"\n📡 Enviando consulta a: {self.config['models'][model_key]['name']}")
        print(f"   Consulta: {query[:100]}{'...' if len(query) > 100 else ''}")
        
        try:
            start_time = time.time()
            result = self.client.generate(query, model_key)
            end_time = time.time()
            
            print(f"\n✅ Respuesta recibida en {end_time - start_time:.2f}s")
            
            if result['success']:
                print(f"   Modelo: {result['model']}")
                print(f"   Tokens: {result.get('token_count', 'N/A')}")
                print(f"   Tiempo total: {result.get('total_duration', 'N/A')}ms")
                print(f"\n💬 RESPUESTA:")
                print(f"   {result['response']}")
                return result
            else:
                print(f"❌ Error del modelo: {result['error']}")
                return result
                
        except Exception as e:
            print(f"❌ Error al comunicar con el modelo: {e}")
            return {"success": False, "error": str(e)}

    def test_all_models_compare(self, query: str):
        """Probar todos los modelos real y comparar respuestas"""
        if not self.vllm_available:
            print("❌ VLLM no disponible, no se pueden probar modelos reales")
            return
            
        print(f"\n🔄 Probando consulta en todos los modelos: '{query[:50]}...'")
        
        models = self.get_available_models()
        results = {}
        
        for i, model_key in enumerate(models):
            print(f"\n ({i+1}/{len(models)}) Prueba con {self.config['models'][model_key]['name']}...")
            
            try:
                start_time = time.time()
                result = self.client.generate(query, model_key)
                end_time = time.time()
                
                results[model_key] = {
                    'response': result.get('response', ''),
                    'time': end_time - start_time,
                    'success': result['success'],
                    'model_name': self.config['models'][model_key]['name'],
                    'error': result.get('error', '') if not result['success'] else None
                }
                
                if result['success']:
                    print(f"   ✅ {end_time - start_time:.2f}s - {len(result['response'])} chars")
                else:
                    print(f"   ❌ Error: {result['error']}")
                    
            except Exception as e:
                print(f"   ❌ Error con {self.config['models'][model_key]['name']}: {e}")
                results[model_key] = {
                    'response': '',
                    'time': 0,
                    'success': False,
                    'model_name': self.config['models'][model_key]['name'],
                    'error': str(e)
                }
        
        # Mostrar resultados comparativos
        print(f"\n📊 RESULTADOS COMPARATIVOS:")
        print("-" * 100)
        print(f"{'Modelo':<20} {'Estado':<8} {'Tiempo (s)':<12} {'Chars':<8} {'Palabras':<10}")
        print("-" * 100)
        
        for model_key, result in results.items():
            model_name = result['model_name'][:18]  # Truncar nombre para formato
            status = "✅" if result['success'] else "❌"
            time_str = f"{result['time']:.2f}" if result['time'] > 0 else "N/A"
            chars_str = str(len(result['response'])) if result['success'] else "N/A"
            words_str = str(len(result['response'].split())) if result['success'] else "N/A"
            
            print(f"{model_name:<20} {status:<8} {time_str:<12} {chars_str:<8} {words_str:<10}")
        
        # Mostrar respuestas
        print(f"\n💬 RESPUESTAS DETALLADAS:")
        for model_key, result in results.items():
            print(f"\n--- {result['model_name']} ---")
            if result['success']:
                print(f"   {result['response']}")
            else:
                print(f"   ❌ Error: {result['error']}")
        
        return results

    def test_endpoint_health(self):
        """Probar la salud del endpoint vLLM"""
        if not self.vllm_available:
            print("❌ VLLM no disponible")
            return False
            
        print(f"\n🏥 Probando salud del endpoint: {self.client.endpoint}")
        
        try:
            # Probar el endpoint de modelos
            health_url = f"{self.client.endpoint}/models"
            response = requests.get(health_url, timeout=10)
            
            if response.status_code == 200:
                models_data = response.json()
                print(f"✅ Endpoint saludable")
                print(f"✅ Modelos disponibles: {len(models_data.get('data', []))}")
                
                for model in models_data.get('data', []):
                    print(f"   • {model.get('id', 'N/A')}")
                
                return True
            else:
                print(f"❌ Health check falló: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error en health check: {e}")
            return False

    def interactive_test_loop(self):
        """Bucle de prueba interactivo real"""
        print("🚀 INICIANDO INTERFAZ DE PRUEBA REAL DE MODELOS (CORREGIDA)")
        print("=" * 60)
        
        if self.vllm_available:
            print(f"✅ Conectado a: {self.client.endpoint}")
            print(f"✅ Modelos configurados: {len(self.config['models'])}")
            for key, cfg in self.config['models'].items():
                print(f"   • {cfg['name']} ({key})")
        else:
            print("⚠️  ADVERTENCIA: VLLM no disponible - Modo de pruebas limitado")
            print("   Puedes revisar la configuración y reiniciar los servicios")
            print("   El sistema está configurado pero el endpoint no responde")
            print("   Verifica que el servidor esté corriendo en el puerto correcto")
            return
        
        while True:
            print("\n" + "="*60)
            print("OPCIONES DE PRUEBA REAL:")
            print("1. Probar modelo individual")
            print("2. Probar todos los modelos (comparativo)")
            print("3. Probar salud del endpoint") 
            print("4. Ver modelos disponibles")
            print("5. Salir")
            print("-"*60)
            
            try:
                choice = input("\nSelecciona opción (1-5): ").strip()
                
                if choice == "1":
                    # Probar modelo individual
                    models = self.get_available_models()
                    print(f"\nModelos disponibles ({len(models)}):")
                    for i, model_key in enumerate(models, 1):
                        model_name = self.config['models'][model_key]['name']
                        description = self.config['models'][model_key]['description']
                        print(f"  {i}. {model_name}")
                        print(f"     → {description}")
                    
                    model_choice = input(f"\nSelecciona modelo (1-{len(models)}): ").strip()
                    try:
                        idx = int(model_choice) - 1
                        if 0 <= idx < len(models):
                            selected_model = models[idx]
                            query = input("Ingresa tu consulta: ").strip()
                            if query:
                                self.test_individual_model(selected_model, query)
                            else:
                                print("Consulta vacía")
                        else:
                            print("Opción inválida")
                    except ValueError:
                        print("Entrada inválida")
                
                elif choice == "2":
                    # Probar todos los modelos
                    query = input("Ingresa tu consulta para todos los modelos: ").strip()
                    if query:
                        self.test_all_models_compare(query)
                    else:
                        print("Consulta vacía")
                
                elif choice == "3":
                    # Probar salud del endpoint
                    self.test_endpoint_health()
                
                elif choice == "4":
                    # Ver modelos disponibles
                    models = self.get_available_models()
                    print(f"\nCONFIGURACIÓN DE MODELOS ({len(models)}):")
                    for model_key in models:
                        cfg = self.config['models'][model_key]
                        print(f"\n• {model_key}: {cfg['name']}")
                        print(f"  Descripción: {cfg['description']}")
                        print(f"  Casos de uso: {', '.join(cfg['use_case'][:3])}")
                        print(f"  Tokens máx: {cfg['max_tokens']}")
                        print(f"  Timeout: {cfg['timeout']}ms")
                
                elif choice == "5":
                    print("\n👋 Saliendo de la interfaz de pruebas real...")
                    break
                
                else:
                    print("❌ Opción inválida, selecciona 1-5")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Interrumpido por el usuario")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    print("🔧 INICIANDO INTERFAZ REAL DE PRUEBA DE MODELOS CAPIBARA6 (CORREGIDA)")
    print("Conexión directa a vLLM con modelos ARM-Axion optimizados")
    print("-" * 60)
    
    try:
        tester = RealModelTester()
        
        # Probar conexión antes de iniciar
        if tester.vllm_available:
            print(f"✅ Cliente VLLM creado, endpoint: {tester.client.endpoint}")
        else:
            print("⚠️  Cliente VLLM no disponible, revisando configuración...")
            print(f"   Endpoint configurado: {tester.config.get('api_settings', {}).get('vllm_endpoint', 'N/A')}")
        
        tester.interactive_test_loop()
        
    except Exception as e:
        print(f"❌ Error fatal en la interfaz: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()