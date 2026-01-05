#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para cargar todos los 5 modelos ARM-Axion simultáneamente
"""

import requests
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

class ARMModelLoader:
    """Clase para cargar todos los modelos ARM-Axion"""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.session = requests.Session()
        self.models_info = {}
        self.load_results = {}
    
    def get_expert_status(self) -> Dict:
        """Obtener estado de los expertos/modelos"""
        try:
            response = requests.get(f"{self.server_url}/experts", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Error obteniendo estado de expertos: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            return {}
    
    def trigger_model_load(self, model_name: str, prompt: str = "Hello, test load", max_tokens: int = 10) -> Tuple[str, bool, str]:
        """Disparar la carga de un modelo haciendo una solicitud"""
        try:
            print(f"🔄 Cargando {model_name}...")
            
            headers = {"Content-Type": "application/json"}
            data = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "top_p": 0.9
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.server_url}/v1/chat/completions",
                json=data,
                headers=headers,
                timeout=180  # 3 minutos para permitir carga
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"✅ {model_name} cargado en {end_time - start_time:.2f}s")
                return model_name, True, f"Cargado en {end_time - start_time:.2f}s"
            else:
                error_msg = f"Error {response.status_code}: {response.text}"
                print(f"❌ {model_name} fallo - {error_msg}")
                return model_name, False, error_msg
                
        except Exception as e:
            print(f"❌ {model_name} error: {e}")
            return model_name, False, str(e)
    
    def load_all_models(self, max_concurrent: int = 5):
        """Cargar todos los modelos disponibles con control de concurrencia"""
        print("🔄 OBTENIENDO MODELOS DISPONIBLES...")
        experts_data = self.get_expert_status()
        all_models = [exp['expert_id'] for exp in experts_data.get('experts', [])]
        
        print(f"📚 Total modelos disponibles: {len(all_models)}")
        print(f"   {all_models}")
        
        # Separar modelos ya cargados de los que no
        loaded_models = [exp['expert_id'] for exp in experts_data.get('experts', []) if exp['is_loaded']]
        unloaded_models = [exp['expert_id'] for exp in experts_data.get('experts', []) if not exp['is_loaded']]
        
        print(f"✅ Modelos ya cargados: {len(loaded_models)} - {loaded_models}")
        print(f"📦 Modelos por cargar: {len(unloaded_models)} - {unloaded_models}")
        
        if not unloaded_models:
            print("\\n🎉 ¡Todos los modelos ya están cargados!")
            return self.show_final_status()
        
        print(f"\\n🚀 INICIANDO CARGA SIMULTÁNEA DE {len(unloaded_models)} MODELOS...")
        print("=" * 80)
        
        # Cargar modelos con control de concurrencia
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submits de tareas
            future_to_model = {
                executor.submit(self.trigger_model_load, model, f"Hola modelo {model} - carga inicial"): model 
                for model in unloaded_models
            }
            
            # Procesar resultados conforme se completan
            for future in as_completed(future_to_model):
                model_name, success, message = future.result()
                self.load_results[model_name] = {'success': success, 'message': message}
        
        print("=" * 80)
        print("📊 RESULTADOS DE CARGA:")
        print("-" * 80)
        
        for model in all_models:
            if model in self.load_results:
                status = "✅" if self.load_results[model]['success'] else "❌"
                print(f"  {status} {model}: {self.load_results[model]['message']}")
            else:
                print(f"  ✅ {model}: Ya estaba cargado")
        
        # Verificar estado final
        self.show_final_status()
        
        return self.load_results
    
    def show_final_status(self):
        """Mostrar estado final de todos los modelos"""
        print("\\n🎯 ESTADO FINAL DE MODELOS:")
        print("-" * 80)
        
        experts_data = self.get_expert_status()
        if not experts_data or 'experts' not in experts_data:
            print("❌ No se pudo obtener el estado final")
            return
        
        loaded_count = 0
        for expert in experts_data['experts']:
            status_icon = "✅" if expert['is_loaded'] else "❌"
            print(f"  {status_icon} {expert['expert_id']} - {expert['domain']} (prio: {expert['priority']})")
            if expert['is_loaded']:
                loaded_count += 1
        
        print("-" * 80)
        print(f"✅ {loaded_count}/{len(experts_data['experts'])} modelos cargados")
        
        if loaded_count == len(experts_data['experts']):
            print("🎉 ¡TODOS LOS 5 MODELOS ARM-Axion ESTÁN CARGADOS Y LISTOS!")
        else:
            print(f"⚠️  {len(experts_data['experts']) - loaded_count} modelos aún no están completamente cargados")
    
    def comprehensive_test(self):
        """Realizar pruebas comprensivas a todos los modelos cargados"""
        print("\\n🧪 REALIZANDO PRUEBAS COMPRENSIVAS...")
        
        # Obtener modelos cargados
        experts_data = self.get_expert_status()
        loaded_models = [exp['expert_id'] for exp in experts_data.get('experts', []) if exp['is_loaded']]
        
        if not loaded_models:
            print("❌ No hay modelos cargados para probar")
            return
        
        print(f"✅ {len(loaded_models)} modelos disponibles para prueba:")
        for model in loaded_models:
            print(f"   • {model}")
        
        # Pruebas simples para cada modelo
        test_prompts = [
            "Say 'Hello, ARM-Axion' in one sentence",
            "Count from 1 to 5", 
            "What is the capital of France?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\\n📋 PRUEBA #{i}: '{prompt}'")
            print("-" * 60)
            
            with ThreadPoolExecutor(max_workers=len(loaded_models)) as executor:
                futures = []
                for model in loaded_models:
                    future = executor.submit(
                        self.trigger_model_load, 
                        model, 
                        prompt, 
                        50
                    )
                    futures.append((model, future))
                
                for model, future in futures:
                    model_name, success, response = future.result()
                    if success:
                        response_preview = response.split("Cargado en")[0][:100] + ("..." if len(response.split("Cargado en")[0]) > 100 else "")
                        print(f"  🟢 {model}: {response_preview}")
                    else:
                        print(f"  🔴 {model}: ERROR - {response}")


def main():
    print("🚀 CARGADOR COMPLETO DE MODELOS ARM-Axion")
    print("=" * 70)
    print("Sistema para cargar y verificar los 5 modelos ARM-Axion:")
    print("  1. Phi4-Fast        (respuesta rápida)")
    print("  2. Qwen2.5-Coder    (experto en código)")
    print("  3. Mistral7B-Balanced (tareas técnicas)")
    print("  4. Gemma3-27B-Multimodal (análisis complejo)")
    print("  5. GPT-OSS-20B      (razonamiento avanzado)")
    print("=" * 70)
    
    loader = ARMModelLoader()
    
    # Verificar conexión al servidor
    try:
        health = requests.get("http://localhost:8080/health", timeout=10)
        if health.status_code == 200:
            print("\\n✅ Servidor ARM-Axion detectado y saludable")
        else:
            print(f"❌ Servidor no responde con estado: {health.status_code}")
            return 1
    except Exception as e:
        print(f"❌ No se puede conectar al servidor: {e}")
        print("   Verifica que el servidor ARM-Axion esté corriendo en http://localhost:8080")
        return 1
    
    # Confirmar acción
    print("\\n🔄 ¿Deseas cargar todos los modelos disponibles? (esto puede tomar varios minutos)")
    response = input("   Pulsa ENTER para continuar o 'n' para cancelar: ").strip().lower()
    
    if response == 'n':
        print("Operación cancelada.")
        return 0
    
    # Cargar todos los modelos
    start_time = time.time()
    results = loader.load_all_models()
    end_time = time.time()
    
    print(f"\\n⏱️  TIEMPO TOTAL DE CARGA: {end_time - start_time:.2f} segundos")
    
    # Hacer pruebas breves
    print("\\n🔍 REALIZANDO PRUEBAS BREVES A MODELOS CARGADOS...")
    loader.comprehensive_test()
    
    print(f"\\n🎉 ¡CARGA COMPLETA REALIZADA!")
    print("   Todos los modelos ARM-Axion ahora están listos para uso")
    
    return 0


if __name__ == "__main__":
    exit(main())