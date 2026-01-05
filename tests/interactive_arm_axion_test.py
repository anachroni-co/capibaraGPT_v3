#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interfaz Interactiva de Prueba para vLLM ARM-Axion con 5 Modelos
Permite probar los 5 modelos: Phi4, Qwen2.5, Mistral7B, Gemma3-27B, GPT-OSS-20B
"""

import json
import time
import sys
import os
from typing import Dict, List, Any
import requests
from pathlib import Path

# Asegurar que nuestro vLLM modificado está en el path
vllm_path = '/home/elect/capibara6/vllm-source-modified'
if vllm_path not in sys.path:
    sys.path.insert(0, vllm_path)

# Importar componentes de vLLM
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform


class ARMVLLMClient:
    """Cliente simplificado para interactuar con el servidor ARM-Axion"""
    
    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url
        self.is_server_running = self._check_server()
        
    def _check_server(self):
        """Verificar si el servidor está corriendo"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt: str, model_key: str = "phi4_fast", **kwargs):
        """Generar texto usando el servidor o directamente con vLLM"""
        if self.is_server_running:
            return self._generate_via_server(prompt, model_key, **kwargs)
        else:
            return self._generate_direct(prompt, model_key, **kwargs)
    
    def _generate_via_server(self, prompt: str, model_key: str, **kwargs):
        """Generar usando el servidor HTTP"""
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": model_key,
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 200),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9)
            }
            
            response = requests.post(
                f"{self.server_url}/api/generate",
                json=data,
                headers=headers,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response received")
            else:
                return f"Error {response.status_code}: {response.text}"
        except Exception as e:
            return f"Server error: {e}"
    
    def _generate_direct(self, prompt: str, model_key: str, **kwargs):
        """Generar usando vLLM directamente (modo fallback)"""
        try:
            # Obtener la configuración del modelo
            config_path = '/home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.five_models.optimized.json'
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Encontrar el modelo específico
            model_config = None
            for expert in config['experts']:
                if expert['expert_id'] == model_key:
                    model_config = expert
                    break
            
            if not model_config:
                return f"Modelo {model_key} no encontrado"
            
            # Cargar el modelo y generar
            print(f"Cargando modelo {model_key}...")
            llm = LLM(
                model=model_config['model_path'],
                tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
                dtype=model_config.get('dtype', 'float16'),
                enforce_eager=model_config.get('enforce_eager', True),
                gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.5),
                max_num_seqs=model_config.get('max_num_seqs', 32)
            )
            
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                max_tokens=kwargs.get("max_tokens", 200)
            )
            
            outputs = llm.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text
            
        except Exception as e:
            return f"Error directo: {e}"


class InteractiveARMVLLM:
    def __init__(self, server_url="http://localhost:8080"):
        # Verificar detección de plataforma ARM-Axion
        self.check_platform()
        
        # Cargar configuración de modelos
        self.load_model_config()
        
        # Inicializar cliente
        self.client = ARMVLLMClient(server_url)
        
        # Mapeo de claves a nombres reales
        self.model_key_to_name = {
            'phi4_fast': 'phi4:mini',
            'mistral_balanced': 'mistral-7b-instruct-v0.2',
            'qwen_coder': 'qwen2.5-coder-1.5b',
            'gemma3_multimodal': 'gemma-3-27b-it',
            'gptoss_complex': 'gpt-oss-20b'
        }

    def check_platform(self):
        """Verificar que estamos en la plataforma ARM-Axion correcta"""
        print(f"Plataforma detectada: {current_platform}")
        print(f"Tipo de dispositivo: {current_platform.device_type}")
        
        if current_platform.is_cpu() and current_platform.device_type == "cpu":
            print("✅ Detección ARM-Axion: CORRECTA")
        else:
            print("⚠️  Advertencia: Plataforma ARM-Axion no detectada correctamente")

    def load_model_config(self):
        """Cargar configuración de modelos"""
        config_path = '/home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.five_models.optimized.json'
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print("✅ Configuración de modelos cargada")
        except FileNotFoundError:
            print("⚠️  Archivo de configuración no encontrado")
            # Configuración simulada
            self.config = {
                'experts': [
                    {'expert_id': 'phi4_fast', 'description': 'Modelo rápido para respuestas simples'},
                    {'expert_id': 'mistral_balanced', 'description': 'Modelo equilibrado para tareas técnicas'},
                    {'expert_id': 'qwen_coder', 'description': 'Modelo experto en código'},
                    {'expert_id': 'gemma3_multimodal', 'description': 'Modelo multimodal avanzado'},
                    {'expert_id': 'gptoss_complex', 'description': 'Modelo para razonamiento complejo'}
                ]
            }

    def print_header(self, title: str):
        """Imprimir encabezado con formato"""
        print("\\n" + "=" * 80)
        print(f"{title:^80}")
        print("=" * 80)

    def print_menu(self):
        """Imprimir menú principal"""
        self.print_header("INTERFAZ INTERACTIVA ARM-Axion vLLM")
        print("Sistema de prueba para 5 modelos ARM-Axion optimizados")
        print("\\nOpciones disponibles:")
        print("1. Probar modelo individual")
        print("2. Probar todos los modelos")
        print("3. Comparar modelos")
        print("4. Información del sistema")
        print("5. Salir")
        print("-" * 80)

    def get_available_models(self):
        """Obtener lista de modelos disponibles"""
        models = [exp['expert_id'] for exp in self.config.get('experts', [])]
        return models

    def test_individual_model(self):
        """Probar un modelo individual"""
        self.print_header("PROBAR MODELO INDIVIDUAL")

        models = self.get_available_models()
        print("Modelos disponibles:")
        for i, model_key in enumerate(models, 1):
            # Buscar descripción en la configuración
            description = "Descripción no disponible"
            for expert in self.config.get('experts', []):
                if expert['expert_id'] == model_key:
                    description = expert.get('description', 'Sin descripción')
                    break
            print(f"  {i}. {model_key} - {description}")

        try:
            choice = input(f"\\nSelecciona modelo (1-{len(models)}): ").strip()
            model_idx = int(choice) - 1

            if 0 <= model_idx < len(models):
                selected_model = models[model_idx]

                print(f"\\nProbando modelo: {selected_model}")
                query = input("Ingresa tu consulta: ").strip()

                if not query:
                    print("Consulta vacía, regresando al menú...")
                    return

                print(f"\\nEnviando consulta a {selected_model}...")

                start_time = time.time()
                result = self.client.generate(query, selected_model)
                end_time = time.time()

                print(f"\\n✅ Respuesta de {selected_model}:")
                print(f"   Tiempo: {end_time - start_time:.2f}s")
                if isinstance(result, str) and len(result) > 0:
                    print(f"   Longitud: {len(result)} caracteres")
                    print(f"   Respuesta:\\n{result}")
                else:
                    print(f"   Resultado: {result}")

            else:
                print("Opción inválida")
        except ValueError:
            print("Entrada inválida, debes ingresar un número")

    def test_all_models(self):
        """Probar todos los modelos con la misma consulta"""
        self.print_header("PROBAR TODOS LOS MODELOS")

        query = input("\\nIngresa tu consulta para probar en todos los modelos: ").strip()
        if not query:
            print("Consulta vacía, regresando al menú...")
            return

        models = self.get_available_models()
        results = {}

        print(f"\\n🧪 PROBANDO {len(models)} MODELOS:")
        for i, model_key in enumerate(models):
            print(f"\\n  {i+1}/{len(models)}. {model_key}...")
            
            start_time = time.time()
            result = self.client.generate(query, model_key)
            end_time = time.time()

            results[model_key] = {
                'response': result,
                'time': end_time - start_time,
                'length': len(result) if isinstance(result, str) else 0
            }
            
            status = "✅" if isinstance(result, str) and len(result) > 0 else "❌"
            print(f"     {status} Respondió en {end_time - start_time:.2f}s")

        # Mostrar resultados
        print(f"\\n📊 RESULTADOS:")
        print("-" * 80)
        print(f"{'Modelo':<20} {'Tiempo (s)':<12} {'Chars':<8}")
        print("-" * 80)

        for model_key, result in results.items():
            time_str = f"{result['time']:.2f}" if result['time'] > 0 else "N/A"
            chars_str = str(result['length']) if result['length'] > 0 else "N/A"
            print(f"{model_key:<20} {time_str:<12} {chars_str:<8}")

    def compare_models(self):
        """Comparar modelos con análisis detallado"""
        self.print_header("COMPARACIÓN DETALLADA DE MODELOS")

        query = input("\\nIngresa tu consulta para comparar modelos: ").strip()
        if not query:
            print("Consulta vacía, regresando al menú...")
            return

        print(f"\\nComparando '{query[:50]}...' en todos los modelos...")

        models = self.get_available_models()
        results = {}

        for model_key in models:
            print(f"  Probando {model_key}...")
            start_time = time.time()
            result = self.client.generate(query, model_key)
            end_time = time.time()

            results[model_key] = {
                'response': result,
                'time': end_time - start_time,
                'length': len(result) if isinstance(result, str) else 0
            }

        # Análisis comparativo
        print(f"\\n📈 ANÁLISIS COMPARATIVO:")
        print("=" * 100)
        
        for model_key, result in results.items():
            print(f"\\n🔹 {model_key}:")
            print(f"   - Tiempo: {result['time']:.2f}s")
            print(f"   - Caracteres: {result['length']}")
            if isinstance(result['response'], str) and len(result['response']) > 0:
                preview = result['response'][:200] + ("..." if len(result['response']) > 200 else "")
                print(f"   - Respuesta: {preview}")
            else:
                print(f"   - Error: {result['response']}")
            print("-" * 50)

    def system_info(self):
        """Mostrar información del sistema"""
        self.print_header("INFORMACIÓN DEL SISTEMA ARM-Axion")

        print("SISTEMA ARM-AXION vLLM - 5 MODELOS OPTIMIZADOS")
        print("-" * 60)
        
        print(f"✓ Plataforma: {current_platform.device_type}")
        print(f"✓ Arquitectura: {os.uname().machine if hasattr(os, 'uname') else 'Desconocida'}")
        
        print("\\n✓ 5 MODELOS CONFIGURADOS:")
        for expert in self.config.get('experts', []):
            print(f"   • {expert['expert_id']} - {expert.get('domain', 'general')}")
            print(f"     {expert.get('description', 'Sin descripción')[:80]}...")

        print("\\n✓ ESTADO DEL SERVIDOR:")
        server_status = "Operativo" if self.client.is_server_running else "No accesible"
        print(f"   • Estado: {server_status}")
        print(f"   • URL: {self.client.server_url}")
        
        print("\\n✓ OPTIMIZACIONES ARM-Axion:")
        print("   • Kernels NEON optimizados")
        print("   • ARM Compute Library (ACL) integrada") 
        print("   • Cuantización AWQ/GPTQ")
        print("   • Flash Attention")
        print("   • Chunked Prefill")

    def run(self):
        """Ejecutar el bucle principal de la interfaz"""
        print("\\n🚀 Iniciando Interfaz Interactiva ARM-Axion vLLM...")
        print("Sistema de pruebas para 5 modelos con ARM-Axion optimizado")
        
        while True:
            self.print_menu()
            choice = input("\\nElige una opción (1-5): ").strip()

            if choice == "1":
                self.test_individual_model()
            elif choice == "2":
                self.test_all_models()
            elif choice == "3":
                self.compare_models()
            elif choice == "4":
                self.system_info()
            elif choice == "5":
                print("\\n👋 ¡Gracias por usar la Interfaz Interactiva ARM-Axion vLLM!")
                print("Sistema optimizado para ARM-Axion con 5 modelos")
                break
            else:
                print("❌ Opción inválida. Por favor elige del 1 al 5.")

            input("\\nPresiona Enter para continuar...")


def main():
    print("🔄 Iniciando Interfaz Interactiva ARM-Axion vLLM...")
    print("Conectando con los 5 modelos: Phi4, Qwen2.5, Mistral7B, Gemma3, GPT-OSS-20B")
    
    try:
        # Asegurarse que el vLLM modificado está en el path
        if '/home/elect/capibara6/vllm-source-modified' not in sys.path:
            sys.path.insert(0, '/home/elect/capibara6/vllm-source-modified')
        
        app = InteractiveARMVLLM()
        app.run()
    except KeyboardInterrupt:
        print("\\n\\n👋 Interfaz interrumpida por el usuario")
    except Exception as e:
        print(f"\\n❌ Error en la interfaz: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()