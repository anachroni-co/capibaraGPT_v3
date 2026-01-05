#!/usr/bin/env python3
"""
Script para probar vLLM en ARM-Axion con la plataforma corregida
"""

import sys
import os
import time
from pathlib import Path

# Asegurar que el path incluye nuestro código modificado de vLLM
sys.path.insert(0, '/home/elect/capibara6/vllm-source-modified')

def test_vllm_platform_detection():
    """Test de detección de plataforma"""
    from vllm.platforms import current_platform
    
    print("="*60)
    print("TEST DE DETECCIÓN DE PLATAFORMA")
    print("="*60)
    print(f"Plataforma detectada: {current_platform}")
    print(f"Tipo de dispositivo: {current_platform.device_type}")
    print(f"¿Es CPU?: {current_platform.is_cpu()}")
    print(f"¿Es CUDA?: {current_platform.is_cuda()}")
    print(f"¿Es ROCm?: {current_platform.is_rocm()}")
    print(f"¿Es TPU?: {current_platform.is_tpu()}")
    print(f"¿Es XPU?: {current_platform.is_xpu()}")
    print(f"¿Es no especificada?: {current_platform.is_unspecified()}")
    
    return current_platform.is_cpu() and current_platform.device_type == "cpu"


def test_model_loading(model_path, model_name="test_model"):
    """Test de carga de modelo"""
    print(f"\n{'='*60}")
    print(f"TEST DE CARGA DE MODELO: {model_name}")
    print(f"{'='*60}")
    
    try:
        from vllm import LLM, SamplingParams
        
        print(f"Cargando modelo desde: {model_path}")
        
        # Parámetros conservadores para pruebas
        start_time = time.time()
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            dtype='float16',
            enforce_eager=True,  # Más estable para pruebas
            gpu_memory_utilization=0.1,  # Muy baja para evitar problemas
            max_num_seqs=1,
            seed=42
            # Nota: 'device' no es un parámetro válido en esta versión de vLLM
        )
        
        load_time = time.time() - start_time
        print(f"✓ Modelo {model_name} cargado exitosamente en {load_time:.2f}s")
        
        return llm
        
    except Exception as e:
        print(f"✗ Error al cargar {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_generation(llm, prompt="Hello, how are you?", max_tokens=30):
    """Test de generación de texto"""
    if llm is None:
        print("\n⚠ No se puede probar generación: modelo no cargado")
        return False
        
    print(f"\n{'='*60}")
    print("TEST DE GENERACIÓN DE TEXTO")
    print("="*60)
    
    try:
        from vllm import SamplingParams
        
        print(f"Generando con prompt: '{prompt}'")
        start_time = time.time()
        
        sampling_params = SamplingParams(
            temperature=0.7, 
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        outputs = llm.generate([prompt], sampling_params)
        
        if outputs and len(outputs) > 0:
            response_time = time.time() - start_time
            generated_text = outputs[0].outputs[0].text
            print(f"✓ Generación exitosa en {response_time:.2f}s")
            print(f"Prompt: {prompt}")
            print(f"Respuesta: {generated_text}")
            return True
        else:
            print("⚠ No se recibió respuesta")
            return False
            
    except Exception as e:
        print(f"✗ Error en generación: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Función principal para probar vLLM en ARM-Axion"""
    print("INICIALIZANDO PRUEBAS DE VLLM EN ARM-AXION")
    print("Con la correción para detección de plataforma ARM64")
    
    # Verificar que los modelos existen
    models_dir = Path("/home/elect/models")
    available_models = {
        'phi4': models_dir / "phi-4-mini",
        'qwen2.5-coder': models_dir / "qwen2.5-coder-1.5b", 
        'mistral': models_dir / "mistral-7b-instruct-v0.2",
        'gemma3': models_dir / "gemma-3-27b-it",
        'gptoss': models_dir / "gpt-oss-20b"
    }
    
    # Filtrar modelos que existen
    existing_models = {}
    for name, path in available_models.items():
        if path.exists():
            existing_models[name] = path
        else:
            print(f"⚠ Modelo no encontrado: {name} en {path}")
    
    if not existing_models:
        print("✗ No se encontraron modelos disponibles para probar")
        return
    
    print(f"\n✓ Modelos disponibles para prueba: {list(existing_models.keys())}")
    
    # Test de detección de plataforma
    platform_ok = test_vllm_platform_detection()
    
    if not platform_ok:
        print("✗ La detección de plataforma no es correcta, abortando pruebas")
        return
    
    print(f"\n✓ Detección de plataforma ARM-Axion: CORRECTA")
    print("Ahora probaremos la carga y generación con los modelos disponibles...")
    
    # Probar cada modelo disponible
    for model_name, model_path in existing_models.items():
        print(f"\n{'#'*80}")
        print(f"PRUEBA CON MODELO: {model_name}")
        print(f"{'#'*80}")
        
        # Cargar el modelo
        llm = test_model_loading(str(model_path), model_name)
        
        # Probar generación si el modelo se cargó
        if llm is not None:
            test_generation(llm, f"Hello, this is a test for {model_name}. How are you?", max_tokens=25)
        else:
            print(f"⚠ No se pudo probar generación para {model_name}")
        
        print(f"\n{'-'*60}")
        print(f"Fin de prueba para {model_name}")
        print(f"{'-'*60}")


if __name__ == "__main__":
    main()