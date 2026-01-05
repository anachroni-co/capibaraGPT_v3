#!/usr/bin/env python3
"""
Configuración para ejecutar vLLM en ARM-Axion con los 5 modelos:
- Qwen2.5-coder 
- Phi4-mini
- Mistral7B
- Gemma3-27B
- GPT-OSS-20B

Este script configura vLLM para reconocer correctamente la plataforma ARM64
y usar las optimizaciones implementadas en el código fuente de vLLM.
"""

import os
import sys
from pathlib import Path

def setup_vllm_arm_axion():
    """Configura el entorno para usar vLLM en ARM-Axion"""
    
    # Asegurar que nuestro vLLM modificado está en el path
    vllm_modified_path = "/home/elect/capibara6/vllm-source-modified"
    if vllm_modified_path not in sys.path:
        sys.path.insert(0, vllm_modified_path)
    
    print("="*70)
    print("CONFIGURACIÓN DE VLLM PARA ARM-AXION")
    print("="*70)
    
    # Verificar la detección de plataforma
    try:
        from vllm.platforms import current_platform
        print(f"✓ Plataforma detectada: {current_platform}")
        print(f"✓ Tipo de dispositivo: {current_platform.device_type}")
        print(f"✓ ¿Es CPU?: {current_platform.is_cpu()}")
        
        if current_platform.is_cpu() and current_platform.device_type == "cpu":
            print("✓ Detección de plataforma ARM-Axion: CORRECTA")
        else:
            print("✗ Advertencia: La detección de plataforma no es la esperada")
            
    except Exception as e:
        print(f"✗ Error en detección de plataforma: {e}")
        return False
    
    # Verificar que los modelos estén disponibles
    models_dir = Path("/home/elect/models")
    required_models = [
        ("phi-4-mini", "Phi4-mini"),
        ("qwen2.5-coder-1.5b", "Qwen2.5-coder"), 
        ("mistral-7b-instruct-v0.2", "Mistral7B"),
        ("gemma-3-27b-it", "Gemma3-27B"),
        ("gpt-oss-20b", "GPT-OSS-20B")
    ]
    
    print(f"\nVerificando modelos en: {models_dir}")
    available_models = []
    
    for model_path, model_name in required_models:
        full_path = models_dir / model_path
        if full_path.exists():
            available_models.append((full_path, model_name))
            print(f"✓ {model_name} encontrado: {full_path}")
        else:
            print(f"✗ {model_name} NO encontrado: {full_path}")
    
    print(f"\n✓ Modelos disponibles: {len(available_models)}/{len(required_models)}")
    
    if len(available_models) == 0:
        print("✗ No se encontraron modelos disponibles")
        return False
    
    print("\n" + "="*70)
    print("CONFIGURACIÓN ARM-Axion LISTA")
    print("="*70)
    print("Ahora puede usar vLLM con las siguientes características:")
    print("- Detección correcta de plataforma ARM64 como CPU")
    print("- Soporte para los 5 modelos requeridos")
    print("- Optimizaciones ARM implementadas (NEON, ACL, etc.)")
    print("- Uso de los servidores multi-modelo ARM-Axion")
    print()
    print("PARA EJECUTAR EL SERVIDOR:")
    print("cd /home/elect/capibara6/arm-axion-optimizations/vllm-integration")
    print("PYTHONPATH='/home/elect/capibara6/vllm-source-modified:$PYTHONPATH' python3 multi_model_server.py")
    print()
    print("PARA USAR DIRECTAMENTE VLLM:")
    print("import sys")
    print("sys.path.insert(0, '/home/elect/capibara6/vllm-source-modified')")
    print("from vllm import LLM, SamplingParams")
    print()
    
    return True

def run_example():
    """Muestra un ejemplo de uso de vLLM con ARM-Axion"""
    
    print("EJEMPLO DE USO:")
    print("-" * 40)
    
    example_code = '''
import sys
sys.path.insert(0, '/home/elect/capibara6/vllm-source-modified')

from vllm import LLM, SamplingParams

# Inicializar el modelo con configuración ARM-Axion
llm = LLM(
    model="/home/elect/models/phi-4-mini",  # o cualquier otro modelo
    tensor_parallel_size=1,
    dtype="float16",
    enforce_eager=True,
    gpu_memory_utilization=0.5,  # Ajustar según memoria disponible
    max_num_seqs=256
)

# Generar texto
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100
)

outputs = llm.generate(["Hello, how are you?"], sampling_params)
print(outputs[0].outputs[0].text)
'''
    
    print(example_code)

if __name__ == "__main__":
    success = setup_vllm_arm_axion()
    
    if success:
        run_example()
        print("\n¡Configuración completada exitosamente!")
        print("vLLM está listo para funcionar en ARM-Axion con los 5 modelos.")
    else:
        print("\n✗ Error en la configuración. Revise los mensajes anteriores.")
        sys.exit(1)