#!/usr/bin/env python3
"""
Implementación de optimizaciones adicionales para ARM Axion
enfocadas en reducir latencia y mejorar el streaming
"""

import json
import os
from pathlib import Path

def apply_fp8_kv_cache_optimization():
    """
    Aplica optimización de KV Cache en FP8 para reducir uso de memoria y mejorar latencia
    """
    config_path = "/home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.json"
    
    # Leer configuración actual
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Modificar cada experto para usar KV cache en FP8
    for i, expert in enumerate(config['experts']):
        print(f"Optimizando KV Cache para {expert['expert_id']}...")
        
        # Aplicar optimización de KV cache en FP8
        expert['kv_cache_dtype'] = 'fp8'
        
        # Ajustar algunos parámetros relacionados para mejorar rendimiento
        if expert['expert_id'] == 'gemma3_multimodal':
            # Para modelo grande, usar valores más conservadores
            if 'num_gpu_blocks_override' not in expert:
                expert['num_gpu_blocks_override'] = 2048  # Reducido para usar menos RAM
            else:
                expert['num_gpu_blocks_override'] = 2048
            if 'block_size' not in expert:
                expert['block_size'] = 8  # Más eficiente para FP8
            else:
                expert['block_size'] = 8
            expert['max_num_seqs'] = 64  # Más bajo para menor latencia
        elif expert['expert_id'] == 'aya_expanse_multilingual':
            if 'num_gpu_blocks_override' not in expert:
                expert['num_gpu_blocks_override'] = 3072
            else:
                expert['num_gpu_blocks_override'] = 3072
            if 'block_size' not in expert:
                expert['block_size'] = 8
            else:
                expert['block_size'] = 8
            expert['max_num_seqs'] = 128
        else:
            # Para otros modelos, también optimizar si es posible
            if 'block_size' not in expert:
                expert['block_size'] = 8
            else:
                expert['block_size'] = 8
            if 'num_gpu_blocks_override' not in expert:
                expert['num_gpu_blocks_override'] = 4096
            else:
                expert['num_gpu_blocks_override'] = 4096
    
    # Guardar la configuración modificada
    backup_config_path = config_path.replace(".json", ".fp8_optimized.backup")
    with open(backup_config_path, 'w') as f:
        original_config = json.load(open(config_path))
        json.dump(original_config, f, indent=2)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Optimización FP8 KV Cache aplicada y guardada en {config_path}")
    print(f"   Backup original en: {backup_config_path}")


def apply_advanced_captured_graphs_optimization():
    """
    Aplica optimización avanzada de Captured Graphs para reducir latencia
    """
    config_path = "/home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.json"
    
    # Leer configuración actual
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Optimizando Captured Graphs para reducir latencia...")
    
    # Aplicarlo en cada experto individualmente tambien
    for expert in config['experts']:
        expert['use_captured_graph'] = True
        if expert['max_model_len'] < 8192:
            expert['max_context_len_to_capture'] = expert['max_model_len']
        else:
            expert['max_context_len_to_capture'] = 8192
    
    # Actualizar la configuración
    perf_tuning = config.get('performance_tuning', {})
    perf_tuning['use_captured_graph'] = True
    perf_tuning['max_context_len_to_capture'] = 8192  # Aumentado para mejor optimización
    config['performance_tuning'] = perf_tuning
    
    # Guardar la configuración modificada
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Optimización Advanced Captured Graphs aplicada y guardada en {config_path}")


def apply_streaming_latency_improvements():
    """
    Aplica mejoras específicas para streaming y latencia
    """
    config_path = "/home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.json"
    
    # Leer configuración actual
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Aplicando mejoras para streaming y latencia...")
    
    # Ajustar para cada experto: reducir el número de secuencias máximas para bajar latencia
    for expert in config['experts']:
        # Reducir max_num_seqs para mejorar latencia individual
        if expert['max_num_seqs'] > 64:  # Si es mayor a 64
            expert['max_num_seqs'] = 64  # Reducir a 64 para mejorar latencia individual
        
        # Ajustar scheduler steps para favorecer latencia sobre throughput
        if 'num_scheduler_steps' not in expert:
            expert['num_scheduler_steps'] = 2  # Valor bajo para mejorar latencia
        else:
            expert['num_scheduler_steps'] = min(expert['num_scheduler_steps'], 2)
        
        # Mejorar la eficiencia del prefill para streaming
        expert['enable_chunked_prefill'] = True
        expert['max_num_batched_tokens'] = max(expert['max_num_batched_tokens'], 4096)
        
        # Asegurarse de que use dtype más eficiente
        if 'dtype' not in expert or expert['dtype'] == 'bfloat16':
            expert['dtype'] = 'float16'  # Para modelos que no requieren alta precisión
    
    # Ajustar también en la sección de performance tuning
    perf_tuning = config.get('performance_tuning', {})
    perf_tuning['num_scheduler_steps'] = 2
    perf_tuning['max_num_batched_tokens'] = 4096
    perf_tuning['enable_prefix_caching'] = True
    
    # Actualizar la configuración
    config['performance_tuning'] = perf_tuning
    
    # Guardar la configuración modificada
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Mejoras para streaming y latencia aplicadas y guardadas en {config_path}")


def main():
    print("🚀 Aplicando optimizaciones adicionales para ARM Axion")
    print("   Objetivo: Reducir latencia y mejorar streaming")
    print("=" * 60)
    
    print("\n1. Aplicando optimización de FP8 KV Cache...")
    apply_fp8_kv_cache_optimization()
    
    print("\n2. Aplicando optimización de Advanced Captured Graphs...")
    apply_advanced_captured_graphs_optimization()
    
    print("\n3. Aplicando mejoras para streaming y latencia...")
    apply_streaming_latency_improvements()
    
    print("\n✅ Todas las optimizaciones adicionales han sido aplicadas")
    print("\n📋 Resumen de optimizaciones:")
    print("   • FP8 KV Cache: Reduce uso de memoria y mejora latencia")
    print("   • Advanced Captured Graphs: Menor overhead de ejecución")
    print("   • Scheduler adjustments: Mejora TTFT y streaming")
    print("   • Chunked Prefill: Mejora respuesta inicial en streaming")
    print("\n💡 Nota: Reiniciar el servidor para que los cambios tengan efecto")
    

if __name__ == "__main__":
    main()