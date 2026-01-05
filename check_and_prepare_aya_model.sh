#!/bin/bash
# Script para verificar y manejar la disponibilidad del modelo Aya Expanse para el sistema de 5 modelos

echo "🔍 Verificando disponibilidad del modelo Aya Expanse 8B..."

# Verificar si tenemos el modelo completo descargado
AYA_MODEL_PATH="/home/elect/models/aya-expanse-8b"
FULL_MODEL_PATH="/home/elect/models/aya-expanse-8b-full"

MODEL_AVAILABLE=false

# Buscar archivos esenciales del modelo
if [ -d "$AYA_MODEL_PATH" ] && [ -f "$AYA_MODEL_PATH/config.json" ]; then
    echo "✅ Modelo encontrado en: $AYA_MODEL_PATH"
    MODEL_PATH="$AYA_MODEL_PATH"
    MODEL_AVAILABLE=true
elif [ -d "$FULL_MODEL_PATH" ] && [ -f "$FULL_MODEL_PATH/config.json" ]; then
    echo "✅ Modelo encontrado en: $FULL_MODEL_PATH"
    MODEL_PATH="$FULL_MODEL_PATH"
    MODEL_AVAILABLE=true
else
    echo "❌ Modelos Aya Expanse no completamente descargados"
    echo ""
    
    # Intentar descargar de nuevo con autenticación
    echo "📡 Intentando descargar el modelo con token de Hugging Face..."
    
    # Crear directorio si no existe
    mkdir -p "$AYA_MODEL_PATH"
    
    # Intentar descargar usando Python con el token configurado
    python3 -c "
import os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

try:
    print('Intentando descargar CohereLabs/aya-expanse-8b...')
    result = snapshot_download(
        repo_id='CohereLabs/aya-expanse-8b',
        local_dir='$AYA_MODEL_PATH',
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=2
    )
    print('✅ Descarga completada exitosamente')
    exit(0)
except GatedRepoError:
    print('❌ Repositorio con acceso restringido - requiere acceso explícito')
    print('💡 Visita: https://huggingface.co/CohereLabs/aya-expanse-8b')
    print('   Y solicita acceso en la pestaña \"Files and versions\"')
    exit(1)
except RepositoryNotFoundError:
    print('❌ Repositorio no encontrado')
    exit(1)
except Exception as e:
    print(f'❌ Error durante la descarga: {e}')
    exit(1)
" 2>&1

    # Verificar si la descarga fue exitosa
    if [ $? -eq 0 ] && [ -f "$AYA_MODEL_PATH/config.json" ]; then
        MODEL_AVAILABLE=true
        MODEL_PATH="$AYA_MODEL_PATH"
        echo "✅ Modelo descargado exitosamente"
    else
        echo "⚠️  No se pudo descargar el modelo Aya Expanse"
        echo "   Usando modelo de reemplazo: NousResearch/Hermes-2-Pro-Mistral-7B"
        echo ""
        
        # Descargar modelo de reemplazo
        REPLACEMENT_MODEL_PATH="/home/elect/models/hermes-2-pro-mistral-7b"
        if [ ! -f "$REPLACEMENT_MODEL_PATH/config.json" ]; then
            echo "📡 Descargando modelo de reemplazo: NousResearch/Hermes-2-Pro-Mistral-7B"
            python3 -c "
import os
from huggingface_hub import snapshot_download

print('Descargando modelo de reemplazo: NousResearch/Hermes-2-Pro-Mistral-7B...')
snapshot_download(
    repo_id='NousResearch/Hermes-2-Pro-Mistral-7B',
    local_dir='$REPLACEMENT_MODEL_PATH',
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=2
)
print('✅ Modelo de reemplazo descargado')
"
        fi
        
        if [ -f "$REPLACEMENT_MODEL_PATH/config.json" ]; then
            # Crear una configuración alternativa con el modelo de reemplazo
            echo "📝 Creando configuración alternativa con modelo de reemplazo..."
            cat > "/home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.five_models_alt.json" << 'EOF'
{
  "experts": [
    {
      "expert_id": "phi4_fast",
      "model_path": "/home/elect/models/phi-4-mini",
      "domain": "general",
      "description": "Modelo rápido para respuestas simples y directas - Optimizado para ARM Axion",
      "priority": 5,
      "quantization": "awq",
      "tensor_parallel_size": 1,
      "gpu_memory_utilization": 0.85,
      "max_num_seqs": 1024,
      "enable_neon": true,
      "enable_chunked_prefill": true,
      "max_num_batched_tokens": 32768,
      "max_model_len": 4096,
      "enforce_eager": false,
      "kv_cache_dtype": "auto",
      "use_v2_block_manager": true,
      "enable_prefix_caching": true,
      "device": "auto",
      "dtype": "float16",
      "use_captured_graph": true,
      "swap_space": 4,
      "cpu_offload_gb": 0,
      "neon_optimizations": {
        "matmul_8x8": true,
        "flash_attention": true,
        "rmsnorm": true,
        "rope": true,
        "swiglu": true,
        "softmax_fast_exp": true
      },
      "use_cases": [
        "preguntas simples",
        "respuestas rápidas",
        "chistes",
        "saludos",
        "respuestas directas"
      ]
    },
    {
      "expert_id": "mistral_balanced",
      "model_path": "/home/elect/models/mistral-7b-instruct-v0.2",
      "domain": "technical",
      "description": "Modelo equilibrado para tareas técnicas intermedias - Optimizado para ARM Axion",
      "priority": 4,
      "quantization": "awq",
      "tensor_parallel_size": 1,
      "gpu_memory_utilization": 0.8,
      "max_num_seqs": 512,
      "enable_neon": true,
      "enable_chunked_prefill": true,
      "max_num_batched_tokens": 16384,
      "max_model_len": 8192,
      "enforce_eager": false,
      "kv_cache_dtype": "auto",
      "use_v2_block_manager": true,
      "enable_prefix_caching": true,
      "device": "auto",
      "dtype": "float16",
      "use_captured_graph": true,
      "swap_space": 4,
      "cpu_offload_gb": 0,
      "neon_optimizations": {
        "matmul_8x8": true,
        "flash_attention": true,
        "rmsnorm": true,
        "rope": true,
        "swiglu": true,
        "softmax_fast_exp": true
      },
      "use_cases": [
        "explicaciones técnicas",
        "código y programación",
        "análisis intermedio",
        "redacción",
        "documentación"
      ]
    },
    {
      "expert_id": "qwen_coder",
      "model_path": "/home/elect/models/qwen2.5-coder-1.5b",
      "domain": "coding",
      "description": "Modelo especializado en código y programación - Optimizado para ARM Axion",
      "priority": 3,
      "quantization": "awq",
      "tensor_parallel_size": 1,
      "gpu_memory_utilization": 0.8,
      "max_num_seqs": 512,
      "enable_neon": true,
      "enable_chunked_prefill": true,
      "max_num_batched_tokens": 16384,
      "max_model_len": 8192,
      "enforce_eager": false,
      "kv_cache_dtype": "auto",
      "use_v2_block_manager": true,
      "enable_prefix_caching": true,
      "device": "auto",
      "dtype": "float16",
      "use_captured_graph": true,
      "swap_space": 4,
      "cpu_offload_gb": 0,
      "neon_optimizations": {
        "matmul_8x8": true,
        "flash_attention": true,
        "rmsnorm": true,
        "rope": true,
        "swiglu": true,
        "softmax_fast_exp": true
      },
      "use_cases": [
        "generación de código",
        "debugging",
        "explicaciones técnicas",
        "refactoring",
        "documentación de código"
      ]
    },
    {
      "expert_id": "gemma3_multimodal",
      "model_path": "/home/elect/models/gemma-3-27b-it",
      "domain": "multimodal_expert",
      "description": "Modelo multimodal para texto + imágenes, análisis complejo y contexto largo - Optimizado para ARM Axion",
      "priority": 2,
      "quantization": null,
      "tensor_parallel_size": 1,
      "gpu_memory_utilization": 0.75,
      "max_num_seqs": 128,
      "enable_neon": true,
      "enable_chunked_prefill": true,
      "enable_flash_attention": true,
      "max_num_batched_tokens": 8192,
      "max_model_len": 24576,
      "trust_remote_code": true,
      "dtype": "bfloat16",
      "enforce_eager": false,
      "kv_cache_dtype": "auto",
      "use_v2_block_manager": true,
      "enable_prefix_caching": true,
      "device": "auto",
      "use_captured_graph": true,
      "swap_space": 8,
      "cpu_offload_gb": 0,
      "neon_optimizations": {
        "rmsnorm": true,
        "rope": true,
        "swiglu": true,
        "flash_attention": true,
        "matmul_8x8": true,
        "softmax_fast_exp": true,
        "quantization_q4": true,
        "acl_gemm": true
      },
      "flash_attention_config": {
        "block_size": 64,
        "enable_for_seq_len_above": 512,
        "max_seq_len": 32768
      },
      "use_cases": [
        "análisis profundo",
        "razonamiento complejo",
        "análisis multimodal",
        "imágenes + texto",
        "contexto largo",
        "multilingüe avanzado (140+ idiomas)",
        "research y documentación extensa",
        "análisis de PDFs con imágenes"
      ]
    },
    {
      "expert_id": "hermes_pro_multilingual",
      "model_path": "/home/elect/models/hermes-2-pro-mistral-7b",
      "domain": "multilingual_reasoning",
      "description": "Modelo de razonamiento avanzado y multilingüe de Hermes-2 Pro - Alternativa a Aya Expanse",
      "priority": 2,
      "quantization": "awq",
      "tensor_parallel_size": 1,
      "gpu_memory_utilization": 0.8,
      "max_num_seqs": 256,
      "enable_neon": true,
      "enable_chunked_prefill": true,
      "enable_flash_attention": true,
      "max_num_batched_tokens": 8192,
      "max_model_len": 8192,
      "trust_remote_code": false,
      "dtype": "float16",
      "enforce_eager": false,
      "kv_cache_dtype": "auto",
      "use_v2_block_manager": true,
      "enable_prefix_caching": true,
      "device": "auto",
      "use_captured_graph": true,
      "swap_space": 8,
      "cpu_offload_gb": 0,
      "neon_optimizations": {
        "rmsnorm": true,
        "rope": true,
        "swiglu": true,
        "flash_attention": true,
        "matmul_8x8": true,
        "softmax_fast_exp": true,
        "quantization_q4": true,
        "acl_gemm": true
      },
      "flash_attention_config": {
        "block_size": 64,
        "enable_for_seq_len_above": 512,
        "max_seq_len": 8192
      },
      "use_cases": [
        "análisis profundo",
        "razonamiento complejo",
        "multilingüe",
        "contexto largo",
        "research y documentación",
        "análisis de instrucciones complejas"
      ]
    }
  ],
  "lazy_loading": {
    "enabled": true,
    "warmup_pool_size": 2,
    "max_loaded_experts": 5,
    "memory_threshold": 0.8,
    "auto_unload_after_s": 300
  },
  "embedding_model": {
    "model_name": "all-MiniLM-L6-v2",
    "cache_size": 20000,
    "use_neon": true
  },
  "rag": {
    "enabled": true,
    "bridge_url": "http://localhost:8001",
    "collection": "capibara_docs",
    "detection_threshold": 0.5,
    "max_context_tokens": 2000,
    "notes": "Parallel RAG fetching: -40% latency on RAG queries"
  },
  "speculative_routing": {
    "enabled": true,
    "speculation_threshold": 0.85,
    "max_speculation_time": 0.5,
    "notes": "Start generation on high-confidence first chunk: -20-30% TTFT on obvious queries"
  },
  "enable_consensus": true,
  "consensus_model": null,
  "chunk_size": 64,
  "routing_threshold": 0.7,
  "use_fast_classifier": true,
  "server_config": {
    "host": "0.0.0.0",
    "port": 8080,
    "log_level": "info",
    "allow_credentials": true
  },
  "performance_tuning": {
    "swap_space": 12,
    "cpu_offload_gb": 0,
    "enforce_eager": true,
    "max_context_len_to_capture": 8192,
    "enable_prefix_caching": false,
    "use_v2_block_manager": false,
    "num_scheduler_steps": 8,
    "enable_chunked_prefill": true,
    "max_num_cached_tokens": 32768
  },
  "optimization_flags": {
    "arm_neon": true,
    "acl_integrated": true,
    "matmul_optimized": true,
    "memory_efficient_sampling": true,
    "batch_scheduler": "vllm_default"
  },
  "notes": {
    "deployment": "Optimizado para Google Cloud ARM Axion C4A-standard-32 (europe-southwest1-b)",
    "vm_name": "models-europe",
    "memory_estimation": {
      "phi4_awq": "~1.2 GB",
      "mistral_awq": "~2.8 GB",
      "qwen_awq": "~3.2 GB",
      "gemma3_original_bfloat16": "~14 GB",
      "hermes_pro_awq": "~4.5 GB (7B params con AWQ)"
    },
    "expected_performance": {
      "phi4_ttft": "~0.15s (con optimizaciones)",
      "mistral_ttft": "~0.3s (con optimizaciones)",
      "qwen_ttft": "~0.4s (con optimizaciones)",
      "gemma3_ttft": "~0.6s (con NEON+ACL, bfloat16)",
      "hermes_pro_ttft": "~0.45s (con optimizaciones, 7B params)",
      "throughput_combined": "100-120 req/min (con 5 modelos multilingües)"
    },
    "model_comparison": {
      "hermes_pro_advantages": [
        "Capacidad de razonamiento avanzado",
        "Buen rendimiento multilingüe",
        "Alternativa viable a Aya Expanse",
        "Menor tamaño que modelos más grandes",
        "Excelente para tareas de consenso complejas"
      ]
    },
    "neon_kernels_impact": {
      "matmul_speedup": "1.5x (8x8 tiles + prefetching + ACL)",
      "attention_speedup": "1.8x (Flash Attention para seq_len > 512)",
      "swiglu_speedup": "1.5x (fusionado + optimizado)",
      "rmsnorm_speedup": "5x (vectorizado + optimizado)",
      "softmax_speedup": "1.6x (fast exp + optimizado)",
      "rope_speedup": "1.4x (vectorizado + optimizado)",
      "global_speedup": "1.7-2.0x (60-80% mejora total con ACL)"
    },
    "quantization_improvement": {
      "awq_reduction": "30-40% memoria",
      "gptq_reduction": "40-50% memoria",
      "q4_reduction": "50-60% memoria",
      "throughput_improvement": "15-25% por modelo"
    }
  }
}
EOF
            echo "✅ Configuración alternativa creada con modelo de reemplazo"
        else
            echo "❌ No se pudo descargar ni el modelo original ni el de reemplazo"
            echo "   Continuando con configuración de 4 modelos"
        fi
    fi
fi

# Verificar si el modelo original está disponible
if [ "$MODEL_AVAILABLE" = true ]; then
    # Actualizar la configuración para incluir el modelo Aya Expanse
    echo "📝 Actualizando configuración para incluir Aya Expanse..."
    sed -i "s|/home/elect/models/aya-expanse-8b|$MODEL_PATH|g" /home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.five_models_with_aya.json
    echo "✅ Configuración actualizada con la ruta correcta del modelo"
    echo "   Modelo Aya Expanse incluido en el sistema de 5 modelos"
else
    # Si no se pudo descargar el Aya Expanse, usar la configuración de reemplazo
    if [ -f "/home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.five_models_alt.json" ]; then
        echo "🔄 Usando configuración alternativa con modelo de reemplazo"
        cp /home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.five_models_alt.json /home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.five_models_with_aya.json
    else
        echo "🔄 No se pudo crear configuración alternativa, manteniendo los 4 modelos actuales"
    fi
fi

echo ""
echo "✅ Proceso de verificación completado"
echo "   - Directorio del modelo: $MODEL_PATH (si disponible)"
echo "   - Configuración actualizada: /home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.five_models_with_aya.json"