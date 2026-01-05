#!/usr/bin/env python3
"""
Multi-Model Server ARM-Axion - Backend Clásico
Con parches para operaciones personalizadas no disponibles en ARM
"""

import os
import sys
import json
import time
import torch

# Configuración crítica para ARM-Axion
os.environ["VLLM_USE_V1"] = "0"  # Deshabilitar V1 engine
os.environ["VLLM_ENABLE_V1_ENGINE"] = "0"  # Deshabilitar V1 engine
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
os.environ["VLLM_USE_FLASHINFER"] = "0"
os.environ["VLLM_NO_DEPRECATION_WARNING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:disabled"  # Para evitar problemas de memoria en ARM

# Asegurar el path
sys.path.insert(0, "/home/elect/capibara6/vllm-source-modified")

# Aplicar patches de fallback antes de importar vLLM
def apply_fallback_patches():
    """Aplicar parches de fallback a operaciones personalizadas"""
    import torch
    
    # Si no hay namespace _C, crearlo
    if not hasattr(torch.ops, "_C"):
        class FakeC:
            pass
        torch.ops._C = FakeC()
    
    # Implementar operaciones de fallback
    if not hasattr(torch.ops._C, "rms_norm"):
        def rms_norm_fallback(output, input, weight, epsilon):
            variance = input.pow(2).mean(-1, keepdim=True)
            inv_rms = torch.rsqrt(variance + epsilon)
            normalized = input * inv_rms
            output.copy_(normalized * weight)
        torch.ops._C.rms_norm = rms_norm_fallback
        
    if not hasattr(torch.ops._C, "fused_add_rms_norm"):
        def fused_add_rms_norm_fallback(input, residual, weight, epsilon):
            input.add_(residual)
            variance = input.pow(2).mean(-1, keepdim=True)
            inv_rms = torch.rsqrt(variance + epsilon)
            input.mul_(inv_rms)
            input.mul_(weight)
        torch.ops._C.fused_add_rms_norm = fused_add_rms_norm_fallback

apply_fallback_patches()

# Ahora importar vLLM
from vllm import LLM, SamplingParams
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="vLLM ARM-Axion Server", version="1.0.0 - Classic Backend")

# Configuración de modelos
MODEL_CONFIGS = {
    "phi4-fast": {
        "path": "/home/elect/models/phi-4-mini",
        "description": "Modelo rápido para respuestas simples"
    },
    "qwen25-coder": {
        "path": "/home/elect/models/qwen2.5-coder-1.5b",
        "description": "Modelo experto en código"
    },
    "mistral7b-balanced": {
        "path": "/home/elect/models/mistral-7b-instruct-v0.2", 
        "description": "Modelo equilibrado para tareas técnicas"
    },
    "gemma3-27b": {
        "path": "/home/elect/models/gemma-3-27b-it-awq",
        "description": "Modelo para tareas complejas y contexto largo"
    },
    "gptoss-20b": {
        "path": "/home/elect/models/gpt-oss-20b",
        "description": "Modelo de razonamiento complejo"
    }
}

models = {}

def load_model(model_id):
    """Cargar un modelo en memoria"""
    if model_id in models:
        return models[model_id]
    
    config = MODEL_CONFIGS[model_id]
    print(f"Loading {model_id} from {config['path']}...")
    
    # Cargar modelo con configuración conservadora para ARM
    model = LLM(
        model=config["path"],
        tensor_parallel_size=1,
        dtype="float16", 
        enforce_eager=True,
        gpu_memory_utilization=0.5,  # Conservador para ARM
        max_num_seqs=64,
        max_model_len=4096,
        quantization=None,  # Usar modelo sin cuantizar para evitar problemas
        trust_remote_code=True
    )
    
    models[model_id] = model
    print(f"✅ {model_id} loaded successfully")
    return model

@app.get("/")
def root():
    return {
        "name": "vLLM ARM-Axion Multi-Model Server",
        "backend": "classic",
        "version": "1.0.0",
        "platform": "ARM64",
        "models_loaded": len(models),
        "models_available": len(MODEL_CONFIGS)
    }

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": len(models)}

@app.get("/models")
def list_models():
    return {
        "models": [
            {
                "id": model_id,
                "object": "model", 
                "description": config["description"],
                "status": "loaded" if model_id in models else "available"
            }
            for model_id, config in MODEL_CONFIGS.items()
        ]
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080) 
    args = parser.parse_args()

    print("="*60)
    print("SERVER vLLM ARM-Axion INICIANDO - BACKEND CLÁSICO")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  VLLM_USE_V1: {os.environ.get('VLLM_USE_V1')}")
    print(f"  VLLM_USE_FLASHINFER: {os.environ.get('VLLM_USE_FLASHINFER')}")
    print("="*60)
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
