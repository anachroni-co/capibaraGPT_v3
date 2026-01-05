#!/usr/bin/env python3
"""
vLLM ARM-Axion Server con Sistema de Fallback para Operaciones Personalizadas
Implementa el sistema multi-modelo con 5 modelos (Qwen2.5, Phi4-mini, Mistral7B, Gemma3-27B, GPT-OSS-20B)
con soporte para plataforma ARM-Axion y fallback automático para operaciones no disponibles
"""

import os
import sys
import asyncio
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import torch

# Asegurar que nuestro código modificado está en el path
vllm_path = '/home/elect/capibara6/vllm-source-modified'
if vllm_path not in sys.path:
    sys.path.insert(0, vllm_path)

# Importar el sistema de fallback antes de importar vllm
from custom_ops_fallback import try_initialize_custom_ops_with_native, get_available_ops_info

# Intentar inicializar las operaciones personalizadas (con fallback si es necesario)
print("🚀 Inicializando sistema de operaciones personalizadas ARM-Axion...")
try_initialize_custom_ops_with_native()

# Ahora importar vLLM
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.platforms import current_platform

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse


app = FastAPI(title="vLLM ARM-Axion Multi-Model Server", version="2.0.0")

# Configuración del modelo
class ARMVLLMManager:
    def __init__(self):
        self.models: Dict[str, LLM] = {}
        self.model_configs = {
            "qwen2.5-coder": {
                "model_path": "/home/elect/models/qwen2.5-coder-1.5b",
                "description": "Modelo especializado en código y programación"
            },
            "phi4-mini": {
                "model_path": "/home/elect/models/phi-4-mini", 
                "description": "Modelo rápido para respuestas simples"
            },
            "mistral-7b": {
                "model_path": "/home/elect/models/mistral-7b-instruct-v0.2",
                "description": "Modelo equilibrado para tareas técnicas"
            },
            "gemma-3-27b": {
                "model_path": "/home/elect/models/gemma-3-27b-it",
                "description": "Modelo multimodal para tareas complejas"
            },
            "gpt-oss-20b": {
                "model_path": "/home/elect/models/gpt-oss-20b",
                "description": "Modelo de razonamiento complejo"
            }
        }
        
        # Verificar disponibilidad de plataforma
        print(f"🌍 Plataforma detectada: {current_platform.device_type}")
        print(f"   ¿Es CPU?: {current_platform.is_cpu()}")
        
        # Verificar operaciones personalizadas
        print("\n🔧 Estado de operaciones personalizadas:")
        ops_info = get_available_ops_info()
        native_ops = sum(1 for info in ops_info.values() if info['implementation'] == 'native')
        fallback_ops = sum(1 for info in ops_info.values() if info['implementation'] == 'fallback')
        print(f"   ✓ Operaciones nativas: {native_ops}")
        print(f"   🔄 Operaciones con fallback: {fallback_ops}")
        print(f"   Total operaciones verificadas: {len(ops_info)}")

    def load_model(self, model_id: str) -> LLM:
        """Cargar un modelo con configuración específica para ARM"""
        if model_id in self.models:
            return self.models[model_id]

        if model_id not in self.model_configs:
            raise ValueError(f"Modelo {model_id} no encontrado en configuración")

        config = self.model_configs[model_id]
        print(f"\n📥 Cargando modelo: {model_id} ({config['description']})")
        print(f"   Ruta: {config['model_path']}")

        try:
            # Configuración conservadora para ARM-Axion
            model = LLM(
                model=config['model_path'],
                tensor_parallel_size=1,  # Conservador para ARM
                dtype="float16",  # Menor uso de memoria
                enforce_eager=True,  # Mayor estabilidad
                gpu_memory_utilization=0.5,  # Conservador
                max_num_seqs=32,  # Limitar batch para estabilidad
                trust_remote_code=True,
                # Forzar uso de CPU para ARM
                device="cpu" if torch.cuda.is_available() else "cpu"
            )

            self.models[model_id] = model
            print(f"   ✅ {model_id} cargado exitosamente")
            return model

        except Exception as e:
            print(f"   ❌ Error al cargar {model_id}: {e}")
            raise

    def generate(self, model_id: str, prompt: str, **kwargs) -> str:
        """Generar texto usando el modelo especificado"""
        model = self.get_model(model_id)
        
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_tokens", 512),
            stop=kwargs.get("stop", None)
        )
        
        start_time = time.time()
        outputs = model.generate([prompt], sampling_params)
        end_time = time.time()
        
        result = outputs[0].outputs[0].text
        tokens_generated = len(outputs[0].outputs[0].token_ids)
        generation_time = end_time - start_time
        tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
        
        return {
            "text": result,
            "tokens_generated": tokens_generated,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_sec
        }


# Instancia global del manager
model_manager = ARMVLLMManager()


@app.on_event("startup")
async def startup_event():
    print("\n✅ Inicialización del servidor ARM-Axion completada")
    print("📡 El servidor está listo para recibir solicitudes")
    print("   - Compatible con API OpenAI")
    print("   - Soporte para 5 modelos ARM-Axion")
    print("   - Sistema de fallback para operaciones personalizadas")


@app.get("/")
async def root():
    return {
        "message": "vLLM ARM-Axion Multi-Model Server",
        "version": "2.0.0",
        "platform": str(current_platform.device_type),
        "models_loaded": len(model_manager.models),
        "models_available": list(model_manager.model_configs.keys()),
        "custom_ops_status": get_available_ops_info()
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "platform": str(current_platform.device_type),
        "cpu_detected": current_platform.is_cpu(),
        "models_loaded": len(model_manager.models)
    }


@app.post("/generate")
async def generate(request: dict):
    try:
        model_id = request.get("model")
        prompt = request.get("prompt")
        
        if not model_id or not prompt:
            raise HTTPException(status_code=400, detail="Model ID and prompt are required")
        
        if model_id not in model_manager.model_configs:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not available")
        
        result = model_manager.generate(model_id, prompt, **request)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    return {
        "models": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "arm-axion",
                "description": config["description"],
                "status": "loaded" if model_id in model_manager.models else "available"
            }
            for model_id, config in model_manager.model_configs.items()
        ]
    }


# Configurar API compatible con OpenAI
@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    try:
        model_id = request.get("model")
        messages = request.get("messages", [])
        
        # Convertir mensajes a un solo prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            prompt_parts.append(f"{role.upper()}: {content}")
        
        prompt = "\n".join(prompt_parts) + "\nASSISTANT:"
        
        if not model_id:
            raise HTTPException(status_code=400, detail="Model is required")
        
        result = model_manager.generate(
            model_id, 
            prompt,
            temperature=request.get("temperature", 0.7),
            max_tokens=request.get("max_tokens", 512),
            top_p=request.get("top_p", 0.9)
        )
        
        return {
            "id": f"chat-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["text"]
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": result["tokens_generated"],
                "total_tokens": len(prompt.split()) + result["tokens_generated"]
            },
            "performance": {
                "tokens_per_second": result["tokens_per_second"],
                "generation_time": result["generation_time"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host para el servidor")
    parser.add_argument("--port", type=int, default=8000, help="Puerto para el servidor")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("   SERVIDOR VLLM ARM-AXION INICIANDO")
    print("   Con sistema de fallback para operaciones personalizadas")
    print("="*60)
    
    # Mostrar estado de modelos disponibles
    print(f"\n📚 Modelos disponibles: {list(model_manager.model_configs.keys())}")
    print(f"🌍 Plataforma ARM-Axion detectada: {current_platform.device_type}")
    print(f"   ¿Es CPU?: {current_platform.is_cpu()}")
    
    print(f"\n📡 Iniciando servidor en {args.host}:{args.port}")
    print("   Presiona Ctrl+C para detener")
    print("="*60)
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")