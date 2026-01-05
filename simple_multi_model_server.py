#!/usr/bin/env python3
"""
Servidor Multi-Modelo Simplificado para ARM-Axion
Sin dependencias complejas - Solo vLLM directo
"""
import os
import sys
import json
import time
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import asyncio

# Forzar backend clásico
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_ENABLE_V1_ENGINE'] = '0'

# Imports de vLLM
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
except ImportError:
    print("ERROR: vLLM no está instalado")
    sys.exit(1)

app = FastAPI(
    title="ARM-Axion Multi-Model Server (Simple)",
    version="1.0.0",
    description="Servidor simplificado para 5 modelos en ARM-Axion"
)

# Configuración de modelos
MODELS_CONFIG = {
    "phi4-fast": {
        "path": "/home/elect/models/phi-4-mini",
        "description": "Phi4-Fast (14B) - Respuestas rápidas",
        "max_model_len": 4096,
        "max_num_seqs": 256
    },
    "qwen25-coder": {
        "path": "/home/elect/models/qwen2.5-coder-1.5b",
        "description": "Qwen2.5-Coder (1.5B) - Experto en código",
        "max_model_len": 8192,
        "max_num_seqs": 512
    },
    "mistral7b-balanced": {
        "path": "/home/elect/models/mistral-7b-instruct-v0.2",
        "description": "Mistral7B (7B) - Equilibrado técnico",
        "max_model_len": 8192,
        "max_num_seqs": 256
    },
    "gemma3-27b": {
        "path": "/home/elect/models/gemma-3-27b-it",
        "description": "Gemma3-27B (27B) - Análisis complejo",
        "max_model_len": 4096,
        "max_num_seqs": 64
    },
    "gptoss-20b": {
        "path": "/home/elect/models/gpt-oss-20b",
        "description": "GPT-OSS-20B (20B) - Razonamiento avanzado",
        "max_model_len": 4096,
        "max_num_seqs": 128
    }
}

# Estado global
loaded_models: Dict[str, LLM] = {}
model_load_times: Dict[str, float] = {}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 150
    top_p: float = 0.9
    stream: bool = False

def format_chat_messages(messages: List[ChatMessage]) -> str:
    """Convertir mensajes de chat a prompt"""
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n"

    prompt += "Assistant:"
    return prompt

def load_model_lazy(model_id: str) -> LLM:
    """Cargar modelo solo cuando se necesita (lazy loading)"""
    global loaded_models, model_load_times

    if model_id in loaded_models:
        print(f"✅ Modelo {model_id} ya está cargado")
        return loaded_models[model_id]

    if model_id not in MODELS_CONFIG:
        raise ValueError(f"Modelo desconocido: {model_id}")

    config = MODELS_CONFIG[model_id]
    print(f"🔄 Cargando modelo {model_id} desde {config['path']}...")

    start_time = time.time()

    try:
        # Configuración básica de vLLM (auto-detecta CPU en ARM)
        llm = LLM(
            model=config['path'],
            trust_remote_code=True,
            max_num_seqs=config.get('max_num_seqs', 256),
            max_model_len=config.get('max_model_len', 4096),
            disable_log_stats=True,
        )

        loaded_models[model_id] = llm
        model_load_times[model_id] = time.time() - start_time

        print(f"✅ Modelo {model_id} cargado en {model_load_times[model_id]:.2f}s")
        return llm

    except Exception as e:
        print(f"❌ Error cargando {model_id}: {e}")
        raise

@app.get("/")
async def root():
    """Información del servidor"""
    return {
        "name": "ARM-Axion Multi-Model Server (Simple)",
        "version": "1.0.0",
        "backend": "vLLM-direct",
        "platform": "ARM64",
        "models_available": len(MODELS_CONFIG),
        "models_loaded": len(loaded_models),
        "loaded_models": list(loaded_models.keys())
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "models_available": len(MODELS_CONFIG)
    }

@app.get("/models")
@app.get("/v1/models")
async def list_models():
    """Listar modelos disponibles"""
    models = []
    for model_id, config in MODELS_CONFIG.items():
        models.append({
            "id": model_id,
            "object": "model",
            "description": config["description"],
            "status": "loaded" if model_id in loaded_models else "available",
            "load_time": model_load_times.get(model_id)
        })

    return {"models": models}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Endpoint de chat completions (OpenAI compatible)"""
    try:
        # Cargar modelo si no está cargado
        llm = load_model_lazy(request.model)

        # Convertir mensajes a prompt
        prompt = format_chat_messages(request.messages)

        # Parámetros de sampling
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        # Generar respuesta
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        generation_time = time.time() - start_time

        # Formatear respuesta
        if outputs and len(outputs) > 0:
            generated_text = outputs[0].outputs[0].text

            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(generated_text.split()),
                    "total_tokens": len(prompt.split()) + len(generated_text.split())
                },
                "generation_time": generation_time
            }
        else:
            raise HTTPException(status_code=500, detail="No se generó respuesta")

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"❌ Error en chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate(request: dict):
    """Endpoint simple de generación"""
    try:
        model_id = request.get("model")
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 150)
        temperature = request.get("temperature", 0.7)

        if not model_id or not prompt:
            raise HTTPException(status_code=400, detail="model y prompt son requeridos")

        llm = load_model_lazy(model_id)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        outputs = llm.generate([prompt], sampling_params)

        if outputs and len(outputs) > 0:
            return {
                "model": model_id,
                "response": outputs[0].outputs[0].text,
                "prompt": prompt
            }
        else:
            raise HTTPException(status_code=500, detail="No response generated")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    import argparse

    parser = argparse.ArgumentParser(description="ARM-Axion Multi-Model Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8080, help="Puerto")

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 ARM-AXION MULTI-MODEL SERVER (SIMPLE)")
    print("=" * 80)
    print(f"\n5 Modelos disponibles:")
    for model_id, config in MODELS_CONFIG.items():
        print(f"  - {model_id}: {config['description']}")

    print(f"\n📍 Servidor: http://{args.host}:{args.port}")
    print(f"📚 Modelos: http://{args.host}:{args.port}/models")
    print(f"💬 Chat: POST http://{args.host}:{args.port}/v1/chat/completions")
    print("\n" + "=" * 80 + "\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
