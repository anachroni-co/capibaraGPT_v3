#!/usr/bin/env python3
"""
Multi-Model vLLM Inference Server for ARM Axion - CLASSIC BACKEND VERSION
Compatible with OpenAI API format
"""

import json
import time
import asyncio
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# FORZAR USO DEL BACKEND CLÁSICO ANTES DE IMPORTAR vLLM
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_ENABLE_V1_ENGINE'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# vLLM imports - Debe ser despues de configurar las variables de entorno
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

app = FastAPI(title="vLLM Multi-Model Server (CLASSIC)", version="1.0.0")

# Global state
models: Dict[str, LLM] = {}
config: Dict = {}
loaded_models: set = set()


@dataclass
class ModelInfo:
    """Model metadata"""
    model_id: str
    model_path: str
    domain: str
    description: str
    quantization: str
    status: str = "unloaded"
    loaded_at: Optional[float] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stream: bool = False


def load_config(config_path: str = "config.production.json") -> Dict:
    """Load configuration file"""
    with open(config_path) as f:
        return json.load(f)


def format_messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert chat messages to prompt format"""
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"<|system|>\n{msg.content}\n"
        elif msg.role == "user":
            prompt += f"<|user|>\n{msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"<|assistant|>\n{msg.content}\n"

    prompt += "<|assistant|>\n"
    return prompt


def load_model(model_id: str) -> LLM:
    """Load a model into memory - Classic backend version"""
    global config, models, loaded_models

    expert = next((e for e in config["experts"] if e["expert_id"] == model_id), None)
    if not expert:
        raise ValueError(f"Model {model_id} not found in config")

    print(f"Loading model: {model_id} from {expert['model_path']} (CLASSIC Backend)")

    # vLLM initialization params - Classic backend specific
    model = LLM(
        model=expert["model_path"],
        tensor_parallel_size=expert.get("tensor_parallel_size", 1),
        dtype=expert.get("dtype", "float16"),
        trust_remote_code=expert.get("trust_remote_code", True),
        # Classic backend parameters - compatible con ARM64 CPU
        gpu_memory_utilization=expert.get("gpu_memory_utilization", 0.9),
        max_num_seqs=expert.get("max_num_seqs", 256),
        max_model_len=expert.get("max_model_len", 4096),
        quantization=expert.get("quantization") if expert.get("quantization") != "q4_0" else None,
        enforce_eager=True,  # Compatible con classic backend
        # Evitar parámetros incompatibles
    )

    models[model_id] = model
    loaded_models.add(model_id)
    print(f"✓ Model {model_id} loaded successfully (CLASSIC)")

    return model


def get_model(model_id: str) -> LLM:
    """Get model, loading if necessary"""
    if model_id not in models:
        load_model(model_id)
    return models[model_id]


@app.on_event("startup")
async def startup_event():
    """Initialize server"""
    global config

    print("="*70)
    print("  vLLM Multi-Model Inference Server")
    print("  ARM Axion Optimized - CLASSIC BACKEND")
    print("="*70)
    print("Environment variables set:")
    print(f"  - VLLM_USE_V1: {os.environ.get('VLLM_USE_V1', 'not set')}")
    print(f"  - VLLM_ENABLE_V1_ENGINE: {os.environ.get('VLLM_ENABLE_V1_ENGINE', 'not set')}")
    print("")

    config = load_config()
    print(f"Loaded configuration with {len(config['experts'])} experts")
    print("")

    # Warm up first model if lazy loading is disabled
    if not config.get("lazy_loading", {}).get("enabled", False):
        warmup_size = config.get("lazy_loading", {}).get("warmup_pool_size", 1)
        for expert in config["experts"][:warmup_size]:
            try:
                load_model(expert["expert_id"])
            except Exception as e:
                print(f"Warning: Failed to load {expert['author_id']}: {e}")

    print("")
    print("✓ Server ready - CLASSIC BACKEND")
    print("")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "vLLM Multi-Model Server",
        "version": "1.0.0",
        "backend": "classic",
        "models_loaded": len(loaded_models),
        "models_available": len(config.get("experts", []))
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    models_list = []

    for expert in config.get("experts", []):
        models_list.append({
            "id": expert["expert_id"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "capibara6",
            "permission": [],
            "root": expert["expert_id"],
            "parent": None,
            "description": expert.get("description", ""),
            "domain": expert.get("domain", ""),
            "status": "loaded" if expert["expert_id"] in loaded_models else "available"
        })

    return {
        "object": "list",
        "data": models_list
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completions endpoint (OpenAI compatible)"""
    try:
        model = get_model(request.model)

        # Convert messages to prompt
        prompt = format_messages_to_prompt(request.messages)

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        # Generate
        start_time = time.time()
        outputs = model.generate([prompt], sampling_params)
        generation_time = time.time() - start_time

        output = outputs[0]
        generated_text = output.outputs[0].text

        # Format response
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
            },
            "performance": {
                "generation_time": round(generation_time, 3),
                "tokens_per_second": round(len(output.outputs[0].token_ids) / generation_time, 2)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Text completions endpoint (OpenAI compatible)"""
    try:
        model = get_model(request.model)

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        # Generate
        start_time = time.time()
        outputs = model.generate([request.prompt], sampling_params)
        generation_time = time.time() - start_time

        output = outputs[0]
        generated_text = output.outputs[0].text

        # Format response
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "text": generated_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
            },
            "performance": {
                "generation_time": round(generation_time, 3),
                "tokens_per_second": round(len(output.outputs[0].token_ids) / generation_time, 2)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
async def ollama_generate(request: dict):
    """Ollama-compatible generate endpoint"""
    try:
        model_name = request.get("model")
        prompt = request.get("prompt")

        # Find model by name (try exact match first, then partial)
        model_id = None
        for expert in config["experts"]:
            if expert["expert_id"] == model_name or model_name in expert["expert_id"]:
                model_id = expert["expert_id"]
                break

        if not model_id:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        model = get_model(model_id)

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=request.get("temperature", 0.7),
            top_p=request.get("top_p", 0.9),
            max_tokens=request.get("max_tokens", 2048),
        )

        # Generate
        start_time = time.time()
        outputs = model.generate([prompt], sampling_params)
        total_duration = time.time() - start_time

        output = outputs[0]
        generated_text = output.outputs[0].text

        # Format Ollama-compatible response
        return {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "response": generated_text,
            "done": True,
            "total_duration": int(total_duration * 1e9),
            "load_duration": 0,
            "prompt_eval_count": len(output.prompt_token_ids),
            "prompt_eval_duration": int(total_duration * 0.1 * 1e9),
            "eval_count": len(output.outputs[0].token_ids),
            "eval_duration": int(total_duration * 0.9 * 1e9),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "backend": "classic",
        "models_loaded": len(loaded_models),
        "models_available": len(config.get("experts", []))
    }


@app.get("/stats")
async def stats():
    """Statistics endpoint"""
    return {
        "backend": "classic",
        "models_loaded": list(loaded_models),
        "models_available": [e["expert_id"] for e in config.get("experts", [])],
        "config": {
            "lazy_loading": config.get("lazy_loading", {}),
            "server": config.get("server_config", {})
        }
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--config", default="config.production.json", help="Config file path")
    args = parser.parse_args()

    print(f"Starting server on {args.host}:{args.port}")
    print(f"Using config: {args.config}")
    print(f"Backend: Classic (not V1 Engine)")
    print(f"Environment variables set:")
    print(f"  - VLLM_USE_V1: {os.environ.get('VLLM_USE_V1', 'not set')}")
    print(f"  - VLLM_ENABLE_V1_ENGINE: {os.environ.get('VLLM_ENABLE_V1_ENGINE', 'not set')}")

    # Establecer las variables de entorno antes de iniciar
    os.environ['VLLM_USE_V1'] = '0'
    os.environ['VLLM_ENABLE_V1_ENGINE'] = '0'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )