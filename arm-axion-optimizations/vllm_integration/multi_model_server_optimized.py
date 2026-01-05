#!/usr/bin/env python3
"""
Multi-Model vLLM Inference Server for ARM Axion
Compatible with OpenAI API format
Enhanced with Intelligent Routing
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

# FORZAR USO DEL BACKEND CLÁSICO Y COMPATIBILIDAD ARM ANTES DE IMPORTAR vLLM
os.environ["VLLM_USE_V1"] = "0"  # Deshabilitar V1 engine
os.environ["VLLM_ENABLE_V1_ENGINE"] = "0"  # Deshabilitar V1 engine
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
os.environ["VLLM_USE_FLASHINFER"] = "0"
os.environ["VLLM_NO_DEPRECATION_WARNING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:disabled"  # Para evitar problemas de memoria en ARM
# DESHABILITAR COMPONENTES PROBLEMÁTICOS EN ARM-Axion
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
os.environ["TORCHINDUCTOR_DISABLED"] = "1"
os.environ["TORCH_COMPILE_BACKEND"] = "eager"  # Usar backend más sencillo en ARM

import sys

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

# vLLM imports - ahora debería usar el backend clásico
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

app = FastAPI(title="vLLM Multi-Model Server with Intelligent Routing", version="2.0.0")

# Importar el router semántico
sys.path.insert(0, "/home/elect/capibara6/arm-axion-optimizations/vllm_integration")
from semantic_router import IncrementalSemanticRouter

# Global state
models: Dict[str, LLM] = {}
config: Dict = {}
loaded_models: set = set()
semantic_router: Optional[IncrementalSemanticRouter] = None


@dataclass
class ModelInfo:
    """Model metadata"""
    model_id: str
    model_path: str
    domain: str
    description: str
    quantization: str
    priority: int  # Lower number = higher priority for speed
    status: str = "unloaded"
    loaded_at: Optional[float] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None  # Now optional - can use semantic routing
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stream: bool = False


class CompletionRequest(BaseModel):
    model: Optional[str] = None  # Now optional - can use semantic routing
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stream: bool = False


def load_config() -> Dict:
    """Load configuration file from environment or default"""
    import os
    config_file = os.environ.get('VLLM_CONFIG_PATH', 'config.production.json')
    with open(config_file) as f:
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
    """Load a model into memory"""
    global config, models, loaded_models

    expert = next((e for e in config["experts"] if e["expert_id"] == model_id), None)
    if not expert:
        raise ValueError(f"Model {model_id} not found in config")

    print(f"Loading model: {model_id} from {expert['model_path']}")

    # vLLM initialization params
    model = LLM(
        model=expert["model_path"],
        tensor_parallel_size=expert.get("tensor_parallel_size", 1),
        gpu_memory_utilization=expert.get("gpu_memory_utilization", 0.9),
        max_num_seqs=expert.get("max_num_seqs", 256),
        max_model_len=expert.get("max_model_len", 4096),
        quantization=expert.get("quantization") if expert.get("quantization") != "q4_0" else None,
        dtype=expert.get("dtype", "float16"),
        trust_remote_code=True,
        enforce_eager=config.get("performance_tuning", {}).get("enforce_eager", False),
    )

    models[model_id] = model
    loaded_models.add(model_id)
    print(f"✓ Model {model_id} loaded successfully")

    return model


def get_model(model_id: str) -> LLM:
    """Get model, loading if necessary"""
    if model_id not in models:
        load_model(model_id)
    return models[model_id]


def initialize_semantic_router():
    """Initialize the semantic router with expert domains"""
    global semantic_router
    
    expert_domains = {expert["expert_id"]: expert["domain"] for expert in config["experts"]}
    
    semantic_router = IncrementalSemanticRouter(
        expert_domains=expert_domains,
        use_neon=True,
        routing_threshold=config.get("routing_threshold", 0.7),
        top_k_experts=1  # For now, just select one best expert
    )
    
    print(f"✅ Semantic router initialized with {len(expert_domains)} experts")


def route_request_to_model(query: str) -> str:
    """
    Route request to the most appropriate model based on content analysis
    
    Args:
        query: The input query to analyze
        
    Returns:
        The expert_id of the most appropriate model
    """
    global semantic_router
    
    if not semantic_router:
        # Fallback to fastest model if router not available
        return get_fastest_model()
    
    # Process the query with the semantic router
    request_id = f"routing_{int(time.time())}"
    semantic_router.start_request(request_id)
    
    # Route based on the query content
    prediction = semantic_router.process_chunk(request_id, query)
    
    # Get the top expert
    if prediction.expert_ids:
        model_id = prediction.expert_ids[0]
        
        # Check if confidence is high enough, otherwise use fastest model
        if prediction.confidence >= semantic_router.routing_threshold:
            return model_id
        else:
            # If confidence is low, use fastest model for simple queries
            return get_fastest_model()
    else:
        # Fallback to fastest model
        return get_fastest_model()


def get_fastest_model() -> str:
    """Return the ID of the fastest model based on priority settings"""
    # Get models sorted by priority (lower number = higher priority/speed)
    experts_with_priority = [
        (expert["expert_id"], expert.get("priority", 10)) 
        for expert in config["experts"]
    ]
    
    # Sort by priority (ascending order)
    experts_sorted = sorted(experts_with_priority, key=lambda x: x[1])
    
    # Return the expert_id with the lowest priority number (highest priority)
    return experts_sorted[0][0] if experts_sorted else config["experts"][0]["expert_id"]


@app.on_event("startup")
async def startup_event():
    """Initialize server"""
    global config

    print("="*80)
    print("  vLLM Multi-Model Inference Server with Intelligent Routing")
    print("  ARM Axion Optimized")
    print("="*80)
    print("")

    config_file = os.environ.get('VLLM_CONFIG_PATH', 'config.production.json')
    config = load_config()
    print(f"Loaded configuration from {config_file} with {len(config['experts'])} experts")
    
    # Initialize semantic router
    initialize_semantic_router()
    
    # Warm up first model if lazy loading is disabled
    if not config.get("lazy_loading", {}).get("enabled", False):
        warmup_size = config.get("lazy_loading", {}).get("warmup_pool_size", 1)
        for expert in config["experts"][:warmup_size]:
            try:
                load_model(expert["expert_id"])
            except Exception as e:
                print(f"Warning: Failed to load {expert['expert_id']}: {e}")

    print("")
    print("✓ Server ready with intelligent routing")
    print("")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "vLLM Multi-Model Server with Intelligent Routing",
        "version": "2.0.0",
        "models_loaded": len(loaded_models),
        "models_available": len(config.get("experts", [])),
        "routing_enabled": True
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
            "status": "loaded" if expert["expert_id"] in loaded_models else "available",
            "priority": expert.get("priority", 10)  # Lower is faster
        })

    return {
        "object": "list",
        "data": models_list
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completions endpoint (OpenAI compatible) with intelligent routing"""
    try:
        # Convert messages to prompt for routing analysis
        prompt = format_messages_to_prompt(request.messages)
        
        # Determine which model to use
        if request.model:
            # Use explicit model from request
            model_id = request.model
        else:
            # Use intelligent routing to select best model
            model_id = route_request_to_model(prompt)
        
        print(f"Routing request to model: {model_id}")
        
        # Get the model
        model = get_model(model_id)

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
            "model": model_id,
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
                "tokens_per_second": round(len(output.outputs[0].token_ids) / generation_time, 2),
                "routed_to": model_id
            }
        }

    except Exception as e:
        print(f"Error in chat_completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Text completions endpoint (OpenAI compatible) with intelligent routing"""
    try:
        # Determine which model to use
        if request.model:
            # Use explicit model from request
            model_id = request.model
        else:
            # Use intelligent routing to select best model
            model_id = route_request_to_model(request.prompt)
        
        print(f"Routing request to model: {model_id}")
        
        # Get the model
        model = get_model(model_id)

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
            "model": model_id,
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
                "tokens_per_second": round(len(output.outputs[0].token_ids) / generation_time, 2),
                "routed_to": model_id
            }
        }

    except Exception as e:
        print(f"Error in completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
async def ollama_generate(request: dict):
    """Ollama-compatible generate endpoint with intelligent routing"""
    try:
        model_name = request.get("model")
        prompt = request.get("prompt")

        # Determine which model to use
        if model_name:
            # Find model by name (try exact match first, then partial)
            model_id = None
            for expert in config["experts"]:
                if expert["expert_id"] == model_name or model_name in expert["expert_id"]:
                    model_id = expert["expert_id"]
                    break

            if not model_id:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        else:
            # Use intelligent routing to select best model
            model_id = route_request_to_model(prompt)
        
        print(f"Routing request to model: {model_id}")
        
        # Get the model
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
            "model": model_id,
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
        print(f"Error in ollama_generate: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "models_available": len(config.get("experts", [])),
        "routing_enabled": semantic_router is not None
    }


@app.get("/stats")
async def stats():
    """Statistics endpoint"""
    return {
        "models_loaded": list(loaded_models),
        "models_available": [e["expert_id"] for e in config.get("experts", [])],
        "config": {
            "lazy_loading": config.get("lazy_loading", {}),
            "server": config.get("server_config", {}),
            "routing_threshold": config.get("routing_threshold", 0.7),
            "use_fast_classifier": config.get("use_fast_classifier", True)
        },
        "routing_info": {
            "enabled": semantic_router is not None,
            "total_experts": len(config.get("experts", [])) if config else 0
        }
    }


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--config", default="config.production.json", help="Config file path")
    args = parser.parse_args()

    # Set the config file path as environment variable
    os.environ['VLLM_CONFIG_PATH'] = args.config

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )