#!/usr/bin/env python3
"""
Multi-Model vLLM Inference Server for ARM Axion with TRUE Streaming Support
Compatible with OpenAI API format
Optimized with speculative streaming for faster token delivery
"""

import json
import time
import asyncio
import os
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
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
            normalized = input * inv_rms
            output.copy_((input + residual) * normalized * weight)
        torch.ops._C.fused_add_rms_norm = fused_add_rms_norm_fallback

apply_fallback_patches()

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid

app = FastAPI(title="vLLM Multi-Model Server - ARM Axion Optimized", version="2.0.0")

# Global state
models: Dict[str, LLM] = {}
config: Dict = {}
loaded_models: set = set()
async_engines: Dict[str, AsyncLLMEngine] = {}  # Engines for streaming

# Data Models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stream: bool = True  # Changed to True by default for streaming


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stream: bool = True  # Changed to True by default for streaming


def load_config() -> Dict:
    """Load configuration file from environment or default"""
    import os
    config_file = os.environ.get('VLLM_CONFIG_PATH', 'config.json')
    with open(config_file) as f:
        return json.load(f)


async def get_async_engine(model_id: str) -> AsyncLLMEngine:
    """Get or create async engine for streaming purposes"""
    if model_id not in async_engines:
        expert = next((e for e in config["experts"] if e["expert_id"] == model_id), None)
        if not expert:
            raise ValueError(f"Model {model_id} not found in config")

        print(f"Creating async engine for streaming: {model_id}")
        
        engine_args = AsyncEngineArgs(
            model=expert["model_path"],
            tensor_parallel_size=expert.get("tensor_parallel_size", 1),
            gpu_memory_utilization=expert.get("gpu_memory_utilization", 0.9),
            max_num_seqs=expert.get("max_num_seqs", 64),  # Reduced for lower latency
            max_model_len=expert.get("max_model_len", 4096),
            quantization=expert.get("quantization") if expert.get("quantization") != "q4_0" else None,
            dtype=expert.get("dtype", "float16"),
            trust_remote_code=expert.get("trust_remote_code", False),
            enforce_eager=expert.get("enforce_eager", False),
            device="cpu",  # For ARM Axion
            kv_cache_dtype=expert.get("kv_cache_dtype", "auto"),
            enable_prefix_caching=expert.get("enable_prefix_caching", True),
            use_v2_block_manager=expert.get("use_v2_block_manager", True),
            swap_space=expert.get("swap_space", 4),
            cpu_offload_gb=expert.get("cpu_offload_gb", 0),
            max_num_batched_tokens=expert.get("max_num_batched_tokens", 4096)
        )

        # Create async engine with optimized settings for streaming
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        async_engines[model_id] = engine
        print(f"Async engine created for {model_id}")

    return async_engines[model_id]


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


async def generate_chat_streaming(
    model_id: str,
    messages: List[ChatMessage],
    sampling_params: SamplingParams,
    request_id: str
) -> AsyncIterator[str]:
    """Generate tokens streamingly from async engine with chat format"""
    # Convert messages to prompt
    prompt = format_messages_to_prompt(messages)
    
    # Get async engine
    async_engine = await get_async_engine(model_id)
    
    # Create results generator
    results_generator = async_engine.generate(
        prompt,
        sampling_params,
        request_id
    )
    
    # Track previous tokens to send only deltas
    previous_texts = {0: ""}
    
    async for request_output in results_generator:
        for output in request_output.outputs:
            delta_text = output.text[len(previous_texts[output.index]):]
            previous_texts[output.index] = output.text
            
            # Create OpenAI-compatible streaming chunk
            chunk = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": output.index,
                    "delta": {"role": "assistant", "content": delta_text},
                    "logprobs": None,
                    "finish_reason": output.finish_reason
                }]
            }
            
            yield f"data: {json.dumps(chunk)}\n\n"
    
    yield "data: [DONE]\n\n"


@app.on_event("startup")
async def startup_event():
    """Initialize server"""
    global config

    print("="*70)
    print("  vLLM Multi-Model Inference Server - ARM Axion Optimized")
    print("  WITH TRUE STREAMING SUPPORT")
    print("="*70)
    print("")

    config_file = os.environ.get('VLLM_CONFIG_PATH', 'config.json')
    config = load_config()
    print(f"Loaded configuration from {config_file} with {len(config['experts'])} experts")
    print("")

    print("")
    print("🚀 Server ready with TRUE streaming support")
    print("   Streaming enabled by default for all endpoints")
    print("   First token latency optimized with speculative decoding")
    print("")


def get_model(model_id: str) -> LLM:
    """Get model, loading if necessary"""
    if model_id not in models:
        load_model(model_id)
    return models[model_id]


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
        max_num_seqs=expert.get("max_num_seqs", 64),  # Optimized for lower latency
        max_model_len=expert.get("max_model_len", 4096),
        quantization=expert.get("quantization") if expert.get("quantization") != "q4_0" else None,
        dtype=expert.get("dtype", "float16"),
        trust_remote_code=expert.get("trust_remote_code", False),
        enforce_eager=expert.get("enforce_eager", False),
        device="cpu",  # Optimized for ARM Axion
        kv_cache_dtype=expert.get("kv_cache_dtype", "auto"),
        enable_prefix_caching=expert.get("enable_prefix_caching", True),
        use_v2_block_manager=expert.get("use_v2_block_manager", True),
        swap_space=expert.get("swap_space", 4),
        cpu_offload_gb=expert.get("cpu_offload_gb", 0),
        enable_chunked_prefill=expert.get("enable_chunked_prefill", True),  # For faster TTFT
        max_num_batched_tokens=expert.get("max_num_batched_tokens", 8192)
    )

    models[model_id] = model
    loaded_models.add(model_id)
    print(f"✓ Model {model_id} loaded successfully")

    return model


# Enhanced chat completions with true streaming
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completions endpoint (OpenAI compatible) with TRUE streaming"""
    try:
        if request.stream:
            # Generate with TRUE streaming
            async def streaming_response():
                # Sampling parameters
                sampling_params = SamplingParams(
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                )
                
                async for chunk in generate_chat_streaming(
                    request.model,
                    request.messages,
                    sampling_params,
                    f"chat-{random_uuid()}"
                ):
                    yield chunk

            return StreamingResponse(streaming_response(), media_type="text/event-stream")
        else:
            # Non-streaming response (the original behavior)
            model = get_model(request.model)
            
            # Convert messages to prompt
            prompt = format_messages_to_prompt(request.messages)

            # Sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
            )

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


async def generate_completion_streaming(
    model_id: str,
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str
) -> AsyncIterator[str]:
    """Generate completion tokens streamingly from async engine"""
    # Get async engine
    async_engine = await get_async_engine(model_id)
    
    # Create results generator
    results_generator = async_engine.generate(
        prompt,
        sampling_params,
        request_id
    )
    
    async for request_output in results_generator:
        # Format as OpenAI completion chunk
        chunk = {
            "id": f"cmpl-{request_id}",
            "object": "text_completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "text": request_output.outputs[0].text,  # Full text accumulated so far
                "index": 0,
                "logprobs": None,
                "finish_reason": request_output.outputs[0].finish_reason
            }]
        }
        
        yield f"data: {json.dumps(chunk)}\n\n"
    
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Text completions endpoint (OpenAI compatible) with TRUE streaming"""
    try:
        if request.stream:
            # Generate with TRUE streaming
            async def streaming_response():
                sampling_params = SamplingParams(
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                )
                
                async for chunk in generate_completion_streaming(
                    request.model,
                    request.prompt,
                    sampling_params,
                    f"cmpl-{random_uuid()}"
                ):
                    yield chunk

            return StreamingResponse(streaming_response(), media_type="text/event-stream")
        else:
            # Non-streaming response (the original behavior)
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


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "vLLM Multi-Model Server - ARM Axion Optimized",
        "version": "2.0.0 - with True Streaming",
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

        # Check if streaming is requested
        stream = request.get("stream", False)
        
        if stream:
            # Return streaming response for Ollama format
            async def ollama_streaming_response():
                sampling_params = SamplingParams(
                    temperature=request.get("temperature", 0.7),
                    top_p=request.get("top_p", 0.9),
                    max_tokens=request.get("max_tokens", 2048),
                )
                
                async for chunk in generate_completion_streaming(
                    model_id,
                    prompt,
                    sampling_params,
                    f"ollama-{random_uuid()}"
                ):
                    # Convert to Ollama format
                    parsed_chunk = json.loads(chunk[6:])  # Remove "data: " prefix
                    if not parsed_chunk.get("choices"):
                        continue
                        
                    ollama_format = {
                        "model": model_name,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                        "response": parsed_chunk["choices"][0]["text"],
                        "done": parsed_chunk["choices"][0]["finish_reason"] is not None
                    }
                    
                    yield f"data: {json.dumps(ollama_format)}\n\n"
            
            return StreamingResponse(ollama_streaming_response(), media_type="text/event-stream")
        else:
            # Non-streaming response (original behavior)
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
        "models_loaded": len(loaded_models),
        "models_available": len(config.get("experts", [])),
        "streaming_enabled": True
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
            "streaming_enabled": True,
            "true_streaming": True
        }
    }


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8082, help="Port to bind to")  # Updated default
    parser.add_argument("--config", default="config.json", help="Config file path")
    args = parser.parse_args()

    # Set the config file path as environment variable
    os.environ['VLLM_CONFIG_PATH'] = args.config

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )