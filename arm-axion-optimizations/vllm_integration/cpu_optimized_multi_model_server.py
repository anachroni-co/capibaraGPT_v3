#!/usr/bin/env python3
"""
Multi-Model vLLM Inference Server for ARM Axion (CPU Optimized)
Compatible with OpenAI API format
"""

import json
import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

app = FastAPI(title="vLLM Multi-Model Server", version="1.0.0")

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


def load_config(config_path: str = "config.five_models.optimized.json") -> Dict:
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
    """Load a model into memory - CPU optimized version"""
    global config, models, loaded_models

    expert_entry = None

    # Check if config uses "experts" list format
    if "experts" in config:
        for expert in config["experts"]:
            if expert["expert_id"] == model_id:
                expert_entry = expert
                break
    # Fallback to "models" dict format if "experts" not found
    elif "models" in config:
        if model_id in config["models"]:
            expert_entry = {"expert_id": model_id, **config["models"][model_id]}
        else:
            # Also check by name field
            for key, value in config["models"].items():
                if value.get("name") == model_id or key == model_id:
                    expert_entry = {"expert_id": key, **value}
                    break

    if not expert_entry:
        raise ValueError(f"Model {model_id} not found in config")

    # Create a copy of the expert config without problematic parameters
    expert = {k: v for k, v in expert_entry.items() if k not in ['device', 'tensor_parallel_size']}

    print(f"Loading model: {model_id} from {expert.get('model_path', expert.get('name', 'N/A'))}")

    # vLLM initialization params - CPU optimized
    # Only explicitly set known compatible parameters to avoid device-related errors
    model_params = {
        "model": expert.get("model_path", expert.get("name")),
        "dtype": expert.get("dtype", "float16"),
        "trust_remote_code": expert.get("trust_remote_code", True),
        "enforce_eager": expert.get("enforce_eager", False),
    }

    # Only safely add gpu_memory_utilization if it's specifically configured in expert
    # This parameter is only valid when using GPU
    if expert.get("gpu_memory_utilization"):
        model_params["gpu_memory_utilization"] = expert["gpu_memory_utilization"]

    # Only safely add quantization if specified
    quantization = expert.get("quantization")
    if quantization and quantization != "null" and quantization != "":
        model_params["quantization"] = quantization

    # Set environment for CPU-only operation
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA devices

    # Ensure we're not passing any device-related parameters
    # Since vLLM for CPU has been specially compiled, we'll use just the essential parameters
    model = LLM(**model_params)

    models[expert["expert_id"]] = model
    loaded_models.add(expert["expert_id"])
    print(f"✓ Model {expert['expert_id']} loaded successfully")

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
    print("  ARM Axion CPU Optimized")
    print("="*70)
    print("")

    # Load config from the correct path - try both possible structures
    try:
        config = load_config()

        # Determine number of models/experts based on config structure
        if "experts" in config:
            num_models = len(config["experts"])
        elif "models" in config:
            num_models = len(config["models"])
        else:
            num_models = 0

        print(f"Loaded configuration with {num_models} experts/models")
        print("Configuration keys:", list(config.keys()))
    except Exception as e:
        print(f"Error loading config: {e}")
        # Load with default structure
        with open("config.five_models.optimized.json") as f:
            config = json.load(f)

    print("")

    # Warm up first model if lazy loading is disabled
    # Disabled initially to focus on getting server running
    print("")
    print("✓ Server ready")
    print("")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "vLLM Multi-Model Server",
        "version": "1.0.0",
        "models_loaded": len(loaded_models),
        "models_available": len(config.get("models", config.get("experts", [])))
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    models_list = []

    # Handle both configuration formats
    if "experts" in config:
        # Use the experts list format
        for expert in config["experts"]:
            expert_id = expert.get("expert_id", expert.get("name", expert.get("model_path", "unknown")))
            models_list.append({
                "id": expert_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "capibara6",
                "permission": [],
                "root": expert_id,
                "parent": None,
                "description": expert.get("description", expert.get("name", "")),
                "domain": expert.get("domain", expert.get("use_case", ["general"])[0] if expert.get("use_case") else "general"),
                "status": "loaded" if expert_id in loaded_models else "available"
            })
    elif "models" in config:
        # Use the models dict format
        for key, model_data in config["models"].items():
            if isinstance(model_data, dict):
                models_list.append({
                    "id": key,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "capibara6",
                    "permission": [],
                    "root": key,
                    "parent": None,
                    "description": model_data.get("description", ""),
                    "domain": model_data.get("domain", ""),
                    "status": "loaded" if key in loaded_models else "available"
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
        print(f"Error in chat completions: {e}")
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
        print(f"Error in completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
async def ollama_generate(request: dict):
    """Ollama-compatible generate endpoint"""
    try:
        model_name = request.get("model")
        prompt = request.get("prompt")

        # Find model by name
        model_id = model_name
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
        print(f"Error in Ollama generate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "models_available": len(config.get("models", config.get("experts", [])))
    }


@app.get("/stats")
async def stats():
    """Statistics endpoint"""
    # Determine the structure of the config to extract model IDs correctly
    models_available = []
    if "experts" in config:
        # Use the experts list format
        for expert in config["experts"]:
            models_available.append(expert.get("expert_id") or expert.get("name") or expert.get("model_path", "unknown"))
    elif "models" in config:
        # Use the models dict format
        for key in config["models"].keys():
            models_available.append(key)

    # Get the actual server config, potentially override with command-line args
    server_config = config.get("server_config", {})

    return {
        "models_loaded": list(loaded_models),
        "models_available": models_available,
        "config": {
            "lazy_loading": config.get("lazy_loading", {}),
            "server": server_config
        }
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", default="config.five_models.optimized.json", help="Config file path")
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )