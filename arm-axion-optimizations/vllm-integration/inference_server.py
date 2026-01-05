"""
vLLM + ARM Axion Inference Server
OpenAI-compatible API with LiveMind optimizations

Features:
- OpenAI-compatible endpoints (/v1/completions, /v1/chat/completions)
- Multi-expert routing with NEON acceleration
- Streaming support
- Health and metrics endpoints
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
import time
import json
import uuid

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from vllm_integration.vllm_axion_backend import (
    AxionMultiExpertVLLM,
    AxionVLLMConfig
)
from vllm_integration.livemind_orchestrator import (
    LiveMindOrchestrator,
    GenerationRequest
)

# FastAPI app
app = FastAPI(
    title="vLLM ARM Axion Inference Server",
    description="Multi-expert inference with NEON optimizations",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
orchestrator: Optional[LiveMindOrchestrator] = None
expert_system: Optional[AxionMultiExpertVLLM] = None


# Pydantic models for API
class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request"""
    prompt: str
    model: str = "default"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    n: int = 1
    stream: bool = False
    stop: Optional[List[str]] = None


class ChatMessage(BaseModel):
    """Chat message"""
    role: str  # system, user, assistant
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    messages: List[ChatMessage]
    model: str = "default"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


class CompletionChoice(BaseModel):
    """Completion choice"""
    text: str
    index: int
    finish_reason: str
    logprobs: Optional[Any] = None


class CompletionResponse(BaseModel):
    """Completion response"""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]


class ChatCompletionChoice(BaseModel):
    """Chat completion choice"""
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """Chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


# Startup/shutdown
@app.on_event("startup")
async def startup():
    """Initialize system on startup"""
    global orchestrator, expert_system

    print("🚀 Starting vLLM ARM Axion Inference Server...")

    # Load config from environment or default
    config_path = Path(__file__).parent / "config.json"

    if config_path.exists():
        print(f"📝 Loading config from {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print("⚠️  No config.json found, using default (single expert)")
        config = {
            "experts": [
                {
                    "expert_id": "default",
                    "model_path": "facebook/opt-125m",
                    "domain": "general",
                    "quantization": None,
                    "enable_neon": True
                }
            ],
            "enable_consensus": False,
            "chunk_size": 64,
            "routing_threshold": 0.7
        }

    # Initialize expert system with lazy loading
    print(f"🔧 Initializing {len(config['experts'])} experts...")

    lazy_config = config.get('lazy_loading', {})

    expert_system = AxionMultiExpertVLLM(
        expert_configs=config['experts'],
        use_lazy_loading=lazy_config.get('enabled', True),
        warmup_pool_size=lazy_config.get('warmup_pool_size', 2),
        max_loaded_experts=lazy_config.get('max_loaded_experts', 3),
        memory_threshold=lazy_config.get('memory_threshold', 0.80)
    )

    # Initialize orchestrator
    print("🔧 Initializing LiveMind orchestrator...")

    # Get RAG configuration
    rag_config = config.get('rag', {})

    orchestrator = LiveMindOrchestrator(
        expert_system=expert_system,
        enable_consensus=config.get('enable_consensus', False),
        chunk_size=config.get('chunk_size', 64),
        routing_threshold=config.get('routing_threshold', 0.7),
        enable_rag=rag_config.get('enabled', True),
        rag_bridge_url=rag_config.get('bridge_url', 'http://localhost:8001'),
        rag_collection=rag_config.get('collection', 'capibara_docs')
    )

    print("✅ Server ready!")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    print("👋 Shutting down...")


# Health endpoint
@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "orchestrator_ready": orchestrator is not None,
        "experts_ready": expert_system is not None
    }


# Stats endpoint
@app.get("/stats")
async def stats():
    """Get server statistics"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    orchestrator_stats = orchestrator.get_stats()

    # Add lazy loading stats if enabled
    if expert_system and expert_system.use_lazy_loading:
        orchestrator_stats['lazy_loading'] = expert_system.get_manager_stats()

    return orchestrator_stats


# Experts endpoint
@app.get("/experts")
async def list_experts():
    """List available experts"""
    if not expert_system:
        raise HTTPException(status_code=503, detail="Expert system not initialized")

    return {
        "experts": expert_system.list_experts()
    }


# OpenAI-compatible completions endpoint
@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    OpenAI-compatible completions endpoint

    POST /v1/completions
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    # Create generation request
    gen_request = GenerationRequest(
        request_id=str(uuid.uuid4()),
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream
    )

    if request.stream:
        # Streaming response
        async def generate_stream():
            async for token in orchestrator.generate_streaming(gen_request):
                chunk = {
                    "id": gen_request.request_id,
                    "object": "text_completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "text": token,
                            "index": 0,
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Final chunk
            final_chunk = {
                "id": gen_request.request_id,
                "object": "text_completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "text": "",
                        "index": 0,
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )

    else:
        # Non-streaming response
        result = await orchestrator.generate(gen_request)

        response = CompletionResponse(
            id=gen_request.request_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    text=result.text,
                    index=0,
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": len(request.prompt.split()),  # Approximate
                "completion_tokens": result.tokens_generated,
                "total_tokens": len(request.prompt.split()) + result.tokens_generated
            }
        )

        return response


# OpenAI-compatible chat completions endpoint
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint

    POST /v1/chat/completions
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    # Convert chat messages to prompt
    prompt_parts = []
    system_prompt = None

    for message in request.messages:
        if message.role == "system":
            system_prompt = message.content
        elif message.role == "user":
            prompt_parts.append(f"User: {message.content}")
        elif message.role == "assistant":
            prompt_parts.append(f"Assistant: {message.content}")

    prompt = "\n".join(prompt_parts) + "\nAssistant:"

    # Create generation request
    gen_request = GenerationRequest(
        request_id=str(uuid.uuid4()),
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream
    )

    if request.stream:
        # Streaming response
        async def generate_stream():
            async for token in orchestrator.generate_streaming(gen_request):
                chunk = {
                    "id": gen_request.request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "delta": {"content": token},
                            "index": 0,
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Final chunk
            final_chunk = {
                "id": gen_request.request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "delta": {},
                        "index": 0,
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )

    else:
        # Non-streaming response
        result = await orchestrator.generate(gen_request)

        response = ChatCompletionResponse(
            id=gen_request.request_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=result.text
                    ),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": result.tokens_generated,
                "total_tokens": len(prompt.split()) + result.tokens_generated
            }
        )

        return response


# Main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="vLLM ARM Axion Inference Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print(f"""
    ╔════════════════════════════════════════════════════════╗
    ║  vLLM ARM Axion Inference Server                       ║
    ║  Optimized for Google Cloud ARM Axion                  ║
    ╚════════════════════════════════════════════════════════╝

    🌐 Server will start on: http://{args.host}:{args.port}

    Endpoints:
      • Health:       GET  /health
      • Stats:        GET  /stats
      • Experts:      GET  /experts
      • Completions:  POST /v1/completions
      • Chat:         POST /v1/chat/completions

    📚 OpenAI-compatible API
    """)

    uvicorn.run(
        "inference_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
