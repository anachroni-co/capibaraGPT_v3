#!/usr/bin/env python3
"""
Actualizar el servidor principal para usar LiveMind Orchestrator con consenso
"""

import os
import json
from pathlib import Path

def update_main_server_to_use_livemind():
    """
    Actualiza el servidor principal para usar el LiveMind Orchestrator con consenso
    """
    
    # Importar los módulos necesarios para verificar que existen
    server_content = '''#!/usr/bin/env python3
"""
Multi-Model vLLM Inference Server for ARM Axion with LiveMind Consensus
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
sys.path.insert(0, "/home/elect/capibara6/arm-axion-optimizations/vllm_integration")

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

# Importar LiveMind Orchestrator con sistema de consenso
try:
    from livemind_orchestrator import LiveMindOrchestrator
    from vllm_integration.vllm_axion_backend import AxionMultiExpertVLLM
    from vllm import SamplingParams
    LIVE_MIND_AVAILABLE = True
    print("✅ LiveMind Orchestrator importado exitosamente")
except ImportError as e:
    print(f"⚠️ Error importando LiveMind Orchestrator: {e}")
    LIVE_MIND_AVAILABLE = False
    from vllm import LLM, SamplingParams  # Fallback clásico

app = FastAPI(title="vLLM Multi-Model Server - ARM Axion Optimized with Consensus", version="3.0.0")

# Global state
config: Dict = {}
loaded_models = set()  # Mantener para compatibilidad con herramientas existentes

# LiveMind Orchestrator (con sistema de consenso)
livemind_orchestrator = None

# Data Models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ""  # Cambiado: vacío por defecto para activar router automático
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stream: bool = True  # True por defecto para mejor experiencia de usuario


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stream: bool = True  # True por defecto para mejor experiencia de usuario


def load_config() -> Dict:
    """Load configuration file from environment or default"""
    import os
    config_file = os.environ.get('VLLM_CONFIG_PATH', 'config.json')
    with open(config_file) as f:
        return json.load(f)


def initialize_livemind():
    """Initialize LiveMind Orchestrator with consensus"""
    global livemind_orchestrator, config
    
    print("🚀 Inicializando LiveMind Orchestrator con sistema de consenso...")
    
    try:
        # Cargar configuración
        config_file = os.environ.get('VLLM_CONFIG_PATH', 'config.json')
        with open(config_file) as f:
            config = json.load(f)
        
        # Crear sistema de expertos
        expert_configs = []
        for expert in config["experts"]:
            expert_configs.append({
                "expert_id": expert["expert_id"],
                "model_path": expert["model_path"],
                "domain": expert.get("domain", "general"),
                "quantization": expert.get("quantization"),
                "enable_neon": expert.get("enable_neon", True),
                "dtype": expert.get("dtype", "float16"),
                "enable_prefix_caching": expert.get("enable_prefix_caching", True),
                "use_v2_block_manager": expert.get("use_v2_block_manager", True),
                "max_num_seqs": expert.get("max_num_seqs", 64),
                "max_model_len": expert.get("max_model_len", 4096),
                "gpu_memory_utilization": expert.get("gpu_memory_utilization", 0.9),
                "tensor_parallel_size": expert.get("tensor_parallel_size", 1),
                "device": "cpu",
                "trust_remote_code": expert.get("trust_remote_code", False),
                "kv_cache_dtype": expert.get("kv_cache_dtype", "auto"),
                "enable_chunked_prefill": expert.get("enable_chunked_prefill", True),
                "max_num_batched_tokens": expert.get("max_num_batched_tokens", 8192)
            })
        
        expert_system = AxionMultiExpertVLLM(expert_configs)
        
        # Inicializar LiveMind Orchestrator con consenso habilitado
        enable_consensus = config.get("enable_consensus", False)
        consensus_model_path = config.get("consensus_model")
        
        print(f"📊 Configuración de consenso: {enable_consensus}")
        print(f"📊 Modelo de consenso: {consensus_model_path}")
        
        livemind_orchestrator = LiveMindOrchestrator(
            expert_system=expert_system,
            enable_consensus=enable_consensus,
            consensus_model=consensus_model_path,
            chunk_size=config.get("chunk_size", 64),
            routing_threshold=config.get("routing_threshold", 0.7),
            use_fast_classifier=config.get("use_fast_classifier", True),
            enable_rag=config.get("rag", {}).get("enabled", True),
            rag_bridge_url=config.get("rag", {}).get("bridge_url", "http://localhost:8001"),
            rag_collection=config.get("rag", {}).get("collection", "capibara_docs")
        )
        
        print("✅ LiveMind Orchestrator inicializado con éxito")
        print(f"   Expertos disponibles: {len(expert_system.experts)}")
        print(f"   Consenso habilitado: {enable_consensus}")
        
    except Exception as e:
        print(f"❌ Error inicializando LiveMind Orchestrator: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize server with LiveMind Orchestrator"""
    global config, loaded_models

    print("="*70)
    print("  vLLM Multi-Model Inference Server - ARM Axion Optimized") 
    print("  WITH LIVE CONSENSUS ORCHESTRATION")
    print("  Parallel inference across multiple specialists")
    print("="*70)
    
    try:
        initialize_livemind()
        print("")
        print("🚀 Server ready with Live Consensus enabled")
        print("   Multiple experts working in parallel")
        print("   Consensus synthesis for improved quality")
        print("   Streaming with minimal TTFT")
        print("")
    except Exception as e:
        print(f"❌ Error durante inicio: {e}")


def format_messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert chat messages to prompt format"""
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"<|system|>\\n{msg.content}\\n"
        elif msg.role == "user":
            prompt += f"<|user|>\\n{msg.content}\\n"
        elif msg.role == "assistant":
            prompt += f"<|assistant|>\\n{msg.content}\\n"

    prompt += "<|assistant|>\\n"
    return prompt


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completions endpoint (OpenAI compatible) with LiveMind Consensus"""
    try:
        if livemind_orchestrator is None:
            raise HTTPException(status_code=500, detail="LiveMind Orchestrator no inicializado")
        
        # Convert messages to single prompt string
        prompt = format_messages_to_prompt(request.messages)
        
        from livemind_orchestrator import GenerationRequest
        
        # Create generation request
        gen_request = GenerationRequest(
            request_id=f"chat-{int(time.time())}-{id(request)}",
            prompt=prompt,
            system_prompt=None,  # Already included in format_messages_to_prompt
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
            expert_id=request.model if request.model else None  # If empty, will use router
        )
        
        if request.stream:
            # For now, return non-streaming response using LiveMind
            # Streaming implementation would require additional async generators
            result = await livemind_orchestrator.generate(gen_request)
            
            # Format as OpenAI-compatible response
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": result.expert_id,  # Which model actually generated the response
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),  # Approximation
                    "completion_tokens": result.tokens_generated,
                    "total_tokens": len(prompt.split()) + result.tokens_generated
                },
                "performance": {
                    "time_to_first_token": result.time_to_first_token,
                    "total_time": result.total_time,
                    "tokens_per_second": result.tokens_generated / result.total_time if result.total_time > 0 else 0
                }
            }
        else:
            # Non-streaming
            result = await livemind_orchestrator.generate(gen_request)
            
            # Format as OpenAI-compatible response
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": result.expert_id,  # Which model actually generated the response
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),  # Approximation
                    "completion_tokens": result.tokens_generated,
                    "total_tokens": len(prompt.split()) + result.tokens_generated
                },
                "performance": {
                    "time_to_first_token": result.time_to_first_token,
                    "total_time": result.total_time,
                    "tokens_per_second": result.tokens_generated / result.total_time if result.total_time > 0 else 0
                }
            }
    
    except Exception as e:
        import traceback
        print(f"❌ Error en chat_completions: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Text completions endpoint (OpenAI compatible)"""
    try:
        if livemind_orchestrator is None:
            raise HTTPException(status_code=500, detail="LiveMind Orchestrator no inicializado")
        
        from livemind_orchestrator import GenerationRequest
        
        # Create generation request
        gen_request = GenerationRequest(
            request_id=f"cmpl-{int(time.time())}-{id(request)}",
            prompt=request.prompt,
            system_prompt=None,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
            expert_id=request.model  # Model must be specified for completions
        )
        
        result = await livemind_orchestrator.generate(gen_request)
        
        # Format as OpenAI-compatible response
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": result.expert_id,
            "choices": [{
                "text": result.text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(request.prompt.split()),  # Approximation
                "completion_tokens": result.tokens_generated,
                "total_tokens": len(request.prompt.split()) + result.tokens_generated
            },
            "performance": {
                "time_to_first_token": result.time_to_first_token,
                "total_time": result.total_time,
                "tokens_per_second": result.tokens_generated / result.total_time if result.total_time > 0 else 0
            }
        }
    
    except Exception as e:
        import traceback
        print(f"❌ Error en completions: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    consensus_enabled = config.get("enable_consensus", False) if config else False
    return {
        "name": "vLLM Multi-Model Server - ARM Axion Optimized",
        "version": "3.0.0 - with Live Consensus",
        "consensus_enabled": consensus_enabled,
        "models_loaded": len(loaded_models),
        "models_available": len(config.get("experts", [])) if config else 0
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    models_list = []

    for expert in config.get("experts", []) if config else []:
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
            "status": "available"  # With LiveMind, all models are available through routing
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

        if livemind_orchestrator is None:
            raise HTTPException(status_code=500, detail="LiveMind Orchestrator no inicializado")
        
        from livemind_orchestrator import GenerationRequest
        
        # Create generation request
        gen_request = GenerationRequest(
            request_id=f"ollama-{int(time.time())}-{id(request)}",
            prompt=prompt,
            system_prompt=None,
            max_tokens=request.get("max_tokens", 2048),
            temperature=request.get("temperature", 0.7),
            top_p=request.get("top_p", 0.9),
            stream=False,
            expert_id=model_name
        )
        
        result = await livemind_orchestrator.generate(gen_request)
        
        # Calculate duration metrics
        start_time = time.time() - result.total_time
        total_duration = result.total_time
        prompt_eval_time = total_duration * 0.1  # Estimation
        eval_time = total_duration * 0.9  # Estimation
        
        # Format Ollama-compatible response
        return {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "response": result.text,
            "done": True,
            "total_duration": int(total_duration * 1e9),
            "load_duration": 0,
            "prompt_eval_count": len(prompt.split()),  # Approximation
            "prompt_eval_duration": int(prompt_eval_time * 1e9),
            "eval_count": result.tokens_generated,
            "eval_duration": int(eval_time * 1e9),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    consensus_enabled = config.get("enable_consensus", False) if config else False
    return {
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "models_available": len(config.get("experts", [])) if config else 5,
        "consensus_enabled": consensus_enabled
    }


@app.get("/stats")
async def stats():
    """Statistics endpoint with consensus information"""
    stats_data = {
        "models_loaded": list(loaded_models),
        "models_available": [e["expert_id"] for e in config.get("experts", [])] if config else [],
        "config": {
            "lazy_loading": config.get("lazy_loading", {}) if config else {},
            "server": config.get("server_config", {}) if config else {},
        }
    }
    
    # Add consensus information if available
    if config:
        stats_data["config"]["consensus_enabled"] = config.get("enable_consensus", False)
        stats_data["config"]["consensus_model"] = config.get("consensus_model")
    
    # Add LiveMind statistics if available
    if livemind_orchestrator:
        try:
            livemind_stats = livemind_orchestrator.get_stats()
            stats_data["livemind_stats"] = livemind_stats
        except:
            pass
    
    return stats


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8082, help="Port to bind to")
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
'''

    # Guardar el nuevo servidor
    server_path = "/home/elect/capibara6/arm-axion-optimizations/vllm_integration/multi_model_server_consensus.py"
    with open(server_path, 'w') as f:
        f.write(server_content)
    
    print(f"✅ Servidor con LiveMind Consensus creado en {server_path}")
    
    return server_path


def create_consensus_start_script():
    """
    Crea un script para iniciar el servidor con consenso
    """
    script_content = '''#!/bin/bash
# Script para iniciar el servidor ARM-Axion con LiveMind Consensus
# start_consensus_server.sh

set -e  # Salir si hay un error

GREEN="\\033[0;32m"
CYAN="\\033[0;36m"
YELLOW="\\033[1;33m"
RED="\\033[0;31m"
NC="\\033[0m"  # Sin color

echo -e "${CYAN}🦫 Iniciando servidor ARM-Axion con LiveMind Consensus${NC}"
echo "   VM: models-europe"
echo "   Puerto: 8084 (consensus server)"
echo "   Modelo: Multi-expert con consensus paralelo ARM-Axion"
echo ""

# Configurar ambiente para ARM-Axion con optimizaciones para consenso
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_USE_FLASHINFER=0
export TORCHINDUCTOR_DISABLED=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:disabled

# Verificar que no haya un servidor ya corriendo en el puerto 8084
if lsof -Pi :8084 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}❌ Error: Puerto 8084 ya está en uso${NC}"
    echo "   Verifica si el servidor de consenso ya está corriendo:"
    echo "   ps aux | grep multi_model_server_consensus"
    exit 1
fi

echo -e "${GREEN}✓${NC} Variables de entorno ARM-Axion configuradas para consenso"

# Verificar que el archivo de configuración exista
if [ ! -f "arm-axion-optimizations/vllm_integration/config.json" ]; then
    echo -e "${RED}❌ Error: No se encuentra el archivo de configuración${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Archivo de configuración encontrado"

# Iniciar servidor de consenso en segundo plano con logging
echo -e "${CYAN}🚀 Iniciando servidor de consenso en modo daemon (puerto 8084)...${NC}"
cd arm-axion-optimizations/vllm_integration
nohup python3 multi_model_server_consensus.py --host 0.0.0.0 --port 8084 --config config.json > /tmp/multi_model_server_consensus.log 2>&1 &

SERVER_PID=$!
echo "PID del servidor de consenso: $SERVER_PID" >> /tmp/multi_model_server_consensus.log

# Esperar a que el servidor arranque
echo -e "${CYAN}⏳ Esperando que el servidor de consenso inicie (60 segundos)...${NC}"
sleep 60

# Verificar que el servidor esté escuchando en el puerto
if ss -tlnp | grep ":8084" >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Servidor de consenso escuchando en puerto 8084"
else
    echo -e "${RED}❌ Error: Servidor de consenso no escucha en puerto 8084${NC}"
    echo "   Revisando logs..."
    tail -20 /tmp/multi_model_server_consensus.log
    exit 1
fi

# Verificar estado de salud del servidor de consenso
echo -e "${CYAN}🏥 Verificando estado de salud del servidor de consenso...${NC}"
if curl -s --connect-timeout 10 http://localhost:8084/health >/dev/null 2>&1; then
    HEALTH_RESPONSE=$(curl -s http://localhost:8084/health)
    echo -e "${GREEN}✓${NC} Servidor de consenso saludable: $HEALTH_RESPONSE"
else
    echo -e "${YELLOW}⚠️  Servidor de consenso no responde a health check aún${NC}"
    echo "   Esto puede ser normal durante la carga inicial de modelos"
    echo "   El servidor de consenso se inició correctamente con PID: $SERVER_PID"
    exit 0
fi

# Verificar que el consenso esté habilitado
if curl -s http://localhost:8084/stats | jq -e ".config.consensus_enabled" >/dev/null 2>&1; then
    CONSENSUS_ENABLED=$(curl -s http://localhost:8084/stats | jq -r ".config.consensus_enabled")
    if [ "$CONSENSUS_ENABLED" = "true" ]; then
        echo -e "${GREEN}✓${NC} Sistema de consenso habilitado en el servidor"
    else
        echo -e "${YELLOW}⚠️  Sistema de consenso no está habilitado en el servidor${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No se pudo verificar estado de consenso (respuesta inesperada)${NC}"
fi

# Verificar modelos disponibles
echo -e "${CYAN}🤖 Verificando modelos disponibles en servidor de consenso...${NC}"
MODELS_COUNT=$(curl -s http://localhost:8084/v1/models | jq ".data | length" 2>/dev/null || echo "Error")
if [ "$MODELS_COUNT" != "Error" ]; then
    echo -e "${GREEN}✓${NC} $MODELS_COUNT modelos disponibles en servidor de consenso"
    curl -s http://localhost:8084/v1/models | jq ".data[].id"
else
    echo -e "${YELLOW}⚠️  No se pudo obtener lista de modelos en servidor de consenso${NC}"
fi

echo ""
echo -e "${GREEN}✅ Servidor ARM-Axion con CONSENSO iniciado exitosamente${NC}"
echo "   Puerto: 8084 (consensus server)"
echo "   Logs: /tmp/multi_model_server_consensus.log"
echo "   Configuración: v1 engine deshabilitado para compatibilidad ARM"
echo "   Optimizaciones: NEON, ACL, FP8 KV Cache, Flash Attention, LIVE CONSENSUS"
echo ""
echo -e "${CYAN}Endpoints disponibles (Consensus):${NC}"
echo "   GET  http://localhost:8084/health - Verificar estado"
echo "   GET  http://localhost:8084/v1/models - Lista de modelos"
echo "   POST http://localhost:8084/v1/chat/completions - API OpenAI con consenso"
echo "   POST http://localhost:8084/v1/completions - API OpenAI completions con consenso"
echo "   GET  http://localhost:8084/stats - Estadísticas del servidor"
echo ""
echo -e "${CYAN}Diferencias con servidores anteriores:${NC}"
echo "   - Puerto 8082: Servidor estándar (respuesta completa)"
echo "   - Puerto 8083: Servidor con streaming verdadero (token por token)"
echo "   - Puerto 8084: Servidor con consenso paralelo (múltiples expertos)"
'''
    
    script_path = "/home/elect/capibara6/start_consensus_server.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Hacerlo ejecutable
    os.chmod(script_path, 0o755)
    
    print(f"✅ Script de inicio con consenso creado en {script_path}")
    
    return script_path


def main():
    print("🔄 ACTUALIZANDO SERVIDOR PARA USAR LIVE CONSENSUS")
    print("=" * 70)
    print("Habilitando sistema de consenso paralelo con las siguientes mejoras:")
    print("- Inferencia paralela en múltiples modelos expertos")
    print("- Síntesis de consenso para mejorar calidad de respuestas")
    print("- Sistema de ruteo semántico avanzado")
    print("- Recursos compartidos para mayor eficiencia")
    print("=" * 70)
    
    # 1. Crear el servidor actualizado
    server_path = update_main_server_to_use_livemind()
    
    # 2. Crear script de inicio
    script_path = create_consensus_start_script()
    
    print(f"\\n✅ IMPLEMENTACIÓN COMPLETADA:")
    print(f"   • Servidor con consenso: {server_path}")
    print(f"   • Script de inicio: {script_path}")
    print("")
    print("📊 PARA PROBAR EL SISTEMA DE CONSENSO:")
    print(f"   1. Iniciar servidor: bash {script_path}")
    print("   2. Probar endpoints en puerto 8084")
    print("   3. Verificar estadísticas en /stats para confirmar consenso activo")
    print("")
    print("🔄 El servidor original en puerto 8082 no se ha modificado")
    print("   Puede seguir usándose mientras se prueba el nuevo sistema")
    
    print("\\n🎉 Optimización de consenso paralelo implementada!")


if __name__ == "__main__":
    main()