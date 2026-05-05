"""
HuggingFace Consensus Strategy - CapibaraGPT v3

Consensus strategy using HuggingFace expert models for multi-domain responses.
More cost-effective than large commercial models (Gemini, DeepSeek, Mixtral).
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer, pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None  # type: ignore[assignment,misc]
    hf_pipeline = None  # type: ignore[assignment]

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class ExpertModelConfig:
    """Configuration for an expert model."""
    name: str
    model_id: str
    domain: str
    max_length: int = 512
    temperature: float = 0.7
    weight: float = 1.0
    use_local: bool = True
    api_endpoint: Optional[str] = None


@dataclass
class HuggingFaceConsensusConfig:
    """Full configuration for HuggingFace consensus strategy."""

    expert_models: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "math": {
            "name": "Math Expert",
            "model_id": "microsoft/DialoGPT-medium",
            "domain": "mathematics",
            "temperature": 0.3,
            "weight": 1.2,
            "max_length": 512,
            "use_local": True
        },
        "code": {
            "name": "Code Expert",
            "model_id": "Salesforce/codet5-small",
            "domain": "programming",
            "temperature": 0.4,
            "weight": 1.3,
            "max_length": 512,
            "use_local": True
        },
        "spanish": {
            "name": "Spanish Expert",
            "model_id": "dccuchile/bert-base-spanish-wwm-cased",
            "domain": "spanish_language",
            "temperature": 0.6,
            "weight": 1.5,
            "max_length": 512,
            "use_local": True
        },
        "reasoning": {
            "name": "Reasoning Expert",
            "model_id": "EleutherAI/gpt-neo-125M",
            "domain": "logical_reasoning",
            "temperature": 0.5,
            "weight": 1.1,
            "max_length": 512,
            "use_local": True
        },
        "medical": {
            "name": "Medical Expert",
            "model_id": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            "domain": "medical",
            "temperature": 0.2,
            "weight": 1.4,
            "max_length": 512,
            "use_local": True
        },
        "legal": {
            "name": "Legal Expert",
            "model_id": "nlpaueb/legal-bert-base-uncased",
            "domain": "legal",
            "temperature": 0.3,
            "weight": 1.2,
            "max_length": 512,
            "use_local": True
        },
        "general": {
            "name": "General Expert",
            "model_id": "microsoft/DialoGPT-medium",
            "domain": "general",
            "temperature": 0.7,
            "weight": 1.0,
            "max_length": 512,
            "use_local": True
        }
    })

    consensus_settings: Dict[str, Any] = field(default_factory=lambda: {
        "max_responses": 5,
        "min_confidence": 0.6,
        "consensus_methods": ["weighted_voting", "semantic_similarity", "majority_voting"],
        "similarity_threshold": 0.7,
        "quality_weights": {"length": 0.3, "relevance": 0.4, "coherence": 0.2, "domain_alignment": 0.1}
    })

    performance_settings: Dict[str, Any] = field(default_factory=lambda: {
        "max_concurrent_requests": 5,
        "timeout_seconds": 30,
        "retry_attempts": 3,
        "batch_size": 1,
        "use_cache": True,
        "cache_ttl": 3600
    })

    domain_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "mathematics": ["equation", "calculus", "number", "algebra", "geometry", "derivative", "integral"],
        "programming": ["code", "function", "variable", "algorithm", "class", "method", "loop"],
        "spanish_language": ["spanish", "grammar", "vocabulary", "conjugation", "syntax"],
        "medical": ["medical", "patient", "treatment", "diagnosis", "symptom", "disease"],
        "legal": ["law", "legal", "contract", "right", "court", "regulation"],
        "logical_reasoning": ["logic", "reasoning", "premise", "conclusion", "inference"]
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_models": self.expert_models,
            "consensus_settings": self.consensus_settings,
            "performance_settings": self.performance_settings,
            "domain_keywords": self.domain_keywords
        }

    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'HuggingFaceConsensusConfig':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        config = cls()
        for key in ['expert_models', 'consensus_settings', 'performance_settings', 'domain_keywords']:
            if key in data:
                setattr(config, key, data[key])
        return config


# =============================================================================
# Main Strategy Class
# =============================================================================

class HuggingFaceConsensusStrategy:
    """Consensus strategy using HuggingFace expert models."""

    def __init__(self, config: Optional[HuggingFaceConsensusConfig] = None):
        self.config = config or HuggingFaceConsensusConfig()
        self.expert_models: Dict[str, ExpertModelConfig] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.pipelines: Dict[str, Any] = {}
        self._load_expert_models()

    def _load_expert_models(self):
        """Load expert model configurations."""
        for domain, cfg in self.config.expert_models.items():
            self.expert_models[domain] = ExpertModelConfig(
                name=cfg.get("name", domain),
                model_id=cfg["model_id"],
                domain=cfg.get("domain", domain),
                max_length=cfg.get("max_length", 512),
                temperature=cfg.get("temperature", 0.7),
                weight=cfg.get("weight", 1.0),
                use_local=cfg.get("use_local", True),
                api_endpoint=cfg.get("api_endpoint")
            )

    def initialize_models(self):
        """Initialize all expert models and tokenizers."""
        logger.info("Initializing HuggingFace expert models...")

        for domain, config in list(self.expert_models.items()):
            try:
                if config.use_local:
                    self.tokenizers[domain] = AutoTokenizer.from_pretrained(config.model_id)

                    device = 0 if torch.cuda.is_available() else -1
                    dtype = torch.float16 if device >= 0 else torch.float32

                    self.pipelines[domain] = hf_pipeline(
                        "text-generation",
                        model=config.model_id,
                        tokenizer=self.tokenizers[domain],
                        device=device,
                        torch_dtype=dtype,
                        model_kwargs={"low_cpu_mem_usage": True, "torch_dtype": dtype}
                    )
                    logger.info(f"Loaded {config.name} ({config.model_id})")
                else:
                    logger.info(f"Using API for {config.name}")

            except Exception as e:
                logger.warning(f"Failed to load {config.name}: {e}")
                del self.expert_models[domain]

    async def get_consensus_response(
        self,
        prompt: str,
        domain_hint: Optional[str] = None,
        max_responses: int = 5
    ) -> Dict[str, Any]:
        """Get consensus response from multiple expert models."""
        selected_models = self._select_expert_models(domain_hint, max_responses)
        responses = await self._generate_responses_async(prompt, selected_models)
        return self._apply_consensus_algorithm(responses, prompt)

    def _select_expert_models(self, domain_hint: Optional[str], max_responses: int) -> List[ExpertModelConfig]:
        """Select expert models based on domain hint and weights."""
        if domain_hint and domain_hint in self.expert_models:
            selected = [self.expert_models[domain_hint]]
            other_models = [c for d, c in self.expert_models.items() if d != domain_hint]
            other_models.sort(key=lambda x: x.weight, reverse=True)
            selected.extend(other_models[:max_responses - 1])
        else:
            all_models = list(self.expert_models.values())
            all_models.sort(key=lambda x: x.weight, reverse=True)
            selected = all_models[:max_responses]
        return selected

    async def _generate_responses_async(
        self,
        prompt: str,
        models: List[ExpertModelConfig]
    ) -> List[Dict[str, Any]]:
        """Generate responses from multiple models asynchronously."""

        async def generate_single(model_config: ExpertModelConfig) -> Dict[str, Any]:
            try:
                if model_config.use_local and model_config.domain in self.pipelines:
                    pipe = self.pipelines[model_config.domain]
                    result = pipe(
                        prompt,
                        max_length=model_config.max_length,
                        temperature=model_config.temperature,
                        do_sample=True,
                        pad_token_id=pipe.tokenizer.eos_token_id
                    )
                    response_text = result[0]['generated_text'][len(prompt):].strip()
                elif model_config.api_endpoint:
                    response_text = await self._call_api(model_config.api_endpoint, prompt, model_config)
                else:
                    return {"success": False, "error": "No pipeline or API"}

                return {
                    "model": model_config.name,
                    "domain": model_config.domain,
                    "response": response_text,
                    "weight": model_config.weight,
                    "success": True
                }
            except Exception as e:
                return {"model": model_config.name, "domain": model_config.domain, "success": False, "error": str(e)}

        tasks = [generate_single(m) for m in models]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, dict) and r.get("success")]

    async def _call_api(self, endpoint: str, prompt: str, config: ExpertModelConfig) -> str:
        """Call external API endpoint."""
        async with aiohttp.ClientSession() as session:
            payload = {"prompt": prompt, "max_length": config.max_length, "temperature": config.temperature}
            async with session.post(endpoint, json=payload, timeout=30) as response:
                if response.status == 200:
                    return (await response.json()).get("response", "")
                raise Exception(f"API error: {response.status}")

    def _apply_consensus_algorithm(self, responses: List[Dict[str, Any]], prompt: str) -> Dict[str, Any]:
        """Apply consensus algorithm to combine responses."""
        if not responses:
            return {"consensus_response": "", "confidence": 0.0, "participating_models": 0, "method": "none"}

        # Score responses
        scored = []
        for r in responses:
            quality = self._calculate_quality(r["response"], prompt, r["domain"])
            scored.append({**r, "quality": quality, "score": quality * r["weight"]})

        scored.sort(key=lambda x: x["score"], reverse=True)

        # Weighted voting (primary method)
        total = sum(r["score"] for r in scored)
        best = scored[0]
        confidence = best["score"] / total if total > 0 else 0.0

        return {
            "consensus_response": best["response"],
            "confidence": confidence,
            "participating_models": len(responses),
            "method": "weighted_voting",
            "all_responses": scored
        }

    def _calculate_quality(self, response: str, prompt: str, domain: str) -> float:
        """Calculate quality score for a response."""
        if not response.strip():
            return 0.0

        # Length score
        length_score = min(len(response) / 100, 1.0)

        # Relevance (keyword overlap)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        relevance = len(prompt_words & response_words) / max(len(prompt_words), 1)

        # Domain alignment
        keywords = self.config.domain_keywords.get(domain, [])
        domain_score = sum(1 for k in keywords if k.lower() in response.lower()) / max(len(keywords), 1) if keywords else 0.5

        weights = self.config.consensus_settings["quality_weights"]
        return (
            weights["length"] * length_score +
            weights["relevance"] * min(relevance, 1.0) +
            weights["domain_alignment"] * domain_score +
            weights["coherence"] * 0.5  # Default coherence
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded models."""
        return {
            "total_models": len(self.expert_models),
            "loaded_pipelines": len(self.pipelines),
            "domains": list(self.expert_models.keys())
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_lightweight_config() -> HuggingFaceConsensusConfig:
    """Configuration for resource-constrained environments."""
    config = HuggingFaceConsensusConfig()
    config.consensus_settings["max_responses"] = 3
    config.performance_settings["max_concurrent_requests"] = 3
    return config


def create_high_quality_config() -> HuggingFaceConsensusConfig:
    """Configuration for maximum accuracy."""
    config = HuggingFaceConsensusConfig()
    config.consensus_settings["max_responses"] = 7
    config.consensus_settings["min_confidence"] = 0.7
    return config


def create_spanish_optimized_config() -> HuggingFaceConsensusConfig:
    """Configuration optimized for Spanish language tasks."""
    config = HuggingFaceConsensusConfig()
    config.expert_models["spanish"]["weight"] = 2.0
    config.expert_models["spanish_secondary"] = {
        "name": "Spanish Secondary",
        "model_id": "PlanTL-GOB-ES/roberta-base-bne",
        "domain": "spanish_language",
        "temperature": 0.6,
        "weight": 1.8,
        "max_length": 512,
        "use_local": True
    }
    return config


def validate_config(config: HuggingFaceConsensusConfig) -> List[str]:
    """Validate configuration, returns list of errors."""
    errors = []
    required = ["name", "model_id", "domain", "temperature", "weight"]

    for domain, model_cfg in config.expert_models.items():
        for field in required:
            if field not in model_cfg:
                errors.append(f"Missing '{field}' for model '{domain}'")

        temp = model_cfg.get("temperature", 0.0)
        if not 0.0 <= temp <= 2.0:
            errors.append(f"Invalid temperature {temp} for '{domain}'")

        weight = model_cfg.get("weight", 0.0)
        if weight <= 0.0:
            errors.append(f"Invalid weight {weight} for '{domain}'")

    return errors


# =============================================================================
# Convenience Function
# =============================================================================

async def get_huggingface_consensus(
    prompt: str,
    domain_hint: Optional[str] = None,
    max_responses: int = 5,
    config: Optional[HuggingFaceConsensusConfig] = None
) -> Dict[str, Any]:
    """Convenience function to get consensus response."""
    strategy = HuggingFaceConsensusStrategy(config)
    return await strategy.get_consensus_response(prompt, domain_hint, max_responses)


if __name__ == "__main__":
    async def main():
        prompt = "What is the capital of Spain?"
        result = await get_huggingface_consensus(prompt, domain_hint="spanish", max_responses=3)
        logger.info(f"Response: {result['consensus_response']}")
        logger.info(f"Confidence: {result['confidence']:.2f}")

    asyncio.run(main())
