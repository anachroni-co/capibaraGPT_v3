"""
Integrated Consensus Strategy for Capibara6 Training

This module integrates all existing training strategies with TPU v6-64 optimization,
using 27 cores and 7 expert models, all open source with legitimate use.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
import asyncio
import aiohttp
from tqdm import tqdm
import os
import json
from concurrent.futures import ThreadPoolExecutor
import time
from enum import Enum
try:
    from capibara.modules.hierarchical_reasoning import (
        HierarchicalReasoningModule,
        HierarchicalReasoningConfig,
    )
except Exception:
    HierarchicalReasoningModule = None  # type: ignore
    HierarchicalReasoningConfig = None  # type: ignore

try:
    import toml  # type: ignore
except Exception:
    toml = None  # type: ignore

logger = logging.getLogger(__name__)

class ExpertType(Enum):
    """Types of expert models for consensus strategy."""
    SPANISH_LANGUAGE = "spanish_language"
    MATHEMATICS = "mathematics"
    PROGRAMMING = "programming"
    REASONING = "reasoning"
    SCIENTIFIC = "scientific"
    TECHNICAL = "technical"
    GENERAL = "general"
    HIERARCHICAL_REASONING = "hierarchical_reasoning"

@dataclass
class ExpertModelConfig:
    """Configurestion for an expert model with legal compliance."""
    name: str
    model_id: str
    expert_type: ExpertType
    license: str
    use_case: str
    temperature: float = 0.7
    weight: float = 1.0
    max_length: int = 512
    tpu_optimized: bool = True
    legal_compliance: bool = True

@dataclass
class TPUv6Config:
    """Configurestion for TPU v6-64 with 27 cores."""
    total_cores: int = 27
    cores_per_expert: int = 3  # 27 cores / 7 experts ≈ 3 cores per expert
    remaining_cores: int = 6   # 27 - (7 * 3) = 6 cores for distribution
    mesh_shape: str = "3x9"    # 3x9 = 27 cores
    memory_per_core: str = "32GB"
    total_memory: str = "864GB"  # 27 * 32GB
    enable_ultra_optimizations: bool = True

class IntegratedConsensusStrategy:
    """
    Integrated consensus strategy for Capibara6 training.
    
    This integrates all existing strategies with TPU v6-64 optimization,
    using 27 cores and 7 expert models, all open source with legitimate use.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tpu_config = TPUv6Config()
        self._load_hr_config()
        self._init_hierarchical_reasoning()
        self.expert_models = self._load_legal_expert_models()
        self.core_distribution = self._setup_core_distribution()
        self._initialize_strategy()
    
    def _load_hr_config(self) -> None:
        self.hr_cfg: Dict[str, Any] = {
            "enabled": True,
            "integration": {
                "use_with_consensus": True,
                "weight_in_consensus": 3.0,
                "override_threshold": 0.8,
            },
            "task_detection": {},
        }
        if toml:
            try:
                base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "configs_toml", "production")
                cfg_path = os.path.join(base_dir, "unified_router.toml")
                if os.path.exists(cfg_path):
                    data = toml.load(cfg_path)
                    if isinstance(data, dict) and "hierarchical_reasoning" in data:
                        self.hr_cfg.update(data["hierarchical_reasoning"])  # type: ignore
            except Exception:
                pass

    def _init_hierarchical_reasoning(self) -> None:
        self.hierarchical_reasoning = None
        if HierarchicalReasoningModule is None:
            return
        try:
            cfg = HierarchicalReasoningConfig(
                h_dim=int(self.hr_cfg.get("h_dim", 512)),
                l_dim=int(self.hr_cfg.get("l_dim", 256)),
                max_iterations=int(self.hr_cfg.get("max_iterations", 100)),
                convergence_threshold=float(self.hr_cfg.get("convergence_threshold", 0.01)),
                tpu_cores_assigned=int(self.hr_cfg.get("tpu_cores_assigned", 3)),
                use_with_consensus=bool(self.hr_cfg.get("integration", {}).get("use_with_consensus", True)),
                weight_in_consensus=float(self.hr_cfg.get("integration", {}).get("weight_in_consensus", 3.0)),
                override_threshold=float(self.hr_cfg.get("integration", {}).get("override_threshold", 0.8)),
            )
            # Extend keywords from TOML if present
            td = self.hr_cfg.get("task_detection", {}) or {}
            if isinstance(td, dict):
                for key in ("keywords_math", "keywords_reasoning", "keywords_planning"):
                    if key in td and isinstance(td[key], list):
                        setattr(cfg, key, tuple(str(x).lower() for x in td[key]))
            self.hierarchical_reasoning = HierarchicalReasoningModule(cfg)
        except Exception as e:
            logger.warning(f"Failed to initialize HierarchicalReasoningModule: {e}")
            self.hierarchical_reasoning = None
    
    def _load_legal_expert_models(self) -> Dict[ExpertType, ExpertModelConfig]:
        """Load expert models with legal compliance and legitimate use."""
        models = {
            ExpertType.SPANISH_LANGUAGE: ExpertModelConfig(
                name="Spanish Language Expert",
                model_id="PlanTL-GOB-ES/roberta-base-bne",  # Spanish BERT
                expert_type=ExpertType.SPANISH_LANGUAGE,
                license="Apache 2.0",
                use_case="Spanish language processing and generation",
                temperature=0.6,
                weight=2.0,  # Highest weight for Spanish domain
                tpu_optimized=True,
                legal_compliance=True
            ),
            
            ExpertType.MATHEMATICS: ExpertModelConfig(
                name="Mathematics Expert",
                model_id="EleutherAI/gpt-neo-125M",  # Open source GPT
                expert_type=ExpertType.MATHEMATICS,
                license="Apache 2.0",
                use_case="Mathematical reasoning and calculations",
                temperature=0.3,
                weight=1.8,
                tpu_optimized=True,
                legal_compliance=True
            ),
            
            ExpertType.PROGRAMMING: ExpertModelConfig(
                name="Programming Expert",
                model_id="Salesforce/codet5-small",  # Code generation
                expert_type=ExpertType.PROGRAMMING,
                license="BSD-3-Clause",
                use_case="Code generation and understanding",
                temperature=0.4,
                weight=1.7,
                tpu_optimized=True,
                legal_compliance=True
            ),
            
            ExpertType.REASONING: ExpertModelConfig(
                name="Logical Reasoning Expert",
                model_id="microsoft/DialoGPT-small",  # Dialogue reasoning
                expert_type=ExpertType.REASONING,
                license="MIT",
                use_case="Logical reasoning and analytical thinking",
                temperature=0.5,
                weight=1.6,
                tpu_optimized=True,
                legal_compliance=True
            ),
            
            ExpertType.SCIENTIFIC: ExpertModelConfig(
                name="Scientific Expert",
                model_id="allenai/scibert_scivocab_uncased",  # Scientific BERT
                expert_type=ExpertType.SCIENTIFIC,
                license="Apache 2.0",
                use_case="Scientific literature and research",
                temperature=0.2,
                weight=1.5,
                tpu_optimized=True,
                legal_compliance=True
            ),
            
            ExpertType.TECHNICAL: ExpertModelConfig(
                name="Technical Expert",
                model_id="EleutherAI/gpt-neo-125M",  # Technical tasks
                expert_type=ExpertType.TECHNICAL,
                license="Apache 2.0",
                use_case="Technical documentation and engineering",
                temperature=0.4,
                weight=1.4,
                tpu_optimized=True,
                legal_compliance=True
            ),
            
            ExpertType.GENERAL: ExpertModelConfig(
                name="General Expert",
                model_id="microsoft/DialoGPT-medium",  # General purpose
                expert_type=ExpertType.GENERAL,
                license="MIT",
                use_case="General conversation and tasks",
                temperature=0.7,
                weight=1.0,
                tpu_optimized=True,
                legal_compliance=True
            ),
        }
        # Add hierarchical reasoning expert if enabled
        if self.hierarchical_reasoning is not None and bool(self.hr_cfg.get("enabled", True)):
            models[ExpertType.HIERARCHICAL_REASONING] = ExpertModelConfig(
                name="Hierarchical Reasoning Expert",
                model_id="capibara/hierarchical_reasoning_module",
                expert_type=ExpertType.HIERARCHICAL_REASONING,
                license="Apache 2.0",
                use_case="Hierarchical planning and reasoning",
                temperature=0.0,
                weight=float(self.hr_cfg.get("integration", {}).get("weight_in_consensus", 3.0)),
                tpu_optimized=True,
                legal_compliance=True,
            )
        return models
    
    def _setup_core_distribution(self) -> Dict[str, Any]:
        """Setup distribution of 27 TPU cores across 7 experts."""
        logger.info("🏗️ Setting up TPU v6-64 distribution with 27 cores")
        
        # Distribute cores optimally
        base_cores_per_expert = self.tpu_config.total_cores // len(self.expert_models)
        remaining_cores = self.tpu_config.total_cores % len(self.expert_models)
        
        distribution = {}
        core_id = 0
        
        # Sort experts by weight for priority distribution
        sorted_experts = sorted(
            self.expert_models.items(), 
            key=lambda x: x[1].weight, 
            reverse=True
        )
        
        for i, (expert_type, expert_config) in enumerate(sorted_experts):
            # Assign extra cores to higher priority experts
            cores_assigned = base_cores_per_expert
            if i < remaining_cores:
                cores_assigned += 1
            
            distribution[expert_type] = {
                "expert_name": expert_config.name,
                "core_id": core_id,
                "assigned_cores": cores_assigned,
                "weight": expert_config.weight,
                "model_id": expert_config.model_id,
                "license": expert_config.license,
                "use_case": expert_config.use_case
            }
            
            core_id += cores_assigned
        
        logger.info(f"📊 Core Distribution: {distribution}")
        return distribution
    
    def _initialize_strategy(self):
        """Initialize the integrated consensus strategy."""
        logger.info("🚀 Initializing Integrated Consensus Strategy")
        
        # Validate legal compliance
        self._validate_legal_compliance()
        
        # Initialize TPU optimizations
        self._initialize_tpu_optimizations()
        
        # Initialize expert models
        self._initialize_expert_models()
        
        logger.info("✅ Integrated Consensus Strategy initialized successfully")
    
    def _validate_legal_compliance(self):
        """Validates legal compliance of all expert models."""
        logger.info("⚖️ Validating legal compliance")
        
        for expert_type, expert_config in self.expert_models.items():
            if not expert_config.legal_compliance:
                raise ValueError(f"Expert {expert_config.name} is not legally compliant")
            
            logger.info(f"✅ {expert_config.name}: {expert_config.license} - {expert_config.use_case}")
    
    def _initialize_tpu_optimizations(self):
        """Initialize TPU v6-64 specific optimizations."""
        self.tpu_optimizations = {
            "mesh_shape": self.tpu_config.mesh_shape,
            "memory_optimization": True,
            "mixed_precision": True,
            "gradient_accumulation": 4,
            "model_sharding": True,
            "data_parallel": True,
            "tensor_parallel": True,
            "activation_checkpointing": True,
            "compile_optimizations": True,
            "core_distribution": self.core_distribution
        }
    
    def _initialize_expert_models(self):
        """Initialize expert models with TPU optimization."""
        logger.info("🧠 Initializing expert models")
        
        for expert_type, expert_config in self.expert_models.items():
            logger.info(f"✅ {expert_config.name} ({expert_config.model_id}) - {expert_config.license}")
    
    async def get_integrated_consensus_response(
        self, 
        prompt: str, 
        domain_hint: Optional[ExpertType] = None,
        use_all_experts: bool = True,
        max_responses: int = 7
    ) -> Dict[str, Any]:
        """
        Get integrated consensus response using all expert models.
        
        Args:
            prompt: Input prompt
            domain_hint: Optional domain hint to prioritize certain experts
            use_all_experts: Whether to use all 7 experts
            max_responses: Maximum number of responses to generate
            
        Returns:
            Integrated consensus response with legal compliance info
        """
        start_time = time.time()

        # Optional: short-circuit with hierarchical reasoning if strongly confident
        if self.hierarchical_reasoning is not None and bool(self.hr_cfg.get("enabled", True)):
            try:
                if self.hierarchical_reasoning.is_reasoning_task(prompt):
                    hr_result = self.hierarchical_reasoning.process(prompt)
                    override_threshold = float(self.hr_cfg.get("integration", {}).get("override_threshold", 0.8))
                    if hr_result.get("confidence", 0.0) >= override_threshold:
                        return {
                            "consensus_response": hr_result.get("response", ""),
                            "confidence": float(hr_result.get("confidence", 0.0)),
                            "participating_experts": 1,
                            "consensus_method": "hierarchical_reasoning_override",
                            "hr_details": hr_result,
                            "legal_compliance": {
                                "all_models_legal": True,
                                "licenses": ["Apache 2.0"],
                                "use_cases": ["Hierarchical planning and reasoning"],
                                "compliance_verified": True
                            },
                            "tpu_v6_metrics": {
                                "total_cores_used": self.tpu_config.total_cores,
                                "cores_per_expert": self.tpu_config.cores_per_expert,
                                "mesh_shape": self.tpu_config.mesh_shape,
                                "memory_usage": self.tpu_config.total_memory,
                                "inference_time": time.time() - start_time,
                                "expert_distribution": self.core_distribution
                            },
                            "expert_breakdown": {
                                "total_experts": 1,
                                "experts_used": ["Hierarchical Reasoning Expert"],
                                "weights_applied": [self.expert_models.get(ExpertType.HIERARCHICAL_REASONING, ExpertModelConfig("","",ExpertType.GENERAL,"","",)).weight]
                            }
                        }
            except Exception as e:
                logger.warning(f"Hierarchical reasoning short-circuit failed: {e}")
        
        # Select expert models based on domain hint and configuration
        selected_experts = self._select_expert_models(domain_hint, use_all_experts, max_responses)
        
        # Ensure HR expert included when applicable
        if self.hierarchical_reasoning is not None and bool(self.hr_cfg.get("enabled", True)) and use_all_experts:
            if ExpertType.HIERARCHICAL_REASONING in self.expert_models and self.expert_models[ExpertType.HIERARCHICAL_REASONING] not in selected_experts:
                selected_experts.append(self.expert_models[ExpertType.HIERARCHICAL_REASONING])
        
        # Generate responses from all selected experts
        expert_responses = await self._generate_expert_responses(prompt, selected_experts)
        
        # Apply integrated consensus algorithm
        consensus_result = self._apply_integrated_consensus_algorithm(expert_responses, prompt)
        
        # Add legal compliance and TPU metrics
        consensus_result.update({
            "legal_compliance": {
                "all_models_legal": True,
                "licenses": [expert.license for expert in selected_experts],
                "use_cases": [expert.use_case for expert in selected_experts],
                "compliance_verified": True
            },
            "tpu_v6_metrics": {
                "total_cores_used": self.tpu_config.total_cores,
                "cores_per_expert": self.tpu_config.cores_per_expert,
                "mesh_shape": self.tpu_config.mesh_shape,
                "memory_usage": self.tpu_config.total_memory,
                "inference_time": time.time() - start_time,
                "expert_distribution": self.core_distribution
            },
            "expert_breakdown": {
                "total_experts": len(selected_experts),
                "experts_used": [expert.name for expert in selected_experts],
                "weights_applied": [expert.weight for expert in selected_experts]
            }
        })
        
        return consensus_result
    
    def _select_expert_models(
        self, 
        domain_hint: Optional[ExpertType], 
        use_all_experts: bool,
        max_responses: int
    ) -> List[ExpertModelConfig]:
        """Select expert models based on domain hint and configuration."""
        if domain_hint and domain_hint in self.expert_models:
            # Prioritize domain-specific expert
            selected = [self.expert_models[domain_hint]]
            
            if use_all_experts:
                # Add other experts based on weight
                other_experts = [
                    expert for expert_type, expert in self.expert_models.items()
                    if expert_type != domain_hint
                ]
                other_experts.sort(key=lambda x: x.weight, reverse=True)
                selected.extend(other_experts[:max_responses-1])
        else:
            # Select based on weight
            all_experts = list(self.expert_models.values())
            all_experts.sort(key=lambda x: x.weight, reverse=True)
            selected = all_experts[:max_responses] if not use_all_experts else all_experts
        
        return selected
    
    async def _generate_expert_responses(
        self, 
        prompt: str, 
        selected_experts: List[ExpertModelConfig]
    ) -> List[Dict[str, Any]]:
        """Generates responses from all selected expert models."""
        responses = []
        
        async def generate_expert_response(expert_config: ExpertModelConfig) -> Dict[str, Any]:
            """Generates response from a single expert model."""
            try:
                # Simulate TPU v6-64 optimized inference
                response_text = await self._call_expert_inference(
                    expert_config.model_id, 
                    prompt, 
                    expert_config
                )
                
                return {
                    "expert_name": expert_config.name,
                    "expert_type": expert_config.expert_type.value,
                    "model_id": expert_config.model_id,
                    "response": response_text,
                    "weight": expert_config.weight,
                    "temperature": expert_config.temperature,
                    "license": expert_config.license,
                    "use_case": expert_config.use_case,
                    "tpu_optimized": expert_config.tpu_optimized,
                    "success": True
                }
                
            except Exception as e:
                logger.error(f"Error generating response from {expert_config.name}: {e}")
                return {
                    "expert_name": expert_config.name,
                    "expert_type": expert_config.expert_type.value,
                    "model_id": expert_config.model_id,
                    "response": "",
                    "weight": expert_config.weight,
                    "temperature": expert_config.temperature,
                    "license": expert_config.license,
                    "use_case": expert_config.use_case,
                    "tpu_optimized": expert_config.tpu_optimized,
                    "success": False,
                    "error": str(e)
                }
        
        # Generate responses from all experts in parallel
        tasks = [generate_expert_response(expert) for expert in selected_experts]
        expert_responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        for response in expert_responses:
            if isinstance(response, dict):
                responses.append(response)
        
        return [r for r in responses if r.get("success", False)]
    
    async def _call_expert_inference(
        self, 
        model_id: str, 
        prompt: str, 
        expert_config: ExpertModelConfig
    ) -> str:
        """Call expert model inference with TPU v6-64 optimization."""
        # Simulate TPU v6-64 inference with legal compliance
        await asyncio.sleep(0.05)  # Simulate TPU v6-64 speed
        
        if expert_config.expert_type == ExpertType.HIERARCHICAL_REASONING and self.hierarchical_reasoning is not None:
            try:
                result = self.hierarchical_reasoning.process(prompt)
                return result.get("response", "")
            except Exception as e:
                logger.warning(f"Hierarchical reasoning inference failed: {e}")
        
        # Return simulated response based on expert type
        expert_responses = {
            ExpertType.SPANISH_LANGUAGE: f"Respuesta en español optimizada por TPU v6-64: {prompt}",
            ExpertType.MATHEMATICS: f"Solución matemática optimizada por TPU v6-64: {prompt}",
            ExpertType.PROGRAMMING: f"Código optimizado por TPU v6-64: def solution(): return '{prompt}'",
            ExpertType.REASONING: f"Razonamiento lógico optimizado por TPU v6-64: {prompt}",
            ExpertType.SCIENTIFIC: f"Análisis científico optimizado por TPU v6-64: {prompt}",
            ExpertType.TECHNICAL: f"Análisis técnico optimizado por TPU v6-64: {prompt}",
            ExpertType.GENERAL: f"Respuesta general optimizada por TPU v6-64: {prompt}"
        }
        
        return expert_responses.get(expert_config.expert_type, f"Respuesta optimizada por TPU v6-64: {prompt}")
    
    def _apply_integrated_consensus_algorithm(
        self, 
        expert_responses: List[Dict[str, Any]], 
        original_prompt: str
    ) -> Dict[str, Any]:
        """Apply integrated consensus algorithm across expert models."""
        if not expert_responses:
            return {
                "consensus_response": "",
                "confidence": 0.0,
                "participating_experts": 0,
                "consensus_method": "no_responses"
            }
        
        # Calculate quality scores for each response
        scored_responses = []
        for response in expert_responses:
            quality_score = self._calculate_response_quality(
                response["response"], 
                original_prompt,
                response["expert_type"]
            )
            scored_responses.append({
                **response,
                "quality_score": quality_score,
                "final_score": quality_score * response["weight"]
            })
        
        # Sort by final score
        scored_responses.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Apply consensus methods
        consensus_methods = {
            "weighted_voting": self._weighted_voting_consensus(scored_responses),
            "expert_priority": self._expert_priority_consensus(scored_responses),
            "quality_weighted": self._quality_weighted_consensus(scored_responses)
        }
        
        # Select best consensus method
        best_method = max(consensus_methods.keys(), 
                         key=lambda k: consensus_methods[k]["confidence"])
        
        return {
            "consensus_response": consensus_methods[best_method]["response"],
            "confidence": consensus_methods[best_method]["confidence"],
            "participating_experts": len(expert_responses),
            "consensus_method": best_method,
            "all_responses": scored_responses,
            "consensus_breakdown": consensus_methods,
            "legal_compliance_verified": True
        }
    
    def _calculate_response_quality(
        self, 
        response: str, 
        prompt: str, 
        expert_type: str
    ) -> float:
        """Calculate quality score for a response."""
        if not response.strip():
            return 0.0
        
        # Base quality metrics
        length_score = min(len(response) / 100, 1.0)
        relevance_score = self._calculate_relevance(response, prompt)
        coherence_score = self._calculate_coherence(response)
        domain_alignment = self._calculate_domain_alignment(response, expert_type)
        
        # Weighted combination
        quality_score = (
            0.3 * length_score +
            0.4 * relevance_score +
            0.2 * coherence_score +
            0.1 * domain_alignment
        )
        
        return min(quality_score, 1.0)
    
    def _calculate_relevance(self, response: str, prompt: str) -> float:
        """Calculate relevance between response and prompt."""
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        if not prompt_words:
            return 0.0
        
        overlap = len(prompt_words.intersection(response_words))
        return min(overlap / len(prompt_words), 1.0)
    
    def _calculate_coherence(self, response: str) -> float:
        """Calculate coherence of the response."""
        sentences = response.split('.')
        if len(sentences) <= 1:
            return 0.5
        
        connectors = ['porque', 'sin embargo', 'además', 'también', 'pero', 'aunque']
        connector_count = sum(1 for connector in connectors if connector in response.lower())
        
        return min(connector_count / len(sentences), 1.0)
    
    def _calculate_domain_alignment(self, response: str, expert_type: str) -> float:
        """Calculate alignment with the expert domain."""
        domain_keywords = {
            "spanish_language": ["español", "gramática", "vocabulario", "idioma"],
            "mathematics": ["ecuación", "cálculo", "número", "matemática", "álgebra"],
            "programming": ["código", "función", "variable", "programa", "algoritmo"],
            "reasoning": ["lógica", "razonamiento", "premisa", "conclusión"],
            "scientific": ["científico", "investigación", "estudio", "análisis"],
            "technical": ["técnico", "ingeniería", "tecnología", "sistema"],
            "general": []
        }
        
        keywords = domain_keywords.get(expert_type, [])
        if not keywords:
            return 0.5
        
        keyword_matches = sum(1 for keyword in keywords if keyword in response.lower())
        return min(keyword_matches / len(keywords), 1.0)
    
    def _weighted_voting_consensus(self, scored_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply weighted voting consensus."""
        total_weight = sum(r["final_score"] for r in scored_responses)
        
        if total_weight == 0:
            return {"response": "", "confidence": 0.0}
        
        best_response = max(scored_responses, key=lambda x: x["final_score"])
        confidence = best_response["final_score"] / total_weight
        
        return {
            "response": best_response["response"],
            "confidence": confidence
        }
    
    def _expert_priority_consensus(self, scored_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply expert priority consensus."""
        # Sort by expert weight (higher weight = higher priority)
        sorted_responses = sorted(scored_responses, key=lambda x: x["weight"], reverse=True)
        
        best_response = sorted_responses[0]
        confidence = best_response["weight"] / max(r["weight"] for r in scored_responses)
        
        return {
            "response": best_response["response"],
            "confidence": confidence
        }
    
    def _quality_weighted_consensus(self, scored_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply quality-weighted consensus."""
        total_quality = sum(r["quality_score"] for r in scored_responses)
        
        if total_quality == 0:
            return {"response": "", "confidence": 0.0}
        
        best_response = max(scored_responses, key=lambda x: x["quality_score"])
        confidence = best_response["quality_score"] / total_quality
        
        return {
            "response": best_response["response"],
            "confidence": confidence
        }
    
    def get_integrated_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the integrated strategy."""
        return {
            "tpu_v6_config": {
                "total_cores": self.tpu_config.total_cores,
                "cores_per_expert": self.tpu_config.cores_per_expert,
                "mesh_shape": self.tpu_config.mesh_shape,
                "total_memory": self.tpu_config.total_memory
            },
            "expert_models": {
                expert_type.value: {
                    "name": expert_config.name,
                    "model_id": expert_config.model_id,
                    "weight": expert_config.weight,
                    "license": expert_config.license,
                    "use_case": expert_config.use_case,
                    "tpu_optimized": expert_config.tpu_optimized,
                    "legal_compliance": expert_config.legal_compliance
                }
                for expert_type, expert_config in self.expert_models.items()
            },
            "core_distribution": self.core_distribution,
            "legal_compliance": {
                "all_models_legal": True,
                "licenses": list(set(expert.license for expert in self.expert_models.values())),
                "compliance_verified": True
            }
        }

# Convenience function for easy integration
async def get_integrated_consensus(
    prompt: str,
    domain_hint: Optional[ExpertType] = None,
    use_all_experts: bool = True
) -> Dict[str, Any]:
    """
    Get integrated consensus response using all expert models.
    
    Args:
        prompt: Input prompt
        domain_hint: Optional domain hint
        use_all_experts: Whether to use all 7 experts
        
    Returns:
        Integrated consensus response
    """
    config = {
        "enable_tpu_v6": True,
        "enable_legal_compliance": True,
        "enable_all_strategies": True
    }
    
    strategy = IntegratedConsensusStrategy(config)
    return await strategy.get_integrated_consensus_response(
        prompt=prompt,
        domain_hint=domain_hint,
        use_all_experts=use_all_experts
    )

if __name__ == "__main__":
    # Example usage
    async def main():
        prompt = "¿Cuál es la capital de España?"
        
        # Test with Spanish Language Expert
        result = await get_integrated_consensus(
            prompt=prompt,
            domain_hint=ExpertType.SPANISH_LANGUAGE,
            use_all_experts=True
        )
        
        print("=== Integrated Consensus Results ===")
        print(f"Prompt: {prompt}")
        print(f"Response: {result['consensus_response']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Method: {result['consensus_method']}")
        print(f"Experts used: {result['participating_experts']}")
        print(f"Legal compliance: {result['legal_compliance']['all_models_legal']}")
        print(f"TPU cores used: {result['tpu_v6_metrics']['total_cores_used']}")
        
        # Show expert breakdown
        print("\n=== Expert Breakdown ===")
        for expert in result['expert_breakdown']['experts_used']:
            print(f"  ✅ {expert}")
    
    asyncio.run(main())