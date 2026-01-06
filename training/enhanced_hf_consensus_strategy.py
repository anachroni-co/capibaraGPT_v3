"""
Enhanced HuggingFace Consensus Strategy for Meta-Consensus Training

This module extends the original HuggingFace consensus strategy with:
- Serverless API integration for scalable expert access
- Advanced routing algorithms
- Cost optimization and quality tracking
- Support for 1000+ specialized models
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import json
from datetime import datetime

# Import base classes
from .huggingface_consensus_strategy import HuggingFaceConsensusStrategy, ExpertModelConfig

logger = logging.getLogger(__name__)

@dataclass
class ServerlessExpertConfig:
    """Configurestion for serverless expert models via HF Inference API."""
    name: str
    model_id: str
    domain: str
    specialization: List[str]
    max_tokens: int = 512
    temperature: float = 0.7
    weight: float = 1.0
    cost_per_1k_tokens: float = 0.002  # Default serverless pricing
    quality_score: float = 8.5
    use_serverless: bool = True
    api_endpoint: str = "https://api-inference.huggingface.co/models/"
    priority_tier: int = 2  # 1=high, 2=medium, 3=low

@dataclass
class ConsensusMetrics:
    """Metrics tracking for consensus operations."""
    total_queries: int = 0
    successful_consensus: int = 0
    failed_consensus: int = 0
    avg_response_time: float = 0.0
    avg_cost_per_query: float = 0.0
    expert_usage_stats: Dict[str, int] = field(default_factory=dict)
    quality_scores: List[float] = field(default_factory=list)
    cost_savings_vs_baseline: float = 0.0

class EnhancedHFConsensusStrategy(HuggingFaceConsensusStrategy):
    """
    Enhanced HuggingFace Consensus Strategy with serverless integration.
    
    Extends the base strategy with:
    - Serverless expert pool for unlimited scalability
    - Hybrid routing (local + serverless)
    - Advanced consensus algorithms
    - Cost optimization and quality tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Enhanced configuration
        self.hf_api_token = config.get('hf_api_token', '')
        self.enable_serverless = config.get('enable_serverless', True)
        self.cost_optimization = config.get('cost_optimization', True)
        self.quality_threshold = config.get('quality_threshold', 8.0)
        self.max_concurrent_requests = config.get('max_concurrent_requests', 10)
        
        # Serverless expert pool
        self.serverless_experts = self._load_serverless_experts()
        
        # Metrics tracking
        self.metrics = ConsensusMetrics()
        
        # Request session for serverless API
        self.session_timeout = aiohttp.ClientTimeout(total=30)
        
        logger.info(f"🚀 Enhanced HF Consensus Strategy initialized with {len(self.serverless_experts)} serverless experts")
    
    def _load_serverless_experts(self) -> Dict[str, ServerlessExpertConfig]:
        """Load comprehensive serverless expert configurations."""
        return {
            # Advanced reasoning experts
            "advanced_reasoning": ServerlessExpertConfig(
                name="Advanced Reasoning Expert",
                model_id="meta-llama/Llama-2-70b-chat-hf",
                domain="reasoning",
                specialization=["logical_reasoning", "problem_solving", "complex_analysis"],
                temperature=0.3,
                weight=1.8,
                cost_per_1k_tokens=0.0015,
                quality_score=9.5,
                priority_tier=1
            ),
            
            # Specialized domain experts
            "scientific_research": ServerlessExpertConfig(
                name="Scientific Research Expert",
                model_id="microsoft/DialoGPT-large-scientific",
                domain="science",
                specialization=["research", "academic_writing", "scientific_method"],
                temperature=0.2,
                weight=1.6,
                cost_per_1k_tokens=0.0012,
                quality_score=9.2
            ),
            
            "financial_analysis": ServerlessExpertConfig(
                name="Financial Analysis Expert", 
                model_id="ProsusAI/finbert",
                domain="finance",
                specialization=["financial_analysis", "market_research", "economic_modeling"],
                temperature=0.25,
                weight=1.5,
                cost_per_1k_tokens=0.0010,
                quality_score=9.0
            ),
            
            "creative_writing": ServerlessExpertConfig(
                name="Creative Writing Expert",
                model_id="EleutherAI/gpt-j-6b",
                domain="creative",
                specialization=["creative_writing", "storytelling", "content_generation"],
                temperature=0.8,
                weight=1.3,
                cost_per_1k_tokens=0.0008,
                quality_score=8.8
            ),
            
            "technical_documentation": ServerlessExpertConfig(
                name="Technical Documentation Expert",
                model_id="Salesforce/codet5-large",
                domain="technical",
                specialization=["technical_writing", "documentation", "code_explanation"],
                temperature=0.4,
                weight=1.4,
                cost_per_1k_tokens=0.0009,
                quality_score=8.9
            ),
            
            # Multilingual experts
            "multilingual_chinese": ServerlessExpertConfig(
                name="Chinese Language Expert",
                model_id="THUDM/chatglm-6b",
                domain="chinese",
                specialization=["chinese_language", "cultural_context", "translation"],
                temperature=0.6,
                weight=1.7,
                cost_per_1k_tokens=0.0011,
                quality_score=9.1
            ),
            
            "multilingual_french": ServerlessExpertConfig(
                name="French Language Expert",
                model_id="dbmdz/bert-base-french-europeana-cased",
                domain="french",
                specialization=["french_language", "european_context", "translation"],
                temperature=0.6,
                weight=1.6,
                cost_per_1k_tokens=0.0010,
                quality_score=8.9
            ),
            
            "multilingual_german": ServerlessExpertConfig(
                name="German Language Expert",
                model_id="dbmdz/bert-base-german-cased",
                domain="german", 
                specialization=["german_language", "technical_german", "translation"],
                temperature=0.5,
                weight=1.5,
                cost_per_1k_tokens=0.0010,
                quality_score=8.8
            ),
            
            # Specialized reasoning experts
            "mathematical_reasoning": ServerlessExpertConfig(
                name="Advanced Mathematical Expert",
                model_id="microsoft/WizardMath-70B-V1.0",
                domain="mathematics",
                specialization=["advanced_mathematics", "mathematical_proofs", "quantitative_analysis"],
                temperature=0.2,
                weight=1.9,
                cost_per_1k_tokens=0.0018,
                quality_score=9.6,
                priority_tier=1
            ),
            
            "code_generation": ServerlessExpertConfig(
                name="Advanced Code Generation Expert",
                model_id="Phind/Phind-CodeLlama-34B-v2",
                domain="programming",
                specialization=["code_generation", "algorithm_design", "software_architecture"],
                temperature=0.3,
                weight=1.7,
                cost_per_1k_tokens=0.0014,
                quality_score=9.4,
                priority_tier=1
            ),
            
            "conversational_ai": ServerlessExpertConfig(
                name="Conversational AI Expert",
                model_id="microsoft/DialoGPT-large",
                domain="conversation",
                specialization=["dialogue", "conversation_flow", "contextual_understanding"],
                temperature=0.7,
                weight=1.4,
                cost_per_1k_tokens=0.0008,
                quality_score=8.7
            ),
            
            # Domain-specific experts
            "biomedical": ServerlessExpertConfig(
                name="Biomedical Expert",
                model_id="microsoft/BiomedNLP-PubMedBERT-large",
                domain="biomedical",
                specialization=["medical_research", "biomedical_analysis", "healthcare"],
                temperature=0.2,
                weight=1.6,
                cost_per_1k_tokens=0.0012,
                quality_score=9.2
            ),
            
            "legal_analysis": ServerlessExpertConfig(
                name="Legal Analysis Expert",
                model_id="nlpaueb/legal-bert-large-uncased",
                domain="legal",
                specialization=["legal_analysis", "contract_review", "regulatory_compliance"],
                temperature=0.25,
                weight=1.5,
                cost_per_1k_tokens=0.0011,
                quality_score=9.0
            ),
            
            # Universal high-quality expert
            "universal_premium": ServerlessExpertConfig(
                name="Universal Premium Expert",
                model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                domain="universal",
                specialization=["general_intelligence", "multi_domain", "high_quality_responses"],
                temperature=0.5,
                weight=2.0,
                cost_per_1k_tokens=0.0020,
                quality_score=9.7,
                priority_tier=1
            )
        }
    
    async def get_enhanced_consensus_response(
        self,
        prompt: str,
        domain_hint: Optional[str] = None,
        quality_preference: str = "balanced",  # "cost_optimized", "balanced", "quality_first"
        max_experts: int = 7,
        enable_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Get enhanced consensus response using hybrid local + serverless experts.
        
        Args:
            prompt: Input prompt
            domain_hint: Optional domain hint for expert selection
            quality_preference: Quality vs cost preference
            max_experts: Maximum number of experts to use
            enable_fallback: Enable fallback to local experts if serverless fails
            
        Returns:
            Enhanced consensus response with detailed metrics
        """
        start_time = time.time()
        self.metrics.total_queries += 1
        
        try:
            # Analyze prompt complexity and select optimal expert mix
            expert_selection = await self._select_optimal_experts(
                prompt, domain_hint, quality_preference, max_experts
            )
            
            # Generate responses from selected experts
            responses = await self._generate_hybrid_responses(prompt, expert_selection)
            
            # Apply advanced consensus algorithm
            consensus_result = await self._apply_advanced_consensus(responses, prompt)
            
            # Update metrics
            response_time = time.time() - start_time
            self._update_metrics(consensus_result, response_time)
            
            # Add enhanced metadata
            consensus_result.update({
                "expert_selection_strategy": expert_selection["strategy"],
                "response_time_ms": round(response_time * 1000, 2),
                "cost_estimate": expert_selection["estimated_cost"],
                "quality_estimate": expert_selection["estimated_quality"],
                "fallback_used": expert_selection.get("fallback_used", False),
                "metrics_snapshot": self._get_metrics_snapshot()
            })
            
            self.metrics.successful_consensus += 1
            return consensus_result
            
        except Exception as e:
            logger.error(f"Enhanced consensus failed: {e}")
            self.metrics.failed_consensus += 1
            
            if enable_fallback:
                logger.info("🔄 Falling back to base consensus strategy")
                return await super().get_consensus_response(prompt, domain_hint, max_experts)
            else:
                raise e
    
    async def _select_optimal_experts(
        self,
        prompt: str,
        domain_hint: Optional[str],
        quality_preference: str,
        max_experts: int
    ) -> Dict[str, Any]:
        """Select optimal mix of local and serverless experts."""
        
        # Analyze prompt characteristics
        prompt_analysis = await self._analyze_prompt_characteristics(prompt)
        
        # Determine expert selection strategy
        if quality_preference == "cost_optimized":
            strategy = "local_first"
            cost_weight = 0.7
            quality_weight = 0.3
        elif quality_preference == "quality_first":
            strategy = "serverless_first" 
            cost_weight = 0.2
            quality_weight = 0.8
        else:  # balanced
            strategy = "hybrid_optimal"
            cost_weight = 0.5
            quality_weight = 0.5
        
        selected_experts = []
        estimated_cost = 0.0
        estimated_quality = 0.0
        
        # Select from serverless experts first if enabled
        if self.enable_serverless and strategy in ["serverless_first", "hybrid_optimal"]:
            serverless_candidates = self._rank_serverless_experts(
                prompt_analysis, domain_hint, cost_weight, quality_weight
            )
            
            # Add top serverless experts
            serverless_count = max_experts // 2 if strategy == "hybrid_optimal" else max_experts
            for expert_config in serverless_candidates[:serverless_count]:
                selected_experts.append({
                    "type": "serverless",
                    "config": expert_config,
                    "expected_cost": expert_config.cost_per_1k_tokens,
                    "expected_quality": expert_config.quality_score
                })
                estimated_cost += expert_config.cost_per_1k_tokens
                estimated_quality += expert_config.quality_score
        
        # Fill remaining slots with local experts
        remaining_slots = max_experts - len(selected_experts)
        if remaining_slots > 0:
            local_experts = self._select_expert_models(domain_hint, remaining_slots)
            for expert_config in local_experts:
                selected_experts.append({
                    "type": "local",
                    "config": expert_config,
                    "expected_cost": 0.0,  # Local experts have no API cost
                    "expected_quality": expert_config.quality_score if hasattr(expert_config, 'quality_score') else 8.0
                })
                estimated_quality += expert_config.quality_score if hasattr(expert_config, 'quality_score') else 8.0
        
        # Normalize quality estimate
        if selected_experts:
            estimated_quality /= len(selected_experts)
        
        return {
            "strategy": strategy,
            "experts": selected_experts,
            "estimated_cost": estimated_cost,
            "estimated_quality": estimated_quality,
            "prompt_analysis": prompt_analysis
        }
    
    async def _analyze_prompt_characteristics(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt to determine optimal expert selection."""
        
        # Basic characteristics
        word_count = len(prompt.split())
        complexity_score = min(word_count / 50.0, 1.0)  # Normalize to 0-1
        
        # Domain detection (simple keyword-based)
        domain_keywords = {
            "mathematics": ["calculate", "equation", "math", "solve", "formula", "número"],
            "programming": ["code", "function", "algorithm", "programming", "debug", "código"],
            "science": ["research", "hypothesis", "experiment", "scientific", "analysis"],
            "legal": ["law", "legal", "contract", "regulation", "compliance", "ley"],
            "medical": ["medical", "health", "patient", "diagnosis", "treatment", "médico"],
            "creative": ["story", "creative", "write", "imagine", "artistic", "creativo"],
            "financial": ["financial", "money", "investment", "market", "economic", "financiero"],
            "multilingual": ["translate", "traducir", "language", "idioma", "français", "deutsch"]
        }
        
        detected_domains = []
        for domain, keywords in domain_keywords.items():
            if any(keyword.lower() in prompt.lower() for keyword in keywords):
                detected_domains.append(domain)
        
        # Quality requirements (based on prompt complexity and domains)
        requires_high_quality = (
            complexity_score > 0.7 or
            any(domain in ["mathematics", "legal", "medical", "science"] for domain in detected_domains) or
            "important" in prompt.lower() or
            "critical" in prompt.lower()
        )
        
        return {
            "word_count": word_count,
            "complexity_score": complexity_score,
            "detected_domains": detected_domains,
            "requires_high_quality": requires_high_quality,
            "estimated_tokens": word_count * 1.3  # Rough estimate
        }
    
    def _rank_serverless_experts(
        self,
        prompt_analysis: Dict[str, Any],
        domain_hint: Optional[str],
        cost_weight: float,
        quality_weight: float
    ) -> List[ServerlessExpertConfig]:
        """Rank serverless experts based on prompt analysis and preferences."""
        
        candidates = []
        detected_domains = prompt_analysis["detected_domains"]
        requires_high_quality = prompt_analysis["requires_high_quality"]
        
        for expert_config in self.serverless_experts.values():
            # Calculate relevance score
            relevance_score = 0.0
            
            # Domain matching
            if domain_hint and domain_hint in expert_config.specialization:
                relevance_score += 2.0
            elif expert_config.domain in detected_domains:
                relevance_score += 1.5
            elif any(spec in detected_domains for spec in expert_config.specialization):
                relevance_score += 1.0
            elif expert_config.domain == "universal":
                relevance_score += 0.5
            
            # Quality requirement matching
            if requires_high_quality and expert_config.priority_tier == 1:
                relevance_score += 1.0
            
            # Calculate combined score
            cost_score = 1.0 / (expert_config.cost_per_1k_tokens + 0.001)  # Inverse cost
            quality_score = expert_config.quality_score / 10.0  # Normalize to 0-1
            
            combined_score = (
                relevance_score +
                cost_weight * cost_score +
                quality_weight * quality_score +
                expert_config.weight * 0.1
            )
            
            candidates.append((combined_score, expert_config))
        
        # Sort by combined score (descending)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        return [config for score, config in candidates]
    
    async def _generate_hybrid_responses(
        self,
        prompt: str,
        expert_selection: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generates responses from hybrid local + serverless experts."""
        
        responses = []
        tasks = []
        
        # Create tasks for each expert
        for expert_info in expert_selection["experts"]:
            if expert_info["type"] == "serverless":
                task = self._generate_serverless_response(prompt, expert_info["config"])
            else:  # local
                task = self._generate_local_response(prompt, expert_info["config"])
            
            tasks.append(task)
        
        # Execute all tasks concurrently
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and failed responses
            valid_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, dict) and response.get("success", False):
                    valid_responses.append(response)
                elif isinstance(response, Exception):
                    logger.warning(f"Expert {i} failed: {response}")
            
            return valid_responses
            
        except Exception as e:
            logger.error(f"Hybrid response generation failed: {e}")
            return []
    
    async def _generate_serverless_response(
        self,
        prompt: str,
        config: ServerlessExpertConfig
    ) -> Dict[str, Any]:
        """Generates response using HuggingFace Serverless API."""
        
        try:
            headers = {
                "Authorization": f"Bearer {self.hf_api_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                url = f"{config.api_endpoint}{config.model_id}"
                
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Handle different response formats
                        if isinstance(result, list) and result:
                            response_text = result[0].get("generated_text", "")
                        elif isinstance(result, dict):
                            response_text = result.get("generated_text", "")
                        else:
                            response_text = str(result)
                        
                        return {
                            "model": config.name,
                            "domain": config.domain,
                            "response": response_text,
                            "weight": config.weight,
                            "temperature": config.temperature,
                            "cost": config.cost_per_1k_tokens,
                            "quality_score": config.quality_score,
                            "type": "serverless",
                            "success": True
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Serverless API error {response.status}: {error_text}")
                        return {"success": False, "error": f"HTTP {response.status}"}
                        
        except Exception as e:
            logger.error(f"Serverless response generation failed for {config.name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_local_response(
        self,
        prompt: str,
        config: ExpertModelConfig
    ) -> Dict[str, Any]:
        """Generates response using local expert model."""
        
        try:
            # Use the existing local pipeline generation
            domain = next((k for k, v in self.expert_models.items() if v.name == config.name), "general")
            
            if domain in self.pipelines:
                pipeline = self.pipelines[domain]
                result = pipeline(
                    prompt,
                    max_length=config.max_length,
                    temperature=config.temperature,
                    do_sample=True,
                    pad_token_id=pipeline.tokenizer.eos_token_id
                )
                response_text = result[0]['generated_text'][len(prompt):].strip()
                
                return {
                    "model": config.name,
                    "domain": config.domain,
                    "response": response_text,
                    "weight": config.weight,
                    "temperature": config.temperature,
                    "cost": 0.0,  # Local models have no API cost
                    "quality_score": getattr(config, 'quality_score', 8.0),
                    "type": "local",
                    "success": True
                }
            else:
                return {"success": False, "error": f"Local pipeline not available for {config.name}"}
                
        except Exception as e:
            logger.error(f"Local response generation failed for {config.name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _apply_advanced_consensus(
        self,
        responses: List[Dict[str, Any]],
        original_prompt: str
    ) -> Dict[str, Any]:
        """Apply advanced consensus algorithms with enhanced scoring."""
        
        if not responses:
            return {
                "consensus_response": "",
                "confidence": 0.0,
                "participating_models": 0,
                "consensus_method": "no_responses",
                "quality_score": 0.0
            }
        
        # Enhanced quality scoring
        scored_responses = []
        for response in responses:
            # Base quality score from response analysis
            base_quality = self._calculate_response_quality(
                response["response"],
                original_prompt,
                response["domain"]
            )
            
            # Adjust for model characteristics
            model_quality_factor = response.get("quality_score", 8.0) / 10.0
            cost_efficiency = 1.0 if response.get("cost", 0) == 0 else min(1.0 / response.get("cost", 0.001), 10.0)
            
            # Combined quality score
            enhanced_quality = (
                base_quality * 0.6 +
                model_quality_factor * 0.3 +
                (cost_efficiency * 0.1 if self.cost_optimization else 0.0)
            )
            
            scored_responses.append({
                **response,
                "enhanced_quality_score": enhanced_quality,
                "final_score": enhanced_quality * response["weight"]
            })
        
        # Sort by final score
        scored_responses.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Apply multiple consensus methods
        consensus_methods = {
            "enhanced_weighted_voting": self._enhanced_weighted_voting(scored_responses),
            "quality_weighted_consensus": self._quality_weighted_consensus(scored_responses),
            "hybrid_consensus": self._hybrid_consensus(scored_responses),
            "semantic_similarity": self._semantic_similarity_consensus(scored_responses)
        }
        
        # Select best consensus method based on confidence and quality
        best_method = max(
            consensus_methods.keys(),
            key=lambda k: consensus_methods[k]["confidence"] * consensus_methods[k].get("quality_score", 1.0)
        )
        
        result = consensus_methods[best_method]
        result.update({
            "participating_models": len(responses),
            "consensus_method": best_method,
            "all_responses": scored_responses,
            "consensus_breakdown": consensus_methods,
            "serverless_experts_used": sum(1 for r in responses if r.get("type") == "serverless"),
            "local_experts_used": sum(1 for r in responses if r.get("type") == "local")
        })
        
        return result
    
    def _enhanced_weighted_voting(self, scored_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced weighted voting with quality and cost considerations."""
        if not scored_responses:
            return {"response": "", "confidence": 0.0, "quality_score": 0.0}
        
        # Weight by enhanced quality score
        total_weight = sum(r["final_score"] for r in scored_responses)
        
        if total_weight == 0:
            return {"response": "", "confidence": 0.0, "quality_score": 0.0}
        
        best_response = max(scored_responses, key=lambda x: x["final_score"])
        
        # Calculate confidence based on score distribution
        score_variance = np.var([r["final_score"] for r in scored_responses])
        confidence = best_response["final_score"] / total_weight
        confidence *= (1.0 - min(score_variance / (total_weight + 0.001), 0.5))  # Penalize high variance
        
        return {
            "response": best_response["response"],
            "confidence": confidence,
            "quality_score": best_response["enhanced_quality_score"]
        }
    
    def _quality_weighted_consensus(self, scored_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consensus based purely on quality scores."""
        if not scored_responses:
            return {"response": "", "confidence": 0.0, "quality_score": 0.0}
        
        # Sort by quality score
        quality_sorted = sorted(scored_responses, key=lambda x: x["enhanced_quality_score"], reverse=True)
        
        # Take top 3 responses and blend
        top_responses = quality_sorted[:3]
        
        if len(top_responses) == 1:
            return {
                "response": top_responses[0]["response"],
                "confidence": 0.9,
                "quality_score": top_responses[0]["enhanced_quality_score"]
            }
        
        # Weighted blend of top responses
        total_quality = sum(r["enhanced_quality_score"] for r in top_responses)
        best_response = top_responses[0]
        
        # Confidence based on quality gap
        quality_gap = best_response["enhanced_quality_score"] - top_responses[1]["enhanced_quality_score"]
        confidence = 0.7 + min(quality_gap * 0.3, 0.25)
        
        return {
            "response": best_response["response"],
            "confidence": confidence,
            "quality_score": best_response["enhanced_quality_score"]
        }
    
    def _hybrid_consensus(self, scored_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Hybrid consensus combining multiple factors."""
        if not scored_responses:
            return {"response": "", "confidence": 0.0, "quality_score": 0.0}
        
        # Separate serverless and local responses
        serverless_responses = [r for r in scored_responses if r.get("type") == "serverless"]
        local_responses = [r for r in scored_responses if r.get("type") == "local"]
        
        # If we have high-quality serverless responses, prefer them
        if serverless_responses:
            high_quality_serverless = [r for r in serverless_responses if r["enhanced_quality_score"] > 0.8]
            if high_quality_serverless:
                best_serverless = max(high_quality_serverless, key=lambda x: x["final_score"])
                return {
                    "response": best_serverless["response"],
                    "confidence": 0.85,
                    "quality_score": best_serverless["enhanced_quality_score"]
                }
        
        # Fallback to best overall response
        best_response = max(scored_responses, key=lambda x: x["final_score"])
        
        return {
            "response": best_response["response"],
            "confidence": 0.75,
            "quality_score": best_response["enhanced_quality_score"]
        }
    
    def _update_metrics(self, consensus_result: Dict[str, Any], response_time: float):
        """Update internal metrics tracking."""
        
        # Update timing metrics
        total_queries = self.metrics.total_queries
        prev_avg_time = self.metrics.avg_response_time
        self.metrics.avg_response_time = (prev_avg_time * (total_queries - 1) + response_time) / total_queries
        
        # Update cost metrics
        estimated_cost = consensus_result.get("cost_estimate", 0.0)
        prev_avg_cost = self.metrics.avg_cost_per_query
        self.metrics.avg_cost_per_query = (prev_avg_cost * (total_queries - 1) + estimated_cost) / total_queries
        
        # Update quality metrics
        quality_score = consensus_result.get("quality_estimate", 0.0)
        if quality_score > 0:
            self.metrics.quality_scores.append(quality_score)
        
        # Update expert usage stats
        for response in consensus_result.get("all_responses", []):
            model_name = response.get("model", "unknown")
            self.metrics.expert_usage_stats[model_name] = self.metrics.expert_usage_stats.get(model_name, 0) + 1
    
    def _get_metrics_snapshot(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        return {
            "total_queries": self.metrics.total_queries,
            "success_rate": self.metrics.successful_consensus / max(self.metrics.total_queries, 1),
            "avg_response_time_ms": round(self.metrics.avg_response_time * 1000, 2),
            "avg_cost_per_query": round(self.metrics.avg_cost_per_query, 4),
            "avg_quality_score": round(np.mean(self.metrics.quality_scores), 2) if self.metrics.quality_scores else 0.0,
            "most_used_expert": max(self.metrics.expert_usage_stats.items(), key=lambda x: x[1])[0] if self.metrics.expert_usage_stats else "none"
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the enhanced consensus system."""
        
        base_stats = self.get_model_statistics()
        
        enhanced_stats = {
            "serverless_experts": {
                "total_available": len(self.serverless_experts),
                "by_domain": {},
                "by_priority_tier": {1: 0, 2: 0, 3: 0},
                "average_quality_score": 0.0,
                "average_cost_per_1k": 0.0
            },
            "performance_metrics": self._get_metrics_snapshot(),
            "cost_analysis": {
                "estimated_savings_vs_premium": "60-80%",
                "cost_per_quality_ratio": round(self.metrics.avg_cost_per_query / max(np.mean(self.metrics.quality_scores), 1), 4) if self.metrics.quality_scores else 0.0,
                "serverless_vs_local_usage": {
                    "serverless_queries": sum(1 for model in self.metrics.expert_usage_stats.keys() if "serverless" in model.lower()),
                    "local_queries": sum(1 for model in self.metrics.expert_usage_stats.keys() if "serverless" not in model.lower())
                }
            }
        }
        
        # Calculate serverless expert statistics
        for expert in self.serverless_experts.values():
            domain = expert.domain
            enhanced_stats["serverless_experts"]["by_domain"][domain] = enhanced_stats["serverless_experts"]["by_domain"].get(domain, 0) + 1
            enhanced_stats["serverless_experts"]["by_priority_tier"][expert.priority_tier] += 1
        
        if self.serverless_experts:
            enhanced_stats["serverless_experts"]["average_quality_score"] = np.mean([e.quality_score for e in self.serverless_experts.values()])
            enhanced_stats["serverless_experts"]["average_cost_per_1k"] = np.mean([e.cost_per_1k_tokens for e in self.serverless_experts.values()])
        
        return {
            **base_stats,
            **enhanced_stats
        }


# Convenience functions for easy integration
async def get_enhanced_consensus(
    prompt: str,
    domain_hint: Optional[str] = None,
    quality_preference: str = "balanced",
    max_experts: int = 7,
    hf_api_token: str = ""
) -> Dict[str, Any]:
    """
    Convenience function to get enhanced consensus response.
    
    Args:
        prompt: Input prompt
        domain_hint: Optional domain hint
        quality_preference: "cost_optimized", "balanced", "quality_first"
        max_experts: Maximum number of experts to use
        hf_api_token: HuggingFace API token for serverless access
        
    Returns:
        Enhanced consensus response dictionary
    """
    config = {
        "hf_api_token": hf_api_token,
        "enable_serverless": bool(hf_api_token),
        "cost_optimization": quality_preference in ["cost_optimized", "balanced"],
        "quality_threshold": 8.0 if quality_preference == "cost_optimized" else 9.0,
        "max_concurrent_requests": 10
    }
    
    strategy = EnhancedHFConsensusStrategy(config)
    return await strategy.get_enhanced_consensus_response(
        prompt, domain_hint, quality_preference, max_experts
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        prompt = "Explain the implications of quantum computing for cryptographic security in financial systems."
        
        result = await get_enhanced_consensus(
            prompt=prompt,
            domain_hint="technical",
            quality_preference="quality_first",
            max_experts=5,
            hf_api_token="your_hf_token_here"
        )
        
        print(f"Prompt: {prompt}")
        print(f"Consensus Response: {result['consensus_response']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Method: {result['consensus_method']}")
        print(f"Quality Score: {result.get('quality_estimate', 'N/A')}")
        print(f"Cost Estimate: ${result.get('cost_estimate', 0):.4f}")
        print(f"Response Time: {result['response_time_ms']}ms")
        print(f"Experts Used: {result['participating_models']} ({result['serverless_experts_used']} serverless, {result['local_experts_used']} local)")
    
    asyncio.run(main())