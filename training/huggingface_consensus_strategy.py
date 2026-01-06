"""
HuggingFace Consensus Strategy for Capibara6 Training

This module implements a consensus strategy using HuggingFace models and smaller expert models
instead of large commercial models like Gemini 3, DeepSeek, and Mixtral.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class ExpertModelConfig:
    """Configurestion for an expert model in the consensus system."""
    name: str
    model_id: str
    domain: str
    max_length: int = 512
    temperature: float = 0.7
    weight: float = 1.0
    use_local: bool = True
    api_endpoint: Optional[str] = None

class HuggingFaceConsensusStrategy:
    """
    Consensus strategy using HuggingFace models and smaller expert models.
    
    This approach is more cost-effective and efficient than using large commercial models
    like Gemini 3, DeepSeek, and Mixtral in separate VMs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.expert_models = self._load_expert_models()
        self.tokenizers = {}
        self.pipelines = {}
        self._initialize_models()
    
    def _load_expert_models(self) -> Dict[str, ExpertModelConfig]:
        """Load expert model configurations."""
        return {
            "math": ExpertModelConfig(
                name="Math Expert",
                model_id="microsoft/DialoGPT-medium",
                domain="mathematics",
                temperature=0.3,
                weight=1.2
            ),
            "code": ExpertModelConfig(
                name="Code Expert", 
                model_id="Salesforce/codet5-small",
                domain="programming",
                temperature=0.4,
                weight=1.3
            ),
            "spanish": ExpertModelConfig(
                name="Spanish Expert",
                model_id="dccuchile/bert-base-spanish-wwm-cased",
                domain="spanish_language",
                temperature=0.6,
                weight=1.5  # Higher weight for Spanish domain
            ),
            "reasoning": ExpertModelConfig(
                name="Reasoning Expert",
                model_id="EleutherAI/gpt-neo-125M",
                domain="logical_reasoning",
                temperature=0.5,
                weight=1.1
            ),
            "medical": ExpertModelConfig(
                name="Medical Expert",
                model_id="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                domain="medical",
                temperature=0.2,
                weight=1.4
            ),
            "legal": ExpertModelConfig(
                name="Legal Expert",
                model_id="nlpaueb/legal-bert-base-uncased",
                domain="legal",
                temperature=0.3,
                weight=1.2
            ),
            "general": ExpertModelConfig(
                name="General Expert",
                model_id="microsoft/DialoGPT-medium",
                domain="general",
                temperature=0.7,
                weight=1.0
            )
        }
    
    def _initialize_models(self):
        """Initialize all expert models and tokenizers."""
        logger.info("Initializing HuggingFace expert models...")
        
        for domain, config in self.expert_models.items():
            try:
                if config.use_local:
                    # FIXED: Load model locally with TPU/GPU optimization
                    self.tokenizers[domain] = AutoTokenizer.from_pretrained(config.model_id)
                    
                    # Determine best available device
                    if torch.cuda.is_available():
                        device = 0  # Use GPU
                        logger.info(f"🚀 Using GPU for {config.name}")
                    else:
                        # Check for TPU or other accelerators
                        try:
                            import jax
                            if jax.devices("tpu"):
                                device = -1  # Use CPU but prepare for TPU offload
                                logger.info(f"🔥 TPU detected for {config.name}, using optimized CPU")
                            else:
                                device = -1  # Fallback to CPU
                                logger.warning(f"⚠️ No accelerator found for {config.name}, using CPU")
                        except:
                            device = -1
                            logger.warning(f"⚠️ Using CPU for {config.name}")
                    
                    self.pipelines[domain] = pipeline(
                        "text-generation",
                        model=config.model_id,
                        tokenizer=self.tokenizers[domain],
                        device=device,
                        torch_dtype=torch.float16 if device >= 0 else torch.float32,  # Use FP16 on GPU
                        model_kwargs={
                            "low_cpu_mem_usage": True,
                            "torch_dtype": torch.float16 if device >= 0 else torch.float32
                        }
                    )
                    logger.info(f"✅ Loaded {config.name} ({config.model_id})")
                else:
                    # Use API endpoint
                    logger.info(f"🌐 Using API for {config.name}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {config.name}: {e}")
                # Remove failed model
                del self.expert_models[domain]
    
    async def get_consensus_response(
        self, 
        prompt: str, 
        domain_hint: Optional[str] = None,
        max_responses: int = 5
    ) -> Dict[str, Any]:
        """
        Get consensus response from multiple expert models.
        
        Args:
            prompt: Input prompt
            domain_hint: Optional domain hint to prioritize certain experts
            max_responses: Maximum number of responses to generate
            
        Returns:
            Dictionary with consensus response and metadata
        """
        # Select expert models based on domain hint
        selected_models = self._select_expert_models(domain_hint, max_responses)
        
        # Generate responses from all selected models
        responses = await self._generate_responses_async(prompt, selected_models)
        
        # Apply consensus algorithm
        consensus_result = self._apply_consensus_algorithm(responses, prompt)
        
        return consensus_result
    
    def _select_expert_models(self, domain_hint: Optional[str], max_responses: int) -> List[ExpertModelConfig]:
        """Select expert models based on domain hint and weights."""
        if domain_hint and domain_hint in self.expert_models:
            # Prioritize domain-specific expert
            selected = [self.expert_models[domain_hint]]
            # Add other models based on weights
            other_models = [
                config for name, config in self.expert_models.items() 
                if name != domain_hint
            ]
            other_models.sort(key=lambda x: x.weight, reverse=True)
            selected.extend(other_models[:max_responses-1])
        else:
            # Select based on weights
            all_models = list(self.expert_models.values())
            all_models.sort(key=lambda x: x.weight, reverse=True)
            selected = all_models[:max_responses]
        
        return selected
    
    async def _generate_responses_async(
        self, 
        prompt: str, 
        models: List[ExpertModelConfig]
    ) -> List[Dict[str, Any]]:
        """Generates responses from multiple models asynchronously."""
        responses = []
        
        async def generate_single_response(model_config: ExpertModelConfig) -> Dict[str, Any]:
            try:
                if model_config.use_local:
                    # Use local pipeline
                    pipeline = self.pipelines[model_config.domain]
                    result = pipeline(
                        prompt,
                        max_length=model_config.max_length,
                        temperature=model_config.temperature,
                        do_sample=True,
                        pad_token_id=pipeline.tokenizer.eos_token_id
                    )
                    response_text = result[0]['generated_text'][len(prompt):].strip()
                else:
                    # Use API endpoint
                    response_text = await self._call_api_endpoint(
                        model_config.api_endpoint, prompt, model_config
                    )
                
                return {
                    "model": model_config.name,
                    "domain": model_config.domain,
                    "response": response_text,
                    "weight": model_config.weight,
                    "temperature": model_config.temperature,
                    "success": True
                }
                
            except Exception as e:
                logger.error(f"Error generating response from {model_config.name}: {e}")
                return {
                    "model": model_config.name,
                    "domain": model_config.domain,
                    "response": "",
                    "weight": model_config.weight,
                    "temperature": model_config.temperature,
                    "success": False,
                    "error": str(e)
                }
        
        # Generate responses concurrently
        tasks = [generate_single_response(model) for model in models]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_responses = [r for r in responses if isinstance(r, dict) and r.get("success", False)]
        
        return valid_responses
    
    async def _call_api_endpoint(
        self, 
        endpoint: str, 
        prompt: str, 
        config: ExpertModelConfig
    ) -> str:
        """Call external API endpoint for model inference."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "prompt": prompt,
                "max_length": config.max_length,
                "temperature": config.temperature
            }
            
            async with session.post(endpoint, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "")
                else:
                    raise Exception(f"API call failed with status {response.status}")
    
    def _apply_consensus_algorithm(
        self, 
        responses: List[Dict[str, Any]], 
        original_prompt: str
    ) -> Dict[str, Any]:
        """
        Apply consensus algorithm to combine multiple responses.
        
        This implements a weighted voting system with quality scoring.
        """
        if not responses:
            return {
                "consensus_response": "",
                "confidence": 0.0,
                "participating_models": 0,
                "consensus_method": "no_responses"
            }
        
        # Calculate quality scores for each response
        scored_responses = []
        for response in responses:
            quality_score = self._calculate_response_quality(
                response["response"], 
                original_prompt,
                response["domain"]
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
            "semantic_similarity": self._semantic_similarity_consensus(scored_responses),
            "majority_voting": self._majority_voting_consensus(scored_responses)
        }
        
        # Select best consensus method
        best_method = max(consensus_methods.keys(), 
                         key=lambda k: consensus_methods[k]["confidence"])
        
        return {
            "consensus_response": consensus_methods[best_method]["response"],
            "confidence": consensus_methods[best_method]["confidence"],
            "participating_models": len(responses),
            "consensus_method": best_method,
            "all_responses": scored_responses,
            "consensus_breakdown": consensus_methods
        }
    
    def _calculate_response_quality(
        self, 
        response: str, 
        prompt: str, 
        domain: str
    ) -> float:
        """Calculate quality score for a response."""
        if not response.strip():
            return 0.0
        
        # Basic quality metrics
        length_score = min(len(response) / 100, 1.0)  # Prefer longer responses
        relevance_score = self._calculate_relevance(response, prompt)
        coherence_score = self._calculate_coherence(response)
        domain_alignment = self._calculate_domain_alignment(response, domain)
        
        # Weighted combination
        quality_score = (
            0.3 * length_score +
            0.4 * relevance_score +
            0.2 * coherence_score +
            0.1 * domain_alignment
        )
        
        return quality_score
    
    def _calculate_relevance(self, response: str, prompt: str) -> float:
        """Calculate relevance between response and prompt."""
        # Simple keyword overlap (could be improved with embeddings)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        if not prompt_words:
            return 0.0
        
        overlap = len(prompt_words.intersection(response_words))
        return min(overlap / len(prompt_words), 1.0)
    
    def _calculate_coherence(self, response: str) -> float:
        """Calculate coherence of the response."""
        # Simple heuristics for coherence
        sentences = response.split('.')
        if len(sentences) <= 1:
            return 0.5
        
        # Check for logical connectors
        connectors = ['porque', 'sin embargo', 'además', 'también', 'pero', 'aunque']
        connector_count = sum(1 for connector in connectors if connector in response.lower())
        
        return min(connector_count / len(sentences), 1.0)
    
    def _calculate_domain_alignment(self, response: str, domain: str) -> float:
        """Calculate alignment with the expert domain."""
        domain_keywords = {
            "mathematics": ["ecuación", "cálculo", "número", "matemática", "álgebra"],
            "programming": ["código", "función", "variable", "programa", "algoritmo"],
            "spanish_language": ["español", "gramática", "vocabulario", "idioma"],
            "medical": ["médico", "paciente", "tratamiento", "diagnóstico", "síntoma"],
            "legal": ["ley", "legal", "jurídico", "contrato", "derecho"],
            "general": []
        }
        
        keywords = domain_keywords.get(domain, [])
        if not keywords:
            return 0.5  # Neutral score for general domain
        
        keyword_matches = sum(1 for keyword in keywords if keyword in response.lower())
        return min(keyword_matches / len(keywords), 1.0)
    
    def _weighted_voting_consensus(self, scored_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply weighted voting consensus."""
        total_weight = sum(r["final_score"] for r in scored_responses)
        
        if total_weight == 0:
            return {"response": "", "confidence": 0.0}
        
        # Select response with highest weighted score
        best_response = max(scored_responses, key=lambda x: x["final_score"])
        
        confidence = best_response["final_score"] / total_weight
        
        return {
            "response": best_response["response"],
            "confidence": confidence
        }
    
    def _semantic_similarity_consensus(self, scored_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply semantic similarity consensus."""
        # For now, use simple similarity based on common words
        # In production, you'd use sentence embeddings
        
        if len(scored_responses) <= 1:
            return {
                "response": scored_responses[0]["response"] if scored_responses else "",
                "confidence": 1.0
            }
        
        # Calculate pairwise similarities
        similarities = []
        for i, resp1 in enumerate(scored_responses):
            for j, resp2 in enumerate(scored_responses[i+1:], i+1):
                similarity = self._calculate_text_similarity(
                    resp1["response"], resp2["response"]
                )
                similarities.append(similarity)
        
        if not similarities:
            return {"response": "", "confidence": 0.0}
        
        avg_similarity = np.mean(similarities)
        
        # Select response closest to the centroid
        best_response = min(scored_responses, 
                           key=lambda x: self._calculate_centroid_distance(x, scored_responses))
        
        return {
            "response": best_response["response"],
            "confidence": avg_similarity
        }
    
    def _majority_voting_consensus(self, scored_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply majority voting consensus."""
        # Group similar responses
        response_groups = []
        used_indices = set()
        
        for i, resp1 in enumerate(scored_responses):
            if i in used_indices:
                continue
                
            group = [resp1]
            used_indices.add(i)
            
            for j, resp2 in enumerate(scored_responses[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                similarity = self._calculate_text_similarity(
                    resp1["response"], resp2["response"]
                )
                
                if similarity > 0.7:  # Threshold for grouping
                    group.append(resp2)
                    used_indices.add(j)
            
            response_groups.append(group)
        
        # Find largest group
        largest_group = max(response_groups, key=len)
        
        # Select best response from largest group
        best_response = max(largest_group, key=lambda x: x["final_score"])
        
        confidence = len(largest_group) / len(scored_responses)
        
        return {
            "response": best_response["response"],
            "confidence": confidence
        }
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_centroid_distance(self, response: Dict[str, Any], all_responses: List[Dict[str, Any]]) -> float:
        """Calculate distance from response to centroid of all responses."""
        similarities = [
            self._calculate_text_similarity(response["response"], other["response"])
            for other in all_responses if other != response
        ]
        
        return 1.0 - np.mean(similarities) if similarities else 1.0
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded models."""
        return {
            "total_models": len(self.expert_models),
            "loaded_models": len(self.pipelines),
            "domains": list(self.expert_models.keys()),
            "model_details": {
                domain: {
                    "name": config.name,
                    "model_id": config.model_id,
                    "weight": config.weight,
                    "loaded": domain in self.pipelines
                }
                for domain, config in self.expert_models.items()
            }
        }

# Convenience function for easy integration
async def get_huggingface_consensus(
    prompt: str,
    domain_hint: Optional[str] = None,
    max_responses: int = 5
) -> Dict[str, Any]:
    """
    Convenience function to get consensus response using HuggingFace models.
    
    Args:
        prompt: Input prompt
        domain_hint: Optional domain hint
        max_responses: Maximum number of responses
        
    Returns:
        Consensus response dictionary
    """
    config = {
        "enable_local_models": True,
        "enable_api_models": False,
        "max_concurrent_requests": 5
    }
    
    strategy = HuggingFaceConsensusStrategy(config)
    return await strategy.get_consensus_response(prompt, domain_hint, max_responses)

if __name__ == "__main__":
    # Example usage
    async def main():
        prompt = "¿Cuál es la capital de España?"
        
        result = await get_huggingface_consensus(
            prompt=prompt,
            domain_hint="spanish",
            max_responses=3
        )
        
        print(f"Prompt: {prompt}")
        print(f"Consensus Response: {result['consensus_response']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Method: {result['consensus_method']}")
        print(f"Models used: {result['participating_models']}")
    
    asyncio.run(main())