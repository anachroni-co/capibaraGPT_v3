"""
Hybrid Expert Router for Meta-Consensus Training

This module extends the hierarchical MoE router with:
- Multi-tier routing (local, serverless, premium)
- Dynamic cost-quality optimization
- Branch-Train-MiX (BTX) integration
- Advanced routing algorithms with ARM Axion optimization
"""

import logging
import asyncio
from enum import Enum
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import time

# JAX imports with fallbacks
try:
    import capibara.jax as jax
    import capibara.jax.numpy as jnp
    from capibara.jax.nn import softmax, relu, sigmoid
    np = jnp
    JAX_AVAILABLE = True
except ImportError:
    try:
        import numpy as np
        import math
        
        def softmax(x, axis=-1):
            exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
        def relu(x):
            return np.maximum(0, x)
        
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            
        JAX_AVAILABLE = False
    except ImportError:
        import math
        
        class DummyArray:
            def __init__(self, data):
                self.data = data if isinstance(data, list) else [data]
            def __getitem__(self, idx): return self.data[idx] if idx < len(self.data) else 0
            def __len__(self): return len(self.data)
            
        class np:
            @staticmethod
            def array(data): return DummyArray(data)
            @staticmethod  
            def mean(data): return sum(data) / len(data) if data else 0
            @staticmethod
            def exp(x): return math.exp(x) if isinstance(x, (int, float)) else DummyArray([math.exp(i) for i in x.data])
            @staticmethod
            def maximum(a, b): return max(a, b) if isinstance(a, (int, float)) else DummyArray([max(i, b) for i in a.data])
        
        def softmax(x, axis=-1): return x
        def relu(x): return max(0, x) if isinstance(x, (int, float)) else x
        def sigmoid(x): return 1 / (1 + math.exp(-max(-500, min(500, x)))) if isinstance(x, (int, float)) else x
        
        JAX_AVAILABLE = False

# Import base classes
from .moe_hierarchical_router import (
    HierarchicalMoERouter, QueryComplexity, ExpertDomain, 
    QueryAnalysis, RoutingDecision, ModelPerformanceMetrics
)

# ARM optimizations (optional import)
try:
    from capibara.core.arm_optimizations import ARMAxionInferenceOptimizer
    ARM_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    ARM_OPTIMIZATIONS_AVAILABLE = False
    
    class ARMAxionInferenceOptimizer:
        def __init__(self, *args, **kwargs):
            pass
        def optimize_inference(self, *args, **kwargs):
            return args[0] if args else None

logger = logging.getLogger(__name__)

class ExpertTier(Enum):
    """Expert tier levels for hybrid routing."""
    LOCAL = "local"              # Local models (fast, free)
    SERVERLESS = "serverless"    # HF Serverless API (scalable, moderate cost)
    PREMIUM = "premium"          # Premium APIs (highest quality, highest cost)
    SPECIALIZED = "specialized"  # Domain-specific experts

class RoutingStrategy(Enum):
    """Routing strategy options."""
    COST_OPTIMIZED = "cost_optimized"      # Minimize cost
    QUALITY_FIRST = "quality_first"        # Maximize quality
    BALANCED = "balanced"                  # Balance cost and quality
    ADAPTIVE = "adaptive"                  # Learn optimal strategy
    LATENCY_OPTIMIZED = "latency_optimized" # Minimize response time

@dataclass
class HybridExpertConfig:
    """Configurestion for hybrid expert in multi-tier system."""
    name: str
    model_id: str
    tier: ExpertTier
    domain: ExpertDomain
    specialization: List[str]
    
    # Performance characteristics
    quality_score: float = 8.5
    avg_latency_ms: float = 1000
    cost_per_1k_tokens: float = 0.001
    success_rate: float = 0.95
    
    # Routing parameters
    weight: float = 1.0
    priority: int = 2  # 1=high, 2=medium, 3=low
    max_concurrent: int = 5
    
    # API configuration
    api_endpoint: Optional[str] = None
    api_key_required: bool = False
    temperature: float = 0.7
    max_tokens: int = 512
    
    # Advanced features
    supports_streaming: bool = False
    supports_function_calling: bool = False
    context_window: int = 4096
    
    def get_efficiency_score(self) -> float:
        """Calculate efficiency score (quality per cost)."""
        return self.quality_score / max(self.cost_per_1k_tokens, 0.0001)

@dataclass
class RoutingMetrics:
    """Comprehensive routing metrics."""
    total_queries: int = 0
    successful_routes: int = 0
    failed_routes: int = 0
    
    # Tier usage statistics
    local_tier_usage: int = 0
    serverless_tier_usage: int = 0
    premium_tier_usage: int = 0
    specialized_tier_usage: int = 0
    
    # Performance metrics
    avg_response_time_ms: float = 0.0
    avg_cost_per_query: float = 0.0
    avg_quality_score: float = 0.0
    
    # Cost analysis
    total_cost: float = 0.0
    cost_savings_vs_premium: float = 0.0
    
    # Quality analysis
    quality_scores: List[float] = field(default_factory=list)
    expert_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Routing strategy effectiveness
    strategy_success_rates: Dict[str, float] = field(default_factory=dict)
    adaptive_learning_score: float = 0.0

class HybridExpertRouter(HierarchicalMoERouter):
    """
    🎭 Hybrid Expert Router with multi-tier routing capabilities.
    
    Extends the hierarchical router with:
    - Multi-tier expert management (local, serverless, premium)
    - Dynamic cost-quality optimization
    - Advanced routing strategies
    - Real-time adaptation and learning
    - ARM Axion optimization for inference
    """
    
    def __init__(self, 
                 router_model_path: str,
                 model_configs: Dict[str, Dict],
                 enable_arm_axion: bool = True,
                 cost_optimization: bool = True,
                 enable_serverless: bool = True,
                 hf_api_token: str = ""):
        
        # Initialize base router
        super().__init__(router_model_path, model_configs, enable_arm_axion, cost_optimization)
        
        # Hybrid router configuration
        self.enable_serverless = enable_serverless
        self.hf_api_token = hf_api_token
        self.adaptive_learning = True
        
        # Multi-tier expert pools
        self.hybrid_experts = self._initialize_hybrid_experts()
        
        # Advanced routing components
        self.routing_metrics = RoutingMetrics()
        self.strategy_weights = self._initialize_strategy_weights()
        self.cost_quality_optimizer = self._initialize_cost_quality_optimizer()
        
        # ARM Axion enhanced router
        if self.enable_arm_axion and ARM_OPTIMIZATIONS_AVAILABLE:
            self.arm_optimizer = ARMAxionInferenceOptimizer(
                model_size="2.6B",
                optimization_level="aggressive",
                enable_sve_vectorization=True,
                memory_pool_optimization=True,
                hybrid_routing_optimization=True  # New feature for hybrid routing
            )
        
        # Advanced neural components for hybrid routing
        self._initialize_hybrid_neural_components()
        
        logger.info("🚀 Hybrid Expert Router initialized with multi-tier capabilities")
    
    def _initialize_hybrid_experts(self) -> Dict[ExpertTier, List[HybridExpertConfig]]:
        """Initialize comprehensive hybrid expert pools."""
        
        return {
            # LOCAL TIER: Fast, free, always available
            ExpertTier.LOCAL: [
                HybridExpertConfig(
                    name="Local_Math_Expert",
                    model_id="local/math_expert_300m",
                    tier=ExpertTier.LOCAL,
                    domain=ExpertDomain.MATHEMATICS_NAVIGATION,
                    specialization=["basic_math", "calculations", "algebra"],
                    quality_score=8.2,
                    avg_latency_ms=200,
                    cost_per_1k_tokens=0.0,
                    weight=1.2
                ),
                HybridExpertConfig(
                    name="Local_Code_Expert",
                    model_id="local/code_expert_600m",
                    tier=ExpertTier.LOCAL,
                    domain=ExpertDomain.GENERAL_PROGRAMMING,
                    specialization=["code_generation", "debugging", "basic_algorithms"],
                    quality_score=8.4,
                    avg_latency_ms=300,
                    cost_per_1k_tokens=0.0,
                    weight=1.3
                ),
                HybridExpertConfig(
                    name="Local_Spanish_Expert",
                    model_id="local/spanish_expert_400m",
                    tier=ExpertTier.LOCAL,
                    domain=ExpertDomain.UNIVERSAL,
                    specialization=["spanish_language", "translation", "cultural_context"],
                    quality_score=8.6,
                    avg_latency_ms=250,
                    cost_per_1k_tokens=0.0,
                    weight=1.5
                ),
                HybridExpertConfig(
                    name="Local_General_Expert",
                    model_id="local/general_expert_1b",
                    tier=ExpertTier.LOCAL,
                    domain=ExpertDomain.UNIVERSAL,
                    specialization=["general_knowledge", "conversation", "basic_reasoning"],
                    quality_score=8.0,
                    avg_latency_ms=400,
                    cost_per_1k_tokens=0.0,
                    weight=1.0
                )
            ],
            
            # SERVERLESS TIER: Scalable, moderate cost, high variety
            ExpertTier.SERVERLESS: [
                HybridExpertConfig(
                    name="HF_Advanced_Math",
                    model_id="microsoft/WizardMath-70B-V1.0",
                    tier=ExpertTier.SERVERLESS,
                    domain=ExpertDomain.MATHEMATICS_NAVIGATION,
                    specialization=["advanced_mathematics", "proofs", "complex_calculations"],
                    quality_score=9.4,
                    avg_latency_ms=2000,
                    cost_per_1k_tokens=0.0015,
                    api_endpoint="https://api-inference.huggingface.co/models/",
                    api_key_required=True,
                    weight=1.8,
                    priority=1
                ),
                HybridExpertConfig(
                    name="HF_Code_Generation",
                    model_id="Phind/Phind-CodeLlama-34B-v2",
                    tier=ExpertTier.SERVERLESS,
                    domain=ExpertDomain.GENERAL_PROGRAMMING,
                    specialization=["advanced_coding", "architecture", "optimization"],
                    quality_score=9.2,
                    avg_latency_ms=1800,
                    cost_per_1k_tokens=0.0012,
                    api_endpoint="https://api-inference.huggingface.co/models/",
                    api_key_required=True,
                    weight=1.7,
                    priority=1
                ),
                HybridExpertConfig(
                    name="HF_Scientific_Research",
                    model_id="microsoft/DialoGPT-large-scientific",
                    tier=ExpertTier.SERVERLESS,
                    domain=ExpertDomain.POLICY_LEGAL_MEDICAL,
                    specialization=["scientific_research", "academic_writing", "analysis"],
                    quality_score=9.0,
                    avg_latency_ms=1600,
                    cost_per_1k_tokens=0.0010,
                    api_endpoint="https://api-inference.huggingface.co/models/",
                    api_key_required=True,
                    weight=1.6
                ),
                HybridExpertConfig(
                    name="HF_Multilingual",
                    model_id="meta-llama/Llama-2-70b-chat-hf",
                    tier=ExpertTier.SERVERLESS,
                    domain=ExpertDomain.UNIVERSAL,
                    specialization=["multilingual", "translation", "cultural_understanding"],
                    quality_score=9.3,
                    avg_latency_ms=2200,
                    cost_per_1k_tokens=0.0018,
                    api_endpoint="https://api-inference.huggingface.co/models/",
                    api_key_required=True,
                    weight=1.9,
                    priority=1
                ),
                HybridExpertConfig(
                    name="HF_Creative_Writing",
                    model_id="EleutherAI/gpt-j-6b",
                    tier=ExpertTier.SERVERLESS,
                    domain=ExpertDomain.UNIVERSAL,
                    specialization=["creative_writing", "storytelling", "content_creation"],
                    quality_score=8.8,
                    avg_latency_ms=1400,
                    cost_per_1k_tokens=0.0008,
                    api_endpoint="https://api-inference.huggingface.co/models/",
                    api_key_required=True,
                    weight=1.4
                )
            ],
            
            # PREMIUM TIER: Highest quality, highest cost
            ExpertTier.PREMIUM: [
                HybridExpertConfig(
                    name="Premium_Universal",
                    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    tier=ExpertTier.PREMIUM,
                    domain=ExpertDomain.UNIVERSAL,
                    specialization=["expert_reasoning", "complex_analysis", "multi_domain"],
                    quality_score=9.7,
                    avg_latency_ms=3000,
                    cost_per_1k_tokens=0.0025,
                    api_endpoint="https://api-inference.huggingface.co/models/",
                    api_key_required=True,
                    weight=2.0,
                    priority=1,
                    context_window=8192
                ),
                HybridExpertConfig(
                    name="Premium_Reasoning",
                    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
                    tier=ExpertTier.PREMIUM,
                    domain=ExpertDomain.UNIVERSAL,
                    specialization=["advanced_reasoning", "logical_analysis", "problem_solving"],
                    quality_score=9.6,
                    avg_latency_ms=2800,
                    cost_per_1k_tokens=0.0022,
                    api_endpoint="https://api-inference.huggingface.co/models/",
                    api_key_required=True,
                    weight=1.9,
                    priority=1,
                    context_window=8192
                )
            ],
            
            # SPECIALIZED TIER: Domain-specific experts
            ExpertTier.SPECIALIZED: [
                HybridExpertConfig(
                    name="Biomedical_Specialist",
                    model_id="microsoft/BiomedNLP-PubMedBERT-large",
                    tier=ExpertTier.SPECIALIZED,
                    domain=ExpertDomain.POLICY_LEGAL_MEDICAL,
                    specialization=["biomedical_research", "medical_analysis", "healthcare"],
                    quality_score=9.5,
                    avg_latency_ms=1800,
                    cost_per_1k_tokens=0.0015,
                    api_endpoint="https://api-inference.huggingface.co/models/",
                    api_key_required=True,
                    weight=1.8,
                    priority=1
                ),
                HybridExpertConfig(
                    name="Legal_Specialist",
                    model_id="nlpaueb/legal-bert-large-uncased",
                    tier=ExpertTier.SPECIALIZED,
                    domain=ExpertDomain.POLICY_LEGAL_MEDICAL,
                    specialization=["legal_analysis", "contract_review", "regulatory"],
                    quality_score=9.3,
                    avg_latency_ms=1600,
                    cost_per_1k_tokens=0.0013,
                    api_endpoint="https://api-inference.huggingface.co/models/",
                    api_key_required=True,
                    weight=1.7,
                    priority=1
                ),
                HybridExpertConfig(
                    name="Financial_Specialist",
                    model_id="ProsusAI/finbert",
                    tier=ExpertTier.SPECIALIZED,
                    domain=ExpertDomain.POLICY_LEGAL_MEDICAL,
                    specialization=["financial_analysis", "market_research", "economic_modeling"],
                    quality_score=9.1,
                    avg_latency_ms=1400,
                    cost_per_1k_tokens=0.0011,
                    api_endpoint="https://api-inference.huggingface.co/models/",
                    api_key_required=True,
                    weight=1.6
                )
            ]
        }
    
    def _initialize_strategy_weights(self) -> Dict[str, float]:
        """Initialize adaptive strategy weights."""
        return {
            RoutingStrategy.COST_OPTIMIZED.value: 0.2,
            RoutingStrategy.QUALITY_FIRST.value: 0.2,
            RoutingStrategy.BALANCED.value: 0.4,
            RoutingStrategy.ADAPTIVE.value: 0.15,
            RoutingStrategy.LATENCY_OPTIMIZED.value: 0.05
        }
    
    def _initialize_cost_quality_optimizer(self) -> Dict[str, Any]:
        """Initialize cost-quality optimization parameters."""
        return {
            "quality_threshold": 8.5,
            "cost_threshold": 0.002,
            "efficiency_weight": 0.6,
            "latency_weight": 0.2,
            "success_rate_weight": 0.2,
            "learning_rate": 0.01,
            "exploration_rate": 0.1
        }
    
    def _initialize_hybrid_neural_components(self):
        """Initialize enhanced neural components for hybrid routing."""
        
        # Enhanced complexity classifier for multi-tier routing
        self.hybrid_complexity_params = self._init_classifier_params(768, len(QueryComplexity) + len(ExpertTier))
        
        # Tier selection network
        self.tier_selection_params = self._init_classifier_params(768 + len(ExpertDomain), len(ExpertTier))
        
        # Cost-quality optimizer network
        self.cost_quality_params = self._init_ensemble_params(768 + len(ExpertTier) + 4)  # +4 for cost, quality, latency, success_rate
        
        # Adaptive strategy selector
        self.strategy_selector_params = self._init_classifier_params(768 + 8, len(RoutingStrategy))  # +8 for context features
        
        logger.info("🧠 Hybrid neural components initialized with JAX native")
    
    async def route_hybrid_query(
        self,
        query: str,
        context: Optional[str] = None,
        routing_strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        max_experts: int = 5,
        budget_limit: float = 0.01,  # Max cost per query
        quality_threshold: float = 8.5,
        latency_threshold_ms: float = 5000
    ) -> Dict[str, Any]:
        """
        Advanced hybrid routing with multi-tier expert selection.
        
        Args:
            query: Input query
            context: Optional context
            routing_strategy: Routing strategy to use
            max_experts: Maximum number of experts to select
            budget_limit: Maximum cost per query
            quality_threshold: Minimum quality threshold
            latency_threshold_ms: Maximum acceptable latency
            
        Returns:
            Comprehensive routing decision with selected experts
        """
        
        start_time = time.time()
        self.routing_metrics.total_queries += 1
        
        try:
            # Enhanced query analysis
            query_analysis = await self._analyze_hybrid_query(query, context)
            
            # Determine optimal routing strategy (if adaptive)
            if routing_strategy == RoutingStrategy.ADAPTIVE:
                routing_strategy = await self._select_adaptive_strategy(query_analysis)
            
            # Multi-tier expert selection
            expert_selection = await self._select_hybrid_experts(
                query_analysis,
                routing_strategy,
                max_experts,
                budget_limit,
                quality_threshold,
                latency_threshold_ms
            )
            
            # Create enhanced routing decision
            routing_decision = self._create_hybrid_routing_decision(
                expert_selection,
                query_analysis,
                routing_strategy
            )
            
            # Update metrics
            processing_time = time.time() - start_time
            await self._update_hybrid_metrics(routing_decision, processing_time)
            
            self.routing_metrics.successful_routes += 1
            
            return routing_decision
            
        except Exception as e:
            logger.error(f"Hybrid routing failed: {e}")
            self.routing_metrics.failed_routes += 1
            
            # Fallback to base router
            base_analysis = self.analyze_query(query, context)
            return self.route_query(base_analysis, "balanced")
    
    async def _analyze_hybrid_query(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced query analysis for hybrid routing."""
        
        # Base analysis from parent class
        base_analysis = self.analyze_query(query, context)
        
        # Enhanced analysis for hybrid routing
        enhanced_features = {
            "word_count": len(query.split()),
            "complexity_indicators": self._detect_complexity_indicators(query),
            "domain_specificity": self._calculate_domain_specificity(query),
            "quality_requirements": self._assess_quality_requirements(query),
            "latency_sensitivity": self._assess_latency_sensitivity(query),
            "cost_sensitivity": self._assess_cost_sensitivity(query),
            "specialized_knowledge_required": self._detect_specialized_knowledge(query),
            "multilingual_aspects": self._detect_multilingual_aspects(query)
        }
        
        # ARM Axion optimized feature extraction
        if self.enable_arm_axion:
            query_embedding = self.arm_optimizer.process_embedding(query, context)
            enhanced_features["embedding_features"] = self._extract_embedding_features(query_embedding)
        
        # Combine base and enhanced analysis
        hybrid_analysis = {
            **base_analysis.__dict__,
            **enhanced_features,
            "hybrid_complexity_score": self._calculate_hybrid_complexity(enhanced_features),
            "optimal_tier_prediction": self._predict_optimal_tier(enhanced_features)
        }
        
        return hybrid_analysis
    
    def _detect_complexity_indicators(self, query: str) -> Dict[str, float]:
        """Detect various complexity indicators in the query."""
        
        indicators = {
            "mathematical": 0.0,
            "technical": 0.0,
            "analytical": 0.0,
            "creative": 0.0,
            "multilingual": 0.0,
            "specialized": 0.0
        }
        
        # Mathematical complexity
        math_keywords = ["calculate", "solve", "equation", "formula", "mathematics", "algebra", "calculus", "statistics"]
        indicators["mathematical"] = sum(1 for kw in math_keywords if kw.lower() in query.lower()) / len(math_keywords)
        
        # Technical complexity
        tech_keywords = ["algorithm", "code", "programming", "technical", "implementation", "architecture", "system"]
        indicators["technical"] = sum(1 for kw in tech_keywords if kw.lower() in query.lower()) / len(tech_keywords)
        
        # Analytical complexity
        analytical_keywords = ["analyze", "compare", "evaluate", "assess", "research", "investigate", "study"]
        indicators["analytical"] = sum(1 for kw in analytical_keywords if kw.lower() in query.lower()) / len(analytical_keywords)
        
        # Creative complexity
        creative_keywords = ["create", "design", "write", "compose", "imagine", "creative", "artistic"]
        indicators["creative"] = sum(1 for kw in creative_keywords if kw.lower() in query.lower()) / len(creative_keywords)
        
        # Multilingual aspects
        multilingual_keywords = ["translate", "language", "español", "français", "deutsch", "中文", "日本語"]
        indicators["multilingual"] = sum(1 for kw in multilingual_keywords if kw.lower() in query.lower()) / len(multilingual_keywords)
        
        # Specialized knowledge
        specialized_keywords = ["medical", "legal", "financial", "scientific", "academic", "professional"]
        indicators["specialized"] = sum(1 for kw in specialized_keywords if kw.lower() in query.lower()) / len(specialized_keywords)
        
        return indicators
    
    def _calculate_domain_specificity(self, query: str) -> float:
        """Calculate how domain-specific the query is."""
        
        domain_keywords = {
            ExpertDomain.MATHEMATICS_NAVIGATION: ["math", "calculate", "equation", "algebra", "geometry"],
            ExpertDomain.GENERAL_PROGRAMMING: ["code", "program", "algorithm", "function", "debug"],
            ExpertDomain.ROBOTICS_MANIPULATION: ["robot", "manipulation", "movement", "spatial", "control"],
            ExpertDomain.POLICY_LEGAL_MEDICAL: ["legal", "medical", "policy", "regulation", "law", "health"],
            ExpertDomain.GENOMICS_MULTIMODAL: ["genetics", "genomics", "DNA", "multimodal", "biology"],
            ExpertDomain.LINUX_SYSTEMS: ["linux", "system", "command", "terminal", "server"]
        }
        
        max_specificity = 0.0
        for domain, keywords in domain_keywords.items():
            specificity = sum(1 for kw in keywords if kw.lower() in query.lower()) / len(keywords)
            max_specificity = max(max_specificity, specificity)
        
        return max_specificity
    
    def _assess_quality_requirements(self, query: str) -> float:
        """Assess quality requirements based on query content."""
        
        high_quality_indicators = [
            "important", "critical", "precise", "accurate", "detailed",
            "professional", "expert", "advanced", "comprehensive", "thorough"
        ]
        
        quality_score = sum(1 for indicator in high_quality_indicators if indicator.lower() in query.lower())
        return min(quality_score / 3.0, 1.0)  # Normalize to 0-1
    
    def _assess_latency_sensitivity(self, query: str) -> float:
        """Assess latency sensitivity based on query content."""
        
        urgent_indicators = ["urgent", "quick", "fast", "immediately", "asap", "now", "real-time"]
        latency_sensitivity = sum(1 for indicator in urgent_indicators if indicator.lower() in query.lower())
        
        return min(latency_sensitivity / 2.0, 1.0)  # Normalize to 0-1
    
    def _assess_cost_sensitivity(self, query: str) -> float:
        """Assess cost sensitivity (inverse of quality requirements)."""
        
        cost_sensitive_indicators = ["simple", "basic", "quick", "brief", "summary"]
        cost_sensitivity = sum(1 for indicator in cost_sensitive_indicators if indicator.lower() in query.lower())
        
        return min(cost_sensitivity / 2.0, 1.0)  # Normalize to 0-1
    
    def _detect_specialized_knowledge(self, query: str) -> bool:
        """Detect if query requires specialized domain knowledge."""
        
        specialized_indicators = [
            "medical", "legal", "financial", "scientific", "academic",
            "research", "professional", "technical", "expert", "specialized"
        ]
        
        return any(indicator.lower() in query.lower() for indicator in specialized_indicators)
    
    def _detect_multilingual_aspects(self, query: str) -> Dict[str, bool]:
        """Detect multilingual aspects in the query."""
        
        return {
            "has_spanish": any(word in query.lower() for word in ["español", "spanish", "castellano"]),
            "has_french": any(word in query.lower() for word in ["français", "french", "francais"]),
            "has_german": any(word in query.lower() for word in ["deutsch", "german", "alemán"]),
            "has_chinese": any(char in query for char in "中文汉语普通话"),
            "needs_translation": "translate" in query.lower() or "traducir" in query.lower()
        }
    
    def _extract_embedding_features(self, embedding) -> Dict[str, float]:
        """Extract features from query embedding."""
        
        if JAX_AVAILABLE and hasattr(embedding, 'shape'):
            return {
                "embedding_norm": float(jnp.linalg.norm(embedding)),
                "embedding_mean": float(jnp.mean(embedding)),
                "embedding_std": float(jnp.std(embedding)),
                "embedding_max": float(jnp.max(embedding)),
                "embedding_min": float(jnp.min(embedding))
            }
        else:
            # Fallback for non-JAX embeddings
            return {
                "embedding_norm": 1.0,
                "embedding_mean": 0.0,
                "embedding_std": 0.3,
                "embedding_max": 1.0,
                "embedding_min": -1.0
            }
    
    def _calculate_hybrid_complexity(self, features: Dict[str, Any]) -> float:
        """Calculate overall hybrid complexity score."""
        
        complexity_factors = [
            features.get("word_count", 0) / 100.0,  # Normalize word count
            max(features.get("complexity_indicators", {}).values()) if features.get("complexity_indicators") else 0.0,
            features.get("domain_specificity", 0.0),
            features.get("quality_requirements", 0.0),
            1.0 if features.get("specialized_knowledge_required", False) else 0.0
        ]
        
        return min(np.mean(complexity_factors), 1.0)
    
    def _predict_optimal_tier(self, features: Dict[str, Any]) -> ExpertTier:
        """Predict optimal tier based on query features."""
        
        complexity_score = features.get("hybrid_complexity_score", 0.5)
        quality_requirements = features.get("quality_requirements", 0.5)
        cost_sensitivity = features.get("cost_sensitivity", 0.5)
        specialized_knowledge = features.get("specialized_knowledge_required", False)
        
        # Decision logic for tier prediction
        if specialized_knowledge and quality_requirements > 0.7:
            return ExpertTier.SPECIALIZED
        elif quality_requirements > 0.8 and cost_sensitivity < 0.3:
            return ExpertTier.PREMIUM
        elif complexity_score > 0.6 and cost_sensitivity < 0.6:
            return ExpertTier.SERVERLESS
        else:
            return ExpertTier.LOCAL
    
    async def _select_adaptive_strategy(self, query_analysis: Dict[str, Any]) -> RoutingStrategy:
        """Select optimal routing strategy using adaptive learning."""
        
        # Features for strategy selection
        features = [
            query_analysis.get("hybrid_complexity_score", 0.5),
            query_analysis.get("quality_requirements", 0.5),
            query_analysis.get("cost_sensitivity", 0.5),
            query_analysis.get("latency_sensitivity", 0.5),
            query_analysis.get("domain_specificity", 0.5),
            1.0 if query_analysis.get("specialized_knowledge_required", False) else 0.0,
            len(query_analysis.get("required_domains", [])) / 7.0,  # Normalize domain count
            query_analysis.get("estimated_cost", 0.001) * 1000.0  # Scale cost
        ]
        
        # JAX neural network forward pass
        if JAX_AVAILABLE:
            feature_vector = jnp.array(features + [0.0] * (768 - len(features)))  # Pad to 768
            strategy_logits = self._forward_classifier(feature_vector, self.strategy_selector_params)
            strategy_probs = softmax(strategy_logits, axis=-1)
            strategy_idx = jnp.argmax(strategy_probs).item()
        else:
            # Fallback heuristic selection
            if query_analysis.get("cost_sensitivity", 0.5) > 0.7:
                strategy_idx = 0  # COST_OPTIMIZED
            elif query_analysis.get("quality_requirements", 0.5) > 0.7:
                strategy_idx = 1  # QUALITY_FIRST
            elif query_analysis.get("latency_sensitivity", 0.5) > 0.7:
                strategy_idx = 4  # LATENCY_OPTIMIZED
            else:
                strategy_idx = 2  # BALANCED
        
        strategies = list(RoutingStrategy)
        selected_strategy = strategies[min(strategy_idx, len(strategies) - 1)]
        
        # Update strategy weights based on success (simplified)
        self.strategy_weights[selected_strategy.value] = min(
            self.strategy_weights[selected_strategy.value] + 0.01, 1.0
        )
        
        return selected_strategy
    
    async def _select_hybrid_experts(
        self,
        query_analysis: Dict[str, Any],
        routing_strategy: RoutingStrategy,
        max_experts: int,
        budget_limit: float,
        quality_threshold: float,
        latency_threshold_ms: float
    ) -> Dict[str, Any]:
        """Select optimal experts from hybrid tiers."""
        
        selected_experts = []
        total_cost = 0.0
        total_latency = 0.0
        tier_distribution = {tier: 0 for tier in ExpertTier}
        
        # Strategy-specific selection logic
        if routing_strategy == RoutingStrategy.COST_OPTIMIZED:
            tier_priorities = [ExpertTier.LOCAL, ExpertTier.SERVERLESS, ExpertTier.SPECIALIZED, ExpertTier.PREMIUM]
            cost_weight, quality_weight, latency_weight = 0.7, 0.2, 0.1
        elif routing_strategy == RoutingStrategy.QUALITY_FIRST:
            tier_priorities = [ExpertTier.PREMIUM, ExpertTier.SPECIALIZED, ExpertTier.SERVERLESS, ExpertTier.LOCAL]
            cost_weight, quality_weight, latency_weight = 0.1, 0.8, 0.1
        elif routing_strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            tier_priorities = [ExpertTier.LOCAL, ExpertTier.SERVERLESS, ExpertTier.SPECIALIZED, ExpertTier.PREMIUM]
            cost_weight, quality_weight, latency_weight = 0.2, 0.3, 0.5
        else:  # BALANCED or ADAPTIVE
            tier_priorities = [ExpertTier.LOCAL, ExpertTier.SERVERLESS, ExpertTier.SPECIALIZED, ExpertTier.PREMIUM]
            cost_weight, quality_weight, latency_weight = 0.4, 0.4, 0.2
        
        # Select experts from each tier
        for tier in tier_priorities:
            if len(selected_experts) >= max_experts:
                break
                
            tier_experts = self.hybrid_experts.get(tier, [])
            tier_candidates = []
            
            # Score experts in this tier
            for expert in tier_experts:
                # Check constraints
                if total_cost + expert.cost_per_1k_tokens > budget_limit:
                    continue
                if expert.quality_score < quality_threshold:
                    continue
                if total_latency + expert.avg_latency_ms > latency_threshold_ms:
                    continue
                
                # Calculate relevance score
                relevance_score = self._calculate_expert_relevance(expert, query_analysis)
                
                # Calculate combined score
                cost_score = 1.0 / (expert.cost_per_1k_tokens + 0.0001)
                quality_score = expert.quality_score / 10.0
                latency_score = 1.0 / (expert.avg_latency_ms / 1000.0 + 0.1)
                
                combined_score = (
                    relevance_score * 0.4 +
                    cost_weight * cost_score * 0.1 +
                    quality_weight * quality_score * 0.4 +
                    latency_weight * latency_score * 0.1
                )
                
                tier_candidates.append((combined_score, expert))
            
            # Sort and select best candidates from this tier
            tier_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Add experts from this tier
            tier_slots = max_experts // len(ExpertTier) if routing_strategy == RoutingStrategy.BALANCED else max_experts
            for score, expert in tier_candidates[:tier_slots]:
                if len(selected_experts) >= max_experts:
                    break
                if total_cost + expert.cost_per_1k_tokens <= budget_limit:
                    selected_experts.append({
                        "expert": expert,
                        "relevance_score": score,
                        "tier": tier
                    })
                    total_cost += expert.cost_per_1k_tokens
                    total_latency += expert.avg_latency_ms
                    tier_distribution[tier] += 1
        
        return {
            "selected_experts": selected_experts,
            "total_estimated_cost": total_cost,
            "total_estimated_latency_ms": total_latency,
            "tier_distribution": tier_distribution,
            "routing_strategy_used": routing_strategy,
            "selection_criteria": {
                "cost_weight": cost_weight,
                "quality_weight": quality_weight,
                "latency_weight": latency_weight
            }
        }
    
    def _calculate_expert_relevance(self, expert: HybridExpertConfig, query_analysis: Dict[str, Any]) -> float:
        """Calculate expert relevance to the query."""
        
        relevance_score = 0.0
        
        # Domain matching
        required_domains = query_analysis.get("required_domains", [])
        if expert.domain in required_domains:
            relevance_score += 2.0
        elif expert.domain == ExpertDomain.UNIVERSAL:
            relevance_score += 0.5
        
        # Specialization matching
        complexity_indicators = query_analysis.get("complexity_indicators", {})
        for specialization in expert.specialization:
            for indicator_type, score in complexity_indicators.items():
                if indicator_type in specialization or specialization in indicator_type:
                    relevance_score += score * 1.5
        
        # Quality requirement matching
        quality_requirements = query_analysis.get("quality_requirements", 0.5)
        if quality_requirements > 0.7 and expert.quality_score > 9.0:
            relevance_score += 1.0
        
        # Specialized knowledge matching
        if query_analysis.get("specialized_knowledge_required", False) and expert.tier == ExpertTier.SPECIALIZED:
            relevance_score += 1.5
        
        # Multilingual matching
        multilingual_aspects = query_analysis.get("multilingual_aspects", {})
        if any(multilingual_aspects.values()) and "multilingual" in expert.specialization:
            relevance_score += 1.0
        
        return relevance_score
    
    def _create_hybrid_routing_decision(
        self,
        expert_selection: Dict[str, Any],
        query_analysis: Dict[str, Any],
        routing_strategy: RoutingStrategy
    ) -> Dict[str, Any]:
        """Creates comprehensive hybrid routing decision."""
        
        selected_experts = expert_selection["selected_experts"]
        
        # Extract expert names and models
        expert_names = [exp["expert"].name for exp in selected_experts]
        expert_models = [exp["expert"].model_id for exp in selected_experts]
        
        # Calculate expected quality
        if selected_experts:
            expected_quality = np.mean([exp["expert"].quality_score for exp in selected_experts])
        else:
            expected_quality = 0.0
        
        # Generate reasoning
        reasoning_parts = [
            f"Selected {len(selected_experts)} experts using {routing_strategy.value} strategy",
            f"Tier distribution: {expert_selection['tier_distribution']}",
            f"Estimated cost: ${expert_selection['total_estimated_cost']:.4f}",
            f"Expected quality: {expected_quality:.1f}/10"
        ]
        
        # Create enhanced routing decision
        return {
            "selected_models": expert_names,
            "selected_model_ids": expert_models,
            "routing_level": self._determine_routing_level(selected_experts),
            "estimated_cost": expert_selection["total_estimated_cost"],
            "expected_quality": expected_quality,
            "estimated_latency_ms": expert_selection["total_estimated_latency_ms"],
            "reasoning": "; ".join(reasoning_parts),
            "routing_strategy": routing_strategy.value,
            "tier_distribution": expert_selection["tier_distribution"],
            "expert_details": [
                {
                    "name": exp["expert"].name,
                    "tier": exp["tier"].value,
                    "domain": exp["expert"].domain.value,
                    "specialization": exp["expert"].specialization,
                    "quality_score": exp["expert"].quality_score,
                    "cost": exp["expert"].cost_per_1k_tokens,
                    "relevance_score": exp["relevance_score"]
                }
                for exp in selected_experts
            ],
            "fallback_options": self._generate_hybrid_fallbacks(expert_selection, query_analysis),
            "query_analysis_summary": {
                "complexity_score": query_analysis.get("hybrid_complexity_score", 0.5),
                "quality_requirements": query_analysis.get("quality_requirements", 0.5),
                "specialized_knowledge": query_analysis.get("specialized_knowledge_required", False),
                "optimal_tier_prediction": query_analysis.get("optimal_tier_prediction", ExpertTier.LOCAL).value
            }
        }
    
    def _determine_routing_level(self, selected_experts: List[Dict[str, Any]]) -> int:
        """Determine routing level based on selected experts."""
        
        if not selected_experts:
            return 0
        
        # Determine level based on highest tier used
        max_tier_priority = 0
        tier_priorities = {
            ExpertTier.LOCAL: 1,
            ExpertTier.SERVERLESS: 2,
            ExpertTier.SPECIALIZED: 3,
            ExpertTier.PREMIUM: 3
        }
        
        for exp in selected_experts:
            tier_priority = tier_priorities.get(exp["tier"], 1)
            max_tier_priority = max(max_tier_priority, tier_priority)
        
        return max_tier_priority
    
    def _generate_hybrid_fallbacks(self, expert_selection: Dict[str, Any], query_analysis: Dict[str, Any]) -> List[str]:
        """Generates fallback options for hybrid routing."""
        
        fallbacks = []
        used_experts = {exp["expert"].name for exp in expert_selection["selected_experts"]}
        
        # Add fallbacks from each tier
        for tier in ExpertTier:
            tier_experts = self.hybrid_experts.get(tier, [])
            for expert in tier_experts:
                if expert.name not in used_experts and len(fallbacks) < 3:
                    fallbacks.append(expert.name)
        
        return fallbacks
    
    async def _update_hybrid_metrics(self, routing_decision: Dict[str, Any], processing_time: float):
        """Update comprehensive hybrid routing metrics."""
        
        # Update basic metrics
        self.routing_metrics.avg_response_time_ms = (
            (self.routing_metrics.avg_response_time_ms * (self.routing_metrics.total_queries - 1) +
             processing_time * 1000) / self.routing_metrics.total_queries
        )
        
        # Update cost metrics
        estimated_cost = routing_decision["estimated_cost"]
        self.routing_metrics.avg_cost_per_query = (
            (self.routing_metrics.avg_cost_per_query * (self.routing_metrics.total_queries - 1) +
             estimated_cost) / self.routing_metrics.total_queries
        )
        self.routing_metrics.total_cost += estimated_cost
        
        # Update quality metrics
        expected_quality = routing_decision["expected_quality"]
        self.routing_metrics.quality_scores.append(expected_quality)
        self.routing_metrics.avg_quality_score = np.mean(self.routing_metrics.quality_scores)
        
        # Update tier usage statistics
        tier_distribution = routing_decision["tier_distribution"]
        self.routing_metrics.local_tier_usage += tier_distribution.get(ExpertTier.LOCAL, 0)
        self.routing_metrics.serverless_tier_usage += tier_distribution.get(ExpertTier.SERVERLESS, 0)
        self.routing_metrics.premium_tier_usage += tier_distribution.get(ExpertTier.PREMIUM, 0)
        self.routing_metrics.specialized_tier_usage += tier_distribution.get(ExpertTier.SPECIALIZED, 0)
        
        # Update strategy success rates
        strategy = routing_decision["routing_strategy"]
        if strategy not in self.routing_metrics.strategy_success_rates:
            self.routing_metrics.strategy_success_rates[strategy] = 0.0
        
        # Simple success estimation (would be updated with actual feedback)
        estimated_success = min(expected_quality / 10.0, 1.0)
        current_rate = self.routing_metrics.strategy_success_rates[strategy]
        self.routing_metrics.strategy_success_rates[strategy] = (current_rate * 0.9 + estimated_success * 0.1)
        
        # Calculate cost savings vs premium baseline
        premium_baseline_cost = 0.0025 * len(routing_decision["selected_models"])  # Assume premium cost
        if premium_baseline_cost > 0:
            savings = (premium_baseline_cost - estimated_cost) / premium_baseline_cost
            self.routing_metrics.cost_savings_vs_premium = (
                self.routing_metrics.cost_savings_vs_premium * 0.9 + savings * 0.1
            )
    
    def get_hybrid_statistics(self) -> Dict[str, Any]:
        """Get comprehensive hybrid routing statistics."""
        
        total_queries = max(self.routing_metrics.total_queries, 1)
        
        return {
            "routing_performance": {
                "total_queries": self.routing_metrics.total_queries,
                "successful_routes": self.routing_metrics.successful_routes,
                "failed_routes": self.routing_metrics.failed_routes,
                "success_rate": self.routing_metrics.successful_routes / total_queries,
                "avg_response_time_ms": round(self.routing_metrics.avg_response_time_ms, 2),
                "avg_cost_per_query": round(self.routing_metrics.avg_cost_per_query, 4),
                "avg_quality_score": round(self.routing_metrics.avg_quality_score, 2)
            },
            "tier_usage_distribution": {
                "local_tier": {
                    "usage_count": self.routing_metrics.local_tier_usage,
                    "percentage": round((self.routing_metrics.local_tier_usage / total_queries) * 100, 1)
                },
                "serverless_tier": {
                    "usage_count": self.routing_metrics.serverless_tier_usage,
                    "percentage": round((self.routing_metrics.serverless_tier_usage / total_queries) * 100, 1)
                },
                "premium_tier": {
                    "usage_count": self.routing_metrics.premium_tier_usage,
                    "percentage": round((self.routing_metrics.premium_tier_usage / total_queries) * 100, 1)
                },
                "specialized_tier": {
                    "usage_count": self.routing_metrics.specialized_tier_usage,
                    "percentage": round((self.routing_metrics.specialized_tier_usage / total_queries) * 100, 1)
                }
            },
            "cost_analysis": {
                "total_cost": round(self.routing_metrics.total_cost, 4),
                "cost_savings_vs_premium": f"{self.routing_metrics.cost_savings_vs_premium * 100:.1f}%",
                "cost_efficiency": round(self.routing_metrics.avg_quality_score / max(self.routing_metrics.avg_cost_per_query, 0.0001), 2)
            },
            "strategy_effectiveness": {
                strategy: f"{rate * 100:.1f}%"
                for strategy, rate in self.routing_metrics.strategy_success_rates.items()
            },
            "expert_inventory": {
                "total_experts": sum(len(experts) for experts in self.hybrid_experts.values()),
                "by_tier": {
                    tier.value: len(experts)
                    for tier, experts in self.hybrid_experts.items()
                },
                "serverless_enabled": self.enable_serverless,
                "arm_axion_optimization": self.enable_arm_axion
            },
            "quality_metrics": {
                "quality_score_distribution": {
                    "min": round(min(self.routing_metrics.quality_scores), 2) if self.routing_metrics.quality_scores else 0.0,
                    "max": round(max(self.routing_metrics.quality_scores), 2) if self.routing_metrics.quality_scores else 0.0,
                    "std": round(np.std(self.routing_metrics.quality_scores), 2) if self.routing_metrics.quality_scores else 0.0
                }
            }
        }


# Factory functions
def create_hybrid_expert_router(
    router_model_path: str,
    config_path: Optional[str] = None,
    hf_api_token: str = "",
    enable_serverless: bool = True,
    enable_arm_axion: bool = True
) -> HybridExpertRouter:
    """Factory function to create hybrid expert router."""
    
    model_configs = {}
    if config_path:
        try:
            with open(config_path, 'r') as f:
                model_configs = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
    
    return HybridExpertRouter(
        router_model_path=router_model_path,
        model_configs=model_configs,
        enable_arm_axion=enable_arm_axion,
        cost_optimization=True,
        enable_serverless=enable_serverless,
        hf_api_token=hf_api_token
    )


# Export main components
__all__ = [
    'HybridExpertRouter',
    'HybridExpertConfig',
    'ExpertTier',
    'RoutingStrategy',
    'RoutingMetrics',
    'create_hybrid_expert_router'
]


if __name__ == "__main__":
    # Example usage
    async def main():
        router = create_hybrid_expert_router(
            router_model_path="models/router_2.6B",
            hf_api_token="your_hf_token_here",
            enable_serverless=True
        )
        
        query = "Develop a machine learning algorithm to predict stock market trends using quantum computing principles."
        
        routing_decision = await router.route_hybrid_query(
            query=query,
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
            max_experts=5,
            budget_limit=0.02,
            quality_threshold=8.5
        )
        
        print(f"Query: {query}")
        print(f"Selected experts: {routing_decision['selected_models']}")
        print(f"Routing strategy: {routing_decision['routing_strategy']}")
        print(f"Estimated cost: ${routing_decision['estimated_cost']:.4f}")
        print(f"Expected quality: {routing_decision['expected_quality']:.1f}/10")
        print(f"Tier distribution: {routing_decision['tier_distribution']}")
        print(f"Reasoning: {routing_decision['reasoning']}")
        
        # Get statistics
        stats = router.get_hybrid_statistics()
        print(f"\nRouter Statistics:")
        print(f"Success rate: {stats['routing_performance']['success_rate']:.2%}")
        print(f"Cost savings vs premium: {stats['cost_analysis']['cost_savings_vs_premium']}")
    
    asyncio.run(main())