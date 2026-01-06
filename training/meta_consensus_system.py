"""
Meta-Consensus System Orchestrator

This is the main orchestrator that integrates all meta-consensus components:
- Enhanced HuggingFace Consensus Strategy
- Hybrid Expert Router
- BTX Training System
- Advanced consensus algorithms
- Real-time optimization and adaptation
"""

import logging
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import json
import numpy as np

# Import meta-consensus components
from .enhanced_hf_consensus_strategy import EnhancedHFConsensusStrategy, ServerlessExpertConfig
from .hybrid_expert_router import HybridExpertRouter, ExpertTier, RoutingStrategy, HybridExpertConfig
from .btx_training_system import BTXTrainingSystem, BTXExpertConfig, BTXStage
from .unified_consensus import UnifiedConsensusStrategy, ConsensusConfig

logger = logging.getLogger(__name__)

class ConsensusMode(Enum):
    """Meta-consensus operation modes."""
    LOCAL_ONLY = "local_only"                    # Use only local experts
    HYBRID = "hybrid"                            # Mix local and serverless
    SERVERLESS_FIRST = "serverless_first"        # Prefer serverless experts
    ADAPTIVE = "adaptive"                        # Learn optimal strategy
    COST_OPTIMIZED = "cost_optimized"           # Minimize costs
    QUALITY_FIRST = "quality_first"             # Maximize quality
    LATENCY_OPTIMIZED = "latency_optimized"     # Minimize response time

class SystemState(Enum):
    """Meta-consensus system states."""
    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    OPTIMIZING = "optimizing"
    SERVING = "serving"
    MAINTENANCE = "maintenance"
    ERROR = "error"

@dataclass
class MetaConsensusConfig:
    """Configurestion for meta-consensus system."""
    
    # Core system settings
    system_name: str = "MetaConsensus-v1"
    enable_serverless: bool = True
    enable_btx_training: bool = True
    enable_adaptive_routing: bool = True
    
    # API and authentication
    hf_api_token: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    
    # Resource limits
    max_concurrent_experts: int = 10
    max_cost_per_query: float = 0.05
    max_latency_ms: float = 10000
    memory_limit_gb: float = 32.0
    
    # Quality thresholds
    min_consensus_confidence: float = 0.7
    min_expert_quality: float = 8.0
    target_consensus_quality: float = 9.0
    
    # Optimization settings
    enable_continuous_learning: bool = True
    adaptation_learning_rate: float = 0.01
    quality_feedback_weight: float = 0.6
    cost_feedback_weight: float = 0.3
    latency_feedback_weight: float = 0.1
    
    # Storage and persistence
    model_cache_dir: str = "cache/models"
    metrics_storage_dir: str = "storage/metrics"
    checkpoint_interval_minutes: int = 30
    
    # Advanced features
    enable_explanation_generation: bool = True
    enable_uncertainty_estimation: bool = True
    enable_bias_detection: bool = True
    enable_safety_filtering: bool = True

@dataclass
class QueryContext:
    """Context information for query processing."""
    query_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: int = 2  # 1=high, 2=medium, 3=low
    
    # Query characteristics
    domain_hint: Optional[str] = None
    complexity_estimate: float = 0.5
    quality_requirement: float = 0.8
    latency_requirement_ms: float = 5000
    cost_limit: float = 0.02
    
    # Context data
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConsensusResult:
    """Comprehensive consensus result."""
    query_id: str
    response: str
    confidence: float
    quality_score: float
    
    # Expert information
    participating_experts: List[str]
    expert_responses: List[Dict[str, Any]]
    routing_decision: Dict[str, Any]
    consensus_method: str
    
    # Performance metrics
    response_time_ms: float
    total_cost: float
    tokens_generated: int
    
    # Quality indicators
    uncertainty_score: float = 0.0
    bias_score: float = 0.0
    safety_score: float = 1.0
    explanation: str = ""
    
    # System information
    consensus_mode: ConsensusMode = ConsensusMode.HYBRID
    system_state: SystemState = SystemState.SERVING
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SystemMetrics:
    """Comprehensive system metrics."""
    
    # Query processing metrics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_response_time_ms: float = 0.0
    avg_quality_score: float = 0.0
    avg_cost_per_query: float = 0.0
    
    # Expert utilization
    local_expert_usage: int = 0
    serverless_expert_usage: int = 0
    premium_expert_usage: int = 0
    
    # Consensus performance
    consensus_success_rate: float = 0.0
    avg_consensus_confidence: float = 0.0
    consensus_method_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Cost analysis
    total_cost: float = 0.0
    cost_savings_vs_premium: float = 0.0
    cost_efficiency_score: float = 0.0
    
    # Quality tracking
    quality_trend: List[float] = field(default_factory=list)
    user_satisfaction_scores: List[float] = field(default_factory=list)
    
    # System health
    system_uptime_hours: float = 0.0
    error_rate: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)

class MetaConsensusSystem:
    """
    🎭 Meta-Consensus System Orchestrator
    
    The main orchestrator that integrates all meta-consensus components:
    - Enhanced HuggingFace Consensus Strategy with serverless API
    - Hybrid Expert Router with multi-tier routing
    - BTX Training System for expert development
    - Advanced consensus algorithms and optimization
    - Real-time adaptation and continuous learning
    """
    
    def __init__(self, config: MetaConsensusConfig):
        self.config = config
        self.state = SystemState.INITIALIZING
        self.metrics = SystemMetrics()
        self.start_time = datetime.now()
        
        # Core components
        self.enhanced_consensus = None
        self.hybrid_router = None
        self.btx_trainer = None
        self.unified_consensus = None
        
        # System state
        self.active_queries: Dict[str, QueryContext] = {}
        self.query_history: List[ConsensusResult] = []
        self.adaptation_weights: Dict[str, float] = {}
        
        # Performance tracking
        self.performance_history: List[Dict[str, float]] = []
        self.cost_tracking: Dict[str, float] = {}
        self.quality_tracking: Dict[str, float] = {}
        
        logger.info(f"🎭 MetaConsensusSystem '{config.system_name}' initializing...")
    
    async def initialize(self) -> bool:
        """Initialize all meta-consensus components."""
        
        try:
            logger.info("🚀 Initializing Meta-Consensus System components...")
            
            # Initialize Enhanced HF Consensus Strategy
            if self.config.enable_serverless and self.config.hf_api_token:
                await self._initialize_enhanced_consensus()
            
            # Initialize Hybrid Expert Router
            await self._initialize_hybrid_router()
            
            # Initialize BTX Training System
            if self.config.enable_btx_training:
                await self._initialize_btx_trainer()
            
            # Initialize Unified Consensus (fallback)
            await self._initialize_unified_consensus()
            
            # Initialize adaptation system
            if self.config.enable_continuous_learning:
                await self._initialize_adaptation_system()
            
            # Setup monitoring and metrics
            await self._setup_monitoring()
            
            self.state = SystemState.READY
            logger.info("✅ Meta-Consensus System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Meta-Consensus System initialization failed: {e}")
            self.state = SystemState.ERROR
            return False
    
    async def process_query(
        self,
        query: str,
        context: Optional[QueryContext] = None,
        consensus_mode: ConsensusMode = ConsensusMode.ADAPTIVE
    ) -> ConsensusResult:
        """
        Process a query using the meta-consensus system.
        
        Args:
            query: Input query
            context: Query context and preferences
            consensus_mode: Consensus operation mode
            
        Returns:
            Comprehensive consensus result
        """
        
        start_time = time.time()
        
        # Create context if not provided
        if context is None:
            context = QueryContext(
                query_id=f"query_{int(time.time() * 1000)}",
                quality_requirement=0.8,
                latency_requirement_ms=5000,
                cost_limit=self.config.max_cost_per_query
            )
        
        self.active_queries[context.query_id] = context
        self.metrics.total_queries += 1
        
        try:
            logger.info(f"🔍 Processing query {context.query_id} in {consensus_mode.value} mode")
            
            # Analyze query and determine optimal strategy
            query_analysis = await self._analyze_query_comprehensive(query, context)
            
            # Select consensus strategy based on mode and analysis
            consensus_strategy = await self._select_consensus_strategy(
                query_analysis, consensus_mode, context
            )
            
            # Execute consensus strategy
            consensus_result = await self._execute_consensus_strategy(
                query, context, query_analysis, consensus_strategy
            )
            
            # Post-process and enhance result
            enhanced_result = await self._post_process_result(
                consensus_result, query_analysis, context
            )
            
            # Update metrics and learning
            processing_time = time.time() - start_time
            await self._update_metrics_and_learning(enhanced_result, processing_time, context)
            
            # Clean up
            del self.active_queries[context.query_id]
            self.query_history.append(enhanced_result)
            
            self.metrics.successful_queries += 1
            logger.info(f"✅ Query {context.query_id} processed successfully")
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"❌ Query {context.query_id} processing failed: {e}")
            self.metrics.failed_queries += 1
            
            # Create error result
            error_result = ConsensusResult(
                query_id=context.query_id,
                response=f"I apologize, but I encountered an error processing your query: {str(e)}",
                confidence=0.0,
                quality_score=0.0,
                participating_experts=[],
                expert_responses=[],
                routing_decision={"error": str(e)},
                consensus_method="error_fallback",
                response_time_ms=(time.time() - start_time) * 1000,
                total_cost=0.0,
                tokens_generated=0,
                consensus_mode=consensus_mode,
                system_state=self.state
            )
            
            if context.query_id in self.active_queries:
                del self.active_queries[context.query_id]
            
            return error_result
    
    async def _initialize_enhanced_consensus(self):
        """Initialize Enhanced HuggingFace Consensus Strategy."""
        
        logger.info("Initializing Enhanced HF Consensus Strategy...")
        
        hf_config = {
            "hf_api_token": self.config.hf_api_token,
            "enable_serverless": self.config.enable_serverless,
            "cost_optimization": True,
            "quality_threshold": self.config.min_expert_quality,
            "max_concurrent_requests": self.config.max_concurrent_experts
        }
        
        self.enhanced_consensus = EnhancedHFConsensusStrategy(hf_config)
        logger.info("✅ Enhanced HF Consensus Strategy initialized")
    
    async def _initialize_hybrid_router(self):
        """Initialize Hybrid Expert Router."""
        
        logger.info("Initializing Hybrid Expert Router...")
        
        # Mock router model path - in real implementation, provide actual path
        router_model_path = "models/router_2.6B"
        model_configs = {}
        
        self.hybrid_router = HybridExpertRouter(
            router_model_path=router_model_path,
            model_configs=model_configs,
            enable_arm_axion=True,
            cost_optimization=True,
            enable_serverless=self.config.enable_serverless,
            hf_api_token=self.config.hf_api_token
        )
        
        logger.info("✅ Hybrid Expert Router initialized")
    
    async def _initialize_btx_trainer(self):
        """Initialize BTX Training System."""
        
        logger.info("Initializing BTX Training System...")
        
        # Mock configuration - in real implementation, provide actual paths and configs
        seed_model_path = "models/seed_model_1b"
        output_base_path = "output/btx_training"
        expert_configs = []  # Would be populated with actual expert configs
        
        self.btx_trainer = BTXTrainingSystem(
            seed_model_path=seed_model_path,
            output_base_path=output_base_path,
            expert_configs=expert_configs,
            max_parallel_jobs=4
        )
        
        logger.info("✅ BTX Training System initialized")
    
    async def _initialize_unified_consensus(self):
        """Initialize Unified Consensus Strategy (fallback)."""
        
        logger.info("Initializing Unified Consensus Strategy...")
        
        consensus_config = ConsensusConfig(
            consensus_threshold=self.config.min_consensus_confidence,
            validation_patience=5,
            consensus_window=10,
            min_agreement_ratio=0.7
        )
        
        self.unified_consensus = UnifiedConsensusStrategy(consensus_config)
        logger.info("✅ Unified Consensus Strategy initialized")
    
    async def _initialize_adaptation_system(self):
        """Initialize adaptive learning system."""
        
        logger.info("Initializing Adaptation System...")
        
        # Initialize adaptation weights
        self.adaptation_weights = {
            "quality_weight": self.config.quality_feedback_weight,
            "cost_weight": self.config.cost_feedback_weight,
            "latency_weight": self.config.latency_feedback_weight,
            "consensus_mode_preferences": {mode.value: 1.0 for mode in ConsensusMode},
            "expert_preferences": {},
            "routing_strategy_preferences": {strategy.value: 1.0 for strategy in RoutingStrategy}
        }
        
        logger.info("✅ Adaptation System initialized")
    
    async def _setup_monitoring(self):
        """Setup monitoring and metrics collection."""
        
        logger.info("Setting up monitoring and metrics...")
        
        # Create storage directories
        Path(self.config.model_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.metrics_storage_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics.system_uptime_hours = 0.0
        self.metrics.resource_utilization = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "gpu_percent": 0.0,
            "disk_percent": 0.0
        }
        
        logger.info("✅ Monitoring and metrics setup completed")
    
    async def _analyze_query_comprehensive(self, query: str, context: QueryContext) -> Dict[str, Any]:
        """Comprehensive query analysis for optimal routing."""
        
        # Basic query characteristics
        word_count = len(query.split())
        char_count = len(query)
        
        # Complexity estimation
        complexity_indicators = {
            "length_complexity": min(word_count / 50.0, 1.0),
            "technical_complexity": self._detect_technical_complexity(query),
            "domain_specificity": self._detect_domain_specificity(query),
            "reasoning_complexity": self._detect_reasoning_complexity(query),
            "multilingual_complexity": self._detect_multilingual_aspects(query)
        }
        
        overall_complexity = np.mean(list(complexity_indicators.values()))
        
        # Quality requirements
        quality_indicators = {
            "explicit_quality_request": self._detect_quality_requests(query),
            "critical_domain": self._detect_critical_domains(query),
            "user_preference": context.quality_requirement,
            "complexity_based": overall_complexity
        }
        
        quality_requirement = max(quality_indicators.values())
        
        # Cost sensitivity
        cost_indicators = {
            "simple_query": 1.0 - overall_complexity,
            "user_cost_limit": 1.0 - (context.cost_limit / self.config.max_cost_per_query),
            "quick_request": self._detect_quick_requests(query)
        }
        
        cost_sensitivity = max(cost_indicators.values())
        
        # Latency sensitivity
        latency_indicators = {
            "urgent_request": self._detect_urgent_requests(query),
            "real_time_requirement": self._detect_realtime_requirements(query),
            "user_latency_limit": 1.0 - (context.latency_requirement_ms / self.config.max_latency_ms)
        }
        
        latency_sensitivity = max(latency_indicators.values())
        
        return {
            "query_characteristics": {
                "word_count": word_count,
                "char_count": char_count,
                "complexity_score": overall_complexity,
                "complexity_indicators": complexity_indicators
            },
            "requirements": {
                "quality_requirement": quality_requirement,
                "cost_sensitivity": cost_sensitivity,
                "latency_sensitivity": latency_sensitivity
            },
            "domain_analysis": {
                "detected_domains": self._detect_domains(query),
                "domain_confidence": self._calculate_domain_confidence(query),
                "specialized_knowledge_required": self._requires_specialized_knowledge(query)
            },
            "optimal_strategy_prediction": self._predict_optimal_strategy(
                overall_complexity, quality_requirement, cost_sensitivity, latency_sensitivity
            )
        }
    
    def _detect_technical_complexity(self, query: str) -> float:
        """Detect technical complexity in query."""
        technical_keywords = [
            "algorithm", "implementation", "optimization", "architecture",
            "framework", "protocol", "specification", "configuration"
        ]
        return min(sum(1 for kw in technical_keywords if kw.lower() in query.lower()) / 3.0, 1.0)
    
    def _detect_domain_specificity(self, query: str) -> float:
        """Detect domain specificity."""
        domain_keywords = {
            "medical": ["medical", "health", "diagnosis", "treatment", "patient"],
            "legal": ["legal", "law", "contract", "regulation", "compliance"],
            "financial": ["financial", "investment", "market", "economic", "trading"],
            "scientific": ["research", "hypothesis", "experiment", "analysis", "study"],
            "technical": ["technical", "engineering", "system", "design", "development"]
        }
        
        max_specificity = 0.0
        for domain, keywords in domain_keywords.items():
            specificity = sum(1 for kw in keywords if kw.lower() in query.lower()) / len(keywords)
            max_specificity = max(max_specificity, specificity)
        
        return max_specificity
    
    def _detect_reasoning_complexity(self, query: str) -> float:
        """Detect reasoning complexity."""
        reasoning_keywords = [
            "analyze", "compare", "evaluate", "explain", "prove",
            "derive", "calculate", "solve", "optimize", "design"
        ]
        return min(sum(1 for kw in reasoning_keywords if kw.lower() in query.lower()) / 3.0, 1.0)
    
    def _detect_multilingual_aspects(self, query: str) -> float:
        """Detect multilingual aspects."""
        multilingual_keywords = ["translate", "translation", "language", "idioma", "traducir"]
        non_english_chars = sum(1 for char in query if ord(char) > 127)
        
        keyword_score = min(sum(1 for kw in multilingual_keywords if kw.lower() in query.lower()) / 2.0, 1.0)
        char_score = min(non_english_chars / len(query), 1.0)
        
        return max(keyword_score, char_score)
    
    def _detect_quality_requests(self, query: str) -> float:
        """Detect explicit quality requests."""
        quality_keywords = [
            "accurate", "precise", "detailed", "comprehensive", "thorough",
            "expert", "professional", "high-quality", "best"
        ]
        return min(sum(1 for kw in quality_keywords if kw.lower() in query.lower()) / 3.0, 1.0)
    
    def _detect_critical_domains(self, query: str) -> float:
        """Detect critical domains requiring high quality."""
        critical_keywords = [
            "medical", "legal", "financial", "safety", "security",
            "critical", "important", "urgent", "emergency"
        ]
        return min(sum(1 for kw in critical_keywords if kw.lower() in query.lower()) / 2.0, 1.0)
    
    def _detect_quick_requests(self, query: str) -> float:
        """Detect requests for quick/simple responses."""
        quick_keywords = ["quick", "simple", "brief", "summary", "short", "fast"]
        return min(sum(1 for kw in quick_keywords if kw.lower() in query.lower()) / 2.0, 1.0)
    
    def _detect_urgent_requests(self, query: str) -> float:
        """Detect urgent requests."""
        urgent_keywords = ["urgent", "asap", "immediately", "now", "quickly", "fast"]
        return min(sum(1 for kw in urgent_keywords if kw.lower() in query.lower()) / 2.0, 1.0)
    
    def _detect_realtime_requirements(self, query: str) -> float:
        """Detect real-time requirements."""
        realtime_keywords = ["real-time", "live", "instant", "immediate", "now"]
        return min(sum(1 for kw in realtime_keywords if kw.lower() in query.lower()) / 2.0, 1.0)
    
    def _detect_domains(self, query: str) -> List[str]:
        """Detect relevant domains."""
        domain_keywords = {
            "mathematics": ["math", "calculate", "equation", "algebra", "geometry"],
            "programming": ["code", "program", "algorithm", "function", "debug"],
            "science": ["research", "hypothesis", "experiment", "analysis", "study"],
            "language": ["translate", "language", "grammar", "vocabulary", "linguistics"],
            "creative": ["write", "create", "design", "compose", "artistic"]
        }
        
        detected_domains = []
        for domain, keywords in domain_keywords.items():
            if any(kw.lower() in query.lower() for kw in keywords):
                detected_domains.append(domain)
        
        return detected_domains
    
    def _calculate_domain_confidence(self, query: str) -> float:
        """Calculate confidence in domain detection."""
        domains = self._detect_domains(query)
        return min(len(domains) / 3.0, 1.0)
    
    def _requires_specialized_knowledge(self, query: str) -> bool:
        """Determine if query requires specialized knowledge."""
        specialized_keywords = [
            "medical", "legal", "financial", "scientific", "technical",
            "expert", "professional", "specialized", "advanced"
        ]
        return any(kw.lower() in query.lower() for kw in specialized_keywords)
    
    def _predict_optimal_strategy(self, complexity: float, quality: float, 
                                cost_sensitivity: float, latency_sensitivity: float) -> str:
        """Predict optimal consensus strategy."""
        
        if latency_sensitivity > 0.8:
            return "local_first"
        elif quality > 0.8 and cost_sensitivity < 0.3:
            return "quality_first"
        elif cost_sensitivity > 0.7:
            return "cost_optimized"
        elif complexity > 0.7:
            return "hybrid_premium"
        else:
            return "balanced_hybrid"
    
    async def _select_consensus_strategy(
        self,
        query_analysis: Dict[str, Any],
        consensus_mode: ConsensusMode,
        context: QueryContext
    ) -> Dict[str, Any]:
        """Select optimal consensus strategy."""
        
        if consensus_mode == ConsensusMode.ADAPTIVE:
            # Use adaptive learning to select strategy
            strategy_scores = {}
            for mode in ConsensusMode:
                if mode == ConsensusMode.ADAPTIVE:
                    continue
                
                score = self._calculate_strategy_score(mode, query_analysis, context)
                strategy_scores[mode] = score
            
            # Select best strategy
            best_mode = max(strategy_scores.keys(), key=lambda k: strategy_scores[k])
            consensus_mode = best_mode
        
        # Configure strategy based on selected mode
        strategy_config = {
            "mode": consensus_mode,
            "max_experts": min(self.config.max_concurrent_experts, 7),
            "quality_threshold": self.config.min_expert_quality,
            "cost_limit": context.cost_limit,
            "latency_limit_ms": context.latency_requirement_ms
        }
        
        if consensus_mode == ConsensusMode.LOCAL_ONLY:
            strategy_config.update({
                "use_serverless": False,
                "use_premium": False,
                "routing_strategy": RoutingStrategy.LATENCY_OPTIMIZED
            })
        elif consensus_mode == ConsensusMode.SERVERLESS_FIRST:
            strategy_config.update({
                "use_serverless": True,
                "prefer_serverless": True,
                "routing_strategy": RoutingStrategy.QUALITY_FIRST
            })
        elif consensus_mode == ConsensusMode.QUALITY_FIRST:
            strategy_config.update({
                "use_serverless": True,
                "use_premium": True,
                "routing_strategy": RoutingStrategy.QUALITY_FIRST,
                "quality_threshold": 9.0
            })
        elif consensus_mode == ConsensusMode.COST_OPTIMIZED:
            strategy_config.update({
                "use_serverless": False,
                "routing_strategy": RoutingStrategy.COST_OPTIMIZED,
                "max_experts": 3
            })
        elif consensus_mode == ConsensusMode.LATENCY_OPTIMIZED:
            strategy_config.update({
                "use_serverless": False,
                "routing_strategy": RoutingStrategy.LATENCY_OPTIMIZED,
                "max_experts": 2
            })
        else:  # HYBRID
            strategy_config.update({
                "use_serverless": True,
                "routing_strategy": RoutingStrategy.BALANCED
            })
        
        return strategy_config
    
    def _calculate_strategy_score(
        self,
        mode: ConsensusMode,
        query_analysis: Dict[str, Any],
        context: QueryContext
    ) -> float:
        """Calculate score for a consensus strategy."""
        
        requirements = query_analysis["requirements"]
        
        # Base scores for each mode
        mode_scores = {
            ConsensusMode.LOCAL_ONLY: 0.6,
            ConsensusMode.HYBRID: 0.8,
            ConsensusMode.SERVERLESS_FIRST: 0.7,
            ConsensusMode.COST_OPTIMIZED: 0.5,
            ConsensusMode.QUALITY_FIRST: 0.9,
            ConsensusMode.LATENCY_OPTIMIZED: 0.7
        }
        
        base_score = mode_scores.get(mode, 0.5)
        
        # Adjust based on requirements
        if mode == ConsensusMode.QUALITY_FIRST and requirements["quality_requirement"] > 0.8:
            base_score += 0.2
        elif mode == ConsensusMode.COST_OPTIMIZED and requirements["cost_sensitivity"] > 0.7:
            base_score += 0.2
        elif mode == ConsensusMode.LATENCY_OPTIMIZED and requirements["latency_sensitivity"] > 0.7:
            base_score += 0.2
        
        # Apply adaptive weights
        adaptation_weight = self.adaptation_weights["consensus_mode_preferences"].get(mode.value, 1.0)
        
        return base_score * adaptation_weight
    
    async def _execute_consensus_strategy(
        self,
        query: str,
        context: QueryContext,
        query_analysis: Dict[str, Any],
        strategy_config: Dict[str, Any]
    ) -> ConsensusResult:
        """Execute the selected consensus strategy."""
        
        mode = strategy_config["mode"]
        
        if mode in [ConsensusMode.SERVERLESS_FIRST, ConsensusMode.HYBRID, ConsensusMode.QUALITY_FIRST]:
            # Use enhanced HF consensus strategy
            if self.enhanced_consensus:
                result = await self._execute_enhanced_consensus(query, context, strategy_config)
            else:
                result = await self._execute_hybrid_routing(query, context, strategy_config)
        elif mode in [ConsensusMode.LOCAL_ONLY, ConsensusMode.LATENCY_OPTIMIZED]:
            # Use hybrid router with local-only configuration
            result = await self._execute_hybrid_routing(query, context, strategy_config)
        else:
            # Fallback to unified consensus
            result = await self._execute_unified_consensus(query, context, strategy_config)
        
        return result
    
    async def _execute_enhanced_consensus(
        self,
        query: str,
        context: QueryContext,
        strategy_config: Dict[str, Any]
    ) -> ConsensusResult:
        """Execute enhanced HF consensus strategy."""
        
        quality_preference = "quality_first" if strategy_config["mode"] == ConsensusMode.QUALITY_FIRST else "balanced"
        
        hf_result = await self.enhanced_consensus.get_enhanced_consensus_response(
            prompt=query,
            domain_hint=context.domain_hint,
            quality_preference=quality_preference,
            max_experts=strategy_config["max_experts"]
        )
        
        return ConsensusResult(
            query_id=context.query_id,
            response=hf_result["consensus_response"],
            confidence=hf_result["confidence"],
            quality_score=hf_result.get("quality_estimate", 8.0),
            participating_experts=hf_result.get("selected_models", []),
            expert_responses=hf_result.get("all_responses", []),
            routing_decision=hf_result.get("expert_selection_strategy", {}),
            consensus_method=hf_result["consensus_method"],
            response_time_ms=hf_result.get("response_time_ms", 0),
            total_cost=hf_result.get("cost_estimate", 0.0),
            tokens_generated=len(hf_result["consensus_response"].split()),
            consensus_mode=strategy_config["mode"]
        )
    
    async def _execute_hybrid_routing(
        self,
        query: str,
        context: QueryContext,
        strategy_config: Dict[str, Any]
    ) -> ConsensusResult:
        """Execute hybrid expert routing."""
        
        routing_decision = await self.hybrid_router.route_hybrid_query(
            query=query,
            context=None,
            routing_strategy=strategy_config.get("routing_strategy", RoutingStrategy.BALANCED),
            max_experts=strategy_config["max_experts"],
            budget_limit=strategy_config["cost_limit"],
            quality_threshold=strategy_config["quality_threshold"],
            latency_threshold_ms=strategy_config["latency_limit_ms"]
        )
        
        # Mock consensus generation based on routing decision
        mock_response = f"Based on the analysis from {len(routing_decision['selected_models'])} expert models, here's the consensus response to your query."
        
        return ConsensusResult(
            query_id=context.query_id,
            response=mock_response,
            confidence=0.85,
            quality_score=routing_decision["expected_quality"],
            participating_experts=routing_decision["selected_models"],
            expert_responses=[],
            routing_decision=routing_decision,
            consensus_method="hybrid_routing",
            response_time_ms=routing_decision.get("estimated_latency_ms", 1000),
            total_cost=routing_decision["estimated_cost"],
            tokens_generated=len(mock_response.split()),
            consensus_mode=strategy_config["mode"]
        )
    
    async def _execute_unified_consensus(
        self,
        query: str,
        context: QueryContext,
        strategy_config: Dict[str, Any]
    ) -> ConsensusResult:
        """Execute unified consensus strategy (fallback)."""
        
        # Mock metrics for unified consensus
        mock_metrics = {
            "loss": 0.5,
            "accuracy": 0.85,
            "perplexity": 2.1
        }
        
        unified_result = self.unified_consensus.update(
            params={},
            metrics=mock_metrics,
            global_step=1000
        )
        
        mock_response = "This is a fallback response from the unified consensus strategy."
        
        return ConsensusResult(
            query_id=context.query_id,
            response=mock_response,
            confidence=unified_result["consensus_score"],
            quality_score=8.0,
            participating_experts=["unified_consensus"],
            expert_responses=[],
            routing_decision=unified_result,
            consensus_method="unified_consensus",
            response_time_ms=500,
            total_cost=0.0,
            tokens_generated=len(mock_response.split()),
            consensus_mode=strategy_config["mode"]
        )
    
    async def _post_process_result(
        self,
        result: ConsensusResult,
        query_analysis: Dict[str, Any],
        context: QueryContext
    ) -> ConsensusResult:
        """Post-process and enhance consensus result."""
        
        # Add explanation if enabled
        if self.config.enable_explanation_generation:
            result.explanation = await self._generate_explanation(result, query_analysis)
        
        # Calculate uncertainty score if enabled
        if self.config.enable_uncertainty_estimation:
            result.uncertainty_score = await self._estimate_uncertainty(result)
        
        # Detect bias if enabled
        if self.config.enable_bias_detection:
            result.bias_score = await self._detect_bias(result)
        
        # Apply safety filtering if enabled
        if self.config.enable_safety_filtering:
            result.safety_score = await self._apply_safety_filtering(result)
        
        return result
    
    async def _generate_explanation(self, result: ConsensusResult, query_analysis: Dict[str, Any]) -> str:
        """Generates explanation for the consensus result."""
        
        explanation_parts = [
            f"This response was generated using the {result.consensus_method} method",
            f"with {len(result.participating_experts)} expert models.",
            f"The consensus confidence is {result.confidence:.2f}",
            f"and the estimated quality score is {result.quality_score:.1f}/10."
        ]
        
        if result.total_cost > 0:
            explanation_parts.append(f"The total cost was ${result.total_cost:.4f}.")
        
        return " ".join(explanation_parts)
    
    async def _estimate_uncertainty(self, result: ConsensusResult) -> float:
        """Estimate uncertainty in the consensus result."""
        
        # Simple uncertainty estimation based on confidence and expert agreement
        base_uncertainty = 1.0 - result.confidence
        
        # Adjust based on number of experts
        expert_factor = 1.0 / (len(result.participating_experts) + 1)
        
        return min(base_uncertainty + expert_factor * 0.5, 1.0)
    
    async def _detect_bias(self, result: ConsensusResult) -> float:
        """Detect potential bias in the consensus result."""
        
        # Mock bias detection - in real implementation, use bias detection models
        return 0.1  # Low bias score
    
    async def _apply_safety_filtering(self, result: ConsensusResult) -> float:
        """Apply safety filtering to the consensus result."""
        
        # Mock safety filtering - in real implementation, use safety models
        return 0.95  # High safety score
    
    async def _update_metrics_and_learning(
        self,
        result: ConsensusResult,
        processing_time: float,
        context: QueryContext
    ):
        """Update system metrics and adaptive learning."""
        
        # Update basic metrics
        self.metrics.avg_response_time_ms = (
            (self.metrics.avg_response_time_ms * (self.metrics.total_queries - 1) +
             processing_time * 1000) / self.metrics.total_queries
        )
        
        self.metrics.avg_quality_score = (
            (self.metrics.avg_quality_score * (self.metrics.total_queries - 1) +
             result.quality_score) / self.metrics.total_queries
        )
        
        self.metrics.avg_cost_per_query = (
            (self.metrics.avg_cost_per_query * (self.metrics.total_queries - 1) +
             result.total_cost) / self.metrics.total_queries
        )
        
        self.metrics.total_cost += result.total_cost
        
        # Update consensus method distribution
        method = result.consensus_method
        self.metrics.consensus_method_distribution[method] = (
            self.metrics.consensus_method_distribution.get(method, 0) + 1
        )
        
        # Update expert utilization
        for expert in result.participating_experts:
            if "serverless" in expert.lower() or "hf_" in expert.lower():
                self.metrics.serverless_expert_usage += 1
            elif "premium" in expert.lower():
                self.metrics.premium_expert_usage += 1
            else:
                self.metrics.local_expert_usage += 1
        
        # Update quality tracking
        self.metrics.quality_trend.append(result.quality_score)
        if len(self.metrics.quality_trend) > 100:
            self.metrics.quality_trend = self.metrics.quality_trend[-100:]
        
        # Update adaptive learning if enabled
        if self.config.enable_continuous_learning:
            await self._update_adaptive_learning(result, context)
    
    async def _update_adaptive_learning(self, result: ConsensusResult, context: QueryContext):
        """Update adaptive learning weights based on result quality."""
        
        # Calculate performance score
        performance_score = (
            result.quality_score * self.config.quality_feedback_weight +
            (1.0 - min(result.total_cost / self.config.max_cost_per_query, 1.0)) * self.config.cost_feedback_weight +
            (1.0 - min(result.response_time_ms / context.latency_requirement_ms, 1.0)) * self.config.latency_feedback_weight
        )
        
        # Update consensus mode preferences
        mode = result.consensus_mode.value
        current_weight = self.adaptation_weights["consensus_mode_preferences"].get(mode, 1.0)
        learning_rate = self.config.adaptation_learning_rate
        
        # Simple gradient-based update
        new_weight = current_weight + learning_rate * (performance_score - 0.5)
        self.adaptation_weights["consensus_mode_preferences"][mode] = max(0.1, min(2.0, new_weight))
        
        # Update expert preferences
        for expert in result.participating_experts:
            current_weight = self.adaptation_weights["expert_preferences"].get(expert, 1.0)
            new_weight = current_weight + learning_rate * (result.quality_score / 10.0 - 0.5)
            self.adaptation_weights["expert_preferences"][expert] = max(0.1, min(2.0, new_weight))
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        uptime = datetime.now() - self.start_time
        self.metrics.system_uptime_hours = uptime.total_seconds() / 3600
        
        # Calculate success rates
        total_queries = max(self.metrics.total_queries, 1)
        success_rate = self.metrics.successful_queries / total_queries
        self.metrics.error_rate = self.metrics.failed_queries / total_queries
        self.metrics.consensus_success_rate = success_rate
        
        # Calculate cost efficiency
        if self.metrics.avg_cost_per_query > 0:
            self.metrics.cost_efficiency_score = self.metrics.avg_quality_score / self.metrics.avg_cost_per_query
        
        return {
            "system_info": {
                "name": self.config.system_name,
                "state": self.state.value,
                "uptime_hours": round(self.metrics.system_uptime_hours, 2),
                "version": "1.0.0"
            },
            "performance_metrics": {
                "total_queries": self.metrics.total_queries,
                "success_rate": f"{success_rate:.2%}",
                "avg_response_time_ms": round(self.metrics.avg_response_time_ms, 2),
                "avg_quality_score": round(self.metrics.avg_quality_score, 2),
                "avg_cost_per_query": f"${self.metrics.avg_cost_per_query:.4f}"
            },
            "expert_utilization": {
                "local_experts": self.metrics.local_expert_usage,
                "serverless_experts": self.metrics.serverless_expert_usage,
                "premium_experts": self.metrics.premium_expert_usage
            },
            "consensus_methods": self.metrics.consensus_method_distribution,
            "cost_analysis": {
                "total_cost": f"${self.metrics.total_cost:.4f}",
                "cost_efficiency_score": round(self.metrics.cost_efficiency_score, 2),
                "cost_savings_estimate": f"{self.metrics.cost_savings_vs_premium:.1%}"
            },
            "quality_tracking": {
                "current_trend": "improving" if len(self.metrics.quality_trend) > 1 and 
                                self.metrics.quality_trend[-1] > self.metrics.quality_trend[-2] else "stable",
                "quality_variance": round(np.std(self.metrics.quality_trend), 2) if self.metrics.quality_trend else 0.0
            },
            "active_queries": len(self.active_queries),
            "recent_queries": len([q for q in self.query_history if 
                                 (datetime.now() - q.timestamp).total_seconds() < 3600])
        }


# Factory functions and utilities
def create_meta_consensus_system(config: Optional[MetaConsensusConfig] = None) -> MetaConsensusSystem:
    """Factory function to create meta-consensus system."""
    
    if config is None:
        config = MetaConsensusConfig()
    
    return MetaConsensusSystem(config)

def create_default_config(
    hf_api_token: str = "",
    enable_serverless: bool = True,
    max_cost_per_query: float = 0.02
) -> MetaConsensusConfig:
    """Creates default configuration for meta-consensus system."""
    
    return MetaConsensusConfig(
        hf_api_token=hf_api_token,
        enable_serverless=enable_serverless,
        max_cost_per_query=max_cost_per_query,
        enable_btx_training=True,
        enable_adaptive_routing=True,
        enable_continuous_learning=True
    )


# Export main components
__all__ = [
    'MetaConsensusSystem',
    'MetaConsensusConfig',
    'QueryContext',
    'ConsensusResult',
    'ConsensusMode',
    'SystemState',
    'SystemMetrics',
    'create_meta_consensus_system',
    'create_default_config'
]


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create configuration
        config = create_default_config(
            hf_api_token="your_hf_token_here",
            enable_serverless=True,
            max_cost_per_query=0.05
        )
        
        # Create meta-consensus system
        system = create_meta_consensus_system(config)
        
        # Initialize system
        if await system.initialize():
            print("🚀 Meta-Consensus System initialized successfully")
            
            # Process example queries
            queries = [
                "Explain quantum computing principles and their applications in cryptography.",
                "Write a Python function to implement a binary search algorithm.",
                "¿Cuáles son los beneficios de la inteligencia artificial en la medicina?",
                "Analyze the economic impact of renewable energy adoption.",
                "Create a creative story about time travel."
            ]
            
            for i, query in enumerate(queries):
                print(f"\n--- Query {i+1} ---")
                print(f"Query: {query}")
                
                # Create query context
                context = QueryContext(
                    query_id=f"example_query_{i+1}",
                    quality_requirement=0.8,
                    cost_limit=0.02,
                    latency_requirement_ms=5000
                )
                
                # Process query
                result = await system.process_query(
                    query=query,
                    context=context,
                    consensus_mode=ConsensusMode.ADAPTIVE
                )
                
                print(f"Response: {result.response[:100]}...")
                print(f"Confidence: {result.confidence:.2f}")
                print(f"Quality Score: {result.quality_score:.1f}")
                print(f"Experts Used: {len(result.participating_experts)}")
                print(f"Cost: ${result.total_cost:.4f}")
                print(f"Response Time: {result.response_time_ms:.0f}ms")
            
            # Get system status
            status = system.get_system_status()
            print(f"\n🎭 System Status:")
            print(f"State: {status['system_info']['state']}")
            print(f"Total Queries: {status['performance_metrics']['total_queries']}")
            print(f"Success Rate: {status['performance_metrics']['success_rate']}")
            print(f"Avg Quality: {status['performance_metrics']['avg_quality_score']}")
            print(f"Avg Cost: {status['performance_metrics']['avg_cost_per_query']}")
            print(f"Avg Response Time: {status['performance_metrics']['avg_response_time_ms']}ms")
        
        else:
            print("❌ Meta-Consensus System initialization failed")
    
    asyncio.run(main())