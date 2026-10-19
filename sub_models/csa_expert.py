#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSA (Counterfactual Scenario Analysis) Expert Sub-Model

Enhanced counterfactual reasoning sub-model for CapibaraGPT with TPU optimization
and integration with the existing router system.
"""

import logging
import time
import math
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, NamedTuple
from enum import Enum
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import nn
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = np  # type: ignore
    logging.warning("JAX not available for CSA Expert; using NumPy fallback")

from interfaces.isub_models import (
    ICounterfactualExpert, 
    ExpertContext, 
    ExpertResult,
    PrecisionMode,
    ConfigTPU
)

logger = logging.getLogger(__name__)

# --------- Enhanced Specifications and Types ---------

class CSATaskType(Enum):
    """Types of tasks that CSA can handle."""
    DIAGNOSIS = "diagnosis"
    PLANNING = "planning"
    DESIGN = "design"
    STRATEGY = "strategy"
    TROUBLESHOOTING = "troubleshooting"
    RISK_ANALYSIS = "risk_analysis"
    OPTIMIZATION = "optimization"

@dataclass
class Hypothesis:
    """Enhanced hypothesis with metadata and scoring."""
    premise: str
    delta: str
    prior_score: float
    rationale: str
    task_type: CSATaskType
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorldState:
    """Enhanced world state with temporal aspects."""
    vars: Dict[str, float]
    evidence: List[str]
    consequences: str
    timestamp: float = field(default_factory=time.time)
    uncertainty: float = 0.0
    stability_score: float = 1.0

@dataclass
class CFResult:
    """Enhanced counterfactual result with rich metadata."""
    hypothesis: Hypothesis
    rollout: WorldState
    plausibility: float
    utility: float
    diversity: float
    score: float
    risk_assessment: float = 0.0
    actionability: float = 0.0
    confidence_interval: tuple = (0.0, 1.0)

@dataclass
class CSAExpertConfig:
    """Configurestion for CSA Expert."""
    max_hypotheses: int = 6
    max_rollout_steps: int = 3
    min_plausibility: float = 0.5
    min_utility: float = 0.5
    weights: Dict[str, float] = field(default_factory=lambda: {
        "plausibility": 0.45,
        "utility": 0.35,
        "diversity": 0.15,
        "actionability": 0.05
    })
    tpu_config: ConfigTPU = field(default_factory=ConfigTPU)
    enable_logging: bool = True
    cache_size: int = 100
    temperature: float = 0.7

# --------- Enhanced Utilities ---------

def sigmoid(x: float) -> float:
    """Safe sigmoid function."""
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

def safe_log(x: float, epsilon: float = 1e-8) -> float:
    """Safe logarithm function."""
    return math.log(max(epsilon, x))

def cosine_similarity(a: set, b: set) -> float:
    """Enhanced cosine similarity with proper normalization."""
    if not a or not b:
        return 0.0
    
    intersection = len(a & b)
    denominator = (len(a) * len(b)) ** 0.5
    
    return intersection / denominator if denominator > 0 else 0.0

def enhanced_tokenize(text: str) -> set:
    """Enhanced tokenization with stop word removal."""
    import re
    
    # Simple stop words (in production, use nltk or spacy)
    stop_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 
                  'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'si', 'the', 
                  'is', 'at', 'and', 'or', 'but', 'in', 'on', 'to', 'be', 'have', 'it', 'that', 'for'}
    
    # Clean and tokenize
    tokens = re.findall(r'\b\w+\b', text.lower())
    return set(token for token in tokens if token not in stop_words and len(token) > 2)

# --------- Enhanced CSA Components ---------

class EnhancedCSAGenerator:
    """Enhanced hypothesis generator with domain-specific patterns."""
    
    def __init__(self, config: CSAExpertConfig):
        self.config = config
        self.domain_patterns = self._load_domain_patterns()
        
    def _load_domain_patterns(self) -> Dict[CSATaskType, List[str]]:
        """Load domain-specific hypothesis patterns."""
        return {
            CSATaskType.DIAGNOSIS: [
                "if the critical component {0} fails intermittently",
                "if the system configuration {0} is outdated",
                "if the operating environment varies in a range of ±{1}%",
                "if the sensor data {0} is biased",
                "if the external dependency {0} does not respond in time",
                "if the environmental conditions {0} are outside the normal range"
            ],
            CSATaskType.PLANNING: [
                "if the available resources {0} are reduced by {1}%",
                "if the project schedule {0} is extended by {1} weeks",
                "if the client requirements {0} change significantly",
                "if unforeseen dependencies arise with {0}",
                "if the budget {0} is affected by {1}% inflation"
            ],
            CSATaskType.TROUBLESHOOTING: [
                "if the problem originates in the {0} layer of the system",
                "if there is a race condition in {0}",
                "if the permissions for {0} are misconfigured",
                "if there is a bottleneck in {0}",
                "if the version of {0} is incompatible"
            ]
        }
    
    def propose(self, context: ExpertContext, max_hypotheses: int) -> List[Hypothesis]:
        """Generates enhanced hypotheses based on context."""
        task_type = self._infer_task_type(context)
        patterns = self.domain_patterns.get(task_type, self.domain_patterns[CSATaskType.DIAGNOSIS])
        
        hypotheses = []
        key_terms = self._extract_key_terms(context.text)
        
        for i, pattern in enumerate(patterns[:max_hypotheses]):
            # Fill pattern with extracted terms
            if key_terms and len(key_terms) > i % len(key_terms):
                filled_pattern = pattern.format(
                    key_terms[i % len(key_terms)], 
                    str(10 + i * 5)  # Variable percentage
                )
            else:
                filled_pattern = pattern.format("system", str(15))
            
            hypothesis = Hypothesis(
                premise=f"Base context: {context.text[:200]}...",
                delta=filled_pattern,
                prior_score=0.5 + (i * 0.05),  # Slight variation
                rationale=f"Counterfactual analysis based on pattern {task_type.value}",
                task_type=task_type,
                confidence=0.6 + (i * 0.02),
                metadata={
                    "pattern_id": i,
                    "key_terms": key_terms,
                    "generation_time": time.time()
                }
            )
            hypotheses.append(hypothesis)
            
        return hypotheses
    
    def _infer_task_type(self, context: ExpertContext) -> CSATaskType:
        """Infer task type from context."""
        hint_mapping = {
            "diagnosis": CSATaskType.DIAGNOSIS,
            "planning": CSATaskType.PLANNING,
            "design": CSATaskType.DESIGN,
            "strategy": CSATaskType.STRATEGY,
            "troubleshooting": CSATaskType.TROUBLESHOOTING,
            "risk": CSATaskType.RISK_ANALYSIS,
            "optimization": CSATaskType.OPTIMIZATION
        }
        
        return hint_mapping.get(context.task_hint, CSATaskType.DIAGNOSIS)
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from input text."""
        tokens = enhanced_tokenize(text)
        # Simple heuristic: longer words often more important
        sorted_tokens = sorted(tokens, key=len, reverse=True)
        return sorted_tokens[:5]

class KnowledgeBaseRetriever:
    """Simple knowledge base retriever for evidence lookup."""

    def __init__(self, knowledge_base: Optional[Dict[str, List[str]]] = None):
        # Default knowledge base with common patterns
        self.knowledge_base = knowledge_base or {
            "risk": [
                "High risk scenarios typically require mitigation strategies",
                "Risk can be reduced through redundancy and monitoring",
                "Historical data shows 30% of high-risk changes cause incidents"
            ],
            "cost": [
                "Cost overruns are common in complex system changes",
                "Budget contingency of 15-20% recommended for technical projects",
                "Resource allocation impacts both cost and timeline"
            ],
            "time": [
                "Schedule delays often compound in complex systems",
                "Critical path analysis helps identify bottlenecks",
                "Parallel execution can reduce overall timeline"
            ],
            "diagnosis": [
                "Intermittent issues often indicate race conditions or resource contention",
                "Configuration drift is a common cause of unexpected behavior",
                "Sensor failures can mask underlying system issues"
            ],
            "planning": [
                "Resource constraints directly impact project feasibility",
                "Schedule compression increases risk of quality issues",
                "Early stakeholder alignment reduces change requests"
            ],
            "troubleshooting": [
                "Layer-by-layer analysis isolates problem domains",
                "Race conditions are difficult to reproduce consistently",
                "Performance bottlenecks often cascade through systems"
            ],
            "stability": [
                "System stability depends on component interdependencies",
                "Gradual rollouts reduce stability risks",
                "Monitoring and alerting are critical for stability"
            ]
        }
        self._vector_store = None

    def set_vector_store(self, vector_store):
        """Set external vector store for semantic retrieval."""
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        hypothesis: Optional['Hypothesis'] = None,
        state: Optional[Dict[str, float]] = None,
        top_k: int = 3
    ) -> List[str]:
        """Retrieve relevant evidence for the query and context.

        Args:
            query: The query text (usually hypothesis delta)
            hypothesis: Optional hypothesis for task-type filtering
            state: Optional world state for metric-based filtering
            top_k: Maximum number of evidence items to return

        Returns:
            List of relevant evidence strings
        """
        evidence = []

        # Try vector store first if available
        if self._vector_store is not None:
            try:
                results = self._vector_store.invoke(query)
                for doc in results[:top_k]:
                    if hasattr(doc, 'page_content'):
                        evidence.append(doc.page_content)
                if evidence:
                    return evidence
            except Exception as e:
                logger.debug(f"Vector store retrieval failed: {e}")

        # Fallback to keyword-based retrieval from local knowledge base
        query_lower = query.lower()

        # Get task-specific evidence
        if hypothesis:
            task_key = hypothesis.task_type.value
            if task_key in self.knowledge_base:
                evidence.extend(self.knowledge_base[task_key][:top_k])

        # Get metric-specific evidence based on state
        if state:
            if state.get("risk", 0) > 0.6:
                evidence.extend(self.knowledge_base.get("risk", [])[:1])
            if state.get("cost", 0) > 1.2:
                evidence.extend(self.knowledge_base.get("cost", [])[:1])
            if state.get("time", 0) > 1.3:
                evidence.extend(self.knowledge_base.get("time", [])[:1])
            if state.get("stability", 1) < 0.5:
                evidence.extend(self.knowledge_base.get("stability", [])[:1])

        # Keyword matching as final fallback
        if not evidence:
            for key, items in self.knowledge_base.items():
                if key in query_lower:
                    evidence.extend(items[:1])

        # Deduplicate and limit
        seen = set()
        unique_evidence = []
        for e in evidence:
            if e not in seen:
                seen.add(e)
                unique_evidence.append(e)

        return unique_evidence[:top_k]


class EnhancedWorldModel:
    """Enhanced world model with uncertainty quantification."""

    def __init__(self, config: CSAExpertConfig, knowledge_retriever: Optional[KnowledgeBaseRetriever] = None):
        self.config = config
        self.knowledge_retriever = knowledge_retriever or KnowledgeBaseRetriever()
        
    def infer_initial(self, context: ExpertContext) -> Dict[str, float]:
        """Infer initial world state with uncertainty."""
        base_state = {
            "risk": 0.4,
            "cost": 1.0,
            "time": 1.0,
            "confidence": 0.6,
            "complexity": 0.5,
            "stability": 0.8
        }
        
        # Adjust based on context constraints
        if context.constraints:
            if "budget" in context.constraints:
                base_state["cost"] *= context.constraints["budget"]
            if "time" in context.constraints:
                base_state["time"] *= context.constraints["time"]
            if "risk_tolerance" in context.constraints:
                base_state["risk"] = context.constraints["risk_tolerance"]
                
        return base_state
    
    def apply_delta(self, state: Dict[str, float], hypothesis: Hypothesis) -> Dict[str, float]:
        """Apply hypothesis delta to state with enhanced logic."""
        new_state = dict(state)
        delta_lower = hypothesis.delta.lower()
        
        # Enhanced delta application based on task type
        if hypothesis.task_type == CSATaskType.DIAGNOSIS:
            self._apply_diagnosis_delta(new_state, delta_lower)
        elif hypothesis.task_type == CSATaskType.PLANNING:
            self._apply_planning_delta(new_state, delta_lower)
        elif hypothesis.task_type == CSATaskType.TROUBLESHOOTING:
            self._apply_troubleshooting_delta(new_state, delta_lower)
        
        # Ensure bounds
        return {k: max(0.0, min(2.0, v)) for k, v in new_state.items()}
    
    def _apply_diagnosis_delta(self, state: Dict[str, float], delta: str):
        """Apply diagnosis-specific deltas."""
        if "intermittent" in delta:
            state["risk"] += 0.3
            state["stability"] -= 0.2
        if "configuration" in delta:
            state["risk"] += 0.2
            state["cost"] += 0.15
        if "sensor" in delta:
            state["confidence"] -= 0.3
            state["risk"] += 0.1
        if "dependency" in delta:
            state["time"] += 0.4
            state["risk"] += 0.2
    
    def _apply_planning_delta(self, state: Dict[str, float], delta: str):
        """Apply planning-specific deltas."""
        if "resources" in delta:
            state["cost"] += 0.3
            state["time"] += 0.2
        if "schedule" in delta:
            state["time"] += 0.5
            state["risk"] += 0.1
        if "budget" in delta:
            state["cost"] += 0.4
            state["risk"] += 0.15
    
    def _apply_troubleshooting_delta(self, state: Dict[str, float], delta: str):
        """Apply troubleshooting-specific deltas."""
        if "layer" in delta:
            state["complexity"] += 0.3
            state["time"] += 0.2
        if "race" in delta:
            state["risk"] += 0.4
            state["stability"] -= 0.3
        if "bottleneck" in delta:
            state["time"] += 0.6
            state["risk"] += 0.1
    
    def rollout(self, context: ExpertContext, hypothesis: Hypothesis, steps: int) -> WorldState:
        """Enhanced rollout with uncertainty propagation."""
        state = self.infer_initial(context)
        uncertainty = 0.1
        
        for step in range(steps):
            state = self.apply_delta(state, hypothesis)
            
            # Add temporal dynamics (gradual return to baseline)
            relaxation_factor = 0.85 ** (step + 1)
            for key in state:
                if key in ["risk", "cost", "time"]:
                    baseline = {"risk": 0.4, "cost": 1.0, "time": 1.0}.get(key, 0.5)
                    state[key] = relaxation_factor * state[key] + (1 - relaxation_factor) * baseline
            
            # Accumulate uncertainty
            uncertainty = min(0.5, uncertainty + 0.05)
        
        # Generate consequence description
        consequences = self._generate_consequences(state, hypothesis)

        # Retrieve evidence from knowledge base
        evidence = self.knowledge_retriever.retrieve(
            query=hypothesis.delta,
            hypothesis=hypothesis,
            state=state,
            top_k=3
        )

        return WorldState(
            vars=state,
            evidence=evidence,
            consequences=consequences,
            uncertainty=uncertainty,
            stability_score=state.get("stability", 0.8)
        )
    
    def _generate_consequences(self, state: Dict[str, float], hypothesis: Hypothesis) -> str:
        """Generates human-readable consequences."""
        consequences = []

        if state["risk"] > 0.7:
            consequences.append("High risk detected")
        if state["cost"] > 1.3:
            consequences.append("Significant cost overrun")
        if state["time"] > 1.5:
            consequences.append("Considerable delays")
        if state["confidence"] < 0.4:
            consequences.append("Low confidence in results")

        if not consequences:
            consequences.append("Moderate impact on the system")

        risk_str = f"risk={state['risk']:.2f}"
        cost_str = f"cost={state['cost']:.2f}"
        time_str = f"time={state['time']:.2f}"
        conf_str = f"confidence={state['confidence']:.2f}"

        return f"Under '{hypothesis.delta}': {', '.join(consequences)}. Metrics: {risk_str}, {cost_str}, {time_str}, {conf_str}"

class EnhancedCSAScorer:
    """Enhanced scoring with multiple evaluation criteria."""
    
    def __init__(self, config: CSAExpertConfig):
        self.config = config
        
    def plausibility(self, context: ExpertContext, world_state: WorldState, hypothesis: Hypothesis) -> float:
        """Enhanced plausibility scoring."""
        score = 0.0
        
        # Base plausibility from world state consistency
        score += 1.0 - abs(world_state.vars["confidence"] - 0.6)
        score += 1.0 - min(1.0, world_state.vars["risk"])
        score += world_state.stability_score
        
        # Task-specific plausibility
        if hypothesis.task_type == CSATaskType.DIAGNOSIS:
            score += self._diagnosis_plausibility(world_state)
        elif hypothesis.task_type == CSATaskType.PLANNING:
            score += self._planning_plausibility(world_state)
        
        # Uncertainty penalty
        score -= world_state.uncertainty
        
        return sigmoid(score - 2.0)
    
    def _diagnosis_plausibility(self, world_state: WorldState) -> float:
        """Diagnosis-specific plausibility."""
        # Diagnostic scenarios should show clear cause-effect
        if world_state.vars["risk"] > 0.3 and world_state.vars["confidence"] < 0.8:
            return 0.5
        return 0.0
    
    def _planning_plausibility(self, world_state: WorldState) -> float:
        """Planning-specific plausibility."""
        # Planning scenarios should balance cost, time, risk
        balance = 1.0 - abs(world_state.vars["cost"] - world_state.vars["time"])
        return min(0.5, balance)
    
    def utility(self, context: ExpertContext, world_state: WorldState) -> float:
        """Enhanced utility scoring."""
        utility_score = 0.0
        
        # Constraint-based utility
        if context.constraints:
            budget = context.constraints.get("budget", 1.0)
            target_time = context.constraints.get("time", 1.0)
            risk_tolerance = context.constraints.get("risk_tolerance", 0.5)
            
            utility_score += max(0, budget - world_state.vars["cost"])
            utility_score += max(0, target_time - world_state.vars["time"])
            utility_score += max(0, risk_tolerance - world_state.vars["risk"])
        
        # General utility preferences
        utility_score += (0.7 - world_state.vars["risk"])  # Prefer lower risk
        utility_score += world_state.vars["confidence"]    # Prefer higher confidence
        
        return sigmoid(utility_score)
    
    def diversity(self, hypothesis: Hypothesis, existing_results: List[CFResult]) -> float:
        """Enhanced diversity scoring."""
        if not existing_results:
            return 1.0
        
        current_tokens = enhanced_tokenize(hypothesis.delta)
        similarities = []
        
        for result in existing_results:
            other_tokens = enhanced_tokenize(result.hypothesis.delta)
            sim = cosine_similarity(current_tokens, other_tokens)
            similarities.append(sim)
        
        # Consider both semantic and task type diversity
        semantic_diversity = max(0.0, 1.0 - max(similarities))
        
        task_type_diversity = 1.0
        existing_task_types = [r.hypothesis.task_type for r in existing_results]
        if hypothesis.task_type in existing_task_types:
            task_type_diversity = 0.5
        
        return 0.7 * semantic_diversity + 0.3 * task_type_diversity
    
    def actionability(self, context: ExpertContext, world_state: WorldState, hypothesis: Hypothesis) -> float:
        """Score how actionable the hypothesis is."""
        actionability_score = 0.0
        
        # Clear, specific hypotheses are more actionable
        if len(hypothesis.delta.split()) > 5:  # Not too vague
            actionability_score += 0.3
        
        # Reasonable confidence levels
        if 0.3 <= hypothesis.confidence <= 0.9:
            actionability_score += 0.3
        
        # Moderate impact (not too extreme)
        total_impact = world_state.vars["risk"] + world_state.vars["cost"] + world_state.vars["time"]
        if 1.5 <= total_impact <= 4.0:
            actionability_score += 0.4
        
        return actionability_score
    
    def score(self, context: ExpertContext, world_state: WorldState, 
              hypothesis: Hypothesis, existing_results: List[CFResult]) -> CFResult:
        """Enhanced comprehensive scoring."""
        plaus = self.plausibility(context, world_state, hypothesis)
        util = self.utility(context, world_state)
        div = self.diversity(hypothesis, existing_results)
        action = self.actionability(context, world_state, hypothesis)
        
        weights = self.config.weights
        score = (
            weights["plausibility"] * plaus +
            weights["utility"] * util +
            weights["diversity"] * div +
            weights["actionability"] * action
        )
        
        # Risk assessment
        risk_assessment = world_state.vars["risk"] * (1 - world_state.vars["confidence"])
        
        # Confidence interval
        uncertainty = world_state.uncertainty
        confidence_interval = (
            max(0.0, score - uncertainty),
            min(1.0, score + uncertainty)
        )
        
        return CFResult(
            hypothesis=hypothesis,
            rollout=world_state,
            plausibility=plaus,
            utility=util,
            diversity=div,
            score=score,
            risk_assessment=risk_assessment,
            actionability=action,
            confidence_interval=confidence_interval
        )

# --------- Main CSA Expert Class ---------

class CSAExpert(ICounterfactualExpert):
    """Enhanced CSA Expert with full integration capabilities."""
    
    def __init__(self, config: Optional[CSAExpertConfig] = None):
        self.config = config or CSAExpertConfig()
        self.generator = EnhancedCSAGenerator(self.config)
        self.world_model = EnhancedWorldModel(self.config)
        self.scorer = EnhancedCSAScorer(self.config)
        self.cache = {}
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "processing_times": []
        }
        
        logger.debug(f"CSA Expert initialized with config: {self.config}")
    
    @property
    def name(self) -> str:
        return "CSA"
    
    @property
    def version(self) -> str:
        return "2.0.0"
    
    def supports(self, context: ExpertContext) -> bool:
        """Enhanced support detection."""
        # Direct task hint support
        supported_tasks = {
            "diagnosis", "planning", "design", "strategy", 
            "troubleshooting", "risk_analysis", "optimization"
        }
        
        if context.task_hint in supported_tasks:
            return True
        
        # Flag-based activation
        if context.flags and context.flags.get("ask_counterfactual", False):
            return True
        
        # Keyword-based detection
        counterfactual_keywords = {
            "if", "what if", "alternatively",
            "scenario", "hypothesis", "assume", "suppose"
        }

        text_lower = context.text.lower()
        if any(keyword in text_lower for keyword in counterfactual_keywords):
            return True

        # Complexity-based activation (for complex problems)
        if len(context.text) > 200 and any(word in text_lower for word in
                                          ["problem", "error", "failure", "issue"]):
            return True
        
        return False
    
    async def process(self, context: ExpertContext) -> ExpertResult:
        """Enhanced processing with full pipeline."""
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # Check cache
            cache_key = self._generate_cache_key(context)
            if cache_key in self.cache:
                self.metrics["cache_hits"] += 1
                cached_result = self.cache[cache_key]
                logger.debug(f"CSA cache hit for key: {cache_key[:20]}...")
                return cached_result
            
            # Generate and evaluate counterfactuals
            results = await self._generate_counterfactuals(context)
            
            processing_time = time.time() - start_time
            self.metrics["processing_times"].append(processing_time)
            
            # Create expert result
            expert_result = ExpertResult(
                success=len(results) > 0,
                output=results,
                confidence=self._calculate_overall_confidence(results),
                processing_time=processing_time,
                expert_name=self.name,
                metadata={
                    "num_results": len(results),
                    "task_type": context.task_hint,
                    "cache_used": False,
                    "version": self.version
                }
            )
            
            # Cache result
            if len(self.cache) < self.config.cache_size:
                self.cache[cache_key] = expert_result
            
            # Only log every 10th request to reduce verbosity
            if not hasattr(self, '_log_counter'):
                self._log_counter = 0
            self._log_counter += 1
            if self._log_counter % 10 == 0:
                logger.info(f"CSA processed {self._log_counter} requests, latest: {processing_time:.3f}s, {len(results)} scenarios")
            return expert_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"CSA processing failed: {e}")
            
            return ExpertResult(
                success=False,
                output=[],
                confidence=0.0,
                processing_time=processing_time,
                expert_name=self.name,
                error_message=str(e)
            )
    
    async def _generate_counterfactuals(self, context: ExpertContext) -> List[CFResult]:
        """Generates and evaluate counterfactual scenarios."""
        # Generate hypotheses
        hypotheses = self.generator.propose(context, self.config.max_hypotheses)
        
        # Evaluate each hypothesis
        results = []
        for hypothesis in hypotheses:
            # Create world state rollout
            world_state = self.world_model.rollout(context, hypothesis, self.config.max_rollout_steps)
            
            # Score the result
            cf_result = self.scorer.score(context, world_state, hypothesis, results)
            results.append(cf_result)
        
        # Filter and sort results
        filtered_results = [
            r for r in results 
            if r.plausibility >= self.config.min_plausibility and 
               r.utility >= self.config.min_utility
        ]
        
        filtered_results.sort(key=lambda r: r.score, reverse=True)
        return filtered_results[:3]  # Return top 3
    
    def generate_hypotheses(self, context: ExpertContext, max_hypotheses: int = 6) -> List[Hypothesis]:
        """Interface implementation for hypothesis generation."""
        return self.generator.propose(context, max_hypotheses)
    
    def evaluate_scenarios(self, context: ExpertContext, hypotheses: List[Hypothesis]) -> List[CFResult]:
        """Interface implementation for scenario evaluation."""
        results = []
        for hypothesis in hypotheses:
            world_state = self.world_model.rollout(context, hypothesis, self.config.max_rollout_steps)
            cf_result = self.scorer.score(context, world_state, hypothesis, results)
            results.append(cf_result)
        return results
    
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities."""
        return [
            "counterfactual_analysis",
            "scenario_planning",
            "risk_assessment", 
            "diagnostic_reasoning",
            "hypothesis_generation",
            "uncertainty_quantification",
            "decision_support"
        ]
    
    def _generate_cache_key(self, context: ExpertContext) -> str:
        """Generates cache key for context."""
        import hashlib
        
        key_components = [
            context.text[:500],  # Limit text length
            context.task_hint,
            str(sorted(context.constraints.items()) if context.constraints else ""),
            str(sorted(context.flags.items()) if context.flags else "")
        ]
        
        combined = "|".join(key_components)
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _calculate_overall_confidence(self, results: List[CFResult]) -> float:
        """Calculate overall confidence from results."""
        if not results:
            return 0.0
        
        # Weighted average based on scores
        total_weight = sum(r.score for r in results)
        if total_weight == 0:
            return 0.0
        
        weighted_confidence = sum(r.plausibility * r.score for r in results) / total_weight
        return weighted_confidence
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        avg_time = sum(self.metrics["processing_times"]) / len(self.metrics["processing_times"]) if self.metrics["processing_times"] else 0
        
        return {
            "total_requests": self.metrics["total_requests"],
            "cache_hits": self.metrics["cache_hits"],
            "cache_hit_rate": self.metrics["cache_hits"] / max(1, self.metrics["total_requests"]),
            "average_processing_time": avg_time,
            "cache_size": len(self.cache)
        }
    
    def __call__(self, inputs, context=None, **kwargs):
        """
        Make CSAExpert callable for direct invocation in training pipeline.
        
        Args:
            inputs: Input tensors or data
            context: Context information (optional)
            **kwargs: Additional keyword arguments
            
        Returns:
            Processed outputs from the expert
        """
        # Convert inputs to ExpertContext for processing
        if hasattr(inputs, 'shape'):
            # If inputs is a tensor/array, create a simple context
            expert_context = ExpertContext(
                text="Tensor input processing",
                task_hint="analysis",
                constraints=None,
                flags={"tensor_input": True}
            )
        elif isinstance(inputs, str):
            # If inputs is a string, use it directly
            expert_context = ExpertContext(
                text=inputs,
                task_hint="analysis",
                constraints=None,
                flags=None
            )
        else:
            # For other input types, create a generic context
            expert_context = ExpertContext(
                text=str(inputs),
                task_hint="analysis", 
                constraints=None,
                flags=None
            )
        
        # Process synchronously (since training loop expects sync results)
        import asyncio
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to handle this differently
                # For now, return a simple processed version of inputs
                return self._sync_process(expert_context, inputs)
            else:
                # Run the async process method
                result = loop.run_until_complete(self.process(expert_context))
                return self._convert_result_to_tensor(result, inputs)
        except RuntimeError:
            # No event loop running, create one
            try:
                result = asyncio.run(self.process(expert_context))
                return self._convert_result_to_tensor(result, inputs)
            except Exception:
                # Fallback to sync processing
                return self._sync_process(expert_context, inputs)
    
    def _sync_process(self, context: ExpertContext, original_inputs):
        """Synchronous processing fallback for training pipeline."""
        try:
            # Generate hypotheses synchronously
            hypotheses = self.generator.propose(context, self.config.max_hypotheses)
            
            # If we have valid hypotheses, return a modified version of inputs
            if hypotheses and hasattr(original_inputs, 'shape'):
                # Apply a simple transformation based on the number of hypotheses
                modification_factor = 1.0 + len(hypotheses) * 0.01  # Small modification
                return jnp.asarray(original_inputs) * modification_factor
            else:
                return original_inputs
                
        except Exception as e:
            logger.warning(f"CSA sync processing failed: {e}")
            return original_inputs
    
    def _convert_result_to_tensor(self, result: ExpertResult, original_inputs):
        """Convert ExpertResult back to tensor format for training pipeline."""
        if not result.success or not hasattr(original_inputs, 'shape'):
            return original_inputs
            
        try:
            # Apply confidence-based modification to inputs
            confidence_factor = 1.0 + result.confidence * 0.1
            return jnp.asarray(original_inputs) * confidence_factor
        except Exception:
            return original_inputs

def main():
    """Main function for testing CSA Expert."""
    # Simple test
    expert = CSAExpert()

    context = ExpertContext(
        text="The production system stops 3 times a day. The sensors show normal values but energy consumption increased by 15%. The ambient temperature is 35°C.",
        task_hint="diagnosis",
        constraints={"budget": 1.0, "time": 1.0, "risk_tolerance": 0.5},
        flags={"ask_counterfactual": True}
    )
    
    logger.info(f"Expert supports context: {expert.supports(context)}")
    logger.info(f"Expert capabilities: {expert.get_capabilities()}")
    
    return True

if __name__ == "__main__":
    main()
