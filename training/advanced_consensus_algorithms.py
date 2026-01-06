"""
Advanced Consensus Algorithms for Meta-Consensus System

This module implements sophisticated consensus algorithms that go beyond basic voting:
- Bayesian consensus with uncertainty quantification
- Multi-criteria decision analysis (MCDA)
- Ensemble confidence calibration
- Dynamic weight adjustment based on expert performance
- Semantic similarity clustering for response grouping
- Quality-aware consensus with bias detection
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import scipy.stats as stats

logger = logging.getLogger(__name__)

class ConsensusAlgorithm(Enum):
    """Available consensus algorithms."""
    BAYESIAN_CONSENSUS = "bayesian_consensus"
    MCDA_WEIGHTED = "mcda_weighted"
    SEMANTIC_CLUSTERING = "semantic_clustering"
    CONFIDENCE_CALIBRATED = "confidence_calibrated"
    DYNAMIC_WEIGHTED = "dynamic_weighted"
    QUALITY_AWARE = "quality_aware"
    HYBRID_ENSEMBLE = "hybrid_ensemble"

class QualityMetric(Enum):
    """Quality assessment metrics."""
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CLARITY = "clarity"
    FACTUALITY = "factuality"
    BIAS_SCORE = "bias_score"
    SAFETY_SCORE = "safety_score"

@dataclass
class ExpertResponse:
    """Enhanced expert response with detailed metadata."""
    expert_id: str
    response_text: str
    confidence: float
    quality_scores: Dict[QualityMetric, float] = field(default_factory=dict)
    
    # Performance metrics
    response_time_ms: float = 0.0
    cost: float = 0.0
    token_count: int = 0
    
    # Expert characteristics
    expert_tier: str = "unknown"
    domain_specialization: List[str] = field(default_factory=list)
    historical_accuracy: float = 0.8
    
    # Response analysis
    semantic_embedding: Optional[np.ndarray] = None
    uncertainty_score: float = 0.0
    bias_indicators: Dict[str, float] = field(default_factory=dict)
    safety_flags: List[str] = field(default_factory=list)

@dataclass
class ConsensusResult:
    """Enhanced consensus result with detailed analysis."""
    final_response: str
    consensus_confidence: float
    algorithm_used: ConsensusAlgorithm
    
    # Quality assessment
    quality_scores: Dict[QualityMetric, float] = field(default_factory=dict)
    overall_quality_score: float = 0.0
    
    # Consensus analysis
    expert_contributions: Dict[str, float] = field(default_factory=dict)
    response_clusters: List[List[str]] = field(default_factory=list)
    dissenting_opinions: List[ExpertResponse] = field(default_factory=list)
    
    # Uncertainty and reliability
    epistemic_uncertainty: float = 0.0  # Model uncertainty
    aleatoric_uncertainty: float = 0.0  # Data uncertainty
    reliability_score: float = 0.0
    
    # Bias and safety
    bias_analysis: Dict[str, float] = field(default_factory=dict)
    safety_assessment: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    consensus_time_ms: float = 0.0
    total_cost: float = 0.0
    efficiency_score: float = 0.0

class AdvancedConsensusEngine:
    """
    🧠 Advanced Consensus Engine
    
    Implements sophisticated consensus algorithms that consider multiple factors:
    - Expert reliability and historical performance
    - Response quality across multiple dimensions
    - Semantic similarity and clustering
    - Uncertainty quantification and calibration
    - Bias detection and mitigation
    - Safety assessment and filtering
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Algorithm weights and preferences
        self.algorithm_weights = {
            ConsensusAlgorithm.BAYESIAN_CONSENSUS: 0.2,
            ConsensusAlgorithm.MCDA_WEIGHTED: 0.2,
            ConsensusAlgorithm.SEMANTIC_CLUSTERING: 0.15,
            ConsensusAlgorithm.CONFIDENCE_CALIBRATED: 0.15,
            ConsensusAlgorithm.DYNAMIC_WEIGHTED: 0.15,
            ConsensusAlgorithm.QUALITY_AWARE: 0.1,
            ConsensusAlgorithm.HYBRID_ENSEMBLE: 0.05
        }
        
        # Quality metric weights
        self.quality_weights = {
            QualityMetric.RELEVANCE: 0.25,
            QualityMetric.ACCURACY: 0.20,
            QualityMetric.COHERENCE: 0.15,
            QualityMetric.COMPLETENESS: 0.15,
            QualityMetric.CLARITY: 0.10,
            QualityMetric.FACTUALITY: 0.10,
            QualityMetric.BIAS_SCORE: 0.03,
            QualityMetric.SAFETY_SCORE: 0.02
        }
        
        # Expert performance tracking
        self.expert_performance_history: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Semantic analysis tools
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        logger.info("🧠 Advanced Consensus Engine initialized")
    
    async def generate_consensus(
        self,
        expert_responses: List[ExpertResponse],
        query: str,
        algorithm: Optional[ConsensusAlgorithm] = None,
        quality_threshold: float = 7.0
    ) -> ConsensusResult:
        """
        Generate consensus using advanced algorithms.
        
        Args:
            expert_responses: List of expert responses with metadata
            query: Original query for context
            algorithm: Specific algorithm to use (None for auto-selection)
            quality_threshold: Minimum quality threshold
            
        Returns:
            Comprehensive consensus result
        """
        
        if not expert_responses:
            return self._create_empty_result()
        
        # Filter responses by quality threshold
        qualified_responses = [
            resp for resp in expert_responses
            if self._calculate_response_quality(resp) >= quality_threshold
        ]
        
        if not qualified_responses:
            qualified_responses = expert_responses  # Fallback to all responses
        
        # Auto-select algorithm if not specified
        if algorithm is None:
            algorithm = self._select_optimal_algorithm(qualified_responses, query)
        
        # Generate semantic embeddings
        await self._generate_semantic_embeddings(qualified_responses)
        
        # Apply consensus algorithm
        consensus_result = await self._apply_consensus_algorithm(
            algorithm, qualified_responses, query
        )
        
        # Post-process and enhance result
        enhanced_result = await self._enhance_consensus_result(
            consensus_result, qualified_responses, query
        )
        
        # Update expert performance tracking
        self._update_expert_performance(qualified_responses, enhanced_result)
        
        return enhanced_result
    
    def _calculate_response_quality(self, response: ExpertResponse) -> float:
        """Calculate overall quality score for a response."""
        
        if not response.quality_scores:
            # Fallback quality estimation
            return self._estimate_quality_heuristic(response)
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in self.quality_weights.items():
            if metric in response.quality_scores:
                weighted_score += response.quality_scores[metric] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 5.0
    
    def _estimate_quality_heuristic(self, response: ExpertResponse) -> float:
        """Estimate quality using heuristics when detailed scores unavailable."""
        
        quality_score = 5.0  # Base score
        
        # Length-based quality (reasonable length indicates thoughtfulness)
        text_length = len(response.response_text.split())
        if 10 <= text_length <= 200:
            quality_score += 1.0
        elif text_length > 200:
            quality_score += 0.5
        
        # Confidence-based adjustment
        quality_score += response.confidence * 2.0
        
        # Historical accuracy
        quality_score += response.historical_accuracy * 2.0
        
        # Expert tier bonus
        tier_bonuses = {
            "premium": 1.0,
            "specialized": 0.8,
            "serverless": 0.6,
            "local": 0.4
        }
        quality_score += tier_bonuses.get(response.expert_tier, 0.0)
        
        return min(quality_score, 10.0)
    
    def _select_optimal_algorithm(
        self,
        responses: List[ExpertResponse],
        query: str
    ) -> ConsensusAlgorithm:
        """Select optimal consensus algorithm based on response characteristics."""
        
        num_responses = len(responses)
        avg_confidence = np.mean([r.confidence for r in responses])
        quality_variance = np.var([self._calculate_response_quality(r) for r in responses])
        
        # Decision logic for algorithm selection
        if num_responses <= 2:
            return ConsensusAlgorithm.CONFIDENCE_CALIBRATED
        elif quality_variance > 2.0:  # High quality variance
            return ConsensusAlgorithm.QUALITY_AWARE
        elif avg_confidence < 0.7:  # Low average confidence
            return ConsensusAlgorithm.BAYESIAN_CONSENSUS
        elif num_responses >= 5:  # Many responses
            return ConsensusAlgorithm.SEMANTIC_CLUSTERING
        else:
            return ConsensusAlgorithm.MCDA_WEIGHTED
    
    async def _generate_semantic_embeddings(self, responses: List[ExpertResponse]):
        """Generates semantic embeddings for responses."""
        
        if len(responses) < 2:
            return
        
        try:
            # Extract response texts
            texts = [resp.response_text for resp in responses]
            
            # Generate TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Store embeddings in response objects
            for i, response in enumerate(responses):
                response.semantic_embedding = tfidf_matrix[i].toarray().flatten()
                
        except Exception as e:
            logger.warning(f"Failed to generate semantic embeddings: {e}")
    
    async def _apply_consensus_algorithm(
        self,
        algorithm: ConsensusAlgorithm,
        responses: List[ExpertResponse],
        query: str
    ) -> ConsensusResult:
        """Apply the specified consensus algorithm."""
        
        algorithm_methods = {
            ConsensusAlgorithm.BAYESIAN_CONSENSUS: self._bayesian_consensus,
            ConsensusAlgorithm.MCDA_WEIGHTED: self._mcda_weighted_consensus,
            ConsensusAlgorithm.SEMANTIC_CLUSTERING: self._semantic_clustering_consensus,
            ConsensusAlgorithm.CONFIDENCE_CALIBRATED: self._confidence_calibrated_consensus,
            ConsensusAlgorithm.DYNAMIC_WEIGHTED: self._dynamic_weighted_consensus,
            ConsensusAlgorithm.QUALITY_AWARE: self._quality_aware_consensus,
            ConsensusAlgorithm.HYBRID_ENSEMBLE: self._hybrid_ensemble_consensus
        }
        
        method = algorithm_methods.get(algorithm, self._mcda_weighted_consensus)
        return await method(responses, query)
    
    async def _bayesian_consensus(
        self,
        responses: List[ExpertResponse],
        query: str
    ) -> ConsensusResult:
        """Bayesian consensus with uncertainty quantification."""
        
        # Calculate Bayesian weights based on expert reliability and confidence
        bayesian_weights = []
        
        for response in responses:
            # Prior: historical accuracy
            prior = response.historical_accuracy
            
            # Likelihood: current confidence
            likelihood = response.confidence
            
            # Quality factor
            quality = self._calculate_response_quality(response)
            
            # Bayesian weight
            weight = prior * likelihood * (quality / 10.0)
            bayesian_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(bayesian_weights)
        if total_weight > 0:
            bayesian_weights = [w / total_weight for w in bayesian_weights]
        else:
            bayesian_weights = [1.0 / len(responses)] * len(responses)
        
        # Select response with highest Bayesian weight
        best_idx = np.argmax(bayesian_weights)
        best_response = responses[best_idx]
        
        # Calculate epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = 1.0 - max(bayesian_weights)
        
        # Calculate aleatoric uncertainty (data uncertainty)
        confidence_variance = np.var([r.confidence for r in responses])
        aleatoric_uncertainty = min(confidence_variance, 1.0)
        
        return ConsensusResult(
            final_response=best_response.response_text,
            consensus_confidence=max(bayesian_weights),
            algorithm_used=ConsensusAlgorithm.BAYESIAN_CONSENSUS,
            expert_contributions={
                resp.expert_id: weight 
                for resp, weight in zip(responses, bayesian_weights)
            },
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            reliability_score=1.0 - (epistemic_uncertainty + aleatoric_uncertainty) / 2
        )
    
    async def _mcda_weighted_consensus(
        self,
        responses: List[ExpertResponse],
        query: str
    ) -> ConsensusResult:
        """Multi-Criteria Decision Analysis weighted consensus."""
        
        # Define criteria and their weights
        criteria = {
            'quality': 0.35,
            'confidence': 0.25,
            'reliability': 0.20,
            'cost_efficiency': 0.10,
            'response_time': 0.10
        }
        
        # Calculate scores for each response across criteria
        mcda_scores = []
        
        for response in responses:
            scores = {}
            
            # Quality score (0-1)
            scores['quality'] = self._calculate_response_quality(response) / 10.0
            
            # Confidence score (already 0-1)
            scores['confidence'] = response.confidence
            
            # Reliability score (historical accuracy)
            scores['reliability'] = response.historical_accuracy
            
            # Cost efficiency (inverse of cost, normalized)
            max_cost = max([r.cost for r in responses]) if responses else 1.0
            scores['cost_efficiency'] = 1.0 - (response.cost / max_cost) if max_cost > 0 else 1.0
            
            # Response time efficiency (inverse of time, normalized)
            max_time = max([r.response_time_ms for r in responses]) if responses else 1000.0
            scores['response_time'] = 1.0 - (response.response_time_ms / max_time) if max_time > 0 else 1.0
            
            # Calculate weighted MCDA score
            mcda_score = sum(scores[criterion] * weight for criterion, weight in criteria.items())
            mcda_scores.append(mcda_score)
        
        # Select response with highest MCDA score
        best_idx = np.argmax(mcda_scores)
        best_response = responses[best_idx]
        
        return ConsensusResult(
            final_response=best_response.response_text,
            consensus_confidence=mcda_scores[best_idx],
            algorithm_used=ConsensusAlgorithm.MCDA_WEIGHTED,
            expert_contributions={
                resp.expert_id: score 
                for resp, score in zip(responses, mcda_scores)
            },
            efficiency_score=mcda_scores[best_idx]
        )
    
    async def _semantic_clustering_consensus(
        self,
        responses: List[ExpertResponse],
        query: str
    ) -> ConsensusResult:
        """Semantic clustering-based consensus."""
        
        if len(responses) < 3:
            return await self._confidence_calibrated_consensus(responses, query)
        
        # Check if embeddings are available
        embeddings = [r.semantic_embedding for r in responses if r.semantic_embedding is not None]
        
        if len(embeddings) < 3:
            return await self._confidence_calibrated_consensus(responses, query)
        
        # Perform K-means clustering
        n_clusters = min(3, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Group responses by cluster
        clusters = defaultdict(list)
        for i, (response, label) in enumerate(zip(responses, cluster_labels)):
            clusters[label].append((i, response))
        
        # Find the largest cluster (majority opinion)
        largest_cluster = max(clusters.values(), key=len)
        
        # Select best response from largest cluster based on quality
        best_response = max(
            largest_cluster,
            key=lambda x: self._calculate_response_quality(x[1])
        )[1]
        
        # Calculate consensus confidence based on cluster size
        consensus_confidence = len(largest_cluster) / len(responses)
        
        # Identify dissenting opinions (other clusters)
        dissenting_opinions = []
        for label, cluster_responses in clusters.items():
            if len(cluster_responses) < len(largest_cluster):
                dissenting_opinions.extend([resp for _, resp in cluster_responses])
        
        return ConsensusResult(
            final_response=best_response.response_text,
            consensus_confidence=consensus_confidence,
            algorithm_used=ConsensusAlgorithm.SEMANTIC_CLUSTERING,
            response_clusters=[[resp.expert_id for _, resp in cluster] for cluster in clusters.values()],
            dissenting_opinions=dissenting_opinions,
            expert_contributions={
                resp.expert_id: 1.0 if resp == best_response else 0.5
                for resp in responses
            }
        )
    
    async def _confidence_calibrated_consensus(
        self,
        responses: List[ExpertResponse],
        query: str
    ) -> ConsensusResult:
        """Confidence-calibrated consensus with uncertainty estimation."""
        
        # Calibrate confidence scores based on historical performance
        calibrated_confidences = []
        
        for response in responses:
            # Simple calibration: adjust confidence by historical accuracy
            calibrated_confidence = response.confidence * response.historical_accuracy
            
            # Apply quality adjustment
            quality_factor = self._calculate_response_quality(response) / 10.0
            calibrated_confidence *= quality_factor
            
            calibrated_confidences.append(calibrated_confidence)
        
        # Select response with highest calibrated confidence
        best_idx = np.argmax(calibrated_confidences)
        best_response = responses[best_idx]
        
        # Calculate uncertainty based on confidence distribution
        confidence_entropy = -sum(
            p * math.log(p + 1e-10) for p in calibrated_confidences
        ) / math.log(len(calibrated_confidences))
        
        uncertainty = confidence_entropy / math.log(len(responses)) if len(responses) > 1 else 0.0
        
        return ConsensusResult(
            final_response=best_response.response_text,
            consensus_confidence=calibrated_confidences[best_idx],
            algorithm_used=ConsensusAlgorithm.CONFIDENCE_CALIBRATED,
            expert_contributions={
                resp.expert_id: conf 
                for resp, conf in zip(responses, calibrated_confidences)
            },
            epistemic_uncertainty=uncertainty,
            reliability_score=1.0 - uncertainty
        )
    
    async def _dynamic_weighted_consensus(
        self,
        responses: List[ExpertResponse],
        query: str
    ) -> ConsensusResult:
        """Dynamic weighted consensus with adaptive expert weights."""
        
        # Calculate dynamic weights based on multiple factors
        dynamic_weights = []
        
        for response in responses:
            weight = 1.0
            
            # Historical performance weight
            perf_history = self.expert_performance_history.get(response.expert_id, {})
            avg_performance = np.mean(list(perf_history.values())) if perf_history else 0.8
            weight *= avg_performance
            
            # Current confidence weight
            weight *= response.confidence
            
            # Quality weight
            quality_score = self._calculate_response_quality(response)
            weight *= quality_score / 10.0
            
            # Domain specialization bonus
            if any(domain in query.lower() for domain in response.domain_specialization):
                weight *= 1.2
            
            # Cost efficiency factor
            if response.cost > 0:
                cost_efficiency = 1.0 / (1.0 + response.cost * 100)  # Normalize cost
                weight *= cost_efficiency
            
            dynamic_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(dynamic_weights)
        if total_weight > 0:
            dynamic_weights = [w / total_weight for w in dynamic_weights]
        else:
            dynamic_weights = [1.0 / len(responses)] * len(responses)
        
        # Weighted combination approach - select best weighted response
        best_idx = np.argmax(dynamic_weights)
        best_response = responses[best_idx]
        
        return ConsensusResult(
            final_response=best_response.response_text,
            consensus_confidence=max(dynamic_weights),
            algorithm_used=ConsensusAlgorithm.DYNAMIC_WEIGHTED,
            expert_contributions={
                resp.expert_id: weight 
                for resp, weight in zip(responses, dynamic_weights)
            },
            efficiency_score=max(dynamic_weights)
        )
    
    async def _quality_aware_consensus(
        self,
        responses: List[ExpertResponse],
        query: str
    ) -> ConsensusResult:
        """Quality-aware consensus focusing on response quality metrics."""
        
        # Calculate comprehensive quality scores
        quality_scores = []
        
        for response in responses:
            if response.quality_scores:
                # Use detailed quality metrics if available
                weighted_quality = sum(
                    response.quality_scores.get(metric, 5.0) * weight
                    for metric, weight in self.quality_weights.items()
                )
            else:
                # Use heuristic quality estimation
                weighted_quality = self._estimate_quality_heuristic(response)
            
            quality_scores.append(weighted_quality)
        
        # Select response with highest quality score
        best_idx = np.argmax(quality_scores)
        best_response = responses[best_idx]
        
        # Calculate quality-based confidence
        max_quality = max(quality_scores)
        quality_confidence = max_quality / 10.0
        
        # Assess quality consistency
        quality_std = np.std(quality_scores)
        quality_consistency = 1.0 / (1.0 + quality_std)
        
        return ConsensusResult(
            final_response=best_response.response_text,
            consensus_confidence=quality_confidence,
            algorithm_used=ConsensusAlgorithm.QUALITY_AWARE,
            quality_scores={
                QualityMetric.ACCURACY: max_quality / 10.0,
                QualityMetric.COHERENCE: quality_consistency,
                QualityMetric.RELEVANCE: quality_confidence
            },
            overall_quality_score=max_quality,
            expert_contributions={
                resp.expert_id: score / sum(quality_scores)
                for resp, score in zip(responses, quality_scores)
            },
            reliability_score=quality_consistency
        )
    
    async def _hybrid_ensemble_consensus(
        self,
        responses: List[ExpertResponse],
        query: str
    ) -> ConsensusResult:
        """Hybrid ensemble consensus combining multiple algorithms."""
        
        # Apply multiple algorithms
        algorithms_to_combine = [
            ConsensusAlgorithm.BAYESIAN_CONSENSUS,
            ConsensusAlgorithm.MCDA_WEIGHTED,
            ConsensusAlgorithm.CONFIDENCE_CALIBRATED
        ]
        
        algorithm_results = []
        for algorithm in algorithms_to_combine:
            try:
                result = await self._apply_consensus_algorithm(algorithm, responses, query)
                algorithm_results.append(result)
            except Exception as e:
                logger.warning(f"Algorithm {algorithm} failed: {e}")
        
        if not algorithm_results:
            # Fallback to simple consensus
            return await self._confidence_calibrated_consensus(responses, query)
        
        # Combine results using voting
        response_votes = defaultdict(float)
        confidence_scores = defaultdict(list)
        
        for result in algorithm_results:
            response_votes[result.final_response] += result.consensus_confidence
            confidence_scores[result.final_response].append(result.consensus_confidence)
        
        # Select response with highest combined vote
        best_response = max(response_votes.keys(), key=lambda k: response_votes[k])
        combined_confidence = np.mean(confidence_scores[best_response])
        
        # Find the original response object
        selected_response_obj = next(
            (r for r in responses if r.response_text == best_response),
            responses[0]
        )
        
        return ConsensusResult(
            final_response=best_response,
            consensus_confidence=combined_confidence,
            algorithm_used=ConsensusAlgorithm.HYBRID_ENSEMBLE,
            expert_contributions={
                resp.expert_id: response_votes.get(resp.response_text, 0.0)
                for resp in responses
            },
            reliability_score=combined_confidence
        )
    
    async def _enhance_consensus_result(
        self,
        result: ConsensusResult,
        responses: List[ExpertResponse],
        query: str
    ) -> ConsensusResult:
        """Enhance consensus result with additional analysis."""
        
        # Calculate overall quality score if not set
        if result.overall_quality_score == 0.0:
            selected_response = next(
                (r for r in responses if r.response_text == result.final_response),
                None
            )
            if selected_response:
                result.overall_quality_score = self._calculate_response_quality(selected_response)
        
        # Perform bias analysis
        result.bias_analysis = await self._analyze_bias(result.final_response, responses)
        
        # Perform safety assessment
        result.safety_assessment = await self._assess_safety(result.final_response)
        
        # Calculate efficiency score
        if result.efficiency_score == 0.0:
            total_cost = sum(r.cost for r in responses)
            avg_time = np.mean([r.response_time_ms for r in responses])
            result.efficiency_score = result.overall_quality_score / (1.0 + total_cost + avg_time / 1000.0)
        
        return result
    
    async def _analyze_bias(
        self,
        response: str,
        all_responses: List[ExpertResponse]
    ) -> Dict[str, float]:
        """Analyze potential bias in the consensus response."""
        
        # Simple bias indicators (in production, use specialized bias detection models)
        bias_indicators = {
            'gender_bias': 0.0,
            'racial_bias': 0.0,
            'cultural_bias': 0.0,
            'confirmation_bias': 0.0,
            'selection_bias': 0.0
        }
        
        # Check for gender-biased language
        gender_terms = ['he', 'she', 'his', 'her', 'man', 'woman', 'male', 'female']
        gender_count = sum(1 for term in gender_terms if term.lower() in response.lower())
        bias_indicators['gender_bias'] = min(gender_count / len(response.split()) * 10, 1.0)
        
        # Check for confirmation bias (response similarity to majority)
        if len(all_responses) > 1:
            response_similarities = []
            for resp in all_responses:
                if resp.semantic_embedding is not None:
                    # Calculate similarity (simplified)
                    similarity = len(set(response.lower().split()) & 
                                   set(resp.response_text.lower().split())) / len(set(response.lower().split()))
                    response_similarities.append(similarity)
            
            if response_similarities:
                bias_indicators['confirmation_bias'] = np.mean(response_similarities)
        
        return bias_indicators
    
    async def _assess_safety(self, response: str) -> Dict[str, float]:
        """Assess safety of the consensus response."""
        
        # Simple safety assessment (in production, use specialized safety models)
        safety_scores = {
            'toxicity': 0.0,
            'harmful_content': 0.0,
            'misinformation_risk': 0.0,
            'privacy_violation': 0.0,
            'overall_safety': 1.0
        }
        
        # Check for potentially harmful keywords
        harmful_keywords = [
            'violence', 'harm', 'dangerous', 'illegal', 'unsafe',
            'toxic', 'hate', 'discriminat', 'threat'
        ]
        
        harmful_count = sum(1 for keyword in harmful_keywords 
                          if keyword.lower() in response.lower())
        
        if harmful_count > 0:
            safety_scores['harmful_content'] = min(harmful_count / 10.0, 1.0)
            safety_scores['overall_safety'] = 1.0 - safety_scores['harmful_content']
        
        return safety_scores
    
    def _update_expert_performance(
        self,
        responses: List[ExpertResponse],
        result: ConsensusResult
    ):
        """Update expert performance history based on consensus result."""
        
        selected_expert_id = None
        for response in responses:
            if response.response_text == result.final_response:
                selected_expert_id = response.expert_id
                break
        
        # Update performance scores
        for response in responses:
            expert_id = response.expert_id
            
            if expert_id not in self.expert_performance_history:
                self.expert_performance_history[expert_id] = {}
            
            # Performance score based on whether response was selected and quality
            if response.expert_id == selected_expert_id:
                performance_score = result.overall_quality_score / 10.0
            else:
                performance_score = self._calculate_response_quality(response) / 10.0 * 0.8
            
            # Update rolling average
            current_avg = self.expert_performance_history[expert_id].get('avg_performance', 0.8)
            new_avg = current_avg * 0.9 + performance_score * 0.1  # Exponential moving average
            self.expert_performance_history[expert_id]['avg_performance'] = new_avg
            
            # Update other metrics
            self.expert_performance_history[expert_id]['last_quality'] = self._calculate_response_quality(response)
            self.expert_performance_history[expert_id]['selection_rate'] = (
                self.expert_performance_history[expert_id].get('selection_rate', 0.5) * 0.9 +
                (1.0 if response.expert_id == selected_expert_id else 0.0) * 0.1
            )
    
    def _create_empty_result(self) -> ConsensusResult:
        """Creates empty consensus result for error cases."""
        
        return ConsensusResult(
            final_response="No valid responses available for consensus.",
            consensus_confidence=0.0,
            algorithm_used=ConsensusAlgorithm.CONFIDENCE_CALIBRATED,
            overall_quality_score=0.0,
            reliability_score=0.0
        )
    
    def get_algorithm_statistics(self) -> Dict[str, Any]:
        """Get statistics about algorithm usage and performance."""
        
        return {
            'algorithm_weights': self.algorithm_weights,
            'quality_weights': self.quality_weights,
            'expert_performance_history': dict(self.expert_performance_history),
            'total_experts_tracked': len(self.expert_performance_history),
            'avg_expert_performance': np.mean([
                data.get('avg_performance', 0.8) 
                for data in self.expert_performance_history.values()
            ]) if self.expert_performance_history else 0.0
        }
    
    def update_algorithm_weights(self, performance_feedback: Dict[ConsensusAlgorithm, float]):
        """Update algorithm weights based on performance feedback."""
        
        for algorithm, performance in performance_feedback.items():
            if algorithm in self.algorithm_weights:
                current_weight = self.algorithm_weights[algorithm]
                # Simple adaptive update
                new_weight = current_weight * 0.9 + (performance / 10.0) * 0.1
                self.algorithm_weights[algorithm] = max(0.01, min(1.0, new_weight))
        
        # Normalize weights
        total_weight = sum(self.algorithm_weights.values())
        if total_weight > 0:
            for algorithm in self.algorithm_weights:
                self.algorithm_weights[algorithm] /= total_weight


# Utility functions
def create_expert_response(
    expert_id: str,
    response_text: str,
    confidence: float = 0.8,
    **kwargs
) -> ExpertResponse:
    """Utility function to create ExpertResponse objects."""
    
    return ExpertResponse(
        expert_id=expert_id,
        response_text=response_text,
        confidence=confidence,
        **kwargs
    )

def calculate_consensus_diversity(responses: List[ExpertResponse]) -> float:
    """Calculate diversity score of expert responses."""
    
    if len(responses) <= 1:
        return 0.0
    
    # Simple diversity based on response length variance
    lengths = [len(resp.response_text.split()) for resp in responses]
    length_diversity = np.std(lengths) / (np.mean(lengths) + 1)
    
    # Confidence diversity
    confidences = [resp.confidence for resp in responses]
    confidence_diversity = np.std(confidences)
    
    return (length_diversity + confidence_diversity) / 2


# Export main components
__all__ = [
    'AdvancedConsensusEngine',
    'ExpertResponse',
    'ConsensusResult',
    'ConsensusAlgorithm',
    'QualityMetric',
    'create_expert_response',
    'calculate_consensus_diversity'
]


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create consensus engine
        engine = AdvancedConsensusEngine()
        
        # Create sample expert responses
        responses = [
            create_expert_response(
                expert_id="math_expert",
                response_text="The solution involves calculus and requires integration by parts.",
                confidence=0.9,
                quality_scores={
                    QualityMetric.ACCURACY: 9.0,
                    QualityMetric.RELEVANCE: 8.5,
                    QualityMetric.COHERENCE: 8.8
                },
                expert_tier="specialized",
                domain_specialization=["mathematics", "calculus"]
            ),
            create_expert_response(
                expert_id="general_expert",
                response_text="This is a mathematical problem that can be solved using standard techniques.",
                confidence=0.7,
                quality_scores={
                    QualityMetric.ACCURACY: 7.5,
                    QualityMetric.RELEVANCE: 8.0,
                    QualityMetric.COHERENCE: 8.2
                },
                expert_tier="local",
                domain_specialization=["general"]
            ),
            create_expert_response(
                expert_id="premium_expert",
                response_text="To solve this calculus problem, apply integration by parts: ∫u dv = uv - ∫v du.",
                confidence=0.95,
                quality_scores={
                    QualityMetric.ACCURACY: 9.5,
                    QualityMetric.RELEVANCE: 9.2,
                    QualityMetric.COHERENCE: 9.0,
                    QualityMetric.COMPLETENESS: 8.8
                },
                expert_tier="premium",
                domain_specialization=["mathematics", "advanced_calculus"],
                cost=0.015
            )
        ]
        
        # Generate consensus
        query = "How do I solve this integration problem?"
        result = await engine.generate_consensus(
            expert_responses=responses,
            query=query,
            algorithm=ConsensusAlgorithm.HYBRID_ENSEMBLE
        )
        
        print(f"Query: {query}")
        print(f"Consensus Response: {result.final_response}")
        print(f"Algorithm Used: {result.algorithm_used.value}")
        print(f"Confidence: {result.consensus_confidence:.2f}")
        print(f"Quality Score: {result.overall_quality_score:.1f}")
        print(f"Reliability: {result.reliability_score:.2f}")
        
        if result.expert_contributions:
            print("\nExpert Contributions:")
            for expert_id, contribution in result.expert_contributions.items():
                print(f"  {expert_id}: {contribution:.3f}")
        
        if result.bias_analysis:
            print(f"\nBias Analysis: {result.bias_analysis}")
        
        if result.safety_assessment:
            print(f"Safety Assessment: {result.safety_assessment}")
    
    import asyncio
    asyncio.run(main())