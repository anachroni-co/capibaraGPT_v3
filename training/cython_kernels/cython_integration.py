"""
Cython Kernels Integration for Meta-Consensus System
Provides fallback to pure Python implementations when Cython modules are not available
"""

import logging
import numpy as np
from typing import Dict, Tuple, Any, Optional
import warnings

logger = logging.getLogger(__name__)

# Try to import compiled Cython modules
CYTHON_AVAILABLE = False
try:
    from . import consensus_kernels
    from . import routing_kernels
    from . import similarity_kernels
    from . import aggregation_kernels
    CYTHON_AVAILABLE = True
    logger.info("✅ Cython kernels loaded successfully - 20x speedup enabled")
except ImportError as e:
    logger.warning(f"⚠️ Cython kernels not available: {e}")
    logger.warning("🔄 Falling back to pure Python implementations")
    consensus_kernels = None
    routing_kernels = None
    similarity_kernels = None
    aggregation_kernels = None


class CythonKernelManager:
    """Manager for Cython kernel operations with Python fallbacks."""
    
    def __init__(self):
        self.cython_available = CYTHON_AVAILABLE
        self.fallback_count = 0
        
    def fast_consensus_calculation(self, response_embeddings, weights, similarity_threshold=0.8):
        """Fast consensus calculation with Cython optimization."""
        if self.cython_available and consensus_kernels:
            try:
                return consensus_kernels.fast_consensus_calculation(
                    response_embeddings.astype(np.float32),
                    weights.astype(np.float32),
                    float(similarity_threshold)
                )
            except Exception as e:
                logger.warning(f"Cython consensus calculation failed: {e}, falling back to Python")
                self.fallback_count += 1
        
        # Python fallback
        return self._python_consensus_calculation(response_embeddings, weights, similarity_threshold)
    
    def fast_expert_routing_scores(self, query_embeddings, expert_embeddings, expert_weights, quality_thresholds):
        """Fast expert routing with Cython optimization."""
        if self.cython_available and routing_kernels:
            try:
                return routing_kernels.fast_expert_routing_scores(
                    query_embeddings.astype(np.float32),
                    expert_embeddings.astype(np.float32),
                    expert_weights.astype(np.float32),
                    quality_thresholds.astype(np.float32)
                )
            except Exception as e:
                logger.warning(f"Cython routing failed: {e}, falling back to Python")
                self.fallback_count += 1
        
        # Python fallback
        return self._python_expert_routing_scores(query_embeddings, expert_embeddings, expert_weights, quality_thresholds)
    
    def fast_cosine_similarity_matrix(self, embeddings_a, embeddings_b):
        """Fast cosine similarity matrix with Cython optimization."""
        if self.cython_available and similarity_kernels:
            try:
                return similarity_kernels.fast_cosine_similarity_matrix(
                    embeddings_a.astype(np.float32),
                    embeddings_b.astype(np.float32)
                )
            except Exception as e:
                logger.warning(f"Cython similarity failed: {e}, falling back to Python")
                self.fallback_count += 1
        
        # Python fallback
        return self._python_cosine_similarity_matrix(embeddings_a, embeddings_b)
    
    def fast_weighted_aggregation(self, response_embeddings, response_weights, quality_scores, quality_threshold=0.7):
        """Fast weighted aggregation with Cython optimization."""
        if self.cython_available and aggregation_kernels:
            try:
                return aggregation_kernels.fast_weighted_aggregation(
                    response_embeddings.astype(np.float32),
                    response_weights.astype(np.float32),
                    quality_scores.astype(np.float32),
                    float(quality_threshold)
                )
            except Exception as e:
                logger.warning(f"Cython aggregation failed: {e}, falling back to Python")
                self.fallback_count += 1
        
        # Python fallback
        return self._python_weighted_aggregation(response_embeddings, response_weights, quality_scores, quality_threshold)
    
    def fast_attention_consensus(self, attention_weights, value_matrices, expert_confidences):
        """Fast attention consensus with Cython optimization."""
        if self.cython_available and consensus_kernels:
            try:
                return consensus_kernels.fast_attention_consensus(
                    attention_weights.astype(np.float32),
                    value_matrices.astype(np.float32),
                    expert_confidences.astype(np.float32)
                )
            except Exception as e:
                logger.warning(f"Cython attention consensus failed: {e}, falling back to Python")
                self.fallback_count += 1
        
        # Python fallback
        return self._python_attention_consensus(attention_weights, value_matrices, expert_confidences)
    
    def fast_quality_assessment(self, response_embeddings, reference_embeddings, response_lengths, 
                               diversity_weight=0.3, coherence_weight=0.4, relevance_weight=0.3):
        """Fast quality assessment with Cython optimization."""
        if self.cython_available and consensus_kernels:
            try:
                return consensus_kernels.fast_quality_assessment(
                    response_embeddings.astype(np.float32),
                    reference_embeddings.astype(np.float32),
                    response_lengths.astype(np.float32),
                    float(diversity_weight), float(coherence_weight), float(relevance_weight)
                )
            except Exception as e:
                logger.warning(f"Cython quality assessment failed: {e}, falling back to Python")
                self.fallback_count += 1
        
        # Python fallback
        return self._python_quality_assessment(response_embeddings, reference_embeddings, response_lengths,
                                             diversity_weight, coherence_weight, relevance_weight)
    
    # Python fallback implementations
    def _python_consensus_calculation(self, response_embeddings, weights, similarity_threshold):
        """Pure Python consensus calculation fallback."""
        num_responses, seq_len, hidden_dim = response_embeddings.shape
        consensus = np.zeros((seq_len, hidden_dim), dtype=np.float32)
        
        for seq_idx in range(seq_len):
            for dim_idx in range(hidden_dim):
                weighted_value = 0.0
                total_weight = 0.0
                
                for i in range(num_responses):
                    # Calculate similarity with other responses
                    similarities = []
                    for j in range(num_responses):
                        if i != j:
                            # Cosine similarity
                            dot_prod = np.dot(response_embeddings[i, seq_idx], response_embeddings[j, seq_idx])
                            norm_i = np.linalg.norm(response_embeddings[i, seq_idx])
                            norm_j = np.linalg.norm(response_embeddings[j, seq_idx])
                            if norm_i > 0 and norm_j > 0:
                                similarities.append(dot_prod / (norm_i * norm_j))
                    
                    avg_similarity = np.mean(similarities) if similarities else 0.0
                    
                    if avg_similarity >= similarity_threshold:
                        weighted_value += weights[i] * avg_similarity * response_embeddings[i, seq_idx, dim_idx]
                        total_weight += weights[i] * avg_similarity
                
                if total_weight > 0:
                    consensus[seq_idx, dim_idx] = weighted_value / total_weight
        
        return consensus
    
    def _python_expert_routing_scores(self, query_embeddings, expert_embeddings, expert_weights, quality_thresholds):
        """Pure Python expert routing fallback."""
        num_experts = expert_embeddings.shape[0]
        scores = np.zeros(num_experts, dtype=np.float32)
        
        for expert_idx in range(num_experts):
            # Calculate cosine similarity between query and expert
            similarities = []
            for seq_idx in range(query_embeddings.shape[0]):
                dot_prod = np.dot(query_embeddings[seq_idx], expert_embeddings[expert_idx, seq_idx])
                norm_query = np.linalg.norm(query_embeddings[seq_idx])
                norm_expert = np.linalg.norm(expert_embeddings[expert_idx, seq_idx])
                if norm_query > 0 and norm_expert > 0:
                    similarities.append(dot_prod / (norm_query * norm_expert))
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            score = avg_similarity * expert_weights[expert_idx]
            
            if score >= quality_thresholds[expert_idx]:
                scores[expert_idx] = score
        
        return scores
    
    def _python_cosine_similarity_matrix(self, embeddings_a, embeddings_b):
        """Pure Python cosine similarity matrix fallback."""
        # Normalize embeddings
        norms_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True)
        norms_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms_a = np.where(norms_a == 0, 1, norms_a)
        norms_b = np.where(norms_b == 0, 1, norms_b)
        
        normalized_a = embeddings_a / norms_a
        normalized_b = embeddings_b / norms_b
        
        # Calculate cosine similarity
        return np.dot(normalized_a, normalized_b.T).astype(np.float32)
    
    def _python_weighted_aggregation(self, response_embeddings, response_weights, quality_scores, quality_threshold):
        """Pure Python weighted aggregation fallback."""
        num_responses, seq_len, hidden_dim = response_embeddings.shape
        aggregated = np.zeros((seq_len, hidden_dim), dtype=np.float32)
        
        for seq_idx in range(seq_len):
            for dim_idx in range(hidden_dim):
                total_weight = 0.0
                
                for response_idx in range(num_responses):
                    if quality_scores[response_idx] >= quality_threshold:
                        combined_weight = response_weights[response_idx] * quality_scores[response_idx]
                        aggregated[seq_idx, dim_idx] += (combined_weight * 
                                                       response_embeddings[response_idx, seq_idx, dim_idx])
                        total_weight += combined_weight
                
                if total_weight > 0:
                    aggregated[seq_idx, dim_idx] /= total_weight
        
        return aggregated
    
    def _python_attention_consensus(self, attention_weights, value_matrices, expert_confidences):
        """Pure Python attention consensus fallback."""
        num_experts, seq_len, _ = attention_weights.shape
        hidden_dim = value_matrices.shape[3]
        consensus = np.zeros((seq_len, hidden_dim), dtype=np.float32)
        
        total_confidence = np.sum(expert_confidences)
        
        for i in range(seq_len):
            for dim_idx in range(hidden_dim):
                weighted_value = 0.0
                
                for expert_idx in range(num_experts):
                    attention_value = np.sum(attention_weights[expert_idx, i, :] * 
                                           value_matrices[expert_idx, i, :, dim_idx])
                    weighted_value += attention_value * expert_confidences[expert_idx]
                
                if total_confidence > 0:
                    consensus[i, dim_idx] = weighted_value / total_confidence
        
        return consensus
    
    def _python_quality_assessment(self, response_embeddings, reference_embeddings, response_lengths,
                                 diversity_weight, coherence_weight, relevance_weight):
        """Pure Python quality assessment fallback."""
        num_responses = response_embeddings.shape[0]
        quality_scores = np.zeros(num_responses, dtype=np.float32)
        
        diversity_scores = np.zeros(num_responses, dtype=np.float32)
        coherence_scores = np.zeros(num_responses, dtype=np.float32)
        relevance_scores = np.zeros(num_responses, dtype=np.float32)
        
        for i in range(num_responses):
            # Diversity score
            similarities = []
            for j in range(num_responses):
                if i != j:
                    sim = np.dot(response_embeddings[i], response_embeddings[j]) / (
                        np.linalg.norm(response_embeddings[i]) * np.linalg.norm(response_embeddings[j]) + 1e-8)
                    similarities.append(1.0 - sim)  # Distance
            diversity_scores[i] = np.mean(similarities) if similarities else 0.0
            
            # Coherence score (embedding norm)
            coherence_scores[i] = np.linalg.norm(response_embeddings[i])
            
            # Relevance score
            if len(reference_embeddings) > 0:
                similarities = []
                for ref_emb in reference_embeddings:
                    sim = np.dot(response_embeddings[i], ref_emb) / (
                        np.linalg.norm(response_embeddings[i]) * np.linalg.norm(ref_emb) + 1e-8)
                    similarities.append(sim)
                relevance_scores[i] = np.mean(similarities)
            
            # Length penalty
            length_penalty = 1.0
            if response_lengths[i] < 10:
                length_penalty = 0.5
            elif response_lengths[i] > 1000:
                length_penalty = 0.8
            
            # Combine metrics
            quality_scores[i] = (diversity_scores[i] * diversity_weight +
                               coherence_scores[i] * coherence_weight +
                               relevance_scores[i] * relevance_weight) * length_penalty
        
        detailed_metrics = {
            'diversity': diversity_scores,
            'coherence': coherence_scores,
            'relevance': relevance_scores
        }
        
        return quality_scores, detailed_metrics
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'cython_available': self.cython_available,
            'fallback_count': self.fallback_count,
            'performance_mode': 'Cython (20x speedup)' if self.cython_available else 'Python fallback'
        }


# Global kernel manager instance
kernel_manager = CythonKernelManager()


def compile_cython_kernels():
    """Compile Cython kernels if not already compiled."""
    import subprocess
    import sys
    import os
    
    kernel_dir = os.path.dirname(__file__)
    setup_path = os.path.join(kernel_dir, 'setup.py')
    
    if os.path.exists(setup_path):
        try:
            logger.info("🔧 Compiling Cython kernels...")
            result = subprocess.run([
                sys.executable, setup_path, 'build_ext', '--inplace'
            ], cwd=kernel_dir, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Cython kernels compiled successfully")
                return True
            else:
                logger.error(f"❌ Cython compilation failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Cython compilation error: {e}")
            return False
    else:
        logger.warning("⚠️ Cython setup.py not found")
        return False


# Auto-compile on import if needed
if not CYTHON_AVAILABLE:
    logger.info("🔄 Attempting to compile Cython kernels...")
    if compile_cython_kernels():
        # Try to reload modules
        try:
            from . import consensus_kernels
            from . import routing_kernels
            from . import similarity_kernels
            from . import aggregation_kernels
            CYTHON_AVAILABLE = True
            kernel_manager.cython_available = True
            logger.info("✅ Cython kernels compiled and loaded successfully")
        except ImportError:
            logger.warning("⚠️ Could not load compiled Cython kernels")