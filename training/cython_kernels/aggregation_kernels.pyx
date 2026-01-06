# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=False

"""
High-performance Cython kernels for response aggregation and consensus
Optimized for weighted aggregation and ensemble methods
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, log, sqrt, fabs, tanh, pow
from libc.stdlib cimport malloc, free
from cython.parallel import prange
cimport cython

ctypedef cnp.float32_t DTYPE_t
ctypedef cnp.int32_t INT_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] fast_weighted_aggregation(
    cnp.ndarray[DTYPE_t, ndim=3] response_embeddings,
    cnp.ndarray[DTYPE_t, ndim=1] response_weights,
    cnp.ndarray[DTYPE_t, ndim=1] quality_scores,
    DTYPE_t quality_threshold=0.7
):
    """
    Fast weighted aggregation of response embeddings with quality filtering.
    
    Args:
        response_embeddings: Shape (num_responses, seq_len, hidden_dim)
        response_weights: Shape (num_responses,)
        quality_scores: Shape (num_responses,)
        quality_threshold: Minimum quality for inclusion
        
    Returns:
        aggregated_embeddings: Shape (seq_len, hidden_dim)
    """
    cdef int num_responses = response_embeddings.shape[0]
    cdef int seq_len = response_embeddings.shape[1]
    cdef int hidden_dim = response_embeddings.shape[2]
    
    cdef cnp.ndarray[DTYPE_t, ndim=2] aggregated = np.zeros((seq_len, hidden_dim), dtype=np.float32)
    
    cdef int response_idx, seq_idx, dim_idx
    cdef DTYPE_t combined_weight, total_weight, final_weight
    
    # Aggregate embeddings with quality filtering
    for seq_idx in range(seq_len):
        for dim_idx in range(hidden_dim):
            total_weight = 0.0
            
            for response_idx in range(num_responses):
                # Check quality threshold
                if quality_scores[response_idx] >= quality_threshold:
                    # Combine response weight and quality score
                    combined_weight = response_weights[response_idx] * quality_scores[response_idx]
                    
                    aggregated[seq_idx, dim_idx] += (combined_weight * 
                        response_embeddings[response_idx, seq_idx, dim_idx])
                    total_weight += combined_weight
            
            # Normalize by total weight
            if total_weight > 0:
                aggregated[seq_idx, dim_idx] /= total_weight
    
    return aggregated


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] fast_consensus_voting(
    cnp.ndarray[DTYPE_t, ndim=3] response_embeddings,
    cnp.ndarray[DTYPE_t, ndim=1] confidence_scores,
    cnp.ndarray[DTYPE_t, ndim=1] expert_priorities,
    DTYPE_t consensus_threshold=0.6,
    int min_votes=2
):
    """
    Fast consensus voting mechanism for response selection.
    
    Args:
        response_embeddings: Shape (num_responses, seq_len, hidden_dim)
        confidence_scores: Shape (num_responses,)
        expert_priorities: Shape (num_responses,)
        consensus_threshold: Minimum agreement threshold
        min_votes: Minimum number of votes for consensus
        
    Returns:
        consensus_embeddings: Shape (seq_len, hidden_dim)
    """
    cdef int num_responses = response_embeddings.shape[0]
    cdef int seq_len = response_embeddings.shape[1]
    cdef int hidden_dim = response_embeddings.shape[2]
    
    cdef cnp.ndarray[DTYPE_t, ndim=2] consensus = np.zeros((seq_len, hidden_dim), dtype=np.float32)
    cdef cnp.ndarray[DTYPE_t, ndim=1] vote_weights = np.zeros(num_responses, dtype=np.float32)
    
    cdef int response_idx, seq_idx, dim_idx, vote_count
    cdef DTYPE_t weighted_vote, total_votes, consensus_strength
    
    # Calculate vote weights
    for response_idx in range(num_responses):
        vote_weights[response_idx] = confidence_scores[response_idx] * expert_priorities[response_idx]
    
    # Apply consensus voting
    for seq_idx in range(seq_len):
        for dim_idx in range(hidden_dim):
            total_votes = 0.0
            vote_count = 0
            
            for response_idx in range(num_responses):
                # Check if this response meets consensus threshold
                consensus_strength = 0.0
                
                # Calculate agreement with other responses
                for j in range(num_responses):
                    if response_idx != j:
                        # Simple agreement based on embedding similarity
                        if (fabs(response_embeddings[response_idx, seq_idx, dim_idx] - 
                               response_embeddings[j, seq_idx, dim_idx]) < 0.1):
                            consensus_strength += vote_weights[j]
                
                # Normalize consensus strength
                if num_responses > 1:
                    consensus_strength /= (num_responses - 1)
                
                # Include vote if consensus threshold is met
                if consensus_strength >= consensus_threshold:
                    weighted_vote = vote_weights[response_idx] * response_embeddings[response_idx, seq_idx, dim_idx]
                    consensus[seq_idx, dim_idx] += weighted_vote
                    total_votes += vote_weights[response_idx]
                    vote_count += 1
            
            # Apply consensus only if minimum votes are met
            if vote_count >= min_votes and total_votes > 0:
                consensus[seq_idx, dim_idx] /= total_votes
            else:
                # Fallback to weighted average of all responses
                total_votes = 0.0
                for response_idx in range(num_responses):
                    consensus[seq_idx, dim_idx] += (vote_weights[response_idx] * 
                        response_embeddings[response_idx, seq_idx, dim_idx])
                    total_votes += vote_weights[response_idx]
                
                if total_votes > 0:
                    consensus[seq_idx, dim_idx] /= total_votes
    
    return consensus


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] fast_ensemble_aggregation(
    cnp.ndarray[DTYPE_t, ndim=4] expert_outputs,
    cnp.ndarray[DTYPE_t, ndim=2] expert_weights,
    cnp.ndarray[DTYPE_t, ndim=1] ensemble_weights,
    DTYPE_t temperature=1.0
):
    """
    Fast ensemble aggregation with temperature scaling.
    
    Args:
        expert_outputs: Shape (num_experts, num_responses, seq_len, hidden_dim)
        expert_weights: Shape (num_experts, num_responses)
        ensemble_weights: Shape (num_experts,)
        temperature: Temperature for ensemble combination
        
    Returns:
        ensemble_output: Shape (seq_len, hidden_dim)
    """
    cdef int num_experts = expert_outputs.shape[0]
    cdef int num_responses = expert_outputs.shape[1]
    cdef int seq_len = expert_outputs.shape[2]
    cdef int hidden_dim = expert_outputs.shape[3]
    
    cdef cnp.ndarray[DTYPE_t, ndim=2] ensemble_output = np.zeros((seq_len, hidden_dim), dtype=np.float32)
    
    cdef int expert_idx, response_idx, seq_idx, dim_idx
    cdef DTYPE_t expert_contribution, response_weight, scaled_weight
    cdef DTYPE_t total_weight, final_value
    
    # Aggregate across experts and responses
    for seq_idx in range(seq_len):
        for dim_idx in range(hidden_dim):
            final_value = 0.0
            total_weight = 0.0
            
            for expert_idx in range(num_experts):
                expert_contribution = 0.0
                
                # Aggregate responses within expert
                for response_idx in range(num_responses):
                    response_weight = expert_weights[expert_idx, response_idx]
                    expert_contribution += (response_weight * 
                        expert_outputs[expert_idx, response_idx, seq_idx, dim_idx])
                
                # Apply temperature scaling to ensemble weight
                scaled_weight = ensemble_weights[expert_idx] / temperature
                scaled_weight = exp(scaled_weight)
                
                final_value += scaled_weight * expert_contribution
                total_weight += scaled_weight
            
            # Normalize by total weight
            if total_weight > 0:
                ensemble_output[seq_idx, dim_idx] = final_value / total_weight
    
    return ensemble_output


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=1] fast_confidence_calibration(
    cnp.ndarray[DTYPE_t, ndim=1] raw_confidences,
    cnp.ndarray[DTYPE_t, ndim=1] historical_accuracies,
    DTYPE_t calibration_strength=0.5
):
    """
    Fast confidence calibration based on historical performance.
    
    Args:
        raw_confidences: Shape (num_responses,)
        historical_accuracies: Shape (num_responses,)
        calibration_strength: Strength of calibration adjustment
        
    Returns:
        calibrated_confidences: Shape (num_responses,)
    """
    cdef int num_responses = raw_confidences.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] calibrated = np.zeros(num_responses, dtype=np.float32)
    
    cdef int i
    cdef DTYPE_t raw_conf, historical_acc, calibration_factor, calibrated_conf
    
    for i in range(num_responses):
        raw_conf = raw_confidences[i]
        historical_acc = historical_accuracies[i]
        
        # Calculate calibration factor
        calibration_factor = 1.0 + calibration_strength * (historical_acc - 0.5)
        
        # Apply calibration
        calibrated_conf = raw_conf * calibration_factor
        
        # Clamp to [0, 1] range
        if calibrated_conf < 0.0:
            calibrated_conf = 0.0
        elif calibrated_conf > 1.0:
            calibrated_conf = 1.0
        
        calibrated[i] = calibrated_conf
    
    return calibrated


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] fast_hierarchical_aggregation(
    cnp.ndarray[DTYPE_t, ndim=3] response_embeddings,
    cnp.ndarray[INT_t, ndim=1] hierarchy_levels,
    cnp.ndarray[DTYPE_t, ndim=1] level_weights,
    int max_level=3
):
    """
    Fast hierarchical aggregation with multi-level consensus.
    
    Args:
        response_embeddings: Shape (num_responses, seq_len, hidden_dim)
        hierarchy_levels: Shape (num_responses,) - level for each response
        level_weights: Shape (max_level+1,) - weight for each level
        max_level: Maximum hierarchy level
        
    Returns:
        hierarchical_output: Shape (seq_len, hidden_dim)
    """
    cdef int num_responses = response_embeddings.shape[0]
    cdef int seq_len = response_embeddings.shape[1]
    cdef int hidden_dim = response_embeddings.shape[2]
    
    cdef cnp.ndarray[DTYPE_t, ndim=2] hierarchical_output = np.zeros(
        (seq_len, hidden_dim), dtype=np.float32)
    cdef cnp.ndarray[DTYPE_t, ndim=3] level_aggregates = np.zeros(
        (max_level + 1, seq_len, hidden_dim), dtype=np.float32)
    cdef cnp.ndarray[DTYPE_t, ndim=1] level_counts = np.zeros(max_level + 1, dtype=np.float32)
    
    cdef int response_idx, level, seq_idx, dim_idx
    cdef DTYPE_t total_weight, weighted_value
    
    # Aggregate responses by hierarchy level
    for response_idx in range(num_responses):
        level = hierarchy_levels[response_idx]
        if level <= max_level:
            level_counts[level] += 1.0
            
            for seq_idx in range(seq_len):
                for dim_idx in range(hidden_dim):
                    level_aggregates[level, seq_idx, dim_idx] += response_embeddings[response_idx, seq_idx, dim_idx]
    
    # Normalize level aggregates
    for level in range(max_level + 1):
        if level_counts[level] > 0:
            for seq_idx in range(seq_len):
                for dim_idx in range(hidden_dim):
                    level_aggregates[level, seq_idx, dim_idx] /= level_counts[level]
    
    # Combine levels with weights
    for seq_idx in range(seq_len):
        for dim_idx in range(hidden_dim):
            total_weight = 0.0
            weighted_value = 0.0
            
            for level in range(max_level + 1):
                if level_counts[level] > 0:
                    weighted_value += (level_weights[level] * 
                        level_aggregates[level, seq_idx, dim_idx])
                    total_weight += level_weights[level]
            
            if total_weight > 0:
                hierarchical_output[seq_idx, dim_idx] = weighted_value / total_weight
    
    return hierarchical_output