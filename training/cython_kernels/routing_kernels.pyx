# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=False

"""
High-performance Cython kernels for expert routing operations
Optimized for meta-consensus expert selection and load balancing
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, log, sqrt, fabs, tanh
from libc.stdlib cimport malloc, free
from cython.parallel import prange
cimport cython

ctypedef cnp.float32_t DTYPE_t
ctypedef cnp.int32_t INT_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[INT_t, ndim=1] fast_expert_selection(
    cnp.ndarray[DTYPE_t, ndim=2] query_embeddings,
    cnp.ndarray[DTYPE_t, ndim=3] expert_embeddings,
    cnp.ndarray[DTYPE_t, ndim=1] expert_capacities,
    cnp.ndarray[DTYPE_t, ndim=1] expert_loads,
    int top_k=3,
    DTYPE_t capacity_threshold=0.8
):
    """
    Fast expert selection with load balancing and capacity constraints.
    
    Args:
        query_embeddings: Shape (seq_len, hidden_dim)
        expert_embeddings: Shape (num_experts, seq_len, hidden_dim)
        expert_capacities: Shape (num_experts,)
        expert_loads: Shape (num_experts,)
        top_k: Number of experts to select
        capacity_threshold: Maximum load ratio
        
    Returns:
        selected_experts: Shape (top_k,) - indices of selected experts
    """
    cdef int num_experts = expert_embeddings.shape[0]
    cdef int seq_len = query_embeddings.shape[0]
    cdef int hidden_dim = query_embeddings.shape[1]
    
    cdef cnp.ndarray[DTYPE_t, ndim=1] expert_scores = np.zeros(num_experts, dtype=np.float32)
    cdef cnp.ndarray[INT_t, ndim=1] selected_experts = np.zeros(top_k, dtype=np.int32)
    cdef cnp.ndarray[INT_t, ndim=1] available_experts = np.zeros(num_experts, dtype=np.int32)
    
    cdef int expert_idx, seq_idx, dim_idx, i, j, max_idx
    cdef DTYPE_t similarity, dot_product, norm_query, norm_expert
    cdef DTYPE_t load_penalty, capacity_ratio, final_score, max_score
    cdef int num_available = 0
    
    # Calculate scores for available experts
    for expert_idx in range(num_experts):
        # Check capacity constraint
        capacity_ratio = expert_loads[expert_idx] / expert_capacities[expert_idx]
        if capacity_ratio < capacity_threshold:
            available_experts[num_available] = expert_idx
            num_available += 1
            
            # Calculate similarity score
            similarity = 0.0
            for seq_idx in range(seq_len):
                dot_product = 0.0
                norm_query = 0.0
                norm_expert = 0.0
                
                for dim_idx in range(hidden_dim):
                    dot_product += (query_embeddings[seq_idx, dim_idx] * 
                                  expert_embeddings[expert_idx, seq_idx, dim_idx])
                    norm_query += query_embeddings[seq_idx, dim_idx] ** 2
                    norm_expert += expert_embeddings[expert_idx, seq_idx, dim_idx] ** 2
                
                if norm_query > 0 and norm_expert > 0:
                    similarity += dot_product / (sqrt(norm_query) * sqrt(norm_expert))
            
            # Average similarity across sequence
            similarity /= seq_len
            
            # Apply load balancing penalty
            load_penalty = 1.0 - (capacity_ratio * 0.5)  # Reduce score for loaded experts
            final_score = similarity * load_penalty
            
            expert_scores[expert_idx] = final_score
        else:
            expert_scores[expert_idx] = -1.0  # Mark as unavailable
    
    # Select top-k experts
    cdef int selected_count = 0
    for i in range(min(top_k, num_available)):
        max_score = -2.0
        max_idx = -1
        
        # Find expert with highest score
        for j in range(num_experts):
            if expert_scores[j] > max_score:
                max_score = expert_scores[j]
                max_idx = j
        
        if max_idx >= 0:
            selected_experts[selected_count] = max_idx
            expert_scores[max_idx] = -2.0  # Mark as selected
            selected_count += 1
    
    return selected_experts[:selected_count]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=1] fast_load_balancing_weights(
    cnp.ndarray[DTYPE_t, ndim=1] expert_loads,
    cnp.ndarray[DTYPE_t, ndim=1] expert_capacities,
    cnp.ndarray[DTYPE_t, ndim=1] expert_qualities,
    DTYPE_t load_weight=0.4,
    DTYPE_t quality_weight=0.6
):
    """
    Fast calculation of load balancing weights for expert routing.
    
    Args:
        expert_loads: Current load for each expert
        expert_capacities: Maximum capacity for each expert
        expert_qualities: Quality scores for each expert
        load_weight: Weight for load balancing factor
        quality_weight: Weight for quality factor
        
    Returns:
        routing_weights: Normalized weights for expert selection
    """
    cdef int num_experts = expert_loads.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] weights = np.zeros(num_experts, dtype=np.float32)
    
    cdef int i
    cdef DTYPE_t load_factor, quality_factor, combined_weight, total_weight = 0.0
    
    # Calculate weights for each expert
    for i in range(num_experts):
        # Load balancing factor (prefer less loaded experts)
        if expert_capacities[i] > 0:
            load_factor = 1.0 - (expert_loads[i] / expert_capacities[i])
        else:
            load_factor = 0.0
        
        # Quality factor (prefer high-quality experts)
        quality_factor = expert_qualities[i]
        
        # Combine factors
        combined_weight = (load_factor * load_weight + 
                          quality_factor * quality_weight)
        
        weights[i] = max(combined_weight, 0.0)  # Ensure non-negative
        total_weight += weights[i]
    
    # Normalize weights
    if total_weight > 0:
        for i in range(num_experts):
            weights[i] /= total_weight
    
    return weights


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] fast_routing_matrix(
    cnp.ndarray[DTYPE_t, ndim=2] query_batch,
    cnp.ndarray[DTYPE_t, ndim=3] expert_embeddings,
    cnp.ndarray[DTYPE_t, ndim=1] expert_weights,
    DTYPE_t temperature=1.0
):
    """
    Fast computation of routing matrix for batch queries.
    
    Args:
        query_batch: Shape (batch_size, hidden_dim)
        expert_embeddings: Shape (num_experts, seq_len, hidden_dim)
        expert_weights: Shape (num_experts,)
        temperature: Softmax temperature for routing probabilities
        
    Returns:
        routing_matrix: Shape (batch_size, num_experts)
    """
    cdef int batch_size = query_batch.shape[0]
    cdef int hidden_dim = query_batch.shape[1]
    cdef int num_experts = expert_embeddings.shape[0]
    cdef int seq_len = expert_embeddings.shape[1]
    
    cdef cnp.ndarray[DTYPE_t, ndim=2] routing_matrix = np.zeros(
        (batch_size, num_experts), dtype=np.float32)
    
    cdef int batch_idx, expert_idx, seq_idx, dim_idx
    cdef DTYPE_t similarity, dot_product, norm_query, norm_expert
    cdef DTYPE_t weighted_score, max_score, sum_exp, exp_score
    
    # Process each query in the batch
    for batch_idx in range(batch_size):
        max_score = -1e10
        
        # Calculate scores for all experts
        for expert_idx in range(num_experts):
            similarity = 0.0
            
            # Calculate average similarity across sequence positions
            for seq_idx in range(seq_len):
                dot_product = 0.0
                norm_query = 0.0
                norm_expert = 0.0
                
                for dim_idx in range(hidden_dim):
                    dot_product += (query_batch[batch_idx, dim_idx] * 
                                  expert_embeddings[expert_idx, seq_idx, dim_idx])
                    norm_query += query_batch[batch_idx, dim_idx] ** 2
                    norm_expert += expert_embeddings[expert_idx, seq_idx, dim_idx] ** 2
                
                if norm_query > 0 and norm_expert > 0:
                    similarity += dot_product / (sqrt(norm_query) * sqrt(norm_expert))
            
            similarity /= seq_len
            weighted_score = similarity * expert_weights[expert_idx]
            routing_matrix[batch_idx, expert_idx] = weighted_score
            
            if weighted_score > max_score:
                max_score = weighted_score
        
        # Apply softmax with temperature
        sum_exp = 0.0
        for expert_idx in range(num_experts):
            exp_score = exp((routing_matrix[batch_idx, expert_idx] - max_score) / temperature)
            routing_matrix[batch_idx, expert_idx] = exp_score
            sum_exp += exp_score
        
        # Normalize probabilities
        if sum_exp > 0:
            for expert_idx in range(num_experts):
                routing_matrix[batch_idx, expert_idx] /= sum_exp
    
    return routing_matrix


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple fast_adaptive_routing(
    cnp.ndarray[DTYPE_t, ndim=2] query_embeddings,
    cnp.ndarray[DTYPE_t, ndim=3] expert_embeddings,
    cnp.ndarray[DTYPE_t, ndim=1] expert_performance_history,
    cnp.ndarray[DTYPE_t, ndim=1] expert_latencies,
    DTYPE_t performance_weight=0.5,
    DTYPE_t latency_weight=0.3,
    DTYPE_t similarity_weight=0.2
):
    """
    Adaptive routing that learns from expert performance and latency.
    
    Args:
        query_embeddings: Shape (seq_len, hidden_dim)
        expert_embeddings: Shape (num_experts, seq_len, hidden_dim)
        expert_performance_history: Shape (num_experts,)
        expert_latencies: Shape (num_experts,)
        performance_weight: Weight for performance factor
        latency_weight: Weight for latency factor
        similarity_weight: Weight for similarity factor
        
    Returns:
        routing_scores: Shape (num_experts,)
        routing_probabilities: Shape (num_experts,)
    """
    cdef int num_experts = expert_embeddings.shape[0]
    cdef int seq_len = query_embeddings.shape[0]
    cdef int hidden_dim = query_embeddings.shape[1]
    
    cdef cnp.ndarray[DTYPE_t, ndim=1] routing_scores = np.zeros(num_experts, dtype=np.float32)
    cdef cnp.ndarray[DTYPE_t, ndim=1] routing_probs = np.zeros(num_experts, dtype=np.float32)
    
    cdef int expert_idx, seq_idx, dim_idx
    cdef DTYPE_t similarity, dot_product, norm_query, norm_expert
    cdef DTYPE_t performance_factor, latency_factor, similarity_factor
    cdef DTYPE_t combined_score, max_score = -1e10, sum_exp = 0.0
    
    # Calculate adaptive scores for each expert
    for expert_idx in range(num_experts):
        # Similarity factor
        similarity = 0.0
        for seq_idx in range(seq_len):
            dot_product = 0.0
            norm_query = 0.0
            norm_expert = 0.0
            
            for dim_idx in range(hidden_dim):
                dot_product += (query_embeddings[seq_idx, dim_idx] * 
                              expert_embeddings[expert_idx, seq_idx, dim_idx])
                norm_query += query_embeddings[seq_idx, dim_idx] ** 2
                norm_expert += expert_embeddings[expert_idx, seq_idx, dim_idx] ** 2
            
            if norm_query > 0 and norm_expert > 0:
                similarity += dot_product / (sqrt(norm_query) * sqrt(norm_expert))
        
        similarity_factor = similarity / seq_len
        
        # Performance factor (normalized to 0-1)
        performance_factor = expert_performance_history[expert_idx]
        
        # Latency factor (inverse of latency, normalized)
        if expert_latencies[expert_idx] > 0:
            latency_factor = 1.0 / (1.0 + expert_latencies[expert_idx] / 1000.0)  # Convert ms to seconds
        else:
            latency_factor = 1.0
        
        # Combine factors
        combined_score = (similarity_factor * similarity_weight + 
                         performance_factor * performance_weight + 
                         latency_factor * latency_weight)
        
        routing_scores[expert_idx] = combined_score
        if combined_score > max_score:
            max_score = combined_score
    
    # Convert to probabilities using softmax
    for expert_idx in range(num_experts):
        routing_probs[expert_idx] = exp(routing_scores[expert_idx] - max_score)
        sum_exp += routing_probs[expert_idx]
    
    # Normalize probabilities
    if sum_exp > 0:
        for expert_idx in range(num_experts):
            routing_probs[expert_idx] /= sum_exp
    
    return np.array(routing_scores), np.array(routing_probs)