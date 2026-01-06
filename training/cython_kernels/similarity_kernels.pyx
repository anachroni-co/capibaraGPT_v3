# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=False

"""
High-performance Cython kernels for similarity calculations
Optimized for large-scale embedding comparisons in meta-consensus
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, log, sqrt, fabs, cos, sin, tanh
from libc.stdlib cimport malloc, free
from cython.parallel import prange
cimport cython

ctypedef cnp.float32_t DTYPE_t
ctypedef cnp.int32_t INT_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] fast_cosine_similarity_matrix(
    cnp.ndarray[DTYPE_t, ndim=2] embeddings_a,
    cnp.ndarray[DTYPE_t, ndim=2] embeddings_b
):
    """
    Fast computation of cosine similarity matrix between two embedding sets.
    
    Args:
        embeddings_a: Shape (num_a, hidden_dim)
        embeddings_b: Shape (num_b, hidden_dim)
        
    Returns:
        similarity_matrix: Shape (num_a, num_b)
    """
    cdef int num_a = embeddings_a.shape[0]
    cdef int num_b = embeddings_b.shape[0]
    cdef int hidden_dim = embeddings_a.shape[1]
    
    cdef cnp.ndarray[DTYPE_t, ndim=2] similarity_matrix = np.zeros(
        (num_a, num_b), dtype=np.float32)
    
    cdef int i, j, k
    cdef DTYPE_t dot_product, norm_a, norm_b, similarity
    
    # Calculate similarity matrix
    for i in range(num_a):
        for j in range(num_b):
            dot_product = 0.0
            norm_a = 0.0
            norm_b = 0.0
            
            # Calculate dot product and norms
            for k in range(hidden_dim):
                dot_product += embeddings_a[i, k] * embeddings_b[j, k]
                norm_a += embeddings_a[i, k] * embeddings_a[i, k]
                norm_b += embeddings_b[j, k] * embeddings_b[j, k]
            
            # Calculate cosine similarity
            if norm_a > 0 and norm_b > 0:
                similarity = dot_product / (sqrt(norm_a) * sqrt(norm_b))
            else:
                similarity = 0.0
            
            similarity_matrix[i, j] = similarity
    
    return similarity_matrix


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=1] fast_semantic_similarity_scores(
    cnp.ndarray[DTYPE_t, ndim=2] query_embeddings,
    cnp.ndarray[DTYPE_t, ndim=3] response_embeddings,
    cnp.ndarray[DTYPE_t, ndim=1] response_weights,
    DTYPE_t position_weight_decay=0.95
):
    """
    Fast semantic similarity calculation with positional weighting.
    
    Args:
        query_embeddings: Shape (seq_len, hidden_dim)
        response_embeddings: Shape (num_responses, seq_len, hidden_dim)
        response_weights: Shape (num_responses,)
        position_weight_decay: Decay factor for position weights
        
    Returns:
        similarity_scores: Shape (num_responses,)
    """
    cdef int num_responses = response_embeddings.shape[0]
    cdef int seq_len = query_embeddings.shape[0]
    cdef int hidden_dim = query_embeddings.shape[1]
    
    cdef cnp.ndarray[DTYPE_t, ndim=1] similarity_scores = np.zeros(num_responses, dtype=np.float32)
    
    cdef int response_idx, seq_idx, dim_idx
    cdef DTYPE_t dot_product, norm_query, norm_response, position_weight
    cdef DTYPE_t weighted_similarity, total_weight, final_score
    
    # Calculate similarity for each response
    for response_idx in range(num_responses):
        weighted_similarity = 0.0
        total_weight = 0.0
        
        # Calculate position-weighted similarity
        for seq_idx in range(seq_len):
            dot_product = 0.0
            norm_query = 0.0
            norm_response = 0.0
            
            # Calculate similarity at this position
            for dim_idx in range(hidden_dim):
                dot_product += (query_embeddings[seq_idx, dim_idx] * 
                              response_embeddings[response_idx, seq_idx, dim_idx])
                norm_query += query_embeddings[seq_idx, dim_idx] ** 2
                norm_response += response_embeddings[response_idx, seq_idx, dim_idx] ** 2
            
            # Position weight (earlier positions are more important)
            position_weight = position_weight_decay ** seq_idx
            
            # Add weighted similarity
            if norm_query > 0 and norm_response > 0:
                weighted_similarity += (position_weight * dot_product / 
                                      (sqrt(norm_query) * sqrt(norm_response)))
            
            total_weight += position_weight
        
        # Normalize by total weight and apply response weight
        if total_weight > 0:
            final_score = (weighted_similarity / total_weight) * response_weights[response_idx]
        else:
            final_score = 0.0
        
        similarity_scores[response_idx] = final_score
    
    return similarity_scores


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] fast_attention_similarity(
    cnp.ndarray[DTYPE_t, ndim=3] query_states,
    cnp.ndarray[DTYPE_t, ndim=3] key_states,
    cnp.ndarray[DTYPE_t, ndim=3] value_states,
    DTYPE_t scale_factor=1.0,
    DTYPE_t dropout_rate=0.1
):
    """
    Fast attention-based similarity calculation.
    
    Args:
        query_states: Shape (num_heads, seq_len, head_dim)
        key_states: Shape (num_heads, seq_len, head_dim)
        value_states: Shape (num_heads, seq_len, head_dim)
        scale_factor: Scaling factor for attention scores
        dropout_rate: Dropout rate for attention weights
        
    Returns:
        attention_output: Shape (seq_len, num_heads * head_dim)
    """
    cdef int num_heads = query_states.shape[0]
    cdef int seq_len = query_states.shape[1]
    cdef int head_dim = query_states.shape[2]
    
    cdef cnp.ndarray[DTYPE_t, ndim=2] attention_output = np.zeros(
        (seq_len, num_heads * head_dim), dtype=np.float32)
    cdef cnp.ndarray[DTYPE_t, ndim=2] attention_weights = np.zeros(
        (seq_len, seq_len), dtype=np.float32)
    
    cdef int head_idx, i, j, k, output_idx
    cdef DTYPE_t dot_product, max_score, sum_exp, attention_score
    cdef DTYPE_t weighted_value
    
    # Process each attention head
    for head_idx in range(num_heads):
        # Calculate attention scores
        max_score = -1e10
        
        for i in range(seq_len):
            for j in range(seq_len):
                dot_product = 0.0
                
                # Calculate query-key dot product
                for k in range(head_dim):
                    dot_product += (query_states[head_idx, i, k] * 
                                  key_states[head_idx, j, k])
                
                attention_score = dot_product * scale_factor
                attention_weights[i, j] = attention_score
                
                if attention_score > max_score:
                    max_score = attention_score
        
        # Apply softmax to attention weights
        for i in range(seq_len):
            sum_exp = 0.0
            
            # Calculate exp and sum
            for j in range(seq_len):
                attention_weights[i, j] = exp(attention_weights[i, j] - max_score)
                sum_exp += attention_weights[i, j]
            
            # Normalize
            if sum_exp > 0:
                for j in range(seq_len):
                    attention_weights[i, j] /= sum_exp
        
        # Apply attention to values and accumulate in output
        for i in range(seq_len):
            for k in range(head_dim):
                weighted_value = 0.0
                
                for j in range(seq_len):
                    weighted_value += (attention_weights[i, j] * 
                                     value_states[head_idx, j, k])
                
                output_idx = head_idx * head_dim + k
                attention_output[i, output_idx] = weighted_value
    
    return attention_output


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=1] fast_diversity_calculation(
    cnp.ndarray[DTYPE_t, ndim=2] embeddings,
    DTYPE_t diversity_threshold=0.5
):
    """
    Fast diversity calculation for response embeddings.
    
    Args:
        embeddings: Shape (num_responses, hidden_dim)
        diversity_threshold: Minimum diversity threshold
        
    Returns:
        diversity_scores: Shape (num_responses,)
    """
    cdef int num_responses = embeddings.shape[0]
    cdef int hidden_dim = embeddings.shape[1]
    
    cdef cnp.ndarray[DTYPE_t, ndim=1] diversity_scores = np.zeros(num_responses, dtype=np.float32)
    
    cdef int i, j, k
    cdef DTYPE_t dot_product, norm_i, norm_j, similarity, diversity
    cdef DTYPE_t total_diversity, avg_diversity
    
    # Calculate diversity for each response
    for i in range(num_responses):
        total_diversity = 0.0
        
        for j in range(num_responses):
            if i != j:
                dot_product = 0.0
                norm_i = 0.0
                norm_j = 0.0
                
                # Calculate cosine similarity
                for k in range(hidden_dim):
                    dot_product += embeddings[i, k] * embeddings[j, k]
                    norm_i += embeddings[i, k] * embeddings[i, k]
                    norm_j += embeddings[j, k] * embeddings[j, k]
                
                if norm_i > 0 and norm_j > 0:
                    similarity = dot_product / (sqrt(norm_i) * sqrt(norm_j))
                    diversity = 1.0 - fabs(similarity)  # Diversity = 1 - |similarity|
                else:
                    diversity = 1.0
                
                total_diversity += diversity
        
        # Average diversity with other responses
        if num_responses > 1:
            avg_diversity = total_diversity / (num_responses - 1)
        else:
            avg_diversity = 1.0
        
        # Apply threshold
        if avg_diversity >= diversity_threshold:
            diversity_scores[i] = avg_diversity
        else:
            diversity_scores[i] = 0.0
    
    return diversity_scores


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] fast_cross_attention_similarity(
    cnp.ndarray[DTYPE_t, ndim=3] embeddings_a,
    cnp.ndarray[DTYPE_t, ndim=3] embeddings_b,
    cnp.ndarray[DTYPE_t, ndim=1] attention_weights_a,
    cnp.ndarray[DTYPE_t, ndim=1] attention_weights_b
):
    """
    Fast cross-attention similarity between two sets of embeddings.
    
    Args:
        embeddings_a: Shape (num_a, seq_len, hidden_dim)
        embeddings_b: Shape (num_b, seq_len, hidden_dim)
        attention_weights_a: Shape (num_a,)
        attention_weights_b: Shape (num_b,)
        
    Returns:
        cross_similarity: Shape (num_a, num_b)
    """
    cdef int num_a = embeddings_a.shape[0]
    cdef int num_b = embeddings_b.shape[0]
    cdef int seq_len = embeddings_a.shape[1]
    cdef int hidden_dim = embeddings_a.shape[2]
    
    cdef cnp.ndarray[DTYPE_t, ndim=2] cross_similarity = np.zeros(
        (num_a, num_b), dtype=np.float32)
    
    cdef int i, j, seq_idx, dim_idx
    cdef DTYPE_t dot_product, norm_a, norm_b, position_similarity
    cdef DTYPE_t weighted_similarity, total_similarity
    
    # Calculate cross-attention similarity
    for i in range(num_a):
        for j in range(num_b):
            total_similarity = 0.0
            
            # Calculate similarity across sequence positions
            for seq_idx in range(seq_len):
                dot_product = 0.0
                norm_a = 0.0
                norm_b = 0.0
                
                # Calculate position-wise similarity
                for dim_idx in range(hidden_dim):
                    dot_product += (embeddings_a[i, seq_idx, dim_idx] * 
                                  embeddings_b[j, seq_idx, dim_idx])
                    norm_a += embeddings_a[i, seq_idx, dim_idx] ** 2
                    norm_b += embeddings_b[j, seq_idx, dim_idx] ** 2
                
                if norm_a > 0 and norm_b > 0:
                    position_similarity = dot_product / (sqrt(norm_a) * sqrt(norm_b))
                else:
                    position_similarity = 0.0
                
                total_similarity += position_similarity
            
            # Average similarity across positions
            total_similarity /= seq_len
            
            # Apply attention weights
            weighted_similarity = (total_similarity * attention_weights_a[i] * 
                                 attention_weights_b[j])
            
            cross_similarity[i, j] = weighted_similarity
    
    return cross_similarity