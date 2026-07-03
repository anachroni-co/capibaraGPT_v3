"""
Cython Kernels Package - High-performance kernels for meta-consensus.

This package provides Cython-optimized implementations of computationally
intensive operations used in meta-consensus, routing, similarity
computation, and aggregation.

Key Components:
    - consensus_kernels: Optimized consensus operations
    - routing_kernels: High-performance routing functions
    - similarity_kernels: Fast similarity computation
    - aggregation_kernels: Efficient aggregation operations

Note:
    These kernels are optional and fall back to pure Python if Cython
    is not available during build.

Author: Skydesk International Dev Team.
"""

# Cython Kernels for Meta-Consensus Critical Loops
# High-performance Cython implementations for computationally intensive operations

__version__ = "1.0.0"
__all__ = [
    "consensus_kernels",
    "routing_kernels", 
    "similarity_kernels",
    "aggregation_kernels"
]