"""
Federated Consensus Package - Multi-node distributed consensus system.

This package provides federated consensus capabilities for distributed
meta-consensus across multiple nodes with fault tolerance and
distributed coordination.

Key Components:
    - FederatedConsensusNode: Individual node in consensus network
    - FederatedConsensusCoordinator: Central coordinator for consensus
    - FederatedConsensusConfig: Configuration for consensus settings
    - ConsensusProtocol: Protocol definitions for consensus operations
    - NodeRole: Enum for node roles in consensus network

Author: Skydesk International Dev Team.
"""

# Federated Consensus System for Meta-Consensus
# Multi-node consensus with distributed coordination and fault tolerance

__version__ = "1.0.0"
__all__ = [
    "FederatedConsensusNode",
    "FederatedConsensusCoordinator", 
    "FederatedConsensusConfig",
    "ConsensusProtocol",
    "NodeRole"
]