"""
Core Verification Package - Safety and alignment verification system.

This package provides verification capabilities for CapibaraGPT,
implementing Constitutional AI-inspired checks and corrections.

Key Components:
    - ComprehensiveVerificationSystem: Main verification class
    - AlignmentConfig: Configuration for verification parameters
    - create_verification_system: Factory function for creating verifiers

Author: Skydesk International Dev Team.
"""

from .constitutional_ai import (
    ComprehensiveVerificationSystem,
    AlignmentConfig,
    create_verification_system,
)

__all__ = [
    "ComprehensiveVerificationSystem",
    "AlignmentConfig",
    "create_verification_system",
]