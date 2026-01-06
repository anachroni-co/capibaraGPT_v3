"""
CapibaraGPT Vector Quantization (VQ) Module v2.0

Advanced VQ implementations with intelligent integration and automatic optimization.
Includes enhanced VQ systems, intelligent manager, and seamless legacy compatibility.
"""

import os
import sys
import logging
import warnings
from typing import Dict, Any, Optional, List, Union, Tuple, Type
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

# ============================================================================
# Version and metadata
# ============================================================================

__version__ = "2024.1.0"
__author__ = "CapibaraGPT Ultra VQ Team"
__description__ = "Ultra-Advanced Vector Quantization System"
__status__ = "Production-Ready Ultra"

# ============================================================================
# Safe import system for ultra components
# ============================================================================

# Component availability flags
ULTRA_VQ_ORCHESTRATOR_AVAILABLE = True
MULTI_MODAL_VQ_INTELLIGENCE_AVAILABLE = True
ADAPTIVE_VQ_PERFORMANCE_AVAILABLE = True
VQ_LEGACY_COMPONENTS_AVAILABLE = True

# Core ultra components
try:
    from .enhanced_vq_system_v2 import (
        # Enhanced classes
        EnhancedVectorQuantizer,
        EnhancedVQConfig,
        
        # Factory functions
        create_enhanced_vq_layer,
        benchmark_vq_variants,
        enhanced_example_usage
    )
    ENHANCED_VQ_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Enhanced VQ system not available: {e}")
    ENHANCED_VQ_AVAILABLE = False

# Import intelligent integration system
try:
    from .intelligent_vq_integration import (
        # Intelligent management
        IntelligentVQManager,
        VQManagerConfig,
        CapibaraVQIntegration,
        
        # High-level API
        create_optimal_vq,
        get_vq_manager,
        get_vq_recommendations,
        get_vq_usage_stats,
        
        # Demo
        demo_intelligent_integration
    )
    INTELLIGENT_VQ_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Intelligent VQ integration not available: {e}")
    INTELLIGENT_VQ_AVAILABLE = False

# Legacy imports for backward compatibility
try:
    from .ultra_vq_orchestrator import *
    from .vqbit_layer import *
    from .multi_modal_vq_intelligence import *
    from .adaptive_vq_performance_intelligence import *
    from .quantum_submodel_fixed import *
    from .demo_ultra_vq_ecosystem import *
    LEGACY_VQ_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import some legacy VQ components: {e}")
    LEGACY_VQ_AVAILABLE = False

# ============================================================================
# Unified VQ Creation Function
# ============================================================================

def create_vq(use_case: str = "general",
              num_embeddings: int = 8192,
              embedding_dim: int = 1024,
              system_preference: str = "auto",  # "auto", "enhanced", "legacy"
              performance_requirements: dict = None,
              **kwargs):
    """
    Unified VQ creation function that automatically selects the best system.
    
    Args:
        use_case: Use case ('general', 'research', 'memory_constrained', etc.)
        num_embeddings: Number of embeddings in codebook
        embedding_dim: Embedding dimension
        system_preference: System preference ("auto", "enhanced", "legacy")
        performance_requirements: Performance constraints
        **kwargs: Additional configuration
        
    Returns:
        VQ module and configuration info
        
    Examples:
        # Automatic selection (recommended)
        vq, info = create_vq("research", 4096, 512)
        
        # Force enhanced system
        vq, info = create_vq("general", system_preference="enhanced")
        
        # Memory-constrained with specific requirements
        vq, info = create_vq(
            "memory_constrained", 
            2048, 256,
            performance_requirements={'max_latency_ms': 5.0}
        )
    """
    
    # Auto-selection logic
    if system_preference == "auto":
        if INTELLIGENT_VQ_AVAILABLE:
            return create_optimal_vq(
                use_case=use_case,
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                performance_requirements=performance_requirements,
                **kwargs
            )
        elif ENHANCED_VQ_AVAILABLE:
            vq = create_enhanced_vq_layer(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                **kwargs
            )
            return vq, {'system': 'enhanced', 'fallback': True}
        else:
            raise ImportError("No VQ systems available")
    
    # Enhanced system
    elif system_preference == "enhanced":
        if ENHANCED_VQ_AVAILABLE:
            vq = create_enhanced_vq_layer(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                **kwargs
            )
            return vq, {'system': 'enhanced'}
        else:
            raise ImportError("Enhanced VQ system not available")
    
    # Legacy system
    elif system_preference == "legacy":
        if LEGACY_VQ_AVAILABLE:
            # This would depend on legacy APIs
            raise NotImplementedError("Legacy VQ system selection not implemented")
        else:
            raise ImportError("Legacy VQ systems not available")
    
    else:
        raise ValueError(f"Unknown system preference: {system_preference}")

# ============================================================================
# Module Configuration and Status
# ============================================================================

def get_vq_status():
    """Get status of all VQ systems."""
    return {
        'enhanced_available': ENHANCED_VQ_AVAILABLE,
        'intelligent_available': INTELLIGENT_VQ_AVAILABLE,
        'legacy_available': LEGACY_VQ_AVAILABLE,
        'recommended_api': 'create_vq' if INTELLIGENT_VQ_AVAILABLE else 'create_enhanced_vq_layer'
    }

def list_available_systems():
    """List all available VQ systems and their capabilities."""
    systems = {}
    
    if INTELLIGENT_VQ_AVAILABLE:
        try:
            manager = get_vq_manager()
            systems['intelligent'] = {
                'available_systems': list(manager.vq_systems.keys()),
                'auto_selection': True,
                'benchmarking': True,
                'performance_optimization': True
            }
        except:
            pass
    
    if ENHANCED_VQ_AVAILABLE:
        systems['enhanced'] = {
            'features': [
                'Product Quantization',
                'Adaptive Commitment',
                'Entropy Regularization',
                'Diversity Loss',
                'Neural Compression',
                'Gumbel Softmax'
            ],
            'research_grade': True
        }
    
    if LEGACY_VQ_AVAILABLE:
        systems['legacy'] = {
            'components': [
                'ultra_vq_orchestrator',
                'vqbit_layer',
                'multi_modal_vq_intelligence',
                'adaptive_vq_performance_intelligence'
            ],
            'backward_compatible': True
        }
    
    return systems

# ============================================================================
# Quick Examples and Demonstrations
# ============================================================================

def run_vq_demo():
    """Run a comprehensive VQ demonstration."""
    print("🚀 CapibaraGPT VQ Module v2.0 Demo")
    print("=" * 50)
    
    # System status
    print("\n📊 System Status:")
    status = get_vq_status()
    for system, available in status.items():
        status_emoji = "✅" if available else "❌"
        print(f"   {status_emoji} {system}")
    
    # Available systems
    print("\n🔧 Available Systems:")
    systems = list_available_systems()
    for name, info in systems.items():
        print(f"   📦 {name.title()}: {len(info)} capabilities")
    
    # Quick creation examples
    print("\n🎯 Quick Creation Examples:")
    
    try:
        # General VQ
        vq, info = create_vq("general", 1024, 256)
        print(f"   ✅ General VQ: {info.get('selected_system', 'created')}")
        
        # Research VQ
        vq_research, info_research = create_vq(
            "research", 
            2048, 512,
            performance_requirements={'min_perplexity': 100.0}
        )
        print(f"   ✅ Research VQ: {info_research.get('selected_system', 'created')}")
        
    except Exception as e:
        print(f"   ❌ Demo failed: {e}")
    
    # Run intelligent demo if available
    if INTELLIGENT_VQ_AVAILABLE:
        print("\n🧠 Running Intelligent Integration Demo:")
        try:
            demo_intelligent_integration()
        except Exception as e:
            print(f"   ❌ Intelligent demo failed: {e}")
    
    print("\n✅ VQ Demo completed!")

# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Main API
    'create_vq',
    
    # Enhanced system (if available)
    'create_enhanced_vq_layer',
    'EnhancedVectorQuantizer', 
    'EnhancedVQConfig',
    
    # Intelligent system (if available)
    'create_optimal_vq',
    'IntelligentVQManager',
    'VQManagerConfig',
    'get_vq_manager',
    'get_vq_recommendations',
    'get_vq_usage_stats',
    
    # Utilities
    'get_vq_status',
    'list_available_systems',
    'run_vq_demo',
    
    # Examples
    'enhanced_example_usage',
    'demo_intelligent_integration',
    
    # Legacy exports (if available)
    # These are preserved for backward compatibility
]

# Version and metadata
__version__ = "2.0.0"
__author__ = "CapibaraGPT Team"
__description__ = "Advanced Vector Quantization with Intelligent Integration"

# Default configurations for quick setup
DEFAULT_ENHANCED_CONFIG = {
    'num_embeddings': 8192,
    'embedding_dim': 1024,
    'use_ema': True,
    'use_adaptive_commitment': False
} if ENHANCED_VQ_AVAILABLE else None

DEFAULT_INTELLIGENT_CONFIG = VQManagerConfig() if INTELLIGENT_VQ_AVAILABLE else None

# Quick start recommendation
RECOMMENDED_USAGE = """
# Recommended usage for CapibaraGPT VQ v2.0:

from capibara.vq import create_vq

# Automatic optimal selection
vq, info = create_vq("research", 4096, 512)

# Or use high-level intelligent API
from capibara.vq import create_optimal_vq
vq, info = create_optimal_vq("memory_constrained", 2048, 256)
"""
