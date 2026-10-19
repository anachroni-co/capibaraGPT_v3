"""
JAX experimental mesh_utils
===========================

Utilidades for manejo de device mesh and topologías distribuidas.
"""

try:
    import numpy as np
except ImportError:
    # Fallback if numpy not está available
    class MockNumpy:
        def prod(self, x): return 1
        def sqrt(self, x): return 1
    np = MockNumpy()
from typing import Tuple, List, Optional, Any

try:
    import jax
    from jax.experimental import mesh_utils as jax_mesh_utils
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

def create_device_mesh(mesh_shape: Tuple[int, ...], devices: Optional[List] = None) -> Any:
    """
    Create device mesh for distributed computation.
    
    Args:
        mesh_shape: Shape of the mesh (e.g., (4, 8) for tpu v4-32)
        devices: Optional list of devices
    
    Returns:
        Device mesh object
    """
    if JAX_AVAILABLE and hasattr(jax_mesh_utils, 'create_device_mesh'):
        if devices is None:
            try:
                devices = jax.devices()
            except:
                devices = list(range(np.prod(mesh_shape)))
        return jax_mesh_utils.create_device_mesh(mesh_shape, devices)
    else:
        # Fallback implementation
        class MockDeviceMesh:
            def __init__(self, shape, devices):
                self.shape = shape
                self.devices = devices or list(range(np.prod(shape)))
                self.size = np.prod(shape)
            
            def __repr__(self):
                return f"MockDeviceMesh(shape={self.shape}, devices={len(self.devices)})"
        
        return MockDeviceMesh(mesh_shape, devices)

def create_hybrid_device_mesh(mesh_shape: Tuple[int, ...], 
                             process_count: Optional[int] = None) -> Any:
    """
    Create hybrid device mesh for multi-process distributed computation.
    
    Args:
        mesh_shape: Shape of the mesh
        process_count: Number of processes
    
    Returns:
        Hybrid device mesh
    """
    if JAX_AVAILABLE and hasattr(jax_mesh_utils, 'create_hybrid_device_mesh'):
        return jax_mesh_utils.create_hybrid_device_mesh(mesh_shape, process_count)
    else:
        # Fallback - use regular device mesh
        return create_device_mesh(mesh_shape)

def get_optimal_mesh_shape(num_devices: int, 
                          workload_type: str = "balanced") -> Tuple[int, ...]:
    """
    Get optimal mesh shape based on number of devices and workload type.
    
    Args:
        num_devices: Total number of devices
        workload_type: Type of workload ("data_parallel", "model_parallel", "balanced")
    
    Returns:
        Optimal mesh shape
    """
    if num_devices == 32:  # tpu v4-32
        if workload_type == "data_parallel":
            return (4, 8)
        elif workload_type == "model_parallel":
            return (8, 4)
        elif workload_type == "cultural_analysis":
            return (2, 16)
        elif workload_type == "adaptive_classical":
            return (8, 4)
        elif workload_type == "spiking_neural":
            return (1, 32)
        else:  # balanced
            return (4, 8)
    elif num_devices == 16:  # tpu v4-16
        return (4, 4) if workload_type == "balanced" else (2, 8)
    elif num_devices == 8:
        return (2, 4) if workload_type == "balanced" else (1, 8)
    elif num_devices == 4:
        return (2, 2) if workload_type == "balanced" else (1, 4)
    else:
        # Default: try to make it as square as possible
        factors = []
        for i in range(1, int(num_devices**0.5) + 1):
            if num_devices % i == 0:
                factors.append((i, num_devices // i))
        
        if factors:
            # Choose the most balanced factorization
            return min(factors, key=lambda x: abs(x[0] - x[1]))
        else:
            return (1, num_devices)

__all__ = [
    'create_device_mesh', 
    'create_hybrid_device_mesh', 
    'get_optimal_mesh_shape'
] 