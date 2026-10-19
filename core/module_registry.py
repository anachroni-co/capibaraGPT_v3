"""Dynamic Module Registry with TPU v6e-64 Optimizations.

This module provides a dynamic module registration and instantiation system
optimized for TPU v6e-64 hardware. It enables runtime discovery, registration,
and creation of pluggable modules within the CapibaraGPT architecture.

The registry system features:
- Dynamic module registration by class or factory function
- TPU-specific optimizations (mesh creation, memory monitoring)
- Automatic memory cleanup before module instantiation
- Backend-specific configuration (v6e, v5e, v4)
- Fallback support when TPU operations unavailable

Key Components:
    - ModuleRegistry: Main registry with TPU optimizations

Example:
    Basic module registration:

    >>> from capibara.core.module_registry import ModuleRegistry
    >>> from capibara.interfaces.imodules import IModule
    >>>
    >>> # Create registry for TPU v6e
    >>> registry = ModuleRegistry(backend="v6e")
    >>>
    >>> # Register a module class
    >>> class CustomModule(IModule):
    ...     def __init__(self, size=768):
    ...         self.size = size
    >>>
    >>> registry.register("custom", CustomModule)
    >>>
    >>> # Create module instance
    >>> module = registry.create_module("custom", size=1024)
    >>> print(module.size)  # 1024

    Factory registration:

    >>> # Register factory function
    >>> def create_vision_module(resolution=224):
    ...     # Complex initialization logic
    ...     module = VisionModule()
    ...     module.configure(resolution)
    ...     return module
    >>>
    >>> registry.register_factory("vision", create_vision_module)
    >>>
    >>> # Create via factory
    >>> vision = registry.create_module("vision", resolution=512)

    TPU-optimized usage:

    >>> # Registry automatically configures TPU optimizations
    >>> # - Creates appropriate mesh for backend (8x8 for v6e)
    >>> # - Sets up memory monitor (64GB limit for v6e)
    >>> # - Initializes TPU-specific operations
    >>>
    >>> # Memory cleanup happens automatically before module creation
    >>> module = registry.create_module("large_module", params=1e9)

Note:
    The registry automatically configures TPU optimizations based on the
    specified backend. For v6e (default), it creates an 8x8 mesh and sets
    64GB memory limits. For v5e, it uses 4x8 mesh. For v4, it uses default
    mesh configuration.

    If TPU operations are unavailable, the registry gracefully falls back
    to CPU operation with disabled optimizations.

See Also:
    - capibara.interfaces.imodules: Module interface definitions
    - capibara.jax.tpu_v4.backend: TPU-specific backend operations
    - capibara.jax.tpu_v4.optimizations: TPU optimization utilities
"""

"""Dynamic Module Registry with TPU optimizations."""

from typing import Dict, Type, Callable, Any

try:
    from capibara.interfaces.imodules import IModule
except ImportError:
    class IModule:
        pass

try:
    from capibara.jax.tpu_v4.optimizations import create_tpu_mesh, TpuMemoryMonitor
except ImportError:
    create_tpu_mesh = None
    TpuMemoryMonitor = None


class ModuleRegistry:
    """Dynamic module registry with TPU optimizations."""

    def __init__(self, backend: str = "v6e"):
        self._modules: Dict[str, Type[IModule]] = {}
        self._factories: Dict[str, Callable[..., IModule]] = {}
        self.backend = backend
        self._setup_tpu()

    def _setup_tpu(self):
        """Configure TPU mesh and memory monitor."""
        self.mesh = None
        self.memory_monitor = None
        
        if create_tpu_mesh:
            mesh_shapes = {"v6e": (8, 8), "v5e": (4, 8)}
            shape = mesh_shapes.get(self.backend)
            try:
                self.mesh = create_tpu_mesh(mesh_shape=shape, axis_names=('dp', 'ep')) if shape else create_tpu_mesh()
            except Exception:
                self.mesh = create_tpu_mesh()
        
        if TpuMemoryMonitor:
            memory_limits = {"v6e": 64, "v5e": 16, "v4": 32}
            limit = memory_limits.get(self.backend, 32)
            self.memory_monitor = TpuMemoryMonitor(memory_limit_gb=limit, cleanup_threshold=0.9)

    def register(self, name: str, module_class: Type[IModule]):
        """Register a module class."""
        self._modules[name] = module_class

    def register_factory(self, name: str, factory: Callable[..., IModule]):
        """Register a factory function."""
        self._factories[name] = factory

    def get_module(self, name: str) -> Type[IModule]:
        """Get registered module class."""
        if name not in self._modules:
            raise KeyError(f"Module {name} not registered")
        return self._modules[name]

    def create_module(self, name: str, **kwargs: Any) -> IModule:
        """Create module instance."""
        if self.memory_monitor and self.memory_monitor.should_cleanup():
            self.memory_monitor.force_cleanup()

        if name in self._factories:
            return self._factories[name](**kwargs)
        if name in self._modules:
            return self._modules[name](**kwargs)
        
        raise KeyError(f"Module {name} not registered")
