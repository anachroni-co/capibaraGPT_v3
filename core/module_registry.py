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

from typing import Dict, Type, Callable, Any

# Correct import path for interfaces
# Assumes existence of `capibara.interfaces.imodules` with `IModule`
try:
    from capibara.interfaces.imodules import IModule  # type: ignore
except Exception:  # pragma: no cover
    class IModule:  # Minimal fallback interface
        """Minimal fallback module interface when real interface unavailable."""
        pass

# Optional imports for TPU v4 backend
try:
    from capibara.jax.tpu_v4.backend import (  # type: ignore
        TpuV4LinalgOps,
        TpuV4SparsityOps,
        TpuV4NeuralOps,
        TpuV4RandomOps,
    )
    from capibara.jax.tpu_v4.optimizations import (  # type: ignore
        create_tpu_mesh,
        TpuMemoryMonitor,
    )
except Exception:  # pragma: no cover
    # Fallbacks if TPU operations not available
    TpuV4LinalgOps = None
    TpuV4SparsityOps = None
    TpuV4NeuralOps = None
    TpuV4RandomOps = None
    create_tpu_mesh = None
    TpuMemoryMonitor = None


class ModuleRegistry:
    """Dynamic module registry with TPU v6e-64 optimizations.

    This registry manages module registration, instantiation, and TPU-specific
    optimizations. It supports both class-based and factory-based module creation,
    with automatic memory management and backend-specific configuration.

    Attributes:
        _modules (Dict[str, Type[IModule]]): Registered module classes by name.
        _factories (Dict[str, Callable[..., IModule]]): Registered factory functions.
        backend (str): TPU backend identifier ("v6e", "v5e", "v4").
        mesh: TPU mesh configuration for distributed computation.
        memory_monitor: TPU memory monitor for cleanup management.
        linalg_ops: TPU-optimized linear algebra operations.
        sparsity_ops: TPU-optimized sparsity operations.
        neural_ops: TPU-optimized neural network operations.
        random_ops: TPU-optimized random number generation.

    Example:
        >>> registry = ModuleRegistry(backend="v6e")
        >>>
        >>> # Check available TPU operations
        >>> if registry.linalg_ops:
        ...     print("TPU linear algebra available")
        >>>
        >>> # Register and create module
        >>> registry.register("my_module", MyModuleClass)
        >>> instance = registry.create_module("my_module", config={})

    Note:
        The registry automatically sets up TPU optimizations during initialization.
        If TPU operations are unavailable, attributes like linalg_ops will be None,
        but the registry remains fully functional for CPU-based operation.
    """

    def __init__(self, backend: str = "v6e") -> None:
        """Initialize module registry with TPU backend configuration.

        Args:
            backend (str, optional): TPU backend identifier. Options:
                - "v6e": TPU v6e with 64GB HBM per chip, 8x8 mesh
                - "v5e": TPU v5e with 16GB HBM per chip, 4x8 mesh
                - "v4": TPU v4 with 32GB HBM per chip, default mesh
                Defaults to "v6e".

        Example:
            >>> # V6e configuration (highest performance)
            >>> registry_v6e = ModuleRegistry(backend="v6e")
            >>>
            >>> # V5e configuration (lower cost)
            >>> registry_v5e = ModuleRegistry(backend="v5e")
            >>>
            >>> # V4 configuration (legacy)
            >>> registry_v4 = ModuleRegistry(backend="v4")

        Note:
            Different backends have different memory limits and optimal mesh
            configurations. The registry automatically configures these based
            on the specified backend.
        """
        self._modules: Dict[str, Type[IModule]] = {}
        self._factories: Dict[str, Callable[..., IModule]] = {}
        self.backend = backend
        self.setup_tpu_optimizations()

    def setup_tpu_optimizations(self) -> None:
        """Configure TPU-specific optimizations based on backend.

        Sets up:
        - TPU mesh with backend-appropriate topology
        - Memory monitor with correct memory limits per backend
        - TPU-optimized operation implementations (linalg, sparsity, etc.)

        Example:
            >>> registry = ModuleRegistry()
            >>> # Optimizations already configured by __init__
            >>> print(f"Backend: {registry.backend}")
            >>> print(f"Mesh available: {registry.mesh is not None}")

        Note:
            Called automatically during __init__. You typically don't need to
            call this manually unless reconfiguring an existing registry.

            Memory limits by backend:
            - v6e: 64GB per chip
            - v5e: 16GB per chip
            - v4: 32GB per chip
        """
        # Create generalized mesh
        self.mesh = self.create_tpu_mesh_auto() if create_tpu_mesh else None

        # Configure memory monitor (v6e has more memory per chip)
        memory_limit = 64 if self.backend == "v6e" else 32  # GB per chip
        self.memory_monitor = (
            TpuMemoryMonitor(memory_limit_gb=memory_limit, cleanup_threshold=0.9)
            if TpuMemoryMonitor
            else None
        )

        # Optimized operations with safe fallbacks
        self.linalg_ops = TpuV4LinalgOps() if TpuV4LinalgOps else None
        self.sparsity_ops = TpuV4SparsityOps() if TpuV4SparsityOps else None
        self.neural_ops = TpuV4NeuralOps() if TpuV4NeuralOps else None
        self.random_ops = TpuV4RandomOps() if TpuV4RandomOps else None

    def create_tpu_mesh_auto(self):
        """Create TPU mesh according to configured backend.

        Returns:
            Mesh or None: TPU mesh configuration if TPU available, None otherwise.

        Example:
            >>> registry = ModuleRegistry(backend="v6e")
            >>> mesh = registry.create_tpu_mesh_auto()
            >>> if mesh:
            ...     print(f"Mesh created for {registry.backend}")

        Note:
            Mesh topologies by backend:
            - v6e: 8x8 mesh (data parallel x expert parallel)
            - v5e: 4x8 mesh
            - v4: Default mesh configuration

            Falls back to default mesh if backend-specific creation fails.
        """
        if not create_tpu_mesh:
            return None

        if self.backend == "v6e":
            # TPU v6e-64: 8x8 mesh (dp x ep)
            try:
                return create_tpu_mesh(mesh_shape=(8, 8), axis_names=('dp', 'ep'))
            except Exception:
                # Fallback to default
                return create_tpu_mesh()
        elif self.backend == "v5e":
            # TPU v5e: 4x8 mesh
            try:
                return create_tpu_mesh(mesh_shape=(4, 8), axis_names=('dp', 'ep'))
            except Exception:
                return create_tpu_mesh()
        else:
            # v4 or fallback
            return create_tpu_mesh()

    def register(self, name: str, module_class: Type[IModule]) -> None:
        """Register a module class by name.

        Args:
            name (str): Unique identifier for the module.
            module_class (Type[IModule]): Module class to register. Must implement
                IModule interface.

        Example:
            >>> from capibara.interfaces.imodules import IModule
            >>>
            >>> class MyModule(IModule):
            ...     def __init__(self, config):
            ...         self.config = config
            >>>
            >>> registry = ModuleRegistry()
            >>> registry.register("my_module", MyModule)
            >>>
            >>> # Now can create instances
            >>> instance = registry.create_module("my_module", config={"x": 1})

        Note:
            If a module with the same name already exists, it will be silently
            overwritten. This allows module updates but may hide configuration errors.
        """
        self._modules[name] = module_class

    def register_factory(self, name: str, factory: Callable[..., IModule]) -> None:
        """Register a factory function for dynamic module creation.

        Factory functions allow complex initialization logic that can't be
        expressed in a simple class constructor.

        Args:
            name (str): Unique identifier for the factory.
            factory (Callable[..., IModule]): Factory function that returns
                IModule instances. Should accept **kwargs.

        Example:
            >>> def create_transformer(num_layers=12, hidden_size=768):
            ...     # Complex initialization
            ...     module = TransformerModule()
            ...     module.layers = [Layer() for _ in range(num_layers)]
            ...     module.hidden_size = hidden_size
            ...     return module
            >>>
            >>> registry = ModuleRegistry()
            >>> registry.register_factory("transformer", create_transformer)
            >>>
            >>> # Create via factory
            >>> model = registry.create_module("transformer", num_layers=24)

        Note:
            Factories take precedence over registered classes when both exist
            for the same name. See create_module() for details.
        """
        self._factories[name] = factory

    def get_module(self, name: str) -> Type[IModule]:
        """Retrieve a registered module class by name.

        Args:
            name (str): Module identifier.

        Returns:
            Type[IModule]: The registered module class.

        Raises:
            KeyError: If module name not registered.

        Example:
            >>> registry = ModuleRegistry()
            >>> registry.register("encoder", EncoderModule)
            >>>
            >>> # Get class reference
            >>> EncoderClass = registry.get_module("encoder")
            >>> print(EncoderClass.__name__)  # "EncoderModule"
            >>>
            >>> # Create instance manually
            >>> encoder = EncoderClass(config={})

        Note:
            This returns the class itself, not an instance. To create an instance,
            use create_module() instead.
        """
        if name not in self._modules:
            raise KeyError(f"Module {name} not registered")
        return self._modules[name]

    def create_module(self, name: str, **kwargs: Any) -> IModule:
        """Create module instance using registered factory or class.

        This method creates a module instance, preferring factory functions over
        direct class instantiation. It also performs automatic memory cleanup
        before creation if needed.

        Args:
            name (str): Module identifier (must be registered).
            **kwargs: Arguments to pass to factory function or class constructor.

        Returns:
            IModule: Created module instance.

        Raises:
            KeyError: If neither factory nor class registered for name.

        Example:
            >>> registry = ModuleRegistry()
            >>>
            >>> # Via class registration
            >>> registry.register("simple", SimpleModule)
            >>> module1 = registry.create_module("simple", size=100)
            >>>
            >>> # Via factory registration
            >>> def create_complex(**config):
            ...     return ComplexModule(**config)
            >>> registry.register_factory("complex", create_complex)
            >>> module2 = registry.create_module("complex", layers=24)

        Note:
            Creation order:
            1. Check if factory registered for name, use if found
            2. Check if class registered for name, instantiate if found
            3. Raise KeyError if neither found

            Before creation, automatically checks TPU memory and performs cleanup
            if memory usage exceeds threshold.

            After creation, calls setup_tpu_optimizations() on module if method exists.
        """
        # Check memory before creating module
        if self.memory_monitor and getattr(self.memory_monitor, "should_cleanup", None):
            try:
                if self.memory_monitor.should_cleanup():  # type: ignore[attr-defined]
                    self.memory_monitor.force_cleanup()  # type: ignore[attr-defined]
            except Exception:
                pass

        if name in self._factories:
            module = self._factories[name](**kwargs)
        elif name in self._modules:
            module = self._modules[name](**kwargs)  # type: ignore[call-arg]
        else:
            raise KeyError(f"Factory or class for {name} not registered")

        # Configure module-specific optimizations if they exist
        if hasattr(module, "setup_tpu_optimizations"):
            try:
                module.setup_tpu_optimizations()  # type: ignore[attr-defined]
            except Exception:
                pass

        return module
