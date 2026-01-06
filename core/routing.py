#!/usr/bin/env python3
"""Core HTTP Routing Module for CapibaraGPT.

This module provides HTTP-style routing functionality for the CapibaraGPT system,
enabling request routing to appropriate handlers based on URL-like paths. While
the module uses HTTP terminology, it's designed for internal request routing
rather than actual HTTP server implementation.

The routing system features:
- Dynamic route registration
- Default route fallback
- Path-based request routing
- No-route error handling
- Health check and info endpoints

Key Components:
    - CoreHttpRouter: Main routing class for handling requests
    - Router: Alias for CoreHttpRouter (compatibility)
    - create_router: Factory function for router setup

Example:
    Basic routing setup:

    >>> from capibara.core.routing import create_router
    >>>
    >>> # Create router with default routes
    >>> router = create_router()
    >>>
    >>> # Test default routes
    >>> health = router.route("/health")
    >>> print(health)  # {"status": "healthy"}
    >>>
    >>> info = router.route("/info")
    >>> print(info["name"])  # "CapibaraGPT"

    Custom routes:

    >>> from capibara.core.routing import CoreHttpRouter
    >>>
    >>> # Create custom router
    >>> router = CoreHttpRouter()
    >>>
    >>> # Add custom route
    >>> def process_text(text=None):
    ...     return {"result": f"Processed: {text}"}
    >>>
    >>> router.add_route("/process", process_text)
    >>>
    >>> # Route request
    >>> result = router.route("/process", text="Hello world")
    >>> print(result)  # {"result": "Processed: Hello world"}

    Default route handling:

    >>> # Set default route for unmatched paths
    >>> def default_handler(**kwargs):
    ...     return {"status": "default", "data": kwargs}
    >>>
    >>> router.set_default_route(default_handler)
    >>>
    >>> # Route to non-existent path
    >>> result = router.route("/unknown", data="test")
    >>> print(result)  # {"status": "default", "data": {"data": "test"}}

Note:
    This routing system is primarily for internal request dispatching within
    the CapibaraGPT system. For actual HTTP server implementation, consider
    using FastAPI, Flask, or similar web frameworks.

    The router uses path strings but does not implement full URL parsing,
    query parameters, or HTTP methods. It's a simplified routing mechanism
    for component-to-component communication.

See Also:
    - capibara.core.router: Advanced router with ML-specific features
    - capibara.services: Service layer implementations
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class CoreHttpRouter:
    """Core HTTP routing class for handling internal request routing.

    This class manages a collection of routes (path -> handler mappings) and
    provides request routing functionality. It's designed for internal use
    within CapibaraGPT for directing requests to appropriate processing handlers.

    Attributes:
        routes (Dict[str, callable]): Mapping of path strings to handler functions.
        default_route (Optional[callable]): Default handler for unmatched paths.

    Example:
        >>> router = CoreHttpRouter()
        >>>
        >>> # Add routes
        >>> router.add_route("/health", lambda: {"status": "ok"})
        >>> router.add_route("/version", lambda: {"version": "4.0"})
        >>>
        >>> # Route requests
        >>> result = router.route("/health")
        >>> print(result)  # {"status": "ok"}

    Note:
        Handlers should be callables that accept **kwargs and return dictionaries.
        This convention ensures consistent routing behavior.
    """

    def __init__(self):
        """Initialize the router with empty route table.

        Example:
            >>> router = CoreHttpRouter()
            >>> print(len(router.routes))  # 0
        """
        self.routes = {}
        self.default_route = None

    def add_route(self, path: str, handler: callable):
        """Register a route handler for a specific path.

        Args:
            path (str): URL-like path string (e.g., "/health", "/process").
            handler (callable): Function to call when path is routed. Should
                accept **kwargs and return a dictionary.

        Example:
            >>> router = CoreHttpRouter()
            >>>
            >>> # Simple handler
            >>> def my_handler():
            ...     return {"message": "Hello"}
            >>> router.add_route("/hello", my_handler)
            >>>
            >>> # Handler with parameters
            >>> def echo_handler(text=None):
            ...     return {"echo": text}
            >>> router.add_route("/echo", echo_handler)
            >>>
            >>> # Test routes
            >>> print(router.route("/hello"))  # {"message": "Hello"}
            >>> print(router.route("/echo", text="test"))  # {"echo": "test"}

        Note:
            If a route already exists for the given path, it will be overwritten
            without warning. This allows route updates but may hide configuration
            errors.
        """
        self.routes[path] = handler
        logger.info(f"Added route: {path}")

    def set_default_route(self, handler: callable):
        """Set the default route handler for unmatched paths.

        The default handler is called when route() is invoked with a path
        that hasn't been registered via add_route().

        Args:
            handler (callable): Default handler function. Should accept **kwargs
                and return a dictionary.

        Example:
            >>> router = CoreHttpRouter()
            >>>
            >>> # Set default handler
            >>> def fallback(**kwargs):
            ...     return {"status": "fallback", "args": kwargs}
            >>>
            >>> router.set_default_route(fallback)
            >>>
            >>> # Route to unregistered path
            >>> result = router.route("/unknown", data="test")
            >>> print(result["status"])  # "fallback"

        Note:
            Only one default route can be active at a time. Calling this method
            multiple times will replace the previous default handler.
        """
        self.default_route = handler
        logger.info("Default route set")

    def route(self, path: str, **kwargs) -> Dict[str, Any]:
        """Route a request to the appropriate handler.

        Looks up the handler for the given path and invokes it with the provided
        keyword arguments. Falls back to default handler or returns error dict
        if path is not found.

        Args:
            path (str): Path to route (e.g., "/health", "/process").
            **kwargs: Keyword arguments to pass to the handler.

        Returns:
            Dict[str, Any]: Dictionary returned by the handler, or error dictionary
                if no handler found and no default route set.

        Example:
            >>> router = create_router()
            >>>
            >>> # Route to existing endpoint
            >>> result = router.route("/health")
            >>> print(result)  # {"status": "healthy"}
            >>>
            >>> # Route with parameters
            >>> def custom_handler(name=None):
            ...     return {"greeting": f"Hello {name}"}
            >>>
            >>> router.add_route("/greet", custom_handler)
            >>> result = router.route("/greet", name="Alice")
            >>> print(result)  # {"greeting": "Hello Alice"}
            >>>
            >>> # Unmatched route
            >>> result = router.route("/nonexistent")
            >>> print(result["status"])  # "no_route"

        Note:
            If the path is not found and no default route is set, returns
            a dictionary with status="no_route" instead of raising an exception.
            This ensures routing never crashes the application.
        """
        if path in self.routes:
            return self.routes[path](**kwargs)
        elif self.default_route:
            return self.default_route(**kwargs)
        else:
            # Return a default response instead of raising an error
            return {"status": "no_route", "path": path, "message": f"No route found for: {path}"}

# Alias for compatibility with core/__init__.py
Router = CoreHttpRouter

def create_router() -> CoreHttpRouter:
    """Create and configure the main router with default routes.

    Factory function that creates a CoreHttpRouter instance and registers
    standard routes for health checks and system information.

    Returns:
        CoreHttpRouter: Configured router instance with default routes:
            - /health: Returns {"status": "healthy"}
            - /info: Returns system information (version, name, mesh, dtype)

    Example:
        >>> # Create router with defaults
        >>> router = create_router()
        >>>
        >>> # Use health check
        >>> health = router.route("/health")
        >>> assert health["status"] == "healthy"
        >>>
        >>> # Get system info
        >>> info = router.route("/info")
        >>> print(f"Version: {info['version']}")
        >>> print(f"Mesh: {info['mesh']}")
        >>>
        >>> # Add custom routes
        >>> def custom():
        ...     return {"custom": True}
        >>> router.add_route("/custom", custom)

    Note:
        Default routes can be overwritten by calling add_route() with the
        same path. This allows customization of default behavior if needed.

        System info includes:
        - version: "4.0" (CapibaraGPT version)
        - name: "CapibaraGPT"
        - mesh: "v6e-64" (TPU mesh configuration)
        - dtype: "bf16" (default dtype)
    """
    router = CoreHttpRouter()

    # Add default routes
    router.add_route("/health", lambda: {"status": "healthy"})
    router.add_route("/info", lambda: {
        "version": "4.0",
        "name": "CapibaraGPT",
        "mesh": "v6e-64",
        "dtype": "bf16"
    })

    return router
