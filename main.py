#!/usr/bin/env python3
"""
CapibaraGPT v3 - Main Entry Point

A modular, experimental AI system supporting both TPU and GPU backends.
Provides intelligent routing, multi-agent orchestration, and advanced
neural network architectures (Transformer/Mamba hybrid).

Usage:
    python main.py                    # Start with default settings
    python main.py --health           # Run health check
    python main.py --demo             # Run demonstration
    python main.py --backend gpu      # Use GPU backend
    python main.py --backend tpu      # Use TPU backend
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("capibaragpt")

__version__ = "3.0.0"


@dataclass
class SystemConfig:
    """Configuration for the CapibaraGPT system."""
    backend: str = "cpu"  # cpu, gpu, tpu
    model_type: str = "hybrid"  # transformer, mamba, hybrid
    enable_routing: bool = True
    enable_safety: bool = True
    cache_size: int = 1000
    log_level: str = "INFO"


@dataclass
class HealthStatus:
    """Health status of system components."""
    overall: str = "unknown"
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class CapibaraGPT:
    """
    Main CapibaraGPT Application.

    Orchestrates the initialization and execution of all system components
    including backends, routing, and safety systems.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize CapibaraGPT.

        Args:
            config: System configuration. Uses defaults if not provided.
        """
        self.config = config or SystemConfig()
        self.backend = None
        self.router = None
        self.initialized = False

    async def initialize(self) -> bool:
        """
        Initialize all system components.

        Returns:
            True if initialization successful, False otherwise.
        """
        logger.info("Initializing CapibaraGPT v%s...", __version__)

        try:
            # Initialize backend
            self.backend = await self._init_backend()
            if not self.backend:
                logger.warning("Backend initialization failed, using CPU fallback")

            # Initialize router (optional)
            if self.config.enable_routing:
                self.router = await self._init_router()

            self.initialized = True
            logger.info("CapibaraGPT initialized successfully")
            return True

        except Exception as e:
            logger.error("Initialization failed: %s", e)
            return False

    async def _init_backend(self):
        """Initialize compute backend based on configuration."""
        try:
            from core.backends import get_backend, BackendType

            backend_map = {
                "cpu": BackendType.CPU,
                "gpu": BackendType.GPU,
                "tpu": BackendType.TPU,
                "auto": BackendType.AUTO,
            }

            backend_type = backend_map.get(self.config.backend, BackendType.CPU)
            backend = get_backend(backend_type)

            if backend and backend.is_available:
                backend.initialize()
                logger.info("Backend initialized: %s", backend.name.upper())
                return backend

        except ImportError:
            logger.warning("Backend system not available")
        except Exception as e:
            logger.warning("Backend initialization error: %s", e)

        return None

    async def _init_router(self):
        """Initialize the semantic router."""
        try:
            from core.router import EnhancedRouter
            router = EnhancedRouter()
            logger.info("Enhanced router initialized")
            return router
        except ImportError:
            logger.debug("Router module not available")
        except Exception as e:
            logger.warning("Router initialization error: %s", e)
        return None

    async def health_check(self) -> HealthStatus:
        """
        Run comprehensive health check.

        Returns:
            HealthStatus with component status information.
        """
        status = HealthStatus()

        # Check backend
        if self.backend:
            try:
                info = self.backend.get_device_info()
                status.components["backend"] = {
                    "status": "healthy",
                    "type": self.backend.name,
                    "device": info.get("device", "unknown"),
                }
            except Exception as e:
                status.components["backend"] = {"status": "error", "error": str(e)}
                status.errors.append(f"Backend: {e}")
        else:
            status.components["backend"] = {"status": "unavailable"}

        # Check router
        if self.router:
            status.components["router"] = {"status": "healthy"}
        else:
            status.components["router"] = {"status": "unavailable"}

        # Determine overall status
        if status.errors:
            status.overall = "degraded"
        elif all(c.get("status") == "healthy" for c in status.components.values()):
            status.overall = "healthy"
        else:
            status.overall = "partial"

        return status

    async def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input through the system.

        Args:
            input_data: Input to process (text, structured data, etc.)

        Returns:
            Processing result dictionary.
        """
        if not self.initialized:
            return {"success": False, "error": "System not initialized"}

        try:
            # Route input if router available
            if self.router:
                route_result = await self.router.route_request(input_data)
                return {
                    "success": True,
                    "route": route_result,
                    "backend": self.backend.name if self.backend else "none",
                }

            return {
                "success": True,
                "message": "Processed without routing",
                "backend": self.backend.name if self.backend else "none",
            }

        except Exception as e:
            logger.error("Processing error: %s", e)
            return {"success": False, "error": str(e)}

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "version": __version__,
            "backend": self.backend.name if self.backend else "none",
            "initialized": self.initialized,
            "config": {
                "backend": self.config.backend,
                "model_type": self.config.model_type,
                "routing_enabled": self.config.enable_routing,
            },
        }


async def run_demo(app: CapibaraGPT):
    """Run demonstration of system capabilities."""
    logger.info("Running CapibaraGPT demonstration...")

    # Health check
    logger.info("Running health check...")
    health = await app.health_check()
    logger.info("Health status: %s", health.overall)

    for name, status in health.components.items():
        logger.info("  - %s: %s", name, status.get("status", "unknown"))

    # Test processing
    logger.info("Testing processing pipeline...")
    test_inputs = [
        "Hello, how are you?",
        "Explain quantum computing",
        {"type": "structured", "query": "test"},
    ]

    for i, test_input in enumerate(test_inputs, 1):
        result = await app.process(test_input)
        status = "OK" if result.get("success") else "FAILED"
        logger.info("  Test %d: %s", i, status)

    logger.info("Demonstration complete")


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CapibaraGPT v3 - Modular AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--backend",
        choices=["cpu", "gpu", "tpu", "auto"],
        default="cpu",
        help="Compute backend (default: cpu)",
    )
    parser.add_argument(
        "--model",
        choices=["transformer", "mamba", "hybrid"],
        default="hybrid",
        help="Model architecture (default: hybrid)",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Run health check and exit",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration mode",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show system information",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create configuration
    config = SystemConfig(
        backend=args.backend,
        model_type=args.model,
    )

    # Create and initialize application
    app = CapibaraGPT(config)

    if not await app.initialize():
        logger.error("Failed to initialize CapibaraGPT")
        return 1

    # Execute requested action
    if args.health:
        health = await app.health_check()
        logger.info("Health Status: %s", health.overall)
        for name, status in health.components.items():
            logger.info("  %s: %s", name, status)
        return 0 if health.overall in ("healthy", "partial") else 1

    elif args.demo:
        await run_demo(app)
        return 0

    elif args.info:
        info = app.get_system_info()
        logger.info("System Information:")
        for key, value in info.items():
            logger.info("  %s: %s", key, value)
        return 0

    else:
        # Default: show info and health
        info = app.get_system_info()
        logger.info("CapibaraGPT v%s", __version__)
        logger.info("Backend: %s", info["backend"])
        logger.info("Model: %s", config.model_type)

        health = await app.health_check()
        logger.info("Status: %s", health.overall)

        logger.info("Ready. Use --help for options.")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
