#!/usr/bin/env python3
"""
CapibaraGPT v3 - Minimal CLI

Provides --info, --health, and --demo entrypoints.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from main import CapibaraGPT, SystemConfig, run_demo, __version__

logger = logging.getLogger("capibaragpt")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CapibaraGPT v3 - CLI",
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
    parser.add_argument("--health", action="store_true", help="Run health check and exit")
    parser.add_argument("--demo", action="store_true", help="Run demonstration mode")
    parser.add_argument("--info", action="store_true", help="Show system information")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser


async def _run(args: argparse.Namespace) -> int:
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = SystemConfig(
        backend=args.backend,
        model_type=args.model,
    )

    app = CapibaraGPT(config)

    if not await app.initialize():
        logger.error("Failed to initialize CapibaraGPT")
        return 1

    if args.health:
        health = await app.health_check()
        logger.info("Health Status: %s", health.overall)
        for name, status in health.components.items():
            logger.info("  %s: %s", name, status)
        return 0 if health.overall in ("healthy", "partial") else 1

    if args.demo:
        await run_demo(app)
        return 0

    if args.info:
        info = app.get_system_info()
        logger.info("System Information:")
        for key, value in info.items():
            logger.info("  %s: %s", key, value)
        return 0

    logger.info("CapibaraGPT v%s", __version__)
    logger.info("Backend: %s", app.get_system_info()["backend"])
    logger.info("Model: %s", config.model_type)
    logger.info("Ready. Use --help for options.")
    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
