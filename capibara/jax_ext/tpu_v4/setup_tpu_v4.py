#!/usr/bin/env python3
"""Automated installation script for TPU v4-32."""

import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def install_dependencies() -> bool:
    """Install all required dependencies."""
    deps = [
        "cmake>=3.18",
        "ninja",
        "clang>=12.0",
        "nanobind>=1.8.0",
        "absl-py>=1.0.0",
        "packaging>=21.0"
    ]

    try:
        for dep in deps:
            logger.info(f"Installing {dep}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", dep],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"Error installing {dep}: {result.stderr}")
                return False

        return True
    except Exception as e:
        logger.error(f"Error during installation: {e}")
        return False


def setup_logging(log_file: Optional[Path] = None):
    """Configure logging system."""
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def main():
    """Main installation function."""
    # Configure logging
    setup_logging(Path("tpu_v4_install.log"))

    logger.info("Installing TPU v4-32 backend for JAX...")

    # 1. Install dependencies
    if not install_dependencies():
        logger.error("Error installing dependencies")
        sys.exit(1)

    # 2. Build
    try:
        result = subprocess.run(
            [sys.executable, "build.py", "--build", "--install", "--test"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Error during build: {result.stderr}")
            sys.exit(1)

        logger.info("Installation complete!")

    except Exception as e:
        logger.error(f"Error during build: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
