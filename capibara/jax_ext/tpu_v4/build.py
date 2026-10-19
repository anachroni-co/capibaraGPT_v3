"""
tpu_v4 build module.

# This module provides functionality for build.
"""

import os
import sys
import argparse

from capibara.jax.tpu_v4.builder import TpuV4Builder
from capibara.jax.tpu_v4.build_config import TpuV4BuildConfig
from capibara.jax.tpu_v4.performance_test import TpuV4PerformanceTest

import logging
logger = logging.getLogger(__name__)

def main():
    """Main build script for tpu v4-32 backend."""
    parser = argparse.ArgumentParser(description="Build JAX TPU v4-32 Backend")
    parser.add_argument("--build", action="store_true", 
                       help="Build the TPU v4-32 backend")
    parser.add_argument("--install", action="store_true",
                       help="Install the TPU v4-32 backend")
    parser.add_argument("--test", action="store_true",
                       help="Run tests for the TPU v4-32 backend")
    parser.add_argument("--clean", action="store_true",
                       help="Clean build directory before building")
    parser.add_argument("--performance-test", action="store_true",
                       help="Run comprehensive performance tests")
    parser.add_argument("--config", type=str,
                       help="Path to custom build configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # Load custom configuration (implementation would parse config file)
        config = TpuV4BuildConfig()
    else:
        config = TpuV4BuildConfig()
    
    builder = TpuV4Builder(config)
    
    success = True
    
    if args.performance_test:
        try:
            results = TpuV4PerformanceTest.run_comprehensive_tests()
            logger.info("\n" + "=" * 60)
            logger.info("Performance test completed successfully!")
        except Exception as e:
            logger.error(f" Performance tests failed: {e}")
            success = False
        return
    
    # Generate build files
    builder.generate_build_files()
    
    if args.build:
        if not builder.build(clean=args.clean):
            success = False
    
    if args.install and success:
        if not builder.install():
            success = False
    
    if args.test and success:
        if not builder.test():
            success = False
    
    if success:
        logger.info("\n JAX TPU v4-32 backend build completed successfully!")
        logger.info("\nTo use the TPU v4-32 backend:")
        logger.info("  import jax")
        logger.info("  # TPU v4 build completed successfully")
        logger.info("  # Your JAX code will automatically use tpu v4-32 optimizations")
    else:
        logger.error("\n Build failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
