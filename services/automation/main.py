#!/usr/bin/env python3
"""
Main script to run Capibara6 N8N Automation Service

This script can be used to run the automation service as a standalone application
with configuration from environment variables and optional config file.
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path

# Add project root to path
# Fixed: Using relative imports instead of sys.path manipulation

from capibara.services.automation import (
    CapibaraN8nAutomationService,
    create_automation_service
)
from capibara.services.automation.config import load_config, get_example_config


def setup_logging(config):
    """Setup logging configuration."""
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(config.logging.format)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if config.logging.file_path:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            config.logging.file_path,
            maxBytes=config.logging.max_file_size,
            backupCount=config.logging.backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


async def run_service(config):
    """Run the automation service."""
    # Create service instance
    service = create_automation_service(config.to_dict())
    
    try:
        # Start the service
        await service.startup()
        
        logging.info(f"Capibara6 N8N Automation Service started successfully")
        logging.info(f"API server running on {config.api.host}:{config.api.port}")
        logging.info(f"Environment: {config.environment}")
        logging.info(f"Debug mode: {config.debug}")
        
        # Run the FastAPI server
        if service.app:
            import uvicorn
            uvicorn_config = uvicorn.Config(
                service.app,
                host=config.api.host,
                port=config.api.port,
                log_level=config.api.log_level,
                reload=config.api.reload
            )
            server = uvicorn.Server(uvicorn_config)
            await server.serve()
        else:
            logging.error("FastAPI application not available")
            return 1
    
    except KeyboardInterrupt:
        logging.info("Received shutdown signal")
    except Exception as e:
        logging.error(f"Service failed: {e}", exc_info=True)
        return 1
    finally:
        # Shutdown the service
        await service.shutdown()
        logging.info("Service shutdown complete")
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Capibara6 N8N Automation Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run with default config
  python main.py --config config.yaml     # Run with config file
  python main.py --example-config         # Show example config
  python main.py --host 0.0.0.0 --port 8080  # Override host and port
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (YAML)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        help="API server host (overrides config)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        help="API server port (overrides config)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--example-config",
        action="store_true",
        help="Print example configuration and exit"
    )
    
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    args = parser.parse_args()
    
    # Show example config if requested
    if args.example_config:
        print(get_example_config())
        return 0
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Apply command line overrides
    if args.host:
        config.api.host = args.host
    if args.port:
        config.api.port = args.port
    if args.debug:
        config.debug = True
        config.logging.level = "DEBUG"
    
    # Validate config if requested
    if args.validate_config:
        errors = config.validate()
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return 1
        else:
            print("Configuration is valid")
            return 0
    
    # Setup logging
    setup_logging(config)
    
    # Run the service
    try:
        return asyncio.run(run_service(config))
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        return 0
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())