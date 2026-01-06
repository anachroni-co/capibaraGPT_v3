"""
Configuration for Capibara6 N8N Automation Service

Configuration management with environment variable support
and default settings for the automation service.
"""

import os

from typing import Dict, Any

class Config:
    """Configurestion manager for automation."""
    
    def __init__(self):
        self.settings = {}
    
    # Service settings
    service_name: str = "capibara6-n8n-automation"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.settings[key] = value


# Default configuration instance
DEFAULT_CONFIG = AutomationServiceConfig()


def load_config(config_file: Optional[str] = None) -> AutomationServiceConfig:
    """
    Load configuration from environment variables and optional config file.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        Loaded configuration
    """
    # Start with environment-based configuration
    config = AutomationServiceConfig.from_env()
    
    # Load from file if provided
    if config_file and os.path.exists(config_file):
        try:
            import yaml
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
            
            # Merge file configuration (simplified merge)
            if 'n8n' in file_config:
                for key, value in file_config['n8n'].items():
                    if hasattr(config.n8n, key):
                        setattr(config.n8n, key, value)
            
            if 'e2b' in file_config:
                for key, value in file_config['e2b'].items():
                    if hasattr(config.e2b, key):
                        setattr(config.e2b, key, value)
            
            # ... similar for other sections
            
        except Exception as e:
            print(f"Warning: Failed to load config file {config_file}: {e}")
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Create necessary directories
    config.create_directories()
    
    return config


def get_example_config() -> str:
    """Get an example configuration file in YAML format."""
    return """
# Capibara6 N8N Automation Service Configuration
# =============================================

n8n:
  base_url: "http://localhost:5678"
  api_key: null  # Set N8N_API_KEY environment variable
  webhook_url: "http://localhost:5678/webhook"
  timeout: 30

e2b:
  api_key: null  # Set E2B_API_KEY environment variable
  default_template: "python3"
  default_timeout: 300
  default_memory_limit: 1024
  max_sandboxes_per_session: 5

agents:
  default_agent_type: "capibara_base"
  max_execution_time: 300
  enable_memory: true
  max_agent_instances: 10

api:
  host: "0.0.0.0"
  port: 8000
  reload: false
  log_level: "info"
  cors_origins: ["*"]

sessions:
  session_timeout_hours: 24
  max_workflows_per_session: 50
  enable_persistence: false
  persistence_backend: "memory"

logging:
  level: "INFO"
  file_path: null  # Set to enable file logging
  enable_structured_logging: true

security:
  enable_api_key_auth: false
  api_key: null
  enable_cors: true
  enable_https: false

performance:
  max_concurrent_workflows: 50
  max_concurrent_executions: 20
  workflow_cache_size: 1000

service:
  environment: "development"
  debug: false
  data_dir: "./data"
  logs_dir: "./logs"
  temp_dir: "./temp"
"""