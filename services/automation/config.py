"""
Configuration for CapibaraGPT v3 N8N Automation Service
==================================================

Configuration management with environment variable support
and default settings for the automation service.
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

import logging
logger = logging.getLogger(__name__)


@dataclass
class N8nConfig:
    """Configuration for n8n integration."""
    base_url: str = "http://localhost:5678"
    api_key: Optional[str] = None
    webhook_url: str = "http://localhost:5678/webhook"
    health_check_endpoint: str = "/healthz"
    workflows_endpoint: str = "/api/v1/workflows"
    executions_endpoint: str = "/api/v1/executions"
    timeout: int = 30


@dataclass
class E2bConfig:
    """Configuration for E2b sandbox integration."""
    api_key: Optional[str] = None
    default_template: str = "python3"
    default_timeout: int = 300
    default_memory_limit: int = 1024
    max_sandboxes_per_session: int = 5
    sandbox_cleanup_interval: int = 1800  # 30 minutes
    max_sandbox_age_hours: int = 6


@dataclass
class AgentConfig:
    """Configuration for Capibara agent integration."""
    default_agent_type: str = "capibara_base"
    max_execution_time: int = 300
    enable_memory: bool = True
    memory_cleanup_interval: int = 3600  # 1 hour
    max_agent_instances: int = 10


@dataclass
class ApiConfig:
    """Configuration for REST API server."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"
    cors_origins: list = field(default_factory=lambda: os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:8000"
    ).split(","))
    api_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"


@dataclass
class SessionConfig:
    """Configuration for session management."""
    session_timeout_hours: int = 24
    max_workflows_per_session: int = 50
    max_executions_per_session: int = 200
    cleanup_interval: int = 3600  # 1 hour
    enable_persistence: bool = False
    persistence_backend: str = "memory"  # memory, redis, database


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_structured_logging: bool = True


@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    enable_api_key_auth: bool = False
    api_key: Optional[str] = None
    enable_cors: bool = True
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    rate_limit_per_minute: int = 100
    enable_https: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None


@dataclass
class PerformanceConfig:
    """Configuration for performance settings."""
    max_concurrent_workflows: int = 50
    max_concurrent_executions: int = 20
    workflow_cache_size: int = 1000
    execution_history_size: int = 1000
    enable_metrics: bool = True
    metrics_export_interval: int = 60


@dataclass
class AutomationServiceConfig:
    """Main configuration class for the automation service."""
    
    # Component configurations
    n8n: N8nConfig = field(default_factory=N8nConfig)
    e2b: E2bConfig = field(default_factory=E2bConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    sessions: SessionConfig = field(default_factory=SessionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Service settings
    service_name: str = "capibara-v3-n8n-automation"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    
    # Data directories
    data_dir: str = "./data"
    logs_dir: str = "./logs"
    temp_dir: str = "./temp"
    
    @classmethod
    def from_env(cls) -> 'AutomationServiceConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # N8N configuration
        config.n8n.base_url = os.getenv("N8N_BASE_URL", config.n8n.base_url)
        config.n8n.api_key = os.getenv("N8N_API_KEY", config.n8n.api_key)
        config.n8n.webhook_url = os.getenv("N8N_WEBHOOK_URL", config.n8n.webhook_url)
        config.n8n.timeout = int(os.getenv("N8N_TIMEOUT", str(config.n8n.timeout)))
        
        # E2b configuration
        config.e2b.api_key = os.getenv("E2B_API_KEY", config.e2b.api_key)
        config.e2b.default_template = os.getenv("E2B_DEFAULT_TEMPLATE", config.e2b.default_template)
        config.e2b.default_timeout = int(os.getenv("E2B_DEFAULT_TIMEOUT", str(config.e2b.default_timeout)))
        config.e2b.default_memory_limit = int(os.getenv("E2B_MEMORY_LIMIT", str(config.e2b.default_memory_limit)))
        
        # Agent configuration
        config.agents.default_agent_type = os.getenv("AGENT_TYPE", config.agents.default_agent_type)
        config.agents.max_execution_time = int(os.getenv("AGENT_MAX_EXECUTION_TIME", str(config.agents.max_execution_time)))
        config.agents.enable_memory = os.getenv("AGENT_ENABLE_MEMORY", "true").lower() == "true"
        
        # API configuration
        config.api.host = os.getenv("API_HOST", config.api.host)
        config.api.port = int(os.getenv("API_PORT", str(config.api.port)))
        config.api.reload = os.getenv("API_RELOAD", "false").lower() == "true"
        config.api.log_level = os.getenv("API_LOG_LEVEL", config.api.log_level)
        
        # Session configuration
        config.sessions.session_timeout_hours = int(os.getenv("SESSION_TIMEOUT_HOURS", str(config.sessions.session_timeout_hours)))
        config.sessions.enable_persistence = os.getenv("SESSION_PERSISTENCE", "false").lower() == "true"
        config.sessions.persistence_backend = os.getenv("SESSION_BACKEND", config.sessions.persistence_backend)
        
        # Logging configuration
        config.logging.level = os.getenv("LOG_LEVEL", config.logging.level)
        config.logging.file_path = os.getenv("LOG_FILE_PATH", config.logging.file_path)
        
        # Security configuration
        config.security.enable_api_key_auth = os.getenv("ENABLE_API_KEY_AUTH", "false").lower() == "true"
        config.security.api_key = os.getenv("API_KEY", config.security.api_key)
        config.security.enable_https = os.getenv("ENABLE_HTTPS", "false").lower() == "true"
        config.security.ssl_cert_path = os.getenv("SSL_CERT_PATH", config.security.ssl_cert_path)
        config.security.ssl_key_path = os.getenv("SSL_KEY_PATH", config.security.ssl_key_path)
        
        # Service settings
        config.environment = os.getenv("ENVIRONMENT", config.environment)
        config.debug = os.getenv("DEBUG", "false").lower() == "true"
        config.data_dir = os.getenv("DATA_DIR", config.data_dir)
        config.logs_dir = os.getenv("LOGS_DIR", config.logs_dir)
        config.temp_dir = os.getenv("TEMP_DIR", config.temp_dir)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "n8n": {
                "base_url": self.n8n.base_url,
                "api_key": "***" if self.n8n.api_key else None,
                "webhook_url": self.n8n.webhook_url,
                "timeout": self.n8n.timeout
            },
            "e2b": {
                "api_key": "***" if self.e2b.api_key else None,
                "default_template": self.e2b.default_template,
                "default_timeout": self.e2b.default_timeout,
                "default_memory_limit": self.e2b.default_memory_limit
            },
            "agents": {
                "default_agent_type": self.agents.default_agent_type,
                "max_execution_time": self.agents.max_execution_time,
                "enable_memory": self.agents.enable_memory
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "log_level": self.api.log_level
            },
            "sessions": {
                "session_timeout_hours": self.sessions.session_timeout_hours,
                "enable_persistence": self.sessions.enable_persistence,
                "persistence_backend": self.sessions.persistence_backend
            },
            "security": {
                "enable_api_key_auth": self.security.enable_api_key_auth,
                "enable_https": self.security.enable_https
            },
            "service": {
                "name": self.service_name,
                "version": self.version,
                "environment": self.environment,
                "debug": self.debug
            }
        }
    
    def create_directories(self):
        """Create necessary directories."""
        directories = [self.data_dir, self.logs_dir, self.temp_dir]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate N8N configuration
        if not self.n8n.base_url.startswith(("http://", "https://")):
            errors.append("N8N base URL must start with http:// or https://")
        
        # Validate API configuration
        if not (1 <= self.api.port <= 65535):
            errors.append("API port must be between 1 and 65535")
        
        # Validate E2b configuration
        if self.e2b.default_timeout <= 0:
            errors.append("E2b timeout must be positive")
        
        if self.e2b.default_memory_limit <= 0:
            errors.append("E2b memory limit must be positive")
        
        # Validate agent configuration
        if self.agents.max_execution_time <= 0:
            errors.append("Agent max execution time must be positive")
        
        # Validate session configuration
        if self.sessions.session_timeout_hours <= 0:
            errors.append("Session timeout must be positive")
        
        # Validate security configuration
        if self.security.enable_https:
            if not self.security.ssl_cert_path or not self.security.ssl_key_path:
                errors.append("SSL certificate and key paths required when HTTPS is enabled")
        
        return errors


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
            logger.error(f"Warning: Failed to load config file {config_file}: {e}")
    
    # Validate configuration
    errors = config.validate()
    if errors:
        logger.error("Configuration validation errors:")
        for error in errors:
            logger.error(f"  - {error}")
    
    # Create necessary directories
    config.create_directories()
    
    return config


def get_example_config() -> str:
    """Get an example configuration file in YAML format."""
    return """
# CapibaraGPT v3 N8N Automation Service Configuration
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
  cors_origins: ["http://localhost:3000", "http://localhost:8000"]

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
