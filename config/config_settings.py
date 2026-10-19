# Settings content
"""
CapibaraGPT v3 system setup.
"""

import os

# import nore  # Fixed: removed incorrect import
from datetime import timedelta

try:
    import yaml
except ImportError:
    yaml = None
from pathlib import Path #type: ignore  
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator #type: ignore

class SecuritySettings(BaseModel):
    """Security configuration."""
    api_key: str = Field(..., min_length=32)
    rate_limit: int = Field(100, ge=1, le=1000)
    jwt_secret: str = Field(..., min_length=32)
    jwt_algorithm: str = Field("HS256")
    token_expiry: timedelta = Field(timedelta(hours=1))

class DatabaseSettings(BaseModel):
    """Database configuration."""
    host: str = Field("localhost")
    port: int = Field(5432, ge=1, le=65535)
    name: str = Field("capibara")
    user: str = Field("postgres")
    password: str = Field(...)
    pool_size: int = Field(5, ge=1, le=20)
    timeout: int = Field(30, ge=1)

class ModelSettings(BaseModel):
    """Model configuration."""
    health_advisor: Dict[str, Any] = Field(
        default_factory=lambda: {
            "min_confidence": 0.7,
            "cache_ttl": 3600
        }
    )
    doc_retriever: Dict[str, Any] = Field(
        default_factory=lambda: {
            "embedding_model": "all-MiniLM-L6-v2",
            "max_results": 5,
            "cache_ttl": 3600
        }
    )
    veracity_verifier: Dict[str, Any] = Field(
        default_factory=lambda: {
            "min_confidence": 0.8,
            "max_evidence": 5,
            "cache_ttl": 3600
        }
    )

class APISettings(BaseModel):
    """API configuration."""
    host: str = Field("0.0.0.0")
    port: int = Field(8000, ge=1, le=65535)
    workers: int = Field(4, ge=1, le=32)
    timeout: int = Field(30, ge=1)
    cors_origins: List[str] = Field(["http://localhost:3000", "http://localhost:8000"])

class LoggingSettings(BaseModel):
    """Logging configuration."""
    level: str = Field("INFO")
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file: Optional[str] = None
    max_size: int = Field(10_000_000)  # 10MB
    backup_count: int = Field(5)

class CacheSettings(BaseModel):
    """cache configuration."""
    enabled: bool = Field(True)
    ttl: int = Field(3600, ge=1)
    max_size: int = Field(1000, ge=1)
    backend: str = Field("memory")  # memory, redis

class Settings(BaseModel):
    """Main system configuration."""
    security: SecuritySettings
    database: DatabaseSettings
    models: ModelSettings
    api: APISettings
    logging: LoggingSettings
    cache: CacheSettings

    @field_validator("security")
    def validate_security(cls, v):
        if len(v.api_key) < 32:
            raise ValueError("API key must have at least 32 characters")
        return v

    @field_validator("database")
    def validate_database(cls, v):
        if not v.password:
            raise ValueError("Database password is required")
        return v

def load_yaml_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary with the configuration
    """
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if yaml is None:
        raise ImportError("PyYAML is required to load YAML config files. Install with: pip install pyyaml")

    with open(config_file) as f:
        return yaml.safe_load(f)

def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    Load and validate the system configuration.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Settings object with the validated configuration
    """
    config_data = load_yaml_config(config_path)
    return Settings(**config_data)

# Global settings instance (lazy initialization to avoid import-time failures)
settings: Optional[Settings] = None

def _get_lazy_settings() -> Optional[Settings]:
    """Get settings with lazy initialization."""
    global settings
    if settings is None:
        try:
            settings = get_settings()
        except (FileNotFoundError, ImportError, Exception):
            # Config file may not exist or have incompatible format
            pass
    return settings

# paths.py content
"""
Path configuration for the Capibara project.
"""

def get_project_root() -> Path:
    """Get the project root path."""
    # First try to get the path from environment variable
    if 'CAPIBARA_ROOT' in os.environ:
        return Path(os.environ['CAPIBARA_ROOT'])

    # If not defined, use the path relative to current module
    return Path(__file__).parent.parent

# Define important project paths
PROJECT_ROOT = get_project_root()
CONFIG_DIR = PROJECT_ROOT / 'config'
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'
CACHE_DIR = PROJECT_ROOT / 'cache'

# Create directories if they don't exist
for directory in [CONFIG_DIR, DATA_DIR, MODELS_DIR, LOGS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configure paths for different environments
def get_model_path(model_name: str) -> Path:
    """Get the path for a specific model."""
    return MODELS_DIR / model_name

def get_data_path(data_name: str) -> Path:
    """Get the path for a specific data set."""
    return DATA_DIR / data_name

def get_config_path(config_name: str) -> Path:
    """Get the path for a specific configuration file."""
    return CONFIG_DIR / config_name 
