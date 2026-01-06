# Contenido de settings.py
"""
setup del system CapibaraGPT v2.
"""

import os

# import nore  # Fixed: removed incorrect import
from datetime import timedelta
from pathlib import Path #type: ignore  
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator #type: ignore

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
    cors_origins: List[str] = Field(["*"])

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

    @validator("security")
    def validate_security(cls, v):
        if len(v.api_key) < 32:
            raise ValueError("API key debe tener al menos 32 caracteres")
        return v

    @validator("database")
    def validate_database(cls, v):
        if not v.password:
            raise ValueError("Se requiere contraseña de base de datos")
        return v

def load_yaml_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    load la setup since un file YAML.
    
    Args:
        config_path: path al file de setup YAML
        
    Returns:
        Diccionario with la setup
    """
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
    
    with open(config_file) as f:
        return yaml.safe_load(f)

def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    load and valida la setup del system.
    
    Args:
        config_path: path al file de setup YAML
        
    Returns:
        object Settings with la setup validada
    """
    config_data = load_yaml_config(config_path)
    return Settings(**config_data)

# instance global de setup
settings = get_settings() 

# Contenido de paths.py
"""
setup de rutas for el proyecto Capibara.
"""

def get_project_root() -> Path:
    
    # Primero intenta obtain la path since la variable de entorno
    if 'CAPIBARA_ROOT' in os.environ:
        return Path(os.environ['CAPIBARA_ROOT'])
    
    # if not está definida, usa la path relativa al module current
    return Path(__file__).parent.parent

# Define important project paths
PROJECT_ROOT = get_project_root()
CONFIG_DIR = PROJECT_ROOT / 'config'
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'
CACHE_DIR = PROJECT_ROOT / 'cache'

# create directorios if not existen
for directory in [CONFIG_DIR, DATA_DIR, MODELS_DIR, LOGS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# configure rutas for diferentes entornos
def get_model_path(model_name: str) -> Path:
    """Gets la path for un model específico."""
    return MODELS_DIR / model_name

def get_data_path(data_name: str) -> Path:
    """Gets la path for un conjunto de data específico."""
    return DATA_DIR / data_name

def get_config_path(config_name: str) -> Path:
    """Gets la path for un file de setup específico."""
    return CONFIG_DIR / config_name 