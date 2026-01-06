"""
Configuration Management and Deployment Tools for Meta-Consensus System

This module provides comprehensive configuration management, validation, and deployment tools:
- Environment-specific configuration management
- Configuration validation and schema enforcement
- Secrets management and security
- Deployment automation scripts
- Configuration versioning and rollback
- Health checks and validation
"""

import logging
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import base64
from datetime import datetime
import shutil
import subprocess
from cryptography.fernet import Fernet
from jsonschema import validate, ValidationError
import tempfile

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class ConfigFormat(Enum):
    """Configurestion file formats."""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"

@dataclass
class ExpertConfig:
    """Configurestion for individual experts."""
    expert_id: str
    name: str
    model_id: str
    tier: str
    domain: str
    specialization: List[str]
    
    # Performance settings
    max_tokens: int = 512
    temperature: float = 0.7
    timeout_seconds: int = 30
    
    # Cost and resource settings
    cost_per_1k_tokens: float = 0.001
    max_concurrent_requests: int = 5
    
    # Quality settings
    quality_threshold: float = 8.0
    confidence_threshold: float = 0.7
    
    # API settings
    api_endpoint: Optional[str] = None
    api_key_name: Optional[str] = None
    use_local: bool = True

@dataclass
class SystemConfig:
    """Main system configuration."""
    
    # System identification
    system_name: str = "MetaConsensus"
    version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    
    # Core settings
    enable_serverless: bool = True
    enable_btx_training: bool = True
    enable_adaptive_routing: bool = True
    enable_monitoring: bool = True
    
    # Performance settings
    max_concurrent_experts: int = 10
    max_queries_per_minute: int = 100
    default_timeout_seconds: int = 30
    
    # Quality settings
    min_consensus_confidence: float = 0.7
    min_expert_quality: float = 8.0
    target_consensus_quality: float = 9.0
    
    # Cost settings
    max_cost_per_query: float = 0.05
    daily_cost_limit: float = 100.0
    cost_optimization_enabled: bool = True
    
    # Security settings
    enable_rate_limiting: bool = True
    enable_input_validation: bool = True
    enable_output_filtering: bool = True
    
    # Storage settings
    data_directory: str = "data"
    cache_directory: str = "cache"
    logs_directory: str = "logs"
    models_directory: str = "models"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    enable_cors: bool = True
    
    # Monitoring settings
    monitoring_enabled: bool = True
    monitoring_port: int = 8080
    metrics_retention_days: int = 30
    
    # Expert configurations
    experts: List[ExpertConfig] = field(default_factory=list)
    
    # Environment-specific overrides
    environment_overrides: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeploymentConfig:
    """Deployment-specific configuration."""
    
    # Infrastructure settings
    deployment_type: str = "standalone"  # standalone, docker, kubernetes
    replicas: int = 1
    cpu_limit: str = "2"
    memory_limit: str = "4Gi"
    
    # Networking
    external_port: int = 80
    internal_port: int = 8000
    load_balancer_enabled: bool = False
    
    # Storage
    persistent_storage: bool = True
    storage_size: str = "10Gi"
    backup_enabled: bool = True
    
    # Security
    tls_enabled: bool = True
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    
    # Scaling
    auto_scaling_enabled: bool = False
    min_replicas: int = 1
    max_replicas: int = 10
    cpu_threshold: int = 70
    
    # Health checks
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    startup_timeout_seconds: int = 120

class SecretManager:
    """Manages encrypted secrets and sensitive configuration."""
    
    def __init__(self, key_path: Optional[str] = None):
        self.key_path = key_path or os.path.join(os.path.expanduser("~"), ".metaconsensus", "secret.key")
        self.key = self._load_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _load_or_create_key(self) -> bytes:
        """Load existing key or create new one."""
        
        key_file = Path(self.key_path)
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Create new key
            key = Fernet.generate_key()
            key_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # Secure the key file
            os.chmod(key_file, 0o600)
            logger.info(f"Created new encryption key at {key_file}")
            
            return key
    
    def encrypt_secret(self, secret: str) -> str:
        """Encrypt a secret string."""
        encrypted = self.cipher.encrypt(secret.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt a secret string."""
        encrypted_bytes = base64.b64decode(encrypted_secret.encode())
        decrypted = self.cipher.decrypt(encrypted_bytes)
        return decrypted.decode()
    
    def encrypt_config_secrets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt secrets in configuration dictionary."""
        
        secret_keys = [
            'api_key', 'hf_api_token', 'openai_api_key', 'anthropic_api_key',
            'database_password', 'redis_password', 'jwt_secret'
        ]
        
        encrypted_config = config.copy()
        
        for key in secret_keys:
            if key in encrypted_config and encrypted_config[key]:
                encrypted_config[key] = self.encrypt_secret(str(encrypted_config[key]))
                encrypted_config[f"{key}_encrypted"] = True
        
        return encrypted_config
    
    def decrypt_config_secrets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt secrets in configuration dictionary."""
        
        decrypted_config = config.copy()
        
        for key, value in config.items():
            if key.endswith('_encrypted') and value:
                secret_key = key.replace('_encrypted', '')
                if secret_key in config:
                    decrypted_config[secret_key] = self.decrypt_secret(config[secret_key])
                    del decrypted_config[key]
        
        return decrypted_config

class ConfigValidator:
    """Validatess configuration against schemas and business rules."""
    
    def __init__(self):
        self.schema = self._load_config_schema()
    
    def _load_config_schema(self) -> Dict[str, Any]:
        """Load JSON schema for configuration validation."""
        
        return {
            "type": "object",
            "properties": {
                "system_name": {"type": "string", "minLength": 1},
                "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
                "environment": {"enum": [env.value for env in Environment]},
                "enable_serverless": {"type": "boolean"},
                "max_concurrent_experts": {"type": "integer", "minimum": 1, "maximum": 100},
                "min_consensus_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "max_cost_per_query": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "api_port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "experts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "expert_id": {"type": "string", "minLength": 1},
                            "name": {"type": "string", "minLength": 1},
                            "model_id": {"type": "string", "minLength": 1},
                            "tier": {"enum": ["local", "serverless", "premium", "specialized"]},
                            "domain": {"type": "string"},
                            "specialization": {"type": "array", "items": {"type": "string"}},
                            "max_tokens": {"type": "integer", "minimum": 1, "maximum": 4096},
                            "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                            "quality_threshold": {"type": "number", "minimum": 0.0, "maximum": 10.0}
                        },
                        "required": ["expert_id", "name", "model_id", "tier", "domain"]
                    }
                }
            },
            "required": ["system_name", "version", "environment"]
        }
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validates configuration and return list of errors."""
        
        errors = []
        
        try:
            # Schema validation
            validate(instance=config, schema=self.schema)
        except ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
        
        # Business rule validation
        errors.extend(self._validate_business_rules(config))
        
        return errors
    
    def _validate_business_rules(self, config: Dict[str, Any]) -> List[str]:
        """Validates business-specific rules."""
        
        errors = []
        
        # Check expert ID uniqueness
        if 'experts' in config:
            expert_ids = [expert.get('expert_id') for expert in config['experts']]
            if len(expert_ids) != len(set(expert_ids)):
                errors.append("Expert IDs must be unique")
        
        # Check cost limits
        max_cost = config.get('max_cost_per_query', 0)
        daily_limit = config.get('daily_cost_limit', 0)
        if max_cost * 1000 > daily_limit:  # Assuming 1000 queries per day max
            errors.append("Daily cost limit may be too low for max cost per query")
        
        # Check port conflicts
        api_port = config.get('api_port', 8000)
        monitoring_port = config.get('monitoring_port', 8080)
        if api_port == monitoring_port:
            errors.append("API port and monitoring port cannot be the same")
        
        # Environment-specific validations
        environment = config.get('environment', 'development')
        if environment == Environment.PRODUCTION.value:
            if not config.get('enable_input_validation', True):
                errors.append("Input validation must be enabled in production")
            if not config.get('tls_enabled', False):
                errors.append("TLS should be enabled in production")
        
        return errors

class ConfigManager:
    """Main configuration management class."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.secret_manager = SecretManager()
        self.validator = ConfigValidator()
        
        # Configuration cache
        self._config_cache: Dict[str, SystemConfig] = {}
        
        logger.info(f"ConfigManager initialized with directory: {self.config_dir}")
    
    def create_default_config(self, environment: Environment = Environment.DEVELOPMENT) -> SystemConfig:
        """Creates default configuration for specified environment."""
        
        # Base configuration
        config = SystemConfig(
            environment=environment,
            system_name=f"MetaConsensus-{environment.value}",
            experts=self._create_default_experts()
        )
        
        # Environment-specific overrides
        if environment == Environment.PRODUCTION:
            config.enable_rate_limiting = True
            config.enable_input_validation = True
            config.enable_output_filtering = True
            config.max_cost_per_query = 0.02
            config.daily_cost_limit = 50.0
            config.api_host = "0.0.0.0"
            config.monitoring_enabled = True
            
        elif environment == Environment.TESTING:
            config.max_concurrent_experts = 3
            config.max_cost_per_query = 0.001
            config.enable_serverless = False
            config.monitoring_enabled = False
            
        elif environment == Environment.DEVELOPMENT:
            config.enable_serverless = True
            config.max_cost_per_query = 0.01
            config.monitoring_enabled = True
            config.api_host = "127.0.0.1"
        
        return config
    
    def _create_default_experts(self) -> List[ExpertConfig]:
        """Creates default expert configurations."""
        
        return [
            ExpertConfig(
                expert_id="local_math",
                name="Local Math Expert",
                model_id="local/math_expert_300m",
                tier="local",
                domain="mathematics",
                specialization=["algebra", "calculus", "statistics"],
                cost_per_1k_tokens=0.0,
                use_local=True
            ),
            ExpertConfig(
                expert_id="local_code",
                name="Local Code Expert", 
                model_id="local/code_expert_600m",
                tier="local",
                domain="programming",
                specialization=["python", "javascript", "algorithms"],
                cost_per_1k_tokens=0.0,
                use_local=True
            ),
            ExpertConfig(
                expert_id="hf_advanced_math",
                name="HF Advanced Math Expert",
                model_id="microsoft/WizardMath-70B-V1.0",
                tier="serverless",
                domain="mathematics",
                specialization=["advanced_mathematics", "proofs", "complex_analysis"],
                cost_per_1k_tokens=0.0015,
                api_endpoint="https://api-inference.huggingface.co/models/",
                api_key_name="HF_API_TOKEN",
                use_local=False,
                quality_threshold=9.0
            ),
            ExpertConfig(
                expert_id="hf_code_generation",
                name="HF Code Generation Expert",
                model_id="Phind/Phind-CodeLlama-34B-v2",
                tier="serverless",
                domain="programming",
                specialization=["advanced_coding", "architecture", "optimization"],
                cost_per_1k_tokens=0.0012,
                api_endpoint="https://api-inference.huggingface.co/models/",
                api_key_name="HF_API_TOKEN",
                use_local=False,
                quality_threshold=8.8
            ),
            ExpertConfig(
                expert_id="premium_universal",
                name="Premium Universal Expert",
                model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                tier="premium",
                domain="universal",
                specialization=["expert_reasoning", "complex_analysis", "multi_domain"],
                cost_per_1k_tokens=0.0025,
                api_endpoint="https://api-inference.huggingface.co/models/",
                api_key_name="HF_API_TOKEN",
                use_local=False,
                quality_threshold=9.5,
                max_tokens=1024
            )
        ]
    
    def save_config(self, config: SystemConfig, filename: Optional[str] = None,
                   format: ConfigFormat = ConfigFormat.YAML, encrypt_secrets: bool = True) -> Path:
        """Save configuration to file."""
        
        if filename is None:
            filename = f"{config.environment.value}.{format.value}"
        
        file_path = self.config_dir / filename
        
        # Convert to dictionary
        config_dict = asdict(config)
        
        # Encrypt secrets if requested
        if encrypt_secrets:
            config_dict = self.secret_manager.encrypt_config_secrets(config_dict)
        
        # Save based on format
        if format == ConfigFormat.YAML:
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format == ConfigFormat.JSON:
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration saved to {file_path}")
        return file_path
    
    def load_config(self, filename: str, decrypt_secrets: bool = True) -> SystemConfig:
        """Load configuration from file."""
        
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Determine format from extension
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filename.endswith('.json'):
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        # Decrypt secrets if needed
        if decrypt_secrets:
            config_dict = self.secret_manager.decrypt_config_secrets(config_dict)
        
        # Validate configuration
        errors = self.validator.validate_config(config_dict)
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        # Convert experts list to ExpertConfig objects
        if 'experts' in config_dict:
            config_dict['experts'] = [
                ExpertConfig(**expert_data) for expert_data in config_dict['experts']
            ]
        
        # Convert environment string to enum
        if 'environment' in config_dict:
            config_dict['environment'] = Environment(config_dict['environment'])
        
        # Create SystemConfig object
        config = SystemConfig(**config_dict)
        
        # Cache the configuration
        self._config_cache[filename] = config
        
        logger.info(f"Configuration loaded from {file_path}")
        return config
    
    def get_config(self, environment: Environment) -> SystemConfig:
        """Get configuration for specified environment."""
        
        filename = f"{environment.value}.yaml"
        
        if filename in self._config_cache:
            return self._config_cache[filename]
        
        try:
            return self.load_config(filename)
        except FileNotFoundError:
            logger.warning(f"Configuration file {filename} not found, creating default")
            config = self.create_default_config(environment)
            self.save_config(config)
            return config
    
    def update_config(self, environment: Environment, updates: Dict[str, Any]) -> SystemConfig:
        """Update configuration with new values."""
        
        config = self.get_config(environment)
        config_dict = asdict(config)
        
        # Apply updates
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(config_dict, updates)
        
        # Recreate config object
        if 'experts' in config_dict:
            config_dict['experts'] = [
                ExpertConfig(**expert_data) if isinstance(expert_data, dict) else expert_data
                for expert_data in config_dict['experts']
            ]
        
        updated_config = SystemConfig(**config_dict)
        
        # Save updated configuration
        self.save_config(updated_config)
        
        return updated_config
    
    def list_configurations(self) -> List[str]:
        """List all available configuration files."""
        
        config_files = []
        for file_path in self.config_dir.glob("*.yaml"):
            config_files.append(file_path.name)
        for file_path in self.config_dir.glob("*.json"):
            config_files.append(file_path.name)
        
        return sorted(config_files)
    
    def backup_config(self, environment: Environment) -> Path:
        """Creates backup of configuration."""
        
        filename = f"{environment.value}.yaml"
        source_path = self.config_dir / filename
        
        if not source_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {source_path}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{environment.value}_{timestamp}.yaml.backup"
        backup_path = self.config_dir / "backups" / backup_filename
        
        backup_path.parent.mkdir(exist_ok=True)
        shutil.copy2(source_path, backup_path)
        
        logger.info(f"Configuration backed up to {backup_path}")
        return backup_path
    
    def restore_config(self, backup_path: Path) -> SystemConfig:
        """Restore configuration from backup."""
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Extract environment from backup filename
        backup_name = backup_path.stem.replace('.yaml', '')
        environment_name = backup_name.split('_')[0]
        environment = Environment(environment_name)
        
        # Copy backup to main config file
        main_config_path = self.config_dir / f"{environment.value}.yaml"
        shutil.copy2(backup_path, main_config_path)
        
        # Clear cache and reload
        filename = f"{environment.value}.yaml"
        if filename in self._config_cache:
            del self._config_cache[filename]
        
        config = self.load_config(filename)
        logger.info(f"Configuration restored from {backup_path}")
        
        return config

class DeploymentManager:
    """Manages deployment configurations and scripts."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.deployment_dir = Path("deployment")
        self.deployment_dir.mkdir(exist_ok=True)
    
    def generate_docker_compose(self, environment: Environment) -> Path:
        """Generates Docker Compose file for deployment."""
        
        config = self.config_manager.get_config(environment)
        
        compose_content = {
            'version': '3.8',
            'services': {
                'metaconsensus': {
                    'build': '.',
                    'ports': [f"{config.api_port}:{config.api_port}"],
                    'environment': [
                        f"ENVIRONMENT={environment.value}",
                        f"API_HOST={config.api_host}",
                        f"API_PORT={config.api_port}",
                        f"ENABLE_MONITORING={config.monitoring_enabled}"
                    ],
                    'volumes': [
                        "./config:/app/config:ro",
                        "./data:/app/data",
                        "./logs:/app/logs"
                    ],
                    'restart': 'unless-stopped',
                    'healthcheck': {
                        'test': ["CMD", "curl", "-f", f"http://localhost:{config.api_port}/health"],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    }
                }
            }
        }
        
        if config.monitoring_enabled:
            compose_content['services']['monitoring'] = {
                'build': '.',
                'command': ['python', '-m', 'capibara.training.monitoring_dashboard'],
                'ports': [f"{config.monitoring_port}:{config.monitoring_port}"],
                'environment': [
                    f"MONITORING_PORT={config.monitoring_port}"
                ],
                'depends_on': ['metaconsensus'],
                'restart': 'unless-stopped'
            }
        
        compose_file = self.deployment_dir / f"docker-compose.{environment.value}.yml"
        
        with open(compose_file, 'w') as f:
            yaml.dump(compose_content, f, default_flow_style=False, indent=2)
        
        logger.info(f"Docker Compose file generated: {compose_file}")
        return compose_file
    
    def generate_dockerfile(self) -> Path:
        """Generates Dockerfile for the application."""
        
        dockerfile_content = '''FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/cache /app/config

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Expose ports
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "capibara.training.meta_consensus_system"]
'''
        
        dockerfile_path = Path("Dockerfile")
        
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        logger.info(f"Dockerfile generated: {dockerfile_path}")
        return dockerfile_path
    
    def generate_kubernetes_manifest(self, environment: Environment) -> Path:
        """Generates Kubernetes deployment manifest."""
        
        config = self.config_manager.get_config(environment)
        
        manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f'metaconsensus-{environment.value}',
                'labels': {
                    'app': 'metaconsensus',
                    'environment': environment.value
                }
            },
            'spec': {
                'replicas': 3 if environment == Environment.PRODUCTION else 1,
                'selector': {
                    'matchLabels': {
                        'app': 'metaconsensus',
                        'environment': environment.value
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'metaconsensus',
                            'environment': environment.value
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'metaconsensus',
                            'image': f'metaconsensus:{environment.value}',
                            'ports': [
                                {'containerPort': config.api_port},
                                {'containerPort': config.monitoring_port}
                            ],
                            'env': [
                                {'name': 'ENVIRONMENT', 'value': environment.value},
                                {'name': 'API_PORT', 'value': str(config.api_port)}
                            ],
                            'resources': {
                                'requests': {
                                    'memory': '1Gi',
                                    'cpu': '500m'
                                },
                                'limits': {
                                    'memory': '4Gi',
                                    'cpu': '2'
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': config.api_port
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': config.api_port
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        manifest_file = self.deployment_dir / f"k8s-deployment.{environment.value}.yaml"
        
        with open(manifest_file, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False, indent=2)
        
        logger.info(f"Kubernetes manifest generated: {manifest_file}")
        return manifest_file
    
    def generate_deployment_script(self, environment: Environment) -> Path:
        """Generates deployment script."""
        
        script_content = f'''#!/bin/bash

# Meta-Consensus Deployment Script for {environment.value}
set -e

echo "🚀 Deploying Meta-Consensus System ({environment.value})"

# Configuration validation
echo "📋 Validating configuration..."
python -c "
from capibara.training.config_manager import ConfigManager
from capibara.training.config_manager import Environment
config_manager = ConfigManager()
config = config_manager.get_config(Environment.{environment.value.upper()})
print('✅ Configuration validated')
"

# Build Docker image
echo "🏗️ Building Docker image..."
docker build -t metaconsensus:{environment.value} .

# Run deployment
if [ "$1" = "docker" ]; then
    echo "🐳 Deploying with Docker Compose..."
    docker-compose -f deployment/docker-compose.{environment.value}.yml up -d
elif [ "$1" = "k8s" ]; then
    echo "☸️ Deploying to Kubernetes..."
    kubectl apply -f deployment/k8s-deployment.{environment.value}.yaml
else
    echo "🖥️ Running standalone..."
    docker run -d \\
        --name metaconsensus-{environment.value} \\
        -p 8000:8000 \\
        -p 8080:8080 \\
        -v $(pwd)/config:/app/config:ro \\
        -v $(pwd)/data:/app/data \\
        -v $(pwd)/logs:/app/logs \\
        metaconsensus:{environment.value}
fi

echo "✅ Deployment completed!"
echo "📊 Dashboard: http://localhost:8080"
echo "🔗 API: http://localhost:8000"
'''
        
        script_path = self.deployment_dir / f"deploy-{environment.value}.sh"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Deployment script generated: {script_path}")
        return script_path

# Utility functions
def create_config_manager(config_dir: str = "config") -> ConfigManager:
    """Creates and initialize configuration manager."""
    return ConfigManager(config_dir)

def setup_environment(environment: Environment, config_dir: str = "config") -> SystemConfig:
    """Setup configuration for specified environment."""
    
    config_manager = ConfigManager(config_dir)
    config = config_manager.create_default_config(environment)
    config_manager.save_config(config)
    
    return config


# Export main components
__all__ = [
    'ConfigManager',
    'DeploymentManager',
    'SystemConfig',
    'ExpertConfig',
    'DeploymentConfig',
    'SecretManager',
    'ConfigValidator',
    'Environment',
    'ConfigFormat',
    'create_config_manager',
    'setup_environment'
]


if __name__ == "__main__":
    # Example usage and CLI
    import argparse
    
    parser = argparse.ArgumentParser(description="Meta-Consensus Configuration Manager")
    parser.add_argument("action", choices=["create", "validate", "deploy"], help="Action to perform")
    parser.add_argument("--environment", "-e", type=str, choices=[env.value for env in Environment],
                       default="development", help="Target environment")
    parser.add_argument("--config-dir", "-c", type=str, default="config", help="Configuration directory")
    parser.add_argument("--deployment-type", "-t", type=str, choices=["docker", "k8s", "standalone"],
                       default="standalone", help="Deployment type")
    
    args = parser.parse_args()
    
    environment = Environment(args.environment)
    config_manager = ConfigManager(args.config_dir)
    
    if args.action == "create":
        print(f"Creating configuration for {environment.value}...")
        config = config_manager.create_default_config(environment)
        config_path = config_manager.save_config(config)
        print(f"✅ Configuration created: {config_path}")
        
    elif args.action == "validate":
        print(f"Validating configuration for {environment.value}...")
        try:
            config = config_manager.get_config(environment)
            print("✅ Configuration is valid")
            print(f"System: {config.system_name}")
            print(f"Version: {config.version}")
            print(f"Experts: {len(config.experts)}")
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            
    elif args.action == "deploy":
        print(f"Generating deployment files for {environment.value}...")
        deployment_manager = DeploymentManager(config_manager)
        
        # Generate deployment files
        dockerfile = deployment_manager.generate_dockerfile()
        compose_file = deployment_manager.generate_docker_compose(environment)
        k8s_manifest = deployment_manager.generate_kubernetes_manifest(environment)
        deploy_script = deployment_manager.generate_deployment_script(environment)
        
        print("✅ Deployment files generated:")
        print(f"  📄 Dockerfile: {dockerfile}")
        print(f"  🐳 Docker Compose: {compose_file}")
        print(f"  ☸️ Kubernetes: {k8s_manifest}")
        print(f"  🚀 Deploy Script: {deploy_script}")
        print(f"\nTo deploy: ./deployment/deploy-{environment.value}.sh {args.deployment_type}")