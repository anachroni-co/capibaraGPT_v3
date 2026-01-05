#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuración específica para infraestructura de Google Cloud.
"""

import os
from typing import Dict, Any

# Configuración de infraestructura
INFRASTRUCTURE_CONFIG = {
    # Modelo 20B en ARM Axion
    'model_20b': {
        'platform': 'gcloud_arm_axion',
        'specs': {
            'vcpus': 32,
            'ram_gb': 64,
            'architecture': 'ARM'
        },
        'performance': {
            'max_tokens': 8000,
            'expected_latency_ms': 2000,
            'throughput_tokens_per_sec': 4000
        },
        'optimization': {
            'use_quantization': True,
            'quantization_bits': 4,
            'batch_size': 1,
            'max_sequence_length': 8192
        }
    },
    
    # Modelo 120B en H100
    'model_120b': {
        'platform': 'gcloud_h100',
        'specs': {
            'gpus': 2,
            'gpu_type': 'H100',
            'gpu_memory_gb': 80,
            'architecture': 'CUDA'
        },
        'performance': {
            'max_tokens': 32000,
            'expected_latency_ms': 10000,
            'throughput_tokens_per_sec': 2000
        },
        'optimization': {
            'use_quantization': False,
            'use_tensor_parallelism': True,
            'tensor_parallel_size': 2,
            'batch_size': 1,
            'max_sequence_length': 32768
        }
    },
    
    # TPU V5e-64 para entrenamiento
    'training_tpu': {
        'platform': 'gcloud_tpu_v5e',
        'specs': {
            'tpu_type': 'v5e-64',
            'tpu_cores': 64,
            'memory_per_core_gb': 16
        },
        'training': {
            'max_sequence_length': 32768,
            'batch_size_per_core': 1,
            'gradient_accumulation_steps': 4,
            'learning_rate': 1e-5,
            'warmup_steps': 1000
        }
    }
}

# Configuración de umbrales optimizada para la infraestructura
OPTIMIZED_THRESHOLDS = {
    'complexity_threshold': 0.7,
    'domain_confidence_threshold': 0.6,
    
    # Umbrales de latencia específicos para la infraestructura
    'max_latency_20b_ms': 2000,  # ARM Axion puede ser más lento
    'max_latency_120b_ms': 10000,  # H100 es más rápido
    
    # Umbrales de tokens optimizados
    'max_tokens_20b': 8000,
    'max_tokens_120b': 32000,
    
    # Umbrales de calidad
    'min_quality_score': 0.7,
    'min_success_rate': 0.85,
    
    # Umbrales de escalación automática
    'auto_escalate_complexity': 0.9,
    'auto_escalate_domain_uncertainty': 0.3,
    
    # Umbrales de fallback
    'fallback_to_20b_threshold': 0.4,
    'emergency_fallback_threshold': 0.2
}

# Configuración de RAG optimizada
RAG_CONFIG = {
    'mini_rag': {
        'timeout_ms': 50,
        'max_results': 5,
        'cache_size': 1000,
        'cache_ttl_seconds': 300
    },
    'full_rag': {
        'max_results': 10,
        'expansion_factor': 2.0,
        'deep_search_timeout_ms': 200
    },
    'vector_store': {
        'type': 'faiss',  # Usar FAISS local para desarrollo
        'index_type': 'IndexFlatIP',  # Inner Product para cosine similarity
        'embedding_dimension': 384
    }
}

# Configuración de CAG optimizada
CAG_CONFIG = {
    'mini_cag': {
        'max_tokens': 8000,
        'token_limits': {
            'static_cache': 3200,  # 40%
            'dynamic_context': 2400,  # 30%
            'rag': 2400  # 30%
        }
    },
    'full_cag': {
        'max_tokens': 32000,
        'token_limits': {
            'static_cache': 8000,  # 25%
            'dynamic_context': 6400,  # 20%
            'mini_rag': 6400,  # 20%
            'full_rag': 11200  # 35%
        }
    }
}

# Configuración de logging para producción
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_rotation': {
        'max_bytes': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    },
    'components': {
        'router': 'INFO',
        'cag': 'INFO',
        'rag': 'INFO',
        'ace': 'INFO',
        'execution': 'INFO',
        'agents': 'INFO',
        'metadata': 'INFO'
    }
}

# Configuración de monitoreo
MONITORING_CONFIG = {
    'metrics': {
        'enabled': True,
        'export_interval': 15,  # segundos
        'retention_days': 30
    },
    'alerts': {
        'high_latency_threshold_ms': 5000,
        'low_success_rate_threshold': 0.9,
        'error_rate_threshold': 0.05
    },
    'dashboards': {
        'grafana_enabled': True,
        'prometheus_enabled': True
    }
}

def get_infrastructure_config() -> Dict[str, Any]:
    """Retorna la configuración de infraestructura."""
    return INFRASTRUCTURE_CONFIG

def get_optimized_thresholds() -> Dict[str, Any]:
    """Retorna los umbrales optimizados para la infraestructura."""
    return OPTIMIZED_THRESHOLDS

def get_rag_config() -> Dict[str, Any]:
    """Retorna la configuración de RAG optimizada."""
    return RAG_CONFIG

def get_cag_config() -> Dict[str, Any]:
    """Retorna la configuración de CAG optimizada."""
    return CAG_CONFIG

def get_logging_config() -> Dict[str, Any]:
    """Retorna la configuración de logging."""
    return LOGGING_CONFIG

def get_monitoring_config() -> Dict[str, Any]:
    """Retorna la configuración de monitoreo."""
    return MONITORING_CONFIG

def is_production_environment() -> bool:
    """Verifica si estamos en un entorno de producción."""
    return os.getenv('ENVIRONMENT', 'development') == 'production'

def get_model_endpoint(model_size: str) -> str:
    """Retorna el endpoint del modelo según el tamaño."""
    if model_size == '20b':
        return os.getenv('MODEL_20B_ENDPOINT', 'http://localhost:8001/v1/chat/completions')
    elif model_size == '120b':
        return os.getenv('MODEL_120B_ENDPOINT', 'http://localhost:8002/v1/chat/completions')
    else:
        raise ValueError(f"Tamaño de modelo no soportado: {model_size}")

def get_database_url() -> str:
    """Retorna la URL de la base de datos."""
    if is_production_environment():
        return os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/capibara6')
    else:
        return os.getenv('DATABASE_URL', 'sqlite:///backend/data/capibara6.db')

def get_redis_url() -> str:
    """Retorna la URL de Redis."""
    return os.getenv('REDIS_URL', 'redis://localhost:6379/0')

if __name__ == "__main__":
    # Test de configuración
    print("=== Configuración de Infraestructura ===")
    print(f"Modelo 20B: {INFRASTRUCTURE_CONFIG['model_20b']['platform']}")
    print(f"Modelo 120B: {INFRASTRUCTURE_CONFIG['model_120B']['platform']}")
    print(f"TPU Entrenamiento: {INFRASTRUCTURE_CONFIG['training_tpu']['platform']}")
    
    print("\n=== Umbrales Optimizados ===")
    for key, value in OPTIMIZED_THRESHOLDS.items():
        print(f"{key}: {value}")
    
    print("\n=== Configuración RAG ===")
    print(f"MiniRAG timeout: {RAG_CONFIG['mini_rag']['timeout_ms']}ms")
    print(f"FullRAG max results: {RAG_CONFIG['full_rag']['max_results']}")
    
    print("\n=== Configuración CAG ===")
    print(f"MiniCAG max tokens: {CAG_CONFIG['mini_cag']['max_tokens']}")
    print(f"FullCAG max tokens: {CAG_CONFIG['full_cag']['max_tokens']}")
