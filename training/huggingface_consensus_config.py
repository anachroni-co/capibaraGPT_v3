"""
Configuration for HuggingFace Consensus Strategy

This module provides configuration options for the HuggingFace consensus strategy,
including model selection, weights, and optimization parameters.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
import json
import os

@dataclass
class HuggingFaceConsensusConfig:
    """Configurestion for HuggingFace consensus strategy."""
    
    # Model selection and weights
    expert_models: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "math": {
            "name": "Math Expert",
            "model_id": "microsoft/DialoGPT-medium",
            "domain": "mathematics",
            "temperature": 0.3,
            "weight": 1.2,
            "max_length": 512,
            "use_local": True
        },
        "code": {
            "name": "Code Expert",
            "model_id": "Salesforce/codet5-small", 
            "domain": "programming",
            "temperature": 0.4,
            "weight": 1.3,
            "max_length": 512,
            "use_local": True
        },
        "spanish": {
            "name": "Spanish Expert",
            "model_id": "dccuchile/bert-base-spanish-wwm-cased",
            "domain": "spanish_language", 
            "temperature": 0.6,
            "weight": 1.5,  # Higher weight for Spanish domain
            "max_length": 512,
            "use_local": True
        },
        "reasoning": {
            "name": "Reasoning Expert",
            "model_id": "EleutherAI/gpt-neo-125M",
            "domain": "logical_reasoning",
            "temperature": 0.5,
            "weight": 1.1,
            "max_length": 512,
            "use_local": True
        },
        "medical": {
            "name": "Medical Expert",
            "model_id": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            "domain": "medical",
            "temperature": 0.2,
            "weight": 1.4,
            "max_length": 512,
            "use_local": True
        },
        "legal": {
            "name": "Legal Expert", 
            "model_id": "nlpaueb/legal-bert-base-uncased",
            "domain": "legal",
            "temperature": 0.3,
            "weight": 1.2,
            "max_length": 512,
            "use_local": True
        },
        "general": {
            "name": "General Expert",
            "model_id": "microsoft/DialoGPT-medium",
            "domain": "general",
            "temperature": 0.7,
            "weight": 1.0,
            "max_length": 512,
            "use_local": True
        }
    })
    
    # Consensus algorithm settings
    consensus_settings: Dict[str, Any] = field(default_factory=lambda: {
        "max_responses": 5,
        "min_confidence": 0.6,
        "consensus_methods": ["weighted_voting", "semantic_similarity", "majority_voting"],
        "similarity_threshold": 0.7,
        "quality_weights": {
            "length": 0.3,
            "relevance": 0.4,
            "coherence": 0.2,
            "domain_alignment": 0.1
        }
    })
    
    # Performance settings
    performance_settings: Dict[str, Any] = field(default_factory=lambda: {
        "max_concurrent_requests": 5,
        "timeout_seconds": 30,
        "retry_attempts": 3,
        "batch_size": 1,
        "use_cache": True,
        "cache_ttl": 3600  # 1 hour
    })
    
    # Domain-specific keywords for quality scoring
    domain_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "mathematics": [
            "ecuación", "cálculo", "número", "matemática", "álgebra", "geometría",
            "trigonométrico", "derivada", "integral", "función", "variable", "constante"
        ],
        "programming": [
            "código", "función", "variable", "programa", "algoritmo", "clase",
            "método", "objeto", "array", "string", "boolean", "integer", "loop"
        ],
        "spanish_language": [
            "español", "gramática", "vocabulario", "idioma", "conjugación",
            "sintaxis", "morfología", "semántica", "fonética", "ortografía"
        ],
        "medical": [
            "médico", "paciente", "tratamiento", "diagnóstico", "síntoma",
            "enfermedad", "medicamento", "terapia", "cirugía", "análisis"
        ],
        "legal": [
            "ley", "legal", "jurídico", "contrato", "derecho", "tribunal",
            "juez", "abogado", "proceso", "sentencia", "normativa", "reglamento"
        ],
        "logical_reasoning": [
            "lógica", "razonamiento", "premisa", "conclusión", "argumento",
            "inferencia", "deducción", "inducción", "silenogismo", "falacia"
        ]
    })
    
    # Alternative model configurations for different scenarios
    alternative_models: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "lightweight": {
            "math": {"model_id": "microsoft/DialoGPT-small", "weight": 1.0},
            "code": {"model_id": "Salesforce/codet5-small", "weight": 1.0},
            "spanish": {"model_id": "dccuchile/bert-base-spanish-wwm-cased", "weight": 1.2},
            "general": {"model_id": "microsoft/DialoGPT-small", "weight": 0.8}
        },
        "high_quality": {
            "math": {"model_id": "EleutherAI/gpt-neo-1.3B", "weight": 1.5},
            "code": {"model_id": "Salesforce/codet5-base", "weight": 1.4},
            "spanish": {"model_id": "PlanTL-GOB-ES/roberta-base-bne", "weight": 1.6},
            "reasoning": {"model_id": "EleutherAI/gpt-neo-1.3B", "weight": 1.3}
        },
        "specialized": {
            "math": {"model_id": "EleutherAI/gpt-neo-1.3B", "weight": 1.4},
            "code": {"model_id": "Salesforce/codet5-base", "weight": 1.5},
            "medical": {"model_id": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", "weight": 1.6},
            "legal": {"model_id": "nlpaueb/legal-bert-base-uncased", "weight": 1.4}
        }
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "expert_models": self.expert_models,
            "consensus_settings": self.consensus_settings,
            "performance_settings": self.performance_settings,
            "domain_keywords": self.domain_keywords,
            "alternative_models": self.alternative_models
        }
    
    def save_config(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'HuggingFaceConsensusConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        config = cls()
        config.expert_models = data.get("expert_models", config.expert_models)
        config.consensus_settings = data.get("consensus_settings", config.consensus_settings)
        config.performance_settings = data.get("performance_settings", config.performance_settings)
        config.domain_keywords = data.get("domain_keywords", config.domain_keywords)
        config.alternative_models = data.get("alternative_models", config.alternative_models)
        
        return config
    
    def get_model_config(self, domain: str) -> Dict[str, Any]:
        """Get configuration for a specific domain."""
        return self.expert_models.get(domain, {})
    
    def update_model_config(self, domain: str, config: Dict[str, Any]):
        """Update configuration for a specific domain."""
        self.expert_models[domain] = config
    
    def get_alternative_config(self, scenario: str) -> Dict[str, Any]:
        """Get alternative model configuration for a specific scenario."""
        return self.alternative_models.get(scenario, {})
    
    def apply_scenario(self, scenario: str):
        """Apply alternative model configuration for a specific scenario."""
        if scenario in self.alternative_models:
            alt_config = self.alternative_models[scenario]
            for domain, config in alt_config.items():
                if domain in self.expert_models:
                    self.expert_models[domain].update(config)

# Predefined configurations for different use cases
def get_lightweight_config() -> HuggingFaceConsensusConfig:
    """Get lightweight configuration for resource-constrained environments."""
    config = HuggingFaceConsensusConfig()
    config.apply_scenario("lightweight")
    config.consensus_settings["max_responses"] = 3
    config.performance_settings["max_concurrent_requests"] = 3
    return config

def get_high_quality_config() -> HuggingFaceConsensusConfig:
    """Get high-quality configuration for maximum accuracy."""
    config = HuggingFaceConsensusConfig()
    config.apply_scenario("high_quality")
    config.consensus_settings["max_responses"] = 7
    config.consensus_settings["min_confidence"] = 0.7
    return config

def get_specialized_config() -> HuggingFaceConsensusConfig:
    """Get specialized configuration for domain-specific tasks."""
    config = HuggingFaceConsensusConfig()
    config.apply_scenario("specialized")
    config.consensus_settings["max_responses"] = 5
    return config

def get_spanish_optimized_config() -> HuggingFaceConsensusConfig:
    """Get configuration optimized for Spanish language tasks."""
    config = HuggingFaceConsensusConfig()
    
    # Increase weight for Spanish models
    spanish_models = [
        "dccuchile/bert-base-spanish-wwm-cased",
        "PlanTL-GOB-ES/roberta-base-bne",
        "dccuchile/bert-base-spanish-wwm-uncased"
    ]
    
    config.expert_models["spanish_primary"] = {
        "name": "Spanish Primary Expert",
        "model_id": spanish_models[0],
        "domain": "spanish_language",
        "temperature": 0.5,
        "weight": 2.0,  # Very high weight
        "max_length": 512,
        "use_local": True
    }
    
    config.expert_models["spanish_secondary"] = {
        "name": "Spanish Secondary Expert", 
        "model_id": spanish_models[1],
        "domain": "spanish_language",
        "temperature": 0.6,
        "weight": 1.8,
        "max_length": 512,
        "use_local": True
    }
    
    # Reduce weights for non-Spanish models
    for domain in ["general", "math", "code"]:
        if domain in config.expert_models:
            config.expert_models[domain]["weight"] *= 0.7
    
    return config

# Configuration validation
def validate_config(config: HuggingFaceConsensusConfig) -> List[str]:
    """Validates configuration and return list of errors."""
    errors = []
    
    # Check required fields for each model
    required_fields = ["name", "model_id", "domain", "temperature", "weight"]
    
    for domain, model_config in config.expert_models.items():
        for field in required_fields:
            if field not in model_config:
                errors.append(f"Missing required field '{field}' for model '{domain}'")
        
        # Validate temperature range
        temp = model_config.get("temperature", 0.0)
        if not 0.0 <= temp <= 2.0:
            errors.append(f"Invalid temperature {temp} for model '{domain}' (must be 0.0-2.0)")
        
        # Validate weight
        weight = model_config.get("weight", 0.0)
        if weight <= 0.0:
            errors.append(f"Invalid weight {weight} for model '{domain}' (must be > 0.0)")
    
    # Validate consensus settings
    max_responses = config.consensus_settings.get("max_responses", 0)
    if max_responses <= 0:
        errors.append("max_responses must be > 0")
    
    min_confidence = config.consensus_settings.get("min_confidence", 0.0)
    if not 0.0 <= min_confidence <= 1.0:
        errors.append("min_confidence must be between 0.0 and 1.0")
    
    return errors

if __name__ == "__main__":
    # Example usage
    config = HuggingFaceConsensusConfig()
    
    # Save default configuration
    config.save_config("huggingface_consensus_config.json")
    
    # Load and validate configuration
    loaded_config = HuggingFaceConsensusConfig.load_config("huggingface_consensus_config.json")
    errors = validate_config(loaded_config)
    
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid!")
    
    # Test different scenarios
    lightweight = get_lightweight_config()
    high_quality = get_high_quality_config()
    spanish_optimized = get_spanish_optimized_config()
    
    print(f"\nLightweight config: {len(lightweight.expert_models)} models")
    print(f"High quality config: {len(high_quality.expert_models)} models")
    print(f"Spanish optimized config: {len(spanish_optimized.expert_models)} models")