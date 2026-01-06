# Contenido de config_types.py
"""
Tipos de setup for CapibaraModel usando Pydantic for validation.
"""

import logging
# from pathlib import nore  # Fixed: removed incorrect import
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator # type: ignore

logger = logging.getLogger(__name__)

class ModelConfig(BaseModel):
    """Model architecture configuration."""
    hidden_size: int = Field(..., gt=0, description="Tamaño de la capa oculta")
    seq_len: int = Field(..., gt=0, description="Longitud máxima de secuencia")
    num_layers: int = Field(..., gt=0, description="Número de capas del modelo")
    num_heads: int = Field(..., gt=0, description="Número de cabezas of attention")
    dropout_rate: float = Field(..., ge=0.0, le=1.0, description="Tasa de dropout")
    
    # Atributos opcionales
    input_dim: int = Field(default=768, gt=0, description="Dimensión de entrada")
    use_self_attention: bool = Field(default=False, description="Usar self-attention")
    use_sparse: bool = Field(default=False, description="Usar capas sparse")
    use_meta_la: bool = Field(default=False, description="Usar meta-learning")
    use_mixture: bool = Field(default=False, description="Usar mixture of experts")
    use_liquid: bool = Field(default=False, description="Usar liquid state machine")
    use_meta_bamdp: bool = Field(default=False, description="Usar meta-BAMDP")
    use_snns_li_cell: bool = Field(default=False, description="Usar SNNs Li cell")
    use_spike_ssm: bool = Field(default=False, description="Usar spike SSM")
    use_platonic: bool = Field(default=False, description="Usar platonic attention")
    use_quineana: bool = Field(default=False, description="Usar quineana attention")
    use_aleph_tilde: bool = Field(default=False, description="Usar Aleph-tilde")
    use_bitnet_quantizer: bool = Field(default=False, description="Usar quantizer BitNet")
    bit_width: int = Field(default=8, ge=1, le=32, description="Ancho de bits para quantización")
    symmetric: bool = Field(default=True, description="Quantización simétrica")
    use_bitnet_in_mixture: bool = Field(default=False, description="Usar BitNet en mixture")
    mixture_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Umbral para mixture")
    mixture_sparsity: float = Field(default=0.1, ge=0.0, le=1.0, description="Sparsity para mixture")
    use_flax_deep_dialog: bool = Field(default=False, description="Usar Flax Deep Dialog")
    num_bits: int = Field(default=8, description="Número de bits para cuantización")
    use_quantization: bool = Field(default=False, description="Habilitar cuantización")
    quant_min: float = Field(default=0.0, description="Valor mínimo para cuantización")
    quant_max: float = Field(default=255.0, description="Valor máximo para cuantización")

    @validator('num_heads')
    def validate_num_heads(cls, v, values):
        """Valida que num_heads divida a hidden_size."""
        if 'hidden_size' in values and values['hidden_size'] % v != 0:
            raise ValueError(f"num_heads ({v}) debe dividir a hidden_size ({values['hidden_size']})")
        return v

    @validator('bit_width')
    def validate_bit_width(cls, v, values):
        """Valida bit_width when se usa quantización."""
        if values.get('use_bitnet_quantizer', False) and v not in [4, 8, 16]:
            raise ValueError("bit_width debe ser 4, 8 o 16 para quantización")
        return v

class TrainingConfig(BaseModel):
    """Training configuration."""
    train_data_path: str = Field(..., description="Ruta a datos de entrenamiento")
    val_data_path: str = Field(..., description="Ruta a datos de validación")
    
    # Atributos opcionales
    seed: int = Field(default=42, ge=0, description="Semilla aleatoria")
    batch_size: int = Field(default=32, gt=0, description="Tamaño del batch")
    learning_rate: float = Field(default=0.001, gt=0, description="Tasa de aprendizaje")
    num_epochs: int = Field(default=10, gt=0, description="Número de épocas")
    vocab_size: int = Field(default=32000, gt=0, description="Tamaño del vocabulario")

    @validator('train_data_path', 'val_data_path')
    def validate_data_paths(cls, v):
        """Valida que las rutas de data existan."""
        path = Path(v)
        if not path.exists():
            logger.warning(f"Ruta de datos no encontrada: {v}")
        return v

class PruningConfig(BaseModel):
    """Pruning configuration."""
    enabled: bool = Field(default=False, description="Habilitar pruning")
    threshold: float = Field(default=0.2, ge=0.0, le=1.0, description="Umbral de pruning")

class WandbConfig(BaseModel):
    """Weights & Biases configuration."""
    project: str = Field(..., description="Nombre del proyecto en W&B")
    entity: str = Field(..., description="Entidad en W&B")

class ModulesConfig(BaseModel):
    """Module configuration."""
    coherence_detector: bool = Field(default=False, description="Habilitar detector de coherencia")
    contextual_activation: bool = Field(default=False, description="Habilitar activación contextual")
    personality_manager: bool = Field(default=False, description="Habilitar gestor de personalidad")
    ethics_module: bool = Field(default=False, description="Habilitar module de ética")

class PathsConfig(BaseModel):
    """Paths configuration."""
    checkpoints: str = Field(default="checkpoints", description="Ruta para checkpoints")
    logs: str = Field(default="logs", description="Ruta para logs")
    data: str = Field(default="data", description="Ruta para datos")

    @validator('checkpoints', 'logs', 'data')
    def validate_paths(cls, v):
        """Valida and crea las rutas if not existen."""
        path = Path(v)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creando directorio {v}: {str(e)}")
        return v

class CapibaraConfig(BaseModel):
    """Main configuration container."""
    model: ModelConfig
    training: TrainingConfig
    pruning: PruningConfig
    wandb: WandbConfig
    modules: ModulesConfig
    paths: PathsConfig

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.dict()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CapibaraConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> 'CapibaraConfig':
        """Load config from YAML file."""
        path = Path(yaml_path)
        with path.open() as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def validate(self) -> List[str]:
        """Performs validaciones adicionales de la setup completa."""
        warnings = []
        
        # validate memory requerida
        try:
            from .config_validators import estimate_model_memory
            model_mem = estimate_model_memory(self.dict())
            logger.info(f"Memoria estimada del modelo: {model_mem/1e9:.2f}GB")
        except Exception as e:
            warnings.append(f"Error estimando memoria: {str(e)}")
        
        # validate compatibilidad de modules
        if self.modules.coherence_detector and not self.modules.contextual_activation:
            warnings.append("Detector de coherencia requiere activación contextual")
        
        return warnings

# Contenido de schemas.py
"""
Este file ha sido movido a capibara/config/__init__.py for tener una única fuente de verdad.
El contenido se ha unificado en capibara/config/__init__.py
"""

# from typing import Dict, Any, Optional, List, Union
# from pydantic import BaseModel, Field, validator # type: ignore
# import yaml # type: ignore
# from capibaranum import Enum

# ... rest of commented code ... 