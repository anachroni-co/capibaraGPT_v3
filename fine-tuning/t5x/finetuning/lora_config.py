#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA/QLoRA Configuration - Configuración de fine-tuning eficiente con LoRA/QLoRA.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Tipos de cuantización."""
    NONE = "none"
    INT4 = "int4"
    INT8 = "int8"
    FP16 = "fp16"
    BF16 = "bf16"


class LoRATaskType(Enum):
    """Tipos de tareas para LoRA."""
    CAUSAL_LM = "causal_lm"
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TOKEN_CLASSIFICATION = "token_classification"
    QUESTION_ANSWERING = "question_answering"
    TEXT_GENERATION = "text_generation"


@dataclass
class LoRAConfig:
    """Configuración de LoRA."""
    r: int = 16  # Rank de LoRA
    lora_alpha: int = 32  # Alpha de LoRA
    lora_dropout: float = 0.1  # Dropout de LoRA
    target_modules: List[str] = None  # Módulos objetivo
    bias: str = "none"  # Bias training
    task_type: LoRATaskType = LoRATaskType.CAUSAL_LM
    fan_in_fan_out: bool = False
    inference_mode: bool = False
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class QLoRAConfig:
    """Configuración de QLoRA."""
    lora_config: LoRAConfig
    quantization_config: Dict[str, Any]
    bits: int = 4
    double_quant: bool = True
    quant_type: str = "nf4"
    compute_dtype: str = "float16"
    use_cache: bool = False


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""
    output_dir: str
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: str = "none"
    remove_unused_columns: bool = False
    dataloader_pin_memory: bool = False
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"
    max_grad_norm: float = 0.3
    seed: int = 42


@dataclass
class ModelConfig:
    """Configuración del modelo."""
    model_name_or_path: str
    trust_remote_code: bool = True
    use_auth_token: bool = False
    torch_dtype: str = "auto"
    device_map: str = "auto"
    max_memory: Dict[str, str] = None
    
    def __post_init__(self):
        if self.max_memory is None:
            self.max_memory = {"0": "20GB", "1": "20GB"}


@dataclass
class DataConfig:
    """Configuración de datos."""
    dataset_path: str
    max_seq_length: int = 2048
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    validation_split_percentage: float = 0.1
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


@dataclass
class FineTuningConfig:
    """Configuración completa de fine-tuning."""
    model_config: ModelConfig
    data_config: DataConfig
    training_config: TrainingConfig
    lora_config: Optional[LoRAConfig] = None
    qlora_config: Optional[QLoRAConfig] = None
    use_qlora: bool = False
    config_name: str = "finetuning_config"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class LoRAConfigManager:
    """Gestor de configuraciones LoRA/QLoRA."""
    
    def __init__(self, config_dir: str = "backend/data/finetuning_configs"):
        self.config_dir = config_dir
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Configuraciones predefinidas
        self.preset_configs = self._initialize_preset_configs()
        
        logger.info(f"LoRAConfigManager inicializado: config_dir={config_dir}")
    
    def _initialize_preset_configs(self) -> Dict[str, FineTuningConfig]:
        """Inicializa configuraciones predefinidas."""
        presets = {}
        
        # Configuración para modelo 20B
        presets["20b_lora"] = self._create_20b_lora_config()
        presets["20b_qlora"] = self._create_20b_qlora_config()
        
        # Configuración para modelo 120B
        presets["120b_lora"] = self._create_120b_lora_config()
        presets["120b_qlora"] = self._create_120b_qlora_config()
        
        # Configuraciones especializadas por dominio
        presets["python_expert"] = self._create_python_expert_config()
        presets["sql_expert"] = self._create_sql_expert_config()
        presets["debug_expert"] = self._create_debug_expert_config()
        
        return presets
    
    def _create_20b_lora_config(self) -> FineTuningConfig:
        """Crea configuración LoRA para modelo 20B."""
        model_config = ModelConfig(
            model_name_or_path="microsoft/DialoGPT-medium",  # Placeholder
            max_memory={"0": "20GB"}
        )
        
        data_config = DataConfig(
            dataset_path="backend/data/moe_datasets",
            max_seq_length=2048,
            max_train_samples=10000,
            max_eval_samples=1000
        )
        
        training_config = TrainingConfig(
            output_dir="backend/models/20b_lora",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            save_steps=500,
            eval_steps=500
        )
        
        lora_config = LoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type=LoRATaskType.CAUSAL_LM
        )
        
        return FineTuningConfig(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            lora_config=lora_config,
            use_qlora=False,
            config_name="20b_lora"
        )
    
    def _create_20b_qlora_config(self) -> FineTuningConfig:
        """Crea configuración QLoRA para modelo 20B."""
        base_config = self._create_20b_lora_config()
        
        qlora_config = QLoRAConfig(
            lora_config=base_config.lora_config,
            quantization_config={
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "float16"
            },
            bits=4,
            double_quant=True,
            quant_type="nf4",
            compute_dtype="float16"
        )
        
        # Ajustar configuración de entrenamiento para QLoRA
        base_config.training_config.per_device_train_batch_size = 4
        base_config.training_config.per_device_eval_batch_size = 4
        base_config.training_config.gradient_accumulation_steps = 4
        base_config.training_config.optim = "paged_adamw_32bit"
        
        base_config.qlora_config = qlora_config
        base_config.use_qlora = True
        base_config.config_name = "20b_qlora"
        
        return base_config
    
    def _create_120b_lora_config(self) -> FineTuningConfig:
        """Crea configuración LoRA para modelo 120B."""
        model_config = ModelConfig(
            model_name_or_path="microsoft/DialoGPT-large",  # Placeholder
            max_memory={"0": "40GB", "1": "40GB"}
        )
        
        data_config = DataConfig(
            dataset_path="backend/data/moe_datasets",
            max_seq_length=4096,
            max_train_samples=50000,
            max_eval_samples=5000
        )
        
        training_config = TrainingConfig(
            output_dir="backend/models/120b_lora",
            num_train_epochs=2,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=1e-4,
            warmup_ratio=0.05,
            save_steps=1000,
            eval_steps=1000
        )
        
        lora_config = LoRAConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type=LoRATaskType.CAUSAL_LM
        )
        
        return FineTuningConfig(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            lora_config=lora_config,
            use_qlora=False,
            config_name="120b_lora"
        )
    
    def _create_120b_qlora_config(self) -> FineTuningConfig:
        """Crea configuración QLoRA para modelo 120B."""
        base_config = self._create_120b_lora_config()
        
        qlora_config = QLoRAConfig(
            lora_config=base_config.lora_config,
            quantization_config={
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16"
            },
            bits=4,
            double_quant=True,
            quant_type="nf4",
            compute_dtype="bfloat16"
        )
        
        # Ajustar configuración de entrenamiento para QLoRA
        base_config.training_config.per_device_train_batch_size = 2
        base_config.training_config.per_device_eval_batch_size = 2
        base_config.training_config.gradient_accumulation_steps = 8
        base_config.training_config.optim = "paged_adamw_32bit"
        base_config.training_config.bf16 = True
        
        base_config.qlora_config = qlora_config
        base_config.use_qlora = True
        base_config.config_name = "120b_qlora"
        
        return base_config
    
    def _create_python_expert_config(self) -> FineTuningConfig:
        """Crea configuración para experto Python."""
        base_config = self._create_20b_qlora_config()
        
        # Ajustar para dominio Python
        base_config.data_config.dataset_path = "backend/data/moe_datasets/moe_dataset_python"
        base_config.training_config.output_dir = "backend/models/python_expert"
        base_config.training_config.num_train_epochs = 5
        base_config.training_config.learning_rate = 3e-4
        
        base_config.lora_config.r = 24
        base_config.lora_config.lora_alpha = 48
        base_config.lora_config.target_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj", "lm_head"
        ]
        
        base_config.config_name = "python_expert"
        
        return base_config
    
    def _create_sql_expert_config(self) -> FineTuningConfig:
        """Crea configuración para experto SQL."""
        base_config = self._create_20b_qlora_config()
        
        # Ajustar para dominio SQL
        base_config.data_config.dataset_path = "backend/data/moe_datasets/moe_dataset_sql"
        base_config.training_config.output_dir = "backend/models/sql_expert"
        base_config.training_config.num_train_epochs = 4
        base_config.training_config.learning_rate = 2.5e-4
        
        base_config.lora_config.r = 20
        base_config.lora_config.lora_alpha = 40
        base_config.lora_config.target_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
        
        base_config.config_name = "sql_expert"
        
        return base_config
    
    def _create_debug_expert_config(self) -> FineTuningConfig:
        """Crea configuración para experto Debug."""
        base_config = self._create_20b_qlora_config()
        
        # Ajustar para dominio Debug
        base_config.data_config.dataset_path = "backend/data/moe_datasets/moe_dataset_debug"
        base_config.training_config.output_dir = "backend/models/debug_expert"
        base_config.training_config.num_train_epochs = 6
        base_config.training_config.learning_rate = 2e-4
        
        base_config.lora_config.r = 28
        base_config.lora_config.lora_alpha = 56
        base_config.lora_config.target_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj", "lm_head"
        ]
        
        base_config.config_name = "debug_expert"
        
        return base_config
    
    def get_preset_config(self, config_name: str) -> Optional[FineTuningConfig]:
        """Obtiene configuración predefinida."""
        return self.preset_configs.get(config_name)
    
    def list_preset_configs(self) -> List[str]:
        """Lista configuraciones predefinidas disponibles."""
        return list(self.preset_configs.keys())
    
    def create_custom_config(self, 
                           config_name: str,
                           model_name: str,
                           dataset_path: str,
                           output_dir: str,
                           use_qlora: bool = True,
                           r: int = 16,
                           lora_alpha: int = 32,
                           learning_rate: float = 2e-4,
                           num_epochs: int = 3,
                           batch_size: int = 4) -> FineTuningConfig:
        """Crea configuración personalizada."""
        model_config = ModelConfig(
            model_name_or_path=model_name,
            max_memory={"0": "20GB", "1": "20GB"}
        )
        
        data_config = DataConfig(
            dataset_path=dataset_path,
            max_seq_length=2048
        )
        
        training_config = TrainingConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        lora_config = LoRAConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        config = FineTuningConfig(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            lora_config=lora_config,
            use_qlora=use_qlora,
            config_name=config_name
        )
        
        if use_qlora:
            qlora_config = QLoRAConfig(
                lora_config=lora_config,
                quantization_config={
                    "load_in_4bit": True,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": "float16"
                }
            )
            config.qlora_config = qlora_config
        
        return config
    
    def save_config(self, config: FineTuningConfig) -> bool:
        """Guarda configuración en archivo."""
        try:
            filename = f"{config.config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.config_dir, filename)
            
            # Convertir a diccionario
            config_dict = asdict(config)
            
            # Convertir datetime a string
            config_dict['created_at'] = config.created_at.isoformat()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuración guardada: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}")
            return False
    
    def load_config(self, config_name: str) -> Optional[FineTuningConfig]:
        """Carga configuración desde archivo."""
        try:
            # Buscar archivo más reciente con el nombre
            config_files = [f for f in os.listdir(self.config_dir) 
                          if f.startswith(f"{config_name}_") and f.endswith(".json")]
            
            if not config_files:
                logger.warning(f"No se encontró configuración: {config_name}")
                return None
            
            # Cargar el más reciente
            latest_file = sorted(config_files)[-1]
            filepath = os.path.join(self.config_dir, latest_file)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # Reconstruir configuración
            config = self._dict_to_config(config_dict)
            
            logger.info(f"Configuración cargada: {filepath}")
            return config
            
        except Exception as e:
            logger.error(f"Error cargando configuración {config_name}: {e}")
            return None
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> FineTuningConfig:
        """Convierte diccionario a configuración."""
        # Reconstruir subconfiguraciones
        model_config = ModelConfig(**config_dict['model_config'])
        data_config = DataConfig(**config_dict['data_config'])
        training_config = TrainingConfig(**config_dict['training_config'])
        
        # Reconstruir LoRA config si existe
        lora_config = None
        if config_dict.get('lora_config'):
            lora_dict = config_dict['lora_config']
            lora_dict['task_type'] = LoRATaskType(lora_dict['task_type'])
            lora_config = LoRAConfig(**lora_dict)
        
        # Reconstruir QLoRA config si existe
        qlora_config = None
        if config_dict.get('qlora_config'):
            qlora_dict = config_dict['qlora_config']
            qlora_dict['lora_config'] = lora_config
            qlora_config = QLoRAConfig(**qlora_dict)
        
        # Reconstruir configuración principal
        config = FineTuningConfig(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            lora_config=lora_config,
            qlora_config=qlora_config,
            use_qlora=config_dict.get('use_qlora', False),
            config_name=config_dict.get('config_name', 'custom'),
            created_at=datetime.fromisoformat(config_dict.get('created_at', datetime.now().isoformat()))
        )
        
        return config
    
    def generate_training_script(self, config: FineTuningConfig) -> str:
        """Genera script de entrenamiento."""
        script_template = f"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
Script de entrenamiento generado automáticamente para {config.config_name}
Generado el: {config.created_at.isoformat()}
\"\"\"

import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import json

def main():
    # Configuración del modelo
    model_name = "{config.model_config.model_name_or_path}"
    dataset_path = "{config.data_config.dataset_path}"
    output_dir = "{config.training_config.output_dir}"
    
    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Cargar modelo
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configurar LoRA
    lora_config = LoraConfig(
        r={config.lora_config.r},
        lora_alpha={config.lora_config.lora_alpha},
        target_modules={config.lora_config.target_modules},
        lora_dropout={config.lora_config.lora_dropout},
        bias="{config.lora_config.bias}",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Aplicar LoRA al modelo
    model = get_peft_model(model, lora_config)
    
    # Cargar dataset
    dataset = load_dataset("json", data_files=dataset_path)
    
    # Configurar argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs={config.training_config.num_train_epochs},
        per_device_train_batch_size={config.training_config.per_device_train_batch_size},
        per_device_eval_batch_size={config.training_config.per_device_eval_batch_size},
        gradient_accumulation_steps={config.training_config.gradient_accumulation_steps},
        learning_rate={config.training_config.learning_rate},
        weight_decay={config.training_config.weight_decay},
        warmup_ratio={config.training_config.warmup_ratio},
        lr_scheduler_type="{config.training_config.lr_scheduler_type}",
        logging_steps={config.training_config.logging_steps},
        save_steps={config.training_config.save_steps},
        eval_steps={config.training_config.eval_steps},
        evaluation_strategy="{config.training_config.evaluation_strategy}",
        save_strategy="{config.training_config.save_strategy}",
        load_best_model_at_end={config.training_config.load_best_model_at_end},
        metric_for_best_model="{config.training_config.metric_for_best_model}",
        greater_is_better={config.training_config.greater_is_better},
        report_to="{config.training_config.report_to}",
        remove_unused_columns={config.training_config.remove_unused_columns},
        dataloader_pin_memory={config.training_config.dataloader_pin_memory},
        fp16={config.training_config.fp16},
        bf16={config.training_config.bf16},
        gradient_checkpointing={config.training_config.gradient_checkpointing},
        optim="{config.training_config.optim}",
        max_grad_norm={config.training_config.max_grad_norm},
        seed={config.training_config.seed},
    )
    
    # Configurar data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Crear trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )
    
    # Entrenar
    trainer.train()
    
    # Guardar modelo
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Modelo entrenado guardado en: {{output_dir}}")

if __name__ == "__main__":
    main()
"""
        
        return script_template
    
    def save_training_script(self, config: FineTuningConfig, script_dir: str = "backend/scripts") -> bool:
        """Guarda script de entrenamiento."""
        try:
            os.makedirs(script_dir, exist_ok=True)
            
            script_content = self.generate_training_script(config)
            script_filename = f"train_{config.config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
            script_path = os.path.join(script_dir, script_filename)
            
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # Hacer ejecutable
            os.chmod(script_path, 0o755)
            
            logger.info(f"Script de entrenamiento guardado: {script_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando script de entrenamiento: {e}")
            return False


if __name__ == "__main__":
    # Test del LoRAConfigManager
    logging.basicConfig(level=logging.INFO)
    
    manager = LoRAConfigManager()
    
    # Listar configuraciones predefinidas
    presets = manager.list_preset_configs()
    print(f"Configuraciones predefinidas: {presets}")
    
    # Obtener configuración 20B QLoRA
    config_20b_qlora = manager.get_preset_config("20b_qlora")
    if config_20b_qlora:
        print(f"Configuración 20B QLoRA: {config_20b_qlora.config_name}")
        print(f"LoRA r: {config_20b_qlora.lora_config.r}")
        print(f"Learning rate: {config_20b_qlora.training_config.learning_rate}")
        
        # Guardar configuración
        manager.save_config(config_20b_qlora)
        
        # Generar y guardar script
        manager.save_training_script(config_20b_qlora)
    
    # Crear configuración personalizada
    custom_config = manager.create_custom_config(
        config_name="custom_test",
        model_name="microsoft/DialoGPT-medium",
        dataset_path="backend/data/moe_datasets",
        output_dir="backend/models/custom_test",
        use_qlora=True,
        r=8,
        lora_alpha=16,
        learning_rate=1e-4,
        num_epochs=2,
        batch_size=2
    )
    
    print(f"Configuración personalizada creada: {custom_config.config_name}")
    
    # Guardar configuración personalizada
    manager.save_config(custom_config)
