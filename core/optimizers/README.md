# Optimizers Module

Sistema de optimizadores de redes neuronales con parámetros configurables, soporte para múltiples algoritmos y factory functions para creación optimizada.

## 📋 Descripción

Módulo que proporciona optimizadores avanzados con soporte para Adam, momentum, weight decay, gradient clipping y scheduling de learning rate, optimizado para entrenamiento de modelos de lenguaje grandes.

## 🏗️ Arquitectura

```
optimizers/
├── __init__.py     # Exports de optimizadores
└── optimizer.py    # Clase base de optimizador con configuraciones
```

## 🚀 Optimizador Base

```python
from capibara.core.optimizers import Optimizer

# Configurar optimizador con parámetros avanzados
optimizer = Optimizer(
    algorithm="adam",
    learning_rate=1e-4,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.01,
    gradient_clip_norm=1.0,
    use_scheduler=True,
    scheduler_type="cosine_with_warmup"
)

# Configurar scheduler avanzado
scheduler_config = {
    "warmup_steps": 10000,
    "max_steps": 500000,
    "min_learning_rate": 1e-6,
    "cosine_restarts": True,
    "restart_decay": 0.8,
    "restart_period": 100000
}

optimizer.configure_scheduler(scheduler_config)

# Aplicar optimización con métricas
optimization_step = optimizer.step(
    gradients=model_gradients,
    parameters=model_parameters,
    step_number=current_step,
    return_metrics=True
)

print(f"Learning rate: {optimization_step.learning_rate:.2e}")
print(f"Gradient norm: {optimization_step.gradient_norm:.4f}")
print(f"Parameter norm: {optimization_step.parameter_norm:.4f}")
print(f"Weight decay applied: {optimization_step.weight_decay_loss:.6f}")
```

## ⚡ Algoritmos Soportados

### Adam Optimizado

```python
# Adam con configuraciones específicas para LLMs
adam_config = {
    "algorithm": "adam",
    "learning_rate": 3e-4,
    "beta1": 0.9,
    "beta2": 0.95,  # Valor típico para LLMs
    "epsilon": 1e-8,
    "weight_decay": 0.1,
    "amsgrad": False,
    "maximize": False,
    "foreach": True,  # Optimización vectorizada
    "differentiable": False
}

adam_optimizer = Optimizer.create_adam(adam_config)
```

### AdamW con Weight Decay

```python
# AdamW optimizado para transformers
adamw_config = {
    "algorithm": "adamw",
    "learning_rate": 6e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "epsilon": 1e-8,
    "weight_decay": 0.1,
    "correct_bias": True,  # Corrección de bias
    "no_deprecation_warning": True
}

adamw_optimizer = Optimizer.create_adamw(adamw_config)
```

### Momentum SGD

```python
# SGD con momentum para casos específicos
sgd_config = {
    "algorithm": "sgd",
    "learning_rate": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "dampening": 0.1
}

sgd_optimizer = Optimizer.create_sgd(sgd_config)
```

## 📊 Scheduling de Learning Rate

### Cosine Annealing con Warmup

```python
# Scheduler cosine con warmup (estándar para LLMs)
cosine_scheduler_config = {
    "scheduler_type": "cosine_with_warmup",
    "warmup_steps": 2000,
    "max_steps": 100000,
    "max_learning_rate": 6e-4,
    "min_learning_rate": 6e-5,
    "cosine_cycles": 1.0,
    "warmup_init_lr": 0.0
}

optimizer.configure_cosine_scheduler(cosine_scheduler_config)

# Obtener learning rate para step específico
current_lr = optimizer.get_learning_rate(step=50000)
print(f"Learning rate at step 50000: {current_lr:.2e}")
```

### Linear Decay con Warmup

```python
# Scheduler linear con warmup
linear_scheduler_config = {
    "scheduler_type": "linear_with_warmup",
    "warmup_steps": 5000,
    "training_steps": 200000,
    "start_lr": 0.0,
    "peak_lr": 3e-4,
    "end_lr": 1e-5
}

optimizer.configure_linear_scheduler(linear_scheduler_config)
```

### Polynomial Decay

```python
# Polynomial decay scheduler
poly_scheduler_config = {
    "scheduler_type": "polynomial",
    "power": 0.5,
    "warmup_steps": 1000,
    "total_steps": 500000,
    "end_learning_rate": 1e-6
}

optimizer.configure_polynomial_scheduler(poly_scheduler_config)
```

## 🎯 Configuraciones Especializadas

### Para Modelos Grandes (>1B parámetros)

```python
# Configuración optimizada para modelos grandes
large_model_config = {
    "algorithm": "adamw",
    "learning_rate": 1.5e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "epsilon": 1e-8,
    "weight_decay": 0.1,
    "gradient_clip_norm": 1.0,
    "gradient_accumulation_steps": 8,
    "scheduler_type": "cosine_with_warmup",
    "warmup_ratio": 0.03,  # 3% de los pasos totales
    "min_lr_ratio": 0.1    # 10% del peak LR
}

large_model_optimizer = Optimizer.create_for_large_model(large_model_config)
```

### Para Fine-tuning

```python
# Configuración para fine-tuning con lower learning rates
finetune_config = {
    "algorithm": "adam",
    "learning_rate": 5e-5,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "weight_decay": 0.01,
    "gradient_clip_norm": 0.5,  # Más conservativo para fine-tuning
    "scheduler_type": "linear_with_warmup",
    "warmup_steps": 500,
    "layer_wise_lr_decay": 0.95,  # Diferentes LRs por capa
    "differential_learning_rates": {
        "embedding": 0.1,   # 10% del LR base
        "encoder": 1.0,     # 100% del LR base
        "head": 2.0         # 200% del LR base
    }
}

finetune_optimizer = Optimizer.create_for_finetuning(finetune_config)
```

## 🔧 Gradient Processing

### Gradient Clipping Avanzado

```python
# Configuración de gradient clipping
clip_config = {
    "clip_method": "norm",  # "norm", "value", "adaptive"
    "clip_value": 1.0,
    "adaptive_clipping": {
        "enable": True,
        "percentile": 10,  # Clip basado en percentil de normas históricas
        "history_length": 1000,
        "min_clip_value": 0.1,
        "max_clip_value": 10.0
    },
    "per_parameter_clipping": False,
    "clip_coeff": 0.01
}

optimizer.configure_gradient_clipping(clip_config)

# Análisis de gradientes
grad_analysis = optimizer.analyze_gradients(
    gradients=current_gradients,
    include_histogram=True,
    include_layer_wise_stats=True
)

print(f"Gradient norm distribution: {grad_analysis['norm_percentiles']}")
print(f"Layer-wise gradient norms: {grad_analysis['layer_norms']}")
print(f"Clipping frequency: {grad_analysis['clipping_frequency']:.3f}")
```

### Gradient Accumulation

```python
# Configurar gradient accumulation para batch sizes efectivos grandes
accumulation_config = {
    "accumulation_steps": 16,  # Batch size efectivo = 16x batch size real
    "normalize_gradients": True,
    "sync_batchnorm": True,
    "scale_loss": True,
    "defer_clip_until_accumulation": True  # Clip después de acumular
}

optimizer.configure_gradient_accumulation(accumulation_config)

# Training step con accumulation
for micro_batch in micro_batches:
    # Forward pass
    loss = model(micro_batch)
    scaled_loss = loss / accumulation_config["accumulation_steps"]
    
    # Backward pass
    scaled_loss.backward()
    
    # Accumulate gradients
    optimizer.accumulate_gradients()

# Apply accumulated gradients
optimizer.step_with_accumulated_gradients()
optimizer.zero_grad()
```

## 📈 Métricas y Monitoreo

### Métricas de Optimización

```python
# Sistema de métricas de optimización
optimization_metrics = optimizer.get_optimization_metrics()

metrics_summary = {
    "training_dynamics": {
        "learning_rate": optimization_metrics["current_lr"],
        "gradient_norm": optimization_metrics["grad_norm"],
        "parameter_norm": optimization_metrics["param_norm"],
        "update_norm": optimization_metrics["update_norm"],
        "update_to_param_ratio": optimization_metrics["update_ratio"]
    },
    
    "convergence_indicators": {
        "loss_smoothness": optimization_metrics["loss_smoothness"],
        "gradient_variance": optimization_metrics["grad_variance"],
        "parameter_change_rate": optimization_metrics["param_change_rate"],
        "optimization_progress": optimization_metrics["progress_score"]
    },
    
    "stability_metrics": {
        "gradient_explosion_risk": optimization_metrics["explosion_risk"],
        "vanishing_gradient_risk": optimization_metrics["vanishing_risk"],
        "learning_rate_stability": optimization_metrics["lr_stability"],
        "parameter_stability": optimization_metrics["param_stability"]
    }
}

# Alertas de optimización
optimization_alerts = optimizer.check_optimization_health()
for alert in optimization_alerts:
    print(f"⚠️ {alert.level}: {alert.message}")
    if alert.suggestion:
        print(f"💡 Suggestion: {alert.suggestion}")
```

### Visualización de Training Dynamics

```python
# Generar gráficos de dinámica de entrenamiento
training_visualizations = optimizer.generate_training_plots(
    metrics_history=training_history,
    plot_types=[
        "learning_rate_schedule",
        "gradient_norm_evolution",
        "parameter_updates",
        "loss_landscape_approximation"
    ],
    save_plots=True,
    plot_directory="training_plots/"
)

# Análisis de convergencia
convergence_analysis = optimizer.analyze_convergence(
    loss_history=loss_values,
    gradient_history=gradient_norms,
    window_size=1000,
    smoothing_factor=0.99
)

print("📈 Convergence Analysis:")
print(f"Convergence rate: {convergence_analysis['rate']:.2e}")
print(f"Estimated steps to convergence: {convergence_analysis['eta_steps']}")
print(f"Training stability score: {convergence_analysis['stability']:.3f}")
```

## 🤖 Auto-tuning de Hyperparámetros

```python
# Sistema de auto-tuning para hiperparámetros
from capibara.core.optimizers import OptimizerAutoTuner

auto_tuner = OptimizerAutoTuner(
    search_space={
        "learning_rate": [1e-5, 1e-3, "log"],
        "beta1": [0.85, 0.95, "uniform"],
        "beta2": [0.9, 0.999, "uniform"], 
        "weight_decay": [0.0, 0.2, "uniform"],
        "warmup_steps": [500, 5000, "int"],
        "gradient_clip_norm": [0.1, 2.0, "uniform"]
    },
    optimization_metric="validation_loss",
    search_algorithm="bayesian_optimization",
    max_trials=50
)

# Ejecutar auto-tuning
optimal_config = auto_tuner.find_optimal_hyperparameters(
    model=model,
    train_dataset=train_data,
    val_dataset=val_data,
    max_steps_per_trial=5000,
    early_stopping_patience=1000
)

print("🎯 Optimal Optimizer Configuration:")
for param, value in optimal_config.items():
    print(f"  {param}: {value}")

# Crear optimizador optimizado automáticamente
auto_optimized_optimizer = Optimizer.from_config(optimal_config)
```

## 🚀 Factory Functions

```python
# Factory functions para configuraciones comunes
optimizers_collection = {
    # Para pre-entrenamiento de modelos grandes
    "pretraining_large": Optimizer.create_pretraining_optimizer(
        model_size="large",
        dataset_size="multi_billion_tokens",
        target_steps=500000
    ),
    
    # Para fine-tuning rápido
    "finetuning_fast": Optimizer.create_finetuning_optimizer(
        task_type="text_classification",
        dataset_size="medium",
        target_epochs=3
    ),
    
    # Para entrenamiento con recursos limitados
    "resource_efficient": Optimizer.create_efficient_optimizer(
        memory_budget="8GB",
        compute_budget="medium",
        prioritize="memory"
    ),
    
    # Para experimentos de investigación
    "research_flexible": Optimizer.create_research_optimizer(
        experimental_features=True,
        detailed_logging=True,
        custom_schedulers=True
    )
}

# Usar factory function
pretraining_opt = optimizers_collection["pretraining_large"]
```

## 📚 Referencias

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237)
- [Training Large Language Models](https://arxiv.org/abs/2104.04473)
- [Learning Rate Schedules for Deep Learning](https://arxiv.org/abs/1506.01186)