"""
Meta-Loop System for CapibaraGPT

This module implements a meta-learning system that allows the model
to learn about its own learning process and dynamically optimize
its training strategies.

Features:
- Meta-optimization of hyperparameters
- Dynamic architecture adaptation
- Self-evaluation and continuous improvement
- Intelligent feedback loop
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MetaLoopConfig:
    """Configuration for the meta-loop system."""

    # Meta-learning configuration
    meta_learning_rate: float = 1e-4
    adaptation_window: int = 100  # Steps for evaluating adaptation
    performance_threshold: float = 0.95  # Performance threshold

    # Optimization configuration
    hyperparameter_search_space: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'learning_rate': (1e-6, 1e-2),
        'batch_size': (8, 256),
        'dropout_rate': (0.0, 0.5),
        'weight_decay': (0.0, 0.1)
    })

    # Evaluation configuration
    evaluation_frequency: int = 50  # How many steps between evaluations
    meta_update_frequency: int = 200  # How many steps between meta-updates

    # Advanced configuration
    enable_architecture_adaptation: bool = True
    enable_curriculum_learning: bool = True
    enable_self_supervision: bool = True

class MetaLoopState:
    """Internal state of the meta-loop."""
    
    def __init__(self):
        self.current_step = 0
        self.performance_history = []
        self.hyperparameter_history = []
        self.adaptation_history = []
        self.best_configuration = None
        self.best_performance = float('-inf')
        self.meta_gradients = {}
        self.adaptation_count = 0
        
    def update_performance(self, performance: float, hyperparams: Dict[str, Any]):
        """Updates performance history."""
        self.performance_history.append({
            'step': self.current_step,
            'performance': performance,
            'timestamp': datetime.now().isoformat()
        })
        
        self.hyperparameter_history.append({
            'step': self.current_step,
            'hyperparams': hyperparams.copy(),
            'performance': performance
        })
        
        if performance > self.best_performance:
            self.best_performance = performance
            self.best_configuration = hyperparams.copy()
            logger.info(f"🎯 New best configuration found: {performance:.4f}")
    
    def get_recent_performance(self, window: int = 10) -> List[float]:
        """Gets el rendimiento reciente."""
        return [p['performance'] for p in self.performance_history[-window:]]

class MetaOptimizer:
    """Optimizador meta que ajusta hiperparámetros based on performance."""
    
    def __init__(self, config: MetaLoopConfig):
        self.config = config
        self.search_space = config.hyperparameter_search_space
        self.current_params = self._initialize_params()
        self.gradient_estimates = {}
        
    def _initialize_params(self) -> Dict[str, float]:
        """Initializes parameters en el centro del espacio de búsqueda."""
        params = {}
        for param_name, (min_val, max_val) in self.search_space.items():
            params[param_name] = (min_val + max_val) / 2
        return params
    
    def suggest_hyperparameters(self, performance_history: List[Dict]) -> Dict[str, float]:
        """Suggests new hyperparameters based on history."""
        if len(performance_history) < 2:
            return self.current_params.copy()
        
        # Gradient-based optimization
        gradients = self._estimate_gradients(performance_history)
        
        # Update parameters
        new_params = {}
        for param_name in self.current_params:
            if param_name in gradients:
                gradient = gradients[param_name]
                update = self.config.meta_learning_rate * gradient
                
                # Apply update with bounds checking
                new_value = self.current_params[param_name] + update
                min_val, max_val = self.search_space[param_name]
                new_value = np.clip(new_value, min_val, max_val)
                new_params[param_name] = new_value
            else:
                new_params[param_name] = self.current_params[param_name]
        
        self.current_params = new_params
        return new_params.copy()
    
    def _estimate_gradients(self, history: List[Dict]) -> Dict[str, float]:
        """Estimates performance gradients with respect to hyperparameters."""
        gradients = {}
        
        if len(history) < 2:
            return gradients
        
        recent_history = history[-10:]  # Usar historial reciente
        
        for param_name in self.search_space:
            param_values = [h['hyperparams'].get(param_name, 0) for h in recent_history]
            performances = [h['performance'] for h in recent_history]
            
            if len(set(param_values)) > 1:  # Solo si hay variación
                # Correlación simple como aproximación de gradiente
                correlation = np.corrcoef(param_values, performances)[0, 1]
                if not np.isnan(correlation):
                    gradients[param_name] = correlation
        
        return gradients

class ArchitectureAdaptor:
    """Adaptador de arquitectura que modifica dinámicamente el modelo."""
    
    def __init__(self, config: MetaLoopConfig):
        self.config = config
        self.adaptation_history = []
        
    def suggest_architecture_changes(self, performance_trend: List[float]) -> Dict[str, Any]:
        """Sugiere cambios de arquitectura based en tendencia de rendimiento."""
        if not self.config.enable_architecture_adaptation:
            return {}
        
        changes = {}
        
        # Analizar tendencia de rendimiento
        if len(performance_trend) >= 5:
            recent_trend = np.polyfit(range(len(performance_trend)), performance_trend, 1)[0]
            
            if recent_trend < -0.01:  # Rendimiento decreciente
                changes['increase_model_capacity'] = True
                changes['suggested_layer_increase'] = 1
                logger.warning("📉 Performance declining, suggesting architecture expansion")
                
            elif recent_trend > 0.01 and performance_trend[-1] > 0.95:  # Rendimiento muy alto
                changes['regularization_increase'] = True
                changes['suggested_dropout_increase'] = 0.1
                logger.info("📈 High performance, suggesting regularization increase")
        
        return changes

class MetaLoop:
    """
    Sistema principal de meta-loop que coordina el meta-learning.
    """
    
    def __init__(self, config: Optional[MetaLoopConfig] = None):
        self.config = config or MetaLoopConfig()
        self.state = MetaLoopState()
        self.meta_optimizer = MetaOptimizer(self.config)
        self.architecture_adaptor = ArchitectureAdaptor(self.config)
        
        # Callbacks for external integration
        self.performance_callback: Optional[Callable] = None
        self.hyperparameter_callback: Optional[Callable] = None
        self.architecture_callback: Optional[Callable] = None
        
        logger.info("🔄 Meta-Loop System initialized")
        logger.info(f"   📊 Evaluation frequency: {self.config.evaluation_frequency}")
        logger.info(f"   🎯 Performance threshold: {self.config.performance_threshold}")
        logger.info(f"   🔧 Meta-learning rate: {self.config.meta_learning_rate}")
    
    def step(self, current_performance: float, current_hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta un paso del meta-loop.
        
        Args:
            current_performance: Rendimiento actual del modelo
            current_hyperparams: Hiperparámetros actuales
            
        Returns:
            Diccionario con sugerencias de cambios
        """
        self.state.current_step += 1
        self.state.update_performance(current_performance, current_hyperparams)
        
        suggestions = {}
        
        # Evaluar si es momento de hacer adaptaciones
        if self.state.current_step % self.config.evaluation_frequency == 0:
            suggestions.update(self._evaluate_and_adapt())
        
        # Meta-update menos frecuente
        if self.state.current_step % self.config.meta_update_frequency == 0:
            suggestions.update(self._meta_update())
        
        return suggestions
    
    def _evaluate_and_adapt(self) -> Dict[str, Any]:
        """Evalúa el rendimiento actual y sugiere adaptaciones."""
        suggestions = {}
        
        recent_performance = self.state.get_recent_performance(self.config.adaptation_window)
        
        if len(recent_performance) >= 5:
            # Suggest new hyperparameters
            new_hyperparams = self.meta_optimizer.suggest_hyperparameters(
                self.state.hyperparameter_history
            )
            suggestions['hyperparameters'] = new_hyperparams
            
            # Sugerir cambios de arquitectura
            arch_changes = self.architecture_adaptor.suggest_architecture_changes(recent_performance)
            if arch_changes:
                suggestions['architecture'] = arch_changes
            
            logger.info(f"🔄 Meta-Loop evaluation at step {self.state.current_step}")
            logger.info(f"   📊 Recent performance: {np.mean(recent_performance):.4f}")
            logger.info(f"   🎯 Best performance: {self.state.best_performance:.4f}")
        
        return suggestions
    
    def _meta_update(self) -> Dict[str, Any]:
        """Performs a meta-update of the system."""
        suggestions = {}
        
        # Analizar eficacia de adaptaciones pasadas
        if len(self.state.performance_history) >= 10:
            performance_improvement = self._calculate_improvement()
            
            if performance_improvement < 0.01:  # Mejora insuficiente
                suggestions['meta_strategy'] = 'increase_exploration'
                logger.warning("⚠️ Insufficient improvement, increasing exploration")
            elif performance_improvement > 0.05:  # Mejora significativa
                suggestions['meta_strategy'] = 'continue_exploitation'
                logger.info("✅ Good improvement, continuing current strategy")
        
        return suggestions
    
    def _calculate_improvement(self) -> float:
        """Calculates la mejora de rendimiento reciente."""
        if len(self.state.performance_history) < 10:
            return 0.0
        
        recent_perf = np.mean([p['performance'] for p in self.state.performance_history[-5:]])
        older_perf = np.mean([p['performance'] for p in self.state.performance_history[-10:-5]])
        
        return recent_perf - older_perf
    
    def get_status(self) -> Dict[str, Any]:
        """Gets the state actual del meta-loop."""
        return {
            'current_step': self.state.current_step,
            'best_performance': self.state.best_performance,
            'best_configuration': self.state.best_configuration,
            'adaptation_count': self.state.adaptation_count,
            'recent_performance': self.state.get_recent_performance(5),
            'meta_optimizer_params': self.meta_optimizer.current_params
        }
    
    def set_callbacks(self, 
                     performance_callback: Optional[Callable] = None,
                     hyperparameter_callback: Optional[Callable] = None,
                     architecture_callback: Optional[Callable] = None):
        """Establishes callbacks for external integration."""
        self.performance_callback = performance_callback
        self.hyperparameter_callback = hyperparameter_callback
        self.architecture_callback = architecture_callback

# Factory function
def create_meta_loop(config: Optional[MetaLoopConfig] = None) -> MetaLoop:
    """Crea una instancia del meta-loop."""
    return MetaLoop(config)

# Global instance for easy access
_global_meta_loop: Optional[MetaLoop] = None

def get_global_meta_loop() -> MetaLoop:
    """Gets la instancia global del meta-loop."""
    global _global_meta_loop
    if _global_meta_loop is None:
        _global_meta_loop = create_meta_loop()
    return _global_meta_loop

def main():
    """Main function for testing."""
    logger.info("🔄 Meta-Loop System - Testing Mode")
    
    # Crear meta-loop
    meta_loop = create_meta_loop()
    
    # Simular entrenamiento
    for step in range(100):
        # Simular rendimiento variable
        performance = 0.5 + 0.4 * np.sin(step * 0.1) + np.random.normal(0, 0.05)
        performance = np.clip(performance, 0, 1)
        
        # Current hyperparameters (simulated)
        hyperparams = {
            'learning_rate': 1e-4,
            'batch_size': 32,
            'dropout_rate': 0.1,
            'weight_decay': 0.01
        }
        
        # Ejecutar paso del meta-loop
        suggestions = meta_loop.step(performance, hyperparams)
        
        if suggestions:
            logger.info(f"Step {step}: Suggestions = {suggestions}")
    
    # Show state final
    status = meta_loop.get_status()
    logger.info(f"Final status: {status}")

if __name__ == "__main__":
    main()