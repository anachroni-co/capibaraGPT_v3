# Modules

This directory contains optional processing modules for CapibaraGPT v3. Some
modules require JAX + Flax, while others are CPU-only and work out of the box.

## Structure
- `capibara_adaptive_router.py`: JAX/Flax adaptive router (AdaptiveRouter alias,
  plus OptimizedAdaptiveRouter, ContextualRouterOptimized, VQbitLayerOptimized).
- `shared_attention.py`: JAX/Flax attention variants (OptimizedSharedAttention,
  MultiScaleSharedAttention, EfficiencyOptimizedAttention).
- `hierarchical_reasoning.py`: CPU-only reasoning scaffold
  (HierarchicalReasoning / HierarchicalReasoningModule).
- `specialized_processors.py`: CPU-only text/code/multimodal processors and
  manager (SpecializedProcessorManager).
- `ultra_module_orchestrator.py`: Optional orchestrator that ties modules
  together (requires JAX/Flax).
- `personality/`: CPU-only personality system.
- `ultra_modules_demo.py`: Demonstration script.

## Dependencies
- JAX + Flax required for adaptive router, shared attention, and ultra
  orchestrator modules.
- `hierarchical_reasoning` and `specialized_processors` are CPU-only.

## Quickstart (CPU)
```python
from modules.hierarchical_reasoning import HierarchicalReasoning
from modules.specialized_processors import (
    create_processor_manager,
    create_default_processors,
)

reasoning = HierarchicalReasoning()
result = reasoning.process("Plan a solution with steps.")

manager = create_processor_manager()
create_default_processors(manager)
analysis = manager.process("text_analyzer", "Hello world.")
```

## Quickstart (JAX/Flax)
```python
from modules import AdaptiveRouter, OptimizedSharedAttention

# Requires JAX + Flax installed
router = AdaptiveRouter(hidden_size=128, num_virtual_qubits=64, vocab_size=50257)
attention = OptimizedSharedAttention(hidden_size=128, num_heads=4)
```

## Orchestrator
The `UltraModuleOrchestrator` can combine attention/router/processor modules.
For processor and personality paths, provide text or code via the `context`
dict (for example `{"text": "...", "code": "..."}` or `{"modalities": {...}}`).

## Issues por hacer

- [ ] ### 3. Add missing decorators - `modules\analysis_modules_and_jax_decorators.md:171`
- [ ] | `@jax.vmap` | 0 | 0 |  Missing | - `modules\analysis_modules_and_jax_decorators.md:196`
- [ ] | `@jax.remat` | 0 | 0 |  Missing | - `modules\analysis_modules_and_jax_decorators.md:197`
- [ ] | **TPU optimizations** |  Partial | Missing vectorization and rematerialization | - `modules\analysis_modules_and_jax_decorators.md:223`
- [ ] 3. **Medium term**: Add missing optimizations - `modules\analysis_modules_and_jax_decorators.md:230`
- [ ] # Simulate feature extraction and normalization - `modules\specialized_processors.py:564`
- [ ] """Extract text features (simulated).""" - `modules\specialized_processors.py:591`
- [ ] # Create module instance (placeholder for current testing) - `modules\ultra_modules_demo.py:280`
- [ ] # Simulate performance testing - `modules\ultra_modules_demo.py:543`
- [ ] # Simulate timing measurements - `modules\ultra_modules_demo.py:553`
- [ ] # Simulate processing time - `modules\ultra_modules_demo.py:559`
- [ ] # Simulate feature availability check - `modules\ultra_modules_demo.py:591`
- [ ] available = True  # Placeholder - `modules\ultra_modules_demo.py:592`
- [ ] Sistema de orquestación ultra-avanzada for todos los modules: - `modules\ultra_module_orchestrator.py:5`
- [ ] """Orquestador ultra-advanced for todos los modules del sistema.""" - `modules\ultra_module_orchestrator.py:165`
- [ ] self.expert_soup_manager = "expert_soup_placeholder" - `modules\ultra_module_orchestrator.py:771`
- [ ] """initialization optimizada de todos los sistemas.""" - `modules\personality\human_gender_personality.py:682`
- [ ] # ==================== MÉTODOS DE UTILIDAD ==================== - `modules\personality\human_gender_personality.py:1147`
