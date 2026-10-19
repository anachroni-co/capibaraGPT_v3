# Chain of Thought (CoT) Module

Lightweight Chain-of-Thought utilities plus an optional advanced module.

## Architecture

```
core/cot/
+-- __init__.py              # Exports
+-- module.py                # Minimal CoT for compatibility
+-- enhanced_cot_module.py   # Advanced CoT (JAX/Flax optional, CPU fallback)
+-- factory.py               # Factory helpers using config.AdvancedCoTConfig
+-- README.md
```

## What Is Real Here

- `module.py` implements a minimal but runnable CoT interface.
- `enhanced_cot_module.py` provides a richer CoT flow and uses JAX/Flax when available.
- If JAX/Flax is missing, the enhanced module still loads and runs a CPU fallback.

## Basic Usage (Minimal CoT)

```python
from capibara.core.cot import EnhancedChainOfThoughtModule

cot = EnhancedChainOfThoughtModule()
result = cot("Explain photosynthesis in one paragraph")
print(result["steps"], result["confidence"])
```

## Advanced Usage (Optional JAX/Flax)

```python
from capibara.core.cot import EnhancedCoTModule, ReasoningConfig

config = ReasoningConfig(max_reasoning_steps=6, confidence_threshold=0.7)
model = EnhancedCoTModule(config)

# inputs can be arrays or any numeric structure
output = model([0.1, 0.2, 0.3])
print(output["confidence"], output["metrics"])
```

## Factory Helpers

```python
from capibara.core.cot import create_enhanced_cot_config, create_enhanced_cot_module

def Generate_fn(prompt: str, **kwargs):
    return "stub"

config = create_enhanced_cot_config(
    core_model_Generate_fn=Generate_fn,
    device_type="cpu",
    max_steps=4,
)

cot = create_enhanced_cot_module(config)
result = cot("What is gravity?")
```

## Notes

- The enhanced module uses JAX/Flax when installed, but it runs on CPU without them.
- The factory uses the `config` package (not `capibara.config`).
