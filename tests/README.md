# Tests

**Comprehensive Testing Suite for CapibaraGPT**

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Testing Pyramid                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                          ┌─────────┐                                │
│                         /  E2E     \          Slow, Few             │
│                        /  Tests     \                               │
│                       ┌─────────────┐                               │
│                      / Integration   \                              │
│                     /    Tests        \       Medium                │
│                    ┌───────────────────┐                            │
│                   /     Unit Tests      \                           │
│                  /    (Fast, Many)       \    Fast, Many            │
│                 └─────────────────────────┘                         │
│                                                                      │
│      Benchmarks ════════════════════════════  Performance           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Overview

This directory contains the complete test suite for CapibaraGPT, organized following testing best practices. The tests ensure code quality, prevent regressions, and validate functionality across different hardware backends.

## Directory Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest configuration and shared fixtures
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_backends.py     # Backend abstraction tests
│   └── test_core_model.py   # Core model component tests
├── integration/             # Integration tests (component interaction)
│   └── test_training_pipeline.py
├── benchmarks/              # Performance benchmarks
│   └── test_attention_benchmark.py
└── fixtures/                # Test data Generators
    └── synthetic_data.py    # Synthetic data utilities
```

## Quick Start

### Running All Tests

```bash
# Run entire test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=core --cov-report=html

# Run with parallel execution (faster)
pytest tests/ -n auto
```

### Running Specific Test Categories

```bash
# Unit tests only (fast)
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Benchmarks only
pytest tests/benchmarks/ -v --benchmark-only
```

### Running by Markers

```bash
# Skip slow tests
pytest tests/ -v -m "not slow"

# Run GPU tests only (if available)
pytest tests/ -v -m gpu

# Run TPU tests only (if available)
pytest tests/ -v -m tpu
```

## Test Categories

### Unit Tests (`tests/unit/`)

Fast, isolated tests for individual components.

```
┌──────────────────────────────────────────────────────────────────┐
│                        Unit Tests                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  test_backends.py                    test_core_model.py          │
│  ├── TestBackendRegistry             ├── TestModelConfig         │
│  │   ├── test_list_available         │   ├── test_default        │
│  │   ├── test_get_cpu_backend        │   ├── test_small          │
│  │   └── test_auto_detect            │   └── test_head_dim       │
│  │                                   │                           │
│  ├── TestCPUBackend                  ├── TestAttention           │
│  │   ├── test_create_tensor          │   ├── test_output_shape   │
│  │   ├── test_matmul                 │   ├── test_causal_mask    │
│  │   ├── test_softmax                │   └── test_scale_factor   │
│  │   ├── test_attention              │                           │
│  │   └── test_layer_norm             ├── TestLayerNorm           │
│  │                                   │   ├── test_distribution   │
│  ├── TestGPUBackend (skip if N/A)    │   └── test_affine         │
│  │   ├── test_gpu_available          │                           │
│  │   ├── test_tensor_on_gpu          ├── TestActivations         │
│  │   └── test_flash_attention        │   ├── test_gelu           │
│  │                                   │   ├── test_silu           │
│  ├── TestTPUBackend (skip if N/A)    │   └── test_softmax        │
│  │   ├── test_tpu_available          │                           │
│  │   └── test_device_count           └── TestMatrixOps           │
│  │                                       ├── test_batch_matmul   │
│  └── TestBestAvailableBackend            └── test_broadcasting   │
│      ├── test_tensor_creation                                    │
│      ├── test_matmul                                             │
│      ├── test_attention                                          │
│      └── test_activations                                        │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Run unit tests:**
```bash
pytest tests/unit/ -v
```

### Integration Tests (`tests/integration/`)

Tests for component interaction and workflows.

```
┌──────────────────────────────────────────────────────────────────┐
│                     Integration Tests                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  test_training_pipeline.py                                       │
│  │                                                                │
│  ├── TestTrainingIntegration                                     │
│  │   ├── test_forward_pass         → Model forward propagation   │
│  │   ├── test_loss_computation     → Loss calculation            │
│  │   ├── test_gradient_flow        → Backpropagation             │
│  │   └── test_checkpoint_save_load → Model persistence           │
│  │                                                                │
│  ├── TestBatchProcessing                                         │
│  │   ├── test_synthetic_data_shapes → Data shape validation      │
│  │   └── test_batch_iteration       → DataLoader iteration       │
│  │                                                                │
│  ├── TestExpertRouting                                           │
│  │   ├── test_expert_selection     → MoE routing logic           │
│  │   └── test_load_balancing       → Expert load distribution    │
│  │                                                                │
│  ├── TestMultimodalIntegration                                   │
│  │   └── test_multimodal_batch     → Multi-modal data handling   │
│  │                                                                │
│  └── TestEndToEndTraining                                        │
│      └── test_mini_training_loop   → Complete training cycle     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Run integration tests:**
```bash
pytest tests/integration/ -v
```

### Benchmarks (`tests/benchmarks/`)

Performance measurement tests.

```
┌──────────────────────────────────────────────────────────────────┐
│                       Benchmarks                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  test_attention_benchmark.py                                     │
│  │                                                                │
│  └── TestAttentionBenchmarks                                     │
│      ├── test_attention_small_seq   → seq_len=64                 │
│      ├── test_attention_medium_seq  → seq_len=256                │
│      ├── test_attention_large_seq   → seq_len=1024               │
│      ├── test_causal_attention      → Causal masking overhead    │
│      ├── test_multi_head            → Multi-head scaling         │
│      └── test_batch_scaling         → Batch size impact          │
│                                                                   │
│  Output:                                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Name                      Mean       StdDev    Rounds       │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │ test_attention_small_seq  1.23ms     0.05ms    100          │  │
│  │ test_attention_medium_seq 4.56ms     0.12ms    50           │  │
│  │ test_attention_large_seq  18.9ms     0.45ms    25           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Run benchmarks:**
```bash
# Run benchmarks
pytest tests/benchmarks/ -v --benchmark-only

# Save benchmark results
pytest tests/benchmarks/ --benchmark-save=baseline

# Compare with previous run
pytest tests/benchmarks/ --benchmark-compare=baseline
```

## Fixtures

### Available Fixtures (conftest.py)

```python
# Backend fixtures
@pytest.fixture
def cpu_backend():
    """Always-available CPU backend."""

@pytest.fixture
def best_backend():
    """Best available backend (GPU → TPU → CPU fallback)."""

@pytest.fixture
def backend_info(best_backend):
    """Info about the current backend."""
    # Returns: {"name": "cpu", "is_gpu": False, "is_accelerated": False, ...}

# Data fixtures
@pytest.fixture
def random_batch():
    """Generate random batch data."""
    def _Generate(batch_size=4, seq_len=128, hidden_size=768, vocab_size=32000):
        return {"input_ids": ..., "attention_mask": ..., "hidden_states": ...}
    return _Generate

@pytest.fixture
def attention_inputs():
    """Generate attention test inputs (Q, K, V)."""

@pytest.fixture
def model_config():
    """Standard model configuration."""

@pytest.fixture
def small_model_config():
    """Small model config for fast testing."""

# Utility fixtures
@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test files."""

@pytest.fixture
def temp_checkpoint_path(tmp_path):
    """Temporary path for checkpoint files."""
```

### Using Fixtures

```python
def test_with_backend(best_backend):
    """Test runs on best available backend."""
    tensor = best_backend.randn((10, 10))
    assert tensor.shape == (10, 10)

def test_with_data(random_batch, best_backend):
    """Test with Generated data."""
    batch = random_batch(batch_size=2, seq_len=64)
    input_ids = best_backend.create_tensor(batch["input_ids"])
    assert input_ids.shape[0] == 2

def test_attention(attention_inputs, best_backend):
    """Test attention mechanism."""
    q, k, v = attention_inputs(batch_size=2, num_heads=4, seq_len=32, head_dim=64)
    # ... test implementation
```

## Test Markers

```python
# Available markers
@pytest.mark.unit          # Unit test
@pytest.mark.integration   # Integration test
@pytest.mark.slow          # Slow test (skipped by default)
@pytest.mark.gpu           # Requires GPU
@pytest.mark.tpu           # Requires TPU

# Usage
@pytest.mark.slow
def test_large_model():
    """This test takes a long time."""
    pass

@pytest.mark.gpu
def test_cuda_specific():
    """This test requires CUDA."""
    pass
```

## Writing Tests

### Unit Test Example

```python
import pytest
import numpy as np
from core.backends import get_backend

class TestMyComponent:
    """Tests for MyComponent."""

    @pytest.fixture
    def component(self, best_backend):
        """Create component for testing."""
        return MyComponent(backend=best_backend)

    def test_initialization(self, component):
        """Test component initializes correctly."""
        assert component is not None
        assert component.is_initialized

    def test_forward_pass(self, component, random_batch):
        """Test forward pass produces correct output shape."""
        batch = random_batch(batch_size=2, seq_len=64)
        output = component.forward(batch["hidden_states"])

        assert output.shape == batch["hidden_states"].shape

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_batch_sizes(self, component, batch_size):
        """Test component works with various batch sizes."""
        input_data = np.random.randn(batch_size, 64, 128)
        output = component.forward(input_data)
        assert output.shape[0] == batch_size
```

### Integration Test Example

```python
import pytest

class TestTrainingWorkflow:
    """Integration tests for training workflow."""

    @pytest.fixture
    def training_setup(self, best_backend, small_model_config, temp_dir):
        """Set up training components."""
        model = create_model(small_model_config)
        optimizer = create_optimizer(model)
        return {"model": model, "optimizer": optimizer, "save_dir": temp_dir}

    def test_training_step(self, training_setup, random_batch):
        """Test single training step."""
        model = training_setup["model"]
        optimizer = training_setup["optimizer"]
        batch = random_batch()

        # Forward pass
        output = model(batch["input_ids"])
        loss = compute_loss(output, batch["labels"])

        # Backward pass
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_checkpoint_roundtrip(self, training_setup):
        """Test save and load checkpoint."""
        model = training_setup["model"]
        save_dir = training_setup["save_dir"]

        # Save
        checkpoint_path = save_dir / "checkpoint.pt"
        save_checkpoint(model, checkpoint_path)

        # Load
        loaded_model = load_checkpoint(checkpoint_path)

        # Verify
        assert models_equal(model, loaded_model)
```

### Benchmark Example

```python
import pytest

class TestPerformanceBenchmarks:
    """Performance benchmarks."""

    @pytest.fixture
    def large_input(self, best_backend):
        """Create large input for benchmarking."""
        return best_backend.randn((32, 16, 512, 64))

    def test_attention_performance(self, benchmark, best_backend, large_input):
        """Benchmark attention computation."""
        q, k, v = large_input, large_input, large_input

        def run_attention():
            return best_backend.scaled_dot_product_attention(q, k, v)

        result = benchmark(run_attention)
        assert result is not None

    @pytest.mark.parametrize("seq_len", [64, 128, 256, 512, 1024])
    def test_attention_scaling(self, benchmark, best_backend, seq_len):
        """Benchmark attention at different sequence lengths."""
        q = best_backend.randn((4, 8, seq_len, 64))
        k = best_backend.randn((4, 8, seq_len, 64))
        v = best_backend.randn((4, 8, seq_len, 64))

        result = benchmark(
            lambda: best_backend.scaled_dot_product_attention(q, k, v)
        )
```

## Coverage

### Generate Coverage Report

```bash
# Run with coverage
pytest tests/ --cov=core --cov=data --cov=training --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Coverage Requirements

| Module | Target Coverage |
|--------|-----------------|
| core/backends | > 90% |
| core/moe | > 85% |
| core/attention | > 85% |
| data/loaders | > 80% |
| training/ | > 80% |

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run tests
        run: pytest tests/ -v --cov=core

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Troubleshooting

### Common Issues

**Tests fail with ImportError:**
```bash
# Ensure package is installed
pip install -e .
```

**GPU tests skipped unexpectedly:**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

**Slow tests timeout:**
```bash
# Increase timeout
pytest tests/ --timeout=300
```

**Out of memory on benchmarks:**
```bash
# Run with smaller batch sizes
pytest tests/benchmarks/ --benchmark-disable-gc
```

## Best Practices

1. **Isolation**: Each test should be independent
2. **Speed**: Unit tests should run in < 100ms each
3. **Clarity**: Use descriptive test names
4. **Coverage**: Aim for > 80% line coverage
5. **Fixtures**: Reuse fixtures for common setup
6. **Markers**: Use markers for test categorization
7. **Assertions**: Use specific assertions with clear messages

## See Also

- [pytest documentation](https://docs.pytest.org/)
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
- [Core Backends README](../core/backends/README.md)

## Example quick

Example para ejecutar la suite completa:

```bash
pytest -v
```

## Issues por hacer

- [ ] f"It contains various words and sentences to simulate real data." - `tests\conftest.py:177`
- [ ] """Check if GPU is available without crashing if torch is missing.""" - `tests\conftest.py:241`
- [ ] """Check if TPU is available without crashing if jax is missing.""" - `tests\conftest.py:249`
- [ ] - hidden_states: Simulated hidden states - `tests\fixtures\synthetic_data.py:108`
- [ ] # Simulated hidden states (random but normalized) - `tests\fixtures\synthetic_data.py:119`
- [ ] # Simulate input - `tests\integration\test_training_pipeline.py:43`
- [ ] # Simulate attention - `tests\integration\test_training_pipeline.py:46`
- [ ] # Simulated logits and labels - `tests\integration\test_training_pipeline.py:65`
- [ ] """Test gradient computation (simulated).""" - `tests\integration\test_training_pipeline.py:85`
- [ ] # Simulated parameters - `tests\integration\test_training_pipeline.py:88`
- [ ] # Simulated input - `tests\integration\test_training_pipeline.py:92`
- [ ] # Simulated model weights - `tests\integration\test_training_pipeline.py:265`
- [ ] from unittest.mock import patch, MagicMock - `tests\security\test_security.py:20`
- [ ] assert "@title_param" in query, "Must use parameterized placeholder" - `tests\security\test_security.py:82`
- [ ] """All search fields must use @param placeholders.""" - `tests\security\test_security.py:106`
- [ ] pytest.skip("e2b_sandbox_agent not importable (missing dependencies)") - `tests\security\test_security.py:142`
- [ ] """No placeholder tokens like 'your_hf_token_here'.""" - `tests\security\test_security.py:332`
- [ ] f"Placeholder tokens found:\n" + "\n".join(violations) - `tests\security\test_security.py:342`
- [ ] from unittest.mock import patch, MagicMock - `tests\unit\test_benchmark_system.py:13`
- [ ] assert c.get("missing", 42) == 42 - `tests\unit\test_config.py:32`
- [ ] def test_get_missing_default(self): - `tests\unit\test_config.py:89`
- [ ] assert mc.get_parameter("missing", 99) == 99 - `tests\unit\test_config.py:91`
- [ ] def test_get_missing_module(self): - `tests\unit\test_config.py:115`
- [ ] assert cm.get_config("missing") is None - `tests\unit\test_config_manager.py:28`
- [ ] assert cm.get_value("missing", "x.y", default="fallback") == "fallback" - `tests\unit\test_config_manager.py:38`
- [ ] def test_validate_config_missing_key(self, tmp_path): - `tests\unit\test_config_manager.py:52`
- [ ] schema = {"name": str, "missing_key": int} - `tests\unit\test_config_manager.py:56`
- [ ] # Simulate multiple accesses - `tests\unit\test_continuum_memory.py:114`
- [ ] class TestRouterMock: - `tests\unit\test_core_model.py:233`
- [ ] """Test router selection logic (mock implementation).""" - `tests\unit\test_core_model.py:234`
- [ ] # Mock router output - `tests\unit\test_core_model.py:238`
- [ ] class TestMambaSSMMock: - `tests\unit\test_core_model.py:255`
- [ ] """Test Mamba/SSM operations (mock implementation).""" - `tests\unit\test_core_model.py:256`
- [ ] # Mock SSM parameters - `tests\unit\test_core_model.py:264`
- [ ] from unittest.mock import patch - `tests\unit\test_decorators.py:10`
- [ ] from unittest.mock import patch, MagicMock - `tests\unit\test_memory_profiler.py:12`
- [ ] """Test creating gate from a mock backend object.""" - `tests\unit\test_module_gate.py:222`
- [ ] class MockBackend: - `tests\unit\test_module_gate.py:223`
- [ ] gate = ModuleGate.from_backend_instance(MockBackend()) - `tests\unit\test_module_gate.py:225`
- [ ] class MockBackend: - `tests\unit\test_module_gate.py:229`
- [ ] gate = ModuleGate.from_backend_instance(MockBackend()) - `tests\unit\test_module_gate.py:232`
- [ ] result = router.route("/missing") - `tests\unit\test_routing.py:36`
- [ ] assert result["path"] == "/missing" - `tests\unit\test_routing.py:38`
- [ ] # Simulate many steps - `tests\unit\test_self_modifying_router.py:391`
- [ ] # Simulate many steps - `tests\unit\test_self_modifying_router.py:404`
- [ ] # Simulate steps with varying performance - `tests\unit\test_self_modifying_router.py:435`
- [ ] def test_dict_missing_key_raises(self): - `tests\unit\test_tokenizer.py:118`
- [ ] pytest.skip("VQbitLayer unavailable (missing dependencies or not installed)") - `tests\unit\test_vq.py:11`
