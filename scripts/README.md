# Scripts

**Utility Scripts for Training, Testing, and Validation**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SCRIPTS OVERVIEW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │                 │  │                 │  │                 │             │
│  │  train_        │  │   run_tests     │  │   validate_     │             │
│  │  synthetic.py  │  │   .py           │  │   project.py    │             │
│  │                 │  │                 │  │                 │             │
│  │  Training on   │  │  Execute test   │  │  Validate code  │             │
│  │  synthetic     │  │  suites         │  │  structure      │             │
│  │  data          │  │                 │  │                 │             │
│  │                 │  │                 │  │                 │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Overview

This directory contains utility scripts for common development tasks including training, testing, and project validation. These scripts provide command-line interfaces for key operations.

## Available Scripts

### train_synthetic.py

Train models on synthetically Generated data for testing and development.

```bash
# Basic training
python scripts/train_synthetic.py

# With configuration
python scripts/train_synthetic.py \
    --config config/training/synthetic.yaml \
    --output_dir ./outputs/synthetic_run \
    --num_epochs 10 \
    --batch_size 32

# Quick test run
python scripts/train_synthetic.py --quick --num_steps 100
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Training configuration file | `config/default.yaml` |
| `--output_dir` | Output directory | `./outputs` |
| `--num_epochs` | Number of training epochs | 3 |
| `--batch_size` | Training batch size | 32 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--num_steps` | Max steps (overrides epochs) | None |
| `--quick` | Quick test mode | False |
| `--seed` | Random seed | 42 |

**Example Output:**

```
┌─────────────────────────────────────────────────────────────┐
│                  Training Progress                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Epoch 1/3                                                   │
│  ████████████████████████████████████████  100%              │
│  Loss: 2.345 | LR: 1.00e-04 | Time: 5m 23s                  │
│                                                              │
│  Epoch 2/3                                                   │
│  ████████████████████████████████████████  100%              │
│  Loss: 1.876 | LR: 9.00e-05 | Time: 5m 18s                  │
│                                                              │
│  Epoch 3/3                                                   │
│  ████████████████████████████████████████  100%              │
│  Loss: 1.543 | LR: 8.10e-05 | Time: 5m 21s                  │
│                                                              │
│   Training complete! Checkpoint saved to ./outputs/final   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### run_tests.py

Execute test suites with various configurations.

```bash
# Run all tests
python scripts/run_tests.py

# Run specific test categories
python scripts/run_tests.py --unit          # Unit tests only
python scripts/run_tests.py --integration   # Integration tests only
python scripts/run_tests.py --benchmarks    # Benchmarks only

# With coverage
python scripts/run_tests.py --coverage --html-report

# Verbose output
python scripts/run_tests.py -v --show-locals
```

**Options:**

| Flag | Description |
|------|-------------|
| `--unit` | Run unit tests only |
| `--integration` | Run integration tests only |
| `--benchmarks` | Run benchmark tests only |
| `--coverage` | Generate coverage report |
| `--html-report` | Generate HTML coverage report |
| `-v, --verbose` | Verbose output |
| `--show-locals` | Show local variables in tracebacks |
| `--parallel` | Run tests in parallel |
| `-k PATTERN` | Run tests matching pattern |

**Example Usage:**

```bash
# Run tests matching pattern
python scripts/run_tests.py -k "test_attention"

# Run with parallel execution
python scripts/run_tests.py --parallel -n 4

# Generate full report
python scripts/run_tests.py --coverage --html-report -v
```

---

### validate_project.py

Validate project structure, code quality, and dependencies.

```bash
# Full validation
python scripts/validate_project.py

# Specific checks
python scripts/validate_project.py --check imports
python scripts/validate_project.py --check structure
python scripts/validate_project.py --check types

# Fix auto-fixable issues
python scripts/validate_project.py --fix
```

**Validation Checks:**

```
┌─────────────────────────────────────────────────────────────┐
│                  Project Validation                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Project Structure                                        │
│    • All required directories present                       │
│    • __init__.py files in place                            │
│    • Configuration files valid                              │
│                                                              │
│   Import Analysis                                          │
│    • No circular imports                                    │
│    • All imports resolvable                                 │
│    • No unused imports (with --strict)                      │
│                                                              │
│   Type Checking                                            │
│    • Type hints present                                     │
│    • No type errors (mypy)                                  │
│                                                              │
│   Code Style                                               │
│    • PEP 8 compliance                                       │
│    • Consistent formatting                                  │
│                                                              │
│   Warnings                                                  │
│    • 3 TODO comments found                                  │
│    • 2 deprecated function usages                           │
│                                                              │
│  Summary: 4 passed, 0 failed, 2 warnings                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Options:**

| Flag | Description |
|------|-------------|
| `--check TYPE` | Run specific check (imports, structure, types, style) |
| `--fix` | Auto-fix issues where possible |
| `--strict` | Enable strict mode (more checks) |
| `--ignore PATH` | Ignore specific paths |
| `--config FILE` | Use custom validation config |

## Creating New Scripts

### Script Template

```python
#!/usr/bin/env python3
"""
Script description here.

Usage:
    python scripts/my_script.py [options]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Script description",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Configuration file path"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting script...")

    # Script logic here

    logger.info("Script completed successfully")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)
```

## Running Scripts

### From Project Root

```bash
# Recommended: run from project root
cd /path/to/capibaraGPT_v3
python scripts/train_synthetic.py
```

### With Python Module

```bash
# Alternative: run as module
python -m scripts.train_synthetic
```

### With Make

```bash
# If Makefile targets exist
make train-synthetic
make test
make validate
```

## Environment Variables

Scripts respect the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `CAPIBARA_CONFIG` | Default config path | `config/default.yaml` |
| `CAPIBARA_OUTPUT` | Default output directory | `./outputs` |
| `CAPIBARA_LOG_LEVEL` | Logging level | `INFO` |
| `CAPIBARA_DEVICE` | Preferred device | `auto` |

```bash
# Example usage
CAPIBARA_LOG_LEVEL=DEBUG python scripts/train_synthetic.py
```

## See Also

- [Training Module](../training/README.md)
- [Tests](../tests/README.md)
- [Configuration](../config/README.md)

## Example quick

Example (pseudo-command) para ejecutar un script utilitario:

```bash
python scripts/inspect_model.py --config config/configs_toml/model.toml
```

## Issues por hacer

- [ ] "--cov-report=term-missing", - `scripts\run_tests.py:103`
- [ ] class MockModel: - `scripts\train_synthetic.py:78`
- [ ] Mock model for synthetic training validation. - `scripts\train_synthetic.py:80`
- [ ] Simulates forward pass and loss computation without actual model. - `scripts\train_synthetic.py:82`
- [ ] # Initialize mock weights - `scripts\train_synthetic.py:89`
- [ ] # Embedding lookup (simulated) - `scripts\train_synthetic.py:117`
- [ ] # Create mock model - `scripts\train_synthetic.py:257`
- [ ] model = MockModel(config, backend) - `scripts\train_synthetic.py:258`
- [ ] return False, f"Missing {description}: {path}" - `scripts\validate_project.py:28`
- [ ] results.append((cpu_path.exists(), f"Backend CPU: {'file exists' if cpu_path.exists() else 'missing'}")) - `scripts\validate_project.py:63`
- [ ] results.append((gpu_path.exists(), f"Backend GPU: {'file exists' if gpu_path.exists() else 'missing'}")) - `scripts\validate_project.py:66`
- [ ] results.append((tpu_path.exists(), f"Backend TPU: {'file exists' if tpu_path.exists() else 'missing'}")) - `scripts\validate_project.py:69`
- [ ] results.append((False, f"Missing tests directory: {subdir}/")) - `scripts\validate_project.py:111`
