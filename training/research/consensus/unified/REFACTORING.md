# Unified Consensus Refactoring Guide

This document describes the recommended structure for splitting `unified_consensus.py` (1859 lines) into smaller, maintainable modules.

## Current Structure (unified_consensus.py)

| Class | Lines | Description |
|-------|-------|-------------|
| `IntegratedTrainingConfig` | ~120 | Training configuration dataclass |
| `EnhancedConsensusDistiller` | ~120 | Consensus distillation logic |
| `EnhancedRefiner` | ~150 | Prediction refinement |
| `IntegratedCapibaraTrainer` | ~390 | Main training orchestrator |
| `DatasetMerger` | ~40 | Dataset merging utilities |
| `ModelEvaluator` | ~100 | Model evaluation metrics |
| `VotingMetrics` | ~10 | Voting metric dataclass |
| `TeacherModel` | ~40 | Teacher model for distillation |
| `CriticModel` | ~30 | Critic model for evaluation |
| `AdvancedVotingSystem` | ~290 | Advanced voting consensus |
| `DistillationManager` | ~90 | Distillation management |
| `UnifiedVotingConsensus` | ~15 | Voting consensus facade |
| `UnifiedRefinementConsensus` | ~15 | Refinement consensus facade |
| `UnifiedDistillationConsensus` | ~15 | Distillation consensus facade |
| `UnifiedCrossTeachingConsensus` | ~15 | Cross-teaching consensus facade |

## Recommended Module Structure

```
training/consensus/unified/
├── __init__.py          # Re-exports all public APIs
├── config.py            # IntegratedTrainingConfig
├── distiller.py         # EnhancedConsensusDistiller
├── refiner.py           # EnhancedRefiner
├── trainer.py           # IntegratedCapibaraTrainer (largest, may need further split)
├── merger.py            # DatasetMerger
├── evaluator.py         # ModelEvaluator
├── voting.py            # VotingMetrics, TeacherModel, CriticModel, AdvancedVotingSystem
├── distillation.py      # DistillationManager
└── facades.py           # Unified*Consensus classes
```

## Migration Steps

1. **Create module files** with imports from unified_consensus.py
2. **Move classes one at a time**, updating imports
3. **Update __init__.py** to re-export from new locations
4. **Update unified_consensus.py** to import from modules
5. **Run tests** after each change to catch regressions
6. **Deprecate unified_consensus.py** after full migration

## Example Migration (config.py)

```python
# training/consensus/unified/config.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class IntegratedTrainingConfig:
    # Move class here
    ...

# training/consensus/unified/__init__.py
from .config import IntegratedTrainingConfig
```

## Backwards Compatibility

During migration, maintain backwards compatibility:

```python
# training/consensus/unified_consensus.py (deprecated)
import warnings
warnings.warn(
    "unified_consensus.py is deprecated, use training.consensus.unified instead",
    DeprecationWarning
)
from .unified import *
```

## Priority Order

1. **High**: `trainer.py` (390 lines) - Core training logic
2. **High**: `voting.py` (370+ lines) - Voting and critic models
3. **Medium**: `distiller.py` (120 lines) - Distillation
4. **Medium**: `evaluator.py` (100 lines) - Evaluation
5. **Low**: Remaining small classes

## Testing

After each module extraction:
```bash
python -m pytest tests/ -k "consensus" -v
```
