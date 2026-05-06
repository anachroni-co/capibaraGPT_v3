"""Sub-Models Module - CapibaraGPT v3.

Surviving submodules after BACKLOG-017 cleanup. Imports are explicit;
no silent-fallback is permitted (CONTRIBUTING.md §1).

Available:
- Byte_TPU (kept for capibaras chain)
- csa_expert.CSAExpert
- deep_dialog.DeepDialog / DeepDialogConfig
- reasoning_enhancement.ReasoningEnhancementExpert
- hybrid.HybridAttentionModule

Removed (see BACKLOG-017):
- SSM_TPU (was a duplicate of capibara/ssm/ssm_tpu.py)
- aleph_Tilde, capibaras.capibara2, csa_expert_tpu_optimized
- experimental.{dual_process, liquid, meta_bamdp, snns_LiCell, spike_ssm}
- semiotic/* (entire package)
- ultra_enhanced_integration, ultra_submodel_orchestrator
- vision.capivision

Quarantined (see sub_models/_quarantine/README.md):
- mamba.mamba_module (broken: PyTorch syntax in JAX repo)
"""
from .csa_expert import CSAExpert
from .deep_dialog import DeepDialog, DeepDialogConfig
from .reasoning_enhancement import ReasoningEnhancementExpert

__all__ = [
    "CSAExpert",
    "DeepDialog",
    "DeepDialogConfig",
    "ReasoningEnhancementExpert",
]
