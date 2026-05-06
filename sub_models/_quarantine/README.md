# `_quarantine/`

Files in this directory are kept for archeology / future rewrites but are
**not part of the live tree**. They will not be imported by anything
productive and exist only so the original implementation intent is not lost.

## `mamba_module.py`

Selective State Space Model (Mamba/S6) draft.
**Broken**: uses PyTorch `.unsqueeze` inside `_selective_scan`, which is not
compatible with JAX/Flax (the rest of the project). The algorithmic intent is
correct and worth preserving.

Rewrite tracked under BACKLOG-018 (not yet promoted as of 2026-04-27).
Do not import from this file.
