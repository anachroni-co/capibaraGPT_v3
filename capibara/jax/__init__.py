"""DEPRECATED shim removed in ISSUE-002.

capibara.jax used to be a wrapper around the real jax package. It caused
circular-import crashes (tree_util) because it shadowed the real jax on
sys.path. ISSUE-002 migrated all imports:

  capibara.jax.tpu_v4.* -> capibara.jax_ext.tpu_v4 (kept in-repo)
  everything else       -> jax (the real PyPI package)

This file is a tombstone: importing it raises so any remaining consumer
is caught loudly instead of silently re-shadowing the real jax.
"""
raise ImportError(
    "capibara.jax has been removed. Use the real 'jax' package "
    "(or capibara.jax_ext.tpu_v4 for the TPU kernels). "
    "See ISSUE-002 / branch fix/issue-002-jax-shim-collision."
)
