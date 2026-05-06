"""
Checkpointing with Orbax (preferred) and a pickle fallback.

Public API:
    cm = CheckpointManager(cfg, params_template)
    cm.save(step, params, opt_state, extra_metadata=...)
    params, opt_state, meta = cm.restore(step=None)   # latest if step is None
    steps = cm.list_steps()                            # ordered list of saved steps

Design notes:
- Async by default when Orbax is installed. We expose `cm.wait_until_finished()`
  for tests and graceful shutdown.
- `keep_last` is enforced AFTER each successful save by deleting the oldest
  checkpoint directories beyond the budget.
- Pickle fallback exists so smoke tests / single-host CPU runs still work
  without orbax installed; the pickle path is NOT recommended for real
  training because it's blocking and not sharded.
"""
from __future__ import annotations

import logging
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config_loader import CheckpointConfig

logger = logging.getLogger(__name__)

try:
    import orbax.checkpoint as ocp
    _HAVE_ORBAX = True
except ImportError:                          # pragma: no cover
    _HAVE_ORBAX = False
    ocp = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _step_dir(root: Path, step: int) -> Path:
    return root / f"step_{step:08d}"


def _list_step_dirs(root: Path) -> List[Tuple[int, Path]]:
    if not root.exists():
        return []
    out: List[Tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("step_"):
            continue
        try:
            step = int(name[len("step_"):])
        except ValueError:
            continue
        out.append((step, child))
    out.sort(key=lambda t: t[0])
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Save / restore training state under cfg.out_dir, keeping `keep_last` steps."""

    def __init__(self, cfg: CheckpointConfig):
        self.cfg = cfg
        self.root = Path(cfg.out_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._orbax_mgr: Optional[Any] = None
        if _HAVE_ORBAX:
            try:
                # CheckpointManager handles step bookkeeping itself, but we keep
                # our own directory layout for fallback compatibility - so we
                # just use the lower-level PyTreeCheckpointer.
                self._handler = ocp.PyTreeCheckpointer()
                self._mode = "orbax"
            except Exception as e:                # pragma: no cover - defensive
                logger.warning("orbax init failed (%s); falling back to pickle", e)
                self._mode = "pickle"
        else:
            self._mode = "pickle"
            logger.info(
                "orbax.checkpoint not importable; falling back to pickle "
                "(slower; not recommended for production)"
            )

    # ------------------------------------------------------------------
    # save / restore
    # ------------------------------------------------------------------

    def save(
        self,
        step: int,
        params: Any,
        opt_state: Any,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist (params, opt_state, metadata) for `step`. Returns the dir."""
        target = _step_dir(self.root, step)
        target.mkdir(parents=True, exist_ok=True)

        payload = {
            "params": params,
            "opt_state": opt_state,
            "metadata": dict(extra_metadata or {}),
            "step": step,
        }

        if self._mode == "orbax":
            # Orbax wants a clean directory it owns - so we save under
            # target/orbax/ and stash metadata pickled alongside (it
            # contains a python int we don't need to shard).
            orbax_dir = target / "orbax"
            if orbax_dir.exists():
                shutil.rmtree(orbax_dir)
            self._handler.save(
                orbax_dir,
                {"params": params, "opt_state": opt_state},
            )
            with open(target / "meta.pkl", "wb") as f:
                pickle.dump(
                    {"metadata": payload["metadata"], "step": step}, f,
                )
        else:
            # Pickle fallback - blocking, single file.
            with open(target / "state.pkl", "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._enforce_keep_last()
        logger.info("checkpoint saved at step=%d under %s (mode=%s)", step, target, self._mode)
        return target

    def restore(
        self, step: Optional[int] = None,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """Load (params, opt_state, metadata). If `step` is None, load the latest."""
        if step is None:
            steps = self.list_steps()
            if not steps:
                raise FileNotFoundError(f"no checkpoints under {self.root}")
            step = steps[-1]

        target = _step_dir(self.root, step)
        if not target.exists():
            raise FileNotFoundError(target)

        if self._mode == "orbax" and (target / "orbax").exists():
            tree = self._handler.restore(target / "orbax")
            params, opt_state = tree["params"], tree["opt_state"]
            with open(target / "meta.pkl", "rb") as f:
                meta = pickle.load(f)
            return params, opt_state, meta.get("metadata", {})

        # Either pickle mode or orbax-mode reading a pickle-saved older ckpt.
        with open(target / "state.pkl", "rb") as f:
            payload = pickle.load(f)
        return payload["params"], payload["opt_state"], payload.get("metadata", {})

    # ------------------------------------------------------------------
    # housekeeping
    # ------------------------------------------------------------------

    def list_steps(self) -> List[int]:
        return [s for s, _ in _list_step_dirs(self.root)]

    def latest_step(self) -> Optional[int]:
        steps = self.list_steps()
        return steps[-1] if steps else None

    def wait_until_finished(self) -> None:
        """No-op when not using a true async manager. Kept for API parity."""
        return None

    def _enforce_keep_last(self) -> None:
        steps_dirs = _list_step_dirs(self.root)
        excess = len(steps_dirs) - self.cfg.keep_last
        if excess <= 0:
            return
        for step, path in steps_dirs[:excess]:
            try:
                shutil.rmtree(path)
                logger.debug("pruned old checkpoint step=%d (%s)", step, path)
            except OSError as e:                  # pragma: no cover
                logger.warning("could not prune %s: %s", path, e)


__all__ = ["CheckpointManager"]
