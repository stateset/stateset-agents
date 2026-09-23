"""
Checkpoint management for the autonomous research loop.

Handles saving and restoring model state between experiments so that
successful experiments can be kept and failures can be reverted.
Uses atomic directory swaps to prevent corruption on crash.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Filesystem-based checkpoint manager for the auto-research loop.

    Maintains a 'best' checkpoint that gets updated when an experiment
    improves on the current best. Failed experiments can revert to the
    last known good state.

    Saves are atomic: writes go to a temporary directory first, then
    atomically renamed into place. This prevents corruption if the process
    crashes mid-save.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.checkpoints_dir = output_dir / "checkpoints"
        self.best_dir = self.checkpoints_dir / "best"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        old_best = self.checkpoints_dir / ".best_old"
        if (
            not self.best_dir.exists()
            and (old_best / "metadata.json").is_file()
            and (old_best / ".complete").is_file()
        ):
            old_best.rename(self.best_dir)
            logger.info("Recovered interrupted best checkpoint swap")

    def save_best(
        self,
        agent: Any,
        experiment_id: str,
        params: dict[str, Any],
        *,
        defer_commit: bool = False,
    ) -> Path:
        """Save current agent state as the new best checkpoint.

        Uses a directory swap to preserve the previous best. When
        ``defer_commit`` is true, the previous checkpoint remains available
        until the caller persists its experiment record and calls
        :meth:`commit_best`.
        Returns the path to the saved checkpoint.
        """
        if (self.best_dir / ".pending").exists():
            raise RuntimeError("previous best checkpoint is not committed")
        # Write to temp directory first
        tmp_dir = Path(tempfile.mkdtemp(dir=self.checkpoints_dir, prefix=".best_tmp_"))
        old_best = self.checkpoints_dir / ".best_old"
        try:
            model_path = tmp_dir / "model"
            model_path.mkdir(exist_ok=True)
            self._save_agent_state(agent, model_path)

            meta = {"experiment_id": experiment_id, "params": params}
            meta_path = tmp_dir / "metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            # Write completion marker
            (tmp_dir / ".complete").touch()
            if defer_commit:
                (tmp_dir / ".pending").touch()

            # Atomic swap: remove old best, rename tmp to best
            if self.best_dir.exists():
                # Rename current best out of the way
                if old_best.exists():
                    shutil.rmtree(old_best)
                self.best_dir.rename(old_best)

            tmp_dir.rename(self.best_dir)

            # Deferred commits keep the prior checkpoint for recovery if the
            # experiment record is never persisted.
            if old_best.exists() and not defer_commit:
                shutil.rmtree(old_best, ignore_errors=True)

            logger.info("Saved best checkpoint: %s → %s", experiment_id, self.best_dir)
            return self.best_dir

        except Exception:
            # Clean up temp dir on failure
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            if not self.best_dir.exists() and old_best.exists():
                old_best.rename(self.best_dir)
            raise

    def commit_best(self, experiment_id: str) -> None:
        """Commit a provisional best after its experiment record is written."""
        metadata = self.load_best_metadata()
        if metadata is None or metadata.get("experiment_id") != experiment_id:
            raise ValueError("best checkpoint does not match experiment record")
        (self.best_dir / ".pending").unlink(missing_ok=True)
        old_best = self.checkpoints_dir / ".best_old"
        if old_best.exists():
            shutil.rmtree(old_best)

    def reconcile_best(self, committed_experiment_id: str | None) -> bool:
        """Match a provisional checkpoint to the replayed experiment log.

        An unrecorded checkpoint is moved aside rather than deleted, then the
        previous best is restored when available. Return whether the active
        checkpoint matches the log's best experiment.
        """
        metadata = self.load_best_metadata() if self.has_best() else None
        if metadata is None:
            return False
        current_id = metadata.get("experiment_id")
        old_best = self.checkpoints_dir / ".best_old"
        if (self.best_dir / ".pending").exists():
            if isinstance(current_id, str) and current_id == committed_experiment_id:
                self.commit_best(current_id)
            else:
                orphan = self.checkpoints_dir / f".uncommitted_{uuid.uuid4().hex}"
                self.best_dir.rename(orphan)
                if (old_best / "metadata.json").is_file() and (
                    old_best / ".complete"
                ).is_file():
                    old_best.rename(self.best_dir)
                logger.warning("Set aside unrecorded best checkpoint %s", current_id)
        elif old_best.exists():
            # A crash after removing the pending marker can leave the backup.
            shutil.rmtree(old_best)

        active = self.load_best_metadata() if self.has_best() else None
        return (
            active is not None
            and active.get("experiment_id") == committed_experiment_id
        )

    def save_experiment(
        self,
        agent: Any,
        experiment_id: str,
        params: dict[str, Any],
    ) -> Path:
        """Save a named experiment checkpoint (kept experiments only)."""
        exp_dir = self.checkpoints_dir / experiment_id
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)

        model_path = exp_dir / "model"
        model_path.mkdir(exist_ok=True)
        self._save_agent_state(agent, model_path)

        meta = {"experiment_id": experiment_id, "params": params}
        with open(exp_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return exp_dir

    def has_best(self) -> bool:
        """Check if a valid best checkpoint exists."""
        meta = self.best_dir / "metadata.json"
        complete = self.best_dir / ".complete"
        # Require both metadata and completion marker
        return meta.exists() and complete.exists()

    def load_best_metadata(self) -> dict[str, Any] | None:
        """Load metadata from the best checkpoint."""
        meta_path = self.best_dir / "metadata.json"
        if not meta_path.exists():
            return None
        with open(meta_path, encoding="utf-8") as f:
            return cast(dict[str, Any], json.load(f))

    def restore_best(self, agent: Any) -> bool:
        """Restore agent state from the best checkpoint.

        Returns True if restoration succeeded, False if no checkpoint exists.
        """
        if not self.has_best():
            logger.debug("No valid best checkpoint to restore from")
            return False

        model_path = self.best_dir / "model"
        if not model_path.exists():
            logger.warning("Best checkpoint exists but has no model directory")
            return False

        return self._restore_agent_state(agent, model_path)

    def _save_agent_state(self, agent: Any, path: Path) -> None:
        """Save agent model state to disk."""
        model = getattr(agent, "model", None)
        if model is None:
            logger.debug("Agent has no .model attribute; skipping weight save")
            return

        # Try PEFT/LoRA save first (saves only adapter weights)
        try:
            save_pretrained = getattr(model, "save_pretrained", None)
            if save_pretrained is not None:
                save_pretrained(str(path))
                logger.debug("Saved model via save_pretrained to %s", path)

                # If this is a PEFT model, also save the base model name
                # so we can re-wrap on restore
                try:
                    from peft import PeftModel

                    if isinstance(model, PeftModel):
                        base_name = getattr(
                            model.base_model.model, "name_or_path", None
                        ) or getattr(model, "name_or_path", None)
                        if base_name:
                            (path / "base_model_name.txt").write_text(
                                base_name, encoding="utf-8"
                            )
                except Exception:
                    pass

                return
        except Exception as exc:
            logger.debug("save_pretrained failed: %s", exc)

        # Fallback: save state_dict via torch
        try:
            import torch

            state_dict = model.state_dict()
            torch.save(state_dict, path / "model_state.pt")
            logger.debug("Saved state_dict to %s", path / "model_state.pt")
        except Exception as exc:
            logger.warning("Could not save model weights: %s", exc)

    def _restore_agent_state(self, agent: Any, path: Path) -> bool:
        """Restore agent model state from disk. Returns True on success."""
        model = getattr(agent, "model", None)
        if model is None:
            logger.debug("Agent has no .model attribute; skipping weight restore")
            return False

        adapter_config = path / "adapter_config.json"
        state_pt = path / "model_state.pt"

        # Strategy 1: PEFT adapter
        if adapter_config.exists():
            try:
                from peft import PeftModel

                if isinstance(model, PeftModel):
                    # Already a PeftModel — load adapter directly
                    model.load_adapter(str(path), adapter_name="default")
                    logger.info("Restored PEFT adapter from %s", path)
                    return True
                else:
                    # Base model without PEFT wrapper — wrap it first,
                    # then load the adapter weights
                    try:
                        wrapped = PeftModel.from_pretrained(model, str(path))
                        agent.model = wrapped
                        logger.info(
                            "Wrapped base model with PEFT and loaded adapter from %s",
                            path,
                        )
                        return True
                    except Exception as wrap_exc:
                        logger.warning(
                            "Could not wrap base model with PEFT adapter: %s. "
                            "Falling back to state_dict.",
                            wrap_exc,
                        )
            except ImportError:
                logger.warning("PEFT not installed; cannot restore adapter")
            except Exception as exc:
                logger.warning("PEFT adapter restore failed: %s", exc)

        # Strategy 2: state_dict
        if state_pt.exists():
            try:
                from ..checkpoint_io import load_checkpoint_file

                state_dict = load_checkpoint_file(state_pt)
                model.load_state_dict(state_dict, strict=False)
                logger.info("Restored state_dict from %s", state_pt)
                return True
            except Exception as exc:
                logger.warning("state_dict restore failed: %s", exc)

        # Strategy 3: from_pretrained
        config_json = path / "config.json"
        if config_json.exists():
            try:
                from_pretrained = getattr(type(model), "from_pretrained", None)
                if from_pretrained is not None:
                    new_model = from_pretrained(str(path))
                    model.load_state_dict(new_model.state_dict(), strict=False)
                    del new_model
                    logger.info("Restored via from_pretrained from %s", path)
                    return True
            except Exception as exc:
                logger.warning("from_pretrained restore failed: %s", exc)

        logger.warning("No restorable model state found in %s", path)
        return False
