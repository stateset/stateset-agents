"""Checkpoint helpers for the single-turn trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .checkpoint_state import load_training_checkpoint, save_training_checkpoint
from .trainer_utils import require_torch


def resolve_checkpoint_path(
    config: Any,
    global_step: int,
    is_best: bool = False,
    checkpoint_name: str | None = None,
) -> Path:
    """Resolve the destination path for a checkpoint."""
    if checkpoint_name is None:
        checkpoint_name = f"checkpoint-{global_step}"
        if is_best:
            checkpoint_name = "best-checkpoint"

    output_dir = getattr(config, "output_dir", "./outputs")
    checkpoint_path = Path(output_dir) / checkpoint_name
    return checkpoint_path


def save_checkpoint_artifacts(
    trainer: Any,
    checkpoint_path: Path,
    exceptions: tuple[type[BaseException], ...],
    logger: Any,
) -> None:
    """Persist model and trainer state, propagating artifact write failures.

    ``exceptions`` and ``logger`` remain accepted for caller compatibility;
    persistence errors are no longer suppressed.
    """
    save_training_checkpoint(trainer, checkpoint_path, require_torch_fn=require_torch)


def load_checkpoint_artifacts(
    trainer: Any,
    checkpoint_path: str | Path,
    require_torch_fn: Any,
    exceptions: tuple[type[BaseException], ...],
    logger: Any,
    *,
    trusted: bool = False,
) -> bool:
    """Load model and trainer state from a checkpoint directory.

    Args:
        trusted: When ``False`` (the default) checkpoints are unpickled with
            ``weights_only=True``, so only tensors and plain data are restored
            and a malicious checkpoint cannot execute code.  Pass ``True`` only
            for checkpoints from a source you control.
    """
    return load_training_checkpoint(
        trainer,
        checkpoint_path,
        require_torch_fn=require_torch_fn,
        trainer_exceptions=exceptions,
        logger=logger,
        trusted=trusted,
    )


__all__ = [
    "load_checkpoint_artifacts",
    "resolve_checkpoint_path",
    "save_checkpoint_artifacts",
]
