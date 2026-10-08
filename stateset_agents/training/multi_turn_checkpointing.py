"""Checkpoint helpers for the multi-turn trainer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .checkpoint_state import load_training_checkpoint, save_training_checkpoint

logger = logging.getLogger(__name__)


async def save_multi_turn_checkpoint(
    trainer: Any,
    *,
    is_best: bool = False,
    checkpoint_name: str | None = None,
    require_torch_fn: Any,
    notify_checkpoint_saved_fn: Any,
) -> None:
    """Save model and trainer state for the multi-turn trainer."""
    if checkpoint_name is None:
        checkpoint_name = f"checkpoint-{trainer.global_step}"
        if is_best:
            checkpoint_name = "best-checkpoint"

    output_dir = getattr(trainer.config, "output_dir", "./outputs")
    checkpoint_path = Path(output_dir) / checkpoint_name
    save_training_checkpoint(
        trainer, checkpoint_path, require_torch_fn=require_torch_fn
    )

    logger.info("Checkpoint saved: %s", checkpoint_path)

    if trainer.wandb_logger:
        trainer.wandb_logger.log_model_checkpoint(
            str(checkpoint_path), trainer.global_step, is_best=is_best
        )

    await notify_checkpoint_saved_fn(
        trainer.callbacks,
        path=str(checkpoint_path),
        step=int(trainer.global_step),
        is_best=bool(is_best),
    )


def load_multi_turn_checkpoint(
    trainer: Any,
    checkpoint_path: Any,
    *,
    require_torch_fn: Any,
    trainer_exceptions: tuple[type[BaseException], ...],
    trusted: bool = False,
) -> bool:
    """Load model and training state from a checkpoint directory.

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
        trainer_exceptions=trainer_exceptions,
        logger=logger,
        trusted=trusted,
    )


__all__ = ["load_multi_turn_checkpoint", "save_multi_turn_checkpoint"]
