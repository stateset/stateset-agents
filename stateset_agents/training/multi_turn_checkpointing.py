"""
Checkpoint helpers for the multi-turn trainer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
    torch = require_torch_fn()

    if checkpoint_name is None:
        checkpoint_name = f"checkpoint-{trainer.global_step}"
        if is_best:
            checkpoint_name = "best-checkpoint"

    output_dir = getattr(trainer.config, "output_dir", "./outputs")
    checkpoint_path = Path(output_dir) / checkpoint_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    trainer.agent.model.save_pretrained(checkpoint_path)
    trainer.agent.tokenizer.save_pretrained(checkpoint_path)

    training_state = {
        "global_step": trainer.global_step,
        "current_epoch": trainer.current_epoch,
        "best_eval_metric": trainer.best_eval_metric,
        "steps_without_improvement": trainer.steps_without_improvement,
        "grad_accum_step": trainer._grad_accum_step,
    }
    if trainer.optimizer is not None:
        training_state["optimizer_state_dict"] = trainer.optimizer.state_dict()

    if hasattr(trainer.config, "__dict__"):
        training_state["config"] = trainer.config.__dict__

    if trainer.lr_scheduler:
        training_state["scheduler_state_dict"] = trainer.lr_scheduler.state_dict()
    if trainer.continual_manager is not None:
        training_state["continual_state"] = trainer.continual_manager.state_dict()
        training_state["current_task_id"] = trainer._current_task_id

    torch.save(training_state, checkpoint_path / "training_state.pt")

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
) -> bool:
    """Load model and training state from a checkpoint directory."""
    try:
        torch = require_torch_fn()
    except ImportError:
        logger.warning("Cannot load checkpoint without PyTorch.")
        return False

    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning("Checkpoint path not found: %s", path)
        return False

    model_dir = path / "model" if (path / "model").is_dir() else path

    weights_loaded = False
    model_file = model_dir / "pytorch_model.bin"
    safetensors_file = model_dir / "model.safetensors"
    if getattr(trainer.agent, "model", None) is not None and hasattr(
        trainer.agent.model, "load_state_dict"
    ):
        try:
            if model_file.is_file():
                state_dict = torch.load(model_file, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                trainer.agent.model.load_state_dict(state_dict, strict=False)
                weights_loaded = True
            elif safetensors_file.is_file():
                try:
                    from safetensors.torch import load_file

                    state_dict = load_file(str(safetensors_file))
                    trainer.agent.model.load_state_dict(state_dict, strict=False)
                    weights_loaded = True
                except trainer_exceptions as exc:
                    logger.debug("Failed to load safetensors: %s", exc)
        except trainer_exceptions as exc:
            logger.warning("Failed to load model weights: %s", exc)

    if getattr(trainer.agent, "tokenizer", None) is not None:
        loader = getattr(trainer.agent.tokenizer, "from_pretrained", None)
        if callable(loader):
            try:
                trainer.agent.tokenizer = loader(model_dir)
            except trainer_exceptions as exc:
                logger.warning("Failed to load tokenizer: %s", exc)

    state_path = path / "training_state.pt"
    if not state_path.exists():
        if not weights_loaded:
            logger.warning("No training_state.pt found in %s", path)
        return False

    try:
        state = torch.load(state_path, map_location="cpu")
    except trainer_exceptions as exc:
        logger.warning("Failed to load training state: %s", exc)
        return False

    if not isinstance(state, dict):
        logger.warning("Unexpected training state format in %s", state_path)
        return False

    trainer.global_step = int(state.get("global_step", trainer.global_step))
    trainer.current_epoch = int(state.get("current_epoch", trainer.current_epoch))
    trainer.best_eval_metric = float(
        state.get("best_eval_metric", trainer.best_eval_metric)
    )
    trainer.steps_without_improvement = int(
        state.get("steps_without_improvement", trainer.steps_without_improvement)
    )
    trainer._grad_accum_step = int(
        state.get("grad_accum_step", trainer._grad_accum_step)
    )

    if trainer.optimizer is not None and state.get("optimizer_state_dict"):
        try:
            trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
        except trainer_exceptions as exc:
            logger.warning("Failed to load optimizer state: %s", exc)
    if trainer.lr_scheduler is not None and state.get("scheduler_state_dict"):
        try:
            trainer.lr_scheduler.load_state_dict(state["scheduler_state_dict"])
        except trainer_exceptions as exc:
            logger.warning("Failed to load scheduler state: %s", exc)

    if trainer.continual_manager is not None and state.get("continual_state"):
        trainer.continual_manager.load_state_dict(state["continual_state"])
    trainer._current_task_id = state.get("current_task_id", trainer._current_task_id)

    logger.info("Loaded checkpoint from %s", path)
    return True


__all__ = ["load_multi_turn_checkpoint", "save_multi_turn_checkpoint"]
