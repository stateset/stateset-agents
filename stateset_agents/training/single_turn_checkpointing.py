"""Checkpoint helpers for the single-turn trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .trainer_utils import get_torch


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
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def save_checkpoint_artifacts(
    trainer: Any,
    checkpoint_path: Path,
    exceptions: tuple[type[BaseException], ...],
    logger: Any,
) -> None:
    """Persist model and trainer state to a checkpoint path."""
    agent = trainer.agent
    if hasattr(agent, "model") and agent.model is not None:
        agent.model.save_pretrained(checkpoint_path)
        if hasattr(agent, "tokenizer") and agent.tokenizer is not None:
            agent.tokenizer.save_pretrained(checkpoint_path)

    training_state = {
        "global_step": int(trainer.global_step),
        "current_epoch": int(trainer.current_epoch),
        "best_eval_metric": float(trainer.best_eval_metric),
        "steps_without_improvement": int(trainer.steps_without_improvement),
        "grad_accum_step": int(trainer._grad_accum_step),
    }
    if trainer.optimizer is not None:
        training_state["optimizer_state_dict"] = trainer.optimizer.state_dict()
    if trainer.lr_scheduler is not None:
        training_state["scheduler_state_dict"] = trainer.lr_scheduler.state_dict()
    if trainer.config is not None and hasattr(trainer.config, "__dict__"):
        training_state["config"] = trainer.config.__dict__
    if trainer.continual_manager is not None:
        training_state["continual_state"] = trainer.continual_manager.state_dict()
        training_state["current_task_id"] = trainer._current_task_id

    torch = get_torch()
    if torch is None:
        return

    try:
        torch.save(training_state, checkpoint_path / "training_state.pt")
    except exceptions as exc:  # pragma: no cover - best effort persistence
        logger.debug("Skipping training state save: %s", exc)


def load_checkpoint_artifacts(
    trainer: Any,
    checkpoint_path: str | Path,
    require_torch_fn: Any,
    exceptions: tuple[type[BaseException], ...],
    logger: Any,
) -> bool:
    """Load model and trainer state from a checkpoint directory."""
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
    agent = trainer.agent
    if getattr(agent, "model", None) is not None and hasattr(
        agent.model,
        "load_state_dict",
    ):
        try:
            if model_file.is_file():
                state_dict = torch.load(model_file, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                agent.model.load_state_dict(state_dict, strict=False)
                weights_loaded = True
            elif safetensors_file.is_file():
                try:
                    from safetensors.torch import load_file  # type: ignore[import-not-found]

                    state_dict = load_file(str(safetensors_file))
                    agent.model.load_state_dict(state_dict, strict=False)
                    weights_loaded = True
                except exceptions as exc:
                    logger.debug("Failed to load safetensors: %s", exc)
        except exceptions as exc:
            logger.warning("Failed to load model weights: %s", exc)

    if getattr(agent, "tokenizer", None) is not None:
        loader = getattr(agent.tokenizer, "from_pretrained", None)
        if callable(loader):
            try:
                agent.tokenizer = loader(model_dir)
            except exceptions as exc:
                logger.warning("Failed to load tokenizer: %s", exc)

    state_path = path / "training_state.pt"
    if not state_path.exists():
        if not weights_loaded:
            logger.warning("No training_state.pt found in %s", path)
        return False

    try:
        state = torch.load(state_path, map_location="cpu")
    except exceptions as exc:
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
        except exceptions as exc:
            logger.warning("Failed to load optimizer state: %s", exc)
    if trainer.lr_scheduler is not None and state.get("scheduler_state_dict"):
        try:
            trainer.lr_scheduler.load_state_dict(state["scheduler_state_dict"])
        except exceptions as exc:
            logger.warning("Failed to load scheduler state: %s", exc)

    if trainer.continual_manager is not None and state.get("continual_state"):
        trainer.continual_manager.load_state_dict(state["continual_state"])
    trainer._current_task_id = state.get("current_task_id", trainer._current_task_id)

    logger.info("Loaded checkpoint from %s", path)
    return True


__all__ = [
    "load_checkpoint_artifacts",
    "resolve_checkpoint_path",
    "save_checkpoint_artifacts",
]
