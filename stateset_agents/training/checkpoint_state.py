"""Shared model, optimizer, gradient accumulation, and AMP checkpoint state."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .checkpoint_io import load_checkpoint_file
from .checkpoint_publication import (
    checkpoint_read,
    checkpoint_write,
    validate_checkpoint_publication,
)
from .model_checkpoint import load_model_checkpoint


def _accumulation_steps(trainer: Any) -> int:
    getter = getattr(trainer, "_get_grad_accum_steps", None)
    if callable(getter):
        return int(getter())
    return max(1, int(getattr(trainer.config, "gradient_accumulation_steps", 1) or 1))


def _parameters(trainer: Any) -> dict[str, Any]:
    model = trainer.agent.model
    named = getattr(model, "named_parameters", None)
    return dict(named()) if callable(named) else {}


def _validate_scaler(state: Any) -> None:
    if state is None or state == {}:
        return
    if not isinstance(state, dict) or set(state) != {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }:
        raise ValueError("Invalid checkpoint AMP scaler state")
    for name in ("scale", "growth_factor", "backoff_factor"):
        value = state[name]
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Invalid checkpoint AMP scaler state")
    if state["growth_factor"] <= 1 or state["backoff_factor"] >= 1:
        raise ValueError("Invalid checkpoint AMP scaler state")
    if (
        type(state["growth_interval"]) is not int
        or state["growth_interval"] < 1
        or type(state["_growth_tracker"]) is not int
        or not 0 <= state["_growth_tracker"] < state["growth_interval"]
    ):
        raise ValueError("Invalid checkpoint AMP scaler state")


def _validate_gradients(gradients: Any, parameters: dict[str, Any], torch: Any) -> None:
    if not isinstance(gradients, dict) or set(gradients) != set(parameters):
        raise ValueError("Checkpoint gradients differ from model parameters")
    if not any(value is not None for value in gradients.values()):
        raise ValueError("Pending accumulation window has no saved gradients")
    for name, gradient in gradients.items():
        if gradient is None:
            continue
        parameter = parameters[name]
        if (
            not torch.is_tensor(gradient)
            or gradient.layout != torch.strided
            or gradient.shape != parameter.shape
            or gradient.dtype != parameter.dtype
            or not bool(torch.isfinite(gradient).all())
        ):
            raise ValueError(f"Invalid checkpoint gradient for {name}")


def _capture_accumulation(trainer: Any, torch: Any) -> dict[str, Any]:
    steps = _accumulation_steps(trainer)
    count = trainer._grad_accum_step
    if type(count) is not int or count < 0:
        raise ValueError("Invalid gradient accumulation counter")
    gradients = {}
    if count % steps:
        if trainer.optimizer is None:
            raise ValueError("Pending accumulation requires an optimizer")
        parameters = _parameters(trainer)
        gradients = {name: parameter.grad for name, parameter in parameters.items()}
        _validate_gradients(gradients, parameters, torch)
        gradients = {
            name: None if value is None else value.detach().cpu().clone()
            for name, value in gradients.items()
        }
    scaler = getattr(trainer, "scaler", None)
    scaler_state = scaler.state_dict() if scaler is not None else None
    _validate_scaler(scaler_state)
    return {
        "schema_version": 1,
        "steps": steps,
        "gradients": gradients,
        "scaler": scaler_state,
    }


def _prepare_accumulation(
    trainer: Any, state: dict[str, Any], torch: Any
) -> tuple[dict[str, Any], Any]:
    """Validate and allocate gradients before mutating live training state."""
    steps = _accumulation_steps(trainer)
    count = state.get("grad_accum_step", 0)
    if type(count) is not int or count < 0:
        raise ValueError("Invalid checkpoint gradient accumulation counter")
    saved = state.get("gradient_accumulation")
    scaler = getattr(trainer, "scaler", None)
    parameters = _parameters(trainer)
    if saved is None:
        config = state.get("config")
        saved_steps = (
            config.get("gradient_accumulation_steps", steps)
            if isinstance(config, dict)
            else steps
        )
        if type(saved_steps) is not int or saved_steps < 1 or count % saved_steps:
            raise ValueError(
                "Legacy checkpoint has a pending accumulation window without gradients"
            )
        if scaler is not None:
            raise ValueError("Legacy checkpoint has no AMP scaler state")
        if count % steps:
            raise ValueError("Checkpoint accumulation schedule differs from trainer")
        return dict.fromkeys(parameters), None
    if (
        not isinstance(saved, dict)
        or type(saved.get("schema_version")) is not int
        or saved["schema_version"] != 1
        or type(saved.get("steps")) is not int
        or saved["steps"] != steps
        or set(saved) != {"schema_version", "steps", "gradients", "scaler"}
    ):
        raise ValueError(
            "Checkpoint accumulation schedule or schema differs from trainer"
        )
    scaler_state = saved["scaler"]
    _validate_scaler(scaler_state)
    if (scaler is None) != (scaler_state is None):
        raise ValueError("Checkpoint AMP scaler differs from trainer")
    if scaler is not None and bool(scaler.state_dict()) != bool(scaler_state):
        raise ValueError("Checkpoint AMP scaler enablement differs from trainer")
    gradients = saved["gradients"]
    if count % steps:
        _validate_gradients(gradients, parameters, torch)
        return {
            name: (
                None
                if value is None
                else value.detach().to(device=parameters[name].device).clone()
            )
            for name, value in gradients.items()
        }, scaler_state
    if not isinstance(gradients, dict) or gradients:
        raise ValueError("Completed accumulation window cannot contain gradients")
    return dict.fromkeys(parameters), scaler_state


def save_training_checkpoint(
    trainer: Any, checkpoint_path: Path, *, require_torch_fn: Any
) -> None:
    """Persist a resumable checkpoint; propagate every artifact write failure."""
    torch = require_torch_fn()
    if getattr(trainer.agent, "model", None) is None:
        raise ValueError("Cannot save a training checkpoint without a model")
    accumulation = _capture_accumulation(trainer, torch)
    with checkpoint_write(checkpoint_path) as staging:
        _write_training_checkpoint(trainer, staging, torch, accumulation)


def _write_training_checkpoint(
    trainer: Any, checkpoint_path: Path, torch: Any, accumulation: dict[str, Any]
) -> None:
    trainer.agent.model.save_pretrained(checkpoint_path)
    if getattr(trainer.agent, "tokenizer", None) is not None:
        trainer.agent.tokenizer.save_pretrained(checkpoint_path)

    training_state = {
        "global_step": trainer.global_step,
        "current_epoch": trainer.current_epoch,
        "best_eval_metric": trainer.best_eval_metric,
        "steps_without_improvement": trainer.steps_without_improvement,
        "grad_accum_step": trainer._grad_accum_step,
        "gradient_accumulation": accumulation,
        "checkpoint_publication": 1,
    }
    if trainer.optimizer is not None:
        training_state["optimizer_state_dict"] = trainer.optimizer.state_dict()

    if hasattr(trainer.config, "__dict__"):
        training_state["config"] = trainer.config.__dict__

    if trainer.lr_scheduler is not None:
        training_state["scheduler_state_dict"] = trainer.lr_scheduler.state_dict()
    if trainer.continual_manager is not None:
        training_state["continual_state"] = trainer.continual_manager.state_dict()
        training_state["current_task_id"] = trainer._current_task_id

    torch.save(training_state, checkpoint_path / "training_state.pt")


def load_training_checkpoint(
    trainer: Any,
    checkpoint_path: Any,
    *,
    require_torch_fn: Any,
    trainer_exceptions: tuple[type[BaseException], ...],
    logger: Any,
    trusted: bool = False,
) -> bool:
    """Load model and training state from a checkpoint directory.

    Args:
        trusted: When ``False`` (the default) checkpoints are unpickled with
            ``weights_only=True``, so only tensors and plain data are restored
            and a malicious checkpoint cannot execute code.  Pass ``True`` only
            for checkpoints from a source you control.
    """
    path = Path(checkpoint_path).resolve()
    with checkpoint_read(path):
        return _load_training_checkpoint(
            trainer,
            path,
            require_torch_fn=require_torch_fn,
            trainer_exceptions=trainer_exceptions,
            logger=logger,
            trusted=trusted,
        )


def _load_training_checkpoint(
    trainer: Any,
    path: Path,
    *,
    require_torch_fn: Any,
    trainer_exceptions: tuple[type[BaseException], ...],
    logger: Any,
    trusted: bool,
) -> bool:
    try:
        torch = require_torch_fn()
    except ImportError:
        logger.warning("Cannot load checkpoint without PyTorch.")
        return False

    if not path.exists():
        logger.warning("Checkpoint path not found: %s", path)
        return False

    published = validate_checkpoint_publication(path)
    model_dir = path / "model" if (path / "model").is_dir() else path

    state_path = path / "training_state.pt"
    state = None
    gradients: dict[str, Any] = {}
    scaler_state = None
    if state_path.exists():
        try:
            state = load_checkpoint_file(
                state_path, trusted=trusted, torch_module=torch
            )
        except trainer_exceptions as exc:
            logger.warning("Failed to load training state: %s", exc)
            return False
        if not isinstance(state, dict):
            logger.warning("Unexpected training state format in %s", state_path)
            return False
        if "checkpoint_publication" in state and (
            type(state["checkpoint_publication"]) is not int
            or state["checkpoint_publication"] != 1
            or not published
        ):
            raise ValueError("Checkpoint publication is incomplete or unsupported")
        gradients, scaler_state = _prepare_accumulation(trainer, state, torch)

    complete_state = state is not None and "gradient_accumulation" in state
    if complete_state:
        assert state is not None
        for component, key in (
            (trainer.optimizer, "optimizer_state_dict"),
            (trainer.lr_scheduler, "scheduler_state_dict"),
        ):
            if (component is None) != (key not in state):
                raise ValueError(f"Checkpoint {key} availability differs from trainer")
            if component is not None and (
                not isinstance(state.get(key), dict)
                or not set(component.state_dict()) <= set(state[key])
            ):
                raise ValueError(f"Checkpoint is missing or incomplete: {key}")

    weights_loaded = False
    if getattr(trainer.agent, "model", None) is not None and hasattr(
        trainer.agent.model, "load_state_dict"
    ):
        try:
            weights_loaded = load_model_checkpoint(
                trainer.agent.model,
                model_dir,
                torch_module=torch,
                trusted=trusted,
                strict=complete_state,
            )
        except trainer_exceptions as exc:
            if complete_state:
                raise
            logger.warning("Failed to load model weights: %s", exc)

    if complete_state and not weights_loaded:
        raise ValueError("Checkpoint is missing loadable model weights")

    if getattr(trainer.agent, "tokenizer", None) is not None:
        loader = getattr(trainer.agent.tokenizer, "from_pretrained", None)
        if callable(loader):
            try:
                trainer.agent.tokenizer = loader(model_dir)
            except trainer_exceptions as exc:
                logger.warning("Failed to load tokenizer: %s", exc)

    if state is None:
        if not weights_loaded:
            logger.warning("No training_state.pt found in %s", path)
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

    if trainer.optimizer is not None and "optimizer_state_dict" in state:
        try:
            trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
        except trainer_exceptions as exc:
            raise ValueError("Failed to load checkpoint optimizer state") from exc
    if trainer.lr_scheduler is not None and "scheduler_state_dict" in state:
        try:
            trainer.lr_scheduler.load_state_dict(state["scheduler_state_dict"])
        except trainer_exceptions as exc:
            raise ValueError("Failed to load checkpoint scheduler state") from exc

    if trainer.continual_manager is not None and state.get("continual_state"):
        trainer.continual_manager.load_state_dict(state["continual_state"])
    trainer._current_task_id = state.get("current_task_id", trainer._current_task_id)
    if scaler_state is not None:
        trainer.scaler.load_state_dict(scaler_state)
        pending_gradient = next(
            (gradient for gradient in gradients.values() if gradient is not None), None
        )
        if scaler_state and pending_gradient is not None:
            # GradScaler restores scale/growth lazily. A resumed final window may
            # unscale saved gradients before any new backward pass initializes
            # it. Use the public API without rescaling the saved gradients.
            trainer.scaler.scale(torch.zeros((), device=pending_gradient.device))
    parameters = _parameters(trainer)
    for name, gradient in gradients.items():
        parameters[name].grad = gradient

    logger.info("Loaded checkpoint from %s", path)
    return True
