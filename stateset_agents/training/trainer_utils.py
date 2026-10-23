"""
Utility functions and lazy imports for GRPO training.

This module provides helper functions for lazy importing optional
dependencies like PyTorch and HuggingFace Transformers.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports - resolved on demand for optional dependency handling
torch: Any | None = None
F: Any | None = None
amp: Any | None = None

DataCollatorForLanguageModeling: Any | None = None
TrainingArguments: Any | None = None
get_cosine_schedule_with_warmup: Any | None = None
get_linear_schedule_with_warmup: Any | None = None

# Try initial imports
try:
    import torch as _torch
    import torch.amp as _amp
    import torch.nn.functional as _F

    torch = _torch
    F = _F
    amp = _amp
except ImportError:  # pragma: no cover - handled via helper functions
    pass

# Transformers imports are lazy to avoid torch/torchvision compatibility issues
_transformers_loaded = False


def _load_transformers_utils() -> bool:
    """Lazily load transformers utilities to avoid import-time errors."""
    global _transformers_loaded, DataCollatorForLanguageModeling, TrainingArguments
    global get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
    if _transformers_loaded:
        return True
    try:
        from transformers import (
            DataCollatorForLanguageModeling as _DataCollatorForLanguageModeling,
        )
        from transformers import TrainingArguments as _TrainingArguments
        from transformers import (
            get_cosine_schedule_with_warmup as _get_cosine_schedule_with_warmup,
        )
        from transformers import (
            get_linear_schedule_with_warmup as _get_linear_schedule_with_warmup,
        )

        DataCollatorForLanguageModeling = _DataCollatorForLanguageModeling
        TrainingArguments = _TrainingArguments
        get_cosine_schedule_with_warmup = _get_cosine_schedule_with_warmup
        get_linear_schedule_with_warmup = _get_linear_schedule_with_warmup
        _transformers_loaded = True
        return True
    except (ImportError, RuntimeError) as e:  # pragma: no cover
        logger.debug("Failed to load transformers: %s", e)
        return False


def require_torch() -> Any:
    """Ensure torch is available, importing lazily if needed."""
    global torch, F, amp
    if torch is None:
        try:
            import torch as _torch
            import torch.amp as _amp
            import torch.nn.functional as _F
        except ImportError as exc:  # pragma: no cover - import guarding
            raise ImportError(
                "PyTorch is required for training features. "
                "Install the 'training' extra: pip install stateset-agents[training]"
            ) from exc
        torch = _torch
        F = _F
        amp = _amp
    return torch


def require_transformers() -> None:
    """Ensure transformers scheduling utilities are available."""
    global DataCollatorForLanguageModeling, TrainingArguments
    global get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

    if (
        DataCollatorForLanguageModeling is None
        or TrainingArguments is None
        or get_cosine_schedule_with_warmup is None
        or get_linear_schedule_with_warmup is None
    ):
        try:
            from transformers import (
                DataCollatorForLanguageModeling as _DataCollatorForLanguageModeling,
            )
            from transformers import TrainingArguments as _TrainingArguments
            from transformers import (
                get_cosine_schedule_with_warmup as _get_cosine_schedule_with_warmup,
            )
            from transformers import (
                get_linear_schedule_with_warmup as _get_linear_schedule_with_warmup,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required for the training utilities. "
                "Install the 'training' extra: pip install stateset-agents[training]"
            ) from exc

        DataCollatorForLanguageModeling = _DataCollatorForLanguageModeling
        TrainingArguments = _TrainingArguments
        get_cosine_schedule_with_warmup = _get_cosine_schedule_with_warmup
        get_linear_schedule_with_warmup = _get_linear_schedule_with_warmup


def get_torch():
    """Get the torch module (may be None if not imported)."""
    return torch


def get_functional():
    """Get torch.nn.functional module (may be None if not imported)."""
    return F


def get_amp():
    """Get torch.amp module (may be None if not imported)."""
    return amp


def backward_training_loss(
    trainer: Any,
    loss: Any,
    torch_module: Any,
    *,
    scaler: Any = None,
    retain_graph: bool = False,
) -> None:
    """Backpropagate only a finite differentiable scalar, discarding failed work."""
    try:
        if (
            not torch_module.is_tensor(loss)
            or loss.ndim != 0
            or not loss.is_floating_point()
            or not loss.requires_grad
            or not bool(torch_module.isfinite(loss))
        ):
            raise ValueError("GRPO loss must be a finite differentiable scalar")
        target = scaler.scale(loss) if scaler is not None else loss
        if retain_graph:
            target.backward(retain_graph=True)
        else:
            target.backward()
    except BaseException:
        _discard_gradients(trainer)
        raise


def _discard_gradients(trainer: Any) -> None:
    trainer._grad_accum_step = 0
    trainer._grpo_accumulation = None
    if trainer.optimizer is not None:
        try:
            trainer.optimizer.zero_grad(set_to_none=True)
        except Exception as error:
            logger.warning("Could not discard failed gradients: %s", error)


def safe_optimizer_step(
    trainer: Any,
    torch_module: Any,
    *,
    max_grad_norm: float,
    scaler: Any = None,
    gradient_scale: float = 1.0,
) -> bool:
    """Validate gradients and count only committed optimizer steps.

    Standard GradScaler overflows consume the window and reduce the scale, but
    do not advance the scheduler, step counter, or rollout synchronization.
    A positive finite gradient_scale corrects partial-window normalization after
    AMP unscaling and before clipping.
    Exceptions after optimizer.step cannot roll back its already applied work.

    The GRPO trainers' configuration of the shared
    :class:`~stateset_agents.training.spine.update.OptimizerStep`: one optimizer
    and scheduler read from ``trainer``, ``global_step`` counting commits,
    normalised accumulation banks finalised first, gradients released and the
    rollout backend synchronised after each commit.
    """
    optimizer = trainer.optimizer
    if optimizer is None:
        return False
    from .grpo_accumulation import finalize_accumulated_gradients
    from .spine.update import OptimizerStep

    def finalize() -> bool:
        return bool(finalize_accumulated_gradients(trainer, torch_module))

    def discard() -> None:
        _discard_gradients(trainer)

    def on_commit() -> None:
        trainer._grpo_accumulation = None
        trainer._sync_rollout_backend()

    step = OptimizerStep(
        trainer,
        torch_module,
        optimizers=[optimizer],
        schedulers=[trainer.lr_scheduler],
        scaler=scaler,
        max_grad_norm=max_grad_norm,
        counter="commits",
        nonfinite_grad_policy="raise",
        zero_grad_after_commit=True,
        finalize=finalize,
        discard=discard,
        on_commit=on_commit,
        label="GRPO",
    )
    return step(gradient_scale=gradient_scale)


__all__ = [
    "get_amp",
    "get_functional",
    "get_torch",
    "require_torch",
    "require_transformers",
]

# Best-effort pre-load of lightweight Transformers utilities.
#
# Some trainer modules import scheduler helpers directly from this module
# (e.g. `from .trainer_utils import get_cosine_schedule_with_warmup`). In that
# pattern, updating globals later won't affect already-imported bindings, so we
# attempt to populate these helpers on import when possible.
_load_transformers_utils()


def disable_dropout_in_model(model: Any) -> int:
    """Set ``p = 0`` on every ``torch.nn.Dropout`` in ``model``; return how
    many modules were changed (TRL's ``disable_dropout_in_model``)."""
    torch = require_torch()
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return 0
    count = 0
    for module in modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            count += 1
    return count


def apply_dropout_setting(config: Any, model: Any) -> None:
    """Disable dropout in ``model`` when ``config.disable_dropout`` is set."""
    if bool(getattr(config, "disable_dropout", False)) and model is not None:
        count = disable_dropout_in_model(model)
        logger.info("disable_dropout: zeroed %d dropout modules", count)
