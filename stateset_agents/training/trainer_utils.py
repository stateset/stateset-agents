"""
Utility functions and lazy imports for GRPO training.

This module provides helper functions for lazy importing optional
dependencies like PyTorch and HuggingFace Transformers.
"""

from __future__ import annotations

import logging
import math
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
    trainer: Any, loss: Any, torch_module: Any, *, scaler: Any = None
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
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
    except BaseException:
        _discard_gradients(trainer)
        raise


def _discard_gradients(trainer: Any) -> None:
    trainer._grad_accum_step = 0
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
    """
    optimizer = trainer.optimizer
    if optimizer is None:
        return False
    active_scaler = None
    unscaled = False
    scaler_updated = False
    previous_scale = 1.0
    try:
        correction = float(gradient_scale)
        if not math.isfinite(correction) or correction <= 0:
            raise ValueError("gradient_scale must be finite and positive")
        max_norm = float(max_grad_norm)
        if not math.isfinite(max_norm) or max_norm < 0:
            raise ValueError("max_grad_norm must be finite and nonnegative")
        parameters = list(
            {
                id(parameter): parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            }.values()
        )
        gradients = [
            parameter.grad for parameter in parameters if parameter.grad is not None
        ]
        if not gradients:
            _discard_gradients(trainer)
            return False
        active_scaler = scaler if scaler is not None and scaler.is_enabled() else None
        if active_scaler is not None:
            previous_scale = float(active_scaler.get_scale())
            if not math.isfinite(previous_scale) or previous_scale <= 0:
                raise ValueError("AMP scale must be finite and positive")
            active_scaler.unscale_(optimizer)
            unscaled = True
            gradients = [
                parameter.grad for parameter in parameters if parameter.grad is not None
            ]
        # Aggregate on each device before converting to bool, avoiding one
        # host/device synchronization for every parameter tensor.
        checks_by_device: dict[Any, list[Any]] = {}
        for gradient in gradients:
            checks_by_device.setdefault(gradient.device, []).append(
                torch_module.isfinite(gradient).all()
            )
        finite = all(
            bool(torch_module.stack(checks).all())
            for checks in checks_by_device.values()
        )
        if not finite:
            if active_scaler is None:
                raise ValueError("GRPO gradients must be finite")
            # unscale_ recorded these non-finite gradients; GradScaler skips the
            # optimizer and updates its overflow scale using those observations.
            active_scaler.step(optimizer)
            active_scaler.update()
            scaler_updated = True
            _discard_gradients(trainer)
            return False
        # Partial windows were accumulated as sum(loss / configured_steps).
        # Correct their mean after AMP unscaling, before clipping. Any overflow
        # introduced here must fail clipping, not masquerade as an AMP skip.
        if correction != 1.0:
            with torch_module.no_grad():
                for gradient in gradients:
                    gradient.mul_(correction)
        torch_module.nn.utils.clip_grad_norm_(
            parameters, max_norm, error_if_nonfinite=True
        )
        if active_scaler is not None:
            active_scaler.step(optimizer)
            active_scaler.update()
            scaler_updated = True
            if float(active_scaler.get_scale()) < previous_scale:
                _discard_gradients(trainer)
                return False
        else:
            optimizer.step()
        # The model/optimizer commit occurred even if later scheduling fails.
        trainer.global_step += 1
        optimizer.zero_grad(set_to_none=True)
        if trainer.lr_scheduler is not None:
            trainer.lr_scheduler.step()
        trainer._sync_rollout_backend()
        return True
    except BaseException:
        _discard_gradients(trainer)
        if active_scaler is not None and unscaled and not scaler_updated:
            try:
                # Reset its per-optimizer phase without advancing the scale's
                # growth tracker after a rejected clipping/backward operation.
                active_scaler.update(new_scale=previous_scale)
            except Exception as error:
                logger.warning("Could not reset AMP state after failure: %s", error)
        raise


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
