"""
Utility functions and lazy imports for GRPO training.

This module provides helper functions for lazy importing optional
dependencies like PyTorch and HuggingFace Transformers.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
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


def _clamped_cosine_lr(step: int, *, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then half-cosine decay to 0, held at 0 past ``total_steps``."""
    if step >= total_steps:
        return 0.0
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def inner_update_scheduler(optimizer: Any, config: Any, *, cosine: bool) -> Any:
    """Schedule one learning-rate tick per committed inner rollout update."""
    from functools import partial

    from torch.optim.lr_scheduler import ConstantLR, LambdaLR

    total = config.num_episodes * config.num_gradient_updates
    if not cosine:
        return ConstantLR(optimizer, factor=1.0, total_iters=total)
    schedule = partial(
        _clamped_cosine_lr,
        total_steps=total,
        warmup_steps=int(config.warmup_ratio * total),
    )
    return LambdaLR(optimizer, schedule)


@contextmanager
def deterministic_scoring(*models: Any) -> Iterator[None]:
    """Disable stochastic layers without disabling gradients; restore all modes."""
    modes = {
        module: module.training
        for model in models
        if model is not None
        for module in model.modules()
    }
    try:
        for model in models:
            if model is not None:
                model.eval()
        yield
    finally:
        # Restore mixed submodule modes as well as the root's flag.
        for module, training in modes.items():
            module.training = training


def encode_prompt_completions(
    tokenizer: Any,
    prompts: list[str],
    completions: list[str],
    *,
    max_prompt_length: int,
    max_completion_length: int,
    device: Any,
) -> tuple[Any, Any, Any]:
    """Tokenize text actions and verify their prompt prefixes and scoring budgets."""
    torch = require_torch()
    # One extra token detects an overlong prompt without allocating an
    # unbounded prompt batch. Never label a truncated prompt's tail as a
    # response merely because the full text has a larger token budget.
    prompt_encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length + 1,
    )
    prompt_lengths = prompt_encodings["attention_mask"].sum(dim=1)
    if (prompt_lengths > max_prompt_length).any():
        raise ValueError("prompts exceed max_prompt_length; truncate them explicitly")

    # Tokenize inputs
    full_texts = [p + c for p, c in zip(prompts, completions, strict=True)]
    encodings = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length + max_completion_length + 1,
    )

    # A BPE merge or an injected special token can change the separately
    # tokenized prompt. Length alone cannot locate the completion boundary.
    for row in range(len(prompts)):
        prompt_ids = prompt_encodings["input_ids"][row][
            prompt_encodings["attention_mask"][row].bool()
        ]
        full_ids = encodings["input_ids"][row][encodings["attention_mask"][row].bool()]
        if prompt_ids.numel() == 0 or not torch.equal(
            prompt_ids, full_ids[: prompt_ids.numel()]
        ):
            raise ValueError("prompt token boundary is ambiguous in combined text")
        if full_ids.numel() - prompt_ids.numel() > max_completion_length:
            raise ValueError(
                "completions exceed max_completion_length; truncated rewards are invalid"
            )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    return input_ids, attention_mask, prompt_lengths.to(device)


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


def _clip_gradient_norm(gradients: list[Any], cap: float, torch_module: Any) -> float:
    """Clip the joint L2 norm and return the pre-clip norm in binary64."""
    from .grpo_accumulation import _gradient_chunks

    if math.isnan(cap) or cap < 0:
        raise ValueError("max_grad_norm must be nonnegative")
    if not gradients:
        return 0.0
    pieces: list[Any] = []
    for gradient in gradients:
        components = (
            (gradient.real, gradient.imag) if gradient.is_complex() else (gradient,)
        )
        for component in components:
            if component.numel():
                pieces.extend(_gradient_chunks(component, 1024**2))

    def wide(piece: Any) -> Any:
        device = "cpu" if piece.device.type == "mps" else piece.device
        return piece.to(device=device, dtype=torch_module.float64)

    # Every binary16/bfloat16/binary32 coordinate squares exactly in binary64.
    # Even 2^63 such squares fit. Binary64 inputs need power-of-two scaling.
    unit_exp = 0
    if any(piece.dtype == torch_module.float64 for piece in pieces):
        peaks = [float(p.abs().amax()) for p in pieces]
        if not all(math.isfinite(p) for p in peaks):
            raise RuntimeError("The total gradient norm is non-finite")
        peak = max(peaks, default=0.0)
        if peak == 0:
            return 0.0
        unit_exp = math.frexp(peak)[1] - 1
    unit = math.ldexp(1.0, unit_exp)

    # Batch scalar transfers by device. Keep the binary64 work buffers bounded
    # even when a parameter is large or its gradient is noncontiguous.
    by_device: dict[Any, list[Any]] = {}
    for piece in pieces:
        value = wide(piece)
        if unit_exp:
            value = value / unit
        by_device.setdefault(value.device, []).append(value.square().sum())
    scaled_norm = math.sqrt(
        math.fsum(
            subtotal
            for group in by_device.values()
            for subtotal in torch_module.stack(group).tolist()
        )
    )
    if not math.isfinite(scaled_norm):
        raise RuntimeError("The total gradient norm is non-finite")
    norm = scaled_norm * unit
    if norm == 0:
        return 0.0
    if cap == math.inf:
        return norm
    if cap == 0:
        for piece in pieces:
            piece.zero_()
        return norm
    cap_mantissa, cap_exp = math.frexp(cap)
    if math.isfinite(norm):
        divisor, divisor_exp = math.frexp(norm + 1e-6)
    else:
        divisor, divisor_exp = scaled_norm, unit_exp
    mantissa, adjustment = math.frexp(cap_mantissa / divisor)
    exponent = cap_exp - divisor_exp + adjustment
    if exponent > 0:  # min(1, cap / (norm + epsilon)) is one.
        return norm
    # A coefficient can itself be below the binary64 subnormal range.
    # Apply it in normal factors, with just one conversion to the leaf dtype.
    factors = []
    while exponent < -1021:
        factors.append(2.0**-512)
        exponent += 512
    factors.append(math.ldexp(mantissa, exponent))
    for piece in pieces:
        result = wide(piece) * factors[0]
        for factor in factors[1:]:
            result.mul_(factor)
        piece.copy_(result)
    return norm


def clip_parameter_gradients(parameters: Iterable[Any], cap: float) -> float:
    """Clip the populated gradients of one parameter group and return its L2 norm."""
    backend = require_torch()
    with backend.no_grad():
        gradients = [p.grad for p in dict.fromkeys(parameters) if p.grad is not None]
        return _clip_gradient_norm(gradients, cap, backend)


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
        from .adamw_weights import ModelAdamW
        from .grpo_accumulation import finalize_accumulated_gradients

        if finalize_accumulated_gradients(trainer, torch_module):
            correction = 1.0
        step_options = {}
        if isinstance(optimizer, ModelAdamW):
            optimizer.prepare_model_gradients()
            step_options["accumulated"] = True
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
            active_scaler.step(optimizer, **step_options)
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
        with torch_module.no_grad():
            _clip_gradient_norm(gradients, max_norm, torch_module)
        if active_scaler is not None:
            active_scaler.step(optimizer, **step_options)
            active_scaler.update()
            scaler_updated = True
            if float(active_scaler.get_scale()) < previous_scale:
                _discard_gradients(trainer)
                return False
        else:
            optimizer.step(**step_options)
        # The model/optimizer commit occurred even if later scheduling fails.
        trainer.global_step += 1
        optimizer.zero_grad(set_to_none=True)
        trainer._grpo_accumulation = None
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
