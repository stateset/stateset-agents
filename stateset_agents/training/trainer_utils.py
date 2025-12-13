"""
Utility functions and lazy imports for GRPO training.

This module provides helper functions for lazy importing optional
dependencies like PyTorch and HuggingFace Transformers.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lazy imports - resolved on demand for optional dependency handling
torch: Optional[Any] = None
F: Optional[Any] = None
amp: Optional[Any] = None

DataCollatorForLanguageModeling: Optional[Any] = None
TrainingArguments: Optional[Any] = None
get_cosine_schedule_with_warmup: Optional[Any] = None
get_linear_schedule_with_warmup: Optional[Any] = None

# Try initial imports
try:
    import torch as _torch  # type: ignore[import-not-found]
    import torch.nn.functional as _F  # type: ignore[import-not-found]
    import torch.amp as _amp  # type: ignore[import-not-found]

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
        from transformers import (  # type: ignore[import-not-found]
            DataCollatorForLanguageModeling as _DataCollatorForLanguageModeling,
            TrainingArguments as _TrainingArguments,
            get_cosine_schedule_with_warmup as _get_cosine_schedule_with_warmup,
            get_linear_schedule_with_warmup as _get_linear_schedule_with_warmup,
        )
        DataCollatorForLanguageModeling = _DataCollatorForLanguageModeling
        TrainingArguments = _TrainingArguments
        get_cosine_schedule_with_warmup = _get_cosine_schedule_with_warmup
        get_linear_schedule_with_warmup = _get_linear_schedule_with_warmup
        _transformers_loaded = True
        return True
    except (ImportError, RuntimeError) as e:  # pragma: no cover
        logger.warning(f"Failed to load transformers: {e}")
        return False


def require_torch() -> Any:
    """Ensure torch is available, importing lazily if needed."""
    global torch, F, amp
    if torch is None:
        try:
            import torch as _torch  # type: ignore[import-not-found]
            import torch.nn.functional as _F  # type: ignore[import-not-found]
            import torch.amp as _amp  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - import guarding
            raise ImportError(
                "PyTorch is required for training features. "
                "Install the 'training' extra: pip install stateset-agents[training]"
            ) from exc
        torch = _torch  # type: ignore[assignment]
        F = _F  # type: ignore[assignment]
        amp = _amp  # type: ignore[assignment]
    return torch  # type: ignore[return-value]


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
            from transformers import (  # type: ignore[import-not-found]
                DataCollatorForLanguageModeling as _DataCollatorForLanguageModeling,
                TrainingArguments as _TrainingArguments,
                get_cosine_schedule_with_warmup as _get_cosine_schedule_with_warmup,
                get_linear_schedule_with_warmup as _get_linear_schedule_with_warmup,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required for the training utilities. "
                "Install the 'training' extra: pip install stateset-agents[training]"
            ) from exc

        DataCollatorForLanguageModeling = _DataCollatorForLanguageModeling  # type: ignore[assignment]
        TrainingArguments = _TrainingArguments  # type: ignore[assignment]
        get_cosine_schedule_with_warmup = _get_cosine_schedule_with_warmup  # type: ignore[assignment]
        get_linear_schedule_with_warmup = _get_linear_schedule_with_warmup  # type: ignore[assignment]


def get_torch():
    """Get the torch module (may be None if not imported)."""
    return torch


def get_functional():
    """Get torch.nn.functional module (may be None if not imported)."""
    return F


def get_amp():
    """Get torch.amp module (may be None if not imported)."""
    return amp


# Best-effort pre-load of lightweight Transformers utilities.
# This keeps common scheduler helpers available for tests and simple consumers,
# while still guarding failures on environments with incompatible deps.
_load_transformers_utils()
