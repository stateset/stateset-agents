"""
Compatibility wrapper for the legacy RULER reward API.

The original ``rewards.ruler_reward.RulerRewardFunction`` used a lightweight
placeholder judge. The real LLM‑as‑judge implementation now lives in
``rewards.llm_reward.RulerRewardFunction`` (litellm‑powered, rubric aware).

This module preserves the old import path and ``RulerConfig`` constructor
while delegating to the modern backend.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional

from .llm_reward import RulerRewardFunction as _LLMRulerRewardFunction


@dataclass
class RulerConfig:
    """Legacy configuration container for RULER rewards."""

    model: str = "openai/gpt-4"
    rubric_type: str = "default"
    custom_rubric: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 256
    weight: float = 1.0
    fallback_enabled: bool = True
    fallback_score: float = 0.5


class RulerRewardFunction(_LLMRulerRewardFunction):
    """Backward compatible RULER reward function.

    Accepts either a legacy ``RulerConfig`` or the modern keyword arguments
    supported by ``rewards.llm_reward.RulerRewardFunction``.
    """

    def __init__(self, config: Optional[RulerConfig] = None, **kwargs: Any):
        if config is not None:
            warnings.warn(
                "Passing RulerConfig is deprecated; instantiate "
                "RulerRewardFunction with keyword args instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            super().__init__(
                model=config.model,
                rubric_type=config.rubric_type,
                custom_rubric=config.custom_rubric,
                temperature=config.temperature,
                weight=config.weight,
                fallback_enabled=config.fallback_enabled,
                **kwargs,
            )
            # Keep legacy attribute for callers that referenced it.
            self.fallback_score = config.fallback_score
        else:
            super().__init__(**kwargs)


def create_customer_service_ruler(
    model: str = "openai/gpt-4", weight: float = 1.0, **kwargs: Any
) -> RulerRewardFunction:
    """Create a RULER judge configured for customer service."""
    return RulerRewardFunction(
        model=model, rubric_type="customer_service", weight=weight, **kwargs
    )


def create_general_ruler(
    model: str = "openai/gpt-4", weight: float = 1.0, **kwargs: Any
) -> RulerRewardFunction:
    """Create a general-purpose RULER judge."""
    return RulerRewardFunction(model=model, rubric_type="default", weight=weight, **kwargs)


def create_technical_support_ruler(
    model: str = "openai/gpt-4", weight: float = 1.0, **kwargs: Any
) -> RulerRewardFunction:
    """Create a RULER judge configured for technical support."""
    return RulerRewardFunction(
        model=model, rubric_type="technical_support", weight=weight, **kwargs
    )


def create_custom_ruler(
    rubric: str, model: str = "openai/gpt-4", weight: float = 1.0, **kwargs: Any
) -> RulerRewardFunction:
    """Create a RULER judge with a custom rubric."""
    return RulerRewardFunction(
        model=model, custom_rubric=rubric, rubric_type="default", weight=weight, **kwargs
    )


__all__ = [
    "RulerConfig",
    "RulerRewardFunction",
    "create_customer_service_ruler",
    "create_general_ruler",
    "create_technical_support_ruler",
    "create_custom_ruler",
]
