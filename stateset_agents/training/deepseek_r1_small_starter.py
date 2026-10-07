"""DeepSeek R1 Distill Qwen LoRA/QLoRA starter helpers."""

from __future__ import annotations

import logging

from stateset_agents.training.small_model_specs import small_model_spec
from stateset_agents.training.starter_factory import build_starter, starter_all

logger = logging.getLogger(__name__)
SPEC = small_model_spec(__name__, "deepseek_r1_small", "DeepSeek R1 Distill Qwen")
_SYMBOLS = build_starter(SPEC, logger)
globals().update(_SYMBOLS)

__all__ = starter_all(_SYMBOLS)
