"""Text-only Ministral 3 LoRA/QLoRA starter helpers."""

from __future__ import annotations

import logging

from stateset_agents.training.small_model_specs import small_model_spec
from stateset_agents.training.starter_factory import build_starter, starter_all

logger = logging.getLogger(__name__)
SPEC = small_model_spec(__name__, "ministral3", "Ministral 3")
_SYMBOLS = build_starter(SPEC, logger)
globals().update(_SYMBOLS)

__all__ = starter_all(_SYMBOLS)
