"""Rollout generation always runs the model in inference mode.

Trainers put the policy in ``train()`` for the loss and backward pass. If a
rollout is then sampled in that state, LoRA dropout and gradient checkpointing
are active during ``generate`` and, with the KV cache disabled by
checkpointing, a PEFT model emits garbage (observed live on Qwen2.5-1.5B:
"To the same thing that we can either (100" in train mode against a correct
700-character solution from the same weights in eval mode). Every generation
path therefore wraps the model in :func:`inference_mode`: eval mode with the
KV cache enabled for the duration of the call, and the previous mode restored
afterwards so the trainer's state is untouched.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def inference_mode(model: Any) -> Iterator[None]:
    """Run ``model`` in eval mode with ``config.use_cache=True`` inside the
    block; restore its training flag and cache setting on exit."""
    was_training = bool(getattr(model, "training", False))
    config = getattr(model, "config", None)
    had_use_cache = getattr(config, "use_cache", None) if config is not None else None
    if callable(getattr(model, "eval", None)):
        model.eval()
    if config is not None and had_use_cache is not None:
        try:
            config.use_cache = True
        except Exception:  # noqa: BLE001 - frozen/odd configs keep their value
            had_use_cache = None
    try:
        yield
    finally:
        if config is not None and had_use_cache is not None:
            try:
                config.use_cache = had_use_cache
            except Exception:  # noqa: BLE001
                pass
        if was_training and callable(getattr(model, "train", None)):
            model.train()


__all__ = ["inference_mode"]
