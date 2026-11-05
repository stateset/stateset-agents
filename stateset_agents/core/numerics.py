"""Stateless tensor numerics shared by ``core`` and ``training``.

These helpers live in ``core`` so that core modules (``value_function``,
``enhanced``) can use them without importing the training layer; training
code may import them from here too. torch is imported lazily, so importing
this module never requires torch.
"""

from __future__ import annotations

from typing import Any

__all__ = ["whiten"]


def whiten(
    x: Any, mask: Any | None = None, *, correction: int = 1, eps: float = 1e-8
) -> Any:
    """Standardise ``x`` with statistics taken over ``mask``-selected entries.

    The batch-level advantage normalisation used by PPO-style trainers. With
    one or no selected entries ``x`` is returned unchanged. When the spread is
    zero or non-finite the result is only centred, never divided by ~``eps``
    (which turned constant advantages into ~1e8-scale noise). ``correction``
    selects sample (1, ``Tensor.std`` default) or population (0) std, so call
    sites keep their numerics.

    ``stateset_agents.training.rl_losses.whiten`` delegates here, so the two
    are one implementation (``tests/unit/test_layering.py`` pins the parity).
    """
    import torch

    values = x[mask > 0] if mask is not None else x.reshape(-1)
    if values.numel() <= 1:
        return x
    mean = values.mean()
    std = values.std(correction=correction)
    if not bool(torch.isfinite(std)) or float(std) <= eps:
        return x - mean
    return (x - mean) / (std + eps)
