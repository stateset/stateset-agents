"""The shipped :class:`~stateset_agents.training.spine.algorithm.AlgorithmSpec`\\s.

Each prompt-group algorithm is one spec, defined next to its trainer and
resolved here by name. Lookups import the trainer module lazily, so
importing this module does not import ``torch``.

VAPO (a critic, value warmup and two optimizers) stays a specialised loop
built from the spine parts; MultiTurn and SingleTurn GRPO run the episode
loop (:class:`~stateset_agents.training.spine.loop.SpineLoopMixin`).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .algorithm import AlgorithmSpec

_SPECS: dict[str, tuple[str, str]] = {
    "gspo": ("stateset_agents.training.gspo_trainer", "GSPO_SPEC"),
    "gspo_token": ("stateset_agents.training.gspo_trainer", "GSPO_TOKEN_SPEC"),
    "dapo": ("stateset_agents.training.dapo_trainer", "DAPO_SPEC"),
    "gepo": ("stateset_agents.training.gepo_trainer", "GEPO_SPEC"),
}

#: Names :func:`algorithm_spec` resolves.
ALGORITHM_NAMES: tuple[str, ...] = tuple(_SPECS)


def algorithm_spec(name: str) -> AlgorithmSpec:
    """The shipped spec called ``name`` (one of :data:`ALGORITHM_NAMES`)."""
    try:
        module, attribute = _SPECS[name]
    except KeyError:
        raise ValueError(
            f"unknown algorithm {name!r}; expected one of {', '.join(ALGORITHM_NAMES)}"
        ) from None
    spec: AlgorithmSpec = getattr(importlib.import_module(module), attribute)
    return spec


__all__ = ["ALGORITHM_NAMES", "algorithm_spec"]
