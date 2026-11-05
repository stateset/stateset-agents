"""The pluggable trainer behind ``train(training_mode="offline"|"hybrid")``.

The stable :func:`stateset_agents.training.train.train` owns the offline and
hybrid *modes* (argument checks, config mapping, seeding, resume, the online
phase of hybrid training) but the offline trainer itself is the experimental
``offline-rl`` component. Importing :mod:`stateset_agents.experimental.offline_rl`
registers it here, so the stable layer depends on this small protocol instead
of importing the experimental layer.

This module imports nothing heavy (no torch).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["load_offline_trainer", "register_offline_trainer"]

#: Returns ``(config_class, trainer_class)``; called lazily on first use so a
#: registration costs nothing until offline training actually runs.
OfflineTrainerLoader = Callable[[], tuple[type[Any], type[Any]]]

_loader: OfflineTrainerLoader | None = None


def register_offline_trainer(loader: OfflineTrainerLoader) -> None:
    """Register the loader for the offline trainer (last registration wins)."""
    global _loader
    _loader = loader


def load_offline_trainer() -> tuple[type[Any], type[Any]] | None:
    """Return ``(config_class, trainer_class)``, or ``None`` if none is registered."""
    if _loader is None:
        return None
    return _loader()
