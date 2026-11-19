"""Retained state for model, tokenizer, configuration, and adapter initialization."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


def _checked_component(name: str, value: Any, *, owned: bool = False) -> Any:
    """Reject missing/async results at the synchronous loader boundary."""
    if value is None:
        raise RuntimeError(f"{name} loader returned None during initialization")
    if inspect.isawaitable(value):
        if owned and inspect.iscoroutine(value):
            value.close()
        raise TypeError(
            f"{name} initialization requires a synchronous component, not an awaitable"
        )
    return value


class BackendInitializationState(str, Enum):
    """Outcome of the agent's explicit backend initialization attempt."""

    NEW = "new"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"


@dataclass
class _AdapterAttachment:
    request: dict[str, Any]
    input_model: Any
    output_model: Any = None
    error: BaseException | None = None


@dataclass
class _BackendInitialization:
    state: BackendInitializationState = BackendInitializationState.NEW
    phase: str = "pending"
    error: BaseException | None = None
    identity: tuple[Any, Any, Any] | None = None
    source: Any = None
    source_kind: str | None = None
    stub_request: dict[str, Any] | None = None
    adapter: _AdapterAttachment | None = None
    lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        with self.lock:
            if self.state is BackendInitializationState.INITIALIZING:
                raise RuntimeError(
                    "Cannot copy an agent while initializing its backend"
                )
            return {name: value for name, value in vars(self).items() if name != "lock"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.lock = RLock()
