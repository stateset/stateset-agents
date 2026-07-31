"""Provider name to executor resolution.

Executors are imported lazily so that listing providers never requires their
SDKs. A user without ``modal`` installed still sees ``modal`` in the options;
the actionable "install the extra" message belongs at submit time, where it
can say what to do about it.
"""

from __future__ import annotations

from collections.abc import Callable

from stateset_agents.remote.executor import RemoteExecutionError, RemoteExecutor

__all__ = ["available_providers", "get_executor"]


def _load_local() -> RemoteExecutor:
    from stateset_agents.remote.local import LocalExecutor

    return LocalExecutor()


def _load_modal() -> RemoteExecutor:
    from stateset_agents.remote.modal import ModalExecutor

    return ModalExecutor()


_PROVIDERS: dict[str, Callable[[], RemoteExecutor]] = {
    "local": _load_local,
    "modal": _load_modal,
}


def available_providers() -> list[str]:
    """Every provider name accepted by :func:`get_executor`."""
    return sorted(_PROVIDERS)


def get_executor(provider: str) -> RemoteExecutor:
    """Construct the executor registered under ``provider``."""
    try:
        loader = _PROVIDERS[provider.strip().lower()]
    except KeyError:
        raise RemoteExecutionError(
            f"unknown provider {provider!r}; "
            f"available: {', '.join(available_providers())}",
            provider=provider,
        ) from None
    return loader()
