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


def _load_fireworks() -> RemoteExecutor:
    from stateset_agents.remote.fireworks import FireworksExecutor

    return FireworksExecutor()


def _load_coreweave() -> RemoteExecutor:
    from stateset_agents.remote.coreweave import CoreWeaveExecutor

    return CoreWeaveExecutor()


def _load_local() -> RemoteExecutor:
    from stateset_agents.remote.local import LocalExecutor

    return LocalExecutor()


def _load_modal() -> RemoteExecutor:
    from stateset_agents.remote.modal import ModalExecutor

    return ModalExecutor()


def _load_nebius() -> RemoteExecutor:
    from stateset_agents.remote.nebius import NebiusExecutor

    return NebiusExecutor()


def _load_runpod() -> RemoteExecutor:
    from stateset_agents.remote.runpod import RunPodExecutor

    return RunPodExecutor()


def _load_river() -> RemoteExecutor:
    from stateset_agents.remote.river import RiverExecutor

    return RiverExecutor()


def _load_huggingface() -> RemoteExecutor:
    from stateset_agents.remote.huggingface import HuggingFaceJobsExecutor

    return HuggingFaceJobsExecutor()


def _load_prime() -> RemoteExecutor:
    from stateset_agents.remote.prime import PrimeLabExecutor

    return PrimeLabExecutor()


def _load_tinker() -> RemoteExecutor:
    from stateset_agents.remote.tinker import TinkerExecutor

    return TinkerExecutor()


def _load_together() -> RemoteExecutor:
    from stateset_agents.remote.together import TogetherExecutor

    return TogetherExecutor()


_PROVIDERS: dict[str, Callable[[], RemoteExecutor]] = {
    "coreweave": _load_coreweave,
    "fireworks": _load_fireworks,
    "huggingface": _load_huggingface,
    "local": _load_local,
    "modal": _load_modal,
    "nebius": _load_nebius,
    "prime": _load_prime,
    "river": _load_river,
    "runpod": _load_runpod,
    "tinker": _load_tinker,
    "together": _load_together,
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
