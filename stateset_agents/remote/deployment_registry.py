"""Lazy registry for managed inference deployment providers."""

from __future__ import annotations

from collections.abc import Callable

from stateset_agents.remote.deployment import InferenceDeploymentProvider
from stateset_agents.remote.executor import RemoteExecutionError

__all__ = ["available_deployment_providers", "get_deployment_provider"]


def _coreweave() -> InferenceDeploymentProvider:
    from stateset_agents.remote.coreweave import CoreWeaveInferenceProvider

    return CoreWeaveInferenceProvider()


def _nebius() -> InferenceDeploymentProvider:
    from stateset_agents.remote.nebius import NebiusEndpointProvider

    return NebiusEndpointProvider()


def _huggingface() -> InferenceDeploymentProvider:
    from stateset_agents.remote.huggingface import HuggingFaceEndpointProvider

    return HuggingFaceEndpointProvider()


_PROVIDERS: dict[str, Callable[[], InferenceDeploymentProvider]] = {
    "coreweave": _coreweave,
    "huggingface": _huggingface,
    "nebius": _nebius,
}


def available_deployment_providers() -> list[str]:
    """Return managed inference providers in stable order."""
    return sorted(_PROVIDERS)


def get_deployment_provider(name: str) -> InferenceDeploymentProvider:
    """Construct a managed inference provider by name."""
    normalized = name.strip().lower()
    try:
        return _PROVIDERS[normalized]()
    except KeyError:
        raise RemoteExecutionError(
            f"unknown deployment provider {name!r}; available: "
            f"{', '.join(available_deployment_providers())}",
            provider=normalized,
        ) from None
