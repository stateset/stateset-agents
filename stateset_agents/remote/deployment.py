"""Provider-neutral managed inference deployment contracts."""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "DeploymentHandle",
    "DeploymentSpec",
    "InferenceDeploymentProvider",
]


@dataclass(frozen=True)
class DeploymentSpec:
    """A complete model deployment request, independent of provider APIs."""

    name: str
    model_name: str
    weights_uri: str
    gpu: str
    gpu_count: int = 1
    min_replicas: int = 1
    max_replicas: int = 1
    runtime: str = "vllm"
    runtime_version: str | None = None
    runtime_config: dict[str, str] = field(default_factory=dict)
    gateway_id: str | None = None
    zone: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("name", "model_name", "weights_uri", "gpu", "runtime"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        for field_name in ("gpu_count", "min_replicas", "max_replicas"):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.max_replicas < self.min_replicas:
            raise ValueError("max_replicas must be >= min_replicas")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeploymentHandle:
    """Durable pointer to a provider-managed inference deployment."""

    provider: str
    deployment_id: str
    model_name: str
    endpoint: str | None = None
    gateway_id: str | None = None
    owns_gateway: bool = False


class InferenceDeploymentProvider(abc.ABC):
    """Lifecycle contract for managed OpenAI-compatible deployments."""

    name: str = "unknown"

    @abc.abstractmethod
    def deploy(self, spec: DeploymentSpec) -> DeploymentHandle:
        """Create a deployment and return its durable handle."""

    @abc.abstractmethod
    def status(self, handle: DeploymentHandle) -> dict[str, Any]:
        """Return the provider's current deployment state."""

    @abc.abstractmethod
    def delete(self, handle: DeploymentHandle) -> None:
        """Delete the deployment so its compute stops billing."""
