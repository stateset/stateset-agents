"""Read-only live-provider canaries with machine-readable results.

Canaries authenticate against the real provider APIs and exercise the exact
SDK surfaces used by the executors, but never create training jobs, pods, or
deployments. They also look for StateSet canary resources left by a previous
run so cleanup regressions become visible without deleting user resources.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import islice
from typing import Any

from stateset_agents.remote.executor import RemoteExecutionError, RemoteExecutor
from stateset_agents.remote.registry import get_executor

CANARY_RESOURCE_PREFIX = "stateset-canary-"
_EPHEMERAL_RUNPOD_PREFIXES = (
    CANARY_RESOURCE_PREFIX,
    "stateset-sft-",
    "gpu-verify-",
    "stateset-conformance-",
)
_CREDENTIAL_ENV = {
    # CKS credentials normally live in kubeconfig and Nebius credentials in
    # the selected CLI profile, so neither has a mandatory environment key.
    # Their read-only probes report authentication errors directly.
    "coreweave": (),
    "fireworks": ("FIREWORKS_API_KEY", "FIREWORKS_ACCOUNT_ID"),
    "huggingface": ("HF_TOKEN",),
    "nebius": (),
    "river": ("RIVER_API_KEY",),
    "runpod": ("RUNPOD_API_KEY",),
    "tinker": ("TINKER_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
}


def _probe_huggingface(executor: RemoteExecutor) -> tuple[dict[str, Any], bool]:
    user = executor._client().whoami()  # type: ignore[attr-defined]
    return {"authenticated": True, "identity": _safe_value(user)}, True


def _probe_together(executor: RemoteExecutor) -> tuple[dict[str, Any], bool]:
    response = executor._client().models.list()  # type: ignore[attr-defined]
    models = list(islice(getattr(response, "data", response), 10))
    return {"authenticated": True, "models_observed": len(models)}, True


def _probe_tinker(executor: RemoteExecutor) -> tuple[dict[str, Any], bool]:
    _, client = executor._sdk()  # type: ignore[attr-defined]
    capabilities = client.get_server_capabilities()
    models = getattr(capabilities, "supported_models", ())
    return {
        "authenticated": True,
        "models_observed": len(models),
    }, True


@dataclass(frozen=True)
class ProviderCanaryResult:
    """Outcome of one provider's non-billable live probe."""

    provider: str
    status: str
    checked_at: str
    duration_ms: int
    checks: dict[str, Any] = field(default_factory=dict)
    cleanup_verified: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the provider passed every canary check."""
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result."""
        return asdict(self)


def missing_credentials(provider: str) -> list[str]:
    """Credential variables required by a provider but absent from the process."""
    names = _CREDENTIAL_ENV.get(provider, ())
    return [name for name in names if not os.environ.get(name, "").strip()]


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _safe_value(model_dump())
    return str(value)


def _resource_name(resource: Any) -> str:
    if isinstance(resource, dict):
        return str(resource.get("displayName") or resource.get("name") or "")
    return str(
        getattr(resource, "display_name", None) or getattr(resource, "name", None) or ""
    )


def _redact_error(exc: BaseException) -> str:
    text = str(exc)
    for names in _CREDENTIAL_ENV.values():
        for name in names:
            value = os.environ.get(name, "").strip()
            if value:
                text = text.replace(value, "[REDACTED]")
    return text


def _probe_river(executor: RemoteExecutor) -> tuple[dict[str, Any], bool]:
    client = executor._get_client()  # type: ignore[attr-defined]
    health = client.health_check()
    capabilities = client.get_capabilities()
    return {
        "health": _safe_value(health),
        "capabilities": _safe_value(capabilities),
        "billable_resources_created": 0,
    }, True


def _probe_fireworks(executor: RemoteExecutor) -> tuple[dict[str, Any], bool]:
    client = executor._get_client()  # type: ignore[attr-defined]
    account_id = executor.account_id  # type: ignore[attr-defined]
    models = list(islice(client.models.list(account_id=account_id, page_size=10), 10))
    jobs = list(
        islice(
            client.supervised_fine_tuning_jobs.list(
                account_id=account_id, page_size=100
            ),
            100,
        )
    )
    deployments = list(
        islice(client.deployments.list(account_id=account_id, page_size=100), 100)
    )
    leftovers = sorted(
        name
        for name in (_resource_name(item) for item in [*jobs, *deployments])
        if CANARY_RESOURCE_PREFIX in name
    )
    return {
        "models_observed": len(models),
        "jobs_observed": len(jobs),
        "deployments_observed": len(deployments),
        "canary_leftovers": leftovers,
        "billable_resources_created": 0,
    }, not leftovers


def _probe_runpod(executor: RemoteExecutor) -> tuple[dict[str, Any], bool]:
    api = executor._require_api()  # type: ignore[attr-defined]
    pods = api.list_pods()
    leases = executor.orphaned_leases()  # type: ignore[attr-defined]
    ephemeral_pods = sorted(
        str(pod.get("id") or pod.get("podId") or pod.get("name") or "")
        for pod in pods
        if str(pod.get("name") or "").startswith(_EPHEMERAL_RUNPOD_PREFIXES)
    )
    return {
        "pods_observed": len(pods),
        "local_cleanup_leases": len(leases),
        "canary_leftovers": ephemeral_pods,
        "ephemeral_training_leftovers": ephemeral_pods,
        "billable_resources_created": 0,
    }, not ephemeral_pods and not leases


def _probe_executor_canary(
    executor: RemoteExecutor,
) -> tuple[dict[str, Any], bool]:
    result = executor.canary()  # type: ignore[attr-defined]
    checks = _safe_value(result)
    assert isinstance(checks, dict)
    checks["billable_resources_created"] = 0
    return checks, bool(checks.get("authenticated"))


_PROBES = {
    "coreweave": _probe_executor_canary,
    "fireworks": _probe_fireworks,
    "huggingface": _probe_huggingface,
    "nebius": _probe_executor_canary,
    "river": _probe_river,
    "runpod": _probe_runpod,
    "tinker": _probe_tinker,
    "together": _probe_together,
}


def run_provider_canary(
    provider: str, *, executor: RemoteExecutor | None = None
) -> ProviderCanaryResult:
    """Run one provider's read-only live canary without raising SDK errors."""
    normalized = provider.strip().lower()
    if normalized not in _PROBES:
        raise RemoteExecutionError(
            f"provider {provider!r} has no live canary; available: "
            f"{', '.join(sorted(_PROBES))}",
            provider=normalized,
        )

    started = time.monotonic()
    checked_at = datetime.now(timezone.utc).isoformat()
    missing = missing_credentials(normalized) if executor is None else []
    if missing:
        return ProviderCanaryResult(
            provider=normalized,
            status="skipped",
            checked_at=checked_at,
            duration_ms=0,
            checks={"missing_credentials": missing},
            error=f"missing credential variables: {', '.join(missing)}",
        )

    try:
        resolved = executor or get_executor(normalized)
        checks, cleanup_verified = _PROBES[normalized](resolved)
        status = "passed" if cleanup_verified else "failed"
        error = None if cleanup_verified else "provider cleanup check failed"
    except Exception as exc:  # noqa: BLE001 - SDK errors become canary evidence
        checks = {}
        cleanup_verified = False
        status = "failed"
        error = _redact_error(exc)

    return ProviderCanaryResult(
        provider=normalized,
        status=status,
        checked_at=checked_at,
        duration_ms=max(0, int((time.monotonic() - started) * 1000)),
        checks=checks,
        cleanup_verified=cleanup_verified,
        error=error,
    )


def run_canary_matrix(providers: list[str]) -> list[ProviderCanaryResult]:
    """Run several provider probes in stable input order."""
    return [run_provider_canary(provider) for provider in providers]


__all__ = [
    "CANARY_RESOURCE_PREFIX",
    "ProviderCanaryResult",
    "missing_credentials",
    "run_canary_matrix",
    "run_provider_canary",
]
