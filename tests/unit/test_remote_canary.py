"""Tests for credential-aware, non-billable provider canaries."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import Any, cast

from stateset_agents.remote.canary import canary_source_identity, run_provider_canary
from stateset_agents.remote.executor import RemoteExecutor


def _executor(**attributes: Any) -> RemoteExecutor:
    return cast(RemoteExecutor, SimpleNamespace(**attributes))


def test_canary_source_identity_requires_clean_matching_checkout(
    monkeypatch, tmp_path
) -> None:
    responses = iter(
        (
            subprocess.CompletedProcess([], 0, "a" * 40 + "\n", ""),
            subprocess.CompletedProcess([], 0, "", ""),
        )
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(responses))
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)

    assert canary_source_identity(tmp_path) == ("a" * 40, True)

    responses = iter(
        (
            subprocess.CompletedProcess([], 0, "a" * 40 + "\n", ""),
            subprocess.CompletedProcess([], 0, " M file\n", ""),
        )
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(responses))
    assert canary_source_identity(tmp_path) == ("a" * 40, False)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("git")),
    )
    assert canary_source_identity(tmp_path) == (None, False)


def test_missing_credentials_skip_without_loading_sdk(monkeypatch) -> None:
    monkeypatch.delenv("RIVER_API_KEY", raising=False)

    result = run_provider_canary("river")

    assert result.status == "skipped"
    assert result.checks == {"missing_credentials": ["RIVER_API_KEY"]}
    assert result.cleanup_verified is False


def test_river_canary_checks_health_and_capabilities() -> None:
    client = SimpleNamespace(
        health_check=lambda: {"status": "ok"},
        get_capabilities=lambda: {"models": ["model-a"]},
    )

    result = run_provider_canary(
        "river", executor=_executor(_get_client=lambda: client)
    )

    assert result.ok
    assert result.cleanup_verified
    assert result.checks["health"] == {"status": "ok"}
    assert result.checks["billable_resources_created"] == 0


def test_fireworks_canary_detects_leftover_deployment() -> None:
    resources = SimpleNamespace(
        models=SimpleNamespace(list=lambda **_: [SimpleNamespace(name="model-a")]),
        supervised_fine_tuning_jobs=SimpleNamespace(list=lambda **_: []),
        deployments=SimpleNamespace(
            list=lambda **_: [SimpleNamespace(display_name="stateset-canary-leaked")]
        ),
    )
    executor = _executor(_get_client=lambda: resources, account_id="acct")

    result = run_provider_canary("fireworks", executor=executor)

    assert result.status == "failed"
    assert result.cleanup_verified is False
    assert result.checks["canary_leftovers"] == ["stateset-canary-leaked"]


def test_runpod_canary_passes_with_no_leases_or_canary_pods() -> None:
    api = SimpleNamespace(list_pods=lambda: [{"id": "pod-1", "name": "production"}])
    executor = _executor(
        _require_api=lambda: api,
        orphaned_leases=lambda: [],
    )

    result = run_provider_canary("runpod", executor=executor)

    assert result.ok
    assert result.checks["pods_observed"] == 1
    assert result.checks["billable_resources_created"] == 0


def test_runpod_canary_fails_on_ephemeral_training_pod() -> None:
    api = SimpleNamespace(
        list_pods=lambda: [{"id": "pod-2", "name": "stateset-sft-leaked"}]
    )
    executor = _executor(_require_api=lambda: api, orphaned_leases=lambda: [])

    result = run_provider_canary("runpod", executor=executor)

    assert result.status == "failed"
    assert result.checks["ephemeral_training_leftovers"] == ["pod-2"]


def test_provider_error_redacts_credentials(monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "secret-value")

    def fail() -> Any:
        raise RuntimeError("request rejected for secret-value")

    result = run_provider_canary(
        "runpod",
        executor=_executor(_require_api=fail, orphaned_leases=lambda: []),
    )

    assert result.status == "failed"
    assert result.error == "request rejected for [REDACTED]"
