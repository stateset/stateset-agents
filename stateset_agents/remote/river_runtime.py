"""Inspect native River prerequisites without opening provider resources."""

from __future__ import annotations

import importlib
import inspect
import math
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

from stateset_agents.remote.executor import RemoteExecutionError

SUPPORTED_RIVER_SDK = ">=0.11.0,<0.12"


def list_accessible_models(client: Any) -> tuple[str, ...]:
    """Read fresh account model names, rejecting unavailable or malformed data."""
    get_capabilities = getattr(client, "get_capabilities", None)
    if not callable(get_capabilities):
        raise RemoteExecutionError(
            "River client lacks get_capabilities(); install the supported River SDK",
            provider="river",
        )
    try:
        models = get_capabilities()
    except Exception as exc:
        # SDK error text may contain credentials or request metadata. Preserve
        # the cause for debugging without copying it into the public message.
        raise RemoteExecutionError(
            f"River model access check failed ({type(exc).__name__}); "
            "check credentials and connectivity before retrying",
            provider="river",
        ) from exc
    if not isinstance(models, (list, tuple)) or any(
        not isinstance(model, str) or not model or model != model.strip()
        for model in models
    ):
        raise RemoteExecutionError(
            "River returned malformed model capabilities; no session was opened",
            provider="river",
        )
    return tuple(dict.fromkeys(models))


def require_model_access(client: Any, base_model: str) -> tuple[str, ...]:
    """Check advertised access before tokenization or opening a training session.

    Access does not establish training correctness, capacity, or funding.
    Never substitute a static catalog for an account response.
    """
    if not isinstance(base_model, str) or not base_model.strip():
        raise ValueError("base_model must be non-empty text")
    models = list_accessible_models(client)
    if base_model not in models:
        raise RemoteExecutionError(
            f"River account does not advertise access to {base_model!r}; "
            "check client.get_capabilities() or request access from River",
            provider="river",
        )
    return models


def river_preflight(
    base_models: Sequence[str] = (),
    *,
    live: bool = False,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    """Report local prerequisites and optionally make one account-access query.

    Offline mode never constructs a client. Live mode owns and closes its
    client, disables retries, and applies a finite SDK request timeout. Neither
    mode loads a tokenizer, samples tokens, or creates provider resources.
    ``passed`` covers only the checks requested, not training readiness.
    """
    if (
        not isinstance(base_models, Sequence)
        or isinstance(base_models, (str, bytes))
        or any(
            not isinstance(model, str) or not model or model != model.strip()
            for model in base_models
        )
    ):
        raise ValueError("base_models must contain non-empty, trimmed model IDs")
    if type(live) is not bool:
        raise ValueError("live must be boolean")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a finite positive number")
    requested = list(dict.fromkeys(base_models))
    checks = inspect_native_runtime()
    issues = native_runtime_issues(checks)
    account: dict[str, Any] = {"status": "not_checked", "available_models": None}
    model_access: dict[str, bool | None] = dict.fromkeys(requested)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "stateset-river-preflight",
        "provider": "river",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "mode": "live" if live else "local",
        "passed": False,
        "checks": checks,
        "account": account,
        "model_access": model_access,
        "issues": issues,
        "request_timeout_seconds": float(timeout_seconds) if live else None,
        "billable_resources_created": 0,
        "unverified": [
            "tokenizer_compatibility",
            "training_correctness",
            "model_quality",
            "capacity_and_funding",
            "hosted_chat_and_streaming",
        ],
    }
    if live and not issues:
        client = None
        try:
            river = importlib.import_module("river_client")
            client = river.Client(
                api_key=os.environ.get("RIVER_API_KEY", "").strip(),
                timeout=float(timeout_seconds),
                enable_retries=False,
            )
            models = list_accessible_models(client)
            account.update(status="checked", available_models=list(models))
            for model in requested:
                model_access[model] = model in models
                if model not in models:
                    issues.append(
                        f"River account does not advertise access to {model!r}"
                    )
        except RemoteExecutionError as exc:
            account["status"] = "failed"
            issues.append(str(exc))
        except Exception as exc:
            account["status"] = "failed"
            issues.append(f"Live River preflight failed ({type(exc).__name__})")
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception as exc:
                    issues.append(f"River client cleanup failed ({type(exc).__name__})")
    report["passed"] = not issues
    return report


def inspect_native_runtime() -> dict[str, Any]:
    """Inspect the active interpreter and SDK API without constructing a client."""
    checks: dict[str, Any] = {
        "python": {
            "version": sys.version.split()[0],
            "passed": sys.version_info >= (3, 12),
        },
        "credentials": {
            "configured": bool(os.environ.get("RIVER_API_KEY", "").strip())
        },
        "river_sdk": {"version": None, "passed": False},
    }
    sdk = checks["river_sdk"]
    try:
        sdk["version"] = metadata.version("river-client")
        # packaging is installed with River's transformers dependency. Keep the
        # import optional so offline dataset preparation works without extras.
        from packaging.specifiers import SpecifierSet

        if sdk["version"] not in SpecifierSet(SUPPORTED_RIVER_SDK):
            sdk["reason"] = "Install the supported stateset-agents[river] extra"
            return checks
        river = importlib.import_module("river_client")
        rl = importlib.import_module("river_client.rl")
        renderers = importlib.import_module("river_client.renderers")
        required = (
            "Env",
            "AsyncTrainer",
            "RolloutEngine",
            "CheckpointSampler",
            "Budget",
            "Schedule",
            "Adam",
            "GroupCentered",
            "Truncation",
            "GroupCompletion",
            "Checkpointing",
            "Evaluator",
        )
        missing = [name for name in required if not callable(getattr(rl, name, None))]
        for name in ("Client", "LoraConfig", "Checkpoint"):
            if not callable(getattr(river, name, None)):
                missing.append(name)
        if not callable(getattr(renderers, "get_renderer", None)):
            missing.append("get_renderer")
        if missing:
            sdk["reason"] = "Missing SDK capabilities: " + ", ".join(missing)
        else:
            callback = inspect.signature(rl.AsyncTrainer.run).parameters.get(
                "after_recovery"
            )
            if callback is None or callback.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                sdk["reason"] = (
                    "SDK trainer lacks the required keyword recovery callback"
                )
            else:
                sdk["passed"] = True
    except metadata.PackageNotFoundError:
        sdk["reason"] = "river-client is not installed in this interpreter"
    except Exception as exc:
        # Import failures are diagnostic evidence, not permission to run. Avoid
        # echoing arbitrary SDK exception text or credential values into reports.
        sdk["reason"] = f"SDK inspection failed ({type(exc).__name__})"
    return checks


def native_runtime_issues(checks: dict[str, Any]) -> list[str]:
    """Describe failed local prerequisites without exposing credentials."""
    issues = []
    if not checks["python"]["passed"]:
        issues.append("River requires Python 3.12 or newer")
    if not checks["river_sdk"]["passed"]:
        issues.append(checks["river_sdk"]["reason"])
    if not checks["credentials"]["configured"]:
        issues.append("RIVER_API_KEY is not configured in this process environment")
    return issues


def require_native_runtime() -> None:
    """Reject unsupported local runtimes before loading models or opening clients.

    Passing checks only local prerequisites, not authentication, funding,
    connectivity, or permission to spend.
    """
    issues = native_runtime_issues(inspect_native_runtime())
    if issues:
        raise RuntimeError("Native River runtime is not ready: " + "; ".join(issues))
