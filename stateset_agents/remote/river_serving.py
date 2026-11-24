"""Managed serving, deployments, and streaming inference for River AI checkpoints.

Turns hosted ``river://`` checkpoint pointers (or local ``river_checkpoint.json``
directories) into first-class servable and deployable StateSet models using
River SDK 0.11's serving APIs:

* Dedicated deployments (``create_deployment``, ``wait_for_deployment``,
  ``get_deployment``, ``list_deployments``, ``scale_on_target``,
  ``delete_deployment``, ``get_deployment_usage``).
* Promoted streaming replicas (``promote_streaming_replica``,
  ``get_streaming_replica``, ``chat_complete_stream``).
* Direct checkpoint chat inference (``chat_complete_from_checkpoint``,
  ``chat_complete``).

All SDK imports are lazy; offline callers and tests may inject any duck-typed
client without installing ``river-client`` or opening network connections.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from stateset_agents.remote.deployment import (
    DeploymentHandle,
    DeploymentSpec,
    InferenceDeploymentProvider,
)
from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.river import (
    RIVER_API_KEY_ENV,
    _checkpoint_from_pointer,
    _inference_checkpoint,
)

__all__ = [
    "RiverDeploymentManager",
    "RiverDeploymentProvider",
    "RiverInferenceClient",
]


def _serialize_sdk_obj(obj: Any) -> Any:
    """Convert River SDK dataclasses/objects into JSON-safe plain Python values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (frozenset, set)):
        return sorted(_serialize_sdk_obj(item) for item in obj)
    if isinstance(obj, (list, tuple)):
        return [_serialize_sdk_obj(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _serialize_sdk_obj(v) for k, v in obj.items()}
    if is_dataclass(obj) and not isinstance(obj, type):
        return _serialize_sdk_obj(asdict(obj))
    if hasattr(obj, "__dict__"):
        return {
            str(k): _serialize_sdk_obj(v)
            for k, v in vars(obj).items()
            if not str(k).startswith("_")
        }
    return str(obj)


def _redact_secret(text: str) -> str:
    """Strip any configured RIVER_API_KEY value from diagnostic strings."""
    key = os.environ.get(RIVER_API_KEY_ENV, "").strip()
    if key:
        return text.replace(key, "[REDACTED]")
    return text


def _checkpoint_base_model(checkpoint_ref: str | Path) -> str:
    """Read declared base_model from a local pointer directory when available."""
    raw = str(checkpoint_ref).strip()
    if not raw or raw.startswith("river://"):
        return ""
    pointer_path = Path(raw) / "river_checkpoint.json"
    if not pointer_path.exists():
        return ""
    try:
        data = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    base = data.get("base_model") if isinstance(data, dict) else None
    return str(base).strip() if isinstance(base, str) else ""


def _parse_chat_result(result: Any) -> dict[str, Any]:
    """Normalize River ChatCompleteResult into an OpenAI-compatible dict."""
    if isinstance(result, dict):
        return result
    raw_json = getattr(result, "response_json", None)
    status_code = getattr(result, "status_code", 200)
    if isinstance(raw_json, str):
        try:
            parsed = json.loads(raw_json)
        except ValueError as exc:
            raise RemoteExecutionError(
                f"River returned invalid chat JSON (HTTP {status_code})",
                provider="river",
            ) from exc
        if isinstance(parsed, dict):
            parsed.setdefault("status_code", status_code)
            return parsed
        return {"status_code": status_code, "data": parsed}
    serialized = _serialize_sdk_obj(result)
    if isinstance(serialized, dict):
        return serialized
    return {"status_code": status_code, "content": str(serialized)}


def _extract_assistant_text(payload: dict[str, Any]) -> str:
    """Extract assistant text from a normalized OpenAI-style chat response."""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return str(msg["content"])
            delta = first.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                return str(delta["content"])
            if isinstance(first.get("text"), str):
                return str(first["text"])
    if isinstance(payload.get("content"), str):
        return str(payload["content"])
    return ""


class RiverDeploymentManager:
    """Lifecycle and serving manager for River AI deployments and checkpoints.

    Follows the same ownership contract as :class:`RiverExecutor`: injected
    clients remain owned by the caller; internally constructed clients are
    closed deterministically when used as a context manager or via :meth:`close`.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._api_key = api_key
        self._timeout = timeout

    def __enter__(self) -> RiverDeploymentManager:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import river_client
        except ImportError as exc:
            raise RemoteExecutionError.wrap(
                exc,
                "the River SDK is not installed. Install `stateset-agents[river]` "
                "on Python >=3.12 or pass an explicit client.",
                provider="river",
            ) from exc
        key = (
            self._api_key
            if self._api_key is not None
            else os.environ.get(RIVER_API_KEY_ENV, "")
        ).strip()
        if not key:
            raise RemoteExecutionError(
                f"{RIVER_API_KEY_ENV} is not set. Export your River API key before "
                "managing deployments or querying hosted checkpoints.",
                provider="river",
            )
        kwargs: dict[str, Any] = {"api_key": key}
        if self._timeout is not None:
            kwargs["timeout"] = float(self._timeout)
        self._client = river_client.Client(**kwargs)
        return self._client

    def close(self) -> None:
        """Close an internally owned River SDK client if open."""
        if self._owns_client and self._client is not None:
            client, self._client = self._client, None
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def _call(self, label: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except RemoteExecutionError:
            raise
        except Exception as exc:
            raise RemoteExecutionError(
                f"River {label} failed ({type(exc).__name__}): {_redact_secret(str(exc))}",
                provider="river",
            ) from exc

    # -- dedicated deployments ---------------------------------------------

    def create_deployment(
        self,
        checkpoint: str | Path,
        *,
        unified_replicas: int | None = None,
        prefill_replicas: int | None = None,
        decode_replicas: int | None = None,
        idempotency_key: str | None = None,
        wait: bool = False,
        wait_timeout: float = 1800.0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Deploy a ``river://`` checkpoint or pointer directory as a live endpoint."""
        uri = _checkpoint_from_pointer(checkpoint)
        if not uri:
            raise RemoteExecutionError(
                "create_deployment requires a river:// URI or checkpoint directory",
                provider="river",
            )
        client = self._get_client()
        ckpt = _inference_checkpoint(client, uri)
        kwargs: dict[str, Any] = {
            "unified_replicas": unified_replicas,
            "prefill_replicas": prefill_replicas,
            "decode_replicas": decode_replicas,
            "idempotency_key": idempotency_key,
            "wait": wait,
            "wait_timeout": wait_timeout,
            "timeout": timeout,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        dep = self._call("create_deployment", client.create_deployment, ckpt, **kwargs)
        return dict(_serialize_sdk_obj(dep))

    def list_deployments(
        self, *, include_deleted: bool = False, timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """List River deployments visible to the authenticated account."""
        client = self._get_client()
        kwargs: dict[str, Any] = {"include_deleted": include_deleted}
        if timeout is not None:
            kwargs["timeout"] = timeout
        deps = self._call("list_deployments", client.list_deployments, **kwargs)
        return [dict(_serialize_sdk_obj(dep)) for dep in deps]

    def get_deployment(
        self, deployment_id: str, *, timeout: float | None = None
    ) -> dict[str, Any]:
        """Fetch current status and replica state for one River deployment."""
        if not deployment_id.strip():
            raise ValueError("deployment_id must be non-empty")
        client = self._get_client()
        kwargs: dict[str, Any] = {"timeout": timeout} if timeout is not None else {}
        dep = self._call(
            "get_deployment", client.get_deployment, deployment_id.strip(), **kwargs
        )
        return dict(_serialize_sdk_obj(dep))

    def wait_for_deployment(
        self,
        deployment_id: str,
        *,
        timeout: float = 1800.0,
        poll_interval: float = 5.0,
        until: tuple[str, ...] = ("ready", "degraded"),
    ) -> dict[str, Any]:
        """Wait until a River deployment reaches one of the target phases."""
        if not deployment_id.strip():
            raise ValueError("deployment_id must be non-empty")
        client = self._get_client()
        dep = self._call(
            "wait_for_deployment",
            client.wait_for_deployment,
            deployment_id.strip(),
            timeout=timeout,
            poll_interval=poll_interval,
            until=until,
        )
        return dict(_serialize_sdk_obj(dep))

    def scale_deployment(
        self,
        deployment_id: str,
        *,
        unified_replicas: int | None = None,
        prefill_replicas: int | None = None,
        decode_replicas: int | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Scale unified or disaggregated prefill/decode replicas on a deployment."""
        if not deployment_id.strip():
            raise ValueError("deployment_id must be non-empty")
        if (
            unified_replicas is None
            and prefill_replicas is None
            and decode_replicas is None
        ):
            raise ValueError(
                "Specify at least one of unified_replicas, prefill_replicas, or decode_replicas"
            )
        client = self._get_client()
        kwargs = {
            k: v
            for k, v in {
                "unified_replicas": unified_replicas,
                "prefill_replicas": prefill_replicas,
                "decode_replicas": decode_replicas,
                "idempotency_key": idempotency_key,
                "timeout": timeout,
            }.items()
            if v is not None
        }
        dep = self._call(
            "scale_on_target",
            client.scale_on_target,
            deployment_id.strip(),
            **kwargs,
        )
        return dict(_serialize_sdk_obj(dep))

    def delete_deployment(
        self,
        deployment_id: str,
        *,
        idempotency_key: str | None = None,
        wait: bool = False,
        wait_timeout: float = 600.0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Delete a River deployment so dedicated serving replicas stop billing."""
        if not deployment_id.strip():
            raise ValueError("deployment_id must be non-empty")
        client = self._get_client()
        kwargs = {
            k: v
            for k, v in {
                "idempotency_key": idempotency_key,
                "wait": wait,
                "wait_timeout": wait_timeout,
                "timeout": timeout,
            }.items()
            if v is not None
        }
        dep = self._call(
            "delete_deployment",
            client.delete_deployment,
            deployment_id.strip(),
            **kwargs,
        )
        return dict(_serialize_sdk_obj(dep))

    def get_deployment_usage(
        self, deployment_id: str, *, timeout: float | None = None
    ) -> dict[str, Any]:
        """Read metered usage metrics for a River deployment."""
        if not deployment_id.strip():
            raise ValueError("deployment_id must be non-empty")
        client = self._get_client()
        kwargs: dict[str, Any] = {"timeout": timeout} if timeout is not None else {}
        usage = self._call(
            "get_deployment_usage",
            client.get_deployment_usage,
            deployment_id.strip(),
            **kwargs,
        )
        return dict(_serialize_sdk_obj(usage))

    # -- streaming replicas ------------------------------------------------

    def promote_streaming_replica(
        self,
        checkpoint: str | Path,
        model: str,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Promote a trained checkpoint to a low-latency streaming replica alias."""
        if not model.strip():
            raise ValueError("model alias must be non-empty")
        uri = _checkpoint_from_pointer(checkpoint)
        if not uri:
            raise RemoteExecutionError(
                "promote_streaming_replica requires a river:// URI or checkpoint directory",
                provider="river",
            )
        client = self._get_client()
        ckpt = _inference_checkpoint(client, uri)
        kwargs: dict[str, Any] = {"timeout": timeout} if timeout is not None else {}
        replica = self._call(
            "promote_streaming_replica",
            client.promote_streaming_replica,
            ckpt,
            model.strip(),
            **kwargs,
        )
        return dict(_serialize_sdk_obj(replica))

    def get_streaming_replica(
        self, model: str, *, timeout: float | None = None
    ) -> dict[str, Any] | None:
        """Inspect the active promoted streaming replica for ``model``, if any."""
        if not model.strip():
            raise ValueError("model alias must be non-empty")
        client = self._get_client()
        kwargs: dict[str, Any] = {"timeout": timeout} if timeout is not None else {}
        replica = self._call(
            "get_streaming_replica",
            client.get_streaming_replica,
            model.strip(),
            **kwargs,
        )
        return None if replica is None else dict(_serialize_sdk_obj(replica))

    # -- hosted chat & streaming completion --------------------------------

    def chat_complete(
        self,
        messages: list[dict[str, Any]],
        *,
        checkpoint: str | Path | None = None,
        base_model: str = "",
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a non-streaming chat completion on a River checkpoint or base model."""
        if not messages:
            raise ValueError("messages must be non-empty")
        client = self._get_client()
        if checkpoint is not None:
            uri = _checkpoint_from_pointer(checkpoint)
            if not uri:
                raise RemoteExecutionError(
                    "Invalid River checkpoint reference", provider="river"
                )
            resolved_base = base_model.strip() or _checkpoint_base_model(checkpoint)
            raw = self._call(
                "chat_complete_from_checkpoint",
                client.chat_complete_from_checkpoint,
                messages,
                checkpoint_path=uri,
                base_model=resolved_base,
                timeout=timeout,
                **kwargs,
            )
            payload = _parse_chat_result(raw)
            payload.setdefault("checkpoint", uri)
            if resolved_base:
                payload.setdefault("base_model", resolved_base)
            payload.setdefault("text", _extract_assistant_text(payload))
            return payload

        if not base_model.strip():
            raise ValueError(
                "Provide either checkpoint or base_model for River chat completion"
            )
        raw = self._call(
            "chat_complete",
            client.chat_complete,
            messages,
            base_model=base_model.strip(),
            timeout=timeout,
            **kwargs,
        )
        payload = _parse_chat_result(raw)
        payload.setdefault("base_model", base_model.strip())
        payload.setdefault("text", _extract_assistant_text(payload))
        return payload

    def chat_complete_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        timeout: float | None = None,
        on_not_ready: str = "raise",
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream chat completion chunks from a promoted River streaming replica."""
        if not messages:
            raise ValueError("messages must be non-empty")
        if not model.strip():
            raise ValueError("model alias must be non-empty for streaming completion")
        client = self._get_client()
        stream = self._call(
            "chat_complete_stream",
            client.chat_complete_stream,
            messages,
            model=model.strip(),
            timeout=timeout,
            on_not_ready=on_not_ready,
            **kwargs,
        )
        for chunk in stream:
            yield dict(_serialize_sdk_obj(chunk))


class RiverDeploymentProvider(InferenceDeploymentProvider):
    """Managed :class:`InferenceDeploymentProvider` backed by River AI deployments."""

    name = "river"

    def __init__(self, client: Any = None, *, api_key: str | None = None) -> None:
        self._client = client
        self._api_key = api_key

    def _manager(self) -> RiverDeploymentManager:
        return RiverDeploymentManager(client=self._client, api_key=self._api_key)

    def deploy(self, spec: DeploymentSpec) -> DeploymentHandle:
        """Create a River deployment from a ``DeploymentSpec``."""
        with self._manager() as manager:
            rc = spec.runtime_config or {}
            prefill = int(rc["prefill_replicas"]) if "prefill_replicas" in rc else None
            decode = int(rc["decode_replicas"]) if "decode_replicas" in rc else None
            unified = (
                int(rc["unified_replicas"])
                if "unified_replicas" in rc
                else (
                    None
                    if prefill is not None or decode is not None
                    else spec.min_replicas
                )
            )
            dep = manager.create_deployment(
                spec.weights_uri,
                unified_replicas=unified,
                prefill_replicas=prefill,
                decode_replicas=decode,
                idempotency_key=rc.get("idempotency_key") or f"stateset-{spec.name}",
                wait=str(rc.get("wait", "")).lower() in {"1", "true", "yes"},
            )
            dep_id = str(dep.get("id") or "")
            if not dep_id:
                raise RemoteExecutionError(
                    "River create_deployment returned no deployment id",
                    provider=self.name,
                )
            return DeploymentHandle(
                provider=self.name,
                deployment_id=dep_id,
                model_name=str(dep.get("model") or spec.model_name),
                endpoint=str(dep["base_url"]) if dep.get("base_url") else None,
            )

    def status(self, handle: DeploymentHandle) -> dict[str, Any]:
        """Return River deployment status plus metered usage when available."""
        with self._manager() as manager:
            state = manager.get_deployment(handle.deployment_id)
            try:
                state["usage"] = manager.get_deployment_usage(handle.deployment_id)
            except RemoteExecutionError:
                state["usage"] = None
            return state

    def delete(self, handle: DeploymentHandle) -> None:
        """Delete a River deployment."""
        with self._manager() as manager:
            manager.delete_deployment(handle.deployment_id)


class RiverInferenceClient:
    """Ergonomic serving client bound to a ``river://`` checkpoint or pointer dir."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        base_model: str = "",
        streaming_model: str | None = None,
        client: Any = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        uri = _checkpoint_from_pointer(checkpoint)
        if not uri:
            raise RemoteExecutionError(
                "RiverInferenceClient requires a river:// URI or pointer directory",
                provider="river",
            )
        self.checkpoint_uri = uri
        self.base_model = base_model.strip() or _checkpoint_base_model(checkpoint)
        self.streaming_model = streaming_model
        self._manager = RiverDeploymentManager(
            client=client, api_key=api_key, timeout=timeout
        )

    def __enter__(self) -> RiverInferenceClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._manager.close()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run chat completion against the bound River checkpoint."""
        return self._manager.chat_complete(
            messages,
            checkpoint=self.checkpoint_uri,
            base_model=self.base_model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Convenience single-prompt completion returning assistant text."""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        res = self.chat(
            messages, max_tokens=max_tokens, temperature=temperature, **kwargs
        )
        return str(res.get("text", ""))

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream chunks from a promoted streaming replica for this checkpoint."""
        target = (model or self.streaming_model or "").strip()
        if not target:
            raise ValueError(
                "Pass model=... or configure streaming_model= when streaming via River"
            )
        return self._manager.chat_complete_stream(messages, model=target, **kwargs)
