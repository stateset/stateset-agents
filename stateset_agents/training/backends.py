"""Versioned contract for delegating one StateSet experiment to an engine.

The contract is intentionally smaller than any framework's configuration
surface.  It preserves the semantic inputs that must not drift across engines
(model/data revisions, environment, reward, seed, and algorithm config) while
letting an adapter own framework-specific launch details.

Secrets are not part of the contract.  Command adapters receive credentials
through their process environment and execute argument arrays without a shell.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

BACKEND_PROTOCOL = "stateset-training-backend-v1"
BACKEND_PROTOCOL_VERSION = 1

_CAPABILITIES = frozenset(
    {"async_rollouts", "distributed", "multi_turn", "multimodal", "tool_use"}
)
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_REQUIRED_COMMAND_PLACEHOLDERS = frozenset({"{request}", "{result}", "{output_dir}"})
_SENSITIVE_KEYS = frozenset(
    {"api_key", "authorization", "credential", "password", "secret", "token"}
)


class BackendError(RuntimeError):
    """Base error raised by the backend contract and registry."""


class BackendExecutionError(BackendError):
    """Raised when an engine command fails or violates its result contract."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("backend contract values must be JSON serializable") from exc


def _reject_secrets(value: Any, path: str = "experiment") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            components = set(normalized.split("_"))
            if (
                normalized in _SENSITIVE_KEYS
                or normalized.endswith("_api_key")
                or components.intersection(_SENSITIVE_KEYS)
            ):
                raise ValueError(
                    f"{path}.{key}: secrets must be supplied via environment"
                )
            _reject_secrets(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_secrets(child, f"{path}[{index}]")


def _reject_uri_credentials(value: str, field_name: str) -> None:
    parsed = urlparse(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            f"{field_name} must not embed credentials, query parameters, or fragments"
        )


def _nonempty(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


@dataclass(frozen=True)
class BackendCapabilities:
    """Capabilities an engine promises without changing experiment meaning."""

    algorithms: frozenset[str]
    features: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        algorithms = frozenset(
            _nonempty(value, "algorithm").lower() for value in self.algorithms
        )
        if not algorithms:
            raise ValueError("backend must support at least one algorithm")
        features = frozenset(
            _nonempty(value, "feature").lower() for value in self.features
        )
        unknown = sorted(features - _CAPABILITIES)
        if unknown:
            raise ValueError("unknown backend capabilities: " + ", ".join(unknown))
        object.__setattr__(self, "algorithms", algorithms)
        object.__setattr__(self, "features", features)

    def supports(self, experiment: TrainingExperiment) -> bool:
        """Whether the engine preserves this experiment's declared semantics."""
        return (
            experiment.algorithm in self.algorithms
            and experiment.requirements.issubset(self.features)
        )

    def to_dict(self) -> dict[str, list[str]]:
        """Return deterministic, JSON-safe capability metadata."""
        return {
            "algorithms": sorted(self.algorithms),
            "features": sorted(self.features),
        }


@dataclass(frozen=True)
class TrainingExperiment:
    """Backend-neutral semantic description of one training experiment."""

    algorithm: str
    model: str
    model_revision: str
    dataset_uri: str
    dataset_sha256: str
    output_dir: Path
    seed: int
    config: Mapping[str, Any]
    task: str = "unspecified"
    environment: Mapping[str, Any] = field(default_factory=dict)
    reward: Mapping[str, Any] = field(default_factory=dict)
    requirements: frozenset[str] = frozenset()
    protocol: str = BACKEND_PROTOCOL

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "algorithm", _nonempty(self.algorithm, "algorithm").lower()
        )
        for name in ("model", "model_revision", "dataset_uri", "task"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        _reject_uri_credentials(self.dataset_uri, "dataset_uri")
        digest = self.dataset_sha256.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("dataset_sha256 must be a 64-character SHA-256 digest")
        object.__setattr__(self, "dataset_sha256", digest)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        requirements = frozenset(
            _nonempty(value, "requirement").lower() for value in self.requirements
        )
        unknown = sorted(requirements - _CAPABILITIES)
        if unknown:
            raise ValueError("unknown experiment requirements: " + ", ".join(unknown))
        object.__setattr__(self, "requirements", requirements)
        for name in ("config", "environment", "reward"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping")
            _canonical_json(value)
            _reject_secrets(value, name)
        if self.protocol != BACKEND_PROTOCOL:
            raise ValueError(f"protocol must be {BACKEND_PROTOCOL!r}")

    def semantic_payload(self) -> dict[str, Any]:
        """Return the fields that must be identical across engine executions."""
        return {
            "protocol": self.protocol,
            "protocol_version": BACKEND_PROTOCOL_VERSION,
            "algorithm": self.algorithm,
            "model": self.model,
            "model_revision": self.model_revision,
            "dataset_uri": self.dataset_uri,
            "dataset_sha256": self.dataset_sha256,
            "seed": self.seed,
            "task": self.task,
            "config": dict(self.config),
            "environment": dict(self.environment),
            "reward": dict(self.reward),
            "requirements": sorted(self.requirements),
        }

    @property
    def sha256(self) -> str:
        """Canonical digest used to reject adapter-side semantic drift."""
        return hashlib.sha256(
            _canonical_json(self.semantic_payload()).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the request passed to a command adapter."""
        return {
            **self.semantic_payload(),
            "experiment_sha256": self.sha256,
            "output_dir": str(self.output_dir.resolve()),
        }


@dataclass(frozen=True)
class BackendResult:
    """Normalized result returned by every training-engine adapter."""

    backend: str
    backend_version: str
    experiment_sha256: str
    artifact_uri: str
    metrics: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    protocol_version: int = BACKEND_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _nonempty(self.backend, "backend").lower())
        object.__setattr__(
            self,
            "backend_version",
            _nonempty(self.backend_version, "backend_version"),
        )
        if not re.fullmatch(r"[0-9a-f]{64}", self.experiment_sha256):
            raise ValueError("experiment_sha256 must be a SHA-256 digest")
        _nonempty(self.artifact_uri, "artifact_uri")
        _reject_uri_credentials(self.artifact_uri, "artifact_uri")
        if self.protocol_version != BACKEND_PROTOCOL_VERSION:
            raise ValueError(
                f"protocol_version must be {BACKEND_PROTOCOL_VERSION}, "
                f"got {self.protocol_version}"
            )
        if not isinstance(self.metrics, Mapping) or not self.metrics:
            raise ValueError("metrics must be a non-empty mapping")
        for name, value in self.metrics.items():
            if not str(name).strip():
                raise ValueError("metric names must be non-empty")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"metric {name!r} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"metric {name!r} must be finite")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        _canonical_json(self.metadata)
        _reject_secrets(self.metadata, "metadata")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BackendResult:
        """Parse and validate an adapter result document."""
        try:
            return cls(
                backend=str(value["backend"]),
                backend_version=str(value["backend_version"]),
                experiment_sha256=str(value["experiment_sha256"]),
                artifact_uri=str(value["artifact_uri"]),
                metrics=value["metrics"],
                metadata=value.get("metadata", {}),
                protocol_version=value.get("protocol_version", 0),
            )
        except KeyError as exc:
            raise ValueError(f"backend result is missing {exc.args[0]!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result document."""
        return asdict(self)


class TrainingBackend:
    """Base interface implemented by native and external training engines."""

    name: str
    version: str
    capabilities: BackendCapabilities

    def validate(self, experiment: TrainingExperiment) -> None:
        """Reject unsupported semantics before allocating compute."""
        if not self.capabilities.supports(experiment):
            missing = sorted(experiment.requirements - self.capabilities.features)
            detail = (
                f"missing capabilities {missing}"
                if missing
                else f"unsupported algorithm {experiment.algorithm!r}"
            )
            raise BackendError(f"backend {self.name!r} cannot run experiment: {detail}")

    def run(self, experiment: TrainingExperiment) -> BackendResult:
        """Execute one experiment and return a normalized result."""
        raise NotImplementedError


class CommandTrainingBackend(TrainingBackend):
    """Execute an external engine adapter using a shell-free command contract."""

    def __init__(
        self,
        *,
        name: str,
        version: str,
        capabilities: BackendCapabilities,
        command: Sequence[str],
        timeout_seconds: int = 14400,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        normalized_name = _nonempty(name, "backend name").lower()
        if not _NAME_RE.fullmatch(normalized_name):
            raise ValueError(
                "backend name must contain only lowercase letters, digits, ._-"
            )
        if not command or any(
            not isinstance(part, str) or not part for part in command
        ):
            raise ValueError("command must be a non-empty sequence of strings")
        missing = _REQUIRED_COMMAND_PLACEHOLDERS.difference(command)
        if missing:
            raise ValueError(
                "command is missing required placeholders: "
                + ", ".join(sorted(missing))
            )
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        self.name = normalized_name
        self.version = _nonempty(version, "backend version")
        self.capabilities = capabilities
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds
        self.cwd = Path(cwd) if cwd is not None else None
        self.env = dict(env or {})
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.env.items()
        ):
            raise ValueError("env keys and values must be strings")

    @staticmethod
    def _format_command(command: Sequence[str], values: Mapping[str, str]) -> list[str]:
        formatted: list[str] = []
        for part in command:
            match = _PLACEHOLDER_RE.fullmatch(part)
            if match is None:
                formatted.append(part)
                continue
            name = match.group(1)
            if name not in values:
                raise BackendExecutionError(f"unknown command placeholder: {name}")
            formatted.append(values[name])
        return formatted

    def run(self, experiment: TrainingExperiment) -> BackendResult:
        """Write a request, run the adapter, and validate its result and artifact."""
        self.validate(experiment)
        output_dir = experiment.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        request_path = output_dir / "backend-request.json"
        result_path = output_dir / "backend-result.json"
        existing = [path for path in (request_path, result_path) if path.exists()]
        if existing:
            raise BackendExecutionError(f"refusing to overwrite existing {existing[0]}")
        request_path.write_text(
            json.dumps(experiment.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        command = self._format_command(
            self.command,
            {
                "request": str(request_path),
                "result": str(result_path),
                "output_dir": str(output_dir),
            },
        )
        process_env = os.environ.copy()
        process_env.update(self.env)
        try:
            completed = subprocess.run(
                command,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
                env=process_env,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", "replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            (output_dir / "backend-stdout.log").write_text(stdout, encoding="utf-8")
            (output_dir / "backend-stderr.log").write_text(stderr, encoding="utf-8")
            raise BackendExecutionError(
                f"backend {self.name!r} timed out after {self.timeout_seconds}s"
            ) from exc
        except OSError as exc:
            raise BackendExecutionError(
                f"could not launch backend {self.name!r}: {exc}"
            ) from exc
        (output_dir / "backend-stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (output_dir / "backend-stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            raise BackendExecutionError(
                f"backend {self.name!r} exited {completed.returncode}"
            )
        if not result_path.is_file():
            raise BackendExecutionError(
                f"backend {self.name!r} did not write {result_path}"
            )
        try:
            raw = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BackendExecutionError(f"{result_path} is not valid JSON") from exc
        if not isinstance(raw, Mapping):
            raise BackendExecutionError("backend result must be a JSON object")
        try:
            result = BackendResult.from_dict(raw)
        except ValueError as exc:
            raise BackendExecutionError(f"invalid backend result: {exc}") from exc
        if result.backend != self.name or result.backend_version != self.version:
            raise BackendExecutionError(
                "backend result identity does not match adapter"
            )
        if result.experiment_sha256 != experiment.sha256:
            raise BackendExecutionError(
                "backend result experiment digest does not match"
            )
        artifact = Path(result.artifact_uri)
        parsed = urlparse(result.artifact_uri)
        if artifact.is_absolute() or not parsed.scheme or parsed.scheme == "file":
            if parsed.scheme == "file":
                artifact = Path(parsed.path)
            if not artifact.is_absolute():
                artifact = output_dir / artifact
            artifact = artifact.resolve()
            if not artifact.is_relative_to(output_dir):
                raise BackendExecutionError(
                    "local artifact must stay inside output_dir"
                )
            if not artifact.exists():
                raise BackendExecutionError(
                    f"backend artifact does not exist: {artifact}"
                )
            if artifact.is_dir() and not any(
                path.is_file() for path in artifact.rglob("*")
            ):
                raise BackendExecutionError(f"backend artifact is empty: {artifact}")
        return result


class BackendRegistry:
    """Explicit registry for training engines; no optional SDK imports at listing time."""

    def __init__(self) -> None:
        self._backends: dict[str, TrainingBackend] = {}

    def register(self, backend: TrainingBackend, *, replace: bool = False) -> None:
        """Register an engine, rejecting accidental identity replacement."""
        name = _nonempty(backend.name, "backend name").lower()
        if not _NAME_RE.fullmatch(name):
            raise ValueError(
                "backend name must contain only lowercase letters, digits, ._-"
            )
        if name in self._backends and not replace:
            raise BackendError(f"backend {name!r} is already registered")
        self._backends[name] = backend

    def available(self) -> list[str]:
        """Return registered backend names in deterministic order."""
        return sorted(self._backends)

    def get(self, name: str) -> TrainingBackend:
        """Resolve a backend by normalized name."""
        normalized = name.strip().lower()
        try:
            return self._backends[normalized]
        except KeyError:
            available = ", ".join(self.available()) or "none"
            raise BackendError(
                f"unknown backend {name!r}; available: {available}"
            ) from None


__all__ = [
    "BACKEND_PROTOCOL",
    "BACKEND_PROTOCOL_VERSION",
    "BackendCapabilities",
    "BackendError",
    "BackendExecutionError",
    "BackendRegistry",
    "BackendResult",
    "CommandTrainingBackend",
    "TrainingBackend",
    "TrainingExperiment",
]
