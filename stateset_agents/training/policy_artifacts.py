"""Content-addressed policy artifacts for distributed rollout workers."""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .async_rollouts import AsyncRolloutError

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_URI_SCHEMES = frozenset({"az", "file", "gs", "https", "s3"})


class PolicyArtifactError(AsyncRolloutError):
    """Raised when a policy artifact is invalid or fails verification."""


class PolicyArtifactUnavailable(PolicyArtifactError):
    """Raised when no verified artifact is published for a policy version."""


@dataclass(frozen=True)
class PolicyArtifact:
    """Immutable, content-addressed weights for one exact policy version."""

    policy_version: int
    uri: str
    sha256: str
    size_bytes: int
    published_at: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.policy_version, bool)
            or not isinstance(self.policy_version, int)
            or self.policy_version < 0
        ):
            raise ValueError("policy_version must be a non-negative integer")
        if not isinstance(self.uri, str) or not self.uri.strip():
            raise ValueError("uri must be a non-empty string")
        parsed = urlsplit(self.uri)
        if parsed.scheme not in _SUPPORTED_URI_SCHEMES:
            raise ValueError(
                "uri scheme must be one of: "
                + ", ".join(sorted(_SUPPORTED_URI_SCHEMES))
            )
        if parsed.scheme == "file":
            if not parsed.path.startswith("/"):
                raise ValueError("file artifact URI must use an absolute path")
        elif not parsed.netloc:
            raise ValueError("remote artifact URI must include an authority or bucket")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("artifact URI must not contain embedded credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("artifact URI must not contain a query or fragment")
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.sha256
        ):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 1
        ):
            raise ValueError("size_bytes must be a positive integer")
        if (
            isinstance(self.published_at, bool)
            or not isinstance(self.published_at, (int, float))
            or not math.isfinite(float(self.published_at))
            or self.published_at < 0
        ):
            raise ValueError("published_at must be finite and non-negative")

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a stable JSON-compatible artifact descriptor."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PolicyArtifact:
        """Restore and validate an artifact descriptor."""
        expected = {
            "policy_version",
            "uri",
            "sha256",
            "size_bytes",
            "published_at",
        }
        if set(value) != expected:
            raise ValueError("policy artifact fields do not match schema")
        return cls(**dict(value))


def compute_policy_artifact_sha256(
    path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024
) -> str:
    """Hash one artifact file without loading model weights into memory."""
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise PolicyArtifactError(f"policy artifact is not a file: {artifact_path}")
    digest = hashlib.sha256()
    with artifact_path.open("rb") as artifact_file:
        while chunk := artifact_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_policy_artifact(
    path: str | os.PathLike[str], artifact: PolicyArtifact
) -> None:
    """Fail closed unless a downloaded artifact matches size and SHA-256."""
    if not isinstance(artifact, PolicyArtifact):
        raise TypeError("artifact must be PolicyArtifact")
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise PolicyArtifactError(f"policy artifact is not a file: {artifact_path}")
    actual_size = artifact_path.stat().st_size
    if actual_size != artifact.size_bytes:
        raise PolicyArtifactError(
            f"policy artifact size mismatch: expected {artifact.size_bytes}, "
            f"got {actual_size}"
        )
    actual_digest = compute_policy_artifact_sha256(artifact_path)
    if actual_digest != artifact.sha256:
        raise PolicyArtifactError("policy artifact SHA-256 mismatch")


__all__ = [
    "PolicyArtifact",
    "PolicyArtifactError",
    "PolicyArtifactUnavailable",
    "compute_policy_artifact_sha256",
    "verify_policy_artifact",
]
