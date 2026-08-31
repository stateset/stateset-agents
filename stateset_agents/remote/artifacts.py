"""Provider-neutral S3 artifact transport for remote training jobs.

CoreWeave AI Object Storage and Nebius Object Storage both expose an S3 API.
Keeping that transport here lets executors move the exact same dataset and
adapter bytes without embedding cloud credentials in a :class:`RemoteJobSpec`.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from stateset_agents.remote.executor import RemoteExecutionError

__all__ = ["S3ArtifactStore", "parse_s3_uri"]


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return ``(bucket, key)`` for a validated ``s3://`` URI."""
    parsed = urlparse(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"expected s3://bucket/key, got {uri!r}")
    key = parsed.path.lstrip("/")
    if any(part in {".", ".."} for part in PurePosixPath(key).parts):
        raise ValueError("S3 object keys cannot contain traversal segments")
    return parsed.netloc, key


class S3ArtifactStore:
    """Upload and retrieve job inputs/outputs through an S3-compatible API."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        endpoint_url: str | None = None,
        region_name: str | None = None,
    ) -> None:
        self._client = client
        self.endpoint_url = endpoint_url
        self.region_name = region_name

    def _require_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RemoteExecutionError.wrap(
                exc,
                "S3 artifact transport requires boto3; install "
                "'stateset-agents[cloud]'",
                provider="s3",
            ) from exc
        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url
            or os.environ.get("STATESET_S3_ENDPOINT_URL")
            or None,
            region_name=self.region_name or os.environ.get("AWS_REGION") or "us-east-1",
        )
        return self._client

    def upload_file(self, source: Path, uri: str) -> None:
        """Upload one local file to ``uri``."""
        bucket, key = parse_s3_uri(uri)
        if not key:
            raise ValueError("an artifact file URI must include an object key")
        try:
            self._require_client().upload_file(str(source), bucket, key)
        except Exception as exc:  # noqa: BLE001 - SDK-specific exception tree
            raise RemoteExecutionError.wrap(
                exc, f"could not upload artifact to {uri}", provider="s3"
            ) from exc

    def upload_directory(self, source: Path, uri: str) -> list[str]:
        """Recursively upload ``source`` beneath an S3 prefix."""
        bucket, prefix = parse_s3_uri(uri)
        uploaded: list[str] = []
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            if path.is_symlink():
                raise RemoteExecutionError(
                    f"refusing to upload symlinked artifact: {path}", provider="s3"
                )
            relative = path.relative_to(source).as_posix()
            key = "/".join(part for part in (prefix.rstrip("/"), relative) if part)
            target = f"s3://{bucket}/{key}"
            self.upload_file(path, target)
            uploaded.append(target)
        return uploaded

    def download_file(self, uri: str, destination: Path) -> Path:
        """Download one object, creating its destination directory."""
        bucket, key = parse_s3_uri(uri)
        if not key:
            raise ValueError("an artifact file URI must include an object key")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._require_client().download_file(bucket, key, str(destination))
        except Exception as exc:  # noqa: BLE001
            raise RemoteExecutionError.wrap(
                exc, f"could not download artifact from {uri}", provider="s3"
            ) from exc
        return destination

    def download_prefix(self, uri: str, destination: Path) -> list[Path]:
        """Download every object beneath ``uri`` without allowing traversal."""
        bucket, prefix = parse_s3_uri(uri)
        normalized = prefix.rstrip("/")
        if not normalized:
            raise ValueError("an artifact prefix URI must include an object prefix")
        client = self._require_client()
        destination_root = destination.resolve()
        written: list[Path] = []
        try:
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=normalized + "/")
            for page in pages:
                for row in page.get("Contents", []):
                    key = str(row["Key"])
                    relative = PurePosixPath(key).relative_to(normalized)
                    if not relative.parts or any(
                        part in {"", ".", ".."} for part in relative.parts
                    ):
                        raise RemoteExecutionError(
                            f"unsafe artifact key returned by object store: {key!r}",
                            provider="s3",
                        )
                    target = destination.joinpath(*relative.parts)
                    if not target.resolve(strict=False).is_relative_to(
                        destination_root
                    ):
                        raise RemoteExecutionError(
                            f"artifact path escapes destination: {key!r}",
                            provider="s3",
                        )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    client.download_file(bucket, key, str(target))
                    written.append(target)
        except RemoteExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RemoteExecutionError.wrap(
                exc, f"could not download artifact prefix {uri}", provider="s3"
            ) from exc
        return written

    def delete_prefix(self, uri: str) -> None:
        """Delete all objects beneath a job-owned prefix."""
        bucket, prefix = parse_s3_uri(uri)
        normalized = prefix.rstrip("/")
        if not normalized:
            raise ValueError("refusing to delete an entire artifact bucket")
        client = self._require_client()
        try:
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=normalized + "/"):
                objects = [{"Key": row["Key"]} for row in page.get("Contents", [])]
                if objects:
                    client.delete_objects(Bucket=bucket, Delete={"Objects": objects})
        except Exception as exc:  # noqa: BLE001
            raise RemoteExecutionError.wrap(
                exc, f"could not delete artifact prefix {uri}", provider="s3"
            ) from exc
