"""Portable memory snapshots with revision checks and explicit publication receipts."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import numpy as np

from ...utils.async_calls import run_async_owned, run_sync_owned
from ...utils.snapshot_files import canonical_json as _canonical
from ...utils.snapshot_files import json_value as _json_value
from ...utils.snapshot_files import open_regular as _open_regular
from ...utils.snapshot_files import publish_bytes
from ...utils.snapshot_files import sync_directory as _sync_directory
from ..agent_initialization import BackendInitializationState
from .vector_memory import MemoryEntry, VectorMemory

_T = TypeVar("_T")


class MemorySnapshotConflict(RuntimeError):
    """The file revision, live memory, or operation ownership has changed."""


def _keys(value: Any, expected: set[str]) -> None:
    if type(value) is not dict or set(value) != expected:
        raise ValueError("Invalid memory snapshot fields")


def _integer(value: Any) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("Snapshot counts must be nonnegative integers")


def _timestamp(value: Any) -> datetime:
    if type(value) is not str:
        raise ValueError("Snapshot timestamps must be ISO datetime strings")
    parsed = datetime.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("Snapshot timestamps must use canonical datetime.isoformat()")
    return parsed


def _date(value: Any) -> str:
    if type(value) is not datetime:
        raise ValueError("Snapshot timestamps must be datetime values")
    result = value.isoformat()
    _timestamp(result)
    return result


def _validate_payload(payload: Any) -> None:
    _keys(payload, {"embedding_model", "max_entries", "entries"})
    if (
        type(payload["embedding_model"]) is not str
        or not payload["embedding_model"].strip()
    ):
        raise ValueError("Snapshot embedding_model must be a nonempty string")
    _integer(payload["max_entries"])
    entries = payload["entries"]
    if type(entries) is not list or len(entries) > payload["max_entries"]:
        raise ValueError("Snapshot entries exceed the declared capacity")
    for entry in entries:
        _keys(
            entry,
            {
                "content",
                "context",
                "timestamp",
                "importance",
                "access_count",
                "last_accessed",
            },
        )
        if type(entry["content"]) is not str or type(entry["context"]) is not dict:
            raise ValueError(
                "Snapshot entries require text content and a context object"
            )
        _json_value(entry["context"])
        if type(entry["importance"]) not in (int, float) or not math.isfinite(
            entry["importance"]
        ):
            raise ValueError("Snapshot importance must be finite")
        _integer(entry["access_count"])
        _timestamp(entry["timestamp"])
        if entry["last_accessed"] is not None:
            _timestamp(entry["last_accessed"])


def _capture(memory: VectorMemory) -> dict[str, Any]:
    entries = []
    for entry in memory.entries:
        if type(entry) is not MemoryEntry:
            raise ValueError("Snapshots support canonical MemoryEntry objects only")
        entries.append(
            {
                "content": entry.content,
                "context": _json_value(entry.context),
                "timestamp": _date(entry.timestamp),
                "importance": entry.importance,
                "access_count": entry.access_count,
                "last_accessed": (
                    _date(entry.last_accessed)
                    if entry.last_accessed is not None
                    else None
                ),
            }
        )
    payload = {
        "embedding_model": memory.embedding_model,
        "max_entries": memory.max_entries,
        "entries": entries,
    }
    _validate_payload(payload)
    return payload


def _revision(value: Any) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise ValueError("Snapshot revision must be a 32-character lowercase hex ID")


def _arguments(max_bytes: int, expected_revision: str | None = None) -> None:
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if expected_revision is not None:
        _revision(expected_revision)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate memory snapshot JSON key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"Invalid JSON constant: {value}")


def _read(path: Path, max_bytes: int) -> dict[str, Any]:
    with os.fdopen(_open_regular(path, os.O_RDONLY), "rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("Memory snapshot exceeds max_bytes")
    document: dict[str, Any] = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
    )
    _keys(
        document,
        {"schema_version", "kind", "revision", "created_at", "payload", "sha256"},
    )
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("Unsupported memory snapshot schema version")
    if document["kind"] != "stateset_vector_memory":
        raise ValueError("Invalid memory snapshot kind")
    _revision(document["revision"])
    _timestamp(document["created_at"])
    _validate_payload(document["payload"])
    checksum = document["sha256"]
    unsigned = {key: value for key, value in document.items() if key != "sha256"}
    if (
        type(checksum) is not str
        or checksum != hashlib.sha256(_canonical(unsigned)).hexdigest()
    ):
        raise ValueError("Memory snapshot checksum mismatch")
    return document


def _metadata(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": document["schema_version"],
        "revision": document["revision"],
        "created_at": document["created_at"],
        "sha256": document["sha256"],
        "entry_count": len(document["payload"]["entries"]),
        "source_embedding_model": document["payload"]["embedding_model"],
        "source_max_entries": document["payload"]["max_entries"],
    }


def _update(memory: VectorMemory, operation: dict[str, Any], **fields: Any) -> None:
    with memory._owner().lock:
        operation.update(fields)


def _admit(memory: VectorMemory, path: Path, action: str) -> dict[str, Any]:
    owner = memory._owner()
    with owner.lock:
        if owner.snapshot_operation is not None:
            raise MemorySnapshotConflict("Another snapshot operation owns this memory")
        if owner.state is BackendInitializationState.INITIALIZING:
            raise MemorySnapshotConflict("Memory backend is initializing")
        operation = {
            "id": uuid4().hex,
            "action": action,
            "path": str(path),
            "state": "running",
            "phase": "capturing",
            "revision": None,
            "observed_revision": None,
            "captured_mutation_revision": owner.mutation_revision,
            "entry_count": None,
            "publication": "not_started",
            "file_synced": False,
            "directory_synced": False,
            "error_type": None,
            "work_error_type": None,
            "cleanup_error_type": None,
        }
        owner.snapshot_operation = operation
        return operation


def _finish(
    memory: VectorMemory, operation: dict[str, Any], error: BaseException | None
) -> dict[str, Any]:
    owner = memory._owner()
    with owner.lock:
        operation["state"] = (
            "completed"
            if error is None
            else (
                "cancelled"
                if isinstance(error, asyncio.CancelledError)
                else (
                    "conflict"
                    if isinstance(error, MemorySnapshotConflict)
                    else "failed"
                )
            )
        )
        operation["error_type"] = type(error).__name__ if error is not None else None
        owner.last_snapshot_operation = copy.deepcopy(operation)
        owner.snapshot_operation = None
        return copy.deepcopy(operation)


def _work(
    memory: VectorMemory, operation: dict[str, Any], function: Callable[[], _T]
) -> _T:
    try:
        return function()
    except BaseException as error:
        _update(memory, operation, work_error_type=type(error).__name__)
        raise


def _publish(
    memory: VectorMemory,
    operation: dict[str, Any],
    path: Path,
    raw: bytes,
    expected_revision: str | None,
    max_bytes: int,
) -> None:
    publish_bytes(
        path,
        raw,
        expected_revision=expected_revision,
        read_revision=lambda: _read(path, max_bytes)["revision"],
        status=lambda fields: _update(memory, operation, **fields),
        conflict=MemorySnapshotConflict,
        sync=_sync_directory,
    )


async def save_snapshot(
    memory: VectorMemory,
    path: str | os.PathLike[str],
    expected_revision: str | None,
    max_bytes: int,
) -> dict[str, Any]:
    """Capture once, then own the file transaction through caller cancellation."""
    _arguments(max_bytes, expected_revision)
    target = Path(path).absolute()
    operation = _admit(memory, target, "save")
    try:
        with memory._owner().lock:
            document: dict[str, Any] = {
                "schema_version": 1,
                "kind": "stateset_vector_memory",
                "revision": uuid4().hex,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "payload": _capture(memory),
            }
            operation["captured_mutation_revision"] = memory._owner().mutation_revision
        document["sha256"] = hashlib.sha256(_canonical(document)).hexdigest()
        raw = _canonical(document)
        if len(raw) > max_bytes:
            raise ValueError("Memory snapshot exceeds max_bytes")
        _update(
            memory,
            operation,
            revision=document["revision"],
            entry_count=len(document["payload"]["entries"]),
        )
        await run_sync_owned(
            _work,
            memory,
            operation,
            lambda: _publish(
                memory, operation, target, raw, expected_revision, max_bytes
            ),
        )
    except BaseException as error:
        _finish(memory, operation, error)
        raise
    return _finish(memory, operation, None)


async def restore_snapshot(
    memory: VectorMemory,
    path: str | os.PathLike[str],
    expected_revision: str | None,
    max_bytes: int,
) -> dict[str, Any]:
    """Stage and reindex portable data before a single fenced in-memory commit."""
    _arguments(max_bytes, expected_revision)
    target = Path(path).absolute()
    operation = _admit(memory, target, "restore")
    owner = memory._owner()
    try:
        with owner.lock:
            token = memory._token()
            baseline = _canonical(_capture(memory))
            revision = owner.mutation_revision
            original_entries = memory.entries
            identities = tuple(map(id, memory.entries))
            operation["captured_mutation_revision"] = revision
        _update(memory, operation, phase="reading")
        document = await run_sync_owned(
            _work, memory, operation, lambda: _read(target, max_bytes)
        )
        _update(
            memory,
            operation,
            revision=document["revision"],
            observed_revision=document["revision"],
            phase="validating",
        )
        if expected_revision is not None and document["revision"] != expected_revision:
            raise MemorySnapshotConflict(
                "Snapshot file revision does not match the requested revision"
            )
        saved_entries = document["payload"]["entries"]
        if len(saved_entries) > memory.max_entries:
            raise ValueError("Snapshot exceeds destination memory capacity")
        _update(memory, operation, phase="reindexing", entry_count=len(saved_entries))
        candidates = []
        dimensions = set()
        for saved in saved_entries:
            entry = MemoryEntry(
                content=saved["content"],
                context=saved["context"],
                timestamp=_timestamp(saved["timestamp"]),
                importance=saved["importance"],
                access_count=saved["access_count"],
                last_accessed=(
                    _timestamp(saved["last_accessed"])
                    if saved["last_accessed"] is not None
                    else None
                ),
            )
            if token[0] is not None:

                async def embed(text: str = entry.content) -> Any:
                    try:
                        return await memory._generate_embedding(text)
                    except BaseException as error:
                        _update(memory, operation, work_error_type=type(error).__name__)
                        raise

                embedding = np.asarray(
                    await run_async_owned(embed()), dtype=np.float64
                ).copy()
                if (
                    embedding.ndim != 1
                    or not embedding.size
                    or not np.isfinite(embedding).all()
                ):
                    raise ValueError(
                        "Memory embeddings must be nonempty finite vectors"
                    )
                dimensions.add(embedding.size)
                if len(dimensions) > 1:
                    raise ValueError(
                        "Embedding model returned inconsistent vector dimensions"
                    )
                entry.embedding = embedding
            candidates.append(entry)
        with owner.lock:
            operation["phase"] = "committing"
            try:
                memory._check_token(token)
                unchanged = (
                    owner.mutation_revision == revision
                    and memory.entries is original_entries
                    and tuple(map(id, memory.entries)) == identities
                    and _canonical(_capture(memory)) == baseline
                )
            except (RuntimeError, ValueError) as error:
                raise MemorySnapshotConflict(
                    "Live memory changed during restore"
                ) from error
            if not unchanged:
                raise MemorySnapshotConflict("Live memory changed during restore")
            memory.entries = candidates
            owner.generation += 1
            owner.mutation_revision += 1
            operation.update(publication="published", phase="finished")
    except BaseException as error:
        _finish(memory, operation, error)
        raise
    return _finish(memory, operation, None)


async def inspect_snapshot(
    path: str | os.PathLike[str], max_bytes: int
) -> dict[str, Any]:
    """Inspect validated metadata; never execute the source model configuration."""
    _arguments(max_bytes)
    document = await run_sync_owned(_read, Path(path).absolute(), max_bytes)
    return _metadata(document)
