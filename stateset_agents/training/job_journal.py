"""Single-owner, atomic job records for local process-crash recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from stateset_agents.remote.river_rl import atomic_json, exclusive_run

from .advanced_training_models import (
    TrainingJob,
    deserialize_training_job,
    serialize_training_job,
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key in training job journal")
        result[key] = value
    return result


class JobJournal:
    """Hold an OS lock and commit JSON records using file and directory fsync.

    Use a persistent local filesystem with reliable locking and atomic rename.
    This journal does not coordinate jobs across hosts or stop remote processes.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._ownership = exclusive_run(self.directory)
        self._ownership.__enter__()
        self._closed = False

    def _path(self, job_id: str) -> Path:
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id must be nonempty text")
        name = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
        return self.directory / f"{name}.json"

    def save(self, snapshot: dict[str, Any]) -> None:
        """Durably replace one record before exposing its transition to workers."""
        if self._closed:
            raise RuntimeError("Training job journal is closed")
        atomic_json(
            self._path(snapshot["job_id"]),
            {"schema_version": 2, "job": snapshot},
        )

    def _read(self, path: Path) -> dict[str, Any]:
        if path.is_symlink():
            raise ValueError("Training job journal records must not be symlinks")
        record = json.loads(path.read_text(), object_pairs_hook=_unique_object)
        # Reject nonfinite numbers, including overflowing JSON numeric literals.
        json.dumps(record, allow_nan=False)
        if (
            not isinstance(record, dict)
            or set(record) != {"schema_version", "job"}
            or type(record["schema_version"]) is not int
            or record["schema_version"] not in (1, 2)
            or not isinstance(record["job"], dict)
        ):
            raise ValueError("Unsupported training job journal record")
        snapshot = record["job"]
        if record["schema_version"] == 1:
            if "submission_fingerprint" in snapshot:
                raise ValueError("Unexpected submission fingerprint in legacy journal")
            snapshot = {**snapshot, "submission_fingerprint": None}
        job = deserialize_training_job(snapshot)
        if serialize_training_job(job) != snapshot:
            raise ValueError("Incomplete training job journal record")
        if path != self._path(job.job_id):
            raise ValueError("Training job journal identity does not match filename")
        fingerprint = job.submission_fingerprint
        if fingerprint is not None and (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)
        ):
            raise ValueError("Invalid submission fingerprint in training job journal")
        if job.job_id.startswith("idem_") and fingerprint is None:
            raise ValueError("Missing submission fingerprint for idempotent job")
        return snapshot

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Read an atomically committed record, including after graceful shutdown."""
        path = self._path(job_id)
        if not path.exists():
            return None
        return self._read(path)

    def load(self) -> list[TrainingJob]:
        """Validate every committed record before the orchestrator starts work."""
        jobs = [
            deserialize_training_job(self._read(path))
            for path in sorted(self.directory.glob("*.json"))
        ]
        for job in jobs:
            for value in (job.created_at, job.started_at, job.completed_at):
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, (int, float))
                ):
                    raise ValueError("Invalid timestamp in training job journal")
            if job.created_at is None:
                raise ValueError("Missing creation timestamp in training job journal")
        return sorted(jobs, key=lambda job: (job.created_at, job.job_id))

    def close(self) -> None:
        """Release ownership only after workers have stopped and writes finish."""
        if not self._closed:
            self._closed = True
            self._ownership.__exit__(None, None, None)
