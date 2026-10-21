"""Transactional training queue for API replicas and independent workers.

SQLite coordinates processes sharing one local persistent filesystem. Use one
queue volume per deployment; SQLite WAL is not supported on network filesystems.
Expired running work is marked failed and requires explicit checkpoint resume:
automatically replaying an unknown GPU operation could duplicate side effects.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

TERMINAL = {"completed", "failed", "cancelled"}
MAX_PENDING_JOBS = 1000


class TrainingLeaseLostError(RuntimeError):
    """The worker no longer owns a live lease and must stop publishing results."""


class TrainingIdempotencyConflictError(ValueError):
    """An idempotency key already identifies a different accepted request."""


class TrainingQueueFullError(ValueError):
    """The durable queue has reached its global pending-job capacity."""


class _PostgresRow(dict):
    def __getitem__(self, key: Any) -> Any:
        return (
            list(self.values())[key]
            if isinstance(key, int)
            else super().__getitem__(key)
        )


class _PostgresCursor:
    def __init__(self, cursor: Any) -> None:
        self.cursor = cursor

    def fetchone(self) -> Any:
        row = self.cursor.fetchone()
        return None if row is None else _PostgresRow(row)

    def fetchall(self) -> list[Any]:
        return [_PostgresRow(row) for row in self.cursor.fetchall()]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.fetchall())


class _PostgresConnection:
    """DB-API adapter preserving the queue's transaction semantics across hosts."""

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise ImportError(
                "PostgreSQL workers require stateset-agents[workers]"
            ) from exc
        self.connection = psycopg.connect(
            dsn, autocommit=True, row_factory=dict_row, client_encoding="utf8"
        )

    def execute(self, statement: str, parameters: Any = ()) -> _PostgresCursor:
        if statement == "BEGIN IMMEDIATE":
            self.connection.execute("BEGIN")
            # One database-wide lock keeps capacity and claim decisions atomic.
            return _PostgresCursor(
                self.connection.execute("SELECT pg_advisory_xact_lock(873421901)")
            )
        return _PostgresCursor(
            self.connection.execute(statement.replace("?", "%s"), parameters)
        )

    def commit(self) -> None:
        self.connection.commit()

    def rollback(self) -> None:
        self.connection.rollback()

    def close(self) -> None:
        self.connection.close()


class TrainingQueue:
    """Durable idempotent submission, global admission, leases and cancellation."""

    def __init__(self, path: str | Path, *, max_concurrent_jobs: int = 1) -> None:
        if type(max_concurrent_jobs) is not int or max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be a positive integer")
        self.postgres_dsn = (
            str(path)
            if str(path).startswith(("postgresql://", "postgres://"))
            else None
        )
        self.path = Path(path).resolve() if self.postgres_dsn is None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(mode=0o600, exist_ok=True)
            self.path.chmod(0o600)
        self.max_concurrent_jobs = max_concurrent_jobs
        with self._connection() as connection:
            if self.postgres_dsn is None:
                connection.execute("PRAGMA journal_mode=WAL")
        with self._transaction() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS stateset_training_jobs (
                id TEXT PRIMARY KEY, owner TEXT, fingerprint TEXT NOT NULL,
                status TEXT NOT NULL, snapshot TEXT NOT NULL, created DOUBLE PRECISION NOT NULL,
                lease_token TEXT, lease_until DOUBLE PRECISION, cancel INTEGER NOT NULL DEFAULT 0
            )""")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS stateset_training_jobs_status ON stateset_training_jobs(status, created)"
            )
            for name, columns in (
                ("created", "created DESC,id DESC"),
                ("owner_created", "owner,created DESC,id DESC"),
                ("status_created", "status,created DESC,id DESC"),
            ):
                connection.execute(
                    f"CREATE INDEX IF NOT EXISTS stateset_training_jobs_{name} "
                    f"ON stateset_training_jobs({columns})"
                )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS stateset_training_queue_settings (
                id INTEGER PRIMARY KEY CHECK(id=1), capacity INTEGER NOT NULL
            )"""
            )
            connection.execute(
                "INSERT INTO stateset_training_queue_settings VALUES (1, ?) ON CONFLICT DO NOTHING",
                (max_concurrent_jobs,),
            )
            if (
                connection.execute(
                    "SELECT capacity FROM stateset_training_queue_settings"
                ).fetchone()[0]
                != max_concurrent_jobs
            ):
                raise ValueError(
                    "All workers and API replicas must use the same queue capacity"
                )
        if self.path is not None:
            self.path.chmod(0o600)

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        connection: Any
        if self.postgres_dsn is not None:
            connection = _PostgresConnection(self.postgres_dsn)
        else:
            assert self.path is not None
            connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA synchronous=FULL")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _now(self, connection: Any) -> float:
        """Read the shared clock after acquiring the queue's transaction lock."""
        if self.postgres_dsn is not None:
            row = connection.execute(
                "SELECT EXTRACT(EPOCH FROM clock_timestamp())"
            ).fetchone()
            return float(row[0])
        return time.time()

    @staticmethod
    def _validate_lease_seconds(lease_seconds: float) -> None:
        if (
            type(lease_seconds) not in (int, float)
            or not 1 <= lease_seconds <= 3600
            or not math.isfinite(lease_seconds)
        ):
            raise ValueError("lease_seconds must be a finite number between 1 and 3600")

    @staticmethod
    def _encode(value: Any) -> str:
        return json.dumps(
            value, sort_keys=True, allow_nan=False, default=lambda x: x.isoformat()
        )

    @staticmethod
    def _decode(value: str) -> dict[str, Any]:
        snapshot: dict[str, Any] = json.loads(value)
        for name in ("created_at", "started_at", "completed_at"):
            if snapshot.get(name) is not None:
                snapshot[name] = datetime.fromisoformat(snapshot[name])
        return snapshot

    def submit(
        self,
        request: dict[str, Any],
        owner: str | None,
        key: str | None = None,
        *,
        output_root: str | None = None,
        resume_from_checkpoint: str | None = None,
        resume_user_id: str | None = None,
    ) -> str:
        """Commit inputs before acknowledging; a reused key must bind identical inputs."""
        fingerprint = hashlib.sha256(
            self._encode([owner, request]).encode()
        ).hexdigest()
        if key is not None and (not isinstance(key, str) or not 1 <= len(key) <= 256):
            raise ValueError("Idempotency key must contain 1 to 256 characters")
        job_id = (
            str(uuid.uuid4())
            if key is None
            else "idem_"
            + hashlib.sha256(self._encode([owner, key]).encode()).hexdigest()
        )
        now = datetime.utcnow()
        snapshot = {
            "status": "queued",
            "user_id": owner,
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "completion_scope": None,
            "progress": 0.0,
            "current_episode": 0,
            "total_episodes": request["num_episodes"],
            "metrics": {},
            "error": None,
            "config": request,
        }
        if output_root is not None:
            # Trusted service execution metadata is separate from client inputs
            # and their idempotency fingerprint. The first accepted plan wins.
            snapshot["output_dir"] = str(Path(output_root) / job_id)
            snapshot["resume_from_checkpoint"] = resume_from_checkpoint
            snapshot["resume_user_id"] = resume_user_id
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT fingerprint FROM stateset_training_jobs WHERE id=?", (job_id,)
            ).fetchone()
            if existing:
                if existing[0] != fingerprint:
                    raise TrainingIdempotencyConflictError(
                        "Idempotency key already binds a different training request"
                    )
                return job_id
            if (
                connection.execute(
                    "SELECT count(*) FROM stateset_training_jobs WHERE status='queued'"
                ).fetchone()[0]
                >= MAX_PENDING_JOBS
            ):
                raise TrainingQueueFullError("Training queue is full")
            connection.execute(
                "INSERT INTO stateset_training_jobs(id,owner,fingerprint,status,snapshot,created) VALUES(?,?,?,?,?,?)",
                (
                    job_id,
                    owner,
                    fingerprint,
                    "queued",
                    self._encode(snapshot),
                    self._now(connection),
                ),
            )
        return job_id

    def jobs(self) -> dict[str, dict[str, Any]]:
        """Read detached snapshots, including jobs submitted by other replicas."""
        with self._connection() as connection:
            return {
                row["id"]: self._decode(row["snapshot"])
                for row in connection.execute(
                    "SELECT id,snapshot FROM stateset_training_jobs"
                )
            }

    def page(
        self,
        *,
        owner: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[tuple[str, dict[str, Any]]], int]:
        """Read one owner-filtered page and its total from a consistent snapshot.

        An omitted owner means unrestricted administrative access, matching
        ``get()``. Ownerless jobs remain visible to authenticated users. Only
        the requested page's payloads are fetched and decoded.
        """
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if status is not None and not isinstance(status, str):
            raise ValueError("status must be a string or None")
        parameters: list[Any] = []
        if owner is not None:
            parameters.append(owner)
        if status:
            parameters.append(status)
        # Only choose complete literal statements. Caller-controlled owner and
        # status values always stay in bound parameters, including quotes/SQL.
        count_query, page_query = {
            (False, False): (
                "SELECT count(*) FROM stateset_training_jobs",
                "SELECT id,snapshot FROM stateset_training_jobs "
                "ORDER BY created DESC,id DESC LIMIT ? OFFSET ?",
            ),
            (True, False): (
                "SELECT count(*) FROM stateset_training_jobs "
                "WHERE (owner=? OR owner IS NULL)",
                "SELECT id,snapshot FROM stateset_training_jobs "
                "WHERE (owner=? OR owner IS NULL) "
                "ORDER BY created DESC,id DESC LIMIT ? OFFSET ?",
            ),
            (False, True): (
                "SELECT count(*) FROM stateset_training_jobs WHERE status=?",
                "SELECT id,snapshot FROM stateset_training_jobs WHERE status=? "
                "ORDER BY created DESC,id DESC LIMIT ? OFFSET ?",
            ),
            (True, True): (
                "SELECT count(*) FROM stateset_training_jobs "
                "WHERE (owner=? OR owner IS NULL) AND status=?",
                "SELECT id,snapshot FROM stateset_training_jobs "
                "WHERE (owner=? OR owner IS NULL) AND status=? "
                "ORDER BY created DESC,id DESC LIMIT ? OFFSET ?",
            ),
        }[(owner is not None, bool(status))]
        with self._connection() as connection:
            # PostgreSQL READ COMMITTED would allow different snapshots for the
            # count and rows. Readers never acquire the writers' advisory lock.
            connection.execute(
                "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY"
                if self.postgres_dsn is not None
                else "BEGIN"
            )
            try:
                total = int(
                    connection.execute(
                        count_query,
                        parameters,
                    ).fetchone()[0]
                )
                # This also avoids binding a caller's enormous Python offset
                # into a signed-64-bit database parameter when the page is empty.
                rows = (
                    connection.execute(
                        page_query,
                        [*parameters, limit, offset],
                    ).fetchall()
                    if offset < total
                    else []
                )
            finally:
                connection.rollback()
        return [(row["id"], self._decode(row["snapshot"])) for row in rows], total

    def get(self, job_id: str, owner: str | None = None) -> dict[str, Any] | None:
        """Read a snapshot subject to the same owner check as the API service."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT owner,snapshot FROM stateset_training_jobs WHERE id=?",
                (job_id,),
            ).fetchone()
        if row is None or (owner is not None and row["owner"] not in (None, owner)):
            return None
        return self._decode(row["snapshot"])

    def claim(
        self, *, lease_seconds: float = 60
    ) -> tuple[str, str, dict[str, Any]] | None:
        """Claim at most global capacity; fence expired workers instead of replaying."""
        self._validate_lease_seconds(lease_seconds)
        with self._transaction() as connection:
            now = self._now(connection)
            for row in connection.execute(
                "SELECT * FROM stateset_training_jobs WHERE lease_until <= ? AND status IN ('running','cancelling')",
                (now,),
            ).fetchall():
                snapshot = self._decode(row["snapshot"])
                snapshot.update(
                    status="failed",
                    completed_at=datetime.utcnow(),
                    error="Worker lease expired; resume explicitly from a verified checkpoint",
                )
                connection.execute(
                    "UPDATE stateset_training_jobs SET status='failed',snapshot=?,lease_token=NULL WHERE id=?",
                    (self._encode(snapshot), row["id"]),
                )
            active = connection.execute(
                "SELECT count(*) FROM stateset_training_jobs WHERE status IN ('running','cancelling')"
            ).fetchone()[0]
            if active >= self.max_concurrent_jobs:
                return None
            row = connection.execute(
                "SELECT * FROM stateset_training_jobs WHERE status='queued' ORDER BY created,id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            snapshot = self._decode(row["snapshot"])
            snapshot.update(status="running", started_at=datetime.utcnow())
            connection.execute(
                "UPDATE stateset_training_jobs SET status='running',snapshot=?,lease_token=?,lease_until=? WHERE id=?",
                (self._encode(snapshot), token, now + lease_seconds, row["id"]),
            )
            return row["id"], token, snapshot

    def heartbeat(
        self,
        job_id: str,
        token: str,
        *,
        lease_seconds: float = 60,
        progress: dict[str, Any] | None = None,
    ) -> bool:
        """Renew an unexpired lease; return cancellation state, rejecting stale owners."""
        self._validate_lease_seconds(lease_seconds)
        with self._transaction() as connection:
            now = self._now(connection)
            row = connection.execute(
                "SELECT * FROM stateset_training_jobs WHERE id=? AND lease_token=? AND lease_until>? AND status IN ('running','cancelling')",
                (job_id, token, now),
            ).fetchone()
            if row is None:
                raise TrainingLeaseLostError("Training worker lost its lease")
            snapshot = self._decode(row["snapshot"])
            if progress is not None:
                for name in (
                    "progress",
                    "current_episode",
                    "metrics",
                    "completion_scope",
                ):
                    if name in progress:
                        snapshot[name] = progress[name]
            connection.execute(
                "UPDATE stateset_training_jobs SET lease_until=?,snapshot=? WHERE id=? AND lease_token=?",
                (now + lease_seconds, self._encode(snapshot), job_id, token),
            )
            return bool(row["cancel"])

    def finish(self, job_id: str, token: str, snapshot: dict[str, Any]) -> None:
        """Commit a terminal result only while holding the current live lease."""
        if snapshot.get("status") not in TERMINAL:
            raise ValueError("Worker result must be terminal")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT cancel,snapshot FROM stateset_training_jobs WHERE id=? AND lease_token=? AND lease_until>? AND status IN ('running','cancelling')",
                (job_id, token, self._now(connection)),
            ).fetchone()
            if row is None:
                raise TrainingLeaseLostError("Training worker lost its lease")
            persisted = self._decode(row["snapshot"])
            persisted.update(
                {
                    k: snapshot[k]
                    for k in (
                        "status",
                        "completed_at",
                        "completion_scope",
                        "progress",
                        "current_episode",
                        "metrics",
                        "error",
                    )
                    if k in snapshot
                }
            )
            if row["cancel"]:
                persisted["status"] = "cancelled"
            connection.execute(
                "UPDATE stateset_training_jobs SET status=?,snapshot=?,lease_token=NULL,lease_until=NULL WHERE id=?",
                (persisted["status"], self._encode(persisted), job_id),
            )

    def cancel(self, job_id: str, owner: str | None = None) -> bool:
        """Cancel queued work atomically or signal a running lease holder."""
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM stateset_training_jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None or (owner is not None and row["owner"] not in (None, owner)):
                return False
            if row["status"] in TERMINAL:
                return True
            snapshot = self._decode(row["snapshot"])
            snapshot["status"] = (
                "cancelled" if row["status"] == "queued" else "cancelling"
            )
            if snapshot["status"] == "cancelled":
                snapshot["completed_at"] = datetime.utcnow()
            connection.execute(
                "UPDATE stateset_training_jobs SET cancel=1,status=?,snapshot=? WHERE id=?",
                (snapshot["status"], self._encode(snapshot), job_id),
            )
        return True
