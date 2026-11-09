"""The durable record of a fan-out, one atomically replaced file per run.

A sandbox bills until it is stopped, and the process driving it can die at any
moment. So the sandbox id reaches disk before anything else happens to it, and a
later process can pick up exactly where the last one stopped: reattach to what is
still running, or stop what nothing is watching.

Secrets never appear here. Forwarded environment values are not part of any
record, and manifests refuse literal secret values.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stateset_agents.fanout.manifest import FanoutManifest, RunSpec
from stateset_agents.utils.atomic_io import atomic_json

__all__ = [
    "AttemptRecord",
    "FanoutJournal",
    "RUN_FILE",
    "RunRecord",
    "TERMINAL_STATUSES",
    "load_records",
    "load_run_file",
]

RUN_FILE = "run.json"
EVENTS_FILE = "events.jsonl"
STATE_FILE = "state.json"
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


@dataclass
class AttemptRecord:
    """One sandbox's worth of trying. A retry is always a new sandbox."""

    number: int
    #: Idempotency key for this attempt's create, on disk before the create is
    #: sent. Asking again with it can only ever answer with the same sandbox.
    create_key: str | None = None
    #: A create went out and nothing has proved it did not land. On disk before
    #: the create is sent, so that a crash cannot lose track of one.
    create_maybe_landed: bool = False
    create_sent_at: float | None = None
    allocation_id: str | None = None
    root: str | None = None
    process_id: str | None = None
    started_at: float = 0.0
    #: When the process started. The run's time limit counts from here, so that
    #: waiting for capacity never eats into it.
    process_started_at: float | None = None
    #: This run's time limit: the manifest's, or less if the sandbox was granted less.
    time_limit_s: int | None = None
    finished_at: float | None = None
    #: succeeded, workload_failed, infra_failed, timed_out, budget_exhausted,
    #: capacity_exhausted, cancelled or fatal. None while in progress.
    outcome: str | None = None
    reason: str | None = None
    exit_code: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    stdout_offset: int = 0
    stderr_offset: int = 0
    cost_cents: float | None = None
    released: bool = False
    release_error: str | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    #: What ``collect`` asked for and did not get. A run with these is not clean.
    collect_errors: list[str] = field(default_factory=list)

    @property
    def open(self) -> bool:
        """In progress, as far as anything on disk knows."""
        return self.outcome is None

    @property
    def allocation_unknown(self) -> bool:
        """Finished, yet a sandbox may exist that no record names."""
        return (
            self.outcome is not None
            and self.allocation_id is None
            and self.create_maybe_landed
        )

    @property
    def needs_release(self) -> bool:
        """Finished, but its sandbox was never confirmed stopped."""
        return (
            self.allocation_id is not None
            and self.outcome is not None
            and not self.released
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttemptRecord:
        names = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in names})


@dataclass
class RunRecord:
    """One run's state and full attempt history."""

    run_id: str
    arm_id: str
    seed: int
    params: dict[str, Any]
    status: str = "pending"
    attempts: list[AttemptRecord] = field(default_factory=list)
    #: Attempts before this index do not count against ``max_attempts``.
    retry_base: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    last_event: dict[str, Any] | None = None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def attempts_used(self) -> int:
        """Attempts that count. A fatal error says nothing about this run."""
        return sum(
            attempt.outcome != "fatal" for attempt in self.attempts[self.retry_base :]
        )

    @property
    def cost_cents(self) -> float | None:
        """Total measured spend. One unknown attempt makes the total unknown."""
        total = 0.0
        for attempt in self.attempts:
            if attempt.allocation_id is None:
                if attempt.create_maybe_landed:
                    return None
                continue
            if attempt.cost_cents is None:
                return None
            total += attempt.cost_cents
        return total

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        names = {item.name for item in fields(cls)}
        values = {key: value for key, value in data.items() if key in names}
        values["attempts"] = [
            AttemptRecord.from_dict(item) for item in values.get("attempts", [])
        ]
        return cls(**values)

    def to_report_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "arm_id": self.arm_id,
            "seed": self.seed,
            "params": self.params,
            "status": self.status,
            "metrics": self.metrics,
            "cost_cents": self.cost_cents,
            "collect_errors": [
                error
                for attempt in self.attempts[-1:]
                for error in attempt.collect_errors
            ],
        }


def load_run_file(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / RUN_FILE
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{run_dir} is not a fan-out run directory: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_records(run_dir: str | Path) -> list[RunRecord]:
    """Read every run record. Safe while a driver runs: writes replace atomically."""
    records = [
        RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted((Path(run_dir) / "runs").glob(f"*/{STATE_FILE}"))
    ]
    return records


class FanoutJournal:
    """Reads and writes one run directory. One driver at a time holds its lock."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def start(self, manifest: FanoutManifest, backend_name: str) -> None:
        """Record what this directory is for, or confirm a resume matches it."""
        path = self.run_dir / RUN_FILE
        digest = manifest.digest()
        if path.exists():
            existing = load_run_file(self.run_dir)
            if existing.get("digest") != digest:
                raise ValueError(
                    f"{self.run_dir} holds a different manifest; refusing to mix "
                    "two experiments in one report. Use a new run directory."
                )
            if existing.get("backend") != backend_name:
                raise ValueError(
                    f"{self.run_dir} ran on backend {existing.get('backend')!r}, "
                    f"not {backend_name!r}: its allocations mean nothing here"
                )
            return
        atomic_json(
            path,
            {
                "schema_version": 1,
                "name": manifest.name,
                "digest": digest,
                "backend": backend_name,
                "metric": manifest.report.metric,
                "mode": manifest.report.mode,
                "manifest": manifest.to_dict(),
                "base_dir": str(manifest.base_dir.resolve()),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def run_root(self, run_id: str) -> Path:
        return self.run_dir / "runs" / run_id

    def attempt_dir(self, run_id: str, number: int) -> Path:
        return self.run_root(run_id) / f"attempt-{number}"

    def ensure_runs(self, runs: list[RunSpec]) -> None:
        for run in runs:
            if not (self.run_root(run.run_id) / STATE_FILE).exists():
                self.save(
                    RunRecord(
                        run_id=run.run_id,
                        arm_id=run.arm_id,
                        seed=run.seed,
                        params=run.params,
                    )
                )

    def load(self, run_id: str) -> RunRecord:
        path = self.run_root(run_id) / STATE_FILE
        return RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, record: RunRecord) -> None:
        atomic_json(self.run_root(record.run_id) / STATE_FILE, record.to_dict())

    def records(self) -> list[RunRecord]:
        return load_records(self.run_dir)

    def append_event(self, event: dict[str, Any]) -> None:
        """Append one event line. The caller serializes concurrent writers."""
        with (self.run_dir / EVENTS_FILE).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
