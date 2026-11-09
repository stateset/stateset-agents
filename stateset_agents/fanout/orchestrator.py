"""Drive N independent runs to completion, whatever happens to each of them.

Each run is one thread, mostly asleep between polls, with a bounded number alive
at once. A run's story is a sequence of attempts, each on a fresh sandbox:

    journal a create key -> allocate -> journal the id -> upload -> start
        -> watch -> collect -> release

Three rules hold throughout. The create key is on disk before the create is sent,
so a lost response or a crash can find its sandbox again rather than orphan it. The
sandbox id is on disk before anything else touches the sandbox. And an ambiguous
start is never repeated in one sandbox, because a failed request may still have
reached the server: only a new sandbox is safe. A 429 is different: it is a refusal,
so asking again cannot start anything twice.

What counts as worth retrying:

* ``infra_failed`` retries: preemption, bootstrap failure, unreachable sandbox.
* ``workload_failed`` does not: the worker ran and the run failed, and it would
  fail the same way again while billing again.
* ``timed_out`` and ``budget_exhausted`` do not: another attempt spends the same
  ceiling to reach the same end.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from stateset_agents.fanout.backend import (
    Allocation,
    AllocationLostError,
    BackendError,
    CapacityError,
    FatalBackendError,
    ProcessState,
    RunBackend,
    TransientBackendError,
)
from stateset_agents.fanout.journal import (
    AttemptRecord,
    FanoutJournal,
    RunRecord,
    load_records,
    load_run_file,
)
from stateset_agents.fanout.manifest import FanoutManifest, Resources, RunSpec
from stateset_agents.fanout.report import build_report
from stateset_agents.fanout.worker import EVENT_MARKER, RESULT_FILE
from stateset_agents.remote.ledger import CostEntry, record_entry
from stateset_agents.utils.atomic_io import atomic_json, exclusive_run

__all__ = [
    "CancelOutcome",
    "FanoutRunner",
    "FanoutSummary",
    "cancel_run",
    "load_records",
    "rebuild_report",
    "remaining_worst_case_usd",
    "request_cancel",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")

CANCEL_FILE = "CANCEL"
REPORT_FILE = "report.json"

#: The process's own limit fires first. This catches a stuck control plane, so it
#: only needs to outlast that enforcement.
_DEADLINE_GRACE_S = 120.0
#: Kept back from a sandbox's lifetime, so that a run reaching its limit can still
#: have its output collected before the sandbox goes.
_COLLECT_RESERVE_S = 300.0
#: Below this there is no point starting: installing alone would not finish.
_MIN_RUN_SECONDS = 60
#: How long after its expiry to keep treating an unresolved create as a live risk.
#: Covers a create being slower than its response timeout.
_CREATE_RESOLUTION_GRACE_S = 900.0
_RELEASE_TRIES = 3
#: Rate-limit windows are a minute: enough tries with doubling waits to outlast one.
_CONTROL_PLANE_TRIES = 6
_MAX_LOG_READS_PER_POLL = 8
_STDERR_TAIL_BYTES = 600
_MAX_BACKOFF_S = 60.0

_FAILED_OUTCOMES = frozenset(
    {"workload_failed", "timed_out", "budget_exhausted", "capacity_exhausted"}
)


def request_cancel(run_dir: str | Path) -> None:
    """Ask whichever driver owns this directory to stop, even from elsewhere."""
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / CANCEL_FILE).touch()


def remaining_worst_case_usd(
    manifest: FanoutManifest, run_dir: str | Path, *, retry_failed: bool = False
) -> float | None:
    """The most that continuing this directory could spend. None when unbounded."""
    cents = manifest.resources.max_cost_cents
    if cents is None:
        return None
    records = {record.run_id: record for record in load_records(run_dir)}
    runs = 0
    for run in manifest.expand():
        record = records.get(run.run_id)
        if record is None or not record.terminal:
            runs += 1
        elif retry_failed and record.status != "succeeded":
            runs += 1
    # Settling a create by asking again may itself start a sandbox, until it stops.
    unsettled = sum(
        attempt.allocation_unknown
        for record in records.values()
        for attempt in record.attempts
    )
    return (runs * manifest.max_attempts + unsettled) * cents / 100.0


def rebuild_report(run_dir: str | Path) -> dict[str, Any]:
    """Build a report from the journal alone. Needs no backend and no manifest files."""
    info = load_run_file(run_dir)
    return build_report(
        name=info["name"],
        digest=info["digest"],
        metric=info["metric"],
        mode=info["mode"],
        runs=[record.to_report_row() for record in load_records(run_dir)],
    )


def _unresolved_create_label(record: RunRecord, attempt: AttemptRecord) -> str:
    return f"unresolved create for {record.run_id} attempt {attempt.number}"


def _create_may_still_exist(
    attempt: AttemptRecord, lifetime: float, now: float
) -> bool:
    """A controller error can hold a create's key forever, but no sandbox can
    outlive its own expiry. Past that point the risk is over; only its cost stays
    unknown."""
    if attempt.create_sent_at is None:
        return True
    return now <= attempt.create_sent_at + lifetime + _CREATE_RESOLUTION_GRACE_S


@dataclass
class CancelOutcome:
    """What a cancel did. With a live driver, the driver does the stopping."""

    driver_active: bool
    stopped: list[str] = field(default_factory=list)
    unreleased: list[str] = field(default_factory=list)


def cancel_run(
    run_dir: str | Path, backend_factory: Callable[[], RunBackend]
) -> CancelOutcome:
    """Stop a fan-out, whether or not anything is still driving it.

    A live driver sees the request and stops its own sandboxes. Otherwise the
    driver is gone and nobody is watching what it started, so stop them here: the
    journal knows every sandbox that could still be billing.
    """
    root = Path(run_dir)
    load_run_file(root)
    request_cancel(root)
    try:
        lock = exclusive_run(root)
        lock.__enter__()
    except ValueError:
        return CancelOutcome(driver_active=True)
    try:
        return _cancel_without_driver(root, backend_factory())
    finally:
        lock.__exit__(None, None, None)


def _stop_allocation(backend: RunBackend, allocation_id: str) -> str | None:
    """Try to stop a sandbox. Returns the error if it may still be running."""
    error: str | None = None
    for _ in range(_RELEASE_TRIES):
        try:
            backend.release(allocation_id)
            return None
        except AllocationLostError:
            return None
        except BackendError as exc:
            error = str(exc)
    return error


def _cancel_without_driver(root: Path, backend: RunBackend) -> CancelOutcome:
    journal = FanoutJournal(root)
    outcome = CancelOutcome(driver_active=False)
    lifetime = Resources.from_dict(
        load_run_file(root)["manifest"]["resources"]
    ).sandbox_lifetime_seconds
    for record in journal.records():
        for attempt in record.attempts:
            if attempt.allocation_id is None:
                if attempt.open:
                    attempt.outcome = "cancelled"
                    attempt.reason = "cancelled"
                # Its create may have landed with nothing here to name it. The
                # sandbox's own limits bound it, and a resume can settle it.
                if attempt.allocation_unknown and _create_may_still_exist(
                    attempt, lifetime, time.time()
                ):
                    outcome.unreleased.append(_unresolved_create_label(record, attempt))
                continue
            if not (attempt.open or attempt.needs_release):
                continue
            error = _stop_allocation(backend, attempt.allocation_id)
            if attempt.open:
                attempt.outcome = "cancelled"
                attempt.reason = "cancelled"
            if error is None:
                attempt.released = True
                attempt.release_error = None
                outcome.stopped.append(attempt.allocation_id)
            else:
                attempt.release_error = error
                outcome.unreleased.append(attempt.allocation_id)
        if not record.terminal:
            _settle_for_cancel(record)
        journal.save(record)
    atomic_json(root / REPORT_FILE, rebuild_report(root))
    return outcome


def _settle_for_cancel(record: RunRecord) -> None:
    """A driver can die after an attempt finished but before recording the run."""
    last = record.attempts[-1] if record.attempts else None
    if last is not None and last.outcome == "succeeded":
        record.status = "succeeded"
        record.metrics = dict(last.metrics)
        record.error = None
    elif last is not None and last.outcome in _FAILED_OUTCOMES:
        record.status = "failed"
        record.metrics = dict(last.metrics)
        record.error = last.reason
    else:
        record.status = "cancelled"
        record.error = "cancelled"


@dataclass
class FanoutSummary:
    """How a driver's pass over the run directory ended."""

    run_dir: Path
    total: int
    succeeded: int
    failed: int
    cancelled: int
    incomplete: int
    report: dict[str, Any]
    aborted_reason: str | None = None
    #: Sandboxes that may still be running, and so still billing.
    unreleased: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.succeeded == self.total
            and not self.unreleased
            and self.aborted_reason is None
            and not self.report.get("artifacts_missing")
        )


class _CapacityWaitExceeded(Exception):
    """Waited as long as allowed for room to allocate."""


class _CreateOutcomeUnknown(Exception):
    """The create may have landed, and asking again kept getting no answer."""


class _Cancelled(Exception):
    """Cancelled while waiting to allocate."""


class FanoutRunner:
    """Runs a manifest on a backend, journaling everything into ``run_dir``."""

    def __init__(
        self,
        manifest: FanoutManifest,
        backend: RunBackend,
        run_dir: str | Path,
        *,
        forwarded_env: Mapping[str, str] | None = None,
        retry_failed: bool = False,
        poll_interval_s: float = 5.0,
        retry_backoff_s: float = 5.0,
        capacity_backoff_s: float = 15.0,
        capacity_wait_limit_s: float = 3600.0,
        create_unknown_wait_s: float = 300.0,
        max_poll_failures: int = 60,
        ledger_path: Path | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.manifest = manifest
        self.backend = backend
        self.run_dir = Path(run_dir)
        self._journal = FanoutJournal(self.run_dir)
        self._forwarded = self._resolve_forwarded(manifest, forwarded_env)
        self._retry_failed = retry_failed
        self._poll_interval_s = poll_interval_s
        self._retry_backoff_s = retry_backoff_s
        self._capacity_backoff_s = capacity_backoff_s
        self._capacity_wait_limit_s = capacity_wait_limit_s
        self._create_unknown_wait_s = create_unknown_wait_s
        self._max_poll_failures = max_poll_failures
        self._ledger_path = ledger_path
        self._on_event = on_event
        self._clock = clock
        self._wake = threading.Event()
        self._events_lock = threading.Lock()
        self._abort_reason: str | None = None
        self._partial_lines: dict[tuple[str, int], bytes] = {}

    @staticmethod
    def _resolve_forwarded(
        manifest: FanoutManifest, provided: Mapping[str, str] | None
    ) -> dict[str, str]:
        source = os.environ if provided is None else provided
        missing = [name for name in manifest.forward_env if name not in source]
        if missing:
            raise ValueError(
                f"forward_env names unset variable(s): {', '.join(missing)}"
            )
        return {name: source[name] for name in manifest.forward_env}

    # ------------------------------------------------------------------
    # Whole fan-out
    # ------------------------------------------------------------------

    def run(self) -> FanoutSummary:
        """Run or resume every unfinished run, then write the report."""
        self.backend.validate(self.manifest)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with exclusive_run(self.run_dir):
            self._journal.start(self.manifest, self.backend.name)
            (self.run_dir / CANCEL_FILE).unlink(missing_ok=True)
            runs = self.manifest.expand()
            self._journal.ensure_runs(runs)
            self._release_abandoned()
            todo = [run for run in runs if self._prepare_for_work(run)]
            self._event("fanout_started", total=len(runs), pending=len(todo))
            if todo:
                self._drive_all(todo)
            unreleased = self._release_abandoned()
            return self._finish(unreleased)

    def _drive_all(self, todo: list[RunSpec]) -> None:
        workers = min(self.manifest.concurrency, len(todo))
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="fanout"
        ) as pool:
            futures: list[Future[None]] = [
                pool.submit(self._drive_safely, run) for run in todo
            ]
            try:
                for future in as_completed(futures):
                    future.result()
            except KeyboardInterrupt:
                logger.warning("Interrupted: stopping every sandbox before exiting")
                self.cancel()

    def cancel(self) -> None:
        """Stop launching and stop everything running, from within this process."""
        request_cancel(self.run_dir)
        self._wake.set()

    def _prepare_for_work(self, run: RunSpec) -> bool:
        record = self._journal.load(run.run_id)
        if not record.terminal:
            return True
        if self._retry_failed and record.status != "succeeded":
            record.status = "pending"
            record.retry_base = len(record.attempts)
            record.metrics = {}
            record.error = None
            self._journal.save(record)
            return True
        return False

    def _finish(self, unreleased: list[str]) -> FanoutSummary:
        records = self._journal.records()
        report = build_report(
            name=self.manifest.name,
            digest=self.manifest.digest(),
            metric=self.manifest.report.metric,
            mode=self.manifest.report.mode,
            runs=[record.to_report_row() for record in records],
        )
        atomic_json(self.run_dir / REPORT_FILE, report)
        statuses = [record.status for record in records]
        summary = FanoutSummary(
            run_dir=self.run_dir,
            total=len(records),
            succeeded=statuses.count("succeeded"),
            failed=statuses.count("failed"),
            cancelled=statuses.count("cancelled"),
            incomplete=sum(not record.terminal for record in records),
            report=report,
            aborted_reason=self._abort_reason,
            unreleased=unreleased,
        )
        self._event(
            "fanout_finished",
            succeeded=summary.succeeded,
            failed=summary.failed,
            cancelled=summary.cancelled,
            incomplete=summary.incomplete,
            aborted_reason=summary.aborted_reason,
            unreleased=unreleased,
        )
        return summary

    def _release_abandoned(self) -> list[str]:
        """Stop what finished attempts left behind. Return what may still be running."""
        unreleased: list[str] = []
        lifetime = self.manifest.resources.sandbox_lifetime_seconds
        for record in self._journal.records():
            changed = False
            for attempt in record.attempts:
                if attempt.allocation_unknown:
                    if not _create_may_still_exist(attempt, lifetime, self._clock()):
                        continue
                    changed = True
                    if not self._resolve_unknown_create(record, attempt):
                        unreleased.append(_unresolved_create_label(record, attempt))
                        continue
                if attempt.open and attempt.allocation_id:
                    # Left running by an abort, with nothing watching it now.
                    unreleased.append(attempt.allocation_id)
                    continue
                if not attempt.needs_release:
                    continue
                self._release(record, attempt)
                changed = True
                if not attempt.released and attempt.allocation_id:
                    unreleased.append(attempt.allocation_id)
            if changed:
                self._journal.save(record)
        return unreleased

    def _resolve_unknown_create(
        self, record: RunRecord, attempt: AttemptRecord
    ) -> bool:
        """Ask again with the same key: it can only answer with the same sandbox."""
        if attempt.create_key is None:
            return False
        try:
            allocation = self.backend.allocate(
                run_id=record.run_id,
                resources=self.manifest.resources,
                idempotency_key=attempt.create_key,
            )
        except BackendError as exc:
            logger.warning("Create for %s is still unresolved: %s", record.run_id, exc)
            return False
        attempt.allocation_id = allocation.id
        attempt.root = allocation.root
        return True

    def _settle_unknown_creates(self, record: RunRecord) -> None:
        """Before another attempt, so that no sandbox idles beside its replacement."""
        lifetime = self.manifest.resources.sandbox_lifetime_seconds
        for attempt in record.attempts:
            if (
                attempt.allocation_unknown
                and _create_may_still_exist(attempt, lifetime, self._clock())
                and self._resolve_unknown_create(record, attempt)
            ):
                self._release(record, attempt)
                self._journal.save(record)

    # ------------------------------------------------------------------
    # One run
    # ------------------------------------------------------------------

    def _drive_safely(self, run: RunSpec) -> None:
        try:
            self._drive(run)
        except Exception as exc:  # noqa: BLE001 - one run must not end the others
            logger.exception("Run %s failed in the orchestrator", run.run_id)
            self._contain_failure(run, exc)

    def _contain_failure(self, run: RunSpec, exc: Exception) -> None:
        """Something unexpected escaped, so stop whatever this run still holds."""
        try:
            record = self._journal.load(run.run_id)
            for attempt in record.attempts:
                if attempt.open:
                    attempt.outcome = "infra_failed"
                    attempt.reason = f"orchestrator error: {exc}"
                    attempt.finished_at = self._clock()
                    self._release(record, attempt)
            record.status = "failed"
            record.error = f"orchestrator error: {exc}"
            self._journal.save(record)
        except Exception:  # noqa: BLE001 - a failed report must not end the others
            logger.exception("Could not record the failure of run %s", run.run_id)
        self._event("run_finished", run_id=run.run_id, status="failed")

    def _drive(self, run: RunSpec) -> None:
        if self._aborted():
            # Whatever this run holds stays as it is, for a later resume.
            return
        record = self._journal.load(run.run_id)
        # Attempts before retry_base already had their say; --retry-failed asked
        # to set them aside, not to apply their outcome again.
        if len(record.attempts) > record.retry_base:
            last = record.attempts[-1]
            if last.open:
                self._event("run_reattached", run_id=run.run_id, attempt=last.number)
                self._finish_attempt(
                    run, record, last, self._resume_attempt(run, record, last)
                )
            # A driver can die between recording an outcome and settling the run.
            # A fatal outcome says nothing about this run, so it runs again now.
            if (
                last.outcome
                and last.outcome != "fatal"
                and self._settle(run, record, last.outcome)
            ):
                return

        while True:
            if self._aborted():
                if record.status != "pending":
                    record.status = "pending"
                    self._journal.save(record)
                return
            if self._cancel_requested():
                self._conclude(run, record, "cancelled", "cancelled")
                return
            used = record.attempts_used
            if used >= self.manifest.max_attempts:
                self._conclude(
                    run,
                    record,
                    "failed",
                    (record.attempts[-1].reason if record.attempts else None)
                    or "no attempts left",
                )
                return
            if used:
                self._settle_unknown_creates(record)
                self._wait(min(_MAX_BACKOFF_S, self._retry_backoff_s * 2 ** (used - 1)))
            self._event(
                "run_started", run_id=run.run_id, attempt=len(record.attempts) + 1
            )
            outcome = self._attempt(run, record)
            if self._settle(run, record, outcome):
                return

    def _settle(self, run: RunSpec, record: RunRecord, outcome: str) -> bool:
        """Apply an attempt's outcome. True when this thread is done with the run."""
        last = record.attempts[-1] if record.attempts else None
        if outcome == "succeeded":
            record.metrics = dict(last.metrics) if last else {}
            self._conclude(run, record, "succeeded", None)
            return True
        if outcome in _FAILED_OUTCOMES:
            record.metrics = dict(last.metrics) if last else {}
            self._conclude(run, record, "failed", last.reason if last else outcome)
            return True
        if outcome == "cancelled":
            self._conclude(run, record, "cancelled", "cancelled")
            return True
        if outcome == "fatal":
            record.status = "pending"
            self._journal.save(record)
            return True
        return False

    def _conclude(
        self, run: RunSpec, record: RunRecord, status: str, error: str | None
    ) -> None:
        record.status = status
        record.error = error
        self._journal.save(record)
        self._event("run_finished", run_id=run.run_id, status=status, error=error)

    def _aborted(self) -> bool:
        """A method, because another thread can set this between any two checks."""
        return self._abort_reason is not None

    def _abort(self, reason: str) -> None:
        if self._abort_reason is None:
            self._abort_reason = reason
            self._event("fanout_aborted", reason=reason)

    def _cancel_requested(self) -> bool:
        return (self.run_dir / CANCEL_FILE).exists()

    def _wait(self, seconds: float) -> None:
        if seconds > 0:
            self._wake.wait(seconds)

    # ------------------------------------------------------------------
    # One attempt
    # ------------------------------------------------------------------

    def _attempt(self, run: RunSpec, record: RunRecord) -> str:
        """One sandbox's worth of trying. Returns the outcome.

        Only ``Exception`` is handled: on anything stronger the attempt stays
        journaled as open, so that a later resume can pick it back up.
        """
        attempt = AttemptRecord(
            number=len(record.attempts) + 1,
            create_key=uuid.uuid4().hex,
            started_at=self._clock(),
        )
        record.status = "running"
        record.attempts.append(attempt)
        # On disk before the create is sent, so that its sandbox can be found again.
        self._journal.save(record)
        outcome = self._guarded(
            run, attempt, lambda: self._allocate_and_run(run, record, attempt)
        )
        return self._finish_attempt(run, record, attempt, outcome)

    def _resume_attempt(
        self, run: RunSpec, record: RunRecord, attempt: AttemptRecord
    ) -> str:
        """Pick up an attempt a previous driver left open."""
        if attempt.allocation_id is None:
            return self._guarded(
                run, attempt, lambda: self._allocate_and_run(run, record, attempt)
            )
        if attempt.process_id is None:
            # The start may have landed, and starting again could run two copies
            # in one sandbox writing the same output. Abandon this sandbox.
            attempt.reason = "driver stopped before the process was known to start"
            return "infra_failed"
        return self._guarded(run, attempt, lambda: self._watch(run, record, attempt))

    def _guarded(
        self, run: RunSpec, attempt: AttemptRecord, step: Callable[[], str]
    ) -> str:
        """Turn whatever a step raises into an outcome, so that release still runs."""
        try:
            return step()
        except _Cancelled:
            attempt.reason = "cancelled"
            return "cancelled"
        except _CapacityWaitExceeded as exc:
            attempt.reason = f"gave up waiting for capacity: {exc}"
            return "capacity_exhausted"
        except _CreateOutcomeUnknown as exc:
            attempt.reason = (
                f"could not learn whether the sandbox was created: {exc}. If it was, "
                "its own time limit and budget bound it, and a resume will stop it."
            )
            return "infra_failed"
        except FatalBackendError as exc:
            self._abort(str(exc))
            attempt.reason = str(exc)
            return "fatal"
        except BackendError as exc:
            attempt.reason = str(exc)
            return "infra_failed"
        except Exception as exc:  # noqa: BLE001 - the sandbox must still be released
            logger.exception("Attempt %s of %s failed", attempt.number, run.run_id)
            attempt.reason = f"orchestrator error: {exc}"
            return "infra_failed"

    def _allocate_and_run(
        self, run: RunSpec, record: RunRecord, attempt: AttemptRecord
    ) -> str:
        allocation = self._allocate(run, record, attempt)
        attempt.allocation_id = allocation.id
        attempt.root = allocation.root
        # On disk before anything else touches the sandbox.
        self._journal.save(record)
        self._event(
            "attempt_started",
            run_id=run.run_id,
            attempt=attempt.number,
            allocation_id=allocation.id,
        )
        if self._cancel_requested():
            raise _Cancelled
        self._stage(run, attempt)
        time_limit = self._time_limit(run, attempt, allocation)
        attempt.time_limit_s = time_limit
        argv = self.backend.worker_argv(
            self.manifest,
            root=allocation.root,
            worker_args=[
                "--spec",
                f"{allocation.root}/spec.json",
                "--output-dir",
                f"{allocation.root}/output",
                "--stage-dir",
                f"{allocation.root}/stage",
            ],
        )
        # A 429 is a refusal, so it is safe to ask again. Anything else is ambiguous
        # and falls through to abandoning this sandbox.
        attempt.process_id = self._with_retries(
            lambda: self.backend.start(
                allocation.id,
                argv,
                env={
                    "PYTHONUNBUFFERED": "1",
                    **self.manifest.env,
                    **self._forwarded,
                },
                timeout_seconds=time_limit,
                name=f"fanout-{run.run_id}",
            ),
            retry_transient=False,
        )
        attempt.process_started_at = self._clock()
        self._journal.save(record)
        return self._watch(run, record, attempt)

    def _time_limit(
        self, run: RunSpec, attempt: AttemptRecord, allocation: Allocation
    ) -> int:
        """The manifest's limit, or as much of it as this sandbox has time for.

        A plan can silently grant less than was asked for, and a slow start uses up
        what was granted. Without this the sandbox expires first: the output is
        deleted, and the loss reads as a preemption and bills a retry.
        """
        wanted = self.manifest.resources.timeout_seconds
        if allocation.lifetime_s is None:
            return wanted
        elapsed = self._clock() - (attempt.create_sent_at or attempt.started_at)
        usable = int(allocation.lifetime_s - elapsed - _COLLECT_RESERVE_S)
        if usable >= wanted:
            return wanted
        if usable < _MIN_RUN_SECONDS:
            raise BackendError(
                f"the sandbox was granted {allocation.lifetime_s:.0f}s, leaving "
                f"{usable}s to run: check the plan's runtime limit"
            )
        self._event(
            "time_limit_shortened",
            run_id=run.run_id,
            requested=wanted,
            limit=usable,
            granted=allocation.lifetime_s,
        )
        return usable

    def _allocate(
        self, run: RunSpec, record: RunRecord, attempt: AttemptRecord
    ) -> Allocation:
        """Allocate, waiting out capacity limits and lost answers.

        A refusal releases the idempotency key on the controller, and a create that
        landed answers with its sandbox, so asking again with one key is always safe.
        """
        key = attempt.create_key or uuid.uuid4().hex
        waiting_since = self._clock()
        unknown_since: float | None = None
        delay = self._capacity_backoff_s
        announced = False
        # Once a create may have landed, later refusals prove nothing: a rate limiter
        # answers before the key is even looked at.
        ambiguous = attempt.create_maybe_landed
        while True:
            if self._cancel_requested():
                raise _Cancelled
            if not attempt.create_maybe_landed:
                attempt.create_maybe_landed = True
                attempt.create_sent_at = self._clock()
                self._journal.save(record)
            try:
                return self.backend.allocate(
                    run_id=run.run_id,
                    resources=self.manifest.resources,
                    idempotency_key=key,
                )
            except CapacityError as exc:
                if not ambiguous:
                    attempt.create_maybe_landed = False
                if self._clock() - waiting_since > self._capacity_wait_limit_s:
                    raise _CapacityWaitExceeded(str(exc)) from exc
                if not announced:
                    announced = True
                    self._event("capacity_wait", run_id=run.run_id, reason=str(exc))
                self._wait(exc.retry_after if exc.retry_after is not None else delay)
                delay = min(_MAX_BACKOFF_S, delay * 2)
            except TransientBackendError as exc:
                ambiguous = True
                now = self._clock()
                if unknown_since is None:
                    unknown_since = now
                if now - unknown_since >= self._create_unknown_wait_s:
                    raise _CreateOutcomeUnknown(str(exc)) from exc
                self._wait(min(_MAX_BACKOFF_S, self._retry_backoff_s))
            except BackendError:
                # A flat refusal: nothing was created, unless an earlier try was.
                if not ambiguous:
                    attempt.create_maybe_landed = False
                raise

    def _with_retries(
        self, operation: Callable[[], T], *, retry_transient: bool = True
    ) -> T:
        """Ride out rate limits, so that no sandbox is thrown away for one."""
        for attempt_number in range(_CONTROL_PLANE_TRIES - 1):
            delay = min(_MAX_BACKOFF_S, self._retry_backoff_s * 2**attempt_number)
            try:
                return operation()
            except CapacityError as exc:
                if self._cancel_requested():
                    raise _Cancelled from exc
                self._wait(exc.retry_after if exc.retry_after is not None else delay)
            except TransientBackendError as exc:
                if not retry_transient:
                    raise
                if self._cancel_requested():
                    raise _Cancelled from exc
                self._wait(delay)
        return operation()

    def _finish_attempt(
        self, run: RunSpec, record: RunRecord, attempt: AttemptRecord, outcome: str
    ) -> str:
        attempt.outcome = outcome
        attempt.finished_at = self._clock()
        self._journal.save(record)
        self._release(record, attempt)
        self._record_cost(run, attempt)
        self._journal.save(record)
        self._event(
            "attempt_finished",
            run_id=run.run_id,
            attempt=attempt.number,
            outcome=outcome,
            reason=attempt.reason,
            cost_cents=attempt.cost_cents,
        )
        return outcome

    def _stage(self, run: RunSpec, attempt: AttemptRecord) -> None:
        allocation_id = attempt.allocation_id or ""
        root = attempt.root

        def write(path: str, content: bytes) -> None:
            self._with_retries(
                lambda: self.backend.write_file(allocation_id, path, content)
            )

        spec = json.dumps(run.worker_spec(self.manifest), sort_keys=True).encode()
        write(f"{root}/spec.json", spec)
        for destination, source in self.manifest.stage_files().items():
            write(f"{root}/stage/{destination}", source.read_bytes())
        wheel = self.manifest.wheel_path()
        if wheel is not None:
            write(f"{root}/package/{wheel.name}", wheel.read_bytes())

    def _watch(self, run: RunSpec, record: RunRecord, attempt: AttemptRecord) -> str:
        allocation_id = attempt.allocation_id or ""
        process_id = attempt.process_id or ""
        started = (
            attempt.process_started_at
            if attempt.process_started_at is not None
            else attempt.started_at
        )
        limit = attempt.time_limit_s or self.manifest.resources.timeout_seconds
        deadline = started + limit + _DEADLINE_GRACE_S
        failures = 0
        while True:
            if self._cancel_requested():
                attempt.reason = "cancelled"
                return "cancelled"
            try:
                state = self.backend.process(allocation_id, process_id)
                self._mirror_logs(record, attempt, state)
            except AllocationLostError as exc:
                return self._classify_loss(attempt, exc)
            except FatalBackendError:
                raise
            except BackendError as exc:
                failures += 1
                if failures > self._max_poll_failures:
                    attempt.reason = f"sandbox unreachable: {exc}"
                    return "infra_failed"
                retry_after = (
                    exc.retry_after if isinstance(exc, CapacityError) else None
                )
                self._wait(
                    retry_after if retry_after is not None else self._poll_interval_s
                )
                continue
            failures = 0
            # State first: a run that finished while nobody was watching is a result,
            # however long ago its deadline passed.
            if not state.running:
                return self._collect(run, record, attempt, state)
            if self._clock() > deadline:
                attempt.reason = f"no result within {limit}s"
                return "timed_out"
            self._journal.save(record)
            self._wait(self._poll_interval_s)

    def _classify_loss(self, attempt: AttemptRecord, exc: BaseException) -> str:
        allocation_id = attempt.allocation_id or ""
        try:
            reason = self.backend.loss_reason(allocation_id)
        except BackendError:
            reason = None
        if reason == "budget_exhausted":
            attempt.reason = "stopped by the run's cost budget"
            return "budget_exhausted"
        attempt.reason = f"sandbox lost: {reason or exc}"
        return "infra_failed"

    # ------------------------------------------------------------------
    # Logs and events
    # ------------------------------------------------------------------

    def _mirror_logs(
        self, record: RunRecord, attempt: AttemptRecord, state: ProcessState
    ) -> None:
        """Mirror new log output. Logs are read only when they grew, since polling
        three endpoints per run per poll is what exhausts an org's rate limit."""
        directory = self._journal.attempt_dir(record.run_id, attempt.number)
        directory.mkdir(parents=True, exist_ok=True)
        for stream, offset_name, size in (
            ("stdout", "stdout_offset", state.stdout_bytes),
            ("stderr", "stderr_offset", state.stderr_bytes),
        ):
            # Always drain at the end: the size was read before the final flush.
            if (
                state.running
                and size is not None
                and size <= getattr(attempt, offset_name)
            ):
                continue
            for _ in range(_MAX_LOG_READS_PER_POLL):
                chunk = self.backend.logs(
                    attempt.allocation_id or "",
                    attempt.process_id or "",
                    stream,
                    getattr(attempt, offset_name),
                )
                if chunk.truncated:
                    self._append_log(
                        directory / f"{stream}.log",
                        b"[older output was rotated away]\n",
                    )
                if chunk.data:
                    self._append_log(directory / f"{stream}.log", chunk.data)
                    if stream == "stdout":
                        self._scan_events(record, attempt, chunk.data)
                setattr(attempt, offset_name, chunk.next_offset)
                if chunk.eof or not chunk.data:
                    break

    @staticmethod
    def _append_log(path: Path, data: bytes) -> None:
        with path.open("ab") as stream:
            stream.write(data)

    def _scan_events(
        self, record: RunRecord, attempt: AttemptRecord, data: bytes
    ) -> None:
        key = (record.run_id, attempt.number)
        buffered = self._partial_lines.pop(key, b"") + data
        lines = buffered.split(b"\n")
        if lines[-1]:
            self._partial_lines[key] = lines[-1]
        marker = EVENT_MARKER.encode()
        for line in lines[:-1]:
            if not line.startswith(marker):
                continue
            try:
                event = json.loads(line[len(marker) :])
            except ValueError:
                continue
            if isinstance(event, dict):
                record.last_event = event

    def _event(self, kind: str, **fields: Any) -> None:
        """Record and deliver an event. Neither may ever disturb a running fan-out:
        ``fanout run | head`` makes every later print raise, with sandboxes live."""
        event = {"ts": time.time(), "type": kind, **fields}
        with self._events_lock:
            try:
                self._journal.append_event(event)
            except OSError as exc:
                logger.warning("Could not journal an event: %s", exc)
            if self._on_event is not None:
                try:
                    self._on_event(event)
                except Exception as exc:  # noqa: BLE001 - see the docstring
                    logger.warning("Event consumer failed; dropping it: %s", exc)
                    self._on_event = None

    # ------------------------------------------------------------------
    # Collecting results
    # ------------------------------------------------------------------

    def _collect(
        self,
        run: RunSpec,
        record: RunRecord,
        attempt: AttemptRecord,
        state: ProcessState,
    ) -> str:
        attempt.exit_code = state.exit_code
        try:
            result = self._read_result(run, attempt)
            self._collect_artifacts(attempt, record)
        except AllocationLostError as exc:
            # Going while being read does not change why the run ended.
            if state.status == "timed_out":
                attempt.reason = "process exceeded its lifetime limit"
                return "timed_out"
            return self._classify_loss(attempt, exc)
        if isinstance(result, dict):
            attempt.metrics = dict(result.get("metrics") or {})
        if state.status == "timed_out":
            attempt.reason = "process exceeded its lifetime limit"
            return "timed_out"
        if not isinstance(result, dict):
            attempt.reason = (
                f"process ended ({state.status}, exit {state.exit_code}) "
                f"without a usable result{self._stderr_tail(record, attempt)}"
            )
            return "infra_failed"
        if result.get("status") == "succeeded" and state.exit_code == 0:
            return "succeeded"
        attempt.reason = result.get("error") or f"workload exited {state.exit_code}"
        return "workload_failed"

    def _read_result(self, run: RunSpec, attempt: AttemptRecord) -> Any:
        """The worker's result for this run, or None if there is no valid one.

        Retries first: giving up here discards finished work and bills it again.
        """
        allocation_id = attempt.allocation_id or ""
        path = f"{attempt.root}/output/{RESULT_FILE}"
        try:
            raw = self._with_retries(
                lambda: self.backend.read_file(allocation_id, path)
            )
        except AllocationLostError:
            raise
        except BackendError as exc:
            logger.warning("Could not read result of %s: %s", run.run_id, exc)
            return None
        if raw is None:
            return None
        try:
            result = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(result, dict) or result.get("run_id") != run.run_id:
            return None
        atomic_json(
            self._journal.attempt_dir(run.run_id, attempt.number) / RESULT_FILE, result
        )
        return result

    def _stderr_tail(self, record: RunRecord, attempt: AttemptRecord) -> str:
        path = self._journal.attempt_dir(record.run_id, attempt.number) / "stderr.log"
        try:
            size = path.stat().st_size
            with path.open("rb") as stream:
                stream.seek(max(0, size - _STDERR_TAIL_BYTES))
                tail = stream.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""
        return f"; stderr: {tail}" if tail else ""

    def _collect_artifacts(self, attempt: AttemptRecord, record: RunRecord) -> None:
        """Bring back what the manifest asked for, reporting anything that did not
        come back. The sandbox is about to be deleted, so this is the last chance,
        and on a failed run these files are often what the money bought."""
        if not self.manifest.collect:
            return
        allocation_id = attempt.allocation_id or ""
        output_root = PurePosixPath(f"{attempt.root}/output")
        destination = (
            self._journal.attempt_dir(record.run_id, attempt.number) / "artifacts"
        )
        limit = self.backend.max_read_bytes
        try:
            entries = self._with_retries(
                lambda: self.backend.list_files(allocation_id, str(output_root))
            )
        except _Cancelled:
            attempt.collect_errors.append("cancelled before everything came back")
            return
        except AllocationLostError:
            raise
        except BackendError as exc:
            attempt.collect_errors.append(f"could not list outputs: {exc}")
            return
        matched: set[str] = set()
        for entry in entries:
            try:
                relative = PurePosixPath(entry.path).relative_to(output_root)
            except ValueError:
                attempt.collect_errors.append(f"unexpected path: {entry.path}")
                continue
            if ".." in relative.parts:
                attempt.collect_errors.append(f"refused unsafe path: {entry.path}")
                continue
            name = relative.as_posix()
            patterns = [p for p in self.manifest.collect if fnmatch.fnmatch(name, p)]
            if not patterns:
                continue
            matched.update(patterns)
            if entry.size is not None and limit is not None and entry.size > limit:
                attempt.collect_errors.append(
                    f"{name} is {entry.size} bytes, above the {limit} byte limit for "
                    "leaving a sandbox: have the workload upload it itself"
                )
                continue
            try:
                content = self._with_retries(
                    partial(self.backend.read_file, allocation_id, entry.path)
                )
                if content is None:
                    attempt.collect_errors.append(f"{name} disappeared before reading")
                    continue
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            except _Cancelled:
                # The run itself finished: keep its result, and name what is missing.
                attempt.collect_errors.append("cancelled before everything came back")
                return
            except AllocationLostError:
                raise
            except (BackendError, OSError) as exc:
                attempt.collect_errors.append(f"{name}: {exc}")
                continue
            attempt.artifacts.append({"path": name, "size": len(content)})
        for pattern in self.manifest.collect:
            if pattern not in matched:
                attempt.collect_errors.append(f"no output matched {pattern!r}")

    # ------------------------------------------------------------------
    # Stopping and accounting
    # ------------------------------------------------------------------

    def _release(self, record: RunRecord, attempt: AttemptRecord) -> None:
        allocation_id = attempt.allocation_id
        if allocation_id is None or attempt.released:
            return
        # Stop first. Reads are paced, and a sandbox waiting its turn is still billing.
        error: BaseException | None = None
        for attempt_number in range(_RELEASE_TRIES):
            try:
                self.backend.release(allocation_id)
                error = None
                break
            except AllocationLostError:
                error = None
                break
            except BackendError as exc:
                error = exc
                self._wait(self._retry_backoff_s * attempt_number)
        attempt.cost_cents = self._read_cost(allocation_id)
        if error is not None:
            attempt.release_error = str(error)
            logger.error(
                "Sandbox %s of run %s did not stop and may still be billing: %s",
                allocation_id,
                record.run_id,
                error,
            )
            self._event(
                "release_failed",
                run_id=record.run_id,
                allocation_id=allocation_id,
                error=str(error),
            )
            return
        attempt.released = True
        attempt.release_error = None

    def _read_cost(self, allocation_id: str) -> float | None:
        try:
            return self.backend.cost_cents(allocation_id)
        except BackendError:
            return None

    def _record_cost(self, run: RunSpec, attempt: AttemptRecord) -> None:
        if not self.backend.metered:
            return
        if attempt.allocation_id is None and not attempt.allocation_unknown:
            return
        workload = self.manifest.workload
        agent = workload.get("agent")
        model = agent.get("model_name") if isinstance(agent, dict) else None
        gpu = self.manifest.resources.gpu
        record_entry(
            CostEntry(
                provider=self.backend.name,
                job_id=f"{self.manifest.name}/{run.run_id}/attempt-{attempt.number}",
                base_model=str(model or workload.get("kind")),
                gpu=(gpu.type or "unspecified") if gpu else "none",
                gpu_count=gpu.count if gpu else 0,
                duration_s=(
                    attempt.finished_at - attempt.started_at
                    if attempt.finished_at is not None
                    else None
                ),
                cost_usd=(
                    round(attempt.cost_cents / 100.0, 6)
                    if attempt.cost_cents is not None
                    else None
                ),
                status=attempt.outcome or "unknown",
            ),
            path=self._ledger_path,
        )
