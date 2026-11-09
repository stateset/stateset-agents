"""Orchestration: what happens to N independent runs when things go wrong."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from stateset_agents.fanout.backend import (
    AllocationLostError,
    BackendError,
    CapacityError,
    FatalBackendError,
    TransientBackendError,
)
from stateset_agents.fanout.journal import AttemptRecord
from stateset_agents.fanout.manifest import FanoutManifest
from stateset_agents.fanout.orchestrator import (
    FanoutRunner,
    cancel_run,
    load_records,
    remaining_worst_case_usd,
    request_cancel,
)
from tests.unit.fanout_fakes import FakeBackend, Script


def _manifest(**overrides: Any) -> FanoutManifest:
    data: dict[str, Any] = {
        "schema_version": 1,
        "name": "demo",
        "seeds": [1, 2, 3],
        "workload": {
            "kind": "command",
            "argv": ["t", "{seed}", "{output_dir}"],
        },
        "report": {"metric": "reward", "mode": "max"},
        "concurrency": 2,
        "max_attempts": 2,
        "resources": {"timeout_seconds": 600},
    }
    data.update(overrides)
    return FanoutManifest.from_dict(data, base_dir=overrides.get("_base", "."))


def _runner(
    manifest: FanoutManifest, backend: FakeBackend, run_dir: Path, **kwargs: Any
) -> FanoutRunner:
    defaults: dict[str, Any] = {
        "poll_interval_s": 0.001,
        "retry_backoff_s": 0.0,
        "capacity_backoff_s": 0.0,
        "ledger_path": run_dir.parent / "ledger.jsonl",
        "forwarded_env": {},
    }
    defaults.update(kwargs)
    return FanoutRunner(manifest, backend, run_dir, **defaults)


def _status(run_dir: Path, run_id: str) -> str:
    return next(r for r in load_records(run_dir) if r.run_id == run_id).status


def _record(run_dir: Path, run_id: str) -> Any:
    return next(r for r in load_records(run_dir) if r.run_id == run_id)


SEED1 = "base-seed1"


class TestHappyPath:
    def test_each_run_gets_its_own_sandbox_and_all_are_released(
        self, tmp_path: Path
    ) -> None:
        backend = FakeBackend()

        summary = _runner(_manifest(), backend, tmp_path / "run").run()

        assert summary.ok
        assert summary.succeeded == 3
        assert len(backend.allocations) == 3
        assert sorted(backend.released) == sorted(a for _, a in backend.allocations)

    def test_concurrency_limit_is_never_exceeded(self, tmp_path: Path) -> None:
        backend = FakeBackend(
            {f"base-seed{seed}": [Script(polls=3)] for seed in range(1, 9)}
        )
        manifest = _manifest(seeds=list(range(1, 9)), concurrency=3)

        summary = _runner(manifest, backend, tmp_path / "run").run()

        assert summary.succeeded == 8
        assert 1 <= backend.max_live <= 3

    def test_writes_a_report_with_each_runs_metrics(self, tmp_path: Path) -> None:
        backend = FakeBackend(
            {
                "base-seed1": [Script(metrics={"reward": 1.0})],
                "base-seed2": [Script(metrics={"reward": 2.0})],
                "base-seed3": [Script(metrics={"reward": 3.0})],
            }
        )

        summary = _runner(_manifest(), backend, tmp_path / "run").run()

        report = json.loads((tmp_path / "run" / "report.json").read_text())
        assert report == summary.report
        assert report["arms"][0]["mean"] == pytest.approx(2.0)
        assert report["runs"]["succeeded"] == 3

    def test_uploads_the_spec_before_starting(self, tmp_path: Path) -> None:
        backend = FakeBackend()

        _runner(_manifest(seeds=[9]), backend, tmp_path / "run").run()

        allocation = backend.allocations[0][1]
        spec = json.loads(backend.written[allocation]["/root/base-seed9/spec.json"])
        assert spec["seed"] == 9
        assert spec["run_id"] == "base-seed9"
        assert spec["workload"]["kind"] == "command"

    def test_stages_files_and_the_wheel(self, tmp_path: Path) -> None:
        (tmp_path / "data.txt").write_text("hello")
        (tmp_path / "pkg.whl").write_bytes(b"wheel")
        manifest = FanoutManifest.from_dict(
            {
                "schema_version": 1,
                "name": "demo",
                "seeds": [1],
                "workload": {
                    "kind": "command",
                    "argv": ["t", "{seed}", "{output_dir}"],
                },
                "report": {"metric": "reward", "mode": "max"},
                "stage": {"sub/data.txt": "data.txt"},
                "package": {"source": "wheel", "wheel": "pkg.whl"},
            },
            base_dir=tmp_path,
        )
        backend = FakeBackend()

        _runner(manifest, backend, tmp_path / "run").run()

        written = backend.written[backend.allocations[0][1]]
        assert written["/root/base-seed1/stage/sub/data.txt"] == b"hello"
        assert written["/root/base-seed1/package/pkg.whl"] == b"wheel"

    def test_worker_args_point_at_the_run_layout(self, tmp_path: Path) -> None:
        backend = FakeBackend()

        _runner(_manifest(seeds=[1]), backend, tmp_path / "run").run()

        argv = backend.started[0]["argv"]
        assert argv == [
            "worker",
            "--spec",
            "/root/base-seed1/spec.json",
            "--output-dir",
            "/root/base-seed1/output",
            "--stage-dir",
            "/root/base-seed1/stage",
        ]

    def test_process_lifetime_comes_from_the_manifest(self, tmp_path: Path) -> None:
        backend = FakeBackend()

        _runner(_manifest(seeds=[1]), backend, tmp_path / "run").run()

        assert backend.started[0]["timeout_seconds"] == 600


class TestRetries:
    def test_lost_sandbox_retries_on_a_fresh_one(self, tmp_path: Path) -> None:
        backend = FakeBackend(
            {
                SEED1: [
                    Script(poll_errors=[AllocationLostError("preempted")]),
                    Script(),
                ]
            }
        )
        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.succeeded == 1
        record = _record(run_dir, SEED1)
        assert [a.outcome for a in record.attempts] == ["infra_failed", "succeeded"]
        assert len(backend.allocations) == 2
        assert len(backend.released) == 2

    def test_retries_stop_at_the_attempt_limit(self, tmp_path: Path) -> None:
        lost = lambda: Script(poll_errors=[AllocationLostError("gone")])  # noqa: E731
        backend = FakeBackend({SEED1: [lost(), lost(), lost()]})
        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1], max_attempts=2), backend, run_dir).run()

        assert summary.failed == 1
        assert len(backend.allocations) == 2
        assert "gone" in (_record(run_dir, SEED1).error or "")

    def test_workload_failure_is_not_retried(self, tmp_path: Path) -> None:
        """A failing seed fails identically on a new sandbox; retrying only bills."""
        backend = FakeBackend({SEED1: [Script(exit_code=3)]})
        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.failed == 1
        assert len(backend.allocations) == 1
        record = _record(run_dir, SEED1)
        assert record.attempts[0].outcome == "workload_failed"
        assert record.attempts[0].exit_code == 3

    def test_exit_without_a_result_means_the_worker_never_ran(
        self, tmp_path: Path
    ) -> None:
        """Bootstrap failures are infrastructure: a retry can help."""
        backend = FakeBackend({SEED1: [Script(exit_code=1, result=None), Script()]})
        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.succeeded == 1
        assert _record(run_dir, SEED1).attempts[0].outcome == "infra_failed"

    def test_clean_exit_without_a_result_is_not_a_success(self, tmp_path: Path) -> None:
        backend = FakeBackend({SEED1: [Script(result=None), Script(result=None)]})

        summary = _runner(_manifest(seeds=[1]), backend, tmp_path / "run").run()

        assert summary.failed == 1

    def test_garbled_result_is_not_a_success(self, tmp_path: Path) -> None:
        backend = FakeBackend(
            {SEED1: [Script(result=b"{not json"), Script(result=b"{not json")]}
        )

        summary = _runner(_manifest(seeds=[1]), backend, tmp_path / "run").run()

        assert summary.failed == 1

    def test_result_from_a_different_run_is_rejected(self, tmp_path: Path) -> None:
        wrong = {
            "run_id": "other",
            "status": "succeeded",
            "exit_code": 0,
            "metrics": {},
        }
        backend = FakeBackend({SEED1: [Script(result=wrong), Script(result=wrong)]})

        summary = _runner(_manifest(seeds=[1]), backend, tmp_path / "run").run()

        assert summary.failed == 1

    def test_staging_failure_retries_and_releases(self, tmp_path: Path) -> None:
        backend = FakeBackend(
            {SEED1: [Script(write_error=BackendError("disk full")), Script()]}
        )

        summary = _runner(_manifest(seeds=[1]), backend, tmp_path / "run").run()

        assert summary.succeeded == 1
        assert len(backend.released) == 2

    def test_ambiguous_start_never_starts_twice_in_one_sandbox(
        self, tmp_path: Path
    ) -> None:
        """The request may have reached the server: only a new sandbox is safe."""
        backend = FakeBackend(
            {SEED1: [Script(start_error=TransientBackendError("timeout")), Script()]}
        )

        summary = _runner(_manifest(seeds=[1]), backend, tmp_path / "run").run()

        assert summary.succeeded == 1
        assert len(backend.allocations) == 2
        assert len(backend.started) == 1
        assert backend.started[0]["allocation_id"] == backend.allocations[1][1]

    def test_transient_poll_failures_are_ridden_out(self, tmp_path: Path) -> None:
        errors = [TransientBackendError("503") for _ in range(4)]
        backend = FakeBackend({SEED1: [Script(poll_errors=errors)]})

        summary = _runner(_manifest(seeds=[1]), backend, tmp_path / "run").run()

        assert summary.succeeded == 1
        assert len(backend.allocations) == 1

    def test_unreachable_sandbox_eventually_counts_as_lost(
        self, tmp_path: Path
    ) -> None:
        errors = [TransientBackendError("503") for _ in range(10)]
        backend = FakeBackend({SEED1: [Script(poll_errors=errors), Script()]})

        summary = _runner(
            _manifest(seeds=[1]), backend, tmp_path / "run", max_poll_failures=3
        ).run()

        assert summary.succeeded == 1
        assert len(backend.allocations) == 2


class TestSpendControl:
    def test_budget_exhaustion_is_final(self, tmp_path: Path) -> None:
        """Retrying a run that hit its ceiling would just spend the ceiling again."""
        backend = FakeBackend(
            {
                SEED1: [
                    Script(
                        poll_errors=[AllocationLostError("sandbox gone")],
                        loss_reason="budget_exhausted",
                    )
                ]
            }
        )
        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.failed == 1
        assert len(backend.allocations) == 1
        assert _record(run_dir, SEED1).attempts[0].outcome == "budget_exhausted"

    def test_exceeding_the_deadline_stops_the_run(self, tmp_path: Path) -> None:
        backend = FakeBackend({SEED1: [Script(polls=10_000)]})
        now = [0.0]

        def clock() -> float:
            now[0] += 100.0
            return now[0]

        run_dir = tmp_path / "run"
        summary = _runner(_manifest(seeds=[1]), backend, run_dir, clock=clock).run()

        assert summary.failed == 1
        assert _record(run_dir, SEED1).attempts[0].outcome == "timed_out"
        assert len(backend.allocations) == 1
        assert backend.released

    def test_process_timeout_is_not_retried(self, tmp_path: Path) -> None:
        backend = FakeBackend(
            {SEED1: [Script(status="timed_out", exit_code=None, result=None)]}
        )
        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.failed == 1
        assert len(backend.allocations) == 1
        assert _record(run_dir, SEED1).attempts[0].outcome == "timed_out"

    def test_cost_is_recorded_per_attempt_and_summed(self, tmp_path: Path) -> None:
        backend = FakeBackend(
            {
                SEED1: [
                    Script(poll_errors=[AllocationLostError("x")], cost_cents=1.5),
                    Script(cost_cents=2.5),
                ]
            }
        )
        run_dir = tmp_path / "run"
        ledger = tmp_path / "ledger.jsonl"

        summary = _runner(
            _manifest(seeds=[1]), backend, run_dir, ledger_path=ledger
        ).run()

        assert summary.report["cost"]["known_cents"] == pytest.approx(4.0)
        entries = [json.loads(line) for line in ledger.read_text().splitlines()]
        assert [e["cost_usd"] for e in entries] == [0.015, 0.025]
        assert entries[0]["provider"] == "fake"

    def test_unknown_cost_stays_unknown(self, tmp_path: Path) -> None:
        backend = FakeBackend({SEED1: [Script(cost_cents=None)]})
        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.report["cost"]["runs_with_unknown_cost"] == 1
        assert _record(run_dir, SEED1).cost_cents is None


class TestCapacity:
    def test_capacity_errors_wait_without_using_an_attempt(
        self, tmp_path: Path
    ) -> None:
        full = [CapacityError("limit reached") for _ in range(5)]
        backend = FakeBackend({SEED1: [Script(allocate_errors=full)]})
        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1], max_attempts=1), backend, run_dir).run()

        assert summary.succeeded == 1
        assert backend.allocate_calls == 6
        assert len(_record(run_dir, SEED1).attempts) == 1

    def test_capacity_wait_is_bounded(self, tmp_path: Path) -> None:
        now = [0.0]

        def clock() -> float:
            now[0] += 1000.0
            return now[0]

        full = [CapacityError("limit") for _ in range(50)]
        backend = FakeBackend({SEED1: [Script(allocate_errors=full)]})
        run_dir = tmp_path / "run"

        summary = _runner(
            _manifest(seeds=[1]),
            backend,
            run_dir,
            clock=clock,
            capacity_wait_limit_s=2500.0,
        ).run()

        assert summary.failed == 1
        assert "capacity" in (_record(run_dir, SEED1).error or "")


class TestCircuitBreaker:
    def test_fatal_error_stops_launching_and_leaves_runs_resumable(
        self, tmp_path: Path
    ) -> None:
        """Bad credentials would fail all N; do not make N doomed requests."""
        fatal = FatalBackendError("invalid API key")
        backend = FakeBackend(
            {
                f"base-seed{seed}": [Script(allocate_errors=[fatal])]
                for seed in range(1, 21)
            }
        )
        run_dir = tmp_path / "run"
        manifest = _manifest(seeds=list(range(1, 21)), concurrency=1)

        summary = _runner(manifest, backend, run_dir).run()

        assert "invalid API key" in (summary.aborted_reason or "")
        assert not summary.ok
        assert backend.allocate_calls == 1
        assert all(r.status == "pending" for r in load_records(run_dir))

    def test_fatal_during_a_run_still_releases_its_sandbox(
        self, tmp_path: Path
    ) -> None:
        backend = FakeBackend(
            {SEED1: [Script(start_error=FatalBackendError("forbidden"))]}
        )

        summary = _runner(_manifest(seeds=[1]), backend, tmp_path / "run").run()

        assert summary.aborted_reason is not None
        assert backend.released == [backend.allocations[0][1]]


class TestResume:
    def _crash_after_start(self, tmp_path: Path) -> tuple[Path, FakeBackend]:
        """Run until the process starts, then die the way a killed driver would."""

        class Crash(BaseException):
            pass

        backend = FakeBackend({SEED1: [Script(polls=1), Script()]})

        def crash(allocation_id: str) -> None:
            raise Crash

        backend.on_poll = crash
        run_dir = tmp_path / "run"
        runner = _runner(_manifest(seeds=[1]), backend, run_dir)
        runner._journal.start(runner.manifest, backend.name)
        runner._journal.ensure_runs(runner.manifest.expand())
        record = runner._journal.load(SEED1)
        with pytest.raises(Crash):
            runner._attempt(runner.manifest.expand()[0], record)
        return run_dir, backend

    def test_interrupted_run_keeps_its_sandbox_for_reattach(
        self, tmp_path: Path
    ) -> None:
        run_dir, _ = self._crash_after_start(tmp_path)

        attempt = _record(run_dir, SEED1).attempts[0]

        assert attempt.allocation_id == "alloc-1"
        assert attempt.process_id == "proc-alloc-1"
        assert attempt.open

    def test_resume_reattaches_instead_of_starting_again(self, tmp_path: Path) -> None:
        run_dir, backend = self._crash_after_start(tmp_path)
        backend.on_poll = None
        starts_before = len(backend.started)
        allocations_before = len(backend.allocations)

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.succeeded == 1
        assert len(backend.started) == starts_before
        assert len(backend.allocations) == allocations_before
        assert "alloc-1" in backend.released

    def test_sandbox_gone_after_a_crash_counts_as_a_failed_attempt(
        self, tmp_path: Path
    ) -> None:
        run_dir, backend = self._crash_after_start(tmp_path)
        backend.on_poll = None
        backend.live["alloc-1"].script.poll_errors = [AllocationLostError("expired")]

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.succeeded == 1
        assert len(backend.allocations) == 2

    def test_crash_before_start_never_double_starts(self, tmp_path: Path) -> None:
        """We cannot know whether the start landed, so that sandbox is abandoned."""

        class Crash(BaseException):
            pass

        backend = FakeBackend()

        def crash(allocation_id: str, path: str) -> None:
            raise Crash

        backend.on_write = crash
        run_dir = tmp_path / "run"
        runner = _runner(_manifest(seeds=[1]), backend, run_dir)
        runner._journal.start(runner.manifest, backend.name)
        runner._journal.ensure_runs(runner.manifest.expand())
        with pytest.raises(Crash):
            runner._attempt(runner.manifest.expand()[0], runner._journal.load(SEED1))
        backend.on_write = None

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.succeeded == 1
        assert "alloc-1" in backend.released
        assert len(backend.allocations) == 2
        assert backend.started[0]["allocation_id"] == "alloc-2"

    def test_allocation_is_journaled_before_anything_else_happens(
        self, tmp_path: Path
    ) -> None:
        """Otherwise a crash here leaks a sandbox nothing knows about."""
        run_dir = tmp_path / "run"
        observed: list[str | None] = []
        backend = FakeBackend()

        def check(allocation_id: str, path: str) -> None:
            if not observed:
                record = _record(run_dir, SEED1)
                observed.append(record.attempts[-1].allocation_id)

        backend.on_write = check

        _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert observed == ["alloc-1"]

    def test_finished_runs_are_not_repeated(self, tmp_path: Path) -> None:
        backend = FakeBackend()
        run_dir = tmp_path / "run"
        _runner(_manifest(), backend, run_dir).run()
        before = len(backend.allocations)

        summary = _runner(_manifest(), backend, run_dir).run()

        assert summary.succeeded == 3
        assert len(backend.allocations) == before

    def test_failed_runs_stay_failed_by_default(self, tmp_path: Path) -> None:
        backend = FakeBackend({SEED1: [Script(exit_code=1)]})
        run_dir = tmp_path / "run"
        _runner(_manifest(seeds=[1]), backend, run_dir).run()

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.failed == 1
        assert len(backend.allocations) == 1

    def test_retry_failed_gets_a_fresh_attempt_budget(self, tmp_path: Path) -> None:
        backend = FakeBackend({SEED1: [Script(exit_code=1), Script()]})
        run_dir = tmp_path / "run"
        _runner(_manifest(seeds=[1], max_attempts=1), backend, run_dir).run()

        summary = _runner(
            _manifest(seeds=[1], max_attempts=1), backend, run_dir, retry_failed=True
        ).run()

        assert summary.succeeded == 1
        record = _record(run_dir, SEED1)
        assert [a.outcome for a in record.attempts] == ["workload_failed", "succeeded"]

    def test_a_different_manifest_is_refused(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        _runner(_manifest(), FakeBackend(), run_dir).run()

        with pytest.raises(ValueError, match="different manifest"):
            _runner(_manifest(seeds=[1, 2, 99]), FakeBackend(), run_dir).run()

    def test_changing_only_concurrency_is_allowed(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        _runner(_manifest(concurrency=1), FakeBackend(), run_dir).run()

        summary = _runner(_manifest(concurrency=9), FakeBackend(), run_dir).run()

        assert summary.succeeded == 3

    def test_two_drivers_cannot_share_a_run_directory(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        inner_error: list[Exception] = []
        backend = FakeBackend()

        def try_second(allocation_id: str) -> None:
            if not inner_error:
                try:
                    _runner(_manifest(seeds=[1]), FakeBackend(), run_dir).run()
                except ValueError as exc:
                    inner_error.append(exc)

        backend.on_poll = try_second

        _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert inner_error
        assert "owns" in str(inner_error[0])

    def test_remaining_worst_case_excludes_finished_runs(self, tmp_path: Path) -> None:
        manifest = _manifest(
            seeds=[1, 2, 3], max_attempts=2, resources={"max_cost_cents": 100}
        )
        backend = FakeBackend({SEED1: [Script(exit_code=1)]})
        run_dir = tmp_path / "run"
        runner = _runner(manifest, backend, run_dir)
        runner.run()

        assert remaining_worst_case_usd(manifest, run_dir) == 0.0
        # Only the failed seed would run again, at up to two attempts.
        assert remaining_worst_case_usd(
            manifest, run_dir, retry_failed=True
        ) == pytest.approx(2.0)


class TestCancel:
    def test_cancel_stops_running_sandboxes_and_pending_runs(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        backend = FakeBackend(
            {f"base-seed{seed}": [Script(polls=100_000)] for seed in range(1, 5)}
        )
        cancelled = threading.Event()

        def cancel_once(allocation_id: str) -> None:
            if not cancelled.is_set():
                cancelled.set()
                request_cancel(run_dir)

        backend.on_poll = cancel_once
        manifest = _manifest(seeds=[1, 2, 3, 4], concurrency=1)

        summary = _runner(manifest, backend, run_dir).run()

        assert summary.cancelled == 4
        assert summary.succeeded == 0
        assert len(backend.allocations) == 1
        assert len(backend.released) == 1

    def test_resume_clears_a_previous_cancel_request(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        request_cancel(run_dir)

        summary = _runner(_manifest(seeds=[1]), FakeBackend(), run_dir).run()

        assert summary.succeeded == 1

    def test_cancelled_runs_rerun_only_when_asked(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        backend = FakeBackend({SEED1: [Script(polls=100_000), Script()]})
        backend.on_poll = lambda allocation_id: request_cancel(run_dir)
        _runner(_manifest(seeds=[1]), backend, run_dir).run()
        backend.on_poll = None

        again = _runner(_manifest(seeds=[1]), backend, run_dir).run()
        assert again.cancelled == 1

        retried = _runner(
            _manifest(seeds=[1]), backend, run_dir, retry_failed=True
        ).run()
        assert retried.succeeded == 1


class TestReleaseFailures:
    def test_transient_release_errors_are_retried(self, tmp_path: Path) -> None:
        backend = FakeBackend()
        backend.release_errors = [TransientBackendError("503")]

        summary = _runner(_manifest(seeds=[1]), backend, tmp_path / "run").run()

        assert summary.ok
        assert backend.released

    def test_a_sandbox_that_will_not_stop_is_surfaced(self, tmp_path: Path) -> None:
        """It may still be billing. Reporting success here would hide that."""
        backend = FakeBackend()
        backend.release_errors = [TransientBackendError("503") for _ in range(50)]
        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.succeeded == 1
        assert summary.unreleased == ["alloc-1"]
        assert not summary.ok
        assert _record(run_dir, SEED1).attempts[0].released is False


class TestSecrets:
    def test_forwarded_env_reaches_the_process_but_never_the_disk(
        self, tmp_path: Path
    ) -> None:
        secret = "hf_super_secret_value"
        backend = FakeBackend()
        run_dir = tmp_path / "run"
        manifest = _manifest(seeds=[1], forward_env=["HF_TOKEN"])

        _runner(manifest, backend, run_dir, forwarded_env={"HF_TOKEN": secret}).run()

        assert backend.started[0]["env"]["HF_TOKEN"] == secret
        for path in run_dir.rglob("*"):
            if path.is_file():
                assert secret not in path.read_text(errors="replace"), path

    def test_missing_forwarded_variable_stops_before_any_allocation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        backend = FakeBackend()
        manifest = _manifest(seeds=[1], forward_env=["HF_TOKEN"])

        with pytest.raises(ValueError, match="HF_TOKEN"):
            FanoutRunner(manifest, backend, tmp_path / "run").run()

        assert backend.allocate_calls == 0

    def test_manifest_env_reaches_the_process(self, tmp_path: Path) -> None:
        backend = FakeBackend()

        _runner(
            _manifest(seeds=[1], env={"LOG_LEVEL": "debug"}), backend, tmp_path / "run"
        ).run()

        assert backend.started[0]["env"]["LOG_LEVEL"] == "debug"
        assert backend.started[0]["env"]["PYTHONUNBUFFERED"] == "1"


class TestLogsAndEvents:
    def test_logs_are_mirrored_locally(self, tmp_path: Path) -> None:
        backend = FakeBackend(
            {SEED1: [Script(stdout=[b"one\n", b"two\n"], stderr=[b"oops\n"])]}
        )
        run_dir = tmp_path / "run"

        _runner(_manifest(seeds=[1]), backend, run_dir).run()

        attempt_dir = run_dir / "runs" / SEED1 / "attempt-1"
        assert (attempt_dir / "stdout.log").read_bytes() == b"one\ntwo\n"
        assert (attempt_dir / "stderr.log").read_bytes() == b"oops\n"

    def test_progress_events_survive_being_split_across_chunks(
        self, tmp_path: Path
    ) -> None:
        line = b'STATESET_FANOUT_EVENT={"type":"episode","episode":3}\n'
        backend = FakeBackend(
            {SEED1: [Script(stdout=[b"noise\n" + line[:20], line[20:]])]}
        )
        run_dir = tmp_path / "run"

        _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert _record(run_dir, SEED1).last_event == {"type": "episode", "episode": 3}

    def test_a_malformed_event_line_is_ignored(self, tmp_path: Path) -> None:
        backend = FakeBackend(
            {SEED1: [Script(stdout=[b"STATESET_FANOUT_EVENT={broken\n"])]}
        )

        summary = _runner(_manifest(seeds=[1]), backend, tmp_path / "run").run()

        assert summary.succeeded == 1

    def test_lifecycle_events_are_journaled_and_delivered(self, tmp_path: Path) -> None:
        seen: list[dict[str, Any]] = []
        run_dir = tmp_path / "run"

        _runner(
            _manifest(seeds=[1]), FakeBackend(), run_dir, on_event=seen.append
        ).run()

        kinds = [event["type"] for event in seen]
        assert "run_started" in kinds
        assert "run_finished" in kinds
        written = [
            json.loads(line)
            for line in (run_dir / "events.jsonl").read_text().splitlines()
        ]
        assert [event["type"] for event in written] == kinds


class TestArtifacts:
    def test_matching_files_come_back_and_others_do_not(self, tmp_path: Path) -> None:
        backend = FakeBackend(
            {
                SEED1: [
                    Script(
                        files={
                            "ckpt/model.bin": b"weights",
                            "scratch/tmp.bin": b"junk",
                        }
                    )
                ]
            }
        )
        run_dir = tmp_path / "run"

        _runner(_manifest(seeds=[1], collect=["ckpt/*"]), backend, run_dir).run()

        artifacts = run_dir / "runs" / SEED1 / "attempt-1" / "artifacts"
        assert (artifacts / "ckpt" / "model.bin").read_bytes() == b"weights"
        assert not (artifacts / "scratch").exists()

    def test_failed_runs_still_return_what_was_paid_for(self, tmp_path: Path) -> None:
        backend = FakeBackend(
            {SEED1: [Script(exit_code=1, files={"ckpt.bin": b"weights"})]}
        )
        run_dir = tmp_path / "run"

        _runner(_manifest(seeds=[1], collect=["*.bin"]), backend, run_dir).run()

        artifacts = run_dir / "runs" / SEED1 / "attempt-1" / "artifacts"
        assert (artifacts / "ckpt.bin").read_bytes() == b"weights"

    def test_a_path_escaping_the_output_is_not_written(self, tmp_path: Path) -> None:
        class Hostile(FakeBackend):
            def list_files(self, allocation_id: str, path: str) -> list[Any]:
                from stateset_agents.fanout.backend import FileEntry

                return [FileEntry(path=f"{path}/../../../../evil.txt", size=4)]

            def read_file(self, allocation_id: str, path: str) -> bytes | None:
                if path.endswith("evil.txt"):
                    return b"evil"
                return super().read_file(allocation_id, path)

        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1], collect=["*"]), Hostile(), run_dir).run()

        assert summary.succeeded == 1
        assert not list(tmp_path.rglob("evil.txt"))

    def test_collection_failure_keeps_the_runs_result(self, tmp_path: Path) -> None:
        class Flaky(FakeBackend):
            def list_files(self, allocation_id: str, path: str) -> list[Any]:
                raise TransientBackendError("503")

        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1], collect=["*"]), Flaky(), run_dir).run()

        assert summary.succeeded == 1
        assert "503" in _record(run_dir, SEED1).attempts[0].collect_errors[0]


class TestTimeLimitAccounting:
    def test_waiting_for_capacity_does_not_eat_the_time_limit(
        self, tmp_path: Path
    ) -> None:
        """The limit is for the run. An hour in the queue must not kill it on start."""
        now = [0.0]

        def clock() -> float:
            return now[0]

        def queue_for_a_long_time(run_id: str, key: str) -> None:
            now[0] += 2000.0

        full = [CapacityError("limit")]
        backend = FakeBackend({SEED1: [Script(allocate_errors=full, polls=2)]})
        backend.on_allocate = queue_for_a_long_time
        run_dir = tmp_path / "run"

        summary = _runner(
            _manifest(seeds=[1]),
            backend,
            run_dir,
            clock=clock,
            capacity_wait_limit_s=100_000.0,
        ).run()

        assert summary.succeeded == 1
        assert _record(run_dir, SEED1).attempts[0].outcome == "succeeded"

    def test_a_run_that_finished_while_the_driver_was_away_is_not_a_timeout(
        self, tmp_path: Path
    ) -> None:
        """Laptop closed overnight, the run finished: resume must collect it."""

        class Crash(BaseException):
            pass

        backend = FakeBackend({SEED1: [Script(polls=1)]})

        def crash(allocation_id: str) -> None:
            raise Crash

        backend.on_poll = crash
        run_dir = tmp_path / "run"
        now = [0.0]
        runner = _runner(_manifest(seeds=[1]), backend, run_dir, clock=lambda: now[0])
        runner._journal.start(runner.manifest, backend.name)
        runner._journal.ensure_runs(runner.manifest.expand())
        with pytest.raises(Crash):
            runner._attempt(runner.manifest.expand()[0], runner._journal.load(SEED1))
        backend.on_poll = None
        backend.live["alloc-1"].polls_seen = 1
        now[0] = 1_000_000.0

        summary = _runner(
            _manifest(seeds=[1]), backend, run_dir, clock=lambda: now[0]
        ).run()

        assert summary.succeeded == 1
        assert _record(run_dir, SEED1).attempts[0].outcome == "succeeded"


class TestControlPlaneLimits:
    """Rate limits reset within a minute; a sandbox must not be thrown away for one."""

    def test_rate_limit_while_staging_keeps_the_sandbox(self, tmp_path: Path) -> None:
        backend = FakeBackend(
            {SEED1: [Script(write_errors=[CapacityError("429", retry_after=0.0)])]}
        )

        summary = _runner(
            _manifest(seeds=[1], max_attempts=1), backend, tmp_path / "run"
        ).run()

        assert summary.succeeded == 1
        assert len(backend.allocations) == 1

    def test_rate_limited_start_is_asked_again_in_the_same_sandbox(
        self, tmp_path: Path
    ) -> None:
        """A 429 is a refusal, so unlike a timeout it cannot have started anything."""
        backend = FakeBackend(
            {SEED1: [Script(start_errors=[CapacityError("429", retry_after=0.0)])]}
        )

        summary = _runner(
            _manifest(seeds=[1], max_attempts=1), backend, tmp_path / "run"
        ).run()

        assert summary.succeeded == 1
        assert len(backend.allocations) == 1
        assert len(backend.started) == 1

    def test_a_finished_run_is_not_discarded_over_a_failed_read(
        self, tmp_path: Path
    ) -> None:
        """Otherwise a rate limit throws away finished work and bills it again."""
        errors = [TransientBackendError("503"), CapacityError("429", retry_after=0.0)]
        backend = FakeBackend({SEED1: [Script(read_errors=errors)]})
        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.succeeded == 1
        assert len(backend.allocations) == 1

    def test_rate_limits_while_polling_do_not_lose_the_sandbox(
        self, tmp_path: Path
    ) -> None:
        errors = [CapacityError("429", retry_after=0.0) for _ in range(10)]
        backend = FakeBackend({SEED1: [Script(poll_errors=errors)]})

        summary = _runner(
            _manifest(seeds=[1]), backend, tmp_path / "run", max_poll_failures=20
        ).run()

        assert summary.succeeded == 1
        assert len(backend.allocations) == 1

    def test_logs_are_read_only_when_they_grew(self, tmp_path: Path) -> None:
        """Three requests per poll per run is what exhausts an org's rate limit."""
        backend = FakeBackend(
            {SEED1: [Script(polls=50, stdout=[b"line\n"], report_log_bytes=True)]}
        )
        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.succeeded == 1
        assert backend.log_reads < 8
        assert (run_dir / "runs" / SEED1 / "attempt-1" / "stdout.log").read_bytes() == (
            b"line\n"
        )

    def test_logs_are_always_drained_at_the_end(self, tmp_path: Path) -> None:
        backend = FakeBackend(
            {SEED1: [Script(polls=0, stdout=[b"final\n"], report_log_bytes=True)]}
        )
        run_dir = tmp_path / "run"

        _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert (run_dir / "runs" / SEED1 / "attempt-1" / "stdout.log").read_bytes() == (
            b"final\n"
        )


class TestAmbiguousCreates:
    """A lost connection can hide a create that landed. It must not become an orphan."""

    def test_the_key_reaches_disk_before_the_create_is_sent(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        seen: list[tuple[str, str | None]] = []
        backend = FakeBackend()

        def check(run_id: str, key: str) -> None:
            seen.append((key, _record(run_dir, run_id).attempts[-1].create_key))

        backend.on_allocate = check

        _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert seen and all(sent == stored for sent, stored in seen)

    def test_a_lost_connection_adopts_the_sandbox_it_created(
        self, tmp_path: Path
    ) -> None:
        errors = [TransientBackendError("503 in progress") for _ in range(3)]
        backend = FakeBackend({SEED1: [Script(ambiguous_errors=errors)]})
        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1], max_attempts=1), backend, run_dir).run()

        assert summary.succeeded == 1
        assert len(backend.allocations) == 1
        assert len(set(backend.create_keys)) == 1

    def test_a_retry_after_a_real_failure_uses_a_new_key(self, tmp_path: Path) -> None:
        backend = FakeBackend(
            {SEED1: [Script(poll_errors=[AllocationLostError("gone")]), Script()]}
        )

        _runner(_manifest(seeds=[1]), backend, tmp_path / "run").run()

        assert len(set(backend.create_keys)) == 2

    def test_a_driver_killed_during_create_finds_the_sandbox_on_resume(
        self, tmp_path: Path
    ) -> None:
        class Crash(BaseException):
            pass

        backend = FakeBackend()
        run_dir = tmp_path / "run"

        def land_then_die(run_id: str, key: str) -> None:
            # The controller finishes the create; the driver never hears back.
            backend.on_allocate = None
            backend.allocate(
                run_id=run_id,
                resources=_manifest(seeds=[1]).resources,
                idempotency_key=key,
            )
            raise Crash

        backend.on_allocate = land_then_die
        runner = _runner(_manifest(seeds=[1]), backend, run_dir)
        runner._journal.start(runner.manifest, backend.name)
        runner._journal.ensure_runs(runner.manifest.expand())
        with pytest.raises(Crash):
            runner._attempt(runner.manifest.expand()[0], runner._journal.load(SEED1))

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.succeeded == 1
        assert len(backend.allocations) == 1, "must adopt it, not create another"
        assert backend.released == ["alloc-1"]

    def test_an_unresolved_create_is_reported_and_its_cost_is_unknown(
        self, tmp_path: Path
    ) -> None:
        errors = [TransientBackendError("503 in progress") for _ in range(5000)]
        backend = FakeBackend({SEED1: [Script(ambiguous_errors=errors)]})
        run_dir = tmp_path / "run"

        summary = _runner(
            _manifest(seeds=[1], max_attempts=1),
            backend,
            run_dir,
            create_unknown_wait_s=0.0,
        ).run()

        assert summary.failed == 1
        assert summary.unreleased, "it may exist, so it must not read as clean"
        assert not summary.ok
        record = _record(run_dir, SEED1)
        assert record.attempts[0].allocation_unknown
        assert record.cost_cents is None, "unknown, never free"

    def test_a_later_resume_stops_the_sandbox_once_the_create_resolves(
        self, tmp_path: Path
    ) -> None:
        errors = [TransientBackendError("503 in progress") for _ in range(2)]
        backend = FakeBackend({SEED1: [Script(ambiguous_errors=errors)]})
        run_dir = tmp_path / "run"
        manifest = _manifest(seeds=[1], max_attempts=1)

        first = _runner(manifest, backend, run_dir, create_unknown_wait_s=0.0).run()
        assert first.unreleased

        second = _runner(manifest, backend, run_dir, create_unknown_wait_s=0.0).run()

        assert not second.unreleased
        assert backend.released == ["alloc-1"]
        assert _record(run_dir, SEED1).attempts[0].allocation_unknown is False


class TestContainment:
    def test_a_broken_event_consumer_does_not_lose_a_sandbox(
        self, tmp_path: Path
    ) -> None:
        """``fanout run | head`` makes every later print raise BrokenPipeError."""

        def broken(event: dict[str, Any]) -> None:
            raise BrokenPipeError

        backend = FakeBackend()

        summary = _runner(
            _manifest(seeds=[1, 2]), backend, tmp_path / "run", on_event=broken
        ).run()

        assert summary.ok, summary
        assert sorted(backend.released) == sorted(a for _, a in backend.allocations)

    def test_an_unexpected_error_releases_what_the_run_holds(
        self, tmp_path: Path
    ) -> None:
        backend = FakeBackend()
        run_dir = tmp_path / "run"
        runner = _runner(_manifest(seeds=[1]), backend, run_dir)
        original = runner._journal.save
        failures = {"left": 1}

        def fail_when_recording_the_outcome(record: Any) -> None:
            if record.attempts and record.attempts[-1].outcome and failures["left"]:
                failures["left"] = 0
                raise RuntimeError("disk full")
            original(record)

        runner._journal.save = fail_when_recording_the_outcome  # type: ignore[method-assign]

        summary = runner.run()

        assert backend.released == ["alloc-1"]
        assert not summary.unreleased

    def test_a_non_backend_error_on_reattach_is_contained(self, tmp_path: Path) -> None:
        class Crash(BaseException):
            pass

        class Odd(FakeBackend):
            explode = False

            def process(self, allocation_id: str, process_id: str) -> Any:
                if self.explode:
                    raise RuntimeError("bug in a backend")
                return super().process(allocation_id, process_id)

        backend = Odd()

        def crash(allocation_id: str) -> None:
            raise Crash

        backend.on_poll = crash
        run_dir = tmp_path / "run"
        runner = _runner(_manifest(seeds=[1], max_attempts=1), backend, run_dir)
        runner._journal.start(runner.manifest, backend.name)
        runner._journal.ensure_runs(runner.manifest.expand())
        with pytest.raises(Crash):
            runner._attempt(runner.manifest.expand()[0], runner._journal.load(SEED1))
        backend.on_poll = None
        backend.explode = True

        summary = _runner(_manifest(seeds=[1], max_attempts=1), backend, run_dir).run()

        assert summary.failed == 1
        assert backend.released == ["alloc-1"]

    def test_fatal_mid_run_can_be_resumed(self, tmp_path: Path) -> None:
        """Fixing the credentials must be enough: the run cannot stay stuck."""
        backend = FakeBackend(
            {SEED1: [Script(start_error=FatalBackendError("forbidden")), Script()]}
        )
        run_dir = tmp_path / "run"
        first = _runner(_manifest(seeds=[1], max_attempts=1), backend, run_dir).run()
        assert first.aborted_reason is not None

        second = _runner(_manifest(seeds=[1], max_attempts=1), backend, run_dir).run()

        assert second.succeeded == 1
        assert second.aborted_reason is None


class TestLostArtifacts:
    """A run that lost what was asked for must not read as clean."""

    def test_a_file_too_large_to_read_is_named_and_the_rest_still_come_back(
        self, tmp_path: Path
    ) -> None:
        backend = FakeBackend(
            {SEED1: [Script(files={"big.bin": b"x" * 2048, "small.bin": b"y"})]}
        )
        backend.max_read_bytes = 1024
        run_dir = tmp_path / "run"

        summary = _runner(
            _manifest(seeds=[1], collect=["*.bin"]), backend, run_dir
        ).run()

        assert summary.succeeded == 1
        assert not summary.ok
        artifacts = run_dir / "runs" / SEED1 / "attempt-1" / "artifacts"
        assert (artifacts / "small.bin").read_bytes() == b"y"
        assert not (artifacts / "big.bin").exists()
        errors = _record(run_dir, SEED1).attempts[0].collect_errors
        assert any("big.bin" in error for error in errors)

    def test_one_failed_read_does_not_stop_the_others(self, tmp_path: Path) -> None:
        class OneBad(FakeBackend):
            def read_file(self, allocation_id: str, path: str) -> bytes | None:
                if path.endswith("bad.bin"):
                    raise BackendError("unreadable")
                return super().read_file(allocation_id, path)

        backend = OneBad({SEED1: [Script(files={"bad.bin": b"x", "good.bin": b"y"})]})
        run_dir = tmp_path / "run"

        _runner(_manifest(seeds=[1], collect=["*.bin"]), backend, run_dir).run()

        artifacts = run_dir / "runs" / SEED1 / "attempt-1" / "artifacts"
        assert (artifacts / "good.bin").exists()

    def test_a_pattern_matching_nothing_is_reported(self, tmp_path: Path) -> None:
        """A typo in ``collect`` would otherwise lose the checkpoint silently."""
        backend = FakeBackend({SEED1: [Script(files={"model.bin": b"x"})]})
        run_dir = tmp_path / "run"

        summary = _runner(
            _manifest(seeds=[1], collect=["chekpoint/*"]), backend, run_dir
        ).run()

        assert not summary.ok
        errors = _record(run_dir, SEED1).attempts[0].collect_errors
        assert any("chekpoint/*" in error for error in errors)

    def test_the_report_names_what_did_not_come_back(self, tmp_path: Path) -> None:
        backend = FakeBackend({SEED1: [Script(files={"big.bin": b"x" * 2048})]})
        backend.max_read_bytes = 1024

        summary = _runner(
            _manifest(seeds=[1], collect=["*.bin"]), backend, tmp_path / "run"
        ).run()

        assert summary.report["artifacts_missing"][0]["run_id"] == SEED1
        assert "big.bin" in summary.report["artifacts_missing"][0]["errors"][0]


class TestCancelAccuracy:
    def test_cancel_keeps_a_success_the_driver_died_before_recording(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        backend = FakeBackend()
        runner = _runner(_manifest(seeds=[1]), backend, run_dir)
        runner._journal.start(runner.manifest, backend.name)
        runner._journal.ensure_runs(runner.manifest.expand())
        runner._attempt(runner.manifest.expand()[0], runner._journal.load(SEED1))
        # The attempt finished, but the run was never settled.
        assert _record(run_dir, SEED1).status == "running"

        cancel_run(run_dir, lambda: backend)

        record = _record(run_dir, SEED1)
        assert record.status == "succeeded"
        assert record.metrics == {"reward": 1.0}

    def test_cancel_reports_a_create_it_cannot_resolve(self, tmp_path: Path) -> None:
        class Crash(BaseException):
            pass

        backend = FakeBackend()

        def die(run_id: str, key: str) -> None:
            raise Crash

        backend.on_allocate = die
        run_dir = tmp_path / "run"
        runner = _runner(_manifest(seeds=[1]), backend, run_dir)
        runner._journal.start(runner.manifest, backend.name)
        runner._journal.ensure_runs(runner.manifest.expand())
        with pytest.raises(Crash):
            runner._attempt(runner.manifest.expand()[0], runner._journal.load(SEED1))

        outcome = cancel_run(run_dir, lambda: backend)

        assert outcome.unreleased
        assert _record(run_dir, SEED1).attempts[0].allocation_unknown


def _crash_on_allocate(tmp_path: Path) -> tuple[Path, FakeBackend]:
    """The driver dies with a create on the wire. The create landed."""

    class Crash(BaseException):
        pass

    backend = FakeBackend()
    run_dir = tmp_path / "run"

    def land_then_die(run_id: str, key: str) -> None:
        backend.on_allocate = None
        backend.allocate(
            run_id=run_id,
            resources=_manifest(seeds=[1]).resources,
            idempotency_key=key,
        )
        raise Crash

    backend.on_allocate = land_then_die
    runner = _runner(_manifest(seeds=[1]), backend, run_dir)
    runner._journal.start(runner.manifest, backend.name)
    runner._journal.ensure_runs(runner.manifest.expand())
    with pytest.raises(Crash):
        runner._attempt(runner.manifest.expand()[0], runner._journal.load(SEED1))
    return run_dir, backend


class TestLandedCreatesAreNeverForgotten:
    """Once a create may have landed, it stays on the books until settled."""

    def test_a_cancel_after_a_lost_answer_still_stops_the_sandbox(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        backend = FakeBackend(
            {SEED1: [Script(ambiguous_errors=[TransientBackendError("timeout")])]}
        )
        backend.on_allocate = lambda run_id, key: request_cancel(run_dir)

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.cancelled == 1
        assert backend.released == ["alloc-1"]
        assert not summary.unreleased

    def test_a_cancel_during_resume_of_a_crashed_create_still_stops_it(
        self, tmp_path: Path
    ) -> None:
        run_dir, backend = _crash_on_allocate(tmp_path)

        def cancel_once_started(event: dict[str, Any]) -> None:
            if event["type"] == "fanout_started":
                request_cancel(run_dir)

        summary = _runner(
            _manifest(seeds=[1]), backend, run_dir, on_event=cancel_once_started
        ).run()

        assert summary.cancelled == 1
        assert backend.released == ["alloc-1"]

    def test_an_unexpected_create_failure_is_settled_by_asking_again(
        self, tmp_path: Path
    ) -> None:
        """Nothing proved it did not land, so its key is asked about, not forgotten."""
        backend = FakeBackend(
            {SEED1: [Script(allocate_errors=[RuntimeError("odd")]), Script()]}
        )
        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.succeeded == 1
        first = _record(run_dir, SEED1).attempts[0]
        assert first.outcome == "infra_failed"
        assert first.create_maybe_landed
        assert first.allocation_id == "alloc-1"
        assert "alloc-1" in backend.released

    def test_a_flat_refusal_leaves_nothing_to_settle(self, tmp_path: Path) -> None:
        """A 4xx releases the key: nothing was created, so the books stay clean."""
        backend = FakeBackend(
            {SEED1: [Script(allocate_errors=[BackendError("400")]), Script()]}
        )
        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.ok, summary
        record = _record(run_dir, SEED1)
        assert not record.attempts[0].allocation_unknown
        assert record.cost_cents == pytest.approx(2.0)

    def test_a_rate_limit_after_a_lost_answer_proves_nothing(
        self, tmp_path: Path
    ) -> None:
        """A limiter answers before the key is even looked at."""
        errors: list[Exception] = [
            TransientBackendError("timeout"),
            CapacityError("429", retry_after=0.0),
        ]
        run_dir = tmp_path / "run"
        backend = FakeBackend({SEED1: [Script(ambiguous_errors=errors)]})

        def cancel_on_the_third_try(run_id: str, key: str) -> None:
            if backend.allocate_calls == 2:
                request_cancel(run_dir)

        backend.on_allocate = cancel_on_the_third_try

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.cancelled == 1
        assert backend.released == ["alloc-1"]

    def test_cancel_ignores_a_create_that_was_never_sent(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        backend = FakeBackend()
        runner = _runner(_manifest(seeds=[1]), backend, run_dir)
        runner._journal.start(runner.manifest, backend.name)
        runner._journal.ensure_runs(runner.manifest.expand())
        record = runner._journal.load(SEED1)
        record.attempts.append(AttemptRecord(number=1, create_key="k"))
        runner._journal.save(record)

        outcome = cancel_run(run_dir, lambda: backend)

        assert outcome.unreleased == []
        assert not _record(run_dir, SEED1).attempts[0].allocation_unknown

    def test_an_unknown_create_is_settled_before_the_next_attempt(
        self, tmp_path: Path
    ) -> None:
        """Otherwise the first sandbox idles, billing, beside its replacement."""
        backend = FakeBackend(
            {
                SEED1: [
                    Script(ambiguous_errors=[TransientBackendError("timeout")]),
                    Script(),
                ]
            }
        )

        summary = _runner(
            _manifest(seeds=[1]),
            backend,
            tmp_path / "run",
            create_unknown_wait_s=0.0,
        ).run()

        assert summary.ok, summary
        assert backend.released == ["alloc-1", "alloc-2"]
        assert backend.max_live == 1

    def test_a_stuck_create_stops_mattering_once_it_must_have_expired(
        self, tmp_path: Path
    ) -> None:
        """A controller 500 can hold a key forever. A sandbox cannot outlive itself."""
        stuck = [TransientBackendError("503 in progress") for _ in range(5000)]
        backend = FakeBackend({SEED1: [Script(ambiguous_errors=stuck), Script()]})
        run_dir = tmp_path / "run"
        manifest = _manifest(seeds=[1])

        first = _runner(manifest, backend, run_dir, create_unknown_wait_s=0.0).run()
        assert first.succeeded == 1
        assert first.unreleased

        later = _runner(
            manifest, backend, run_dir, clock=lambda: time.time() + 10_000_000
        ).run()

        assert later.ok, later
        assert _record(run_dir, SEED1).cost_cents is None, "unknown, never free"

    def test_the_spend_gate_counts_a_create_still_to_settle(
        self, tmp_path: Path
    ) -> None:
        """Settling one may start a sandbox, for as long as it takes to stop it."""
        stuck = [TransientBackendError("503") for _ in range(5000)]
        backend = FakeBackend({SEED1: [Script(ambiguous_errors=stuck), Script()]})
        run_dir = tmp_path / "run"
        manifest = _manifest(seeds=[1], resources={"max_cost_cents": 100})

        _runner(manifest, backend, run_dir, create_unknown_wait_s=0.0).run()

        assert remaining_worst_case_usd(manifest, run_dir) == pytest.approx(1.0)


class TestLifetimeThePlatformGranted:
    """A plan can silently clamp a sandbox's lifetime below what was asked for."""

    def test_a_clamped_sandbox_shortens_the_run_to_fit(self, tmp_path: Path) -> None:
        backend = FakeBackend({SEED1: [Script(lifetime_s=1000.0)]})
        seen: list[dict[str, Any]] = []

        summary = _runner(
            _manifest(seeds=[1], resources={"timeout_seconds": 3600}),
            backend,
            tmp_path / "run",
            on_event=seen.append,
        ).run()

        assert summary.succeeded == 1
        limit = backend.started[0]["timeout_seconds"]
        assert 600 <= limit <= 700, "room to collect before the sandbox expires"
        assert any(event["type"] == "time_limit_shortened" for event in seen)

    def test_a_full_lifetime_changes_nothing(self, tmp_path: Path) -> None:
        backend = FakeBackend({SEED1: [Script(lifetime_s=4200.0)]})
        seen: list[dict[str, Any]] = []

        _runner(
            _manifest(seeds=[1], resources={"timeout_seconds": 3600}),
            backend,
            tmp_path / "run",
            on_event=seen.append,
        ).run()

        assert backend.started[0]["timeout_seconds"] == 3600
        assert not any(event["type"] == "time_limit_shortened" for event in seen)

    def test_too_little_time_to_do_anything_starts_nothing(
        self, tmp_path: Path
    ) -> None:
        backend = FakeBackend(
            {SEED1: [Script(lifetime_s=100.0), Script(lifetime_s=100.0)]}
        )

        summary = _runner(_manifest(seeds=[1]), backend, tmp_path / "run").run()

        assert summary.failed == 1
        assert backend.started == []

    def test_the_deadline_follows_the_shortened_limit(self, tmp_path: Path) -> None:
        backend = FakeBackend({SEED1: [Script(lifetime_s=1000.0, polls=100_000)]})
        now = [0.0]

        def clock() -> float:
            now[0] += 50.0
            return now[0]

        run_dir = tmp_path / "run"

        _runner(
            _manifest(seeds=[1], resources={"timeout_seconds": 3600}),
            backend,
            run_dir,
            clock=clock,
        ).run()

        attempt = _record(run_dir, SEED1).attempts[0]
        assert attempt.outcome == "timed_out"
        assert attempt.time_limit_s <= 700
        assert attempt.finished_at < 1500.0


class TestStoppingPromptly:
    def test_a_sandbox_is_stopped_before_anything_is_read_from_it(
        self, tmp_path: Path
    ) -> None:
        """Reads are paced. A sandbox waiting its turn is a sandbox still billing."""
        backend = FakeBackend()

        _runner(_manifest(seeds=[1]), backend, tmp_path / "run").run()

        assert backend.calls[0] == "release"

    def test_a_cancel_during_allocation_skips_staging(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        backend = FakeBackend()
        backend.on_allocate = lambda run_id, key: request_cancel(run_dir)

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.cancelled == 1
        assert backend.written == {}
        assert backend.started == []
        assert backend.released == ["alloc-1"]

    def test_retries_notice_a_cancel_from_another_process(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"

        class Limited(FakeBackend):
            writes = 0

            def write_file(self, allocation_id: str, path: str, content: bytes) -> None:
                self.writes += 1
                request_cancel(run_dir)
                raise CapacityError("429", retry_after=0.0)

        backend = Limited()

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.cancelled == 1
        assert backend.writes == 1

    def test_a_cancel_mid_collection_keeps_the_result(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"

        class Limited(FakeBackend):
            def read_file(self, allocation_id: str, path: str) -> bytes | None:
                if path.endswith(".bin"):
                    request_cancel(run_dir)
                    raise CapacityError("429", retry_after=0.0)
                return super().read_file(allocation_id, path)

        backend = Limited({SEED1: [Script(files={"a.bin": b"x"})]})

        summary = _runner(
            _manifest(seeds=[1], collect=["*.bin"]), backend, run_dir
        ).run()

        record = _record(run_dir, SEED1)
        assert record.status == "succeeded"
        assert record.metrics == {"reward": 1.0}
        assert record.attempts[0].collect_errors
        assert not summary.ok


class TestAbortKeepsTheBooksStraight:
    def _crash_after_start(
        self, tmp_path: Path, scripts: dict[str, list[Script]]
    ) -> tuple[Path, FakeBackend]:
        class Crash(BaseException):
            pass

        backend = FakeBackend(scripts)

        def crash(allocation_id: str) -> None:
            raise Crash

        backend.on_poll = crash
        run_dir = tmp_path / "run"
        manifest = _manifest(seeds=[1, 2], concurrency=1)
        runner = _runner(manifest, backend, run_dir)
        runner._journal.start(manifest, backend.name)
        runner._journal.ensure_runs(manifest.expand())
        target = manifest.expand()[1]
        with pytest.raises(Crash):
            runner._attempt(target, runner._journal.load(target.run_id))
        backend.on_poll = None
        return run_dir, backend

    def test_a_reattached_success_counts_even_if_an_abort_lands_meanwhile(
        self, tmp_path: Path
    ) -> None:
        run_dir, backend = self._crash_after_start(tmp_path, {})
        manifest = _manifest(seeds=[1, 2], concurrency=2)
        runner = _runner(manifest, backend, run_dir)
        backend.on_poll = lambda allocation_id: runner._abort("forbidden")

        runner.run()

        record = _record(run_dir, "base-seed2")
        assert record.status == "succeeded"
        assert record.metrics == {"reward": 1.0}

    def test_a_failed_attempt_during_an_abort_leaves_the_run_pending(
        self, tmp_path: Path
    ) -> None:
        backend = FakeBackend({SEED1: [Script(poll_errors=[AllocationLostError("x")])]})
        run_dir = tmp_path / "run"
        runner = _runner(_manifest(seeds=[1]), backend, run_dir)
        backend.on_poll = lambda allocation_id: runner._abort("forbidden")

        runner.run()

        assert _record(run_dir, SEED1).status == "pending"

    def test_after_an_abort_a_sandbox_left_running_is_reported(
        self, tmp_path: Path
    ) -> None:
        """Nothing is watching it now, so it must not read as clean."""
        fatal = FatalBackendError("forbidden")
        run_dir, backend = self._crash_after_start(
            tmp_path, {SEED1: [Script(allocate_errors=[fatal])]}
        )
        held = backend.allocations[0][1]
        manifest = _manifest(seeds=[1, 2], concurrency=1)

        summary = _runner(manifest, backend, run_dir).run()

        assert summary.aborted_reason is not None
        assert held in summary.unreleased
        assert held not in backend.released


class TestLossDuringCollection:
    def test_a_budget_stop_while_reading_the_result_is_still_final(
        self, tmp_path: Path
    ) -> None:
        backend = FakeBackend(
            {
                SEED1: [
                    Script(
                        read_errors=[AllocationLostError("gone")],
                        loss_reason="budget_exhausted",
                    ),
                    Script(),
                ]
            }
        )
        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.failed == 1
        assert len(backend.allocations) == 1
        assert _record(run_dir, SEED1).attempts[0].outcome == "budget_exhausted"

    def test_a_time_limit_is_final_even_if_the_sandbox_goes_while_collecting(
        self, tmp_path: Path
    ) -> None:
        backend = FakeBackend(
            {
                SEED1: [
                    Script(
                        status="timed_out",
                        exit_code=None,
                        read_errors=[AllocationLostError("gone")],
                    ),
                    Script(),
                ]
            }
        )
        run_dir = tmp_path / "run"

        summary = _runner(_manifest(seeds=[1]), backend, run_dir).run()

        assert summary.failed == 1
        assert len(backend.allocations) == 1
        assert _record(run_dir, SEED1).attempts[0].outcome == "timed_out"
