"""``stateset-agents fanout``: the spend gate, and real local runs end to end."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.main import get_group
from typer.testing import CliRunner

from stateset_agents import cli as stateset_cli
from stateset_agents import cli_fanout
from stateset_agents.fanout.journal import load_records
from stateset_agents.fanout.manifest import FanoutManifest
from stateset_agents.fanout.orchestrator import FanoutRunner
from tests.unit.fanout_fakes import FakeBackend, Script

runner = CliRunner()

WRITE_METRICS = (
    "import json, pathlib, sys; "
    "print('log line for seed', sys.argv[1]); "
    "pathlib.Path(sys.argv[2], 'metrics.json').write_text(json.dumps("
    "{'reward': int(sys.argv[1]) * 1.0}))"
)
FAIL_ON_SEED_2 = (
    "import json, pathlib, sys; "
    "sys.exit(9) if sys.argv[1] == '2' else "
    "pathlib.Path(sys.argv[2], 'metrics.json').write_text("
    "json.dumps({'reward': 1.0}))"
)


def _write_manifest(tmp_path: Path, **overrides: Any) -> Path:
    data: dict[str, Any] = {
        "schema_version": 1,
        "name": "cli-demo",
        "seeds": [1, 2, 3],
        "workload": {
            "kind": "command",
            "argv": ["{python}", "-c", WRITE_METRICS, "{seed}", "{output_dir}"],
        },
        "report": {"metric": "reward", "mode": "max"},
        "concurrency": 3,
    }
    data.update(overrides)
    path = tmp_path / "fanout.json"
    path.write_text(json.dumps(data))
    return path


def _invoke(*args: str) -> Any:
    return runner.invoke(stateset_cli.app, ["fanout", *args])


def test_registered_on_the_main_cli() -> None:
    app: typer.Typer = stateset_cli.app
    assert "fanout" in get_group(app).commands


class TestPlan:
    def test_shows_what_would_run_and_what_it_could_cost(self, tmp_path: Path) -> None:
        path = _write_manifest(
            tmp_path,
            matrix={"x": [1, 2]},
            workload={
                "kind": "command",
                "argv": ["t", "{seed}", "{output_dir}", "{x}"],
            },
            max_attempts=2,
            resources={"max_cost_cents": 10},
        )

        result = _invoke("plan", str(path))

        assert result.exit_code == 0, result.output
        assert "6 runs" in result.output
        assert "2 arms" in result.output
        assert "3 seeds" in result.output
        # 6 runs x 2 attempts x $0.10
        assert "$1.20" in result.output

    def test_says_when_spend_is_unbounded(self, tmp_path: Path) -> None:
        result = _invoke("plan", str(_write_manifest(tmp_path)))

        assert result.exit_code == 0
        assert "max_cost_cents" in result.output

    def test_json_output_lists_every_run(self, tmp_path: Path) -> None:
        result = _invoke("plan", str(_write_manifest(tmp_path)), "--json")

        plan = json.loads(result.output)
        assert plan["total_runs"] == 3
        assert [run["seed"] for run in plan["runs"]] == [1, 2, 3]

    def test_invalid_manifest_exits_2_with_the_reason(self, tmp_path: Path) -> None:
        path = _write_manifest(tmp_path, seeds=[1, 1])

        result = _invoke("plan", str(path))

        assert result.exit_code == 2
        assert "unique" in result.output

    def test_missing_manifest_exits_2(self, tmp_path: Path) -> None:
        result = _invoke("plan", str(tmp_path / "missing.yaml"))

        assert result.exit_code == 2


class TestSpendGate:
    """Paid compute starts only against a confirmed worst case."""

    def _forbid_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(*_: Any) -> None:
            raise AssertionError("a backend was constructed before the spend gate")

        monkeypatch.setattr(cli_fanout, "_make_backend", explode)

    def test_refuses_a_fan_out_with_no_per_run_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._forbid_backend(monkeypatch)
        path = _write_manifest(tmp_path)

        result = _invoke(
            "run",
            str(path),
            "--run-dir",
            str(tmp_path / "run"),
            "--backend",
            "sandbox",
            "--confirm-max-cost-usd",
            "1000",
        )

        assert result.exit_code == 2
        assert "max_cost_cents" in result.output

    def test_refuses_without_confirmation_and_names_the_amount(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._forbid_backend(monkeypatch)
        path = _write_manifest(
            tmp_path, max_attempts=2, resources={"max_cost_cents": 50}
        )

        result = _invoke(
            "run", str(path), "--run-dir", str(tmp_path / "run"), "--backend", "sandbox"
        )

        assert result.exit_code == 2
        assert "--confirm-max-cost-usd" in result.output
        assert "$3.00" in result.output

    def test_refuses_a_confirmation_below_the_worst_case(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._forbid_backend(monkeypatch)
        path = _write_manifest(
            tmp_path, max_attempts=2, resources={"max_cost_cents": 50}
        )

        result = _invoke(
            "run",
            str(path),
            "--run-dir",
            str(tmp_path / "run"),
            "--backend",
            "sandbox",
            "--confirm-max-cost-usd",
            "2.99",
        )

        assert result.exit_code == 2
        assert "$3.00" in result.output

    def test_nothing_is_written_when_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._forbid_backend(monkeypatch)
        path = _write_manifest(tmp_path, resources={"max_cost_cents": 50})
        run_dir = tmp_path / "run"

        _invoke("run", str(path), "--run-dir", str(run_dir), "--backend", "sandbox")

        assert not run_dir.exists()

    def test_sufficient_confirmation_proceeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = FakeBackend()
        monkeypatch.setattr(cli_fanout, "_make_backend", lambda *_: backend)
        monkeypatch.setattr(
            "stateset_agents.remote.ledger.DEFAULT_LEDGER_PATH",
            tmp_path / "ledger.jsonl",
        )
        path = _write_manifest(
            tmp_path, max_attempts=2, resources={"max_cost_cents": 50}
        )

        result = _invoke(
            "run",
            str(path),
            "--run-dir",
            str(tmp_path / "run"),
            "--backend",
            "sandbox",
            "--confirm-max-cost-usd",
            "3",
            "--poll-interval",
            "0.01",
        )

        assert result.exit_code == 0, result.output
        assert len(backend.allocations) == 3

    def test_local_runs_need_no_confirmation(self, tmp_path: Path) -> None:
        result = _invoke(
            "run",
            str(_write_manifest(tmp_path)),
            "--run-dir",
            str(tmp_path / "run"),
            "--poll-interval",
            "0.02",
        )

        assert result.exit_code == 0, result.output

    def test_missing_api_key_is_explained(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("stateset_sandbox")
        monkeypatch.delenv("STATESET_API_KEY", raising=False)
        path = _write_manifest(tmp_path, resources={"max_cost_cents": 1})

        result = _invoke(
            "run",
            str(path),
            "--run-dir",
            str(tmp_path / "run"),
            "--backend",
            "sandbox",
            "--confirm-max-cost-usd",
            "1",
        )

        assert result.exit_code == 2
        assert "STATESET_API_KEY" in result.output


class TestRun:
    def test_real_local_fan_out_writes_a_report(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"

        result = _invoke(
            "run",
            str(_write_manifest(tmp_path)),
            "--run-dir",
            str(run_dir),
            "--poll-interval",
            "0.02",
        )

        assert result.exit_code == 0, result.output
        report = json.loads((run_dir / "report.json").read_text())
        assert report["arms"][0]["mean"] == pytest.approx(2.0)
        assert "single_arm" in result.output

    def test_a_failed_run_makes_the_exit_code_nonzero(self, tmp_path: Path) -> None:
        path = _write_manifest(
            tmp_path,
            workload={
                "kind": "command",
                "argv": ["{python}", "-c", FAIL_ON_SEED_2, "{seed}", "{output_dir}"],
            },
        )

        result = _invoke(
            "run",
            str(path),
            "--run-dir",
            str(tmp_path / "run"),
            "--poll-interval",
            "0.02",
        )

        assert result.exit_code == 1
        assert "failed seeds [2]" in result.output

    def test_a_billing_warning_survives_a_closed_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``fanout run | head`` must not swallow the one line that costs money."""
        from stateset_agents.fanout.backend import TransientBackendError

        backend = FakeBackend()
        backend.release_errors = [TransientBackendError("503") for _ in range(50)]
        monkeypatch.setattr(cli_fanout, "_make_backend", lambda *_: backend)
        monkeypatch.setattr(
            "stateset_agents.remote.ledger.DEFAULT_LEDGER_PATH",
            tmp_path / "ledger.jsonl",
        )

        def closed(_: Any) -> str:
            raise BrokenPipeError

        monkeypatch.setattr("stateset_agents.fanout.report.render_report", closed)

        result = _invoke(
            "run",
            str(_write_manifest(tmp_path, seeds=[1])),
            "--run-dir",
            str(tmp_path / "run"),
            "--poll-interval",
            "0.01",
        )

        assert "may still be running" in result.output

    def test_an_existing_run_directory_points_at_resume(self, tmp_path: Path) -> None:
        path = _write_manifest(tmp_path)
        run_dir = tmp_path / "run"
        _invoke("run", str(path), "--run-dir", str(run_dir), "--poll-interval", "0.02")

        result = _invoke("run", str(path), "--run-dir", str(run_dir))

        assert result.exit_code == 2
        assert "fanout resume" in result.output

    def test_unknown_backend_is_refused(self, tmp_path: Path) -> None:
        result = _invoke(
            "run",
            str(_write_manifest(tmp_path)),
            "--run-dir",
            str(tmp_path / "run"),
            "--backend",
            "lambda",
        )

        assert result.exit_code == 2
        assert "sandbox" in result.output


class TestResume:
    def test_retry_failed_reruns_only_what_failed(self, tmp_path: Path) -> None:
        marker = tmp_path / "healed"
        script = (
            "import json, pathlib, sys; "
            f"healed = pathlib.Path({str(marker)!r}).exists(); "
            "sys.exit(9) if sys.argv[1] == '2' and not healed else "
            "pathlib.Path(sys.argv[2], 'metrics.json').write_text("
            "json.dumps({'reward': 1.0}))"
        )
        path = _write_manifest(
            tmp_path,
            workload={
                "kind": "command",
                "argv": ["{python}", "-c", script, "{seed}", "{output_dir}"],
            },
        )
        run_dir = tmp_path / "run"
        first = _invoke(
            "run", str(path), "--run-dir", str(run_dir), "--poll-interval", "0.02"
        )
        assert first.exit_code == 1

        plain = _invoke("resume", str(run_dir), "--poll-interval", "0.02")
        assert plain.exit_code == 1

        marker.write_text("1")
        retried = _invoke(
            "resume", str(run_dir), "--retry-failed", "--poll-interval", "0.02"
        )

        assert retried.exit_code == 0, retried.output
        assert all(record.status == "succeeded" for record in load_records(run_dir))

    def test_paid_resume_is_gated_on_what_remains(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = FakeBackend({"base-seed1": [Script(exit_code=1)]})
        monkeypatch.setattr(cli_fanout, "_make_backend", lambda *_: backend)
        monkeypatch.setattr(
            "stateset_agents.remote.ledger.DEFAULT_LEDGER_PATH",
            tmp_path / "ledger.jsonl",
        )
        path = _write_manifest(
            tmp_path, max_attempts=2, resources={"max_cost_cents": 100}
        )
        run_dir = tmp_path / "run"
        _invoke(
            "run",
            str(path),
            "--run-dir",
            str(run_dir),
            "--backend",
            "sandbox",
            "--confirm-max-cost-usd",
            "6",
            "--poll-interval",
            "0.01",
        )
        # Fake backends record their own name; resume reads it from the journal.
        info = json.loads((run_dir / "run.json").read_text())
        assert info["backend"] == "fake"

        refused = _invoke("resume", str(run_dir), "--retry-failed")

        assert refused.exit_code == 2
        assert "$2.00" in refused.output

        accepted = _invoke(
            "resume",
            str(run_dir),
            "--retry-failed",
            "--confirm-max-cost-usd",
            "2",
            "--poll-interval",
            "0.01",
        )

        assert accepted.exit_code == 0, accepted.output

    def test_not_a_run_directory_exits_2(self, tmp_path: Path) -> None:
        result = _invoke("resume", str(tmp_path))

        assert result.exit_code == 2


class TestInspection:
    @pytest.fixture
    def run_dir(self, tmp_path: Path) -> Path:
        path = _write_manifest(tmp_path)
        run_dir = tmp_path / "run"
        result = _invoke(
            "run", str(path), "--run-dir", str(run_dir), "--poll-interval", "0.02"
        )
        assert result.exit_code == 0, result.output
        return run_dir

    def test_status_lists_every_run(self, run_dir: Path) -> None:
        result = _invoke("status", str(run_dir))

        assert result.exit_code == 0
        for seed in (1, 2, 3):
            assert f"base-seed{seed}" in result.output
        assert "succeeded" in result.output

    def test_status_json(self, run_dir: Path) -> None:
        result = _invoke("status", str(run_dir), "--json")

        status = json.loads(result.output)
        assert status["name"] == "cli-demo"
        assert status["counts"]["succeeded"] == 3
        assert len(status["runs"]) == 3

    def test_logs_show_what_the_run_printed(self, run_dir: Path) -> None:
        result = _invoke("logs", str(run_dir), "base-seed2")

        assert result.exit_code == 0
        assert "log line for seed 2" in result.output

    def test_logs_for_an_unknown_run_exits_2(self, run_dir: Path) -> None:
        result = _invoke("logs", str(run_dir), "nope")

        assert result.exit_code == 2

    def test_logs_path_cannot_escape_the_run_directory(self, run_dir: Path) -> None:
        result = _invoke("logs", str(run_dir), "../../etc")

        assert result.exit_code == 2

    def test_report_renders_from_the_journal(self, run_dir: Path) -> None:
        result = _invoke("report", str(run_dir))

        assert result.exit_code == 0
        assert "cli-demo" in result.output
        assert "Verdict" in result.output

    def test_report_json(self, run_dir: Path) -> None:
        result = _invoke("report", str(run_dir), "--json")

        assert json.loads(result.output)["name"] == "cli-demo"

    def test_status_of_a_non_run_directory_exits_2(self, tmp_path: Path) -> None:
        result = _invoke("status", str(tmp_path))

        assert result.exit_code == 2


class TestCancel:
    def _orphaned_run(self, tmp_path: Path) -> tuple[Path, FakeBackend]:
        """A driver died mid-run: a sandbox is live and nothing is watching it."""

        class Crash(BaseException):
            pass

        backend = FakeBackend()

        def crash(allocation_id: str) -> None:
            raise Crash

        backend.on_poll = crash
        run_dir = tmp_path / "run"
        manifest = FanoutManifest.from_dict(
            {
                "schema_version": 1,
                "name": "orphan",
                "seeds": [1, 2],
                "workload": {
                    "kind": "command",
                    "argv": ["t", "{seed}", "{output_dir}"],
                },
                "report": {"metric": "reward", "mode": "max"},
            }
        )
        fan = FanoutRunner(
            manifest,
            backend,
            run_dir,
            forwarded_env={},
            ledger_path=tmp_path / "ledger.jsonl",
        )
        fan._journal.start(manifest, backend.name)
        fan._journal.ensure_runs(manifest.expand())
        with pytest.raises(Crash):
            fan._attempt(manifest.expand()[0], fan._journal.load("base-seed1"))
        backend.on_poll = None
        return run_dir, backend

    def test_with_no_driver_it_stops_orphaned_sandboxes_itself(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir, backend = self._orphaned_run(tmp_path)
        monkeypatch.setattr(cli_fanout, "_make_backend", lambda *_: backend)

        result = _invoke("cancel", str(run_dir))

        assert result.exit_code == 0, result.output
        assert backend.released == ["alloc-1"]
        statuses = {record.run_id: record.status for record in load_records(run_dir)}
        assert statuses == {"base-seed1": "cancelled", "base-seed2": "cancelled"}
        assert "alloc-1" in result.output

    def test_reports_a_sandbox_that_will_not_stop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from stateset_agents.fanout.backend import TransientBackendError

        run_dir, backend = self._orphaned_run(tmp_path)
        backend.release_errors = [TransientBackendError("503") for _ in range(20)]
        monkeypatch.setattr(cli_fanout, "_make_backend", lambda *_: backend)

        result = _invoke("cancel", str(run_dir))

        assert result.exit_code == 1
        assert "alloc-1" in result.output

    def test_not_a_run_directory_exits_2(self, tmp_path: Path) -> None:
        result = _invoke("cancel", str(tmp_path))

        assert result.exit_code == 2
