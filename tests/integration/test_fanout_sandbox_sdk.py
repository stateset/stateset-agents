"""RL fan-out through the real sandbox SDK, over real HTTP, on real processes.

The controller is a local stand-in (see ``fake_sandbox_controller``) but the SDK,
its retries and error types, the worker, and the RL code are all the real thing.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("stateset_sandbox")

from stateset_sandbox import StateSetSandbox  # noqa: E402

from stateset_agents.fanout.manifest import FanoutManifest  # noqa: E402
from stateset_agents.fanout.orchestrator import FanoutRunner, load_records  # noqa: E402
from stateset_agents.fanout.sandbox_backend import SandboxBackend  # noqa: E402
from tests.integration.fake_sandbox_controller import (  # noqa: E402
    ApiFailure,
    FakeController,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED1 = "base-seed1"

WRITE_METRICS = (
    "import json, pathlib, sys; "
    "pathlib.Path(sys.argv[2], 'metrics.json').write_text(json.dumps("
    "{'reward': int(sys.argv[1]) * 1.0}))"
)
#: Runs until told to finish, so a test can do something to it mid-flight.
WAIT_FOR_RELEASE = (
    "import json, pathlib, sys, time; "
    "out = pathlib.Path(sys.argv[2]); "
    "(out / 'started').write_text('1'); "
    "marker = pathlib.Path(sys.argv[3]); "
    "deadline = time.time() + 60\n"
    "while not marker.exists() and time.time() < deadline:\n"
    "    time.sleep(0.02)\n"
    "(out / 'metrics.json').write_text(json.dumps({'reward': 1.0}))"
)


@pytest.fixture
def controller(tmp_path: Path) -> Iterator[FakeController]:
    with FakeController(tmp_path / "cluster") as running:
        yield running


def _backend(controller: FakeController, **kwargs: Any) -> SandboxBackend:
    # The throttle has its own tests; here it would only slow things down.
    kwargs.setdefault("requests_per_minute", 600_000)
    return SandboxBackend(
        api_key=controller.api_key, base_url=controller.base_url, **kwargs
    )


def _manifest(tmp_path: Path, **overrides: Any) -> FanoutManifest:
    data: dict[str, Any] = {
        "schema_version": 1,
        "name": "sdk-demo",
        "seeds": [1, 2, 3],
        "workload": {
            "kind": "command",
            "argv": ["{python}", "-c", WRITE_METRICS, "{seed}", "{output_dir}"],
        },
        "report": {"metric": "reward", "mode": "max"},
        "resources": {
            "cpus": "2",
            "memory": "4Gi",
            "timeout_seconds": 300,
            "resource_class": "shared",
            "image": "org:stateset-rl",
            "max_cost_cents": 10,
        },
        "concurrency": 3,
        "max_attempts": 2,
    }
    data.update(overrides)
    return FanoutManifest.from_dict(data, base_dir=tmp_path)


def _runner(
    manifest: FanoutManifest, backend: SandboxBackend, run_dir: Path, **kwargs: Any
) -> FanoutRunner:
    defaults: dict[str, Any] = {
        "poll_interval_s": 0.02,
        "retry_backoff_s": 0.0,
        "capacity_backoff_s": 0.01,
        "ledger_path": run_dir.parent / "ledger.jsonl",
        "forwarded_env": {},
    }
    defaults.update(kwargs)
    return FanoutRunner(manifest, backend, run_dir, **defaults)


def _wait_until(predicate: Any, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.02)


class TestFanOut:
    def test_each_seed_runs_in_its_own_sandbox(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        summary = _runner(
            _manifest(tmp_path), _backend(controller), tmp_path / "run"
        ).run()

        assert summary.ok, summary
        assert len(controller.sandboxes) == 3
        assert summary.report["arms"][0]["mean"] == pytest.approx(2.0)

    def test_every_sandbox_is_stopped(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        _runner(_manifest(tmp_path), _backend(controller), tmp_path / "run").run()

        assert controller.live() == []

    def test_create_carries_resources_and_a_budget(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        _runner(
            _manifest(tmp_path, seeds=[1]), _backend(controller), tmp_path / "run"
        ).run()

        body = controller.create_bodies[0]
        assert body["cpus"] == "2"
        assert body["memory"] == "4Gi"
        # The sandbox outlives the run, so a run at its limit can still be collected.
        assert body["timeout_seconds"] > 300
        assert body["resource_class"] == "shared"
        assert body["image"] == "org:stateset-rl"
        assert body["budget"] == {
            "max_cost_cents": 10,
            "max_runtime_seconds": body["timeout_seconds"],
            "on_exhausted": "stop",
        }
        assert controller.process_starts[0]["timeout_seconds"] == 300

    def test_gpu_request_reaches_the_controller(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        """The SDK's typed options have no ``gpu``, but the controller takes one."""
        manifest = _manifest(
            tmp_path,
            seeds=[1],
            resources={
                "image": "org:stateset-rl-gpu",
                "gpu": {"type": "nvidia-l4", "count": 2},
            },
        )

        _runner(manifest, _backend(controller), tmp_path / "run").run()

        assert controller.create_bodies[0]["gpu"] == {"type": "nvidia-l4", "count": 2}

    def test_measured_cost_reaches_the_report_and_ledger(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        ledger = tmp_path / "ledger.jsonl"

        summary = _runner(
            _manifest(tmp_path),
            _backend(controller),
            tmp_path / "run",
            ledger_path=ledger,
        ).run()

        assert summary.report["cost"]["known_cents"] == pytest.approx(3 * 1.25)
        entries = [json.loads(line) for line in ledger.read_text().splitlines()]
        assert {entry["provider"] for entry in entries} == {"sandbox"}
        assert all(entry["cost_usd"] == pytest.approx(0.0125) for entry in entries)

    def test_real_stub_training_runs_in_a_sandbox(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        manifest = _manifest(
            tmp_path,
            seeds=[1, 2],
            matrix={"objective": ["gspo", "gspo_token"]},
            workload={
                "kind": "train",
                "agent": {
                    "model_name": "stub://fanout",
                    "use_stub_model": True,
                    "max_new_tokens": 32,
                    "stub_responses": ["I'd be happy to help you with that!"],
                },
                "environment": {
                    "max_turns": 3,
                    "scenarios": [
                        {
                            "id": "s1",
                            "context": "c",
                            "user_responses": ["Hi", "Order 12345.", "Thanks!"],
                        }
                    ],
                },
                "training": {
                    "num_episodes": 2,
                    "allow_sequence_fallback": True,
                    "allow_unmasked_labels": True,
                    "zero_signal_patience": 0,
                    "num_generations": 1,
                    "max_steps_per_episode": 3,
                    "bf16": False,
                },
            },
            report={"metric": "eval_reward", "mode": "max"},
            concurrency=4,
        )
        run_dir = tmp_path / "run"

        summary = _runner(manifest, _backend(controller), run_dir).run()

        errors = [record.error for record in load_records(run_dir)]
        assert summary.ok, errors
        assert len(controller.sandboxes) == 4
        assert len(summary.report["arms"]) == 2


class TestSecrets:
    def test_forwarded_env_reaches_the_process_only(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        secret = "hf_top_secret"
        script = (
            "import json, os, pathlib, sys; "
            "pathlib.Path(sys.argv[1], 'metrics.json').write_text(json.dumps("
            "{'has_token': float(os.environ.get('HF_TOKEN') == sys.argv[3])}))"
        )
        manifest = _manifest(
            tmp_path,
            seeds=[1],
            forward_env=["HF_TOKEN"],
            workload={
                "kind": "command",
                "argv": [
                    "{python}",
                    "-c",
                    script,
                    "{output_dir}",
                    "{seed}",
                    secret,
                ],
            },
            report={"metric": "has_token", "mode": "max"},
        )
        # The workload needs the value to compare against, so scan for a second
        # token that only ever travels through the forwarded environment.
        forwarded = {"HF_TOKEN": secret}
        run_dir = tmp_path / "run"

        summary = _runner(
            manifest, _backend(controller), run_dir, forwarded_env=forwarded
        ).run()

        assert summary.report["run_results"][0]["metrics"]["has_token"] == 1.0
        assert "env" not in controller.create_bodies[0]
        assert controller.process_starts[0]["env"]["HF_TOKEN"] == secret

    def test_the_control_plane_key_never_enters_a_sandbox(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        run_dir = tmp_path / "run"

        _runner(_manifest(tmp_path, seeds=[1]), _backend(controller), run_dir).run()

        assert controller.api_key not in json.dumps(controller.create_bodies)
        assert controller.api_key not in json.dumps(controller.process_starts)
        for path in run_dir.rglob("*"):
            if path.is_file():
                assert controller.api_key not in path.read_text(errors="replace"), path


class TestFailures:
    def test_preempted_sandbox_retries_on_a_new_one(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        marker = tmp_path / "finish"
        manifest = _manifest(
            tmp_path,
            seeds=[1],
            workload={
                "kind": "command",
                "argv": [
                    "{python}",
                    "-c",
                    WAIT_FOR_RELEASE,
                    "{seed}",
                    "{output_dir}",
                    str(marker),
                ],
            },
        )
        run_dir = tmp_path / "run"
        result: dict[str, Any] = {}

        def drive() -> None:
            result["summary"] = _runner(manifest, _backend(controller), run_dir).run()

        thread = threading.Thread(target=drive)
        thread.start()
        _wait_until(lambda: any(s.processes for s in controller.live()))
        controller.preempt(controller.live()[0].sandbox_id)
        marker.write_text("go")
        thread.join(60)

        assert result["summary"].succeeded == 1
        record = load_records(run_dir)[0]
        assert [a.outcome for a in record.attempts] == ["infra_failed", "succeeded"]
        assert "lost" in (record.attempts[0].reason or "")
        assert len(controller.sandboxes) == 2

    def test_budget_stop_is_final(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        marker = tmp_path / "never"
        manifest = _manifest(
            tmp_path,
            seeds=[1],
            workload={
                "kind": "command",
                "argv": [
                    "{python}",
                    "-c",
                    WAIT_FOR_RELEASE,
                    "{seed}",
                    "{output_dir}",
                    str(marker),
                ],
            },
        )
        run_dir = tmp_path / "run"
        result: dict[str, Any] = {}

        def drive() -> None:
            result["summary"] = _runner(manifest, _backend(controller), run_dir).run()

        thread = threading.Thread(target=drive)
        thread.start()
        _wait_until(lambda: any(s.processes for s in controller.live()))
        controller.exhaust_budget(controller.live()[0].sandbox_id)
        thread.join(60)

        assert result["summary"].failed == 1
        assert len(controller.sandboxes) == 1
        attempt = load_records(run_dir)[0].attempts[0]
        assert attempt.outcome == "budget_exhausted"

    def test_org_concurrency_limit_is_waited_out(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        """A 429 below the requested concurrency must slow down, not fail seeds."""
        controller.concurrency_limit = 2
        manifest = _manifest(tmp_path, seeds=list(range(1, 9)), concurrency=8)

        summary = _runner(manifest, _backend(controller), tmp_path / "run").run()

        assert summary.ok, summary
        assert controller.max_live <= 2
        assert controller.concurrency_refusals > 0

    def test_bad_credentials_stop_the_whole_fan_out(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        backend = SandboxBackend(
            api_key="sk-wrong",
            base_url=controller.base_url,
            requests_per_minute=600_000,
        )
        manifest = _manifest(tmp_path, seeds=list(range(1, 21)), concurrency=1)
        run_dir = tmp_path / "run"

        summary = _runner(manifest, backend, run_dir).run()

        assert summary.aborted_reason is not None
        creates = [r for r in controller.requests if r[1].endswith("/sandbox/create")]
        assert len(creates) == 1
        assert all(record.status == "pending" for record in load_records(run_dir))

    @pytest.mark.parametrize(
        ("status", "code"),
        [
            (402, "PAYMENT_REQUIRED"),
            (403, "PLAN_UPGRADE_REQUIRED"),
            (400, "FEATURE_DISABLED"),
            (400, "INVALID_FIELD_VALUE"),
        ],
    )
    def test_refusals_no_run_can_get_past_abort_the_fan_out(
        self, tmp_path: Path, controller: FakeController, status: int, code: str
    ) -> None:
        controller.refuse_creates = ApiFailure(status, code, "refused")
        manifest = _manifest(tmp_path, seeds=[1, 2, 3], concurrency=1)

        summary = _runner(manifest, _backend(controller), tmp_path / "run").run()

        assert summary.aborted_reason is not None
        assert len(controller.create_bodies) == 1

    def test_failing_workload_is_not_retried(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        manifest = _manifest(
            tmp_path,
            seeds=[1],
            workload={
                "kind": "command",
                "argv": [
                    "{python}",
                    "-c",
                    "raise SystemExit(5)",
                    "{seed}",
                    "{output_dir}",
                ],
            },
        )
        run_dir = tmp_path / "run"

        summary = _runner(manifest, _backend(controller), run_dir).run()

        assert summary.failed == 1
        assert len(controller.sandboxes) == 1
        assert load_records(run_dir)[0].attempts[0].exit_code == 5

    def test_worker_that_cannot_start_is_retried_and_explained(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        """The bootstrap failing leaves no result: infrastructure, with its stderr."""
        manifest = _manifest(
            tmp_path,
            seeds=[1],
            package={"source": "pypi", "version": "0.0.0-does-not-exist"},
            resources={"timeout_seconds": 120},
            max_attempts=2,
        )
        run_dir = tmp_path / "run"
        backend = _backend(controller)
        original = backend.worker_argv

        def broken_bootstrap(*args: Any, **kwargs: Any) -> list[str]:
            original(*args, **kwargs)
            return ["sh", "-c", "echo 'pip: no matching distribution' >&2; exit 1"]

        backend.worker_argv = broken_bootstrap  # type: ignore[method-assign]

        summary = _runner(manifest, backend, run_dir).run()

        assert summary.failed == 1
        assert len(controller.sandboxes) == 2
        assert "no matching distribution" in (load_records(run_dir)[0].error or "")


class TestRealControllerBehavior:
    """Rules the real controller enforces and a too-friendly fake would hide."""

    def test_every_install_source_gets_past_command_validation(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        """A newline in the boot script was refused by the real controller."""
        (tmp_path / "pkg-1.0-py3-none-any.whl").write_bytes(b"not a real wheel")
        manifest = _manifest(
            tmp_path,
            seeds=[1],
            package={"source": "wheel", "wheel": "pkg-1.0-py3-none-any.whl"},
            resources={"timeout_seconds": 120},
            max_attempts=1,
        )
        run_dir = tmp_path / "run"

        summary = _runner(manifest, _backend(controller), run_dir).run()

        # The fake wheel cannot install, but the process must have been accepted.
        assert summary.aborted_reason is None, summary.aborted_reason
        assert len(controller.process_starts) == 1

    def test_a_run_at_its_time_limit_ends_with_its_output_intact(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        """Final, and collected: not retried, and not deleted with the sandbox."""
        controller.process_time_limit_s = 2.0
        script = (
            "import pathlib, sys, time; "
            "pathlib.Path(sys.argv[1], 'partial.txt').write_text('progress'); "
            "time.sleep(60)"
        )
        manifest = _manifest(
            tmp_path,
            seeds=[1],
            collect=["partial.txt"],
            workload={
                "kind": "command",
                "argv": ["{python}", "-c", script, "{output_dir}", "{seed}"],
            },
        )
        run_dir = tmp_path / "run"

        summary = _runner(manifest, _backend(controller), run_dir).run()

        assert summary.failed == 1
        assert len(controller.sandboxes) == 1, "a time limit must not be retried"
        attempt = load_records(run_dir)[0].attempts[0]
        assert attempt.outcome == "timed_out"
        collected = run_dir / "runs" / SEED1 / "attempt-1" / "artifacts" / "partial.txt"
        assert collected.read_text() == "progress"

    def test_an_expired_sandbox_is_lost_not_a_hang(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        """SANDBOX_NOT_RUNNING must read as gone, not as five minutes of polling."""
        marker = tmp_path / "finish"
        manifest = _manifest(
            tmp_path,
            seeds=[1],
            workload={
                "kind": "command",
                "argv": [
                    "{python}",
                    "-c",
                    WAIT_FOR_RELEASE,
                    "{seed}",
                    "{output_dir}",
                    str(marker),
                ],
            },
        )
        run_dir = tmp_path / "run"
        result: dict[str, Any] = {}

        def drive() -> None:
            result["summary"] = _runner(manifest, _backend(controller), run_dir).run()

        thread = threading.Thread(target=drive)
        thread.start()
        _wait_until(lambda: any(s.processes for s in controller.live()))
        controller.mark_not_running(controller.live()[0].sandbox_id)
        marker.write_text("go")
        thread.join(60)

        assert result["summary"].succeeded == 1
        assert load_records(run_dir)[0].attempts[0].outcome == "infra_failed"

    def test_a_create_whose_response_was_lost_is_adopted(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        """The controller holds the key and answers 503 until the create resolves."""
        controller.creates_in_progress = 6
        client = StateSetSandbox(
            base_url=controller.base_url,
            auth_token=controller.api_key,
            retry_base_delay_ms=1,
            retry_max_delay_ms=2,
        )
        backend = SandboxBackend(client=client, requests_per_minute=600_000)
        manifest = _manifest(tmp_path, seeds=[1], max_attempts=1)

        summary = _runner(manifest, backend, tmp_path / "run").run()

        assert summary.ok, summary
        assert len(controller.sandboxes) == 1
        keys = controller.create_keys
        assert len(set(keys)) == 1 and len(keys) > 1

    def test_a_plan_that_clamps_the_lifetime_shortens_the_run_to_fit(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        """Hobby caps runtime below what was asked, without saying so in an error."""
        controller.clamp_lifetime_s = 450
        manifest = _manifest(tmp_path, seeds=[1])
        run_dir = tmp_path / "run"

        summary = _runner(manifest, _backend(controller), run_dir).run()

        assert summary.ok, summary
        assert controller.process_starts[0]["timeout_seconds"] <= 150

    def test_ip_rate_limit_responses_wait_rather_than_burn_attempts(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        """The per-IP limiter's 429 has a plain body the SDK reads as UNKNOWN_ERROR."""
        controller.plain_429s_on_create = 3
        client = StateSetSandbox(
            base_url=controller.base_url,
            auth_token=controller.api_key,
            max_retries=0,
        )
        backend = SandboxBackend(client=client, requests_per_minute=600_000)
        run_dir = tmp_path / "run"

        summary = _runner(
            _manifest(tmp_path, seeds=[1], max_attempts=1), backend, run_dir
        ).run()

        assert summary.ok, summary
        assert len(load_records(run_dir)[0].attempts) == 1

    def test_polling_stays_near_one_request_per_poll(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        marker = tmp_path / "finish"
        manifest = _manifest(
            tmp_path,
            seeds=[1],
            workload={
                "kind": "command",
                "argv": [
                    "{python}",
                    "-c",
                    WAIT_FOR_RELEASE,
                    "{seed}",
                    "{output_dir}",
                    str(marker),
                ],
            },
        )
        run_dir = tmp_path / "run"
        result: dict[str, Any] = {}

        def drive() -> None:
            result["summary"] = _runner(manifest, _backend(controller), run_dir).run()

        thread = threading.Thread(target=drive)
        thread.start()
        _wait_until(lambda: any(s.processes for s in controller.live()))
        time.sleep(1.0)
        marker.write_text("go")
        thread.join(60)

        assert result["summary"].ok
        polls = sum(
            1
            for method, path in controller.requests
            if method == "GET" and "/processes/" in path and not path.endswith("/logs")
        )
        log_reads = sum(1 for _, path in controller.requests if path.endswith("/logs"))
        assert polls > 10
        assert log_reads < polls / 2, f"{log_reads} log reads for {polls} polls"


DRIVER = """
import sys
from pathlib import Path
from stateset_agents.fanout.manifest import FanoutManifest
from stateset_agents.fanout.orchestrator import FanoutRunner
from stateset_agents.fanout.sandbox_backend import SandboxBackend

run_dir, manifest_path, api_key, base_url = sys.argv[1:5]
FanoutRunner(
    FanoutManifest.from_file(manifest_path),
    SandboxBackend(api_key=api_key, base_url=base_url, requests_per_minute=600000),
    Path(run_dir),
    poll_interval_s=0.02,
    forwarded_env={},
    ledger_path=Path(run_dir).parent / "ledger.jsonl",
).run()
"""


class TestDriverCrash:
    def test_sigkilled_driver_resumes_without_restarting_the_run(
        self, tmp_path: Path, controller: FakeController
    ) -> None:
        """The real thing: kill -9 the driver while the sandbox keeps training."""
        marker = tmp_path / "finish"
        manifest_data = {
            "schema_version": 1,
            "name": "crash-demo",
            "seeds": [1],
            "workload": {
                "kind": "command",
                "argv": [
                    "{python}",
                    "-c",
                    WAIT_FOR_RELEASE,
                    "{seed}",
                    "{output_dir}",
                    str(marker),
                ],
            },
            "report": {"metric": "reward", "mode": "max"},
            "resources": {"image": "org:stateset-rl", "timeout_seconds": 300},
        }
        manifest_path = tmp_path / "fanout.json"
        manifest_path.write_text(json.dumps(manifest_data))
        run_dir = tmp_path / "run"
        driver = subprocess.Popen(
            [
                sys.executable,
                "-c",
                DRIVER,
                str(run_dir),
                str(manifest_path),
                controller.api_key,
                controller.base_url,
            ],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _wait_until(lambda: any(s.processes for s in controller.live()))
        _wait_until(
            lambda: load_records(run_dir)
            and load_records(run_dir)[0].attempts[-1].process_id is not None
        )
        driver.send_signal(signal.SIGKILL)
        driver.wait(timeout=30)
        driver.stdout.close()  # type: ignore[union-attr]
        driver.stderr.close()  # type: ignore[union-attr]

        record = load_records(run_dir)[0]
        assert record.attempts[0].open, "the journal must show the sandbox as in use"
        survivor = controller.live()[0]
        assert any(p.popen.poll() is None for p in survivor.processes.values())

        marker.write_text("go")
        summary = _runner(
            FanoutManifest.from_file(manifest_path),
            _backend(controller),
            run_dir,
        ).run()

        assert summary.ok, summary
        assert len(controller.sandboxes) == 1, "must reattach, not allocate again"
        assert len(controller.process_starts) == 1, "must not start the run twice"
        assert controller.live() == []
