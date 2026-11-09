"""End to end on real processes: real worker, real RL updates, real parallelism.

Nothing is mocked here. Each run is its own OS process in its own directory, the
same worker module a sandbox would launch.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from stateset_agents.fanout.local_backend import LocalBackend
from stateset_agents.fanout.manifest import FanoutManifest
from stateset_agents.fanout.orchestrator import FanoutRunner, cancel_run, load_records

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]

WRITE_METRICS = (
    "import json, pathlib, sys, time; "
    "time.sleep(float(sys.argv[3])); "
    "pathlib.Path(sys.argv[2], 'metrics.json').write_text(json.dumps("
    "{'reward': int(sys.argv[1]) * float(sys.argv[4])}))"
)


def _command_manifest(tmp_path: Path, **overrides: Any) -> FanoutManifest:
    data: dict[str, Any] = {
        "schema_version": 1,
        "name": "local-demo",
        "seeds": [1, 2, 3],
        "matrix": {"scale": [1.0, 10.0]},
        "workload": {
            "kind": "command",
            "argv": [
                "{python}",
                "-c",
                WRITE_METRICS,
                "{seed}",
                "{output_dir}",
                "0.0",
                "{scale}",
            ],
        },
        "report": {"metric": "reward", "mode": "max"},
        "concurrency": 3,
    }
    data.update(overrides)
    return FanoutManifest.from_dict(data, base_dir=tmp_path)


def _run(manifest: FanoutManifest, run_dir: Path, **kwargs: Any) -> Any:
    return FanoutRunner(
        manifest,
        LocalBackend(run_dir / "scratch"),
        run_dir,
        poll_interval_s=0.05,
        retry_backoff_s=0.0,
        capacity_backoff_s=0.0,
        forwarded_env={},
        **kwargs,
    ).run()


def test_matrix_of_real_processes_rolls_up_into_a_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    summary = _run(_command_manifest(tmp_path), run_dir)

    assert summary.ok, summary
    assert summary.total == 6
    means = sorted(arm["mean"] for arm in summary.report["arms"])
    assert means == pytest.approx([2.0, 20.0])
    assert summary.report["ranking"][0] == next(
        arm["arm_id"]
        for arm in summary.report["arms"]
        if arm["params"] == {"scale": 10.0}
    )


def test_runs_really_execute_in_parallel(tmp_path: Path) -> None:
    """Four 1s sleeps at concurrency 4 must not take four seconds."""
    argv = ["{python}", "-c", WRITE_METRICS, "{seed}", "{output_dir}", "1.0", "1"]
    manifest = _command_manifest(
        tmp_path,
        seeds=[1, 2, 3, 4],
        matrix={},
        workload={"kind": "command", "argv": argv},
        concurrency=4,
    )

    started = time.monotonic()
    summary = _run(manifest, tmp_path / "run")
    elapsed = time.monotonic() - started

    assert summary.ok, summary
    assert elapsed < 3.5, f"took {elapsed:.1f}s: the runs did not overlap"


def test_one_bad_seed_does_not_touch_the_others(tmp_path: Path) -> None:
    script = (
        "import json, pathlib, sys; "
        "sys.exit(9) if sys.argv[1] == '2' else "
        "pathlib.Path(sys.argv[2], 'metrics.json').write_text("
        "json.dumps({'reward': 1.0}))"
    )
    manifest = _command_manifest(
        tmp_path,
        matrix={},
        workload={
            "kind": "command",
            "argv": ["{python}", "-c", script, "{seed}", "{output_dir}"],
        },
    )

    summary = _run(manifest, tmp_path / "run")

    assert summary.succeeded == 2
    assert summary.failed == 1
    assert summary.report["arms"][0]["failed_seeds"] == [2]
    assert summary.report["arms"][0]["complete"] is False


def test_artifacts_and_logs_come_back(tmp_path: Path) -> None:
    script = (
        "import pathlib, sys; "
        "print('training log line', flush=True); "
        "p = pathlib.Path(sys.argv[1], 'ckpt'); p.mkdir(); "
        "(p / 'model.bin').write_bytes(b'weights')"
    )
    manifest = _command_manifest(
        tmp_path,
        seeds=[5],
        matrix={},
        collect=["ckpt/*"],
        workload={
            "kind": "command",
            "argv": ["{python}", "-c", script, "{output_dir}", "{seed}"],
        },
    )
    run_dir = tmp_path / "run"

    assert _run(manifest, run_dir).ok

    attempt_dir = run_dir / "runs" / "base-seed5" / "attempt-1"
    assert (attempt_dir / "artifacts" / "ckpt" / "model.bin").read_bytes() == b"weights"
    assert "training log line" in (attempt_dir / "stdout.log").read_text()


def test_scratch_is_cleaned_up(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    assert _run(
        _command_manifest(
            tmp_path,
            seeds=[1],
            matrix={},
            workload={
                "kind": "command",
                "argv": ["{python}", "-c", "pass", "{seed}", "{output_dir}"],
            },
        ),
        run_dir,
    ).ok

    assert list((run_dir / "scratch").iterdir()) == []


def test_real_rl_updates_fan_out_across_seeds(tmp_path: Path) -> None:
    """The offline workflow task: real policy gradients, each seed in its own process."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    manifest = FanoutManifest.from_dict(
        {
            "schema_version": 1,
            "name": "workflow-regression",
            "seeds": [42, 43],
            "workload": {"kind": "workflow_regression", "updates": 5, "device": "cpu"},
            "report": {"metric": "improvement", "mode": "max"},
            "concurrency": 2,
            "collect": ["workflow/result.json"],
        },
        base_dir=tmp_path,
    )
    run_dir = tmp_path / "run"

    summary = _run(manifest, run_dir)

    assert summary.ok, [record.error for record in load_records(run_dir)]
    for record in load_records(run_dir):
        assert record.metrics["optimizer_updates"] == 5
        assert record.metrics["reload_verified"] == 1.0
        assert "improvement" in record.metrics
    # Seeds must really differ: identical metrics would mean fake replicates.
    results = [
        json.loads(
            (
                run_dir
                / "runs"
                / r.run_id
                / "attempt-1"
                / "artifacts"
                / "workflow"
                / "result.json"
            ).read_text()
        )
        for r in load_records(run_dir)
    ]
    assert {result["seed"] for result in results} == {42, 43}


DRIVER = """
import sys
from pathlib import Path
from stateset_agents.fanout.local_backend import LocalBackend
from stateset_agents.fanout.manifest import FanoutManifest
from stateset_agents.fanout.orchestrator import FanoutRunner

run_dir = Path(sys.argv[1])
manifest = FanoutManifest.from_file(sys.argv[2])
summary = FanoutRunner(
    manifest,
    LocalBackend(run_dir / "scratch"),
    run_dir,
    poll_interval_s=0.05,
    forwarded_env={},
).run()
print("cancelled", summary.cancelled)
"""


def test_interrupting_the_driver_stops_every_process(tmp_path: Path) -> None:
    """Ctrl-C must not leave compute running: nothing would ever stop it."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    script = (
        "import os, pathlib, sys, time; "
        f"pathlib.Path({str(pid_dir)!r}, sys.argv[1]).write_text(str(os.getpid())); "
        "time.sleep(120)"
    )
    manifest_path = tmp_path / "fanout.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "interrupt",
                "seeds": [1, 2],
                "workload": {
                    "kind": "command",
                    "argv": ["{python}", "-c", script, "{seed}", "{output_dir}"],
                },
                "report": {"metric": "reward", "mode": "max"},
                "concurrency": 2,
            }
        )
    )
    run_dir = tmp_path / "run"
    driver = subprocess.Popen(
        [sys.executable, "-c", DRIVER, str(run_dir), str(manifest_path)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 30
    while len(list(pid_dir.iterdir())) < 2:
        assert time.monotonic() < deadline, "workloads never started"
        assert driver.poll() is None, driver.communicate()
        time.sleep(0.05)
    pids = [int(path.read_text()) for path in pid_dir.iterdir()]

    driver.send_signal(signal.SIGINT)
    stdout, stderr = driver.communicate(timeout=30)

    assert "cancelled 2" in stdout, stderr
    # The workload's parent died too, so init reaps it a moment later.
    deadline = time.monotonic() + 10
    alive = set(pids)
    while alive and time.monotonic() < deadline:
        for pid in list(alive):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                alive.discard(pid)
        time.sleep(0.05)
    assert not alive, f"still running: {alive}"


def test_cancel_after_a_driver_crash_really_stops_local_processes(
    tmp_path: Path,
) -> None:
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    script = (
        "import os, pathlib, sys, time; "
        f"pathlib.Path({str(pid_dir)!r}, sys.argv[1]).write_text(str(os.getpid())); "
        "time.sleep(120)"
    )
    manifest_path = tmp_path / "fanout.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "orphans",
                "seeds": [1],
                "workload": {
                    "kind": "command",
                    "argv": ["{python}", "-c", script, "{seed}", "{output_dir}"],
                },
                "report": {"metric": "reward", "mode": "max"},
            }
        )
    )
    run_dir = tmp_path / "run"
    driver = subprocess.Popen(
        [sys.executable, "-c", DRIVER, str(run_dir), str(manifest_path)],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 30
    while not list(pid_dir.iterdir()):
        assert time.monotonic() < deadline, "workload never started"
        time.sleep(0.05)
    pid = int(next(pid_dir.iterdir()).read_text())
    driver.send_signal(signal.SIGKILL)
    driver.wait(timeout=30)
    os.kill(pid, 0)  # Orphaned, and still running.

    outcome = cancel_run(run_dir, lambda: LocalBackend(run_dir / "scratch"))

    assert outcome.stopped and not outcome.unreleased
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail(f"process {pid} is still running")
