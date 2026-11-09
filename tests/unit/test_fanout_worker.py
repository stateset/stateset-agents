"""The in-sandbox worker, exercised as the real module a sandbox would launch."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

WORKER_MODULE = "stateset_agents.fanout.worker"
REPO_ROOT = Path(__file__).resolve().parents[2]

STUB_TRAIN_WORKLOAD: dict[str, Any] = {
    "kind": "train",
    "agent": {
        "model_name": "stub://fanout-test",
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
                "user_responses": ["Hi, I need help.", "Order 12345.", "Thanks!"],
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
}


def _spec(workload: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    spec = {
        "schema_version": 1,
        "name": "demo",
        "run_id": "base-seed7",
        "arm_id": "base",
        "seed": 7,
        "params": {},
        "workload": workload,
    }
    spec.update(overrides)
    return spec


def _command(script: str, *args: str) -> dict[str, Any]:
    return {"kind": "command", "argv": ["{python}", "-c", script, *args]}


def _run_worker(
    tmp_path: Path, spec: dict[str, Any]
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            WORKER_MODULE,
            "--spec",
            str(spec_path),
            "--output-dir",
            str(output),
            "--stage-dir",
            str(tmp_path / "stage"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    return completed, json.loads((output / "result.json").read_text())


WRITE_METRICS = (
    "import json, pathlib, sys; "
    "pathlib.Path(sys.argv[2], 'metrics.json').write_text(json.dumps("
    "{'reward': float(sys.argv[1]) / 10}))"
)


class TestCommandWorkload:
    def test_success_reports_the_metrics_the_command_wrote(
        self, tmp_path: Path
    ) -> None:
        completed, result = _run_worker(
            tmp_path, _spec(_command(WRITE_METRICS, "{seed}", "{output_dir}"))
        )

        assert completed.returncode == 0, completed.stderr
        assert result["status"] == "succeeded"
        assert result["exit_code"] == 0
        assert result["metrics"] == {"reward": 0.7}
        assert result["run_id"] == "base-seed7"
        assert result["seed"] == 7

    def test_matrix_params_reach_the_command(self, tmp_path: Path) -> None:
        script = (
            "import json, pathlib, sys; "
            "pathlib.Path(sys.argv[2], 'metrics.json').write_text("
            "json.dumps({'lr': float(sys.argv[1])}))"
        )
        workload = {
            "kind": "command",
            "argv": ["{python}", "-c", script, "{lr}", "{output_dir}", "{seed}"],
        }

        _, result = _run_worker(tmp_path, _spec(workload, params={"lr": 0.25}))

        assert result["metrics"] == {"lr": 0.25}

    def test_stage_dir_placeholder_points_at_staged_files(self, tmp_path: Path) -> None:
        stage = tmp_path / "stage"
        stage.mkdir()
        (stage / "data.txt").write_text("42")
        script = (
            "import json, pathlib, sys; "
            "value = pathlib.Path(sys.argv[1], 'data.txt').read_text(); "
            "pathlib.Path(sys.argv[2], 'metrics.json').write_text("
            "json.dumps({'value': int(value)}))"
        )
        workload = {
            "kind": "command",
            "argv": ["{python}", "-c", script, "{stage_dir}", "{output_dir}", "{seed}"],
        }

        _, result = _run_worker(tmp_path, _spec(workload))

        assert result["metrics"] == {"value": 42}

    def test_failure_keeps_the_exit_code(self, tmp_path: Path) -> None:
        completed, result = _run_worker(
            tmp_path,
            _spec(_command("import sys; sys.exit(3)", "{seed}", "{output_dir}")),
        )

        assert completed.returncode == 3
        assert result["status"] == "failed"
        assert result["exit_code"] == 3

    def test_failed_run_keeps_metrics_it_paid_for(self, tmp_path: Path) -> None:
        script = (
            "import json, pathlib, sys; "
            "pathlib.Path(sys.argv[1], 'metrics.json').write_text("
            "json.dumps({'reward': 0.5})); sys.exit(4)"
        )

        _, result = _run_worker(
            tmp_path, _spec(_command(script, "{output_dir}", "{seed}"))
        )

        assert result["status"] == "failed"
        assert result["metrics"] == {"reward": 0.5}

    def test_missing_executable_is_a_reported_failure(self, tmp_path: Path) -> None:
        workload = {
            "kind": "command",
            "argv": ["/nonexistent/binary", "{seed}", "{output_dir}"],
        }

        completed, result = _run_worker(tmp_path, _spec(workload))

        assert completed.returncode == 127
        assert result["status"] == "failed"
        assert "binary" in result["error"]

    def test_no_metrics_file_is_not_a_failure(self, tmp_path: Path) -> None:
        _, result = _run_worker(
            tmp_path, _spec(_command("pass", "{seed}", "{output_dir}"))
        )

        assert result["status"] == "succeeded"
        assert result["metrics"] == {}

    def test_non_finite_metrics_are_dropped_and_named(self, tmp_path: Path) -> None:
        script = (
            "import pathlib, sys; "
            "pathlib.Path(sys.argv[1], 'metrics.json').write_text("
            '\'{"bad": NaN, "inf": Infinity, "good": 1.5, "text": "x"}\')'
        )

        _, result = _run_worker(
            tmp_path, _spec(_command(script, "{output_dir}", "{seed}"))
        )

        assert result["metrics"] == {"good": 1.5}
        assert sorted(result["dropped_metrics"]) == ["bad", "inf", "text"]

    def test_metrics_file_that_is_not_an_object_is_named(self, tmp_path: Path) -> None:
        script = (
            "import pathlib, sys; "
            "pathlib.Path(sys.argv[1], 'metrics.json').write_text('[1, 2]')"
        )

        _, result = _run_worker(
            tmp_path, _spec(_command(script, "{output_dir}", "{seed}"))
        )

        assert result["metrics"] == {}
        assert "metrics.json" in result["error"]


class TestMalformedSpec:
    def test_unknown_workload_is_a_reported_failure(self, tmp_path: Path) -> None:
        completed, result = _run_worker(tmp_path, _spec({"kind": "nope"}))

        assert completed.returncode != 0
        assert result["status"] == "failed"
        assert "nope" in result["error"]

    def test_unreadable_spec_still_writes_a_result(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "spec.json"
        spec_path.write_text("{not json")
        output = tmp_path / "output"

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                WORKER_MODULE,
                "--spec",
                str(spec_path),
                "--output-dir",
                str(output),
                "--stage-dir",
                str(tmp_path / "stage"),
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=60,
        )

        assert completed.returncode != 0
        assert json.loads((output / "result.json").read_text())["status"] == "failed"


class TestTrainWorkload:
    def test_stub_training_produces_episode_and_eval_metrics(
        self, tmp_path: Path
    ) -> None:
        completed, result = _run_worker(tmp_path, _spec(STUB_TRAIN_WORKLOAD))

        assert completed.returncode == 0, completed.stderr
        assert result["status"] == "succeeded"
        assert result["metrics"]["episodes"] == 2
        assert result["metrics"]["episode_reward_mean"] > 0
        assert result["metrics"]["eval_reward"] > 0
        assert result["metrics"]["eval_complete"] == 1

    def test_progress_events_reach_stdout(self, tmp_path: Path) -> None:
        completed, _ = _run_worker(tmp_path, _spec(STUB_TRAIN_WORKLOAD))

        events = [
            json.loads(line.removeprefix("STATESET_FANOUT_EVENT="))
            for line in completed.stdout.splitlines()
            if line.startswith("STATESET_FANOUT_EVENT=")
        ]

        assert [event["type"] for event in events][0] == "started"
        assert any(event["type"] == "episode" for event in events)

    def test_a_training_parameter_typo_fails_the_run(self, tmp_path: Path) -> None:
        workload = {
            **STUB_TRAIN_WORKLOAD,
            "training": {**STUB_TRAIN_WORKLOAD["training"], "learning_rat": 1.0},
        }

        completed, result = _run_worker(tmp_path, _spec(workload))

        assert completed.returncode != 0
        assert result["status"] == "failed"
        assert "learning_rat" in result["error"]

    def test_save_checkpoint_writes_inside_the_output_dir(self, tmp_path: Path) -> None:
        workload = {**STUB_TRAIN_WORKLOAD, "save_checkpoint": True}

        completed, result = _run_worker(tmp_path, _spec(workload))

        assert completed.returncode == 0, completed.stderr
        assert result["status"] == "succeeded"
        assert any((tmp_path / "output" / "checkpoint").iterdir())


def test_worker_import_needs_only_the_stdlib() -> None:
    """A bare image must still run ``command`` workloads and report failures."""
    code = (
        "import sys, stateset_agents.fanout.worker; "
        "heavy = {'torch', 'numpy', 'pydantic', 'transformers', 'typer', 'yaml'}; "
        "print(sorted(heavy & set(sys.modules)))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


@pytest.mark.parametrize("name", ["helpfulness", "safety", "conciseness", "engagement"])
def test_named_rewards_construct_with_a_weight(name: str) -> None:
    from stateset_agents.fanout.workloads import build_reward

    assert build_reward({name: 0.5}) is not None
