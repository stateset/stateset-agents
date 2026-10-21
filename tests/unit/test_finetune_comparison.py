"""Read-only experiment comparisons never promote training reward to quality."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as api
from stateset_agents.finetune_inspection import compare_runs
from stateset_agents.finetune_workspace import (
    FinetuneConfig,
    create_run,
    plan_run,
    read_json,
    write_json,
)
from stateset_agents.tui import Studio


@pytest.fixture
def experiments(tmp_path):
    data = tmp_path / "chat.jsonl"
    data.write_text(
        '{"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]}\n'
    )
    config = FinetuneConfig(dataset=str(data))
    root = tmp_path / "studio"
    left = create_run(root, plan_run(config))
    right = create_run(root, plan_run(replace(config, learning_rate=5e-6)))
    write_json(
        left / "metrics.json", {"step": 10, "metrics": {"reward": 0.6, "loss": 1.5}}
    )
    write_json(
        right / "metrics.json", {"step": 20, "metrics": {"reward": 0.7, "kl": 0.05}}
    )
    return root, left, right, config


def test_comparison_reports_recorded_differences_missing_metrics_and_no_winner(
    experiments,
):
    _, left, right, _ = experiments
    report = compare_runs(left, right)
    assert report["dataset_identity"] == "same"
    assert (
        report["reference"]["plan_sha256"]
        == hashlib.sha256((left / "plan.json").read_bytes()).hexdigest()
    )
    assert report["config_differences"] == [
        {
            "field": "learning_rate",
            "reference": 1e-5,
            "candidate": 5e-6,
            "reference_present": True,
            "candidate_present": True,
        }
    ]
    metrics = {row["metric"]: row for row in report["metrics"]}
    assert metrics["reward"]["delta"] == pytest.approx(0.1)
    assert metrics["loss"]["candidate"] is None and metrics["loss"]["delta"] is None
    assert metrics["kl"]["reference"] is None and metrics["kl"]["delta"] is None
    assert report["reference"]["latest_metrics"]["step"] == 10
    assert report["candidate"]["latest_metrics"]["step"] == 20
    assert report["quality_gain_verified"] is False and "winner" not in report
    assert any("step, time, data and objective" in note for note in report["notes"])
    assert "metrics.json" not in report["reference"]["unreadable_records"]


@pytest.mark.parametrize(
    "digest,identity",
    [("f" * 64, "different"), ("unknown", "unknown"), (None, "unknown")],
)
def test_dataset_identity_is_only_a_valid_recorded_digest(
    experiments, digest, identity
):
    _, left, right, _ = experiments
    plan = read_json(right / "plan.json")
    plan["dataset_sha256"] = digest
    write_json(right / "plan.json", plan)
    report = compare_runs(left, right)
    assert report["dataset_identity"] == identity
    assert "not reverified" in report["notes"][0]


def test_comparison_does_not_open_data_logs_completion_or_launch_workers(
    experiments, monkeypatch
):
    _, left, right, _ = experiments
    for run in (left, right):
        (run / "dataset").unlink()
        (run / "run.log").write_text("PRIVATE CUSTOMER CONTENT")
        write_json(
            run / "completion.json",
            {"quality_gain_verified": True, "secret": "PRIVATE"},
        )
    before = {p: p.read_bytes() for p in left.parent.rglob("*") if p.is_file()}
    original = os.open
    opened = []

    def open_record(path, *args, **kwargs):
        opened.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_record)
    report = compare_runs(left, right)
    assert set(opened) == {"plan.json", "state.json", "metrics.json"}
    assert "PRIVATE" not in json.dumps(report)
    assert report["quality_gain_verified"] is False
    assert before == {p: p.read_bytes() for p in left.parent.rglob("*") if p.is_file()}
    assert not list(left.parent.glob("*/launch.lock"))


@pytest.mark.parametrize(
    "kind", ["missing", "oversize", "broken", "nested", "symlink", "directory", "fifo"]
)
def test_damaged_and_special_records_are_unavailable_without_blocking(
    experiments, kind
):
    _, left, right, _ = experiments
    path = right / "metrics.json"
    path.unlink()
    if kind == "oversize":
        path.write_bytes(b" " * (128 * 1024 + 1))
    elif kind == "broken":
        path.write_text("{")
    elif kind == "nested":
        path.write_text("[" * 2000)
    elif kind == "symlink":
        path.symlink_to(left / "metrics.json")
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO unavailable")
        os.mkfifo(path)
    report = compare_runs(left, right)
    assert report["candidate"]["latest_metrics"]["metrics"] == {}
    assert "metrics.json" in report["candidate"]["unreadable_records"]
    assert all(row["delta"] is None for row in report["metrics"])


def test_unknown_config_fields_and_malformed_scalars_are_not_reflected(experiments):
    _, left, right, _ = experiments
    plan = read_json(right / "plan.json")
    plan["config"]["CUSTOMER-SECRET"] = "PRIVATE"
    plan["config"]["epochs"] = {"PRIVATE": "bad"}
    plan["config"]["model"] = "x" * 5000
    plan["config"]["learning_rate"] = float("inf")
    plan["examples"] = True
    plan["package_implementation_sha256"] = "PRIVATE"
    (right / "plan.json").write_text(json.dumps(plan))
    write_json(right / "state.json", {"status": {"PRIVATE": 1}})
    report = compare_runs(left, right)
    candidate = report["candidate"]
    assert candidate["status"] == "unreadable"
    assert (
        candidate["examples"] is None
        and candidate["package_implementation_sha256"] is None
    )
    assert not {"epochs", "model", "learning_rate"} & candidate["config"].keys()
    assert "PRIVATE" not in json.dumps(report, allow_nan=False)
    missing = {
        r["field"] for r in report["config_differences"] if not r["candidate_present"]
    }
    assert {"epochs", "model", "learning_rate"} <= missing


def test_metrics_are_bounded_finite_and_overflow_deltas_are_null(experiments):
    _, left, right, _ = experiments
    (left / "metrics.json").write_text(
        json.dumps({"step": 1, "metrics": {"overflow": -1e308}})
    )
    metrics = {
        "overflow": 1e308,
        "bool": True,
        "nan": float("nan"),
        "huge": 10**1000,
        "nested": {"bad": 1},
        "x" * 129: 1,
        **{f"metric-{i}": i for i in range(80)},
    }
    (right / "metrics.json").write_text(
        json.dumps({"step": 1, "metrics": metrics, "updated_at": []})
    )
    report = compare_runs(left, right)
    actual = report["candidate"]["latest_metrics"]
    assert (
        len(actual["metrics"]) == 64 and actual["omitted_fields"] == len(metrics) - 64
    )
    assert actual["updated_at"] is None
    assert {r["metric"]: r for r in report["metrics"]}["overflow"]["delta"] is None
    json.dumps(report, allow_nan=False)


def test_recorded_boolean_is_distinct_from_an_equal_numeric_setting(experiments):
    _, left, right, _ = experiments
    plan = read_json(right / "plan.json")
    plan["config"]["gpu_count"] = False
    write_json(right / "plan.json", plan)
    changes = {
        row["field"]: row for row in compare_runs(left, right)["config_differences"]
    }
    assert changes["gpu_count"]["reference"] == 0
    assert changes["gpu_count"]["candidate"] is False


def test_root_cli_comparison_works_without_training_imports(experiments):
    root, left, right, _ = experiments
    program = """
import json,sys
from typer.testing import CliRunner
from stateset_agents.cli import app
result = CliRunner().invoke(app, ['finetune','compare',sys.argv[1],sys.argv[2],'--workspace',sys.argv[3]])
assert result.exit_code == 0, result.stdout
assert not json.loads(result.stdout)['result']['quality_gain_verified']
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
print('comparison works without training imports')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, left.name, right.name, str(root)],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("step", [True, -1, 1.5, 10**1000])
def test_invalid_metric_steps_are_not_treated_as_snapshots(experiments, step):
    _, left, right, _ = experiments
    (right / "metrics.json").write_text(
        json.dumps({"step": step, "metrics": {"reward": 1}})
    )
    report = compare_runs(left, right)
    assert report["candidate"]["latest_metrics"]["step"] is None
    assert report["candidate"]["latest_metrics"]["metrics"] == {}


def test_cli_comparison_redacts_credentials_and_discovery_is_read_only(
    experiments, monkeypatch
):
    root, left, right, _ = experiments
    plan = read_json(right / "plan.json")
    plan["config"]["model"] = "synthetic-compare-secret"
    write_json(right / "plan.json", plan)
    monkeypatch.setenv("RUNPOD_API_KEY", "synthetic-compare-secret")
    result = CliRunner().invoke(
        api.app, ["compare", left.name, right.name, "--workspace", str(root)]
    )
    assert result.exit_code == 0, result.stdout
    response = json.loads(result.stdout)
    assert response["command"] == "compare" and response["schema_version"] == 1
    assert response["result"]["candidate"]["config"]["model"] == "[REDACTED]"
    assert "synthetic-compare-secret" not in result.stdout
    schema = json.loads(CliRunner().invoke(api.app, ["schema"]).stdout)
    assert schema["result"]["effects"]["compare"] == "read_only"


@pytest.mark.parametrize(
    "other,code",
    [
        ("self", "invalid_comparison"),
        ("../outside", "invalid_run_id"),
        ("missing", "run_not_found"),
        ("linked", "run_not_found"),
    ],
)
def test_cli_comparison_rejects_self_unknown_traversal_and_linked_runs(
    experiments, other, code
):
    root, left, right, _ = experiments
    if other == "linked":
        (root / other).symlink_to(right, target_is_directory=True)
    result = CliRunner().invoke(
        api.app,
        [
            "compare",
            left.name,
            left.name if other == "self" else other,
            "--workspace",
            str(root),
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["code"] == code


def test_reference_pin_survives_filters_and_new_runs_and_comparison_is_frozen(
    experiments,
):
    root, left, right, config = experiments
    studio = Studio(root)
    studio.tab = "runs"
    studio.run_index = next(
        i for i, row in enumerate(studio.runs) if row["id"] == left.name
    )
    draft, preview = dict(studio.values), studio.plan
    studio.handle("a")
    assert studio.comparison_reference == left
    studio.handle("x")
    assert studio.overlay is None and "different run" in studio.message
    studio.run_query = right.name
    studio.refresh()
    assert len(studio.runs) == 1 and studio.comparison_reference == left
    studio.handle("x")
    report = studio.comparison
    assert studio.overlay == "compare" and report["reference"]["run_id"] == left.name
    create_run(root, plan_run(config))
    write_json(right / "metrics.json", {"step": 100, "metrics": {"NEW": 99}})
    studio.refresh()
    assert studio.comparison == report and "NEW" not in json.dumps(studio.comparison)
    studio.handle("PAGEDOWN")
    assert studio.dialog_offset == 8
    studio.handle("ESC")
    assert studio.comparison is None and studio.comparison_reference == left
    studio.handle("x")
    assert studio.comparison["candidate"]["latest_metrics"]["step"] == 100
    assert studio.values == draft and studio.plan is preview
    studio.handle("ESC")
    studio.run_query = ""
    studio.refresh()
    studio.run_index = next(
        i for i, row in enumerate(studio.runs) if row["id"] == left.name
    )
    studio.handle("a")
    assert studio.comparison_reference is None


@pytest.mark.parametrize("width,height", [(72, 22), (74, 44), (110, 32)])
def test_comparison_wraps_scrolls_and_preserves_controls(experiments, width, height):
    root, left, right, _ = experiments
    studio = Studio(root)
    studio.tab = "runs"
    studio.comparison_reference = left
    studio.run_index = next(
        i for i, row in enumerate(studio.runs) if row["id"] == right.name
    )
    studio.handle("x")
    frame = studio.lines(width, height)
    assert len(frame) == height
    assert "RUN COMPARISON" in "\n".join(frame)
    assert "PgUp/Dn scroll" in "\n".join(frame) and "reopen x" in "\n".join(frame)
    for _ in range(100):
        studio.handle("PAGEDOWN")
    text = "\n".join(studio.lines(width, height))
    assert "independent quality" in text and "Esc back" in text
    studio.handle("ENTER")
    assert not list(root.glob("*/launch.lock"))


def test_missing_reference_and_empty_selection_give_actionable_guidance(experiments):
    root, left, right, _ = experiments
    studio = Studio(root)
    studio.tab = "runs"
    studio.handle("x")
    assert "Pin a reference" in studio.message
    studio.comparison_reference = root / "removed"
    studio.handle("x")
    assert studio.overlay is None and "unavailable" in studio.message
    studio.run_query = "no-such-run"
    studio.refresh()
    studio.handle("x")
    assert "candidate" in studio.message
    studio.handle("a")
    assert "Select a retained run" in studio.message
