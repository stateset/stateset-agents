"""Shared configuration diagnostics and terminal repair without data or cloud work."""

import json
import subprocess
import sys
from dataclasses import asdict

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as api
from stateset_agents.finetune_workspace import (
    check_config_values,
    read_json,
)
from stateset_agents.tui import Studio


def test_reports_independent_backend_and_numeric_problems_together_without_echoing_input():
    values = {
        "provider": "river",
        "method": "gspo",
        "epochs": 0,
        "learning_rate": "PRIVATE-TEXT",
        "lora_rank": 64,
        "gpu": "H100",
        "gpu_count": 2,
        "PRIVATE-KEY": "PRIVATE-VALUE",
    }
    original = dict(values)
    _, issues = check_config_values(values, parse_text=True)
    fields = {issue.field for issue in issues}
    assert {
        "config",
        "dataset",
        "method",
        "cost_policy",
        "epochs",
        "learning_rate",
        "lora_rank",
        "gpu",
        "gpu_count",
    } <= fields
    assert len(fields) == len(issues)
    assert values == original
    assert "PRIVATE" not in json.dumps([asdict(issue) for issue in issues])


@pytest.mark.parametrize(
    "name,value",
    [
        ("epochs", True),
        ("epochs", 1.5),
        ("epochs", []),
        ("learning_rate", False),
        ("learning_rate", {}),
        ("learning_rate", float("nan")),
        ("memory_gb", float("inf")),
        ("memory_gb", 10**400),
        ("gpu_count", True),
        ("gpu_count", -1),
        ("provider", []),
        ("method", {}),
        ("cost_policy", []),
        ("model", None),
        ("revision", "main"),
        ("generations", 1),
    ],
)
def test_malformed_values_return_controlled_field_issues(name, value):
    values = {"dataset": "not-read.jsonl", "method": "gspo", name: value}
    _, issues = check_config_values(values)
    assert name in {issue.field for issue in issues}
    assert all(issue.field and issue.code and issue.message for issue in issues)


def test_terminal_numeric_text_is_opt_in_and_boolean_values_are_never_numbers():
    values = {
        "dataset": "not-read.jsonl",
        "epochs": " 2 ",
        "learning_rate": "5e-6",
        "gpu_count": "0",
    }
    parsed, issues = check_config_values(values, parse_text=True)
    assert not issues and parsed["epochs"] == 2 and parsed["learning_rate"] == 5e-6
    _, strict = check_config_values(values)
    assert {issue.field for issue in strict} == {"epochs", "learning_rate", "gpu_count"}
    for text in (False, True):
        _, issues = check_config_values(
            {"dataset": "x", "learning_rate": text}, parse_text=True
        )
        assert issues[0].field == "learning_rate" and issues[0].code == "invalid_type"


def test_checklist_keeps_draft_and_prepared_preview_unchanged(tmp_path, monkeypatch):
    studio = Studio(tmp_path / "studio")
    before = dict(studio.values)
    marker = object()
    studio.plan = marker
    monkeypatch.setattr(
        "stateset_agents.tui.plan_run",
        lambda *a: pytest.fail("no dataset read in settings check"),
    )
    studio.handle("v")
    assert studio.overlay == "validation"
    assert [issue.field for issue in studio.validation_issues] == ["dataset"]
    studio.handle("ESC")
    assert studio.values == before and studio.plan is marker
    assert not studio.workspace.exists()


@pytest.mark.parametrize("width,height", [(72, 22), (74, 44), (110, 32)])
def test_checklist_navigates_to_hidden_setting_and_rechecks_after_edit(
    tmp_path, width, height
):
    studio = Studio(tmp_path / "studio")
    studio.values.update(
        provider="modal",
        cost_policy="provider_billed",
        dataset="not-read.jsonl",
        scratch_gb=-1,
        epochs=0,
        learning_rate="bad-number",
    )
    assert "scratch_gb" not in {field.name for field in studio.fields()}
    studio.handle("v")
    seen = set()
    for _ in range(len(studio.validation_issues)):
        seen.add(studio.validation_issues[studio.validation_index].field)
        frame = studio.lines(width, height)
        assert len(frame) == height
        assert "Enter fix field" in "\n".join(frame[-5:])
        assert "Selected:" in "\n".join(frame)
        studio.handle("DOWN")
    assert {"scratch_gb", "epochs", "learning_rate"} == seen
    studio.validation_index = next(
        index
        for index, issue in enumerate(studio.validation_issues)
        if issue.field == "scratch_gb"
    )
    studio.handle("ENTER")
    assert studio.fields()[studio.field].name == "scratch_gb"
    studio.handle("ENTER")
    studio.handle("CLEAR")
    studio.handle("3")
    studio.handle("ENTER")
    assert studio.values["scratch_gb"] == "3"
    studio.handle("v")
    assert "scratch_gb" not in {issue.field for issue in studio.validation_issues}
    assert "scratch_gb" not in {field.name for field in studio.fields()}
    assert not studio.workspace.exists()


def test_recheck_updates_stale_report_and_pass_does_not_skip_data_validation(
    tmp_path, settle_preview
):
    studio = Studio(tmp_path / "studio")
    studio.handle("v")
    studio.values["dataset"] = "file-does-not-exist.jsonl"
    assert "Settings changed" in "\n".join(studio.lines(74, 44))
    studio.handle("v")
    assert not studio.validation_issues
    text = "\n".join(studio.lines(74, 44))
    assert "Configuration settings passed" in text and "No dataset contents" in text
    studio.handle("ENTER")
    assert studio.overlay == "validation" and studio.plan is None
    studio.handle("ESC")
    studio.handle("p")
    settle_preview(studio)
    assert studio.plan is None and "Dataset must be a file" in studio.message
    assert not studio.workspace.exists()


def test_incomplete_numeric_draft_still_saves_but_validation_does_not_coerce_booleans(
    tmp_path,
):
    studio = Studio(tmp_path / "studio")
    studio.handle("s")
    assert read_json(studio.workspace / "draft.json")["dataset"] == ""
    studio.values["learning_rate"] = True
    with pytest.raises(ValueError, match="learning_rate"):
        studio.config()


@pytest.mark.parametrize(
    "record,valid",
    [
        ({"dataset": "not-read.jsonl"}, True),
        ({"provider": "river", "dataset": "x", "method": "gspo", "epochs": 0}, False),
        ({"dataset": "x", "epochs": "2"}, False),
        ({"dataset": "x", "PRIVATE-KEY": "PRIVATE-VALUE"}, False),
    ],
)
def test_json_validation_returns_all_issues_without_snapshot_or_model_loading(
    tmp_path, monkeypatch, record, valid
):
    source = tmp_path / "config.json"
    source.write_text(json.dumps(record))
    monkeypatch.setattr(
        api, "plan_run", lambda *a: pytest.fail("validation must not prepare")
    )
    result = CliRunner().invoke(api.app, ["validate", "--config", str(source)])
    assert result.exit_code == 0, result.stdout
    response = json.loads(result.stdout)
    assert response["ok"] and response["command"] == "validate"
    report = response["result"]
    assert report["valid"] is valid
    assert (
        not report["dataset_contents_checked"] and not report["provider_access_checked"]
    )
    assert not report["training_started"] and "PRIVATE" not in result.stdout
    assert len(list(tmp_path.iterdir())) == 1
    _, expected = check_config_values(record)
    assert report["issues"] == [asdict(issue) for issue in expected]


def test_json_validation_handles_overflow_and_unreadable_config_without_echoes(
    tmp_path,
):
    source = tmp_path / "config.json"
    source.write_text('{"dataset":"x","learning_rate":1e999}')
    runner = CliRunner()
    result = runner.invoke(api.app, ["validate", "--config", str(source)])
    # The shared record parser rejects nonfinite numbers before field validation.
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["code"] == "invalid_record"
    source.write_text("PRIVATE INVALID JSON")
    result = runner.invoke(api.app, ["validate", "--config", str(source)])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["code"] == "invalid_record"
    assert "PRIVATE" not in result.stdout


def test_root_json_validation_does_not_import_training_dependencies(tmp_path):
    source = tmp_path / "config.json"
    source.write_text('{"dataset":"not-read.jsonl"}')
    program = "from typer.testing import CliRunner; from stateset_agents.cli import app; import sys,json; r=CliRunner().invoke(app,['finetune','validate','--config',sys.argv[1]]); assert r.exit_code==0,r.stdout; assert json.loads(r.stdout)['result']['valid']; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", program, str(source)],
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
