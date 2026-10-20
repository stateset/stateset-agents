"""Agent JSON protocol, reviewed inputs and once-only submission without GPUs."""

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as api
from stateset_agents import finetune_workspace as workspace
from stateset_agents.finetune_workspace import FinetuneConfig, read_json, write_json
from stateset_agents.tui import Studio


@pytest.fixture
def setup(tmp_path, monkeypatch):
    dataset = tmp_path / "chat with spaces.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "PRIVATE CUSTOMER QUESTION"},
                    {"role": "assistant", "content": "PRIVATE SUPPORT REPLY"},
                ]
            }
        )
        + "\n"
    )
    settings = FinetuneConfig(dataset=str(dataset))
    config = tmp_path / "config.json"
    write_json(config, asdict(settings))
    root = tmp_path / "workspace with spaces"
    calls = []

    def start(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(workspace.subprocess, "Popen", start)
    return SimpleNamespace(config=config, root=root, settings=settings, calls=calls)


def invoke(*args):
    result = CliRunner().invoke(api.app, list(map(str, args)))
    response = json.loads(result.stdout)
    assert response["schema_version"] == 1
    assert response["ok"] == (result.exit_code == 0)
    return result, response


def prepared(setup):
    _, response = invoke("prepare", "--config", setup.config, "--workspace", setup.root)
    assert response["ok"]
    return response["result"]["run"]


def launch_args(setup, run):
    return [
        "launch",
        run["run_id"],
        "--workspace",
        setup.root,
        "--expected-plan-sha256",
        run["plan_sha256"],
    ]


def test_schema_is_machine_readable_without_workspace_mutation(setup):
    result, response = invoke("schema")
    schema = response["result"]["config_schema"]
    assert schema["required"] == ["dataset"] and not schema["additionalProperties"]
    assert schema["properties"]["steps"]["type"] == "integer"
    assert response["result"]["effects"]["prepare"] == "local_snapshot_only"
    assert response["result"]["checkpoint_resume_supported"]
    recovery = response["result"]["checkpoint_recovery"]
    assert recovery["providers"] == ["local"]
    assert recovery["launch"] == "separate_exact_plan_review"
    assert set(recovery["methods"]) == {"sft", "grpo", "gspo", "dapo"}
    assert not setup.root.exists() and not setup.calls
    assert len(result.stdout.splitlines()) == 1


def test_model_discovery_includes_maturity_without_backend_certification(setup):
    _, response = invoke("models")
    assert response["result"]["models"]
    assert all(
        set(row) == {"model", "maturity"} for row in response["result"]["models"]
    )
    assert not response["result"]["backend_fit_verified"] and not setup.calls


def test_prepare_freezes_data_and_is_visible_in_tui_without_launching(setup):
    result, response = invoke(
        "prepare", "--config", setup.config, "--workspace", setup.root
    )
    run = response["result"]["run"]
    path = Path(run["path"])
    assert run["state"]["status"] == "prepared" and len(run["plan_sha256"]) == 64
    assert not setup.calls and not (path / "launch.lock").exists()
    assert (path / "dataset").read_bytes() == Path(setup.settings.dataset).read_bytes()
    assert "PRIVATE CUSTOMER QUESTION" not in result.stdout
    assert "PRIVATE SUPPORT REPLY" not in result.stdout
    assert path.stat().st_mode & 0o077 == 0
    studio = Studio(setup.root)
    assert studio.runs[0]["id"] == run["run_id"]
    assert "AI AGENT ACCESS / JSON" in "\n".join(studio._run_details())
    assert response["result"]["launch_argv"][-1] == run["plan_sha256"]
    assert response["result"]["launch_argv"][:3] == [
        sys.executable,
        "-m",
        "stateset_agents.cli",
    ]


def test_launch_is_once_only_and_snapshot_bound_even_after_source_file_changes(setup):
    run = prepared(setup)
    Path(setup.settings.dataset).write_text("Changed original after review")
    _, response = invoke(*launch_args(setup, run))
    assert response["result"]["launched"] and len(setup.calls) == 1
    _, replay = invoke(*launch_args(setup, run))
    assert replay["result"]["already_submitted"] and not replay["result"]["launched"]
    assert len(setup.calls) == 1


@pytest.mark.parametrize("change", ["checksum", "plan", "snapshot", "implementation"])
def test_changed_review_cannot_launch(setup, change):
    run = prepared(setup)
    path = Path(run["path"])
    if change == "checksum":
        run["plan_sha256"] = "0" * 64
    elif change == "plan":
        plan = read_json(path / "plan.json")
        plan["config"]["epochs"] += 1
        write_json(path / "plan.json", plan)
    elif change == "snapshot":
        (path / "dataset").write_text("Changed retained snapshot")
    else:
        plan = read_json(path / "plan.json")
        plan["package_implementation_sha256"] = "0" * 64
        write_json(path / "plan.json", plan)
        import hashlib

        run["plan_sha256"] = hashlib.sha256(
            (path / "plan.json").read_bytes()
        ).hexdigest()
    _, response = invoke(*launch_args(setup, run))
    assert response["error"]["code"] in {"plan_mismatch", "review_invalid"}
    assert not setup.calls and not (path / "launch.lock").exists()


def test_billable_launch_requires_explicit_opt_in_and_environment_key(
    setup, monkeypatch
):
    write_json(setup.config, asdict(replace(setup.settings, provider="runpod")))
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    run = prepared(setup)
    _, response = invoke(*launch_args(setup, run))
    assert response["error"]["code"] == "paid_opt_in_required"
    _, response = invoke(*launch_args(setup, run), "--allow-paid")
    assert response["error"]["code"] == "credential_missing" and not setup.calls
    monkeypatch.setenv("RUNPOD_API_KEY", "synthetic-secret-only")
    result, response = invoke(*launch_args(setup, run), "--allow-paid")
    assert response["result"]["launched"] and len(setup.calls) == 1
    assert "synthetic-secret-only" not in result.stdout
    # A read-only retry requires no new paid opt-in and starts no new worker.
    _, replay = invoke(*launch_args(setup, run))
    assert replay["result"]["already_submitted"] and len(setup.calls) == 1


def test_status_and_logs_are_bounded_and_redact_known_credentials(setup, monkeypatch):
    run = prepared(setup)
    path = Path(run["path"])
    monkeypatch.setenv("RUNPOD_API_KEY", "synthetic-secret-only")
    write_json(
        path / "state.json",
        {
            "status": "failed",
            "error": "synthetic-secret-only",
            "credentials": "not-an-output-field",
        },
    )
    write_json(
        path / "provider-job.json",
        {
            "provider": "runpod",
            "job_id": "fixture-handle",
            "api_key": "not-an-output-field",
        },
    )
    (path / "run.log").write_text("old\n\x1b[31msynthetic-secret-only\x1b[0m\nlatest\n")
    result, response = invoke("status", run["run_id"], "--workspace", setup.root)
    assert response["result"]["run"]["terminal"]
    assert response["result"]["run"]["failure"]["code"] == "unknown"
    assert (
        "synthetic-secret-only" not in result.stdout
        and "not-an-output-field" not in result.stdout
    )
    result, response = invoke(
        "logs", run["run_id"], "--workspace", setup.root, "--limit", "2"
    )
    assert response["result"]["lines"] == ["[REDACTED]", "latest"]
    assert "\x1b" not in result.stdout
    _, listing = invoke("status", "--workspace", setup.root, "--limit", "1")
    assert listing["result"]["runs"][0]["run_id"] == run["run_id"]


def test_damaged_optional_records_and_metric_timestamps_do_not_break_status(setup):
    run = prepared(setup)
    path = Path(run["path"])
    (path / "provider-result.json").write_text("invalid json")
    (path / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "metrics": {"loss": 1.0}, "updated_at": float("nan")})
    )
    _, response = invoke("status", run["run_id"], "--workspace", setup.root)
    result = response["result"]["run"]
    assert result["unreadable_records"] == ["provider-result.json"]
    assert result["latest_metrics"][0]["updated_at"] is None
    assert result["quality_gain_verified"] is False
    (path / "state.json").write_text("invalid json")
    _, response = invoke("status", run["run_id"], "--workspace", setup.root)
    assert response["result"]["run"]["terminal"] is None


@pytest.mark.parametrize(
    "record",
    [
        [],
        {"steps": True, "dataset": "ignored"},
        {"RUNPOD_API_KEY": "do-not-echo"},
        {"dataset": "ignored", "provider": "runpod", "method": "gspo"},
    ],
)
def test_invalid_configs_return_json_errors_without_writes(setup, record):
    write_json(setup.config, record)
    result, response = invoke(
        "prepare", "--config", setup.config, "--workspace", setup.root
    )
    assert not response["ok"] and "do-not-echo" not in result.stdout
    assert not setup.root.exists() and not setup.calls


def test_invalid_dataset_does_not_echo_customer_content(setup):
    Path(setup.settings.dataset).write_text("PRIVATE CUSTOMER QUESTION invalid JSON")
    result, response = invoke(
        "prepare", "--config", setup.config, "--workspace", setup.root
    )
    assert response["error"]["code"] == "invalid_dataset_or_plan"
    assert "PRIVATE CUSTOMER QUESTION" not in result.stdout and not setup.root.exists()


@pytest.mark.parametrize(
    "run_id", ["../outside", "../../etc", "a/b", ".", "..", "/tmp"]
)
def test_workspace_run_ids_cannot_traverse(setup, run_id):
    _, response = invoke("status", run_id, "--workspace", setup.root)
    assert response["error"]["code"] == "invalid_run_id"
    assert not setup.root.exists()


def test_symlink_run_cannot_escape_workspace(setup):
    run = prepared(setup)
    (setup.root / "linked").symlink_to(Path(run["path"]), target_is_directory=True)
    _, response = invoke("status", "linked", "--workspace", setup.root)
    assert response["error"]["code"] == "run_not_found"


@pytest.mark.parametrize(
    "command,extra",
    [
        ("status", ["--limit", "0"]),
        ("logs", ["--limit", "201"]),
        ("logs", ["--offset", "-1"]),
    ],
)
def test_invalid_limits_return_machine_errors(setup, command, extra):
    args = [command] + (["unused"] if command == "logs" else [])
    _, response = invoke(*args, "--workspace", setup.root, *extra)
    assert response["error"]["code"] == "invalid_limit"


def test_readiness_does_not_launch_or_create_workspace(setup, monkeypatch):
    monkeypatch.setattr(
        api,
        "inspect_readiness",
        lambda config, root: {"failed": 0, "warnings": 1, "provisions_hardware": False},
    )
    _, response = invoke(
        "readiness", "--config", setup.config, "--workspace", setup.root
    )
    assert not response["result"]["provisions_hardware"]
    assert not setup.root.exists() and not setup.calls


def test_unexpected_library_output_goes_to_stderr(setup, monkeypatch):
    def inspect(*_):
        print("library diagnostic")
        return {"provisions_hardware": False}

    monkeypatch.setattr(api, "inspect_readiness", inspect)
    result, response = invoke("readiness", "--config", setup.config)
    assert response["ok"] and len(result.stdout.splitlines()) == 1
    assert "library diagnostic" in result.stderr


def test_root_cli_is_registered_and_schema_does_not_import_trainers():
    program = "from typer.testing import CliRunner; from stateset_agents.cli import app; import sys,json; r=CliRunner().invoke(app,['finetune','schema']); assert r.exit_code==0,r.stdout; assert json.loads(r.stdout)['schema_version']==1; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; print('agent JSON schema without training imports')"
    result = subprocess.run(
        [sys.executable, "-c", program],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_redaction_preserves_protocol_keys(setup, monkeypatch):
    run = prepared(setup)
    monkeypatch.setenv("RUNPOD_API_KEY", "state")
    _, response = invoke("status", run["run_id"], "--workspace", setup.root)
    assert response["result"]["run"]["state"]["status"] == "prepared"


def test_native_rl_preparation_uses_the_shared_scenario_validator(setup):
    Path(setup.settings.dataset).write_text(
        json.dumps(
            [
                {
                    "id": "fixture",
                    "context": "A parcel arrived damaged.",
                    "policy_facts": ["Replace damaged items free."],
                    "expected_resolution": "Offer a replacement.",
                    "user_responses": ["Please replace it."],
                }
            ]
        )
    )
    write_json(setup.config, asdict(replace(setup.settings, method="gspo")))
    run = prepared(setup)
    assert run["config"]["method"] == "gspo" and not setup.calls
