"""Tests for the shell-free paired upstream agent-harness adapter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.adapters.paired_agent_harness import (
    PairedHarnessError,
    build_parser,
    canonical_digest,
    load_suite_config,
    load_task_records,
    main,
    run,
    verify_repository,
)


def _command() -> list[str]:
    return [
        "python",
        "driver.py",
        "--policy",
        "{policy}",
        "--model",
        "{model}",
        "--model-revision",
        "{model_revision}",
        "--seed",
        "{seed}",
        "--suite",
        "{suite}",
        "--suite-revision",
        "{suite_revision}",
        "--split",
        "{split}",
        "--evaluation-config-json",
        "{evaluation_config_json}",
        "--output",
        "{output}",
        "--artifact-dir",
        "{artifact_dir}",
    ]


def _write_config(path: Path, repository: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "stateset-paired-agent-harness-config",
                "suites": {
                    "tau-bench": {
                        "repository_path": str(repository),
                        "timeout_seconds": 30,
                        "cost_source": "provider-api",
                        "command": _command(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _args(config: Path, output: Path) -> Any:
    return build_parser().parse_args(
        [
            "run",
            "--harness-config",
            str(config),
            "--seed",
            "42",
            "--suite",
            "tau-bench",
            "--suite-revision",
            "a" * 40,
            "--split",
            "test",
            "--framework-version",
            "0.47.0",
            "--baseline-model",
            "example/base",
            "--baseline-revision",
            "b" * 40,
            "--trained-model",
            "example/trained",
            "--trained-revision",
            "c" * 40,
            "--evaluation-config-json",
            '{"temperature":0.0}',
            "--adapter-output",
            str(output / "adapter.json"),
            "--artifact-dir",
            str(output / "artifact"),
        ]
    )


def test_config_requires_complete_shell_free_command(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _write_config(config, tmp_path)
    assert load_suite_config(config, "tau-bench")["timeout_seconds"] == 30

    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["suites"]["tau-bench"]["command"].remove("{model_revision}")
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PairedHarnessError, match="missing placeholders"):
        load_suite_config(config, "tau-bench")


def test_validate_command_checks_complete_suite_roster(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _write_config(config, tmp_path)
    assert main(["validate", "--harness-config", str(config)]) == 2


def test_task_records_reject_duplicates_and_invalid_cost(tmp_path: Path) -> None:
    records = tmp_path / "tasks.jsonl"
    records.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "one", "success": True, "cost_usd": 0.1}),
                json.dumps({"task_id": "one", "success": False, "cost_usd": 0.2}),
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(PairedHarnessError, match="duplicate task_id"):
        load_task_records(records, policy="baseline")

    records.write_text(
        json.dumps({"task_id": "one", "success": True, "cost_usd": -1}),
        encoding="utf-8",
    )
    with pytest.raises(PairedHarnessError, match="finite and non-negative"):
        load_task_records(records, policy="baseline")


def test_repository_must_match_revision_and_be_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout="b" * 40 + "\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="a" * 40 + "\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="?? generated.txt\n", stderr=""),
        ]
    )
    monkeypatch.setattr(
        "benchmarks.adapters.paired_agent_harness.subprocess.run",
        lambda *_args, **_kwargs: next(responses),
    )
    with pytest.raises(PairedHarnessError, match="revision mismatch"):
        verify_repository(tmp_path, "a" * 40)
    with pytest.raises(PairedHarnessError, match="worktree must be clean"):
        verify_repository(tmp_path, "a" * 40)


def test_run_emits_paired_scores_cost_and_task_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    repository = tmp_path / "upstream"
    repository.mkdir()
    _write_config(config, repository)
    monkeypatch.setattr(
        "benchmarks.adapters.paired_agent_harness.verify_repository",
        lambda *_args: None,
    )
    seen_models: list[str] = []

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        model = command[command.index("--model") + 1]
        seen_models.append(model)
        output = Path(command[command.index("--output") + 1])
        success = model == "example/trained"
        output.write_text(
            "\n".join(
                [
                    json.dumps(
                        {"task_id": "task-1", "success": success, "cost_usd": 0.2}
                    ),
                    json.dumps({"task_id": "task-2", "success": True, "cost_usd": 0.3}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "benchmarks.adapters.paired_agent_harness.subprocess.run", fake_run
    )
    result = run(_args(config, tmp_path / "output"))

    assert seen_models == ["example/base", "example/trained"]
    assert result["tasks"] == 2
    assert result["baseline_score"] == 0.5
    assert result["trained_score"] == 1.0
    assert result["evaluation_cost_usd"] == pytest.approx(1.0)
    assert result["paired_task_ids_sha256"] == canonical_digest(["task-1", "task-2"])
    assert Path(result["artifact_path"], "paired-summary.json").is_file()


def test_run_rejects_task_order_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    repository = tmp_path / "upstream"
    repository.mkdir()
    _write_config(config, repository)
    monkeypatch.setattr(
        "benchmarks.adapters.paired_agent_harness.verify_repository",
        lambda *_args: None,
    )

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        policy = command[command.index("--policy") + 1]
        task_ids = ["one", "two"] if policy == "baseline" else ["two", "one"]
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            "\n".join(
                json.dumps({"task_id": task_id, "success": True, "cost_usd": 0})
                for task_id in task_ids
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "benchmarks.adapters.paired_agent_harness.subprocess.run", fake_run
    )
    with pytest.raises(PairedHarnessError, match="identical ordered task IDs"):
        run(_args(config, tmp_path / "output"))
