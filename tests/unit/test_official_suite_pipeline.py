"""Tests for the shell-free official benchmark suite pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.adapters.official_suite_pipeline import (
    OfficialPipelineError,
    build_parser,
    execute_pipeline,
    load_pipeline_config,
    main,
)


def _evaluation_config(**overrides: Any) -> dict[str, Any]:
    pipeline: dict[str, Any] = {
        "command_timeout_seconds": 30,
        "results_path": "official/results.json",
        "commands": [
            [
                "suite-runner",
                "--model",
                "{model}",
                "--seed",
                "{seed}",
                "--results",
                "{official_results}",
            ]
        ],
    }
    pipeline.update(overrides)
    return {
        "temperature": 0.0,
        "official_suite_pipelines": {"tau3-bench": pipeline},
    }


def _args(tmp_path: Path, config: dict[str, Any], *, suite: str = "tau3-bench") -> Any:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    repository = tmp_path / "upstream"
    repository.mkdir()
    return build_parser().parse_args(
        [
            "--policy",
            "baseline",
            "--model",
            "example/model",
            "--model-revision",
            "b" * 40,
            "--seed",
            "42",
            "--suite",
            suite,
            "--suite-revision",
            "a" * 40,
            "--split",
            "test",
            "--evaluation-config-json",
            json.dumps(config),
            "--output",
            str(artifact_dir / "tasks.jsonl"),
            "--artifact-dir",
            str(artifact_dir),
            "--upstream-repository",
            str(repository),
        ]
    )


def test_pipeline_paths_are_confined_and_distinct(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    config = _evaluation_config(results_path="../escaped.json")
    with pytest.raises(OfficialPipelineError, match="escapes artifact_dir"):
        load_pipeline_config(config, "tau3-bench", artifact_dir)

    config = _evaluation_config(
        cost_records_path="official/results.json",
        artifact_paths={"extra": "official/results.json"},
    )
    with pytest.raises(OfficialPipelineError, match="must be distinct"):
        load_pipeline_config(config, "tau3-bench", artifact_dir)


def test_pipeline_requires_shell_free_model_binding(tmp_path: Path) -> None:
    config = _evaluation_config(commands=[["suite-runner", "--model={model}"]])
    with pytest.raises(OfficialPipelineError, match="missing placeholders"):
        load_pipeline_config(config, "tau3-bench", tmp_path)


def test_pipeline_executes_and_normalizes_fresh_tau3_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, _evaluation_config())
    clean_checks = 0

    def fake_clean(_: Path) -> bool:
        nonlocal clean_checks
        clean_checks += 1
        return True

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        assert kwargs["cwd"] == Path(args.upstream_repository).resolve()
        assert "shell" not in kwargs or kwargs["shell"] is False
        results = Path(command[command.index("--results") + 1])
        results.write_text(
            json.dumps(
                {
                    "simulations": [
                        {
                            "task_id": "retail-1",
                            "reward_info": {"reward": 1.0},
                            "agent_cost": 0.125,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="official stdout", stderr="")

    monkeypatch.setattr(
        "benchmarks.adapters.official_suite_pipeline._repository_is_clean",
        fake_clean,
    )
    monkeypatch.setattr(
        "benchmarks.adapters.official_suite_pipeline.subprocess.run", fake_run
    )
    records = execute_pipeline(args)

    assert clean_checks == 2
    assert records == [{"task_id": "retail-1", "success": True, "cost_usd": 0.125}]
    artifact_dir = Path(args.artifact_dir)
    assert json.loads(Path(args.output).read_text(encoding="utf-8"))["success"]
    manifest = json.loads(
        (artifact_dir / "execution-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model_revision"] == "b" * 40
    assert manifest["suite_revision"] == "a" * 40
    assert (artifact_dir / "command-01.stdout.log").read_text() == "official stdout"


def test_pipeline_executes_and_normalizes_bfcl_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {
        "official_suite_pipelines": {
            "bfcl-v4": {
                "results_path": "official/results",
                "scores_path": "official/scores",
                "cost_records_path": "official/costs.jsonl",
                "commands": [
                    ["bfcl-generate", "{model}", "{official_results}"],
                    [
                        "bfcl-evaluate",
                        "{model}",
                        "{official_scores}",
                        "{cost_records}",
                    ],
                ],
            }
        }
    }
    args = _args(tmp_path, config, suite="bfcl-v4")

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        if command[0] == "bfcl-generate":
            results = Path(command[2])
            results.mkdir()
            (results / "tasks.json").write_text(
                json.dumps([{"id": "bfcl-1"}, {"id": "bfcl-2"}]),
                encoding="utf-8",
            )
        else:
            scores = Path(command[2])
            scores.mkdir()
            (scores / "scores.json").write_text(
                json.dumps(
                    [
                        {"total_count": 2, "correct_count": 1},
                        {"id": "bfcl-2"},
                    ]
                ),
                encoding="utf-8",
            )
            Path(command[3]).write_text(
                '{"task_id":"bfcl-1","cost_usd":0.01}\n'
                '{"task_id":"bfcl-2","cost_usd":0.02}\n',
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "benchmarks.adapters.official_suite_pipeline._repository_is_clean",
        lambda _: True,
    )
    monkeypatch.setattr(
        "benchmarks.adapters.official_suite_pipeline.subprocess.run", fake_run
    )

    assert execute_pipeline(args) == [
        {"task_id": "bfcl-1", "success": True, "cost_usd": 0.01},
        {"task_id": "bfcl-2", "success": False, "cost_usd": 0.02},
    ]


def test_pipeline_executes_and_normalizes_swe_bench_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {
        "official_suite_pipelines": {
            "swe-bench-verified": {
                "results_path": "official/report",
                "cost_records_path": "official/costs.jsonl",
                "commands": [
                    [
                        "swe-runner",
                        "{model}",
                        "{official_results}",
                        "{cost_records}",
                    ]
                ],
            }
        }
    }
    args = _args(tmp_path, config, suite="swe-bench-verified")

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        report = Path(command[2])
        report.mkdir()
        (report / "results.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "total_instances": 2,
                    "submitted_ids": ["swe-1", "swe-2"],
                    "resolved_ids": ["swe-1"],
                }
            ),
            encoding="utf-8",
        )
        Path(command[3]).write_text(
            '{"task_id":"swe-1","cost_usd":0.11}\n'
            '{"task_id":"swe-2","cost_usd":0.22}\n',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "benchmarks.adapters.official_suite_pipeline._repository_is_clean",
        lambda _: True,
    )
    monkeypatch.setattr(
        "benchmarks.adapters.official_suite_pipeline.subprocess.run", fake_run
    )

    assert execute_pipeline(args) == [
        {"task_id": "swe-1", "success": True, "cost_usd": 0.11},
        {"task_id": "swe-2", "success": False, "cost_usd": 0.22},
    ]


def test_pipeline_rejects_stale_artifacts_before_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, _evaluation_config())
    results = Path(args.artifact_dir, "official/results.json")
    results.parent.mkdir(parents=True)
    results.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "benchmarks.adapters.official_suite_pipeline._repository_is_clean",
        lambda _: True,
    )
    with pytest.raises(OfficialPipelineError, match="already exists"):
        execute_pipeline(args)


def test_pipeline_rejects_output_artifact_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, _evaluation_config(results_path="tasks.jsonl"))
    monkeypatch.setattr(
        "benchmarks.adapters.official_suite_pipeline._repository_is_clean",
        lambda _: True,
    )
    with pytest.raises(OfficialPipelineError, match="paths must be distinct"):
        execute_pipeline(args)


def test_pipeline_rejects_stale_provenance_before_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, _evaluation_config())
    Path(args.artifact_dir, "execution-manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    monkeypatch.setattr(
        "benchmarks.adapters.official_suite_pipeline._repository_is_clean",
        lambda _: True,
    )
    with pytest.raises(OfficialPipelineError, match="already exists"):
        execute_pipeline(args)


def test_pipeline_rejects_upstream_mutation_after_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, _evaluation_config())
    cleanliness = iter([True, False])
    monkeypatch.setattr(
        "benchmarks.adapters.official_suite_pipeline._repository_is_clean",
        lambda _: next(cleanliness),
    )
    monkeypatch.setattr(
        "benchmarks.adapters.official_suite_pipeline.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    with pytest.raises(OfficialPipelineError, match="modified the upstream"):
        execute_pipeline(args)


def test_cli_fails_closed_on_missing_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, {"temperature": 0.0})
    monkeypatch.setattr(
        "benchmarks.adapters.official_suite_pipeline._repository_is_clean",
        lambda _: True,
    )
    argv = []
    for key, value in vars(args).items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    assert main(argv) == 2
