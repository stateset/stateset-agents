"""Public CLI contracts for repeatable, explicitly configured River RL."""

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app


@pytest.fixture
def cli_case(tmp_path, monkeypatch):
    train, validation = tmp_path / "train.json", tmp_path / "validation.json"
    train.write_text(json.dumps([{"prompt": "train", "expect": ["done"]}]))
    validation.write_text(json.dumps([{"prompt": "validation", "expect": ["done"]}]))
    specs = []

    class Executor:
        supported_job_kinds = {"rl"}

        def supports(self, kind):
            return kind == "rl"

        def submit(self, spec):
            specs.append(spec)
            spec.output_dir.mkdir(parents=True, exist_ok=True)
            report_name = "rl_preview.json" if spec.dry_run else "rl_report.json"
            (spec.output_dir / report_name).write_text(
                json.dumps({"best_round": 1, "best_eval": {"passed": 1, "total": 1}})
            )
            return spec

        def wait(self, spec):
            return SimpleNamespace(succeeded=True, is_preview=spec.dry_run, logs=[])

    monkeypatch.setattr(
        "stateset_agents.cli_flywheel.get_executor", lambda name: Executor()
    )
    args = [
        "flywheel",
        "--provider",
        "river",
        "--algorithm",
        "cispo",
        "--base-model",
        "test/model",
        "--harvest-prompts",
        str(train),
        "--eval-prompts",
        str(validation),
        "--output-root",
        str(tmp_path / "out"),
    ]
    return args, specs, tmp_path / "out"


def test_repeats_forward_all_config_and_split_budget(cli_case):
    args, specs, output = cli_case
    result = CliRunner().invoke(
        app,
        args
        + [
            "--repeats",
            "3",
            "--seed",
            "42",
            "--learning-rate",
            "0.00001",
            "--lora-r",
            "8",
            "--max-generated-tokens",
            "9000",
            "--normalization",
            "sequence",
        ],
    )
    assert result.exit_code == 0, result.output
    assert [s.harvest["seed"] for s in specs] == [42, 43, 44]
    assert all(s.learning_rate == 1e-5 and s.lora_r == 8 for s in specs)
    assert all(s.harvest["max_generated_tokens"] == 3000 for s in specs)
    assert all(
        s.harvest["temperature"] == 1 and s.harvest["normalization"] == "sequence"
        for s in specs
    )
    assert json.loads((output / "rl_repeats_report.json").read_text())["completed"] == 3


def test_preview_repeats_preserve_live_scores(cli_case):
    args, _, output = cli_case
    output.mkdir()
    live = output / "rl_repeats_report.json"
    live.write_bytes(b"existing scores")
    result = CliRunner().invoke(app, args + ["--repeats", "2", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert live.read_bytes() == b"existing scores"
    plan = json.loads((output / "rl_repeats_preview.json").read_text())
    assert len(plan["runs"]) == 2 and plan["completed"] == 0
    assert plan["pass_rate_mean"] is None


def test_default_provider_sft_preview_is_local(tmp_path, monkeypatch):
    from stateset_agents.remote.runpod import RunPodExecutor

    monkeypatch.setattr(
        RunPodExecutor, "submit", lambda *_: pytest.fail("submitted to RunPod")
    )
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([{"prompt": "Help", "expect": ["done"]}]))
    output = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        [
            "flywheel",
            "--dry-run",
            "--base-model",
            "test/model",
            "--harvest-prompts",
            str(prompts),
            "--eval-prompts",
            str(prompts),
            "--output-root",
            str(output),
            "--repeats",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["provider"] == "runpod" and plan["dry_run"]
    assert plan["max_jobs"] == 12
    assert not output.exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--max-cost", "10"],
        ["--generations", "3"],
        ["--repeats", "0"],
        ["--resume", "--repeats", "2"],
        ["--num-epochs", "1"],
        ["--best-of", "1"],
    ],
)
def test_unsupported_or_invalid_options_fail_before_submit(cli_case, extra):
    args, specs, _ = cli_case
    result = CliRunner().invoke(app, args + extra)
    assert result.exit_code != 0
    assert specs == []
