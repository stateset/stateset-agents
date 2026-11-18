"""Contracts for the packaged swarm recipes and their returned evidence."""

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app

RUNNER = CliRunner()
EXAMPLES = Path(__file__).resolve().parents[2] / "examples/swarm/evidence/local-cpu"


@pytest.fixture
def worker_repo(tmp_path: Path) -> Path:
    """Use a self-contained checkout so tests also work in shallow CI clones."""
    repo = tmp_path / "worker checkout"
    repo.mkdir()
    for name in (
        "stateset_agents/fanout/worker.py",
        "stateset_agents/experimental/weight_space/checkpoints.py",
        "examples/fanout/weight_edit/fact-1.json",
    ):
        path = repo / name
        path.parent.mkdir(parents=True)
        path.touch()
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.org",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Worker fixture",
        ],
        ["tag", "v0.73.2"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True)
    return repo


@pytest.mark.parametrize(
    ("recipe", "keys"),
    [
        ("starter", ["rl-seed-42", "checkpoint-edit"]),
        ("rl-seeds", ["rl-seed-42", "rl-seed-43", "rl-seed-44"]),
        ("qwen-layers", ["qwen-layer-6", "qwen-layer-14"]),
    ],
)
def test_coordinator_plan_contains_local_scripts(
    worker_repo: Path, recipe: str, keys: list[str]
) -> None:
    result = RUNNER.invoke(app, ["swarm", "plan", recipe, "--repo", str(worker_repo)])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=worker_repo, text=True
    ).strip()
    assert Path(plan["repository"]["path"]).resolve() == worker_repo.resolve()
    assert plan["repository"]["base"] == sha
    assert plan["defaults"]["where"] == "sandbox"
    assert plan["defaults"]["max_budget_usd"] == "1"
    assert [t["id"] for t in plan["tasks"]] == keys
    for task in plan["tasks"]:
        assert not task.get("depends_on")
        script = RUNNER.invoke(app, ["swarm", "script", task["id"]])
        assert script.exit_code == 0
        assert script.output in task["objective"]
        subprocess.run(["bash", "-n"], input=script.output, text=True, check=True)


def test_unknown_recipe_task_and_unavailable_base_fail(worker_repo: Path) -> None:
    for arguments in (
        ["plan", "typo"],
        ["script", "typo"],
        ["plan", "starter", "--base", "missing", "--repo", str(worker_repo)],
    ):
        assert RUNNER.invoke(app, ["swarm", *arguments]).exit_code != 0


def test_worker_failure_does_not_publish_or_clobber_evidence(tmp_path: Path) -> None:
    script = RUNNER.invoke(app, ["swarm", "script", "rl-seed-42"]).output
    # Stop exactly where the worker would start its Python process.
    failing = "python() { return 7; }\n" + script
    result = subprocess.run(["bash", "-c", failing], cwd=tmp_path)
    assert result.returncode == 7
    results = tmp_path / "swarm-results"
    assert list(results.iterdir()) == []
    target = results / "rl-seed-42"
    target.mkdir()
    (target / "result.json").write_text("original")
    result = subprocess.run(["bash", "-c", failing], cwd=tmp_path)
    assert result.returncode != 0
    assert (target / "result.json").read_text() == "original"


def test_starter_and_seed_reports_include_observed_results() -> None:
    starter = RUNNER.invoke(
        app, ["swarm", "report", "starter", "--results", str(EXAMPLES)]
    )
    assert starter.exit_code == 0
    rows = json.loads(starter.output)
    assert rows["checkpoint-edit"]["reload_verified"]
    assert rows["checkpoint-edit"]["after"] == [
        x * 0.5 for x in rows["checkpoint-edit"]["before"]
    ]
    seeds = RUNNER.invoke(
        app, ["swarm", "report", "rl-seeds", "--results", str(EXAMPLES)]
    )
    assert seeds.exit_code == 0
    rows = json.loads(seeds.output)
    assert len(rows) == 3 and all(
        row["optimizer_updates"] == 20 for row in rows.values()
    )
    assert rows["rl-seed-44"]["improvement"] < 0


def test_report_keeps_other_arms_when_one_is_missing_or_invalid(tmp_path: Path) -> None:
    layer = tmp_path / "qwen-layer-6"
    layer.mkdir()
    (layer / "result.json").write_text(
        json.dumps(
            {
                "status": "rejected",
                "export": None,
                "attempts": [{"module": "model.layers.6.mlp.down_proj", "error": None}],
            }
        )
    )
    (layer / "metrics.json").write_text('{"accepted": 0}')
    args = ["swarm", "report", "qwen-layers", "--results", str(tmp_path)]
    missing = RUNNER.invoke(app, args)
    assert missing.exit_code == 1
    rows = json.loads(missing.output)
    assert rows["qwen-layer-6"]["status"] == "rejected"
    assert "error" in rows["qwen-layer-14"]
    (tmp_path / "qwen-layer-14").mkdir()
    (tmp_path / "qwen-layer-14/result.json").write_text("invalid")
    assert RUNNER.invoke(app, args).exit_code == 1


def test_completed_qwen_trials_report_the_recipe_decisions() -> None:
    result = RUNNER.invoke(
        app, ["swarm", "report", "qwen-layers", "--results", str(EXAMPLES)]
    )
    assert result.exit_code == 0
    for row in json.loads(result.output).values():
        assert row["status"] == "rejected" and row["accepted"] == 0
        assert row["export"] is None
