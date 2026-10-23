"""``stateset-agents preference``: dry-run plans, errors, and a real tiny run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _pairs(count: int = 6) -> list[dict]:
    return [
        {
            "prompt": [{"role": "user", "content": f"question {i}"}],
            "chosen": "12",
            "rejected": "ab",
        }
        for i in range(count)
    ]


def _last_json(output: str) -> dict:
    """The JSON report, which the command prints last (warnings may precede it)."""
    # Progress bars may share the line the report starts on.
    return json.loads(output[output.rindex('{\n  "method"') :])


def _invoke(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(app, ["preference", *args])
    return result.exit_code, result.output


def test_dry_run_prints_a_prompt_disjoint_plan(tmp_path: Path) -> None:
    data = _write(tmp_path / "pairs.jsonl", _pairs())
    code, output = _invoke(
        "--train-data",
        str(data),
        "--validation-fraction",
        "0.34",
        "--beta",
        "0.2",
        "--method",
        "simpo",
        "--dry-run",
    )
    assert code == 0, output
    plan = json.loads(output)
    assert plan["method"] == "simpo" and plan["config"]["beta"] == 0.2
    assert plan["train"]["pairs"] + plan["eval"]["pairs"] == 6
    assert plan["eval"]["pairs"] == 2


def test_dry_run_reward_model_with_config_file(tmp_path: Path) -> None:
    data = _write(tmp_path / "pairs.jsonl", _pairs())
    config = tmp_path / "rm.json"
    config.write_text(json.dumps({"center_rewards_coefficient": 0.01, "num_epochs": 2}))
    code, output = _invoke(
        "--train-data",
        str(data),
        "--method",
        "rm",
        "--config",
        str(config),
        "--dry-run",
    )
    assert code == 0, output
    plan = json.loads(output)
    assert plan["config"]["center_rewards_coefficient"] == 0.01
    assert plan["config"]["num_epochs"] == 2 and "beta" not in plan["config"]


@pytest.mark.parametrize(
    ("rows", "args", "message"),
    [
        (_pairs(), ["--method", "ppo"], "--method must be one of"),
        (_pairs(), ["--method", "rm", "--beta", "0.1"], "does not take"),
        ([{"prompt": "Q", "completion": "a", "label": True}], [], "need --method kto"),
        (_pairs(), ["--beta", "0"], "beta"),
        (_pairs(), [], "--model is required"),
    ],
)
def test_invalid_requests_exit_2(
    tmp_path: Path, rows: list[dict], args: list[str], message: str
) -> None:
    data = _write(tmp_path / "pairs.jsonl", rows)
    dry = [] if message == "--model is required" else ["--dry-run"]
    code, output = _invoke("--train-data", str(data), *args, *dry)
    assert code == 2
    assert message in output


def test_unknown_config_keys_are_rejected(tmp_path: Path) -> None:
    data = _write(tmp_path / "pairs.jsonl", _pairs())
    config = tmp_path / "bad.json"
    config.write_text(json.dumps({"betta": 0.1}))
    code, output = _invoke(
        "--train-data", str(data), "--config", str(config), "--dry-run"
    )
    assert code == 2 and "unknown" in output


def test_dry_run_does_not_import_torch(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    data = _write(tmp_path / "pairs.jsonl", _pairs())
    script = (
        "import sys\n"
        "from typer.testing import CliRunner\n"
        "from stateset_agents.cli import app\n"
        f"r = CliRunner().invoke(app, ['preference', '--train-data', {str(data)!r}, '--dry-run'])\n"
        "assert r.exit_code == 0, r.output\n"
        "sys.exit(1 if 'torch' in sys.modules else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tests.learning._toy import make_model, make_tokenizer

    directory = tmp_path_factory.mktemp("tiny-gpt2")
    make_model(0, n_embd=16).save_pretrained(directory)
    make_tokenizer().save_pretrained(directory)
    return directory


@pytest.mark.parametrize("method", ["dpo", "rm"])
def test_trains_a_local_model_end_to_end(
    tmp_path: Path, tiny_model_dir: Path, method: str
) -> None:
    data = _write(tmp_path / "pairs.jsonl", _pairs(8))
    output_dir = tmp_path / "out"
    code, output = _invoke(
        "--train-data",
        str(data),
        "--model",
        str(tiny_model_dir),
        "--method",
        method,
        "--validation-fraction",
        "0.25",
        "--max-steps",
        "2",
        "--batch-size",
        "2",
        "--learning-rate",
        "0.001",
        "--max-length",
        "48",
        "--max-prompt-length",
        "32",
        "--device",
        "cpu",
        "--output-dir",
        str(output_dir),
    )
    assert code == 0, output
    report = _last_json(output)
    assert report["global_step"] == 2 and report["eval_metrics"]
    checkpoint = Path(report["checkpoint"])
    if method == "rm":
        assert (checkpoint / "reward_model" / "reward_model.json").exists()
    else:
        assert (checkpoint / "preference_checkpoint.json").exists()
