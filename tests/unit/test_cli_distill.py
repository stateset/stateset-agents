"""``stateset-agents distill``: dry-run plans, errors, and real tiny runs."""

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


def _rows(count: int = 6, *, completions: bool = True) -> list[dict]:
    return [
        {
            "prompt": [{"role": "user", "content": f"question {i}"}],
            **({"completion": "12"} if completions else {}),
        }
        for i in range(count)
    ]


def _invoke(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(app, ["distill", *args])
    return result.exit_code, result.output


def _last_json(output: str) -> dict:
    return json.loads(output[output.rindex('{\n  "model"') :])


def test_dry_run_prints_a_prompt_disjoint_plan(tmp_path: Path) -> None:
    data = _write(tmp_path / "data.jsonl", _rows())
    code, output = _invoke(
        "--train-data",
        str(data),
        "--validation-fraction",
        "0.34",
        "--divergence",
        "jsd",
        "--jsd-beta",
        "0.3",
        "--lmbda",
        "0.5",
        "--temperature",
        "2.0",
        "--dry-run",
    )
    assert code == 0, output
    plan = json.loads(output)
    config = plan["config"]
    assert (config["divergence"], config["jsd_beta"], config["lmbda"]) == (
        "jsd",
        0.3,
        0.5,
    )
    assert config["temperature"] == 2.0
    assert plan["train"]["examples"] + plan["eval"]["examples"] == 6
    assert plan["eval"]["examples"] == 2


def test_dry_run_with_config_file_and_prompt_only_on_policy_data(
    tmp_path: Path,
) -> None:
    data = _write(tmp_path / "prompts.jsonl", _rows(completions=False))
    config = tmp_path / "kd.json"
    config.write_text(
        json.dumps({"on_policy_loss": "sampled", "divergence": "reverse_kl"})
    )
    code, output = _invoke(
        "--train-data", str(data), "--config", str(config), "--lmbda", "1", "--dry-run"
    )
    assert code == 0, output
    plan = json.loads(output)
    assert plan["config"]["on_policy_loss"] == "sampled"
    assert plan["train"]["with_completion"] == 0


@pytest.mark.parametrize(
    ("rows", "args", "message"),
    [
        (_rows(), ["--divergence", "kl"], "divergence"),
        (_rows(completions=False), [], "needs a completion"),
        (_rows(), ["--teacher-top-k", "4", "--lmbda", "0.5"], "live teacher"),
        (_rows(), ["--vocab-alignment", "fuzzy"], "vocab_alignment"),
        ([{"prompt": "Q", "chosen": "a", "rejected": "b"}], [], "preference field"),
        (_rows(), [], "--model is required"),
        (_rows(), ["--model", "student"], "--teacher"),
    ],
)
def test_invalid_requests_exit_2(
    tmp_path: Path, rows: list[dict], args: list[str], message: str
) -> None:
    data = _write(tmp_path / "data.jsonl", rows)
    runs = message in {"--model is required", "--teacher"}
    code, output = _invoke(
        "--train-data", str(data), *args, *([] if runs else ["--dry-run"])
    )
    assert code == 2
    assert message in output


def test_unknown_config_keys_are_rejected(tmp_path: Path) -> None:
    data = _write(tmp_path / "data.jsonl", _rows())
    config = tmp_path / "bad.json"
    config.write_text(json.dumps({"beta": 0.1}))
    code, output = _invoke(
        "--train-data", str(data), "--config", str(config), "--dry-run"
    )
    assert code == 2 and "unknown" in output


def test_dry_run_does_not_import_torch(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    data = _write(tmp_path / "data.jsonl", _rows())
    script = (
        "import sys\n"
        "from typer.testing import CliRunner\n"
        "from stateset_agents.cli import app\n"
        f"r = CliRunner().invoke(app, ['distill', '--train-data', {str(data)!r}, '--dry-run'])\n"
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
def model_dirs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tests.learning._toy import make_model, make_tokenizer

    directories = []
    for name, seed in (("student", 0), ("teacher", 7)):
        directory = tmp_path_factory.mktemp(f"tiny-{name}")
        make_model(seed, n_embd=16).save_pretrained(directory)
        make_tokenizer().save_pretrained(directory)
        directories.append(directory)
    return directories[0], directories[1]


_COMMON = [
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
    "--max-new-tokens",
    "4",
    "--device",
    "cpu",
]


@pytest.mark.parametrize("mode", [["--lmbda", "0.5"], ["--teacher-top-k", "8"]])
def test_distills_local_models_end_to_end(
    tmp_path: Path, model_dirs: tuple[Path, Path], mode: list[str]
) -> None:
    student, teacher = model_dirs
    data = _write(tmp_path / "data.jsonl", _rows(8))
    cache = tmp_path / "cache.pt"
    extra = ["--save-teacher-cache", str(cache)] if "--teacher-top-k" in mode else []
    code, output = _invoke(
        "--train-data",
        str(data),
        "--model",
        str(student),
        "--teacher",
        str(teacher),
        *_COMMON,
        *mode,
        *extra,
        "--output-dir",
        str(tmp_path / "out"),
    )
    assert code == 0, output
    report = _last_json(output)
    assert report["global_step"] == 2 and report["eval_metrics"]
    assert (Path(report["checkpoint"]) / "distillation_checkpoint.json").exists()
    if extra:
        assert cache.exists()
        code, output = _invoke(
            "--train-data",
            str(data),
            "--model",
            str(student),
            "--teacher-cache",
            str(cache),
            "--teacher-top-k",
            "8",
            *_COMMON,
        )
        assert code == 0, output
        assert _last_json(output)["global_step"] == 2
