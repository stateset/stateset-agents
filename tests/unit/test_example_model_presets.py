"""Tests for the model preset registry and unified GSPO finetune driver."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT))

from examples.model_presets import PRESETS, ModelPreset, get_preset, list_preset_names  # noqa: E402
from examples import finetune_gspo  # noqa: E402


def test_presets_registry_has_expected_models():
    expected = {
        "kimi-k3",
        "kimi-k2.5",
        "kimi-k2.6",
        "glm5.1",
        "glm5.2",
        "qwen3",
        "qwen3.5-0.8b",
        "qwen3.5-27b",
        "gemma3",
        "gemma4-31b",
        "llama3",
        "mistral",
    }
    assert set(PRESETS.keys()) == expected
    assert set(list_preset_names()) == expected


def test_preset_is_frozen_dataclass():
    preset = get_preset("kimi-k3")
    assert isinstance(preset, ModelPreset)
    with pytest.raises(AttributeError):
        preset.model_id = "changed"  # type: ignore[misc]


def test_get_preset_unknown_name_raises_with_available_list():
    with pytest.raises(KeyError, match="kimi-k3"):
        get_preset("not-a-real-model")


def test_kimi_k2_5_notes_document_discrepancy():
    """The two clone scripts for Kimi-K2.5 disagreed; the newer one wins
    and the discrepancy must be recorded in ``notes`` per the task brief."""
    preset = get_preset("kimi-k2.5")
    assert "finetune_kimi_k25_gspo.py" in preset.notes
    assert "finetune_kimi_k2_5_gspo.py" in preset.notes
    assert "newer" in preset.notes.lower()


@pytest.mark.parametrize("preset_name", list_preset_names())
def test_dry_run_in_process_exits_zero(preset_name, capsys):
    """Every preset round-trips through the driver's --dry-run path,
    in-process, using the stub backend (fast path for the full matrix)."""
    exit_code = finetune_gspo.main(["--model", preset_name, "--dry-run"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert preset_name in captured.out


def test_list_models_in_process_prints_all_names(capsys):
    exit_code = finetune_gspo.main(["--list-models"])
    assert exit_code == 0

    captured = capsys.readouterr()
    printed = set(captured.out.splitlines())
    assert printed == set(list_preset_names())


@pytest.mark.parametrize(
    "preset_name",
    ["kimi-k3", "glm5.2", "qwen3.5-27b", "mistral"],
)
def test_dry_run_subprocess_exits_zero(preset_name):
    """Cover a representative subset via a real subprocess invocation, to
    guard against import-path/argv issues that in-process calls can mask."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "examples" / "finetune_gspo.py"),
            "--model",
            preset_name,
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert preset_name in result.stdout


def test_list_models_subprocess_prints_all_names():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / "finetune_gspo.py"), "--list-models"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    printed = set(result.stdout.splitlines())
    assert printed == set(list_preset_names())
