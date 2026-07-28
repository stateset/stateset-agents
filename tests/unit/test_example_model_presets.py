"""Tests for the model preset registry and unified GSPO finetune driver."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT))

from examples.model_presets import PRESETS, ModelPreset, get_preset, list_preset_names  # noqa: E402
from examples import finetune_gspo  # noqa: E402

STARTER_BACKED_PRESETS = [
    name for name, preset in PRESETS.items() if preset.starter_module is not None
]
FORWARDER_SCRIPTS = [
    "finetune_kimi_k3_gspo.py",
    "finetune_kimi_k2_6_gspo.py",
    "finetune_gemma4_31b_gspo.py",
    "finetune_qwen3_5_0_8b_gspo.py",
]


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


def test_starter_module_field_set_for_packaged_starters():
    expected = {
        "kimi-k3": "kimi_k3_starter",
        "kimi-k2.6": "kimi_k2_6_starter",
        "glm5.1": "glm5_1_starter",
        "glm5.2": "glm5_2_starter",
        "gemma4-31b": "gemma4_starter",
        "qwen3.5-0.8b": "qwen3_5_starter",
    }
    for name, module in expected.items():
        assert get_preset(name).starter_module == module

    # Presets without a packaged starter must leave the field unset.
    for name in ("kimi-k2.5", "qwen3", "qwen3.5-27b", "gemma3", "llama3", "mistral"):
        assert get_preset(name).starter_module is None


def test_starter_profile_matches_packaged_starter_config(capsys):
    """--starter-profile memory on the driver must resolve to the same
    values as calling the packaged starter's get_*_config directly."""
    from stateset_agents.training.glm5_2_starter import get_glm5_2_config

    exit_code = finetune_gspo.main(
        ["--model", "glm5.2", "--starter-profile", "memory", "--dry-run"]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    driver_config = payload["config"]

    direct_config = get_glm5_2_config(starter_profile="memory").to_dict()

    for key in ("learning_rate", "num_generations", "max_prompt_length", "use_4bit"):
        assert driver_config[key] == direct_config[key], key


@pytest.mark.parametrize("preset_name", STARTER_BACKED_PRESETS)
def test_write_config_round_trips(preset_name, tmp_path):
    """--write-config for starter-backed presets must produce a file the
    same starter's load_*_config_file can read back."""
    out_path = tmp_path / f"{preset_name.replace('.', '_')}.json"
    exit_code = finetune_gspo.main(
        [
            "--model",
            preset_name,
            "--starter-profile",
            "balanced",
            "--write-config",
            str(out_path),
        ]
    )
    assert exit_code == 0
    assert out_path.exists()

    preset = get_preset(preset_name)
    fns = finetune_gspo._load_starter_functions(preset, preset_name)
    loaded = fns["load_config_file"](out_path)
    assert loaded.model_name == preset.model_id


@pytest.mark.parametrize("script_name", FORWARDER_SCRIPTS)
def test_forwarder_scripts_exit_zero_under_dry_run(script_name):
    """Converted forwarder scripts must still work under --dry-run."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / script_name), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script_name", FORWARDER_SCRIPTS)
def test_forwarder_scripts_are_at_most_15_lines_of_code(script_name):
    """Forwarders must be a thin deprecation print + delegate, per the
    task brief's <=15-line budget (excluding blank lines, comments, and the
    leading module docstring)."""
    import ast

    path = REPO_ROOT / "examples" / script_name
    source = path.read_text()
    tree = ast.parse(source)
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        docstring_end = body[0].end_lineno
    else:
        docstring_end = 0

    lines_after_docstring = source.splitlines()[docstring_end:]
    code_lines = [
        line for line in lines_after_docstring if line.strip() and not line.strip().startswith("#")
    ]
    assert len(code_lines) <= 15, f"{script_name} has {len(code_lines)} code lines"


def test_real_run_invokes_train_with_gspo_for_non_starter_preset(monkeypatch):
    """The non-dry-run path must actually attempt to invoke the real
    training entry point instead of being a no-op (regression test for the
    previously flagged NO-OP real-run mode). Agent construction is stubbed
    out so the test doesn't require real model weights -- per the task
    brief, real-run wiring is tested "up to the point of trainer
    construction being attempted"."""
    calls = []

    class _FakeAgent:
        def __init__(self, config):
            self.config = config

        async def initialize(self):
            return None

    async def _fake_train_with_gspo(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        "stateset_agents.core.agent.MultiTurnAgent", _FakeAgent
    )
    monkeypatch.setattr(
        "stateset_agents.training.gspo_entrypoints.train_with_gspo",
        _fake_train_with_gspo,
    )

    exit_code = finetune_gspo.main(["--model", "llama3", "--no-dry-run"])
    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["config"].model_name == get_preset("llama3").model_id


def test_real_run_invokes_starter_run_config_for_starter_backed_preset(monkeypatch):
    """Same regression coverage for starter-backed presets: the real-run
    path must call the packaged starter's run_*_config, not silently pass."""
    calls = []

    async def _fake_run_kimi_k3_config(config, dry_run=False):
        calls.append((config, dry_run))
        return object()

    monkeypatch.setattr(
        "stateset_agents.training.kimi_k3_starter.run_kimi_k3_config",
        _fake_run_kimi_k3_config,
    )

    exit_code = finetune_gspo.main(["--model", "kimi-k3", "--no-dry-run"])
    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0][1] is False


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
