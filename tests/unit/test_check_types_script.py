"""Tests for the type-check wrapper script."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_check_types_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_types.py"
    spec = importlib.util.spec_from_file_location("check_types_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_types_default_uses_stable_mypy_flags(monkeypatch) -> None:
    module = _load_check_types_module()
    run_calls: list[list[str]] = []

    monkeypatch.setattr(module, "_run", lambda cmd: run_calls.append(cmd) or 0)
    monkeypatch.setattr(module, "Path", Path)
    monkeypatch.setattr("sys.argv", ["check_types.py"])

    assert module.main() == 0
    assert run_calls == [
        [
            "mypy",
            "--config-file",
            "mypy.ini",
            "--no-incremental",
            "--cache-dir=/dev/null",
        ]
    ]


def test_check_types_all_uses_stable_mypy_flags(monkeypatch) -> None:
    module = _load_check_types_module()
    run_calls: list[list[str]] = []

    monkeypatch.setattr(module, "_run", lambda cmd: run_calls.append(cmd) or 0)
    monkeypatch.setattr("sys.argv", ["check_types.py", "--all"])

    assert module.main() == 0
    assert run_calls == [
        [
            "mypy",
            "--config-file",
            "mypy.ini",
            "--no-incremental",
            "--cache-dir=/dev/null",
            "stateset_agents",
        ]
    ]
