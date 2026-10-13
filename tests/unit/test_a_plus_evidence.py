"""The A+ evidence layout maps every gate flag to one place under a root."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

import a_plus_evidence  # noqa: E402
import a_plus_gate  # noqa: E402


def test_layout_covers_every_gate_flag() -> None:
    args = a_plus_gate.parse_args.__code__.co_consts
    gate_flags = {c for c in args if isinstance(c, str) and c.startswith("--")}
    gate_flags -= {"--expected-commit", "--output"}
    assert {item.flag for item in a_plus_evidence.LAYOUT} == gate_flags


def test_status_reports_present_and_missing(tmp_path: Path) -> None:
    (tmp_path / "reliability" / "evidence").mkdir(parents=True)
    (tmp_path / "reliability" / "evidence" / "run.json").write_text("{}")

    rows = {row["flag"]: row for row in a_plus_evidence.status(tmp_path)}

    assert rows["--reliability-evidence"]["present"] is True
    assert rows["--flagship-evidence"]["present"] is False
    assert rows["--scaling-billing-evidence"]["required"] is False


def test_gate_argv_refuses_missing_required_evidence(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--framework-evidence"):
        a_plus_evidence.gate_argv(tmp_path, "a" * 40, tmp_path / "r.json")


def test_gate_argv_includes_optional_only_when_present(tmp_path: Path) -> None:
    for item in a_plus_evidence.LAYOUT:
        if not item.required:
            continue
        path = tmp_path / item.relative_path
        if path.suffix == ".json":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        else:
            path.mkdir(parents=True, exist_ok=True)
            (path / "x.json").write_text("{}")

    argv = a_plus_evidence.gate_argv(tmp_path, "a" * 40, tmp_path / "r.json")

    assert "--scaling-billing-evidence" not in argv
    assert argv[:2] == ["--expected-commit", "a" * 40]
    assert argv.count("--release-readiness") == 1
