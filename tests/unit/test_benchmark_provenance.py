"""Tests for measured benchmark checkout provenance."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_provenance", BENCHMARKS / "benchmark_provenance.py"
)
assert SPEC is not None and SPEC.loader is not None
provenance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = provenance
SPEC.loader.exec_module(provenance)


def _completed(
    command: list[str], code: int, output: str
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(command, code, output, "")


def test_resolves_only_a_clean_exact_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40

    def clean(command: list[str], **_: object) -> subprocess.CompletedProcess:
        if "status" in command:
            return _completed(command, 0, "")
        return _completed(command, 0, commit + "\n")

    monkeypatch.setattr(provenance.subprocess, "run", clean)
    assert provenance.resolve_harness_commit(tmp_path, commit) == commit


def test_rejects_dirty_unresolvable_or_falsely_claimed_checkouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def dirty(command: list[str], **_: object) -> subprocess.CompletedProcess:
        return _completed(command, 0, " M trainer.py\n")

    monkeypatch.setattr(provenance.subprocess, "run", dirty)
    with pytest.raises(provenance.BenchmarkProvenanceError, match="must be clean"):
        provenance.resolve_harness_commit(tmp_path)

    def clean(command: list[str], **_: object) -> subprocess.CompletedProcess:
        if "status" in command:
            return _completed(command, 0, "")
        return _completed(command, 0, "b" * 40 + "\n")

    monkeypatch.setattr(provenance.subprocess, "run", clean)
    with pytest.raises(provenance.BenchmarkProvenanceError, match="does not match"):
        provenance.resolve_harness_commit(tmp_path, "c" * 40)

    def broken(command: list[str], **_: object) -> subprocess.CompletedProcess:
        return _completed(command, 1, "")

    monkeypatch.setattr(provenance.subprocess, "run", broken)
    with pytest.raises(provenance.BenchmarkProvenanceError, match="could not inspect"):
        provenance.resolve_harness_commit(tmp_path)
