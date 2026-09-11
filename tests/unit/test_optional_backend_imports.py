"""Regression tests for optional backend import logging."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rust_accelerator_import_does_not_emit_warning(caplog) -> None:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "stateset_agents"
        / "core"
        / "rust_accelerator.py"
    )

    with caplog.at_level(logging.INFO):
        _load_module_from_path("rust_accelerator_test_import", module_path)

    rust_records = [
        record for record in caplog.records if "Rust acceleration" in record.message
    ]
    assert rust_records
    assert all(record.levelno < logging.WARNING for record in rust_records)


def test_vllm_backend_import_does_not_emit_warning(caplog) -> None:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "stateset_agents"
        / "training"
        / "vllm_backend.py"
    )

    with caplog.at_level(logging.INFO):
        _load_module_from_path("vllm_backend_test_import", module_path)

    vllm_records = [record for record in caplog.records if "vLLM" in record.message]
    assert vllm_records
    assert all(record.levelno < logging.WARNING for record in vllm_records)


def test_vllm_backend_falls_back_when_installed_runtime_is_unusable(
    monkeypatch,
) -> None:
    """A broken optional wheel must not make the training package unimportable."""

    class _BrokenVllm(ModuleType):
        def __getattr__(self, name: str):
            raise RuntimeError(f"vLLM runtime cannot initialize: {name}")

    monkeypatch.setitem(sys.modules, "vllm", _BrokenVllm("vllm"))
    module_path = (
        Path(__file__).resolve().parents[2]
        / "stateset_agents"
        / "training"
        / "vllm_backend.py"
    )

    module = _load_module_from_path("vllm_backend_broken_runtime", module_path)

    assert module.VLLM_AVAILABLE is False
    assert module.LLM is None
