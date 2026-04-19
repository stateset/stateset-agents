"""Regression tests for Sphinx configuration."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from stateset_agents import __version__


def test_docs_conf_uses_package_version() -> None:
    conf_path = Path(__file__).resolve().parents[2] / "docs" / "conf.py"
    spec = importlib.util.spec_from_file_location("docs_conf", conf_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.version == __version__
    assert module.release == __version__


def test_docs_conf_sets_docs_safe_env_defaults(monkeypatch) -> None:
    conf_path = Path(__file__).resolve().parents[2] / "docs" / "conf.py"
    spec = importlib.util.spec_from_file_location("docs_conf_env", conf_path)
    assert spec is not None
    assert spec.loader is not None

    monkeypatch.delenv("API_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("INFERENCE_BACKEND", raising=False)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert os.environ["API_REQUIRE_AUTH"] == "false"
    assert os.environ["INFERENCE_BACKEND"] == "stub"
