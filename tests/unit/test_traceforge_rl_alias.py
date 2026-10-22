"""Phase 1 rename: ``traceforge_rl`` is a non-breaking alias.

See ``docs/RENAME_TRACEFORGE.md``. The alias must preserve module-object
identity (so ``isinstance`` and class checks agree across both names) and
must stay lazy like the legacy root package.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_traceforge_rl_version_matches_legacy() -> None:
    legacy = importlib.import_module("stateset_agents")
    alias = importlib.import_module("traceforge_rl")

    assert alias.__version__ == legacy.__version__
    assert sorted(alias.__all__) == sorted(legacy.__all__)


def test_traceforge_rl_public_symbol_identity() -> None:
    legacy = importlib.import_module("stateset_agents")
    alias = importlib.import_module("traceforge_rl")

    assert alias.Agent is legacy.Agent
    assert alias.MultiTurnAgent is legacy.MultiTurnAgent


def test_traceforge_rl_submodule_identity() -> None:
    alias_objectives = importlib.import_module("traceforge_rl.training.objectives")
    legacy_objectives = importlib.import_module("stateset_agents.training.objectives")

    assert alias_objectives is legacy_objectives
    assert alias_objectives.PolicyObjective is legacy_objectives.PolicyObjective
    assert sys.modules["traceforge_rl.training.objectives"] is legacy_objectives


def test_traceforge_rl_missing_submodule_raises() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("traceforge_rl.no_such_module_xyz")


def test_traceforge_rl_bare_import_is_lazy() -> None:
    code = (
        "import sys; import traceforge_rl; "
        "assert 'stateset_agents.core.agent' not in sys.modules, 'not lazy'; "
        "assert 'stateset_agents.training.train' not in sys.modules, 'not lazy'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_traceforge_rl_console_script_declared() -> None:
    scripts = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["scripts"]

    assert scripts["traceforge-rl"] == "stateset_agents.cli:run"
    assert scripts["stateset-agents"] == "stateset_agents.cli:run"


def test_python_m_traceforge_rl_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "traceforge_rl", "--help"],
        capture_output=True,
        text=True,
        # The CLI always writes UTF-8; Windows would otherwise decode as cp1252.
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "ingest" in result.stdout
