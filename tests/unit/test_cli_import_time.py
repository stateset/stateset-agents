"""Guard: importing the CLI must not pull in heavy optional ML dependencies.

``stateset_agents.cli`` is the entry point for every ``stateset-agents``
invocation (including ``--help``). Eagerly importing ``torch`` /
``transformers`` / ``sentence_transformers`` at import time adds seconds to
every command, so those imports must stay lazy.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

HEAVY_MODULES = ("sentence_transformers", "torch", "transformers")


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@pytest.mark.parametrize("heavy", HEAVY_MODULES)
def test_cli_import_does_not_load_heavy_module(heavy: str) -> None:
    if not _installed(heavy):
        pytest.skip(f"{heavy} is not installed; nothing to guard against")

    result = subprocess.run(
        [sys.executable, "-c", "import stateset_agents.cli"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import stateset_agents.cli; "
            "print('\\n'.join(sorted(sys.modules)))",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert loaded.returncode == 0, loaded.stderr
    modules = set(loaded.stdout.split())
    assert heavy not in modules, (
        f"importing stateset_agents.cli eagerly imported {heavy!r}; "
        "keep heavy optional dependencies behind a lazy import"
    )
