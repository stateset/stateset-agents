"""
Layering meta-tests: the stable ``stateset_agents.core`` layer must not depend
on the unstable ``stateset_agents.experimental`` layer at import time.

Structured so later layering rules can be appended: helpers first, then the
individual rule tests.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest

CORE_DIR = Path(__file__).resolve().parents[2] / "stateset_agents" / "core"

# These ``core/<name>.py`` modules ARE the old public import paths; they exist
# only to re-export ``stateset_agents.experimental.<name>`` with a
# DeprecationWarning, so their module-level experimental import is by design.
# The size assertion below keeps the exemption from ever hiding real code.
DEPRECATION_SHIMS = frozenset(
    {
        "adaptive_learning_controller.py",
        "few_shot_adaptation.py",
        "intelligent_orchestrator.py",
        "intelligent_orchestrator_logic.py",
        "intelligent_orchestrator_models.py",
        "long_term_planning.py",
        "multi_agent_coordination.py",
        "multimodal_processing.py",
        "neural_architecture_search.py",
    }
)

MAX_SHIM_LINES = 30

DEPRECATED_ROOT_PLANNING_NAMES = (
    "PlanningConfig",
    "PlanningManager",
    "Plan",
    "PlanStep",
    "PlanStatus",
)


def _core_modules() -> list[Path]:
    return sorted(p for p in CORE_DIR.rglob("*.py"))


def _experimental_module_level_imports(path: Path) -> list[str]:
    """Return module-level imports of the experimental layer in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("stateset_agents.experimental"):
                    offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            absolute = node.level == 0 and module.startswith(
                "stateset_agents.experimental"
            )
            relative = node.level > 0 and (
                module == "experimental" or module.startswith("experimental.")
            )
            if absolute or relative:
                dots = "." * node.level
                offenders.append(f"{path.name}:{node.lineno} from {dots}{module}")
    return offenders


def test_core_does_not_import_experimental_at_module_level() -> None:
    offenders: list[str] = []
    for path in _core_modules():
        if path.name in DEPRECATION_SHIMS:
            continue
        offenders.extend(_experimental_module_level_imports(path))
    assert offenders == [], (
        "stateset_agents.core must not import stateset_agents.experimental at "
        "module level (move the import into the function that uses it): "
        + ", ".join(offenders)
    )


def test_deprecation_shims_stay_tiny() -> None:
    for name in sorted(DEPRECATION_SHIMS):
        path = CORE_DIR / name
        assert path.exists(), f"exempted shim {name} no longer exists"
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        assert line_count <= MAX_SHIM_LINES, (
            f"{name} is {line_count} lines; exempted shims must stay "
            f"<= {MAX_SHIM_LINES} lines so the layering exemption cannot hide "
            "real code"
        )


@pytest.mark.parametrize("name", DEPRECATED_ROOT_PLANNING_NAMES)
def test_root_planning_exports_warn(name: str) -> None:
    import stateset_agents

    # Drop any cached value so __getattr__ runs again.
    vars(stateset_agents).pop(name, None)
    with pytest.warns(DeprecationWarning, match="experimental.long_term_planning"):
        assert getattr(stateset_agents, name) is not None
    vars(stateset_agents).pop(name, None)


def test_experimental_import_path_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from stateset_agents.experimental.long_term_planning import (  # noqa: F401
            PlanningConfig,
        )
