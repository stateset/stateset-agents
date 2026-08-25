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


def _is_warnings_warn_call(node: ast.stmt) -> bool:
    """True for a bare ``warnings.warn(...)`` expression statement."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "warn"
        and isinstance(func.value, ast.Name)
        and func.value.id == "warnings"
    )


def _non_shim_statements(path: Path) -> list[str]:
    """Return descriptions of statements a deprecation shim may not contain.

    A shim may only hold: a module docstring, imports, plain assignments
    (re-exports, ``__all__``), and a ``warnings.warn(...)`` call.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad: list[str] = []
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.Import | ast.ImportFrom | ast.Assign | ast.AnnAssign):
            continue
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue  # module docstring
        if _is_warnings_warn_call(node):
            continue
        bad.append(f"{path.name}:{node.lineno} {type(node).__name__}")
    return bad


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
        bad = _non_shim_statements(path)
        assert bad == [], (
            f"{name} is exempted from the layering rule because it is a pure "
            "deprecation shim, but it contains statements a shim may not have "
            "(only a docstring, imports, assignments and warnings.warn are "
            "allowed): " + ", ".join(bad)
        )


@pytest.mark.parametrize("name", DEPRECATED_ROOT_PLANNING_NAMES)
def test_root_planning_exports_warn(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import stateset_agents

    # Drop any cached value so __getattr__ runs again; monkeypatch restores it.
    monkeypatch.delitem(vars(stateset_agents), name, raising=False)
    with pytest.warns(DeprecationWarning, match="experimental.long_term_planning"):
        assert getattr(stateset_agents, name) is not None


def test_experimental_import_path_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from stateset_agents.experimental.long_term_planning import (  # noqa: F401
            PlanningConfig,
        )


# --- Task 5.3/5.4: core must not depend on training; one TrainingConfig -----

PACKAGE_DIR = CORE_DIR.parent


def _training_module_level_imports(path: Path) -> list[str]:
    """Return module-level imports of the training layer in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("stateset_agents.training"):
                    offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            absolute = node.level == 0 and module.startswith("stateset_agents.training")
            relative = node.level > 0 and (
                module == "training" or module.startswith("training.")
            )
            if absolute or relative:
                dots = "." * node.level
                offenders.append(f"{path.name}:{node.lineno} from {dots}{module}")
    return offenders


def _training_config_class_definitions() -> list[str]:
    """Every ``class TrainingConfig`` defined under ``stateset_agents/``."""
    found: list[str] = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TrainingConfig":
                found.append(
                    f"{path.relative_to(PACKAGE_DIR).as_posix()}:{node.lineno}"
                )
    return found


def test_core_does_not_import_training_at_module_level() -> None:
    offenders: list[str] = []
    for path in _core_modules():
        offenders.extend(_training_module_level_imports(path))
    assert offenders == [], (
        "stateset_agents.core must not import stateset_agents.training at "
        "module level (move the import into the function that uses it): "
        + ", ".join(offenders)
    )


def test_exactly_one_training_config_class() -> None:
    found = _training_config_class_definitions()
    paths = [entry.split(":")[0] for entry in found]
    assert paths == ["training/config.py"], (
        "exactly one class named TrainingConfig may be defined under "
        "stateset_agents/ and it must live in training/config.py; found: "
        + ", ".join(found)
    )


def test_enhanced_gspo_config_is_the_canonical_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from stateset_agents.core.enhanced import advanced_rl_algorithms
    from stateset_agents.training.gspo_config import GSPOConfig

    monkeypatch.delitem(vars(advanced_rl_algorithms), "GSPOConfig", raising=False)
    with pytest.warns(DeprecationWarning, match="training.gspo_config"):
        assert advanced_rl_algorithms.GSPOConfig is GSPOConfig
