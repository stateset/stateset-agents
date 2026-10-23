"""
Layering meta-tests.

The layer contracts (``LAYER_CONTRACTS``) hold for *every* import a module
can execute -- module level, inside functions and methods, and string
imports through ``importlib.import_module`` -- not only module-level ones:

* ``stateset_agents.core`` imports neither ``training``, ``api`` nor
  ``experimental``;
* ``stateset_agents.training`` (home of the stable ``train()`` and trainers)
  does not import ``experimental``.

``if TYPE_CHECKING:`` blocks never run and are exempt. The few edges that
cannot move yet are in ``LAYERING_ALLOWLIST`` with a reason and, where they
are deprecation bridges, the release that removes them.

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

# Deprecated in v0.38.0 ("one release"); removed from both lazy-export tables.
REMOVED_PLANNING_NAMES = (
    "PlanningConfig",
    "PlanningManager",
    "Planner",
    "Plan",
    "PlanStep",
    "PlanStatus",
    "HeuristicPlanner",
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
    """True for a bare ``warnings.warn(...)`` or ``warn_deprecated(...)`` call."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    if isinstance(func, ast.Name):
        return func.id == "warn_deprecated"
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "warn"
        and isinstance(func.value, ast.Name)
        and func.value.id == "warnings"
    )


def _non_shim_statements(path: Path) -> list[str]:
    """Return descriptions of statements a deprecation shim may not contain.

    A shim may only hold: a module docstring, imports, plain assignments
    (re-exports, ``__all__``), and a ``warnings.warn(...)`` /
    ``warn_deprecated(...)`` call.
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
            "(only a docstring, imports, assignments and a deprecation warning "
            "call are allowed): " + ", ".join(bad)
        )


@pytest.mark.parametrize("name", REMOVED_PLANNING_NAMES)
def test_removed_planning_exports_are_gone(name: str) -> None:
    import stateset_agents
    import stateset_agents.core

    for package in (stateset_agents, stateset_agents.core):
        assert name not in package.__all__
        with pytest.raises(AttributeError):
            getattr(package, name)


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


# --- No bottom-of-file imports under core/ ---------------------------------


def _is_type_checking_guard(test: ast.expr) -> bool:
    """True for ``if TYPE_CHECKING:`` / ``if typing.TYPE_CHECKING:``."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _record(path: Path, node: ast.stmt, offenders: list[str]) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
    elif isinstance(node, ast.ImportFrom):
        dots = "." * node.level
        offenders.append(f"{path.name}:{node.lineno} from {dots}{node.module or ''}")


def _scan_body(path: Path, body: list[ast.stmt], offenders: list[str]) -> None:
    """Record module-level imports in ``body``, descending into if/try only."""
    for node in body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            _record(path, node, offenders)
        elif isinstance(node, ast.If):
            if _is_type_checking_guard(node.test):
                continue
            _scan_body(path, node.body, offenders)
            _scan_body(path, node.orelse, offenders)
        elif isinstance(node, ast.Try):
            _scan_body(path, node.body, offenders)
            for handler in node.handlers:
                _scan_body(path, handler.body, offenders)
            _scan_body(path, node.orelse, offenders)
            _scan_body(path, node.finalbody, offenders)


def _imports_after_first_definition(path: Path) -> list[str]:
    """Return module-level imports that appear after the first def/class.

    A top-level ``import`` placed below the definitions it feeds is a cycle
    worked around by import order: the module only loads because something
    else imported it first. Imports inside functions are not module-level and
    do not count -- but imports nested in a top-level ``if``/``try`` DO
    execute at import time and are the same cycle one indent down, so those
    bodies are walked too. ``if TYPE_CHECKING:`` is exempt: it never runs.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    seen_definition = False
    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            seen_definition = True
            continue
        if not seen_definition:
            continue
        _scan_body(path, [node], offenders)
    return offenders


def test_core_has_no_bottom_of_file_imports() -> None:
    offenders: list[str] = []
    for path in _core_modules():
        offenders.extend(_imports_after_first_definition(path))
    assert offenders == [], (
        "no module under stateset_agents/core may place a module-level import "
        "after its first function or class definition: such an import is a "
        "circular dependency that only resolves by import order. Re-export "
        "lazily with a module-level __getattr__ instead: " + ", ".join(offenders)
    )


# --- the bottom-of-file import detector itself ------------------------------


def _detect(tmp_path: Path, source: str) -> list[str]:
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8")
    return _imports_after_first_definition(path)


def test_detector_finds_a_plain_bottom_import(tmp_path: Path) -> None:
    assert _detect(tmp_path, "def f():\n    pass\n\nimport os\n")


def test_detector_finds_an_import_hidden_in_a_top_level_if(tmp_path: Path) -> None:
    """`if True: import x` at the bottom is the same cycle, one indent down."""
    offenders = _detect(
        tmp_path,
        "def f():\n    pass\n\nif True:\n    from .other import thing\n",
    )
    assert offenders and "other" in offenders[0]


def test_detector_finds_an_import_hidden_in_a_top_level_try(tmp_path: Path) -> None:
    offenders = _detect(
        tmp_path,
        "def f():\n    pass\n\ntry:\n    import torch\nexcept ImportError:\n"
        "    import fallback\n",
    )
    assert len(offenders) == 2


def test_detector_ignores_if_type_checking(tmp_path: Path) -> None:
    """TYPE_CHECKING imports never execute, so they cannot form a cycle."""
    assert (
        _detect(
            tmp_path,
            "from typing import TYPE_CHECKING\n\ndef f():\n    pass\n\n"
            "if TYPE_CHECKING:\n    from .other import Thing\n",
        )
        == []
    )


def test_detector_ignores_typing_dot_type_checking(tmp_path: Path) -> None:
    assert (
        _detect(
            tmp_path,
            "import typing\n\ndef f():\n    pass\n\n"
            "if typing.TYPE_CHECKING:\n    from .other import Thing\n",
        )
        == []
    )


def test_detector_ignores_imports_inside_functions(tmp_path: Path) -> None:
    assert _detect(tmp_path, "def f():\n    import os\n    return os\n") == []


def test_detector_ignores_imports_above_the_first_definition(tmp_path: Path) -> None:
    assert _detect(tmp_path, "import os\n\ndef f():\n    return os\n") == []


# --- Layer contracts over every executable import ---------------------------

#: (layer, layers it must never import).
LAYER_CONTRACTS: tuple[tuple[str, frozenset[str]], ...] = (
    ("core", frozenset({"training", "api", "experimental"})),
    ("training", frozenset({"experimental"})),
)

#: (importing file, imported module prefix) -> (reason, removal release or "").
#: An entry with a removal release must be a deprecation bridge whose module
#: states that release; ``test_deprecation_schedule.py`` then fails the build
#: once it is due, so the edge cannot outlive its deprecation.
LAYERING_ALLOWLIST: dict[tuple[str, str], tuple[str, str]] = {
    (
        "stateset_agents/core/agent.py",
        "stateset_agents.experimental.long_term_planning",
    ): (
        "AgentConfig(enable_planning=True) without planning_manager= builds the "
        "experimental planner implicitly; deprecated, inject planning_manager",
        "0.72.0",
    ),
    (
        "stateset_agents/training/train.py",
        "stateset_agents.experimental.offline_rl",
    ): (
        "train(training_mode='offline'|'hybrid') loads the experimental offline "
        "trainer when nothing registered one (training.offline_backend); "
        "deprecated, import stateset_agents.experimental.offline_rl",
        "0.72.0",
    ),
    (
        "stateset_agents/core/multiturn_agent.py",
        "stateset_agents.experimental.long_term_planning",
    ): ("the whole module is deprecated (DialogueSessionAgent)", "0.71.0"),
    ("stateset_agents/core/types.py", "stateset_agents.training.config"): (
        "deprecated core.types.TrainingConfig alias, served lazily; the module "
        "is deprecated",
        "0.71.0",
    ),
    (
        "stateset_agents/core/enhanced/advanced_rl_algorithms.py",
        "stateset_agents.training.gspo_config",
    ): (
        "deprecated core.enhanced GSPOConfig alias, served lazily; the module "
        "is deprecated",
        "0.71.0",
    ),
}

_PACKAGE_NAME = "stateset_agents"


def _module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve(node: ast.ImportFrom, module: str, is_package: bool) -> str:
    if node.level == 0:
        return node.module or ""
    package = module.split(".") if is_package else module.split(".")[:-1]
    base = package[: len(package) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _string_import(node: ast.Call) -> str | None:
    """``importlib.import_module("x")`` / ``import_module("x")`` / ``__import__``."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if name not in {"import_module", "__import__"} or not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def executable_imports(path: Path, root: Path) -> list[tuple[int, str]]:
    """``(line, module)`` for every import ``path`` can execute.

    Walks the whole tree (functions, methods, nested blocks) but skips
    ``if TYPE_CHECKING:`` bodies. ``from pkg import name`` yields both ``pkg``
    and ``pkg.name`` (``name`` may be a submodule).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module = _module_name(path, root)
    is_package = path.name == "__init__.py"
    found: list[tuple[int, str]] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.If) and _is_type_checking_guard(node.test):
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(node, module, is_package)
            found.append((node.lineno, base))
            found.extend((node.lineno, f"{base}.{alias.name}") for alias in node.names)
        elif isinstance(node, ast.Call):
            target = _string_import(node)
            if target is not None:
                found.append((node.lineno, target))
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return found


def _layer_of(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != _PACKAGE_NAME:
        return None
    return parts[1]


def _allowlisted(rel: str, target: str) -> tuple[str, str] | None:
    for path, prefix in LAYERING_ALLOWLIST:
        if rel == path and (target == prefix or target.startswith(prefix + ".")):
            return path, prefix
    return None


def layering_violations(
    root: Path = PACKAGE_DIR,
) -> tuple[list[str], set[tuple[str, str]]]:
    """Return ``(violations, allowlist entries in use)`` for ``root``."""
    violations: list[str] = []
    used: set[tuple[str, str]] = set()
    for layer, forbidden in LAYER_CONTRACTS:
        layer_dir = root / layer
        if not layer_dir.is_dir():
            continue
        for path in sorted(layer_dir.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root.parent).as_posix()
            if (
                layer == "core"
                and path.parent == CORE_DIR
                and path.name in (DEPRECATION_SHIMS)
            ):
                continue  # pure re-export shims, policed by the tests above
            seen: set[tuple[int, str]] = set()
            for lineno, target in executable_imports(path, root):
                if _layer_of(target) not in forbidden:
                    continue
                key = _allowlisted(rel, target)
                if key is not None:
                    used.add(key)
                    continue
                top = ".".join(target.split(".")[:2])
                if (lineno, top) in seen:
                    continue
                seen.add((lineno, top))
                violations.append(f"{rel}:{lineno} -> {target}")
    return violations, used


def test_layer_contracts_hold_for_every_executable_import() -> None:
    violations, _ = layering_violations()
    assert violations == [], (
        "Layer contract broken (core must not import training/api/experimental; "
        "training must not import experimental), including imports inside "
        "functions. Move the shared code down a layer, inject it, or invert "
        "the dependency (see training.offline_backend); only a reasoned, dated "
        "LAYERING_ALLOWLIST entry may excuse an edge:\n  " + "\n  ".join(violations)
    )


def test_layering_allowlist_has_no_stale_entries() -> None:
    _, used = layering_violations()
    stale = sorted(set(LAYERING_ALLOWLIST) - used)
    assert stale == [], f"remove LAYERING_ALLOWLIST entries no longer needed: {stale}"


@pytest.mark.parametrize(
    "key", sorted(LAYERING_ALLOWLIST), ids=lambda key: f"{key[0]}->{key[1]}"
)
def test_dated_allowlist_entries_are_deprecation_bridges(key: tuple[str, str]) -> None:
    from tests.unit.test_deprecation_schedule import (
        _searchable_text,
        alias_table_removals,
        scheduled_removals,
    )

    reason, removal = LAYERING_ALLOWLIST[key]
    assert reason.strip(), "every allowlist entry needs a reason"
    assert removal, "every edge left in the allowlist must have a removal date"
    path = PACKAGE_DIR.parent / key[0]
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    stated = set(scheduled_removals(_searchable_text(text, tree)))
    stated |= set(alias_table_removals(tree))
    assert removal in stated, (
        f"{key[0]} is allowlisted until v{removal} but does not state that "
        f"removal release (states {sorted(stated)})"
    )


# --- the executable-import detector itself ----------------------------------


def _probe(tmp_path: Path, rel: str, source: str) -> list[str]:
    root = tmp_path / _PACKAGE_NAME
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    for parent in path.parents:
        if parent == tmp_path:
            break
        (parent / "__init__.py").touch()
    path.write_text(source, encoding="utf-8")
    violations, _ = layering_violations(root)
    return violations


def test_detector_finds_function_level_imports(tmp_path: Path) -> None:
    source = (
        "def f():\n    from ..training.rl_losses import whiten\n    return whiten\n"
    )
    assert _probe(tmp_path, "core/mod.py", source) == [
        "stateset_agents/core/mod.py:2 -> stateset_agents.training.rl_losses"
    ]


def test_detector_finds_method_and_absolute_imports(tmp_path: Path) -> None:
    source = (
        "class A:\n    def m(self):\n"
        "        import stateset_agents.experimental.ppo as ppo\n"
        "        return ppo\n"
    )
    assert _probe(tmp_path, "training/mod.py", source) == [
        "stateset_agents/training/mod.py:3 -> stateset_agents.experimental.ppo"
    ]


def test_detector_resolves_package_relative_imports(tmp_path: Path) -> None:
    source = "def f():\n    from .. import api\n    return api\n"
    assert _probe(tmp_path, "core/sub/__init__.py", source) == []
    assert _probe(tmp_path, "core/mod2.py", source) == [
        "stateset_agents/core/mod2.py:2 -> stateset_agents.api"
    ]


def test_detector_finds_string_imports(tmp_path: Path) -> None:
    source = (
        "import importlib\n\ndef f():\n"
        "    importlib.import_module('stateset_agents.experimental.bayes')\n"
    )
    assert _probe(tmp_path, "training/mod.py", source) == [
        "stateset_agents/training/mod.py:4 -> stateset_agents.experimental.bayes"
    ]


def test_detector_ignores_type_checking_and_allowed_directions(tmp_path: Path) -> None:
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    from ..experimental.ppo import PPOTrainer\n"
        "def f():\n    from ..core.errors import StateSetError\n"
    )
    assert _probe(tmp_path, "training/mod.py", source) == []


def test_core_whiten_matches_the_training_loss_primitive() -> None:
    """``core.numerics.whiten`` replaced value_function's import of
    ``training.rl_losses.whiten``; the two must stay numerically identical."""
    torch = pytest.importorskip("torch")
    from stateset_agents.core.numerics import whiten as core_whiten
    from stateset_agents.training.rl_losses import whiten as loss_whiten

    generator = torch.Generator().manual_seed(0)
    x = torch.randn(4, 7, generator=generator)
    mask = (torch.rand(4, 7, generator=generator) > 0.3).float()
    cases = [
        ((x,), {}),
        ((x, mask), {}),
        ((x, mask), {"correction": 0}),
        ((torch.full((5,), 2.5),), {}),
        ((torch.tensor([3.0]),), {}),
        ((torch.tensor([1.0, float("nan")]),), {}),
    ]
    for args, kwargs in cases:
        assert torch.equal(
            core_whiten(*args, **kwargs).nan_to_num(7.0),
            loss_whiten(*args, **kwargs).nan_to_num(7.0),
        )
