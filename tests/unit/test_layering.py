"""
Layering meta-tests.

The layer contracts live in ``scripts/layer_graph.py`` (``LAYERS``): every
unit of ``stateset_agents`` -- each subpackage, the ``cli``/``tui``/
``finetune`` module families and the ``errors`` root -- sits in one layer and
may import only itself and units in *lower* layers. They hold for *every*
import a module can execute -- module level, inside functions and methods,
literal ``importlib.import_module`` strings, ``from stateset_agents import
Name`` (resolved through the root lazy-export table) and the targets of
``deprecated_alias_getattr`` tables. ``if TYPE_CHECKING:`` blocks never run
and are exempt.

The only upward edges are dated: pure deprecation shims at old import paths
(``DEPRECATION_SHIMS``) and ``LAYERING_ALLOWLIST`` entries. Without them the
unit graph is a DAG, and the cycle check fails on any new edge that closes a
cycle. ``docs/ARCHITECTURE.md``'s Layering section and package layout are
rendered from the same table and must stay current.

Structured so later layering rules can be appended: helpers first, then the
individual rule tests.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "stateset_agents"
CORE_DIR = PACKAGE_DIR / "core"

_spec = importlib.util.spec_from_file_location(
    "layer_graph", ROOT / "scripts" / "layer_graph.py"
)
assert _spec is not None and _spec.loader is not None
layer_graph = importlib.util.module_from_spec(_spec)
# Registered first: the script's dataclasses resolve their module by name.
sys.modules[_spec.name] = layer_graph
_spec.loader.exec_module(layer_graph)

#: Every pure deprecation shim, any package (paths relative to the package).
#: They may re-export a module that moved up a layer; the shape test below
#: keeps them from hiding real code.
DEPRECATION_SHIMS = frozenset(
    path.removeprefix("stateset_agents/") for path in layer_graph.DEPRECATION_SHIMS
)

MAX_SHIM_LINES = 35

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


def _is_shim(path: Path) -> bool:
    return path.relative_to(PACKAGE_DIR).as_posix() in DEPRECATION_SHIMS


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


def _is_main_guard(node: ast.stmt) -> bool:
    """``if __name__ == "__main__":`` -- a shim may keep ``python -m`` working."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    return (
        isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _is_plain_statement(node: ast.stmt) -> bool:
    if isinstance(node, ast.Import | ast.ImportFrom | ast.Assign | ast.AnnAssign):
        return True
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)


def _non_shim_statements(path: Path) -> list[str]:
    """Return descriptions of statements a deprecation shim may not contain.

    A shim may only hold: a module docstring, imports, plain assignments
    (re-exports, ``__all__``, the ``sys.modules`` alias), a
    ``warnings.warn(...)`` / ``warn_deprecated(...)`` call, and one
    ``if __name__ == "__main__":`` block of imports, assignments and calls
    that forwards ``python -m`` to the new module.
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
        if (
            isinstance(node, ast.If)
            and _is_main_guard(node)
            and all(_is_plain_statement(child) for child in [*node.body, *node.orelse])
        ):
            continue
        bad.append(f"{path.name}:{node.lineno} {type(node).__name__}")
    return bad


def test_core_does_not_import_experimental_at_module_level() -> None:
    offenders: list[str] = []
    for path in _core_modules():
        if _is_shim(path):
            continue
        offenders.extend(_experimental_module_level_imports(path))
    assert offenders == [], (
        "stateset_agents.core must not import stateset_agents.experimental at "
        "module level (move the import into the function that uses it): "
        + ", ".join(offenders)
    )


@pytest.mark.parametrize("name", sorted(DEPRECATION_SHIMS))
def test_deprecation_shims_stay_tiny(name: str) -> None:
    from tests.unit.test_deprecation_schedule import (
        _searchable_text,
        scheduled_removals,
    )

    path = PACKAGE_DIR / name
    assert path.exists(), f"exempted shim {name} no longer exists"
    text = path.read_text(encoding="utf-8")
    line_count = len(text.splitlines())
    assert line_count <= MAX_SHIM_LINES, (
        f"{name} is {line_count} lines; exempted shims must stay "
        f"<= {MAX_SHIM_LINES} lines so the layering exemption cannot hide "
        "real code"
    )
    bad = _non_shim_statements(path)
    assert bad == [], (
        f"{name} is exempted from the layer contracts because it is a pure "
        "deprecation shim, but it contains statements a shim may not have "
        "(only a docstring, imports, assignments, a deprecation warning call "
        "and a `python -m` forwarder are allowed): " + ", ".join(bad)
    )
    assert scheduled_removals(
        _searchable_text(text, ast.parse(text))
    ), f"{name} must state the release that removes it"


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
    return bool(layer_graph.is_type_checking_guard(test))


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

#: Re-exported for readers of this file: the contract table, the dated
#: exceptions, and the unit assignment all live in scripts/layer_graph.py.
LAYERS = layer_graph.LAYERS
LAYERING_ALLOWLIST = layer_graph.LAYERING_ALLOWLIST


def layering_violations(
    root: Path = PACKAGE_DIR,
) -> tuple[list[str], set[tuple[str, str]]]:
    """Return ``(violation keys, allowlist entries in use)`` for ``root``."""
    report = layer_graph.check_contracts(root)
    return [v.key for v in report.violations], report.used_allowlist


@pytest.fixture(scope="module")
def contract_report():
    return layer_graph.check_contracts(PACKAGE_DIR)


@pytest.fixture(scope="module")
def graph():
    return layer_graph.build_graph(PACKAGE_DIR)


def test_every_unit_has_exactly_one_layer(graph) -> None:
    placements = [unit for layer in LAYERS for unit in layer.units]
    duplicates = sorted({u for u in placements if placements.count(u) > 1})
    assert duplicates == [], f"units placed in more than one layer: {duplicates}"
    placed = set(placements)
    unplaced = sorted(set(graph.units) - placed - {layer_graph.FACADE})
    assert unplaced == [], (
        f"new units {unplaced} have no layer: add them to LAYERS in "
        "scripts/layer_graph.py (and regenerate docs/ARCHITECTURE.md)"
    )
    assert placed <= set(graph.units), "LAYERS names a unit that no longer exists"


def test_layer_contracts_hold_for_every_executable_import(contract_report) -> None:
    violations = [v.explain() for v in contract_report.violations]
    assert violations == [], (
        "Layer contract broken (docs/ARCHITECTURE.md, Layering): each unit may "
        "import only units in lower layers, including imports inside "
        "functions. Move the shared code down a layer, inject it, or invert "
        "the dependency (see training.offline_backend); only a pure dated "
        "deprecation shim or a reasoned, dated LAYERING_ALLOWLIST entry in "
        "scripts/layer_graph.py may excuse an edge:\n  " + "\n  ".join(violations)
    )


def test_framework_layers_never_import_app_layers(contract_report) -> None:
    """``core``/``data``/``evaluation``/``rewards``/``training`` (and the
    layers under them) never reach into ``remote``, the studio, ``api``,
    ``tui``, ``cli`` or ``mcp_server`` -- function-level imports included,
    and no allowlist entry may excuse such an edge."""
    apps = {
        unit
        for layer in LAYERS
        if layer.name in layer_graph.APP_LAYERS
        for unit in layer.units
    }
    framework = {
        unit
        for layer in LAYERS
        if layer.name in layer_graph.FRAMEWORK_LAYERS
        for unit in layer.units
    }
    assert apps.isdisjoint(framework)
    assert apps | framework == {u for layer in LAYERS for u in layer.units}
    assert {"cli", "tui", "finetune", "api", "remote"} <= apps
    assert {"core", "rewards", "data", "training", "evaluation"} <= framework
    offenders = [
        v.explain()
        for v in contract_report.violations
        if v.source_unit in framework and v.target_unit in apps
    ]
    excused = [
        item.key
        for item in contract_report.excused
        if layer_graph.unit_of(item.source) in framework
        and layer_graph.unit_of(item.target) in apps
    ]
    assert (
        offenders == [] and excused == []
    ), "a framework layer imports an app layer:\n  " + "\n  ".join(offenders + excused)


def test_unit_graph_has_no_cycles(graph) -> None:
    cycles = layer_graph.strongly_connected_components(graph.units, graph.enforced)
    detail = []
    for component in cycles:
        members = set(component)
        for (src, dst), items in sorted(graph.enforced.items()):
            if src in members and dst in members:
                detail += [f"{src} -> {dst}: {item.key}" for item in items[:3]]
    assert cycles == [], (
        "import cycle between units (ignoring dated deprecation bridges): "
        + "; ".join(" <-> ".join(c) for c in cycles)
        + "\n  "
        + "\n  ".join(detail)
    )


def test_cycles_left_run_only_through_dated_bridges(graph) -> None:
    """Every cycle in the full graph needs a sanctioned (dated) edge to close."""
    sanctioned_only = {key for key in graph.edges if key not in graph.enforced}
    for cycle in layer_graph.elementary_cycles(graph.edges):
        hops = list(zip(cycle, cycle[1:] + cycle[:1], strict=True))
        assert any(
            hop in sanctioned_only for hop in hops
        ), "cycle with no dated bridge: " + " -> ".join(cycle + cycle[:1])


def test_layering_allowlist_has_no_stale_entries(contract_report) -> None:
    stale = sorted(set(LAYERING_ALLOWLIST) - contract_report.used_allowlist)
    assert stale == [], f"remove LAYERING_ALLOWLIST entries no longer needed: {stale}"


@pytest.mark.parametrize(
    "key", sorted(LAYERING_ALLOWLIST), ids=lambda key: f"{key[0]}->{key[1]}"
)
def test_allowlist_entries_are_dated(key: tuple[str, str]) -> None:
    from tests.unit.test_deprecation_schedule import (
        _package_version,
        _searchable_text,
        _version_tuple,
        alias_table_removals,
        scheduled_removals,
    )

    kind, reason, removal = LAYERING_ALLOWLIST[key]
    assert reason.strip(), "every allowlist entry needs a reason"
    assert re.fullmatch(r"\d+\.\d+\.\d+", removal), "removal must be X.Y.Z"
    assert kind in {"bridge", "debt"}
    assert _version_tuple(removal) > _package_version(), (
        f"{key[0]} -> {key[1]} was allowed until v{removal}; that release is "
        "due, so remove the edge"
    )
    if kind == "bridge":
        text = (ROOT / key[0]).read_text(encoding="utf-8")
        tree = ast.parse(text)
        stated = set(scheduled_removals(_searchable_text(text, tree)))
        stated |= set(alias_table_removals(tree))
        assert removal in stated, (
            f"{key[0]} is allowlisted until v{removal} but does not state that "
            f"removal release (states {sorted(stated)})"
        )


def test_documented_layering_is_current() -> None:
    doc = ROOT / "docs" / "ARCHITECTURE.md"
    assert layer_graph.documented_section(doc) == layer_graph.render_section(), (
        "docs/ARCHITECTURE.md's generated Layering section is stale (a unit "
        "gained or lost a dependency, or the contract table changed); "
        "regenerate with `python scripts/layer_graph.py --write "
        "docs/ARCHITECTURE.md`"
    )


def test_documented_package_layout_is_current() -> None:
    doc = ROOT / "docs" / "ARCHITECTURE.md"
    problems = layer_graph.layout_drift(layer_graph.documented_layout(doc))
    assert problems == [], (
        "docs/ARCHITECTURE.md's generated package layout is stale ("
        + "; ".join(problems)
        + "); regenerate with `python scripts/layer_graph.py --write "
        "docs/ARCHITECTURE.md`"
    )


def test_layout_drift_tolerates_small_line_changes() -> None:
    actual = layer_graph.render_layout()
    assert layer_graph.layout_drift(actual) == []
    assert layer_graph.layout_drift(None)
    first_row = next(line for line in actual.splitlines() if line.startswith("| `"))
    unit = first_row.split("|")[1]
    renamed = actual.replace(first_row, first_row.replace(unit, " `renamed` "))
    assert layer_graph.layout_drift(renamed)


# --- product verticals stay out of core -------------------------------------

#: Product-vertical modules that predate the rule. ``core`` holds framework
#: primitives; verticals belong in ``stateset_agents.verticals``. These stay
#: this pass because data/evaluation/remote/training still import them
#: (docs/ARCHITECTURE.md, Layering); the set only shrinks.
CORE_VERTICAL_MODULES = frozenset(
    {
        "core/environments/industry_admin_policies.py",
        "core/environments/industry_operations_policies.py",
        "core/environments/industry_support_environment.py",
        "core/environments/refund_delegation_environment.py",
        "core/environments/refund_environment.py",
        "core/environments/refund_policy_environment.py",
        "core/environments/refund_recovery_environment.py",
        "core/model_presets.py",
        "core/refund_tools.py",
    }
)

#: Generic environments that may live in core/environments. A new module
#: there must be generic (add it here) or go to stateset_agents.verticals.
CORE_GENERIC_ENVIRONMENTS = frozenset(
    {
        "core/environments/json_action.py",
        "core/environments/multimodal_environment.py",
        "core/environments/tool_use.py",
    }
)

_VERTICAL_NAME = re.compile(
    r"refund|industry|insurance|ecommerce|retail|logistics|healthcare|"
    r"manufactur|customer|support_|banking|telecom|model_preset|model_catalog",
    re.IGNORECASE,
)


def test_core_holds_no_new_product_verticals() -> None:
    core = {p.relative_to(PACKAGE_DIR).as_posix() for p in _core_modules()}
    named = {
        rel
        for rel in core
        if _VERTICAL_NAME.search(rel.rsplit("/", 1)[-1])
        and not rel.endswith("__init__.py")
    }
    environments = {rel for rel in core if rel.startswith("core/environments/")}
    new = sorted(
        (named - CORE_VERTICAL_MODULES)
        | (environments - CORE_VERTICAL_MODULES - CORE_GENERIC_ENVIRONMENTS)
    )
    assert new == [], (
        "core holds framework primitives; put product verticals (refund, "
        "industry, support, model catalogs, ...) in stateset_agents.verticals, "
        "or list a generic environment in CORE_GENERIC_ENVIRONMENTS: " + ", ".join(new)
    )
    stale = sorted((CORE_VERTICAL_MODULES | CORE_GENERIC_ENVIRONMENTS) - core)
    assert stale == [], f"remove moved modules from the core vertical ratchet: {stale}"


# --- the executable-import detector itself ----------------------------------


def _probe(tmp_path: Path, rel: str, source: str) -> list[str]:
    root = tmp_path / "stateset_agents"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    for parent in path.parents:
        if parent == tmp_path:
            break
        (parent / "__init__.py").touch()
    path.write_text(source, encoding="utf-8")
    layer_graph.clear_caches()
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


def test_detector_follows_the_facade_lazy_exports(tmp_path: Path) -> None:
    """``from stateset_agents import train`` loads training.train; reading the
    facade's own ``__version__`` is allowed everywhere."""
    root = tmp_path / "stateset_agents"
    root.mkdir()
    (root / "__init__.py").write_text(
        "_LAZY_EXPORTS = {}\n"
        "_LAZY_EXPORTS.update(_export_group('stateset_agents.training.train',"
        " ['train']))\n",
        encoding="utf-8",
    )
    source = "def f():\n    from stateset_agents import __version__, train\n"
    assert _probe(tmp_path, "core/mod3.py", source) == [
        "stateset_agents/core/mod3.py:2 -> stateset_agents.training.train"
    ]


def test_detector_counts_deprecated_alias_tables(tmp_path: Path) -> None:
    source = (
        "from stateset_agents._deprecation import deprecated_alias_getattr\n"
        "__getattr__ = deprecated_alias_getattr(\n"
        "    __name__, {'Old': ('stateset_agents.training.config', 'New', '9.0.0')}\n"
        ")\n"
    )
    assert _probe(tmp_path, "core/mod4.py", source) == [
        "stateset_agents/core/mod4.py:2 -> stateset_agents.training.config"
    ]


def test_detector_flags_sibling_units_in_one_layer(tmp_path: Path) -> None:
    """``data`` and ``evaluation`` share a layer, so neither imports the other."""
    source = "from stateset_agents.data.jsonl import read_jsonl_lines\n"
    assert _probe(tmp_path, "evaluation/mod.py", source) == [
        "stateset_agents/evaluation/mod.py:1 -> stateset_agents.data.jsonl"
    ]


def test_violation_message_names_the_contract(tmp_path: Path) -> None:
    _probe(tmp_path, "utils/mod.py", "from stateset_agents.core import agent\n")
    report = layer_graph.check_contracts(tmp_path / "stateset_agents")
    (violation,) = report.violations
    message = violation.explain()
    assert message.startswith("stateset_agents/utils/mod.py:1 -> ")
    assert "utils (utilities) must not import core (core)" in message


def test_cycle_detection_finds_every_component() -> None:
    edges = [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("e", "f"), ("f", "e")]
    assert layer_graph.strongly_connected_components("abcdef", edges) == [
        ["a", "b", "c"],
        ["e", "f"],
    ]
    assert layer_graph.elementary_cycles(edges) == [["e", "f"], ["a", "b", "c"]]
    assert layer_graph.strongly_connected_components("ab", [("a", "b")]) == []
    assert layer_graph.mutual_pairs(edges) == [("e", "f")]


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
