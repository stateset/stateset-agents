"""Map, and enforce, the import graph between ``stateset_agents`` units.

Every import a module can *execute* counts: module level, inside functions
and methods, nested blocks, literal ``importlib.import_module("...")`` /
``__import__("...")`` calls, ``from stateset_agents import Name`` resolved
through the root package's lazy export table, and the targets of
``deprecated_alias_getattr`` tables. Only ``if TYPE_CHECKING:`` bodies are
skipped: they never run.

Modules are grouped into *units* -- one per subpackage (``core``,
``training``, ...), plus the top-level application families (``cli*.py``,
``__main__.py`` and ``mcp_server.py`` -> ``cli``; ``tui*.py`` -> ``tui``;
``finetune_*.py`` -> ``finetune``) and the exception root
(``core.errors`` + ``core.error_handling`` -> ``errors``). Units sit in the
ordered ``LAYERS`` below; a unit may import itself and units in *lower*
layers only. ``stateset_agents/__init__.py`` (the ``facade``) is the top of
the stack: every unit may read its own metadata (``__version__``), and its
lazy exports resolve to the module that defines them.

Edges that break a contract are either a dated deprecation bridge (a pure
shim in ``DEPRECATION_SHIMS``, or a ``bridge`` entry in ``LAYERING_ALLOWLIST``
whose module states its removal release) or dated ``debt`` -- nothing else
passes. ``tests/unit/test_layering.py`` enforces all of it; the section of
``docs/ARCHITECTURE.md`` between the ``layer-graph`` markers is rendered from
this module.

Usage::

    python scripts/layer_graph.py                  # unit edges, cycles
    python scripts/layer_graph.py --edges          # every cross-unit import
    python scripts/layer_graph.py --unit training  # one unit's in/out edges
    python scripts/layer_graph.py --json           # machine-readable graph
    python scripts/layer_graph.py --violations     # contract breaks (exit 1)
    python scripts/layer_graph.py --write docs/ARCHITECTURE.md
    python scripts/layer_graph.py --check docs/ARCHITECTURE.md
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "stateset_agents"
PACKAGE_DIR = ROOT / PACKAGE
DOC = ROOT / "docs" / "ARCHITECTURE.md"

FACADE = "facade"

# --- units -------------------------------------------------------------------

#: Module (and its submodules) -> unit, checked before the defaults below.
UNIT_OVERRIDES: dict[str, str] = {
    # The exception root every layer raises (AGENTS.md: errors derive from
    # core.errors.StateSetError). Both modules import only _deprecation and
    # each other, so they sit in the foundation, under utils.
    f"{PACKAGE}.core.errors": "errors",
    f"{PACKAGE}.core.error_handling": "errors",
    # ``stateset-agents mcp`` is the only way to start the MCP server, and its
    # tools wrap the CLI's improve loop: it is a command-line entry point.
    f"{PACKAGE}.mcp_server": "cli",
    f"{PACKAGE}.__main__": "cli",
    # A curses panel imported only by tui.py; ``finetune_*`` naming aside it
    # is terminal UI, not fine-tuning studio logic.
    f"{PACKAGE}.finetune_evaluation_tui": "tui",
}

#: Top-level module name prefix -> unit, for ``stateset_agents/<name>.py``.
TOP_LEVEL_FAMILIES: tuple[tuple[str, str], ...] = (
    ("cli", "cli"),
    ("tui", "tui"),
    ("finetune_", "finetune"),
)


@dataclass(frozen=True)
class Layer:
    """One rank of the stack: its units may import only lower layers."""

    name: str
    units: tuple[str, ...]
    role: str


#: Bottom to top. A unit may import its own modules and any unit in a lower
#: layer -- never a sibling in the same layer, never a higher layer.
LAYERS: tuple[Layer, ...] = (
    Layer(
        "foundation",
        ("_deprecation", "exceptions"),
        "deprecation helper and shared exception tuples; no package imports",
    ),
    Layer(
        "errors",
        ("errors",),
        "the StateSetError hierarchy (core.errors, core.error_handling)",
    ),
    Layer(
        "utilities",
        ("utils", "config"),
        "clock, logging, monitoring, atomic I/O, provenance; config presets",
    ),
    Layer("core", ("core",), "agents, environments, trajectories, reward primitives"),
    Layer(
        "data & evaluation",
        ("data", "evaluation"),
        "datasets and loaders; run records, judges, isolation, traces",
    ),
    Layer("rewards", ("rewards",), "reward functions, judges as rewards, verifiers"),
    Layer("packs", ("environments", "tools"), "optional environment and tool packs"),
    Layer("training", ("training",), "trainers, the training spine, train()"),
    Layer(
        "research", ("experimental",), "experimental algorithms; no stability guarantee"
    ),
    Layer("remote", ("remote",), "remote providers, River runtime, deployment"),
    Layer(
        "studio",
        ("finetune", "flywheel", "verticals", "scaffolding", "testing"),
        "fine-tuning studio, flywheel, product verticals, scaffolding, test kit",
    ),
    Layer("interfaces", ("api", "tui"), "FastAPI gateway; terminal UI"),
    Layer("entry points", ("cli",), "the stateset-agents command and its MCP server"),
)

#: The framework proper; ``APP_LAYERS`` are the applications shipped with it.
FRAMEWORK_LAYERS = frozenset(
    {"foundation", "errors", "utilities", "core", "data & evaluation"}
    | {"rewards", "packs", "training", "research"}
)
APP_LAYERS = frozenset({"remote", "studio", "interfaces", "entry points"})

# --- sanctioned exceptions ---------------------------------------------------

#: Pure deprecation shims at old import paths (docstring, imports, assignments
#: and one warn_deprecated call -- policed by test_layering.py). They re-export
#: a module that moved, often upward, until their stated removal release,
#: when test_deprecation_schedule.py makes CI delete them.
DEPRECATION_SHIMS: frozenset[str] = frozenset(
    f"{PACKAGE}/{rel}"
    for rel in (
        "core/adaptive_learning_controller.py",
        "core/few_shot_adaptation.py",
        "core/intelligent_orchestrator.py",
        "core/intelligent_orchestrator_logic.py",
        "core/intelligent_orchestrator_models.py",
        "core/long_term_planning.py",
        "core/multi_agent_coordination.py",
        "core/multimodal_processing.py",
        "core/neural_architecture_search.py",
        "data/industry_synthetic.py",
        "evaluation/acceptance.py",
        "evaluation/agent_study.py",
        "evaluation/industry.py",
        "evaluation/study_preflight.py",
        "training/river_progress.py",
        "training/river_refund.py",
    )
)

#: (importing file, imported module prefix) -> (kind, reason, removal).
#: ``bridge``: a deprecation path whose module states the removal release.
#: ``debt``: a placement that cannot move this pass; dated here, and the
#: build fails once the package version reaches the date.
LAYERING_ALLOWLIST: dict[tuple[str, str], tuple[str, str, str]] = {
    (f"{PACKAGE}/core/agent.py", f"{PACKAGE}.experimental.long_term_planning"): (
        "bridge",
        "AgentConfig(enable_planning=True) without planning_manager= builds the "
        "experimental planner implicitly; deprecated, inject planning_manager",
        "0.72.0",
    ),
    (f"{PACKAGE}/training/train.py", f"{PACKAGE}.experimental.offline_rl"): (
        "bridge",
        "train(training_mode='offline'|'hybrid') loads the experimental offline "
        "trainer when nothing registered one (training.offline_backend); "
        "deprecated, import stateset_agents.experimental.offline_rl",
        "0.72.0",
    ),
    (
        f"{PACKAGE}/core/multiturn_agent.py",
        f"{PACKAGE}.experimental.long_term_planning",
    ): ("bridge", "the whole module is deprecated (DialogueSessionAgent)", "0.71.0"),
    (f"{PACKAGE}/core/types.py", f"{PACKAGE}.training.config"): (
        "bridge",
        "deprecated core.types.TrainingConfig alias, served lazily; the module "
        "is deprecated",
        "0.71.0",
    ),
    (
        f"{PACKAGE}/core/enhanced/advanced_rl_algorithms.py",
        f"{PACKAGE}.training.gspo_config",
    ): (
        "bridge",
        "deprecated core.enhanced GSPOConfig alias, served lazily; the module "
        "is deprecated",
        "0.71.0",
    ),
    (
        f"{PACKAGE}/evaluation/reward_audit.py",
        f"{PACKAGE}.rewards.support_reward_audit",
    ): (
        "bridge",
        "audit_support_rewards and `python -m ...evaluation.reward_audit` moved "
        "up to rewards.support_reward_audit; served lazily with a warning",
        "0.71.0",
    ),
}


@dataclass(frozen=True)
class Import:
    """One executable import: file ``path`` (module ``source``) loads ``target``."""

    path: str  # repo-relative posix path of the importing file
    lineno: int
    source: str
    target: str

    @property
    def key(self) -> str:
        return f"{self.path}:{self.lineno} -> {self.target}"


@dataclass(frozen=True)
class Violation:
    """An import that breaks its unit's layer contract."""

    item: Import
    source_unit: str
    target_unit: str

    @property
    def key(self) -> str:
        return self.item.key

    def explain(self, package: str = PACKAGE) -> str:
        layer = layer_of(self.source_unit)
        target_layer = layer_of(self.target_unit)
        allowed = ", ".join(allowed_units(self.source_unit)) or "nothing"
        return (
            f"{self.key}\n      {self.source_unit} ({layer}) must not import "
            f"{self.target_unit} ({target_layer}); it may import: {allowed}"
        )


# --- AST helpers -------------------------------------------------------------


def is_type_checking_guard(test: ast.expr) -> bool:
    """True for ``if TYPE_CHECKING:`` / ``if typing.TYPE_CHECKING:``."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def module_name(path: Path, package_dir: Path) -> str:
    """Dotted module name of ``path`` inside ``package_dir``."""
    parts = list(path.relative_to(package_dir.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def resolve_from(node: ast.ImportFrom, module: str, is_package: bool) -> str:
    """Absolute module named by ``from <...> import``."""
    if node.level == 0:
        return node.module or ""
    package = module.split(".") if is_package else module.split(".")[:-1]
    base = package[: len(package) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _call_name(node: ast.Call) -> str:
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def string_import(node: ast.Call) -> str | None:
    """``importlib.import_module("x")`` / ``import_module("x")`` / ``__import__``."""
    if _call_name(node) not in {"import_module", "__import__"} or not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def alias_table_targets(node: ast.Call) -> list[str]:
    """Modules a ``deprecated_alias_getattr(name, {old: (module, new, ver)})``
    table loads on attribute access."""
    if _call_name(node) != "deprecated_alias_getattr" or len(node.args) < 2:
        return []
    table = node.args[1]
    if not isinstance(table, ast.Dict):
        return []
    targets: list[str] = []
    for value in table.values:
        if (
            isinstance(value, ast.Tuple)
            and value.elts
            and isinstance(value.elts[0], ast.Constant)
            and isinstance(value.elts[0].value, str)
        ):
            targets.append(value.elts[0].value)
    return targets


@lru_cache(maxsize=8)
def known_modules(package_dir: Path = PACKAGE_DIR) -> frozenset[str]:
    return frozenset(
        module_name(path, package_dir) for path in package_files(package_dir)
    )


@lru_cache(maxsize=8)
def facade_exports(package_dir: Path = PACKAGE_DIR) -> dict[str, str]:
    """Name -> defining module for the root package's lazy exports."""
    init = package_dir / "__init__.py"
    if not init.exists():
        return {}
    tree = ast.parse(init.read_text(encoding="utf-8"))
    exports: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "_export_group":
            if len(node.args) >= 2 and isinstance(node.args[0], ast.Constant):
                module = node.args[0].value
                names = node.args[1]
                if isinstance(module, str) and isinstance(names, ast.List):
                    for elt in names.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            exports[elt.value] = module
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "_LAZY_EXPORTS"
                and isinstance(target.slice, ast.Constant)
                and isinstance(node.value, ast.Tuple)
                and node.value.elts
                and isinstance(node.value.elts[0], ast.Constant)
            ):
                exports[str(target.slice.value)] = str(node.value.elts[0].value)
    return exports


def clear_caches() -> None:
    """Forget every cached scan (for callers that edit a tree and rescan)."""
    for cached in (
        known_modules,
        facade_exports,
        _collect_imports,
        check_contracts,
        build_graph,
    ):
        cached.cache_clear()


def _longest_module(target: str, modules: Iterable[str]) -> str | None:
    known = set(modules)
    parts = target.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in known:
            return candidate
        parts.pop()
    return None


def executable_imports(
    path: Path, package_dir: Path = PACKAGE_DIR
) -> list[tuple[int, str]]:
    """``(line, module)`` for every import ``path`` can execute.

    Walks the whole tree (functions, methods, nested blocks) but skips
    ``if TYPE_CHECKING:`` bodies. Each import resolves to the module it
    loads: ``from a import b`` is ``a.b`` when that is a module and ``a``
    otherwise; ``from stateset_agents import Name`` follows the root lazy
    export table (or the facade's submodule lookup); a dotted name that is
    not in the scanned tree is kept as written.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = package_dir.name
    modules = known_modules(package_dir)
    exports = facade_exports(package_dir)
    module = module_name(path, package_dir)
    is_package = path.name == "__init__.py"
    found: list[tuple[int, str]] = []

    def resolve(target: str) -> str:
        if target in modules:
            return target
        prefix = _longest_module(target, modules)
        if prefix is None or prefix == package:
            return target
        return prefix

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.If) and is_type_checking_guard(node.test):
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, ast.Import):
            found.extend((node.lineno, resolve(alias.name)) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = resolve_from(node, module, is_package)
            for alias in node.names:
                full = f"{base}.{alias.name}"
                if full in modules:
                    found.append((node.lineno, full))
                elif base == package:
                    if alias.name in exports:
                        found.append((node.lineno, exports[alias.name]))
                    elif alias.name.startswith("__"):
                        found.append((node.lineno, base))  # facade metadata
                    else:  # a subpackage served by the facade's __getattr__
                        found.append((node.lineno, full))
                else:
                    found.append((node.lineno, resolve(base)))
        elif isinstance(node, ast.Call):
            target = string_import(node)
            if target is not None:
                found.append((node.lineno, resolve(target)))
            found.extend(
                (node.lineno, resolve(target)) for target in alias_table_targets(node)
            )
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return found


# --- package model -----------------------------------------------------------


def package_files(package_dir: Path = PACKAGE_DIR) -> list[Path]:
    return sorted(
        path for path in package_dir.rglob("*.py") if "__pycache__" not in path.parts
    )


def unit_of(module: str, package: str = PACKAGE) -> str | None:
    """The unit a dotted module belongs to, or ``None`` outside the package."""
    parts = module.split(".")
    if parts[0] != package:
        return None
    if len(parts) == 1:
        return FACADE
    for prefix, unit in UNIT_OVERRIDES.items():
        prefix = package + prefix[len(PACKAGE) :]
        if module == prefix or module.startswith(prefix + "."):
            return unit
    head = parts[1]
    if len(parts) == 2:
        for family, unit in TOP_LEVEL_FAMILIES:
            if head.startswith(family):
                return unit
    return head


def unit_of_path(path: Path, package_dir: Path = PACKAGE_DIR) -> str:
    unit = unit_of(module_name(path, package_dir), package_dir.name)
    assert unit is not None
    return unit


def collect_imports(package_dir: Path = PACKAGE_DIR) -> list[Import]:
    """Every executable intra-package import, resolved to the module it loads."""
    return list(_collect_imports(package_dir))


@lru_cache(maxsize=8)
def _collect_imports(package_dir: Path) -> tuple[Import, ...]:
    package = package_dir.name
    imports: list[Import] = []
    for path in package_files(package_dir):
        source = module_name(path, package_dir)
        rel = path.relative_to(package_dir.parent).as_posix()
        seen: set[tuple[int, str]] = set()
        for lineno, target in executable_imports(path, package_dir):
            if not (target == package or target.startswith(package + ".")):
                continue
            if (lineno, target) in seen:
                continue
            seen.add((lineno, target))
            imports.append(Import(rel, lineno, source, target))
    return tuple(imports)


# --- contracts ---------------------------------------------------------------


def _rank() -> dict[str, int]:
    return {unit: index for index, layer in enumerate(LAYERS) for unit in layer.units}


def layer_of(unit: str) -> str:
    if unit == FACADE:
        return "facade"
    for layer in LAYERS:
        if unit in layer.units:
            return layer.name
    return "unassigned"


def allowed_units(unit: str) -> list[str]:
    """Units ``unit`` may import (besides itself and the facade metadata)."""
    rank = _rank()
    if unit == FACADE:
        return [u for layer in LAYERS for u in layer.units]
    if unit not in rank:
        return []
    return [u for layer in LAYERS[: rank[unit]] for u in layer.units]


def may_import(source_unit: str, target_unit: str) -> bool:
    if source_unit == target_unit or target_unit == FACADE or source_unit == FACADE:
        return True
    rank = _rank()
    if source_unit not in rank or target_unit not in rank:
        return False
    return rank[target_unit] < rank[source_unit]


def allowlist_entry(
    path: str,
    target: str,
    allowlist: dict[tuple[str, str], tuple[str, str, str]] | None = None,
) -> tuple[str, str] | None:
    """The ``LAYERING_ALLOWLIST`` key excusing ``path`` importing ``target``."""
    table = LAYERING_ALLOWLIST if allowlist is None else allowlist
    for key in table:
        entry_path, prefix = key
        if path == entry_path and (target == prefix or target.startswith(prefix + ".")):
            return key
    return None


@dataclass(frozen=True)
class ContractReport:
    """Result of checking every import against the layer contracts."""

    violations: list[Violation]
    used_allowlist: set[tuple[str, str]]
    bridged: list[Import]  # upward imports from DEPRECATION_SHIMS
    excused: list[Import]  # upward imports excused by LAYERING_ALLOWLIST


@lru_cache(maxsize=8)
def check_contracts(package_dir: Path = PACKAGE_DIR) -> ContractReport:
    """Check every executable import in ``package_dir`` against ``LAYERS``."""
    package = package_dir.name
    shims = {package + rel[len(PACKAGE) :] for rel in DEPRECATION_SHIMS}
    violations: list[Violation] = []
    used: set[tuple[str, str]] = set()
    bridged: list[Import] = []
    excused: list[Import] = []
    for item in collect_imports(package_dir):
        src = unit_of(item.source, package)
        dst = unit_of(item.target, package)
        if src is None or dst is None or may_import(src, dst):
            continue
        if item.path in shims:
            bridged.append(item)
            continue
        key = allowlist_entry(package + item.path[len(package) :], item.target)
        if key is None and package != PACKAGE:
            key = allowlist_entry(PACKAGE + item.path[len(package) :], item.target)
        if key is not None:
            used.add(key)
            excused.append(item)
            continue
        violations.append(Violation(item, src, dst))
    return ContractReport(violations, used, bridged, excused)


# --- graph -------------------------------------------------------------------


def unit_edges(
    imports: Iterable[Import], package: str = PACKAGE
) -> dict[tuple[str, str], list[Import]]:
    """Cross-unit edges, each with the imports that create it."""
    edges: dict[tuple[str, str], list[Import]] = defaultdict(list)
    for item in imports:
        src = unit_of(item.source, package)
        dst = unit_of(item.target, package)
        if src is None or dst is None or src == dst:
            continue
        edges[(src, dst)].append(item)
    return dict(edges)


def strongly_connected_components(
    nodes: Iterable[str], edges: Iterable[tuple[str, str]]
) -> list[list[str]]:
    """Tarjan's SCCs with more than one node, each sorted, sorted by size."""
    graph: dict[str, set[str]] = defaultdict(set)
    all_nodes = set(nodes)
    for src, dst in edges:
        graph[src].add(dst)
        all_nodes.update((src, dst))
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    result: list[list[str]] = []
    counter = 0

    for root in sorted(all_nodes):
        if root in index:
            continue
        # Iterative Tarjan: recursion limits must never decide whether CI passes.
        work: list[tuple[str, Iterator[str]]] = [(root, iter(sorted(graph[root])))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            current, children = work[-1]
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(graph[child]))))
                    break
                if child in on_stack:
                    low[current] = min(low[current], index[child])
            else:
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[current])
                if low[current] == index[current]:
                    component: list[str] = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.append(member)
                        if member == current:
                            break
                    if len(component) > 1:
                        result.append(sorted(component))
    return sorted(result, key=lambda comp: (-len(comp), comp))


def mutual_pairs(edges: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Unordered pairs of units that import each other directly."""
    edge_set = set(edges)
    pairs = {(min(a, b), max(a, b)) for a, b in edge_set if (b, a) in edge_set}
    return sorted(pairs)


def elementary_cycles(
    edges: Iterable[tuple[str, str]], limit: int = 10_000
) -> list[list[str]]:
    """Every elementary cycle, each listed once from its smallest unit."""
    graph: dict[str, set[str]] = defaultdict(set)
    for src, dst in edges:
        graph[src].add(dst)
    nodes = sorted(set(graph) | {d for targets in graph.values() for d in targets})
    cycles: list[list[str]] = []

    def dfs(start: str, node: str, path: list[str], blocked: set[str]) -> None:
        for nxt in sorted(graph[node]):
            if len(cycles) >= limit:
                return
            if nxt == start:
                cycles.append(list(path))
            elif nxt > start and nxt not in blocked:
                path.append(nxt)
                blocked.add(nxt)
                dfs(start, nxt, path, blocked)
                blocked.discard(nxt)
                path.pop()

    for start in nodes:
        dfs(start, start, [start], {start})
    return sorted(cycles, key=lambda cycle: (len(cycle), cycle))


def all_units(package_dir: Path = PACKAGE_DIR) -> list[str]:
    return sorted({unit_of_path(p, package_dir) for p in package_files(package_dir)})


@dataclass(frozen=True)
class Graph:
    """The unit-level graph, split into enforced and sanctioned edges."""

    units: list[str]
    edges: dict[tuple[str, str], list[Import]]
    sanctioned: dict[tuple[str, str], list[Import]]  # bridges + debt only

    @property
    def enforced(self) -> dict[tuple[str, str], list[Import]]:
        """Edges with at least one import that is not a bridge or debt."""
        result: dict[tuple[str, str], list[Import]] = {}
        for key, items in self.edges.items():
            rest = [i for i in items if i not in set(self.sanctioned.get(key, []))]
            if rest:
                result[key] = rest
        return result


@lru_cache(maxsize=8)
def build_graph(package_dir: Path = PACKAGE_DIR) -> Graph:
    imports = collect_imports(package_dir)
    edges = unit_edges(imports, package_dir.name)
    report = check_contracts(package_dir)
    sanctioned = unit_edges(report.bridged + report.excused, package_dir.name)
    return Graph(all_units(package_dir), edges, sanctioned)


# --- documentation -----------------------------------------------------------

BEGIN = "<!-- layer-graph:begin -->"


def _cell(text: str) -> str:
    """Escape a Markdown table cell."""
    return text.replace("|", "\\|")


END = "<!-- layer-graph:end -->"


def render_section(package_dir: Path = PACKAGE_DIR) -> str:
    """The generated part of the Layering section of docs/ARCHITECTURE.md."""
    graph = build_graph(package_dir)
    lines = [
        BEGIN,
        "<!-- Generated by `python scripts/layer_graph.py --write "
        "docs/ARCHITECTURE.md`; tests/unit/test_layering.py fails when it is "
        "stale. -->",
        "",
        "| # | Layer | Units | Contains |",
        "|---|-------|-------|----------|",
    ]
    for index, layer in reversed(list(enumerate(LAYERS))):
        units = ", ".join(f"`{u}`" for u in layer.units)
        lines.append(f"| {index} | {layer.name} | {units} | {layer.role} |")
    lines += [
        "",
        "Each unit imports only itself and units in lower layers; siblings in "
        "one layer never import each other. Every unit may read the facade's "
        "own metadata (`stateset_agents.__version__`); a lazy facade name "
        "counts as an import of the module that defines it.",
        "",
        "**Contract per unit** (the facade's own metadata, "
        "`stateset_agents.__version__`, is readable from every unit):",
        "",
        "| Unit | Layer | May import | Must not import |",
        "|------|-------|------------|-----------------|",
        *render_contracts(),
        "",
        "**Layer DAG** (who imports whom, transitively reduced; arrows point "
        "at the layer imported):",
        "",
        *render_layer_dag(graph),
        "",
        "**Unit dependencies** (every executable import, including "
        "function-level and literal `importlib` imports):",
        "",
        "| Unit | Imports |",
        "|------|---------|",
    ]
    by_source: dict[str, list[str]] = defaultdict(list)
    for (src, dst), items in sorted(graph.edges.items()):
        sanctioned = len(graph.sanctioned.get((src, dst), []))
        mark = "" if sanctioned < len(items) else " (bridge/debt only)"
        if 0 < sanctioned < len(items):
            mark = " (+ bridge/debt)"
        by_source[src].append(f"`{dst}`{mark}")
    order = [u for layer in reversed(LAYERS) for u in layer.units] + [FACADE]
    for unit in order:
        if unit in by_source:
            lines.append(f"| `{unit}` | {', '.join(by_source[unit])} |")
    cycles = strongly_connected_components(graph.units, graph.enforced)
    sanctioned_cycles = strongly_connected_components(graph.units, graph.edges)
    lines += [
        "",
        f"**Cycles:** {len(cycles)} among enforced edges"
        + (
            "; the only cycles left run through dated deprecation bridges or "
            "debt: " + "; ".join(" <-> ".join(c) for c in sanctioned_cycles)
            if sanctioned_cycles
            else ""
        )
        + ".",
        "",
        "**Sanctioned upward edges** (each dated; CI fails at the release):",
        "",
        "| Importing module | Imports | Kind | Removed in | Reason |",
        "|------------------|---------|------|------------|--------|",
    ]
    for (path, prefix), (kind, reason, removal) in sorted(LAYERING_ALLOWLIST.items()):
        short = prefix.removeprefix(PACKAGE + ".")
        lines.append(
            f"| `{path.removeprefix(PACKAGE + '/')}` | `{short}` | {kind} | "
            f"v{removal} | {_cell(reason)} |"
        )
    shim_rows = sorted(DEPRECATION_SHIMS)
    lines.append(
        "| "
        + ", ".join(f"`{p.removeprefix(PACKAGE + '/')}`" for p in shim_rows)
        + " | the module they alias | shim | as stated in each shim | pure "
        "deprecation shims at old import paths |"
    )
    lines.append(END)
    return "\n".join(lines) + "\n"


def render_contracts() -> list[str]:
    """One Markdown row per unit: what it may and must not import."""
    every = [u for layer in LAYERS for u in layer.units]
    rows = []
    for index, layer in reversed(list(enumerate(LAYERS))):
        for unit in layer.units:
            allowed = allowed_units(unit)
            forbidden = [u for u in every if u != unit and u not in allowed]
            rows.append(
                f"| `{unit}` | {index} {layer.name} | "
                + (", ".join(f"`{u}`" for u in allowed) or "nothing")
                + " | "
                + (", ".join(f"`{u}`" for u in forbidden) or "nothing")
                + " |"
            )
    return rows


def _layer_index() -> dict[str, int]:
    return {unit: index for index, layer in enumerate(LAYERS) for unit in layer.units}


def layer_edges(graph: Graph) -> set[tuple[int, int]]:
    """Layer-level edges implied by the enforced unit edges."""
    index = _layer_index()
    return {
        (index[src], index[dst])
        for src, dst in graph.enforced
        if src in index and dst in index and index[src] != index[dst]
    }


def transitive_reduction(edges: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Drop every edge implied by a longer path (the input must be a DAG)."""
    successors: dict[int, set[int]] = defaultdict(set)
    for src, dst in edges:
        successors[src].add(dst)

    def reachable(start: int) -> set[int]:
        seen: set[int] = set()
        stack = list(successors[start])
        while stack:
            node = stack.pop()
            if node not in seen:
                seen.add(node)
                stack.extend(successors[node])
        return seen

    return {
        (src, dst)
        for src, dst in edges
        if not any(dst in reachable(mid) for mid in successors[src] if mid != dst)
    }


def render_layer_dag(graph: Graph) -> list[str]:
    """A Mermaid flowchart of the layers, top layer first."""
    lines = ["```mermaid", "flowchart TD"]
    for index, layer in reversed(list(enumerate(LAYERS))):
        units = ", ".join(layer.units)
        lines.append(f'    L{index}["{index} {layer.name}<br/>{units}"]')
    for src, dst in sorted(transitive_reduction(layer_edges(graph)), reverse=True):
        lines.append(f"    L{src} --> L{dst}")
    lines.append("```")
    return lines


# --- package layout ----------------------------------------------------------

LAYOUT_BEGIN = "<!-- package-layout:begin -->"
LAYOUT_END = "<!-- package-layout:end -->"

#: Largest relative drift between documented and actual line counts.
LINE_TOLERANCE = 0.10


@dataclass(frozen=True)
class UnitSize:
    """What the package layout table states about one unit."""

    unit: str
    layer: str
    paths: str
    modules: int
    lines: int


def _unit_paths(unit: str, files: list[str]) -> str:
    """Compact path description for a unit's files (relative to the package)."""
    dirs = sorted({f.split("/")[0] for f in files if "/" in f})
    if unit == FACADE:
        return "`__init__.py`"
    if unit == "errors":
        return ", ".join(f"`{f}`" for f in sorted(files))
    if len(dirs) == 1 and all(f.startswith(dirs[0] + "/") for f in files):
        return f"`{dirs[0]}/`"
    patterns = []
    for family, family_unit in TOP_LEVEL_FAMILIES:
        if family_unit == unit and any(f.startswith(family) for f in files):
            patterns.append(f"`{family}*.py`")
    named = [
        f"`{f}`"
        for f in sorted(files)
        if "/" not in f
        and not any(f.startswith(fam) for fam, u in TOP_LEVEL_FAMILIES if u == unit)
    ]
    return ", ".join(patterns + named + [f"`{d}/`" for d in dirs])


def unit_sizes(package_dir: Path = PACKAGE_DIR) -> list[UnitSize]:
    files: dict[str, list[str]] = defaultdict(list)
    lines: dict[str, int] = defaultdict(int)
    for path in package_files(package_dir):
        unit = unit_of_path(path, package_dir)
        files[unit].append(path.relative_to(package_dir).as_posix())
        with path.open("rb") as stream:
            lines[unit] += sum(1 for _ in stream)
    order = [u for layer in reversed(LAYERS) for u in layer.units] + [FACADE]
    return [
        UnitSize(
            unit,
            layer_of(unit),
            _unit_paths(unit, files[unit]),
            len(files[unit]),
            lines[unit],
        )
        for unit in order
        if unit in files
    ]


def _kloc(lines: int) -> str:
    return f"~{lines / 1000:.1f}K" if lines < 10_000 else f"~{round(lines / 1000)}K"


def render_layout(package_dir: Path = PACKAGE_DIR) -> str:
    sizes = unit_sizes(package_dir)
    rows = [
        LAYOUT_BEGIN,
        "<!-- Generated by `python scripts/layer_graph.py --write "
        "docs/ARCHITECTURE.md`; tests/unit/test_layering.py fails when a unit, "
        "path or module count changes or a line count drifts > 10%. -->",
        "",
        "| Unit | Layer | Path (under `stateset_agents/`) | Modules | Lines |",
        "|------|-------|-------------------------------|--------:|------:|",
    ]
    for size in sizes:
        rows.append(
            f"| `{size.unit}` | {size.layer} | {size.paths} | {size.modules} | "
            f"{_kloc(size.lines)} |"
        )
    total_modules = sum(s.modules for s in sizes)
    total_lines = sum(s.lines for s in sizes)
    rows.append(f"| **total** | | | {total_modules} | {_kloc(total_lines)} |")
    rows.append(LAYOUT_END)
    return "\n".join(rows) + "\n"


def _between(text: str, begin: str, end: str) -> str | None:
    match = re.search(re.escape(begin) + r".*?" + re.escape(end), text, re.S)
    return match.group(0) + "\n" if match else None


def documented_layout(doc: Path) -> str | None:
    return _between(doc.read_text(encoding="utf-8"), LAYOUT_BEGIN, LAYOUT_END)


_ROW = re.compile(r"^\| (?P<unit>[^|]+) \|(?P<rest>.*)\| (?P<lines>[^|]+) \|$")


def _parse_kloc(text: str) -> float:
    return float(text.strip().lstrip("~").rstrip("K")) * 1000


def layout_drift(documented: str | None, package_dir: Path = PACKAGE_DIR) -> list[str]:
    """Problems between a documented layout table and the tree."""
    if documented is None:
        return [f"no {LAYOUT_BEGIN} ... {LAYOUT_END} section"]
    actual = render_layout(package_dir)

    def rows(text: str) -> list[tuple[str, str, float]]:
        found = []
        for line in text.splitlines():
            match = _ROW.match(line)
            if match and match.group("unit").strip() not in {"Unit", "------"}:
                if set(match.group("lines").strip()) <= set("-:"):
                    continue
                found.append(
                    (
                        match.group("unit").strip(),
                        match.group("rest"),
                        _parse_kloc(match.group("lines")),
                    )
                )
        return found

    problems = []
    doc_rows, tree_rows = rows(documented), rows(actual)
    if [(u, r) for u, r, _ in doc_rows] != [(u, r) for u, r, _ in tree_rows]:
        problems.append("units, layers, paths or module counts changed")
    if len(doc_rows) != len(tree_rows):
        return problems
    for (unit, _, doc_lines), (_, _, tree_lines) in zip(
        doc_rows, tree_rows, strict=True
    ):
        if abs(doc_lines - tree_lines) > max(LINE_TOLERANCE * tree_lines, 1000):
            problems.append(
                f"{unit}: documents {doc_lines:.0f} lines, tree has ~{tree_lines:.0f}"
            )
    return problems


def _replace_between(text: str, begin: str, end: str, section: str) -> str:
    if begin not in text or end not in text:
        raise SystemExit(f"document has no {begin} ... {end} markers")
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    return head + section.rstrip("\n") + tail


def write_section(doc: Path, package_dir: Path = PACKAGE_DIR) -> None:
    """Regenerate the layering section and the package layout in ``doc``."""
    text = doc.read_text(encoding="utf-8")
    text = _replace_between(text, BEGIN, END, render_section(package_dir))
    text = _replace_between(text, LAYOUT_BEGIN, LAYOUT_END, render_layout(package_dir))
    doc.write_text(text, encoding="utf-8")


def documented_section(doc: Path) -> str | None:
    return _between(doc.read_text(encoding="utf-8"), BEGIN, END)


# --- CLI ---------------------------------------------------------------------


def _summary(package_dir: Path) -> dict[str, object]:
    graph = build_graph(package_dir)
    return {
        "units": graph.units,
        "edges": {f"{s} -> {d}": len(v) for (s, d), v in sorted(graph.edges.items())},
        "edge_count": len(graph.edges),
        "import_count": sum(len(v) for v in graph.edges.values()),
        "mutual_pairs": [list(p) for p in mutual_pairs(graph.edges)],
        "sccs": strongly_connected_components(graph.units, graph.edges),
        "enforced_sccs": strongly_connected_components(graph.units, graph.enforced),
        "cycles": elementary_cycles(graph.edges),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="print the graph as JSON")
    parser.add_argument("--edges", action="store_true", help="every cross-unit import")
    parser.add_argument("--unit", help="only edges into or out of this unit")
    parser.add_argument(
        "--violations", action="store_true", help="list contract breaks; exit 1 if any"
    )
    parser.add_argument("--write", type=Path, help="regenerate the doc section")
    parser.add_argument("--check", type=Path, help="exit 1 if the doc or code is stale")
    parser.add_argument(
        "--package-dir", type=Path, default=PACKAGE_DIR, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    package_dir: Path = args.package_dir

    if args.write:
        write_section(args.write, package_dir)
        print(f"wrote the layering section and package layout to {args.write}")
        return 0
    if args.check or args.violations:
        report = check_contracts(package_dir)
        status = 0
        for violation in report.violations:
            print("layer contract broken: " + violation.explain())
            status = 1
        if args.check:
            stale = documented_section(args.check) != render_section(package_dir)
            drift = layout_drift(documented_layout(args.check), package_dir)
            if stale or drift:
                print(
                    f"{args.check} layering section or package layout is stale "
                    f"({'; '.join(drift) or 'layer graph changed'}); regenerate "
                    f"with `python scripts/layer_graph.py --write {args.check}`"
                )
                status = 1
        return status
    if args.json:
        print(json.dumps(_summary(package_dir), indent=2))
        return 0

    graph = build_graph(package_dir)
    edges = graph.edges
    if args.unit:
        edges = {k: v for k, v in edges.items() if args.unit in k}
    if args.edges:
        for (src, dst), items in sorted(edges.items()):
            print(f"{src} -> {dst} ({len(items)})")
            for item in items:
                print(f"    {item.path}:{item.lineno} -> {item.target}")
        return 0
    print(
        f"{len(edges)} cross-unit edges, "
        f"{sum(len(v) for v in edges.values())} imports"
    )
    for (src, dst), items in sorted(edges.items()):
        sanctioned = len(graph.sanctioned.get((src, dst), []))
        note = f"  ({sanctioned} bridge/debt)" if sanctioned else ""
        print(f"  {src:<14} -> {dst:<14} {len(items):>4}{note}")
    pairs = mutual_pairs(edges)
    print(f"\n{len(pairs)} mutual pairs")
    for a, b in pairs:
        print(f"  {a} <-> {b}  ({len(edges[(a, b)])} / {len(edges[(b, a)])})")
    for label, edge_set in (("all", graph.edges), ("enforced", graph.enforced)):
        sccs = strongly_connected_components(graph.units, edge_set)
        print(f"\n{len(sccs)} strongly connected components ({label} edges)")
        for comp in sccs:
            print("  " + ", ".join(comp))
    cycles = elementary_cycles(graph.edges)
    print(f"\n{len(cycles)} elementary cycles (all edges)")
    for cycle in cycles[:60]:
        print("  " + " -> ".join(cycle + [cycle[0]]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
