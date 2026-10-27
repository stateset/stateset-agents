"""Every exception class the package defines derives from ``StateSetError``.

``core/errors.py`` promises one hierarchy: callers catch ``StateSetError`` (or
a category under it) for any framework failure, and a class that also keeps
its old builtin base (``ValueError``, ``RuntimeError``, ...) leaves existing
``except`` clauses working. This test walks every ``class`` statement under
``stateset_agents/`` statically -- no module is imported -- resolves its bases
through each module's imports, and fails on an exception class outside the
hierarchy that is not in ``OUTSIDE_HIERARCHY`` with a reason and a removal
release. The allowlist only shrinks: stale entries fail too.
"""

from __future__ import annotations

import ast
import builtins
import importlib
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "stateset_agents"
PACKAGE = "stateset_agents"
ROOT_ERROR = f"{PACKAGE}.core.errors.StateSetError"

_STRING_CODE = (
    "carries a string diagnostic ``code`` that the fine-tuning studio, TUI and "
    "API read, while StateSetError.code is an ErrorCode; re-base once the "
    "diagnostic moves to its own attribute"
)

#: ``module.Class`` -> (reason, removal release). Exception classes allowed
#: outside ``StateSetError`` until the stated release; the build fails once it
#: is due. Only shrinks.
OUTSIDE_HIERARCHY: dict[str, tuple[str, str]] = {
    f"{PACKAGE}.finetune_evaluation.EvaluationProtocolError": (_STRING_CODE, "0.71.0"),
    f"{PACKAGE}.finetune_validation.ValidationComparisonError": (
        _STRING_CODE,
        "0.71.0",
    ),
    f"{PACKAGE}.finetune_workspace.FinetuneFailure": (_STRING_CODE, "0.71.0"),
}


@dataclass
class ClassInfo:
    qualname: str
    path: str
    lineno: int
    bases: list[str] = field(default_factory=list)


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_from(node: ast.ImportFrom, module: str, is_package: bool) -> str:
    if node.level == 0:
        return node.module or ""
    package = module.split(".") if is_package else module.split(".")[:-1]
    base = package[: len(package) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _dotted(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        root = _dotted(expr.value)
        return f"{root}.{expr.attr}" if root else None
    if isinstance(expr, ast.Subscript):
        return _dotted(expr.value)
    return None


@cache
def _scan() -> tuple[dict[str, ClassInfo], dict[str, dict[str, str]]]:
    """Every class in the package, and each module's name -> qualified map."""
    classes: dict[str, ClassInfo] = {}
    namespaces: dict[str, dict[str, str]] = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        module = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        names[alias.asname] = alias.name
                    else:
                        root = alias.name.split(".")[0]
                        names[root] = root
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_from(node, module, path.name == "__init__.py")
                for alias in node.names:
                    names[alias.asname or alias.name] = f"{base}.{alias.name}"
        pending: list[tuple[ast.AST, str]] = [(tree, module)]
        while pending:
            parent, prefix = pending.pop()
            for node in ast.iter_child_nodes(parent):
                if isinstance(node, ast.ClassDef):
                    qualname = f"{prefix}.{node.name}"
                    if prefix == module:
                        names[node.name] = qualname
                    classes[qualname] = ClassInfo(
                        qualname,
                        path.relative_to(ROOT).as_posix(),
                        node.lineno,
                        [b for b in (_dotted(base) for base in node.bases) if b],
                    )
                    pending.append((node, qualname))
                elif isinstance(
                    node, ast.FunctionDef | ast.AsyncFunctionDef | ast.If | ast.Try
                ):
                    pending.append((node, prefix))
        namespaces[module] = names
    return classes, namespaces


def _qualify(name: str, module: str) -> str:
    """Resolve a base expression written in ``module`` to a qualified name."""
    classes, namespaces = _scan()
    names = namespaces.get(module, {})
    head, _, rest = name.partition(".")
    if head in names:
        target = names[head] + (f".{rest}" if rest else "")
    elif hasattr(builtins, head) and not rest:
        return f"builtins.{head}"
    else:
        target = name
    return _follow(target)


def _follow(target: str, seen: frozenset[str] = frozenset()) -> str:
    """Chase re-exports (``from .errors import X`` in a package) to the class."""
    classes, namespaces = _scan()
    if target in classes or target in seen:
        return target
    module, _, attr = target.rpartition(".")
    names = namespaces.get(module)
    if names and attr in names and names[attr] != target:
        return _follow(names[attr], seen | {target})
    return target


@cache
def _external_is_exception(qualified: str) -> bool:
    module, _, attr = qualified.rpartition(".")
    if module == "builtins":
        value = getattr(builtins, attr, None)
    else:
        try:
            value = getattr(importlib.import_module(module), attr, None)
        except Exception:  # noqa: BLE001 - an optional dependency is absent
            return False
    return isinstance(value, type) and issubclass(value, BaseException)


@cache
def _ancestry(qualname: str) -> tuple[bool, bool]:
    """``(is an exception, derives from StateSetError)`` for a package class."""
    if qualname == ROOT_ERROR:
        return True, True
    classes, _ = _scan()
    info = classes[qualname]
    module = info.qualname.rsplit(".", 1)[0]
    while module not in _scan()[1] and "." in module:
        module = module.rsplit(".", 1)[0]
    is_exception = in_hierarchy = False
    for base in info.bases:
        target = _qualify(base, module)
        if target in classes and target != qualname:
            exc, root = _ancestry(target)
        else:
            exc, root = _external_is_exception(target), False
        is_exception |= exc
        in_hierarchy |= root
    return is_exception, in_hierarchy


@cache
def _package_text() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in PACKAGE_DIR.rglob("*.py")
        if "__pycache__" not in path.parts
    }


def is_private_sentinel(info: ClassInfo) -> bool:
    """A ``_Private`` exception no other module mentions is local control flow
    (raised and caught in its own module), not part of the error surface."""
    name = info.qualname.rsplit(".", 1)[-1]
    if not name.startswith("_"):
        return False
    return not any(
        name in text for path, text in _package_text().items() if path != info.path
    )


def outside_hierarchy() -> dict[str, ClassInfo]:
    """Exception classes outside ``StateSetError`` (private sentinels excepted)."""
    classes, _ = _scan()
    return {
        name: info
        for name, info in sorted(classes.items())
        if _ancestry(name) == (True, False) and not is_private_sentinel(info)
    }


def exception_classes() -> list[str]:
    classes, _ = _scan()
    return [name for name in sorted(classes) if _ancestry(name)[0]]


def test_private_sentinels_stay_private() -> None:
    """The exemption covers only module-local control-flow exceptions."""
    classes, _ = _scan()
    sentinels = sorted(
        name
        for name, info in classes.items()
        if _ancestry(name) == (True, False) and is_private_sentinel(info)
    )
    assert all(name.rsplit(".", 1)[-1].startswith("_") for name in sentinels)
    assert f"{PACKAGE}.data.text_sft._InvalidJson" in sentinels


def test_the_walk_sees_the_hierarchy() -> None:
    found = set(exception_classes())
    assert ROOT_ERROR in found
    assert f"{PACKAGE}.core.errors.ConfigurationError" in found
    assert f"{PACKAGE}.core.reward_context.RewardUnavailableError" in found
    assert _ancestry(f"{PACKAGE}.core.errors.ConfigurationError") == (True, True)
    assert len(found) > 50


def test_every_exception_class_derives_from_stateset_error() -> None:
    offenders = [
        f"{info.path}:{info.lineno} {name} (bases: {', '.join(info.bases)})"
        for name, info in outside_hierarchy().items()
        if name not in OUTSIDE_HIERARCHY
    ]
    assert offenders == [], (
        "exception classes must derive from stateset_agents.core.errors."
        "StateSetError (keep the builtin base too, e.g. "
        "`class X(StateSetError, ValueError)` with `message_only_str = True`), "
        "or be listed in OUTSIDE_HIERARCHY with a reason and removal date:\n  "
        + "\n  ".join(offenders)
    )


def test_outside_hierarchy_allowlist_is_current_and_dated() -> None:
    from tests.unit.test_deprecation_schedule import _package_version, _version_tuple

    stale = sorted(set(OUTSIDE_HIERARCHY) - set(outside_hierarchy()))
    assert stale == [], f"re-based; remove from OUTSIDE_HIERARCHY: {stale}"
    for name, (reason, removal) in OUTSIDE_HIERARCHY.items():
        assert reason.strip(), name
        assert (
            _version_tuple(removal) > _package_version()
        ), f"{name} was allowed outside StateSetError until v{removal}"


@pytest.mark.parametrize(
    "source,expected",
    [
        ("class A(Exception):\n    pass\n", (True, False)),
        ("class B(dict):\n    pass\n", (False, False)),
    ],
)
def test_walk_classifies_builtin_bases(
    source: str, expected: tuple[bool, bool]
) -> None:
    tree = ast.parse(source)
    (node,) = tree.body
    assert isinstance(node, ast.ClassDef)
    base = _dotted(node.bases[0])
    assert base is not None
    assert (_external_is_exception(f"builtins.{base}"), False) == expected
