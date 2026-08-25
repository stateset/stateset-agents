"""Completeness meta-tests for ``stateset_agents.data``'s lazy export table.

``data/__init__.py`` resolves its public names lazily (PEP 562) so that
importing a light submodule does not drag in ``torch`` /
``sentence_transformers`` through ``conversation_dataset``. The hand-written
``_LAZY_EXPORTS`` table is easy to forget when a submodule grows a new public
name, so these tests keep it in sync with the submodules' ``__all__``.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import stateset_agents.data as data_pkg

DATA_DIR = Path(data_pkg.__file__).resolve().parent


def _submodules() -> list[Path]:
    return sorted(p for p in DATA_DIR.glob("*.py") if p.name != "__init__.py")


def _declared_all(path: Path) -> list[str] | None:
    """Return the literal ``__all__`` of ``path``, or ``None`` if absent."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            if node.value is not None:
                names = ast.literal_eval(node.value)
                return [n for n in names if isinstance(n, str)]
    return None


@pytest.mark.parametrize("submodule", _submodules(), ids=lambda p: p.stem)
def test_submodule_public_names_are_lazily_exported(submodule: Path) -> None:
    """Every name in a submodule's ``__all__`` must resolve on the package."""
    declared = _declared_all(submodule)
    if declared is None:
        pytest.skip(f"{submodule.name} declares no __all__")

    table = data_pkg._LAZY_EXPORTS
    missing = [name for name in declared if name not in table]
    assert not missing, (
        f"{submodule.name} exports {missing} but stateset_agents.data."
        "_LAZY_EXPORTS does not list them"
    )
    mismapped = {
        name: table[name] for name in declared if table[name] != submodule.stem
    }
    assert not mismapped, (
        f"_LAZY_EXPORTS maps these {submodule.name} names to another module: "
        f"{mismapped}"
    )


def test_lazy_exports_all_resolve() -> None:
    """No stale entry: every listed name is importable from the package."""
    unresolved: list[str] = []
    for name in sorted(data_pkg._LAZY_EXPORTS):
        try:
            getattr(data_pkg, name)
        except (AttributeError, ImportError) as exc:  # pragma: no cover
            unresolved.append(f"{name} ({exc})")
    assert not unresolved, f"unresolvable _LAZY_EXPORTS entries: {unresolved}"


def test_dunder_all_matches_lazy_exports() -> None:
    assert data_pkg.__all__ == sorted(data_pkg._LAZY_EXPORTS)


def test_lazy_export_modules_exist() -> None:
    for name, module_name in sorted(data_pkg._LAZY_EXPORTS.items()):
        assert (
            DATA_DIR / f"{module_name}.py"
        ).is_file(), f"_LAZY_EXPORTS[{name!r}] points at missing module {module_name!r}"
        importlib.import_module(f"stateset_agents.data.{module_name}")
