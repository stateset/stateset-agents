"""TraceForge RL — Phase 1 public alias for ``stateset_agents``.

The library is being renamed to TraceForge RL (see
``docs/RENAME_TRACEFORGE.md``). During Phase 1 the canonical implementation
stays in ``stateset_agents`` and this package re-exports the *same* module
objects under the new name, so identity checks (``is``, ``isinstance``)
agree across both spellings and nothing is imported eagerly beyond the
lightweight legacy root package.

Bare ``traceforge`` is deliberately not used: an unrelated distribution
already owns that name, import, and CLI on PyPI.
"""

from __future__ import annotations

import sys
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
from types import ModuleType
from typing import Any

import stateset_agents as _canonical_pkg

__version__ = _canonical_pkg.__version__
__author__ = _canonical_pkg.__author__
__all__ = list(_canonical_pkg.__all__)

_ALIAS = __name__  # "traceforge_rl"
_CANONICAL = "stateset_agents"


class _AliasLoader(Loader):
    """Loader that binds an alias submodule name to the canonical module."""

    def __init__(self, canonical_name: str) -> None:
        self._canonical_name = canonical_name

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        canonical_module = import_module(self._canonical_name)
        sys.modules[module.__name__] = canonical_module
        module.__dict__.clear()
        module.__dict__.update(canonical_module.__dict__)


class _AliasFinder(MetaPathFinder):
    """Map ``traceforge_rl.*`` submodule imports onto ``stateset_agents.*``."""

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if not fullname.startswith(_ALIAS + "."):
            return None
        if fullname == _ALIAS + ".__main__":
            # Real file in this package; let runpy/PathFinder load it so
            # ``python -m traceforge_rl`` executes as ``__main__``.
            return None
        canonical_name = _CANONICAL + fullname[len(_ALIAS) :]
        try:
            canonical_spec = find_spec(canonical_name)
        except (AttributeError, ImportError, ValueError):
            return None
        if canonical_spec is None:
            return None
        spec = ModuleSpec(
            fullname,
            _AliasLoader(canonical_name),
            origin=canonical_spec.origin,
            is_package=canonical_spec.submodule_search_locations is not None,
        )
        spec.submodule_search_locations = canonical_spec.submodule_search_locations
        spec.cached = canonical_spec.cached
        spec.has_location = canonical_spec.has_location
        return spec


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())


def __getattr__(name: str) -> Any:
    value = getattr(_canonical_pkg, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
