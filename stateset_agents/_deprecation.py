"""One implementation for deprecation warnings.

Every deprecation names the release that removes it, and
``tests/unit/test_deprecation_schedule.py`` fails CI once the package version
reaches that release. Pass ``removal`` as a string literal (``removal="X.Y.Z"``)
so the guard can read it from the source.

Policy (docs/API_COMPATIBILITY.md): a name in ``stateset_agents.__all__`` or
``stateset_agents.api.__all__`` stays available for at least one minor
release after the release that deprecates it; provisional names may go after
the next minor release.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any

__all__ = [
    "deprecated_alias_getattr",
    "deprecation_message",
    "warn_deprecated",
]


def deprecation_message(
    what: str, *, removal: str, replacement: str | None = None
) -> str:
    """Return the canonical deprecation sentence for ``what``."""
    message = f"{what} is deprecated and will be removed in v{removal.lstrip('v')}"
    if replacement:
        message += f"; use {replacement} instead"
    return message + "."


def warn_deprecated(
    what: str,
    *,
    removal: str,
    replacement: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Emit a ``DeprecationWarning`` attributed to the caller's caller.

    ``stacklevel`` has the meaning it has for :func:`warnings.warn` called from
    the function that invokes this helper.
    """
    warnings.warn(
        deprecation_message(what, removal=removal, replacement=replacement),
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )


def deprecated_alias_getattr(
    module_name: str,
    aliases: Mapping[str, tuple[str, str, str]],
) -> Callable[[str], Any]:
    """Build a module ``__getattr__`` that serves deprecated aliases lazily.

    ``aliases`` maps an old attribute name to ``(target_module, target_name,
    removal)``. Each access warns and returns the canonical object, so the old
    name and the new one are the same object (``isinstance`` and ``except``
    clauses keep working).
    """

    def __getattr__(name: str) -> Any:
        target = aliases.get(name)
        if target is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        target_module, target_name, removal = target
        warn_deprecated(
            f"{module_name}.{name}",
            removal=removal,
            replacement=f"{target_module}.{target_name}",
        )
        return getattr(import_module(target_module), target_name)

    return __getattr__
