"""Shared path normalisation for the AST-walking meta-tests.

Several meta-tests compare a repository-relative path against a committed
list (``torch_import_allowlist.txt``) or embed one in an assertion message.
``str(Path(...))`` renders ``\\`` separators on Windows, so a naive
``str(path.relative_to(root))`` never matches a POSIX-style literal and the
tests fail on Windows only. Every such path must go through
:func:`rel_posix`.
"""

from __future__ import annotations

from pathlib import PurePath

__all__ = ["rel_posix"]


def rel_posix(path: PurePath, root: PurePath) -> str:
    """Return ``path`` relative to ``root`` with forward slashes, always.

    >>> from pathlib import PureWindowsPath
    >>> rel_posix(
    ...     PureWindowsPath(r"D:\\repo\\stateset_agents\\training\\ema.py"),
    ...     PureWindowsPath(r"D:\\repo"),
    ... )
    'stateset_agents/training/ema.py'
    """
    return path.relative_to(root).as_posix()
