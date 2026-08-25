"""Meta-tests enforcing the project's torch-import policy.

``torch`` is a heavy optional dependency. Two rules keep it from creeping into
cheap code paths:

1. **Guarded imports** -- no module may import ``torch`` at module level unless
   it is listed in ``torch_import_allowlist.txt`` (genuine trainers/backends
   that cannot function without it). Everything else must guard the import
   (``try``/``except ImportError``), use ``get_torch()``, import inside the
   function that needs it, or import it under ``TYPE_CHECKING``.
2. **Torch-free entry points** -- importing the public packages must not pull
   ``torch`` into ``sys.modules``.

The allowlist is also checked for stale entries, so it can only ever shrink
deliberately.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG = REPO_ROOT / "stateset_agents"
ALLOWLIST_PATH = Path(__file__).with_name("torch_import_allowlist.txt")

#: Public packages that must stay importable without ``torch``.
TORCH_FREE_ENTRY_POINTS = (
    "stateset_agents",
    "stateset_agents.core",
    "stateset_agents.training",
    "stateset_agents.cli",
)


def _load_allowlist() -> list[str]:
    """Return the allowlisted relative paths (``path  # reason`` per line)."""
    entries: list[str] = []
    for raw in ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line.split("#", 1)[0].strip())
    return entries


def _module_level_torch_imports(path: Path) -> list[int]:
    """Line numbers of unguarded module-level torch imports in ``path``.

    Only ``tree.body`` is inspected, so imports nested in ``try``/``except``,
    ``if TYPE_CHECKING:`` or a function body are guarded by construction.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[int] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        if any(n == "torch" or n.startswith("torch.") for n in names):
            hits.append(node.lineno)
    return hits


def _offenders() -> dict[str, list[int]]:
    return {
        str(path.relative_to(PKG.parent)): hits
        for path in sorted(PKG.rglob("*.py"))
        if (hits := _module_level_torch_imports(path))
    }


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def test_allowlist_entries_have_reasons() -> None:
    """Every allowlist line must carry a ``# reason`` comment."""
    missing = [
        raw
        for raw in ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.lstrip().startswith("#") and "#" not in raw
    ]
    assert not missing, f"allowlist entries without a '# reason' comment: {missing}"


def test_no_unguarded_torch_import_outside_allowlist() -> None:
    allow = set(_load_allowlist())
    bad = {rel: hits for rel, hits in _offenders().items() if rel not in allow}
    assert not bad, (
        "unguarded module-level torch imports (wrap in try/except, use "
        "get_torch(), import inside the function, or guard with "
        f"TYPE_CHECKING): {bad}"
    )


def test_allowlist_has_no_stale_entries() -> None:
    """The allowlist may only shrink: every entry must still be needed."""
    offenders = _offenders()
    stale: list[str] = []
    for rel in _load_allowlist():
        if not (PKG.parent / rel).is_file():
            stale.append(f"{rel} (file no longer exists)")
        elif rel not in offenders:
            stale.append(f"{rel} (torch import is now guarded)")
    assert not stale, f"stale entries in {ALLOWLIST_PATH.name}; remove them: {stale}"


@pytest.mark.parametrize("module", TORCH_FREE_ENTRY_POINTS)
def test_entry_points_do_not_import_torch(module: str) -> None:
    if not _installed("torch"):
        pytest.skip("torch is not installed; nothing to guard against")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; import {module}; "
            "sys.exit(1 if 'torch' in sys.modules else 0)",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"importing {module!r} eagerly imported torch "
        f"(stderr: {result.stderr.strip()!r})"
    )
