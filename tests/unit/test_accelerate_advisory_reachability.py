"""Keep CVE-2026-69112 (Accelerate sharded-checkpoint path traversal) unreachable.

``accelerate.utils.load_checkpoint_in_model`` and ``load_checkpoint_and_dispatch``
join ``weight_map`` entries from a checkpoint index onto the checkpoint folder
without validating them, through at least Accelerate 1.15.0. StateSet never
calls them; this test fails if package code starts to, so the release guide's
reachability statement stays true.
"""

from __future__ import annotations

import ast
from pathlib import Path

import stateset_agents

AFFECTED = frozenset({"load_checkpoint_in_model", "load_checkpoint_and_dispatch"})
PACKAGE_ROOT = Path(stateset_agents.__file__).resolve().parent


def _references(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.alias):
            name = node.name.rsplit(".", 1)[-1]
        if name in AFFECTED:
            found.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}")
    return found


def test_package_never_calls_affected_accelerate_loaders() -> None:
    hits = [hit for path in PACKAGE_ROOT.rglob("*.py") for hit in _references(path)]
    assert hits == [], (
        "Accelerate's sharded checkpoint loaders are affected by CVE-2026-69112; "
        "validate weight_map paths before using them and update "
        f"docs/RELEASE_GUIDE.md: {hits}"
    )
