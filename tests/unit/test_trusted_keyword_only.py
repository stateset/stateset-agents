"""Every loader taking ``trusted`` accepts it keyword-only."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "stateset_agents"


def _offenders() -> list[str]:
    bad: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            names = [a.arg for a in node.args.args + node.args.posonlyargs]
            if "trusted" in names:
                rel = path.relative_to(PACKAGE.parent)
                bad.append(f"{rel}:{node.lineno} {node.name}")
    return bad


def test_trusted_is_keyword_only_everywhere() -> None:
    assert _offenders() == []
