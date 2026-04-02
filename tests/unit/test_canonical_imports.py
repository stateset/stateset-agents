"""Guard against new imports from deprecated top-level shim modules."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = ("stateset_agents", "examples", "scripts")
DEPRECATED_SHIMS = {"api", "core", "training", "rewards", "environments"}


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _find_deprecated_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_root = alias.name.split(".", 1)[0]
                if imported_root in DEPRECATED_SHIMS:
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno} imports {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                continue

            imported_root = node.module.split(".", 1)[0]
            if imported_root in DEPRECATED_SHIMS:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno} imports from {node.module}"
                )

    return violations


def test_canonical_package_imports_do_not_use_deprecated_shims() -> None:
    violations: list[str] = []

    for root_name in SCAN_ROOTS:
        root = REPO_ROOT / root_name
        for path in _iter_python_files(root):
            violations.extend(_find_deprecated_imports(path))

    assert not violations, "Deprecated shim imports found:\n" + "\n".join(violations)
