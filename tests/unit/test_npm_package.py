"""Release-contract checks for the dependency-free Node API client."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "npm" / "package.json"


def _python_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_npm_package_is_public_scoped_and_version_aligned() -> None:
    package = json.loads(PACKAGE.read_text(encoding="utf-8"))

    assert package["name"] == "@stateset/agents"
    assert package["version"] == _python_version()
    assert package["publishConfig"]["access"] == "public"
    assert package["publishConfig"]["provenance"] is True
    assert package["engines"]["node"] == ">=18"


def test_npm_package_is_dependency_free_and_exports_types() -> None:
    package = json.loads(PACKAGE.read_text(encoding="utf-8"))

    assert "dependencies" not in package
    assert package["exports"]["."]["import"] == "./src/index.js"
    assert package["exports"]["."]["types"] == "./src/index.d.ts"
    assert (ROOT / "npm" / "src" / "index.js").is_file()
    assert (ROOT / "npm" / "src" / "index.d.ts").is_file()


def test_npm_tarball_carries_complete_project_license() -> None:
    assert (ROOT / "npm" / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
