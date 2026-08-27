"""Guard: meta-tests must compare POSIX paths, on every platform.

The AST-walking meta-tests build repository-relative path strings and compare
them against committed POSIX literals (e.g. ``torch_import_allowlist.txt``).
``str(Path(...))`` uses ``\\`` on Windows, so a naive ``str(relative_to(...))``
silently fails there and nowhere else. :func:`tests.unit._paths.rel_posix` is
the single normalisation point; these tests keep it that way.
"""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath

import pytest

from tests.unit._paths import rel_posix

UNIT_DIR = Path(__file__).parent

#: Meta-tests that turn a filesystem path into a comparable/printable string.
PATH_BUILDING_META_TESTS = (
    "test_torch_import_policy.py",
    "test_layering.py",
    "test_checkpoint_trust.py",
    "test_trusted_keyword_only.py",
    "test_data_package_lazy_exports.py",
    "test_training_lazy_exports.py",
    "test_docs_version_freshness.py",
    "forwarder_asserts.py",
)

#: ``str(x.relative_to(y))`` / ``f"{x.relative_to(y)}"`` — the Windows bug.
_UNNORMALISED = re.compile(r"(?:str\(|\{)\s*\w+\.relative_to\([^)]*\)\s*[)}]")


def test_rel_posix_normalises_windows_separators() -> None:
    """A Windows path must come out with forward slashes."""
    rel = rel_posix(
        PureWindowsPath(r"D:\a\repo\stateset_agents\training\dapo_trainer.py"),
        PureWindowsPath(r"D:\a\repo"),
    )
    assert rel == "stateset_agents/training/dapo_trainer.py"
    assert "\\" not in rel


def test_rel_posix_is_identity_on_posix_input() -> None:
    assert rel_posix(Path("/repo/pkg/mod.py"), Path("/repo")) == "pkg/mod.py"


@pytest.mark.parametrize("name", PATH_BUILDING_META_TESTS)
def test_meta_tests_do_not_stringify_relative_paths(name: str) -> None:
    """No meta-test may render a relative path with the native separator."""
    source = (UNIT_DIR / name).read_text(encoding="utf-8")
    offenders = [m.group(0) for m in _UNNORMALISED.finditer(source)]
    assert not offenders, (
        f"{name} builds a native-separator path string {offenders}; "
        "use tests.unit._paths.rel_posix(path, root) instead"
    )


def test_torch_allowlist_is_posix() -> None:
    """The committed allowlist itself must stay POSIX-only."""
    from tests.unit.test_torch_import_policy import _load_allowlist, _offenders

    assert all("\\" not in entry for entry in _load_allowlist())
    assert all("\\" not in rel for rel in _offenders())
