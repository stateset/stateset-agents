"""CI guard: every deprecation is dated, and nothing outlives its date.

The API policy (docs/API_COMPATIBILITY.md) lets a deprecated name live for
at least one minor release after the release that deprecates it, and every
deprecation states the release that removes it. This test turns that promise
into a gate:

* every scheduled-removal marker in ``stateset_agents/`` ("... removed in
  vX.Y.Z", ``removal="X.Y.Z"``, ``*_REMOVAL = "vX.Y.Z"``) must name a version
  above the package version, so bumping ``__version__`` to a removal release
  fails CI until the deprecated path is deleted;
* no package source may promise removal "in a future release" (undated);
* every module that emits a ``DeprecationWarning`` must state a removal
  version, either in its own source or in ``EXTERNAL_SCHEDULE`` below.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "stateset_agents"
#: The shared helper formats the dated message from its ``removal`` argument.
HELPER = "stateset_agents/_deprecation.py"

#: "will be removed in v0.71.0", "is removed in 0.71", "and removed in v0.70.0".
#: Past-tense history ("were removed in v0.65.0") is skipped via ``_HISTORICAL``.
_SCHEDULED = re.compile(
    r"(?:\b(\w+)\s+)?removed\s+in\s+v?(\d+\.\d+(?:\.\d+)?)\b", re.IGNORECASE
)
_HISTORICAL = {"was", "were", "been", "already", "got"}
#: ``removal="0.72.0"``, ``removal_version: "v0.72.0"``, ``FOO_REMOVAL = "v0.72.0"``.
_ASSIGNED = re.compile(
    r"\b(?:removal|removal_version|removed_in|\w+_REMOVAL)\s*[:=]\s*"
    r"[\"']v?(\d+\.\d+(?:\.\d+)?)[\"']"
)
_UNDATED = re.compile(
    r"removed\s+in\s+a\s+future|in\s+a\s+future\s+(?:release|version)",
    re.IGNORECASE,
)

#: Deprecations in modules owned by other workstreams whose warning text does
#: not (yet) carry the date. The removal release is recorded here and in the
#: CHANGELOG; drop an entry once the module's own warning states the version.
EXTERNAL_SCHEDULE: dict[str, str] = {
    # ``RulerRewardFunction(config=RulerConfig(...))`` (deprecated since v0.6.0).
    "stateset_agents/rewards/ruler_reward.py": "0.71.0",
    # ``DistributedGRPOTrainer`` (deprecated in v0.38.0, non-functional).
    "stateset_agents/training/distributed_trainer.py": "0.71.0",
}


def _version_tuple(text: str) -> tuple[int, int, int]:
    parts = [int(part) for part in text.split(".")]
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def _package_version() -> tuple[int, int, int]:
    source = (PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', source, re.MULTILINE)
    assert match, "stateset_agents/__init__.py must define __version__"
    return _version_tuple(match.group(1).split("+")[0].split("rc")[0])


def _package_sources() -> list[Path]:
    return sorted(
        path for path in PACKAGE_DIR.rglob("*.py") if "__pycache__" not in path.parts
    )


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _searchable_text(text: str, tree: ast.AST) -> str:
    """Source plus every string constant, so implicitly concatenated literals
    ("will be removed " "in v0.71.0") read as the one sentence they are."""
    constants = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    return text + "\n" + "\n".join(constants)


def scheduled_removals(text: str) -> list[str]:
    """Return every future-tense removal version stated in ``text``."""
    versions = [
        match.group(2)
        for match in _SCHEDULED.finditer(text)
        if (match.group(1) or "").lower() not in _HISTORICAL
    ]
    versions.extend(match.group(1) for match in _ASSIGNED.finditer(text))
    return versions


def _is_deprecation_category(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "DeprecationWarning") or (
        isinstance(node, ast.Attribute) and node.attr == "DeprecationWarning"
    )


def _call_name(node: ast.Call) -> str:
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def alias_table_removals(tree: ast.AST) -> list[str]:
    """Removal versions in ``deprecated_alias_getattr(name, {old: (mod, new,
    "X.Y.Z")})`` tables, which state the version as a bare tuple item."""
    versions: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and _call_name(node) == "deprecated_alias_getattr"
        ):
            continue
        for arg in node.args[1:2]:
            if not isinstance(arg, ast.Dict):
                continue
            for value in arg.values:
                if (
                    isinstance(value, ast.Tuple)
                    and len(value.elts) == 3
                    and isinstance(value.elts[2], ast.Constant)
                    and isinstance(value.elts[2].value, str)
                ):
                    versions.append(value.elts[2].value.lstrip("v"))
    return versions


def _emits_deprecation_warning(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name in {"warn_deprecated", "deprecated_alias_getattr"}:
            return True
        if name != "warn":
            continue
        if len(node.args) >= 2 and _is_deprecation_category(node.args[1]):
            return True
        if any(
            kw.arg == "category" and _is_deprecation_category(kw.value)
            for kw in node.keywords
        ):
            return True
    return False


def test_marker_parser_distinguishes_schedule_from_history() -> None:
    assert scheduled_removals("it will be removed in v0.71.0.") == ["0.71.0"]
    assert scheduled_removals("deprecated in v0.69.0 and removed in v0.70.0") == [
        "0.70.0"
    ]
    assert scheduled_removals('warn_deprecated(x, removal="0.72.0")') == ["0.72.0"]
    assert scheduled_removals('LEGACY_REMOVAL = "v0.69.0"') == ["0.69.0"]
    assert scheduled_removals("the re-exports were removed in v0.65.0") == []
    assert _UNDATED.search("will be removed in a future release")


def test_no_scheduled_removal_is_past_due() -> None:
    current = _package_version()
    overdue: list[str] = []
    for path in _package_sources():
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        stated = scheduled_removals(_searchable_text(text, tree))
        for version in set(stated + alias_table_removals(tree)):
            if _version_tuple(version) <= current:
                overdue.append(f"{_rel(path)}: removal v{version}")
    for rel, version in EXTERNAL_SCHEDULE.items():
        if _version_tuple(version) <= current:
            overdue.append(f"{rel}: removal v{version} (EXTERNAL_SCHEDULE)")
    assert not overdue, (
        "Deprecated code has reached its stated removal version; delete it "
        "(with a CHANGELOG 'Removed' entry and, for stable names, "
        "`python scripts/check_api_compatibility.py --write`):\n  "
        + "\n  ".join(overdue)
    )


def test_no_undated_removal_promises() -> None:
    offenders = [
        f"{_rel(path)}:{lineno}"
        for path in _package_sources()
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if _UNDATED.search(line)
    ]
    assert not offenders, (
        "Deprecations must name the release that removes them ('will be "
        "removed in vX.Y.Z'), not 'a future release':\n  " + "\n  ".join(offenders)
    )


def test_every_deprecation_warning_states_a_removal_version() -> None:
    undated: list[str] = []
    for path in _package_sources():
        rel = _rel(path)
        if rel in EXTERNAL_SCHEDULE or rel == HELPER:
            continue
        text = path.read_text(encoding="utf-8")
        if "DeprecationWarning" not in text and "warn_deprecated" not in text:
            continue
        tree = ast.parse(text)
        if _emits_deprecation_warning(tree) and not (
            scheduled_removals(_searchable_text(text, tree))
            or alias_table_removals(tree)
        ):
            undated.append(rel)
    assert not undated, (
        "These modules emit a DeprecationWarning without stating when the "
        "deprecated path is removed; add 'will be removed in vX.Y.Z' (or use "
        "stateset_agents._deprecation.warn_deprecated):\n  " + "\n  ".join(undated)
    )


@pytest.mark.parametrize("rel", sorted(EXTERNAL_SCHEDULE))
def test_external_schedule_entries_are_live(rel: str) -> None:
    path = REPO_ROOT / rel
    assert path.exists(), f"{rel} is gone; drop it from EXTERNAL_SCHEDULE"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    assert _emits_deprecation_warning(tree), (
        f"{rel} no longer emits a DeprecationWarning; drop it from " "EXTERNAL_SCHEDULE"
    )
    assert not scheduled_removals(_searchable_text(text, tree)), (
        f"{rel} now states its own removal version; drop it from " "EXTERNAL_SCHEDULE"
    )


def test_helper_message_states_the_removal_release() -> None:
    from stateset_agents._deprecation import deprecation_message, warn_deprecated

    message = deprecation_message("old.name", removal="v0.72.0", replacement="new")
    assert message == (
        "old.name is deprecated and will be removed in v0.72.0; use new instead."
    )
    assert scheduled_removals(message) == ["0.72.0"]
    with pytest.warns(DeprecationWarning, match=r"removed in v0\.72\.0") as record:
        warn_deprecated("old.name", removal="0.72.0", stacklevel=1)
    assert record[0].filename == __file__


def test_alias_getattr_serves_the_canonical_object() -> None:
    from stateset_agents._deprecation import deprecated_alias_getattr

    getter = deprecated_alias_getattr(
        "legacy.mod", {"OldPath": ("pathlib", "Path", "0.72.0")}
    )
    with pytest.warns(DeprecationWarning, match="legacy.mod.OldPath"):
        assert getter("OldPath") is Path
    with pytest.raises(AttributeError):
        getter("Missing")


def test_implicitly_concatenated_literals_are_read_as_one_sentence() -> None:
    source = 'MESSAGE = ("the alias will be removed "\n    "in v0.71.0")\n'
    assert scheduled_removals(source) == []
    assert scheduled_removals(_searchable_text(source, ast.parse(source))) == ["0.71.0"]


def test_alias_tables_state_their_removal_versions() -> None:
    source = (
        "__getattr__ = deprecated_alias_getattr(\n"
        "    __name__, {'Old': ('pkg.mod', 'New', '0.71.0')}\n"
        ")\n"
    )
    tree = ast.parse(source)
    assert _emits_deprecation_warning(tree)
    assert alias_table_removals(tree) == ["0.71.0"]
