"""Stable names the framework never uses are deprecated (removed in v0.72.0).

The policy (docs/API_COMPATIBILITY.md) keeps a deprecated stable name for at
least one minor release after the deprecating one, so names deprecated in
v0.70.0 go in v0.72.0. They stay in ``__all__`` and the v1 contract until then
and warn on every access.
"""

from __future__ import annotations

import ast
import importlib
import sys
import warnings
from pathlib import Path

import pytest

import stateset_agents as sa

PACKAGE = Path(__file__).resolve().parents[2] / "stateset_agents"

DEPRECATED = sorted(sa._DEPRECATED_STABLE_EXPORTS)

#: Root names from the same modules that the framework does use, so they stay.
KEPT = (
    "GRPOException",
    "ValidationException",  # raised by core.input_validation / utils.security
    "ResourceException",  # raised by core.async_pool
    "ErrorCode",  # passed by those raise sites
    "StateSetError",
    "ConfigurationError",
)


def test_the_deprecated_set_is_the_unused_type_error_and_optimizer_names() -> None:
    modules = {sa._LAZY_EXPORTS[name][0] for name in DEPRECATED}
    assert modules == {
        "stateset_agents.core.error_handling",
        "stateset_agents.core.performance_optimizer",
        "stateset_agents.core.type_system",
    }
    assert len(DEPRECATED) == 34
    assert set(DEPRECATED) <= set(sa.__all__)  # contract unchanged until removal
    assert sa._DEPRECATED_STABLE_EXPORTS_REMOVAL == "v0.72.0"


@pytest.mark.parametrize("name", DEPRECATED)
def test_deprecated_root_names_warn_on_every_access(name: str) -> None:
    for _ in range(2):  # not cached in the module namespace
        with pytest.warns(DeprecationWarning, match=r"removed in v0\.72\.0") as rec:
            value = getattr(sa, name)
        assert value is not None
        assert f"stateset_agents.{name}" in str(rec[0].message)
    assert name not in vars(sa)


@pytest.mark.parametrize("name", KEPT)
def test_names_the_framework_uses_do_not_warn(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert getattr(sa, name) is not None


def test_deprecated_names_are_unused_inside_the_package() -> None:
    """Nothing in the package may reach for a deprecated root name."""
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "from stateset_agents import" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom) and node.module == "stateset_agents":
                used = {alias.name for alias in node.names} & set(DEPRECATED)
                if used:
                    offenders.append(f"{path.relative_to(PACKAGE)}: {sorted(used)}")
    assert offenders == []


def test_one_conversation_turn_class() -> None:
    """``core.trajectory.ConversationTurn`` is the only class of that name."""
    definitions = []
    for path in PACKAGE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "class ConversationTurn" not in text:
            continue
        for node in ast.parse(text).body:
            if isinstance(node, ast.ClassDef) and node.name == "ConversationTurn":
                definitions.append(path.relative_to(PACKAGE).as_posix())
    assert definitions == ["core/trajectory.py"]


def test_type_system_conversation_turn_is_a_deprecated_alias() -> None:
    from stateset_agents.core import type_system

    with pytest.warns(DeprecationWarning, match=r"removed in v0\.71\.0"):
        alias = type_system.ConversationTurn
    assert alias is type_system.ConversationTurnDict
    with pytest.warns(DeprecationWarning):
        assert sa.TypedConversationTurn is type_system.ConversationTurnDict


def test_core_types_is_deprecated_and_aliases_the_canonical_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stateset_agents.core.trajectory import ConversationTurn

    monkeypatch.delitem(sys.modules, "stateset_agents.core.types", raising=False)
    with pytest.warns(DeprecationWarning, match=r"core\.types.*removed in v0\.71\.0"):
        legacy = importlib.import_module("stateset_agents.core.types")
    assert legacy.ConversationTurn is ConversationTurn
    turn = legacy.ConversationTurn("hi", "hello", 0.5)
    assert (turn.user_message, turn.assistant_response, turn.reward) == (
        "hi",
        "hello",
        0.5,
    )
    trajectory = legacy.Trajectory(turns=[turn, legacy.ConversationTurn("a", "b", 1.0)])
    assert trajectory.total_reward == pytest.approx(1.5)
