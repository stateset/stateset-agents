"""Dependency stubs must not evict real modules or duplicate their classes."""

import sys
from pathlib import Path
from types import ModuleType

from tests import _module_stubs


def test_foreign_stub_hiding_preserves_real_submodules(monkeypatch):
    stub = ModuleType("fixture_dependency")
    nested_stub = ModuleType("fixture_dependency.fake")
    real = ModuleType("fixture_dependency.real")
    owner = Path("owner.py").resolve()
    monkeypatch.setattr(
        _module_stubs,
        "STUBS",
        {
            "fixture_dependency": (stub, owner),
            "fixture_dependency.fake": (nested_stub, owner),
        },
    )
    monkeypatch.setitem(sys.modules, "fixture_dependency", stub)
    monkeypatch.setitem(sys.modules, "fixture_dependency.fake", nested_stub)
    monkeypatch.setitem(sys.modules, "fixture_dependency.real", real)

    hidden = _module_stubs.hide_foreign_stubs(Path("consumer.py"))
    assert hidden == {
        "fixture_dependency": stub,
        "fixture_dependency.fake": nested_stub,
    }
    assert sys.modules["fixture_dependency.real"] is real
    _module_stubs.restore_stubs(hidden)
    assert sys.modules["fixture_dependency"] is stub
    assert sys.modules["fixture_dependency.fake"] is nested_stub


def test_hiding_does_not_remove_real_replacement(monkeypatch):
    owner = Path("owner.py").resolve()
    stub = ModuleType("fixture_dependency")
    real = ModuleType("fixture_dependency")
    child = ModuleType("fixture_dependency.child")
    monkeypatch.setattr(_module_stubs, "STUBS", {"fixture_dependency": (stub, owner)})
    monkeypatch.setitem(sys.modules, "fixture_dependency", real)
    monkeypatch.setitem(sys.modules, "fixture_dependency.child", child)

    assert _module_stubs.hide_foreign_stubs(Path("consumer.py")) == {}
    assert sys.modules["fixture_dependency"] is real
    assert sys.modules["fixture_dependency.child"] is child


def test_owner_keeps_its_stub(monkeypatch):
    owner = Path("owner.py").resolve()
    stub = ModuleType("fixture_dependency")
    monkeypatch.setattr(_module_stubs, "STUBS", {"fixture_dependency": (stub, owner)})
    monkeypatch.setitem(sys.modules, "fixture_dependency", stub)

    assert _module_stubs.hide_foreign_stubs(owner) == {}
    assert sys.modules["fixture_dependency"] is stub
