"""The secondary GRPO app is deprecated as a whole (removed in v0.71.0).

Nothing outside ``stateset_agents.api.grpo`` imports it: the supported gateway
(``stateset_agents.api.main``) has its own middleware and rate limiter. The app
surface was deprecated in v0.16.0 without a removal date; since v0.70.0 any
import of the package warns with the removal version.
"""

import importlib
import sys
import warnings

import pytest


def _fresh_import(module_name: str):
    """Import module_name after purging it and its `grpo`-tree relatives."""
    for name in list(sys.modules):
        if name == module_name or name.startswith(f"{module_name}."):
            del sys.modules[name]
    return importlib.import_module(module_name)


def _grpo_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "stateset_agents.api.grpo" in str(w.message)
    ]


def test_importing_api_main_raises_no_grpo_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fresh_import("stateset_agents.api.main")

    assert _grpo_warnings(caught) == []


def test_nothing_outside_the_grpo_package_imports_it():
    import ast
    from pathlib import Path

    package = Path(__file__).resolve().parents[2] / "stateset_agents"
    grpo_dir = package / "api" / "grpo"
    offenders = []
    for path in package.rglob("*.py"):
        if grpo_dir in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        if "grpo" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom) and node.module:
                target = "." * node.level + node.module
                if target.endswith("api.grpo") or ".api.grpo." in f"{target}.":
                    offenders.append(f"{path.relative_to(package)}:{node.lineno}")
                if node.level and node.module.split(".")[0] == "grpo":
                    offenders.append(f"{path.relative_to(package)}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("stateset_agents.api.grpo"):
                        offenders.append(f"{path.relative_to(package)}:{node.lineno}")
    assert offenders == []


def test_importing_grpo_package_warns_with_removal_version():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fresh_import("stateset_agents.api.grpo")

    messages = _grpo_warnings(caught)
    assert len(messages) == 1
    assert "removed in v0.71.0" in messages[0]
    assert "stateset_agents.api.main" in messages[0]


def test_importing_an_app_module_directly_warns():
    sys.modules.pop("stateset_agents.api.grpo.service", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fresh_import("stateset_agents.api.grpo")
        importlib.import_module("stateset_agents.api.grpo.service")

    assert any("removed in v0.71.0" in message for message in _grpo_warnings(caught))


def test_app_submodules_load_lazily_without_a_second_warning():
    with pytest.warns(DeprecationWarning):
        grpo_pkg = _fresh_import("stateset_agents.api.grpo")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert grpo_pkg.service is not None

    assert _grpo_warnings(caught) == []


def test_unknown_attribute_raises_attribute_error():
    with pytest.warns(DeprecationWarning):
        grpo_pkg = _fresh_import("stateset_agents.api.grpo")

    with pytest.raises(AttributeError):
        _ = grpo_pkg.this_symbol_does_not_exist
