"""Deprecation warning must be scoped to the deprecated grpo app surface.

`stateset_agents.api.grpo` re-exports shared infrastructure (rate_limiter,
state, config, handlers, metrics, models) that is used by the normal app
(`stateset_agents.api.main` -> `middleware.py`). Only the secondary,
deprecated app surface (`service`, `service_routes`, `router_v1`, `auth`
submodules) should warn on access.
"""

import importlib
import sys
import warnings


def _fresh_import(module_name: str):
    """Import module_name after purging it and its `grpo`-tree relatives.

    Ensures we observe warnings from a real (re-)import rather than a
    cached module object from a previous test.
    """
    for name in list(sys.modules):
        if name == module_name or name.startswith(f"{module_name}."):
            del sys.modules[name]
    return importlib.import_module(module_name)


def test_importing_api_main_raises_no_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fresh_import("stateset_agents.api.main")

    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    grpo_warnings = [
        w for w in deprecation_warnings if "stateset_agents.api.grpo" in str(w.message)
    ]
    assert grpo_warnings == []


def test_importing_grpo_package_alone_raises_no_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fresh_import("stateset_agents.api.grpo")

    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecation_warnings == []


def test_accessing_deprecated_service_symbol_warns():
    grpo_pkg = _fresh_import("stateset_agents.api.grpo")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grpo_pkg.service  # noqa: B018 - attribute access triggers __getattr__

    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 1
    assert "grpo.service" in str(deprecation_warnings[0].message)


def test_accessing_rate_limiter_symbol_does_not_warn():
    grpo_pkg = _fresh_import("stateset_agents.api.grpo")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grpo_pkg.get_rate_limiter  # noqa: B018

    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecation_warnings == []


def test_unknown_attribute_raises_attribute_error():
    grpo_pkg = _fresh_import("stateset_agents.api.grpo")

    import pytest

    with pytest.raises(AttributeError):
        _ = grpo_pkg.this_symbol_does_not_exist
