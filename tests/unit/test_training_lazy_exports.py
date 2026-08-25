"""Resolution meta-tests for ``stateset_agents.training``'s lazy export table.

``training/__init__.py`` resolves nearly everything it advertises through a
PEP 562 ``__getattr__`` backed by the string-keyed ``_OPTIONAL_EXPORTS`` table,
so that importing the package stays free of ``torch`` (see
``test_torch_import_policy.py``). String keys are not checked by any linter: a
typo in a module path or attribute name would only surface at runtime, for the
one user who happened to import that name.

These tests walk the whole advertised surface and prove every name resolves.
A missing optional *extra* (torch / transformers / trl / peft not installed) is
tolerated; a broken ``stateset_agents.*`` path is not.

Mirrors ``test_data_package_lazy_exports.py``.
"""

from __future__ import annotations

import importlib

import pytest

import stateset_agents.training as training_pkg

#: Optional heavy extras. An ImportError naming one of these means the extra is
#: simply not installed, which is not a table bug.
OPTIONAL_EXTRAS = ("torch", "transformers", "trl", "peft", "vllm", "bitsandbytes")


def _is_missing_extra(exc: BaseException) -> bool:
    """True if ``exc`` is an optional extra being absent, not a broken entry."""
    if not isinstance(exc, ImportError):
        return False

    # A ModuleNotFoundError for one of our OWN modules is always a real bug,
    # even if the message happens to mention an extra.
    name = getattr(exc, "name", None) or ""
    if name.startswith("stateset_agents"):
        return False

    text = f"{name} {exc}".lower()
    return any(extra in text for extra in OPTIONAL_EXTRAS)


@pytest.mark.parametrize("name", sorted(training_pkg.__all__))
def test_public_name_resolves(name: str) -> None:
    """Every name in ``__all__`` must resolve (or fail only on a missing extra)."""
    try:
        assert getattr(training_pkg, name) is not None or True
    except AttributeError as exc:
        pytest.fail(
            f"stateset_agents.training.{name} is advertised in __all__ but does "
            f"not resolve: {exc}"
        )
    except ImportError as exc:
        if _is_missing_extra(exc):
            pytest.skip(f"{name} needs an optional extra that is not installed: {exc}")
        raise


@pytest.mark.parametrize("name", sorted(training_pkg._OPTIONAL_EXPORTS))
def test_lazy_export_entry_resolves(name: str) -> None:
    """Every ``_OPTIONAL_EXPORTS`` entry must point at a real module attribute."""
    module_name, attr_name = training_pkg._OPTIONAL_EXPORTS[name]

    assert module_name.startswith("stateset_agents.training"), (
        f"_OPTIONAL_EXPORTS[{name!r}] points outside the training package: "
        f"{module_name!r}"
    )

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        if _is_missing_extra(exc):
            pytest.skip(f"{module_name} needs an optional extra: {exc}")
        pytest.fail(f"_OPTIONAL_EXPORTS[{name!r}] module {module_name!r}: {exc}")

    assert hasattr(module, attr_name), (
        f"_OPTIONAL_EXPORTS[{name!r}] promises {module_name}.{attr_name}, "
        f"but that module has no such attribute"
    )


def test_previously_eager_names_still_resolve() -> None:
    """The 13 names moved out of the eager import block must still work.

    These were plain ``from .config import ...`` bindings before the torch-free
    entry-point work; they are now lazy table entries and must behave the same.
    """
    formerly_eager = (
        "ContinualLearningConfig",
        "ContinualLearningManager",
        "EvaluationConfig",
        "GRPOTrainer",
        "MultiTurnGRPOTrainer",
        "SingleTurnGRPOTrainer",
        "TrainingConfig",
        "TrainingMode",
        "TrainingProfile",
        "TrajectoryReplayBuffer",
        "evaluate_agent",
        "get_config_for_task",
        "train",
    )
    for name in formerly_eager:
        assert name in training_pkg._OPTIONAL_EXPORTS, (
            f"{name} was an eager export and must stay reachable via the lazy table"
        )
        assert getattr(training_pkg, name) is not None


def test_all_names_are_reachable() -> None:
    """No ``__all__`` entry may be missing from both globals and the table."""
    unreachable = [
        name
        for name in training_pkg.__all__
        if name not in training_pkg._OPTIONAL_EXPORTS
        and name not in vars(training_pkg)
        and name != "TRL_AVAILABLE"
    ]
    assert not unreachable, (
        f"names in __all__ with no eager binding and no lazy entry: {unreachable}"
    )


def test_trl_available_resolves_lazily() -> None:
    """``TRL_AVAILABLE`` is computed on first access, not at import time."""
    assert isinstance(training_pkg.TRL_AVAILABLE, bool)


def test_table_is_the_registry_module() -> None:
    """``_OPTIONAL_EXPORTS`` is the data file's dict, not a copy."""
    from stateset_agents.training import _registry

    assert training_pkg._OPTIONAL_EXPORTS is _registry.OPTIONAL_EXPORTS
    assert training_pkg.__all__ == list(_registry.PUBLIC_NAMES)


#: Names bound eagerly in ``training/__init__.py`` (availability flags), which
#: therefore need no lazy-table entry.
EAGER_NAMES = frozenset(
    {
        "VLLM_BACKEND_AVAILABLE",
        "VLLM_AVAILABLE",
        "TRL_AVAILABLE",
        "GSPO_AVAILABLE",
        "GEPO_AVAILABLE",
        "DAPO_AVAILABLE",
        "VAPO_AVAILABLE",
        "PPO_AVAILABLE",
        "KL_CONTROLLERS_AVAILABLE",
        "EMA_AVAILABLE",
        "RLAIF_AVAILABLE",
        "OFFLINE_RL_AVAILABLE",
        "BCQ_AVAILABLE",
        "BEAR_AVAILABLE",
        "DECISION_TRANSFORMER_AVAILABLE",
        "SIM_TO_REAL_AVAILABLE",
        "AUTO_RESEARCH_AVAILABLE",
    }
)


def test_registry_covers_every_non_eager_public_name() -> None:
    """``OPTIONAL_EXPORTS`` must serve all of ``__all__`` bar the eager flags.

    This is the anti-drift guard between ``_registry.py`` and ``__init__.py``:
    adding a name to ``__all__`` without a table entry (or without making it an
    eager flag) fails here rather than at some user's ``import``.
    """
    from stateset_agents.training import _registry

    missing = sorted(
        set(_registry.PUBLIC_NAMES) - EAGER_NAMES - set(_registry.OPTIONAL_EXPORTS)
    )
    assert not missing, f"__all__ names with no lazy-table entry: {missing}"


def test_eager_names_are_actually_eager() -> None:
    """Every name in ``EAGER_NAMES`` is bound in ``__init__``, not in the table.

    Keeps the exemption list above honest: a flag that becomes lazy (or
    disappears) must be removed from ``EAGER_NAMES`` rather than silently
    widening the hole in the previous test.
    """
    from stateset_agents.training import _registry

    for name in EAGER_NAMES:
        if name == "TRL_AVAILABLE":  # resolved lazily by design, never a table entry
            continue
        assert name in vars(training_pkg), f"{name} is not eagerly bound"
        assert name not in _registry.OPTIONAL_EXPORTS, f"{name} is now a table entry"
