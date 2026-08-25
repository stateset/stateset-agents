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
