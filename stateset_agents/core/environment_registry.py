"""Find environments by name, including ones shipped by other packages.

A package publishes an environment by declaring an entry point in the
``stateset_agents.environments`` group whose value is a factory returning an
:class:`~stateset_agents.core.environment_base.Environment`:

.. code-block:: toml

    [project.entry-points."stateset_agents.environments"]
    my-task = "my_package.envs:make_environment"

After ``pip install my-package`` the environment is available as
``make_environment("my-task", **options)`` with no change to this package.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module, metadata
from typing import Any

from .environment_base import Environment

ENTRY_POINT_GROUP = "stateset_agents.environments"

EnvironmentFactory = Callable[..., Environment]

# Built-ins are named by import path so listing them imports nothing heavy.
_BUILTIN: dict[str, str] = {
    "refund-policy": (
        "stateset_agents.core.environments.refund_policy_environment:"
        "RefundPolicyEnvironment"
    ),
    "tau-retail": "stateset_agents.training.tau_retail:make_environment",
}
_REGISTERED: dict[str, EnvironmentFactory] = {}


def _entry_points() -> dict[str, Any]:
    return {
        entry.name: entry for entry in metadata.entry_points(group=ENTRY_POINT_GROUP)
    }


def register_environment(
    name: str, factory: EnvironmentFactory, *, replace: bool = False
) -> None:
    """Register ``factory`` under ``name`` for this process."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Environment name must be a nonempty string")
    if not callable(factory):
        raise TypeError("Environment factory must be callable")
    taken = name in _REGISTERED or name in _BUILTIN or name in _entry_points()
    if taken and not replace:
        raise ValueError(f"Environment {name!r} is already registered")
    _REGISTERED[name] = factory


def available_environments() -> list[str]:
    """Return every known environment name, sorted."""
    return sorted({*_BUILTIN, *_REGISTERED, *_entry_points()})


def environment_factory(name: str) -> EnvironmentFactory:
    """Resolve ``name``: process registrations, then plugins, then built-ins."""
    if name in _REGISTERED:
        return _REGISTERED[name]
    plugins = _entry_points()
    if name in plugins:
        factory: EnvironmentFactory = plugins[name].load()
        return factory
    if name in _BUILTIN:
        module, _, attribute = _BUILTIN[name].partition(":")
        builtin: EnvironmentFactory = getattr(import_module(module), attribute)
        return builtin
    known = ", ".join(available_environments())
    raise KeyError(f"Unknown environment {name!r}. Available: {known}")


def make_environment(name: str, **options: Any) -> Environment:
    """Build the environment registered as ``name``."""
    environment = environment_factory(name)(**options)
    if not isinstance(environment, Environment):
        raise TypeError(
            f"Factory for {name!r} returned {type(environment).__name__}, "
            "not an Environment"
        )
    return environment
