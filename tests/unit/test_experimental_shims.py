"""Deprecation shims for modules moved to ``stateset_agents.experimental``.

v0.64.0 moved the least-proven training/evaluation modules (PPO, RLAIF, offline
RL, Decision Transformer, sim-to-real, Bayesian reward model) into the
experimental namespace. Their old import paths keep working for one release:
every old module path and every package-level re-export must emit a
``DeprecationWarning`` that names the new location and the removal version, and
must hand back the *same* objects as the new location.

``pytest.ini`` turns these warnings into errors everywhere else, so this module
is the only place in the suite that is allowed to touch the old paths.
"""

from __future__ import annotations

import importlib
import sys

import pytest

REMOVAL = "v0.65.0"

#: old module path -> new module path
MOVED_MODULES = {
    "stateset_agents.training.ppo_trainer": "stateset_agents.experimental.ppo",
    "stateset_agents.training.rlaif_trainer": "stateset_agents.experimental.rlaif",
    "stateset_agents.training.offline_rl_algorithms": (
        "stateset_agents.experimental.offline_rl.algorithms"
    ),
    "stateset_agents.training.offline_rl_bcq": (
        "stateset_agents.experimental.offline_rl.bcq"
    ),
    "stateset_agents.training.offline_rl_bear": (
        "stateset_agents.experimental.offline_rl.bear"
    ),
    "stateset_agents.training.offline_grpo_trainer": (
        "stateset_agents.experimental.offline_rl.offline_grpo"
    ),
    "stateset_agents.training.decision_transformer": (
        "stateset_agents.experimental.offline_rl.decision_transformer"
    ),
    "stateset_agents.training.sim_to_real": (
        "stateset_agents.experimental.sim_to_real.transfer"
    ),
    "stateset_agents.training.domain_randomization": (
        "stateset_agents.experimental.sim_to_real.domain_randomization"
    ),
    "stateset_agents.evaluation.sim_to_real_metrics": (
        "stateset_agents.experimental.sim_to_real.metrics"
    ),
    "stateset_agents.environments.conversation_simulator": (
        "stateset_agents.experimental.sim_to_real.conversation_simulator"
    ),
    "stateset_agents.rewards.bayesian_reward_model": (
        "stateset_agents.experimental.bayesian_reward_model"
    ),
}

#: (package, name, new module) for package-level re-exports outside the
#: training registry (which test_training_lazy_exports.py covers).
PACKAGE_EXPORTS = [
    (
        "stateset_agents",
        "SimToRealMetrics",
        "stateset_agents.experimental.sim_to_real.metrics",
    ),
    (
        "stateset_agents",
        "SimToRealEvaluator",
        "stateset_agents.experimental.sim_to_real.metrics",
    ),
    (
        "stateset_agents",
        "ConversationSimulator",
        "stateset_agents.experimental.sim_to_real.conversation_simulator",
    ),
    (
        "stateset_agents",
        "ConversationSimulatorConfig",
        "stateset_agents.experimental.sim_to_real.conversation_simulator",
    ),
    *(
        (
            "stateset_agents.evaluation",
            name,
            "stateset_agents.experimental.sim_to_real.metrics",
        )
        for name in (
            "SimToRealMetrics",
            "SimToRealEvaluator",
            "compute_distribution_divergence",
            "compute_response_statistics",
        )
    ),
    *(
        (
            "stateset_agents.environments",
            name,
            "stateset_agents.experimental.sim_to_real.conversation_simulator",
        )
        for name in (
            "ConversationSimulator",
            "ConversationSimulatorConfig",
            "UserSimulator",
        )
    ),
    ("stateset_agents.training", "PPOTrainer", "stateset_agents.experimental.ppo"),
    ("stateset_agents.training", "RLAIFTrainer", "stateset_agents.experimental.rlaif"),
    (
        "stateset_agents.training",
        "BCQTrainer",
        "stateset_agents.experimental.offline_rl.bcq",
    ),
    (
        "stateset_agents.training",
        "OfflineGRPOTrainer",
        "stateset_agents.experimental.offline_rl.offline_grpo",
    ),
    (
        "stateset_agents.training",
        "SimToRealTransfer",
        "stateset_agents.experimental.sim_to_real.transfer",
    ),
]


def _import_new(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        pytest.skip(f"{module_name} needs an optional extra: {exc}")


def _public_names(module) -> list[str]:
    """Names a ``from module import *``-style user could have relied on."""
    return [
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and getattr(value, "__module__", None) == module.__name__
    ]


@pytest.mark.parametrize("old,new", sorted(MOVED_MODULES.items()))
def test_old_module_path_warns_and_aliases_new_module(
    monkeypatch: pytest.MonkeyPatch, old: str, new: str
) -> None:
    new_module = _import_new(new)
    # Force a fresh import of the shim so its module-level warning fires.
    monkeypatch.delitem(sys.modules, old, raising=False)

    with pytest.warns(DeprecationWarning) as record:
        old_module = importlib.import_module(old)

    messages = [str(warning.message) for warning in record]
    assert any(
        old in message and new in message and REMOVAL in message for message in messages
    ), messages
    assert old_module is new_module

    public = _public_names(new_module)
    assert public, f"{new} defines no public names"
    for name in public:
        assert getattr(old_module, name) is getattr(new_module, name)


@pytest.mark.parametrize("old,new", sorted(MOVED_MODULES.items()))
def test_from_import_via_old_path_returns_same_objects(
    monkeypatch: pytest.MonkeyPatch, old: str, new: str
) -> None:
    new_module = _import_new(new)
    monkeypatch.delitem(sys.modules, old, raising=False)
    name = _public_names(new_module)[0]

    with pytest.warns(DeprecationWarning, match=REMOVAL.replace(".", r"\.")):
        namespace: dict[str, object] = {}
        exec(f"from {old} import {name}", namespace)  # noqa: S102 - fixed input

    assert namespace[name] is getattr(new_module, name)


@pytest.mark.parametrize("package,name,new", PACKAGE_EXPORTS)
def test_package_reexport_warns_and_returns_same_object(
    package: str, name: str, new: str
) -> None:
    new_module = _import_new(new)
    pkg = importlib.import_module(package)

    with pytest.warns(DeprecationWarning) as record:
        value = getattr(pkg, name)

    messages = [str(warning.message) for warning in record]
    assert any(
        f"{package}.{name}" in message and new in message and REMOVAL in message
        for message in messages
    ), messages
    assert value is getattr(new_module, name)
    # Deprecated names are never cached, so every access keeps warning.
    assert name not in vars(pkg)


@pytest.mark.timeout(240)
def test_new_paths_do_not_route_through_shims() -> None:
    """The relocated modules must not import each other via the old paths."""
    import subprocess

    code = (
        "import importlib, warnings\n"
        "warnings.filterwarnings('error', message=r'.*removed in v0\\.65\\.0', "
        "category=DeprecationWarning)\n"
        # Sanity check: the filter really turns a shim import into an error.
        "try:\n"
        "    importlib.import_module('stateset_agents.rewards.bayesian_reward_model')\n"
        "except DeprecationWarning:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('shim did not warn')\n"
        "import sys\n"
        "sys.modules.pop('stateset_agents.rewards.bayesian_reward_model', None)\n"
    ) + "".join(f"importlib.import_module({new!r})\n" for new in MOVED_MODULES.values())
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=200,
    )
    assert result.returncode == 0, result.stderr
