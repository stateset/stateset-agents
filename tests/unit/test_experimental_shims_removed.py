"""The v0.64.0 experimental-move shims are gone in v0.65.0.

v0.64.0 moved PPO, RLAIF, offline RL, Decision Transformer, sim-to-real and the
Bayesian reward model into :mod:`stateset_agents.experimental` and left
deprecation shims at the old locations. v0.65.0 deleted them. These tests keep
the old paths from silently coming back (a stray module file, a lazy-registry
entry, or a package ``__getattr__`` fallback) and pin the experimental paths as
the only home.
"""

from __future__ import annotations

import importlib

import pytest

#: removed module path -> experimental replacement
REMOVED_MODULES = {
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

_METRICS = "stateset_agents.experimental.sim_to_real.metrics"
_SIMULATOR = "stateset_agents.experimental.sim_to_real.conversation_simulator"

#: experimental module -> names formerly re-exported by stateset_agents.training
_TRAINING_NAMES = {
    "stateset_agents.experimental.ppo": (
        "PPOConfig",
        "PPOTrainer",
        "PPOValueHead",
        "AdaptiveKLController",
        "compute_gae",
        "train_ppo",
    ),
    "stateset_agents.experimental.rlaif": (
        "RLAIFConfig",
        "RLAIFTrainer",
        "ConstitutionalAI",
        "train_rlaif",
    ),
    "stateset_agents.experimental.offline_rl.algorithms": (
        "CQLConfig",
        "IQLConfig",
        "ConservativeQLearning",
        "ImplicitQLearning",
        "OfflineRLTrainer",
    ),
    "stateset_agents.experimental.offline_rl.bcq": (
        "BCQConfig",
        "BatchConstrainedQLearning",
        "ConversationalVAE",
        "BCQTrainer",
    ),
    "stateset_agents.experimental.offline_rl.bear": (
        "BEARConfig",
        "ConversationalBEAR",
        "MMDKernel",
        "BEARTrainer",
    ),
    "stateset_agents.experimental.offline_rl.decision_transformer": (
        "DecisionTransformerConfig",
        "DecisionTransformer",
        "DecisionTransformerTrainer",
        "ConversationEmbedder",
    ),
    "stateset_agents.experimental.offline_rl.offline_grpo": (
        "OfflineGRPOConfig",
        "OfflineGRPOTrainer",
        "OfflineRLAlgorithm",
    ),
    "stateset_agents.experimental.sim_to_real.domain_randomization": (
        "DomainRandomizationConfig",
        "DomainRandomizer",
        "PersonaGenerator",
        "ScenarioGenerator",
        "UserPersona",
    ),
    "stateset_agents.experimental.sim_to_real.transfer": (
        "SimToRealConfig",
        "SimToRealTransfer",
        "UserBehaviorModel",
        "DomainAdaptationModule",
    ),
}

#: (package, removed name, experimental module that now provides it).
#: The four root ``stateset_agents`` sim-to-real exports are deliberately absent:
#: they are stable v1-contract names, so docs/API_COMPATIBILITY.md keeps them
#: (deprecated) through v0.65.x; test_public_api_contracts.py covers them.
REMOVED_PACKAGE_EXPORTS = [
    *(
        ("stateset_agents.evaluation", name, _METRICS)
        for name in (
            "SimToRealMetrics",
            "SimToRealEvaluator",
            "compute_distribution_divergence",
            "compute_response_statistics",
        )
    ),
    *(
        ("stateset_agents.environments", name, _SIMULATOR)
        for name in (
            "ConversationSimulator",
            "ConversationSimulatorConfig",
            "UserSimulator",
        )
    ),
    *(
        ("stateset_agents.training", name, module)
        for module, names in _TRAINING_NAMES.items()
        for name in names
    ),
]


def _import_experimental(module_name: str) -> object:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        if (exc.name or "").split(".")[0] in {"torch", "transformers", "numpy"}:
            pytest.skip(f"{module_name} needs an optional extra: {exc}")
        raise


@pytest.mark.parametrize("old_path", sorted(REMOVED_MODULES))
def test_removed_module_paths_do_not_import(old_path: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old_path)


@pytest.mark.parametrize(
    ("package", "name", "_new_module"),
    REMOVED_PACKAGE_EXPORTS,
    ids=[f"{pkg}.{name}" for pkg, name, _ in REMOVED_PACKAGE_EXPORTS],
)
def test_removed_package_exports_are_gone(
    package: str, name: str, _new_module: str
) -> None:
    module = importlib.import_module(package)
    assert name not in getattr(module, "__all__", ())
    with pytest.raises(AttributeError):
        getattr(module, name)


@pytest.mark.parametrize("new_path", sorted(set(REMOVED_MODULES.values())))
def test_experimental_module_paths_import(new_path: str) -> None:
    _import_experimental(new_path)


@pytest.mark.parametrize(
    ("_package", "name", "new_module"),
    REMOVED_PACKAGE_EXPORTS,
    ids=[f"{new}.{name}" for _, name, new in REMOVED_PACKAGE_EXPORTS],
)
def test_removed_names_resolve_from_experimental(
    _package: str, name: str, new_module: str
) -> None:
    module = _import_experimental(new_module)
    assert hasattr(module, name), f"{new_module} no longer provides {name}"
