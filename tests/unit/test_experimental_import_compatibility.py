"""Published v0.74.0 imports and experimental imports share their implementation."""

from __future__ import annotations

import importlib
import pickle
from types import ModuleType

import pytest

#: Published module name -> canonical experimental module.
COMPAT_MODULES = {
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

COMPAT_PACKAGE_EXPORTS = [
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


def _import_experimental(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        if (exc.name or "").split(".")[0] in {"torch", "transformers", "numpy"}:
            pytest.skip(f"{module_name} needs an optional extra: {exc}")
        raise


@pytest.mark.parametrize(("published", "canonical"), COMPAT_MODULES.items())
def test_module_alias_identity(published: str, canonical: str) -> None:
    assert _import_experimental(published) is _import_experimental(canonical)


@pytest.mark.parametrize(("pkg", "name", "canonical"), COMPAT_PACKAGE_EXPORTS)
def test_published_package_exports(pkg: str, name: str, canonical: str) -> None:
    package = importlib.import_module(pkg)
    assert name in package.__all__
    assert getattr(package, name) is getattr(_import_experimental(canonical), name)


def test_checkpoint_global_from_published_module() -> None:
    canonical = _import_experimental(
        "stateset_agents.experimental.offline_rl.algorithms"
    )
    # Protocol 0 GLOBAL records the module and symbol names used by an old writer.
    record = b"cstateset_agents.training.offline_rl_algorithms\nIQLConfig\n."
    assert pickle.loads(record) is canonical.IQLConfig  # nosec B301
