"""Explicit layer boundaries and compatibility paths awaiting later retirement."""

from __future__ import annotations

import importlib
import sys
import warnings
from typing import Any

import pytest

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.errors import ConfigurationError

# --- core/agent.py: the planner is injected, the implicit path is removed ---


def test_implicit_planner_construction_requires_explicit_injection() -> None:
    config = AgentConfig(
        model_name="stub://planning",
        use_stub_model=True,
        enable_planning=True,
        planning_config={"max_steps": 3},
    )
    with pytest.raises(ConfigurationError, match="planning_manager_from_config"):
        MultiTurnAgent(config)


def test_injected_planner_needs_no_bridge() -> None:
    from stateset_agents.experimental.long_term_planning import (
        planning_manager_from_config,
    )

    manager = planning_manager_from_config({"max_steps": 2, "enabled": False})
    assert manager.config.enabled is True and manager.config.max_steps == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        agent = MultiTurnAgent(
            AgentConfig(model_name="stub://planning", use_stub_model=True),
            planning_manager=manager,
        )
    assert agent.planning_manager is manager


def test_core_does_not_load_the_planner_on_a_missing_dependency() -> None:
    _run_python(
        "import sys\n"
        "from stateset_agents.core.agent import AgentConfig, MultiTurnAgent\n"
        "from stateset_agents.core.errors import ConfigurationError\n"
        "name = 'stateset_agents.experimental.long_term_planning'\n"
        "assert name not in sys.modules\n"
        "try:\n"
        "    MultiTurnAgent(AgentConfig(model_name='stub://planning', enable_planning=True))\n"
        "except ConfigurationError as exc:\n"
        "    assert 'planning_manager_from_config' in str(exc)\n"
        "else:\n"
        "    raise AssertionError('missing planner silently accepted')\n"
        "assert name not in sys.modules\n"
    )


@pytest.mark.parametrize("bad", [{"max_stepz": 2}, "not-a-dict", 5])
def test_planning_factory_rejects_bad_config_with_a_typed_error(bad: Any) -> None:
    from stateset_agents.experimental.long_term_planning import (
        planning_manager_from_config,
    )

    with pytest.raises(ConfigurationError):
        planning_manager_from_config(bad)


def test_missing_planner_cannot_silently_disable_requested_planning() -> None:
    config = AgentConfig(model_name="stub://planning", use_stub_model=True)
    config.enable_planning = True
    config.planning_config = {"max_stepz": 2}
    with pytest.raises(ConfigurationError, match="explicit planning_manager"):
        MultiTurnAgent(config)


@pytest.mark.asyncio
async def test_gateway_injects_the_planner_itself() -> None:
    from stateset_agents.api.schemas import AgentConfigRequest
    from stateset_agents.api.services.agent_service import AgentService
    from stateset_agents.utils.security import SecurityMonitor

    service = AgentService(SecurityMonitor())
    request = AgentConfigRequest(
        model_name="stub://planning",
        enable_planning=True,
        planning_config={"max_steps": 4},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        agent_id = await service.create_agent(request, client_model=False)
    manager = service.agents[agent_id].planning_manager
    assert manager is not None and manager.config.max_steps == 4


# --- training/train.py: the offline trainer is registered, not imported ----


def test_offline_trainer_comes_from_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stateset_agents.training import offline_backend

    train_module = importlib.import_module("stateset_agents.training.train")

    sentinel = (type("Cfg", (), {}), type("Trainer", (), {}))
    monkeypatch.setattr(offline_backend, "_loader", lambda: sentinel)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert train_module._offline_trainer() == sentinel


def test_missing_offline_registration_raises_without_importing_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stateset_agents.training import offline_backend

    train_module = importlib.import_module("stateset_agents.training.train")

    monkeypatch.setattr(offline_backend, "_loader", None)
    monkeypatch.setitem(sys.modules, "stateset_agents.experimental.offline_rl", None)
    with pytest.raises(ConfigurationError, match="no offline trainer is registered"):
        train_module._offline_trainer()
    assert offline_backend._loader is None
    assert sys.modules["stateset_agents.experimental.offline_rl"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["offline", "hybrid", "auto"])
async def test_dataset_training_requires_registration_before_model_initialization(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from stateset_agents.training import offline_backend

    train_module = importlib.import_module("stateset_agents.training.train")
    monkeypatch.setattr(offline_backend, "_loader", None)
    agent = SimpleNamespace(model=None, initialize=AsyncMock(), tokenizer=None)
    environment = SimpleNamespace(reward_fn=None)
    with pytest.raises(ConfigurationError, match="no offline trainer is registered"):
        await train_module.train(
            agent, environment, training_mode=mode, dataset=[object()], num_episodes=1
        )
    agent.initialize.assert_not_awaited()


@pytest.mark.asyncio
async def test_hybrid_without_dataset_needs_only_the_online_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from stateset_agents.training import offline_backend

    train_module = importlib.import_module("stateset_agents.training.train")
    monkeypatch.setattr(offline_backend, "_loader", None)
    agent = SimpleNamespace()
    trainer = SimpleNamespace(
        initialize=AsyncMock(), train=AsyncMock(return_value=agent), add_callback=Mock()
    )
    monkeypatch.setattr(train_module, "MultiTurnGRPOTrainer", lambda **kwargs: trainer)
    result = await train_module.train(
        agent, SimpleNamespace(reward_fn=None), training_mode="hybrid", num_episodes=1
    )
    assert result is agent
    trainer.initialize.assert_awaited_once()
    trainer.train.assert_awaited_once()


def test_importing_the_component_registers_it_without_torch() -> None:
    import subprocess
    from pathlib import Path

    code = (
        "import sys\n"
        "import stateset_agents.experimental.offline_rl\n"
        "from stateset_agents.training import offline_backend\n"
        "assert offline_backend._loader is not None\n"
        "assert 'torch' not in sys.modules, 'registration imported torch'\n"
    )
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_uncertainty_wrapper_does_not_depend_on_the_experimental_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It used to probe-import experimental.bayesian_reward_model and silently
    skip the wrapper when that failed, though the wrapper never used it."""
    from stateset_agents.core.reward import HelpfulnessReward

    train_module = importlib.import_module("stateset_agents.training.train")

    monkeypatch.setitem(
        sys.modules, "stateset_agents.experimental.bayesian_reward_model", None
    )
    base = HelpfulnessReward()
    wrapped = train_module._wrap_with_uncertainty(base)
    assert isinstance(wrapped, train_module.UncertaintyWeightedReward)
    assert wrapped.base_reward is base


@pytest.mark.asyncio
async def test_auto_trainer_baseline_propagates_programmer_errors() -> None:
    """A TypeError in a baseline episode is a bug, not a failed episode."""
    from stateset_agents.training.train import AutoTrainer

    class BrokenEnvironment:
        async def run_episode(self, agent_fn: Any) -> Any:
            raise TypeError("run_episode() got an unexpected keyword argument")

    with pytest.raises(TypeError, match="unexpected keyword"):
        await AutoTrainer()._analyze_task(
            object(), BrokenEnvironment(), None  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_auto_trainer_baseline_still_tolerates_episode_failures() -> None:
    from stateset_agents.training.train import AutoTrainer

    class FlakyEnvironment:
        async def run_episode(self, agent_fn: Any) -> Any:
            raise RuntimeError("model server unavailable")

    analysis = await AutoTrainer()._analyze_task(
        object(), FlakyEnvironment(), None  # type: ignore[arg-type]
    )
    assert analysis["difficulty"] == "unknown"


# --- modules moved to keep the layers acyclic (v0.70.0) ---------------------

MOVED_MODULES = [
    ("stateset_agents.evaluation.agent_study", "stateset_agents.verticals.agent_study"),
    (
        "stateset_agents.evaluation.study_preflight",
        "stateset_agents.verticals.study_preflight",
    ),
    ("stateset_agents.evaluation.acceptance", "stateset_agents.verticals.acceptance"),
    (
        "stateset_agents.evaluation.industry",
        "stateset_agents.verticals.industry_evaluation",
    ),
    (
        "stateset_agents.data.industry_synthetic",
        "stateset_agents.verticals.industry_synthetic",
    ),
    ("stateset_agents.training.river_refund", "stateset_agents.verticals.river_refund"),
    (
        "stateset_agents.training.river_progress",
        "stateset_agents.evaluation.river_progress",
    ),
]


@pytest.mark.parametrize("old,new", MOVED_MODULES, ids=[m[0] for m in MOVED_MODULES])
def test_old_module_path_warns_and_aliases_the_new_module(
    old: str, new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = importlib.import_module(new)
    monkeypatch.delitem(sys.modules, old, raising=False)
    with pytest.warns(DeprecationWarning, match=r"removed in v0\.72\.0") as caught:
        module = importlib.import_module(old)
    assert new in str(caught[0].message)
    assert module is target
    assert sys.modules[old] is target


def test_new_module_paths_do_not_warn() -> None:
    code = (
        "import warnings\n"
        "warnings.simplefilter('error', DeprecationWarning)\n"
        + "".join(f"import {new}\n" for _, new in MOVED_MODULES)
        + "import stateset_agents.rewards.support_reward_audit\n"
        "import stateset_agents.evaluation.reward_audit\n"
    )
    _run_python(code)


def test_python_m_on_the_old_river_refund_path_still_runs() -> None:
    result = _run_python(
        "import runpy, sys\n"
        "sys.argv = ['river_refund', '--help']\n"
        "try:\n"
        "    runpy.run_module('stateset_agents.training.river_refund',"
        " run_name='__main__', alter_sys=True)\n"
        "except SystemExit as exc:\n"
        "    assert exc.code == 0, exc.code\n"
    )
    assert "--steps" in result.stdout


def _run_python(code: str) -> Any:
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


def test_support_reward_audit_uses_only_the_canonical_path() -> None:
    from stateset_agents.evaluation import reward_audit
    from stateset_agents.rewards import support_reward_audit

    with pytest.raises(AttributeError):
        _ = reward_audit.audit_support_rewards
    assert not hasattr(reward_audit, "main")
    assert callable(support_reward_audit.audit_support_rewards)
    # The generic audit stays in evaluation.
    assert callable(reward_audit.audit_reward)


def test_old_reward_audit_cli_explains_the_replacement() -> None:
    result = _run_python(
        "import runpy\n"
        "try:\n"
        "    runpy.run_module('stateset_agents.evaluation.reward_audit', run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    assert 'python -m stateset_agents.rewards.support_reward_audit' in str(exc)\n"
        "else:\n"
        "    raise AssertionError('retired CLI silently succeeded')\n"
    )
    assert result.returncode == 0


def test_trust_remote_code_policy_has_one_public_home() -> None:
    """remote.chat_repl used the private training.sft._trust_remote_code; the
    public core.model_presets.trust_remote_code_for must give the same answer
    until the SFT module delegates to it."""
    pytest.importorskip("transformers")
    from stateset_agents.core.model_presets import PRESETS, trust_remote_code_for
    from stateset_agents.training.sft import _trust_remote_code

    names = [preset.model_id for preset in PRESETS.values()]
    names += ["someone/unvetted-model", ""]
    for name in names:
        assert trust_remote_code_for(name) is _trust_remote_code(name)
    assert any(trust_remote_code_for(p.model_id) for p in PRESETS.values()) == any(
        p.trust_remote_code for p in PRESETS.values()
    )


def test_evaluation_rng_capture_matches_the_training_runtime() -> None:
    """evaluation.isolation captures RNGs with utils.reproducibility instead of
    training.runtime_state; both copies must stay interchangeable."""
    torch = pytest.importorskip("torch")
    import random

    import numpy as np

    from stateset_agents.training import runtime_state
    from stateset_agents.utils import reproducibility

    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    utils_state = reproducibility.capture_rng(torch)
    training_state = runtime_state.capture_rng(torch)
    assert utils_state["python"] == training_state["python"]
    assert utils_state["numpy"] == training_state["numpy"]
    assert torch.equal(utils_state["torch"], training_state["torch"])

    expected = (random.random(), float(np.random.rand()), torch.rand(1).item())
    for restore, state in (
        (reproducibility.restore_rng, training_state),
        (runtime_state.restore_rng, utils_state),
    ):
        restore(state, torch)
        assert (random.random(), float(np.random.rand()), torch.rand(1).item()) == (
            expected
        )
