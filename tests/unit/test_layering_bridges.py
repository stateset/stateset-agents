"""Behaviour of the layering fixes: stable code no longer reaches into the
experimental layer implicitly (see ``LAYERING_ALLOWLIST`` in test_layering.py).
"""

from __future__ import annotations

import importlib
import sys
import warnings
from typing import Any

import pytest

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.errors import ConfigurationError

# --- core/agent.py: the planner is injected, the implicit path is dated -----


def test_implicit_planner_construction_is_a_dated_deprecation() -> None:
    config = AgentConfig(
        model_name="stub://planning",
        use_stub_model=True,
        enable_planning=True,
        planning_config={"max_steps": 3},
    )
    with pytest.warns(DeprecationWarning, match=r"removed in v0\.72\.0") as caught:
        agent = MultiTurnAgent(config)
    assert "planning_manager_from_config" in str(caught[0].message)
    assert caught[0].filename == __file__
    # Behaviour is unchanged until the removal release.
    assert agent.planning_manager is not None
    assert agent.planning_manager.config.max_steps == 3


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


@pytest.mark.parametrize("bad", [{"max_stepz": 2}, "not-a-dict", 5])
def test_planning_factory_rejects_bad_config_with_a_typed_error(bad: Any) -> None:
    from stateset_agents.experimental.long_term_planning import (
        planning_manager_from_config,
    )

    with pytest.raises(ConfigurationError):
        planning_manager_from_config(bad)


def test_bad_implicit_planning_config_still_leaves_planning_off() -> None:
    config = AgentConfig(model_name="stub://planning", use_stub_model=True)
    config.enable_planning = True
    config.planning_config = {"max_stepz": 2}
    with pytest.warns(DeprecationWarning):
        agent = MultiTurnAgent(config)
    assert agent.planning_manager is None


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


def test_implicit_offline_import_is_a_dated_deprecation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from stateset_agents.training import offline_backend

    train_module = importlib.import_module("stateset_agents.training.train")

    monkeypatch.setattr(offline_backend, "_loader", None)
    # Force the bridge to import the package afresh so it re-registers.
    for name in list(sys.modules):
        if name == "stateset_agents.experimental.offline_rl":
            monkeypatch.delitem(sys.modules, name)
    with pytest.warns(DeprecationWarning, match=r"removed in v0\.72\.0"):
        config_cls, trainer_cls = train_module._offline_trainer()
    assert (config_cls.__name__, trainer_cls.__name__) == (
        "OfflineGRPOConfig",
        "OfflineGRPOTrainer",
    )


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
