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
    with pytest.warns(DeprecationWarning, match=r"removed in v0\.71\.0") as caught:
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


def test_support_reward_audit_moved_with_a_dated_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stateset_agents.evaluation import reward_audit
    from stateset_agents.rewards import support_reward_audit

    with pytest.warns(DeprecationWarning, match=r"removed in v0\.71\.0"):
        alias = reward_audit.audit_support_rewards
    assert alias is support_reward_audit.audit_support_rewards

    called: list[bool] = []
    monkeypatch.setattr(support_reward_audit, "main", lambda: called.append(True))
    with pytest.warns(DeprecationWarning, match="support_reward_audit"):
        reward_audit.main()
    assert called == [True]
    # The generic audit stays in evaluation.
    assert callable(reward_audit.audit_reward)


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
