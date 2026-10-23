"""Training-layer catches no longer swallow programmer errors.

Each test pins one narrowed handler: the expected failure is still absorbed
(with its logging), while a TypeError/AttributeError -- a bug -- propagates.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

orchestrator = importlib.import_module(
    "stateset_agents.training.advanced_training_orchestrator"
)


class _Stop(BaseException):
    """Ends a supervisor loop at its retry sleep."""


# --- advanced_training_orchestrator ------------------------------------------


def _orchestrator(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(
        orchestrator.ResourceManager, "_detect_resources", lambda s: None
    )
    monkeypatch.setattr(orchestrator, "get_state_service", SimpleNamespace)
    monkeypatch.setattr(orchestrator, "get_monitoring_service", SimpleNamespace)
    return orchestrator.AdvancedTrainingOrchestrator(
        start_background_tasks=False, enable_experiment_tracking=False
    )


async def _stop_at_sleep(_seconds: float) -> None:
    raise _Stop


@pytest.mark.asyncio
async def test_orchestration_loop_propagates_programmer_errors(monkeypatch) -> None:
    service = _orchestrator(monkeypatch)
    monkeypatch.setattr(orchestrator.asyncio, "sleep", _stop_at_sleep)
    service._cleanup_completed_tasks = AsyncMock(side_effect=TypeError("bad job"))
    with pytest.raises(TypeError, match="bad job"):
        await service._orchestration_loop()


@pytest.mark.asyncio
async def test_orchestration_loop_still_retries_runtime_failures(monkeypatch) -> None:
    service = _orchestrator(monkeypatch)
    monkeypatch.setattr(orchestrator.asyncio, "sleep", _stop_at_sleep)
    service._cleanup_completed_tasks = AsyncMock(side_effect=OSError("state down"))
    with pytest.raises(_Stop):  # logged, then the retry sleep
        await service._orchestration_loop()


def test_resource_probe_bug_is_not_reported_as_missing_capacity(monkeypatch) -> None:
    def broken(path: Any) -> Any:
        raise TypeError("disk_usage() got an unexpected argument")

    monkeypatch.setattr(orchestrator.shutil, "disk_usage", broken)
    with pytest.raises(TypeError, match="unexpected argument"):
        orchestrator.ResourceManager()


# --- auto_research.checkpoint_manager ----------------------------------------


def test_checkpoint_restore_propagates_model_bugs(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from stateset_agents.training.auto_research.checkpoint_manager import (
        CheckpointManager,
    )

    torch.save({"w": torch.zeros(1)}, tmp_path / "model_state.pt")

    class Model:
        def load_state_dict(self, state: Any, strict: bool = True) -> None:
            raise AttributeError("'Model' object has no attribute 'layers'")

    agent = SimpleNamespace(model=Model())
    manager = CheckpointManager(tmp_path / "out")
    with pytest.raises(AttributeError, match="layers"):
        manager._restore_agent_state(agent, tmp_path)


def test_checkpoint_restore_still_falls_through_unreadable_state(tmp_path) -> None:
    pytest.importorskip("torch")
    from stateset_agents.training.auto_research.checkpoint_manager import (
        CheckpointManager,
    )

    (tmp_path / "model_state.pt").write_bytes(b"not a checkpoint")
    agent = SimpleNamespace(model=SimpleNamespace(load_state_dict=lambda *a, **k: 0))
    manager = CheckpointManager(tmp_path / "out")
    assert manager._restore_agent_state(agent, tmp_path) is False


def test_checkpoint_save_propagates_save_pretrained_bugs(tmp_path) -> None:
    from stateset_agents.training.auto_research.checkpoint_manager import (
        CheckpointManager,
    )

    def save_pretrained(path: str) -> None:
        raise AttributeError("'NoneType' object has no attribute 'save'")

    agent = SimpleNamespace(model=SimpleNamespace(save_pretrained=save_pretrained))
    with pytest.raises(AttributeError, match="no attribute 'save'"):
        CheckpointManager(tmp_path)._save_agent_state(agent, tmp_path)


# --- auto_research.experiment_loop -------------------------------------------


def _loop(tmp_path) -> Any:
    from stateset_agents.training.auto_research.config import AutoResearchConfig
    from stateset_agents.training.auto_research.experiment_loop import (
        AutoResearchLoop,
    )
    from stateset_agents.training.auto_research.proposer import PerturbationProposer

    config = AutoResearchConfig(
        time_budget=10,
        max_experiments=1,
        output_dir=str(tmp_path),
        search_space_name="quick",
        eval_episodes=1,
        save_checkpoints=False,
    )
    agent = AsyncMock()
    agent.model = None
    loop = AutoResearchLoop(
        agent=agent,
        environment=MagicMock(),
        eval_scenarios=[{"id": "s"}],
        reward_fn=MagicMock(),
        config=config,
        proposer=PerturbationProposer(search_space=None),
        baseline_params={"learning_rate": 1e-5},
    )
    return loop


@pytest.mark.asyncio
async def test_baseline_validation_errors_are_not_swallowed(
    tmp_path, monkeypatch
) -> None:
    from stateset_agents.training.auto_research import search_spaces

    loop = _loop(tmp_path)

    def broken(params: Any, space: Any) -> list[str]:
        raise TypeError("'<' not supported")

    monkeypatch.setattr(search_spaces, "validate_params_against_space", broken)
    monkeypatch.setattr(loop, "_init_wandb", lambda: None)
    monkeypatch.setattr(loop, "_try_resume", lambda: (_ for _ in ()).throw(_Stop()))
    with pytest.raises(TypeError, match="not supported"):
        await loop.run()


def test_gpu_cleanup_propagates_programmer_errors(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from stateset_agents.training.auto_research.experiment_loop import (
        AutoResearchLoop,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def broken() -> None:
        raise TypeError("empty_cache() takes no arguments")

    monkeypatch.setattr(torch.cuda, "empty_cache", broken)
    with pytest.raises(TypeError, match="no arguments"):
        AutoResearchLoop._cleanup_gpu()


def test_wandb_log_failures_are_visible(tmp_path, caplog) -> None:
    from stateset_agents.training.auto_research.experiment_tracker import (
        ExperimentRecord,
    )

    loop = _loop(tmp_path)
    loop._wandb = SimpleNamespace(log=MagicMock(side_effect=RuntimeError("api down")))
    record = ExperimentRecord(
        experiment_id="e",
        params={},
        metrics={},
        objective_value=0.5,
        training_time=1.0,
        status="keep",
    )
    with caplog.at_level(logging.WARNING):
        loop._log_wandb(record)
    assert "wandb log failed: api down" in caplog.text


def test_report_failures_are_logged_as_warnings(tmp_path, monkeypatch, caplog) -> None:
    from stateset_agents.training.auto_research import analysis

    loop = _loop(tmp_path)
    monkeypatch.setattr(type(loop.tracker), "num_experiments", property(lambda s: 6))

    def broken(*args: Any, **kwargs: Any) -> str:
        raise ValueError("no records")

    monkeypatch.setattr(analysis, "generate_report", broken)
    with caplog.at_level(logging.WARNING):
        loop._print_analysis()
    assert "Analysis generation failed: no records" in caplog.text


# --- auto_research.experiment_tracker ----------------------------------------


def test_tracker_analysis_propagates_programmer_errors(tmp_path, monkeypatch) -> None:
    from stateset_agents.training.auto_research import analysis
    from stateset_agents.training.auto_research.experiment_tracker import (
        ExperimentRecord,
        ExperimentTracker,
    )

    tracker = ExperimentTracker(output_dir=tmp_path)
    tracker.record(
        ExperimentRecord(
            experiment_id="e",
            params={},
            metrics={},
            objective_value=0.5,
            training_time=1.0,
            status="keep",
        )
    )

    def broken(records: Any, direction: str) -> Any:
        raise AttributeError("'ExperimentRecord' object has no attribute 'value'")

    monkeypatch.setattr(analysis, "compute_convergence_curve", broken)
    with pytest.raises(AttributeError, match="'value'"):
        tracker.get_analysis()


# --- auto_research proposers -------------------------------------------------


def _fake_optuna(monkeypatch: pytest.MonkeyPatch) -> Any:
    class OptunaError(Exception):
        pass

    module = types.ModuleType("optuna")
    module.exceptions = SimpleNamespace(OptunaError=OptunaError)  # type: ignore[attr-defined]
    module.trial = SimpleNamespace(TrialState=SimpleNamespace(FAIL="FAIL"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "optuna", module)
    return module


def _bayesian(study: Any) -> Any:
    from stateset_agents.training.auto_research.proposer import BayesianProposer

    proposer = object.__new__(BayesianProposer)
    proposer._study = study
    proposer._current_trial = object()
    proposer._trial_count = 0
    return proposer


def test_optuna_report_absorbs_optuna_errors(monkeypatch, caplog) -> None:
    optuna = _fake_optuna(monkeypatch)
    study = SimpleNamespace(
        tell=MagicMock(side_effect=optuna.exceptions.OptunaError("finished"))
    )
    proposer = _bayesian(study)
    with caplog.at_level(logging.WARNING):
        proposer.report_result(0.5)
    assert "Failed to report result to Optuna" in caplog.text
    assert proposer._trial_count == 1 and proposer._current_trial is None


def test_optuna_report_propagates_programmer_errors(monkeypatch) -> None:
    _fake_optuna(monkeypatch)
    study = SimpleNamespace(tell=MagicMock(side_effect=TypeError("bad tell")))
    proposer = _bayesian(study)
    with pytest.raises(TypeError, match="bad tell"):
        proposer.report_result(0.5)
    assert proposer._trial_count == 1 and proposer._current_trial is None


def _llm_proposer(monkeypatch: pytest.MonkeyPatch, reply: Any) -> Any:
    from stateset_agents.training.auto_research import llm_proposer

    proposer = object.__new__(llm_proposer.LLMProposer)
    proposer.search_space = SimpleNamespace(dimensions=[])
    monkeypatch.setattr(proposer, "_ensure_client", lambda: None, raising=False)
    monkeypatch.setattr(proposer, "_call_llm", reply, raising=False)
    fallback = MagicMock(return_value=({"fallback": True}, "fallback"))
    monkeypatch.setattr(proposer, "_fallback_propose", fallback, raising=False)
    return proposer


def test_llm_proposer_falls_back_on_api_errors(monkeypatch) -> None:
    class APIConnectionError(Exception):  # SDK errors are plain Exceptions
        pass

    proposer = _llm_proposer(
        monkeypatch, MagicMock(side_effect=APIConnectionError("offline"))
    )
    assert proposer.propose({}, []) == ({"fallback": True}, "fallback")


def test_llm_proposer_falls_back_on_unparseable_replies(monkeypatch) -> None:
    proposer = _llm_proposer(monkeypatch, MagicMock(return_value="no json here"))
    assert proposer.propose({}, []) == ({"fallback": True}, "fallback")


def test_llm_proposer_propagates_parser_bugs(monkeypatch) -> None:
    from stateset_agents.training.auto_research import llm_proposer

    proposer = _llm_proposer(monkeypatch, MagicMock(return_value="{}"))

    def broken(text: str) -> dict[str, Any]:
        raise TypeError("expected str")

    monkeypatch.setattr(llm_proposer, "_extract_json", broken)
    with pytest.raises(TypeError, match="expected str"):
        proposer.propose({}, [])


# --- sft ---------------------------------------------------------------------


def test_sft_judge_factory_bugs_propagate(monkeypatch) -> None:
    from stateset_agents.training import sft

    def broken(name: str) -> Any:
        raise TypeError("create_domain_reward() missing 1 required argument")

    monkeypatch.setattr(sft, "_create_domain_reward", broken)
    with pytest.raises(TypeError, match="missing 1 required"):
        sft.judge_completion("customer_support", "q", "a")


def test_sft_cuda_diagnostics_still_tolerate_cuda_errors(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from stateset_agents.training import sft

    def unavailable() -> bool:
        raise RuntimeError("CUDA driver initialization failed")

    monkeypatch.setattr(torch.cuda, "is_available", unavailable)
    sft.log_cuda_memory_per_device()  # diagnostics only: no exception
