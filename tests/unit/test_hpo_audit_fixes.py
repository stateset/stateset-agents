"""HPO trials must train, evaluate, and score for real and in isolation.

Regression tests for: GRPO HPO trials that never initialized the trainer and
read the metric from ``train()``'s return value (the agent), so every trial
scored -inf; trials that trained on top of each other's weights; Ray/W&B
reporting -inf for failed trials even when minimizing; Optuna pruning that
was never wired; ``log_scale`` ignored for float dimensions; and auto_research
treating a missing objective metric as 0.0.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.config import TrainingConfig  # noqa: E402
from stateset_agents.training.hpo import (  # noqa: E402
    grpo_hpo_trainer,
    optuna_backend,
    ray_tune_backend,
    wandb_backend,
)
from stateset_agents.training.hpo.base import (  # noqa: E402
    SearchDimension,
    SearchSpace,
    SearchSpaceType,
)
from stateset_agents.training.hpo.config import HPOConfig  # noqa: E402

# --- GRPO HPO objective ---------------------------------------------------------


class _FakeTrainer:
    """Stands in for MultiTurnGRPOTrainer: requires initialize() before train(),
    mutates the agent's weights while training, returns the agent (like the
    real trainer), and evaluates to configurable metrics."""

    instances: list[_FakeTrainer] = []
    eval_metrics: dict[str, float] = {"eval_reward": 0.7, "eval_num_episodes": 2.0}
    intermediate: list[dict[str, float]] = []

    def __init__(self, agent, environment, reward_fn, config, callbacks=None):
        self.agent = agent
        self.config = config
        self.callbacks = list(callbacks or [])
        self.optimizer = None
        self.global_step = 0
        self.start_weight: Any = None
        _FakeTrainer.instances.append(self)

    async def initialize(self):
        self.optimizer = object()

    async def train(self):
        if self.optimizer is None:
            raise AttributeError("'NoneType' object has no attribute 'param_groups'")
        weight = self.agent.model.weight
        self.start_weight = weight.detach().clone()
        for step, metrics in enumerate(self.intermediate, start=1):
            self.global_step = step
            for callback in self.callbacks:
                await callback.on_evaluation_end(metrics)
        with torch.no_grad():
            weight.add_(1.0)
        self.global_step = max(self.global_step, 1)
        return self.agent

    def _get_eval_scenarios(self):
        return [{"id": "eval"}]

    async def evaluate(self, eval_scenarios, num_eval_episodes=10):
        return dict(self.eval_metrics)


@pytest.fixture
def fake_trainer(monkeypatch):
    _FakeTrainer.instances = []
    _FakeTrainer.eval_metrics = {"eval_reward": 0.7, "eval_num_episodes": 2.0}
    _FakeTrainer.intermediate = []
    monkeypatch.setattr(grpo_hpo_trainer, "MultiTurnGRPOTrainer", _FakeTrainer)
    return _FakeTrainer


def _hpo_trainer(tmp: str, **hpo_kwargs):
    torch.manual_seed(0)
    agent = SimpleNamespace(model=torch.nn.Linear(2, 2), initialize=AsyncMock())
    return grpo_hpo_trainer.GRPOHPOTrainer(
        agent=agent,
        environment=MagicMock(),
        reward_function=MagicMock(),
        base_config=TrainingConfig(num_episodes=5, seed=7, output_dir=tmp),
        hpo_config=HPOConfig(
            backend="optuna",
            search_space_name="grpo",
            output_dir=Path(tmp),
            **hpo_kwargs,
        ),
    )


@pytest.mark.asyncio
async def test_trial_initializes_trains_and_scores_from_evaluation(fake_trainer):
    with tempfile.TemporaryDirectory() as tmp:
        hpo = _hpo_trainer(tmp)
        value = await hpo._objective_function({"learning_rate": 1e-5})
    assert value == pytest.approx(0.7)
    assert fake_trainer.instances[0].optimizer is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "direction,worst", [("maximize", -math.inf), ("minimize", math.inf)]
)
async def test_missing_objective_metric_scores_worst_not_zero(
    fake_trainer, direction, worst
):
    fake_trainer.eval_metrics = {"eval_reward": 0.7, "eval_num_episodes": 2.0}
    with tempfile.TemporaryDirectory() as tmp:
        hpo = _hpo_trainer(tmp, objective_metric="accuracy", direction=direction)
        assert await hpo._objective_function({}) == worst


@pytest.mark.asyncio
async def test_trial_without_completed_evaluation_scores_worst(fake_trainer):
    fake_trainer.eval_metrics = {"eval_reward": 0.0, "eval_num_episodes": 0.0}
    with tempfile.TemporaryDirectory() as tmp:
        hpo = _hpo_trainer(tmp)
        assert await hpo._objective_function({}) == -math.inf


@pytest.mark.asyncio
async def test_trials_start_from_the_same_weights_with_distinct_seeds(fake_trainer):
    with tempfile.TemporaryDirectory() as tmp:
        hpo = _hpo_trainer(tmp)
        initial = hpo.agent.model.weight.detach().clone()
        await hpo._objective_function({})
        await hpo._objective_function({})
        hpo.hpo_summary = SimpleNamespace(best_params={})
        await hpo.train_with_best_params()
    starts = [t.start_weight for t in fake_trainer.instances]
    assert len(starts) == 3
    for start in starts:
        assert torch.equal(start, initial)
    assert [t.config.seed for t in fake_trainer.instances] == [7, 8, 7]


@pytest.mark.asyncio
async def test_intermediate_evaluations_reach_the_pruning_reporter(fake_trainer):
    fake_trainer.intermediate = [{"eval_reward": 0.1}, {"eval_reward": 0.3}]
    reports: list[tuple[float, int]] = []
    with tempfile.TemporaryDirectory() as tmp:
        hpo = _hpo_trainer(tmp)
        value = await hpo._objective_function(
            {}, report=lambda v, s: reports.append((v, s))
        )
    assert reports == [(0.1, 1), (0.3, 2)]
    assert value == pytest.approx(0.7)


# --- Optuna backend -------------------------------------------------------------


class _Pruned(Exception):
    pass


class _FakeTrial:
    def __init__(self, prune_after: int | None = None):
        self.number = 0
        self.reports: list[tuple[float, int]] = []
        self.suggested: list[tuple[str, dict[str, Any]]] = []
        self._prune_after = prune_after

    def suggest_float(self, name, low, high, log=False):
        self.suggested.append((name, {"log": log}))
        return low

    def suggest_int(self, name, low, high, log=False):
        self.suggested.append((name, {"log": log}))
        return low

    def report(self, value, step):
        self.reports.append((value, step))

    def should_prune(self):
        return self._prune_after is not None and len(self.reports) >= self._prune_after


def _space(log_scale: bool = True) -> SearchSpace:
    return SearchSpace(
        dimensions=[
            SearchDimension(
                "lr", SearchSpaceType.FLOAT, low=1e-6, high=1e-3, log_scale=log_scale
            )
        ]
    )


@pytest.fixture
def fake_optuna(monkeypatch):
    monkeypatch.setattr(
        optuna_backend, "optuna", SimpleNamespace(TrialPruned=_Pruned), raising=False
    )


def test_optuna_honours_log_scale_for_float_dimensions(fake_optuna):
    backend = optuna_backend.OptunaBackend(search_space=_space())
    trial = _FakeTrial()
    backend._objective_wrapper(trial, lambda params: 1.0)
    assert trial.suggested == [("lr", {"log": True})]


def test_optuna_reports_intermediate_values_and_prunes(fake_optuna):
    backend = optuna_backend.OptunaBackend(search_space=_space())
    trial = _FakeTrial(prune_after=2)

    def objective(params, report):
        for step in (1, 2, 3):
            report(0.1 * step, step)
        return 1.0

    with pytest.raises(_Pruned):
        backend._objective_wrapper(trial, objective)
    assert trial.reports == [(pytest.approx(0.1), 1), (pytest.approx(0.2), 2)]
    assert backend.results[-1].status == "pruned"


def test_optuna_objectives_without_a_report_parameter_still_work(fake_optuna):
    backend = optuna_backend.OptunaBackend(search_space=_space())
    assert backend._objective_wrapper(_FakeTrial(), lambda params: 0.5) == 0.5


# --- Ray Tune / W&B failed trials and log scale -----------------------------------


class _FakeTune:
    def __init__(self):
        self.reported: list[dict[str, float]] = []

    def get_trial_id(self):
        return "t0"

    def report(self, **metrics):
        self.reported.append(metrics)

    def uniform(self, low, high):
        return ("uniform", low, high)

    def loguniform(self, low, high):
        return ("loguniform", low, high)

    def randint(self, low, high):
        return ("randint", low, high)

    def lograndint(self, low, high):
        return ("lograndint", low, high)


def _failing(params):
    raise RuntimeError("boom")


@pytest.mark.parametrize(
    "direction,worst", [("maximize", -math.inf), ("minimize", math.inf)]
)
def test_ray_failed_trial_reports_the_worst_value_for_the_direction(
    monkeypatch, direction, worst
):
    tune = _FakeTune()
    monkeypatch.setattr(ray_tune_backend, "tune", tune)
    backend = ray_tune_backend.RayTuneBackend(
        search_space=_space(), objective_metric="loss", direction=direction
    )
    backend._build_trainable(_failing)({"lr": 1e-4})
    assert tune.reported == [{"loss": worst}]
    assert backend.results[-1].best_metric == worst
    assert backend.results[-1].status == "failed"


def test_ray_honours_log_scale(monkeypatch):
    monkeypatch.setattr(ray_tune_backend, "tune", _FakeTune())
    monkeypatch.setattr(ray_tune_backend, "_require_ray", lambda: None)
    backend = ray_tune_backend.RayTuneBackend(search_space=_space())
    assert backend._convert_dimension(_space().dimensions[0])[0] == "loguniform"
    int_dim = SearchDimension("n", SearchSpaceType.INT, low=2, high=64, log_scale=True)
    assert backend._convert_dimension(int_dim) == ("lograndint", 2, 65)


class _FakeWandb:
    def __init__(self):
        self.logged: list[dict[str, float]] = []
        self.config: dict[str, Any] = {"lr": 1e-4}

    def init(self, **kwargs):
        return SimpleNamespace(id="run0")

    def log(self, metrics):
        self.logged.append(metrics)

    def finish(self):
        pass


@pytest.mark.parametrize(
    "direction,worst", [("maximize", -math.inf), ("minimize", math.inf)]
)
def test_wandb_failed_trial_logs_the_worst_value_for_the_direction(
    monkeypatch, direction, worst
):
    fake = _FakeWandb()
    monkeypatch.setattr(wandb_backend, "wandb", fake)
    backend = wandb_backend.WandBSweepsBackend(
        search_space=_space(), objective_metric="loss", direction=direction
    )
    backend._build_run_trial(_failing)()
    assert fake.logged == [{"loss": worst}]
    assert backend.results[-1].best_metric == worst


def test_wandb_honours_log_scale():
    backend = wandb_backend.WandBSweepsBackend(search_space=_space())
    assert backend._convert_dimension(_space().dimensions[0]) == {
        "distribution": "log_uniform_values",
        "min": 1e-6,
        "max": 1e-3,
    }
    linear = backend._convert_dimension(_space(log_scale=False).dimensions[0])
    assert linear["distribution"] == "uniform"


# --- auto_research ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_research_missing_baseline_metric_is_a_crash():
    from stateset_agents.training.auto_research.experiment_loop import (
        AutoResearchLoop,
    )

    loop = AutoResearchLoop.__new__(AutoResearchLoop)
    loop.config = SimpleNamespace(objective_metric="eval_accuracy")
    loop.baseline_params = {}
    loop._current_experiment_timeout = lambda: 1.0
    loop._evaluate = AsyncMock(return_value={"eval_reward": 0.4})
    loop._cleanup_gpu = MagicMock()
    loop._record_crash = MagicMock()
    loop.checkpoint_mgr = MagicMock()

    await loop._run_baseline()

    loop._record_crash.assert_called_once()
    assert "eval_accuracy" in loop._record_crash.call_args.args[2]
    loop.checkpoint_mgr.save_best.assert_not_called()


@pytest.mark.asyncio
async def test_auto_research_missing_experiment_metric_is_a_crash():
    from stateset_agents.training.auto_research.experiment_loop import (
        AutoResearchLoop,
    )

    loop = AutoResearchLoop.__new__(AutoResearchLoop)
    loop.config = SimpleNamespace(objective_metric="eval_accuracy")
    loop.baseline_params = {}
    loop.tracker = MagicMock(best_record=None)
    loop.proposer = MagicMock()
    loop.proposer.propose.return_value = ({"lr": 0.01}, "candidate")
    loop._current_experiment_timeout = lambda: 1.0
    loop._execute_experiment = AsyncMock(return_value={"eval_reward": 0.9})
    loop._cleanup_gpu = MagicMock()
    loop._record_crash = MagicMock()
    loop._report_to_proposer = MagicMock()
    loop.checkpoint_mgr = MagicMock()
    loop.agent = MagicMock()

    await loop._run_experiment("exp_0001")

    loop._record_crash.assert_called_once()
    assert "eval_accuracy" in loop._record_crash.call_args.args[2]
    loop.checkpoint_mgr.save_best.assert_not_called()
    loop._report_to_proposer.assert_called_once_with(0.0, crashed=True)
