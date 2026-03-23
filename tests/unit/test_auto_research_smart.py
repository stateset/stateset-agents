"""
Tests for smart proposers, early abort, and analysis features.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stateset_agents.training.auto_research.analysis import (
    compute_convergence_curve,
    compute_diminishing_returns,
    compute_parameter_importance,
    generate_report,
)
from stateset_agents.training.auto_research.config import AutoResearchConfig
from stateset_agents.training.auto_research.early_abort import EarlyAbortCallback
from stateset_agents.training.auto_research.experiment_loop import (
    AutoResearchLoop,
    run_auto_research,
)
from stateset_agents.training.auto_research.experiment_tracker import ExperimentRecord
from stateset_agents.training.auto_research.proposer import (
    AdaptivePerturbationProposer,
    SmartPerturbationProposer,
    create_proposer,
)
from stateset_agents.training.auto_research.search_spaces import (
    create_quick_search_space,
)


EVAL_SCENARIOS = [
    {"topic": "test", "context": "Test.", "user_responses": ["Hi"]},
]


def _make_record(exp_id, params, objective, status="keep", desc=""):
    return ExperimentRecord(
        experiment_id=exp_id,
        params=params,
        metrics={"eval_reward": objective},
        objective_value=objective,
        training_time=10.0,
        status=status,
        description=desc,
    )


# ---------------------------------------------------------------------------
# Early abort callback
# ---------------------------------------------------------------------------


class TestEarlyAbortCallback:
    def test_no_abort_on_normal_training(self):
        cb = EarlyAbortCallback()
        for i in range(100):
            cb.on_step_end(step=i, metrics={"loss": 1.0 - i * 0.005})
        assert cb.should_abort is False

    def test_abort_on_nan_loss(self):
        cb = EarlyAbortCallback(nan_patience=2)
        cb.on_step_end(step=1, metrics={"loss": 0.5})
        cb.on_step_end(step=2, metrics={"loss": float("nan")})
        assert cb.should_abort is False  # 1 NaN, patience is 2
        cb.on_step_end(step=3, metrics={"loss": float("nan")})
        assert cb.should_abort is True
        assert "NaN" in cb.abort_reason

    def test_abort_on_loss_explosion(self):
        cb = EarlyAbortCallback(loss_explosion_factor=10.0)
        cb.on_step_end(step=1, metrics={"loss": 0.5})
        cb.on_step_end(step=2, metrics={"loss": 100.0})  # 200x increase
        assert cb.should_abort is True
        assert "explosion" in cb.abort_reason

    def test_abort_on_plateau(self):
        cb = EarlyAbortCallback(
            plateau_window=10,
            plateau_min_steps=15,
            plateau_min_relative_change=0.01,
        )
        # Feed constant loss for enough steps
        for i in range(30):
            cb.on_step_end(step=i, metrics={"loss": 0.5})
        assert cb.should_abort is True
        assert "plateau" in cb.abort_reason.lower()

    def test_no_abort_on_improving_loss(self):
        cb = EarlyAbortCallback(
            plateau_window=10,
            plateau_min_steps=15,
            plateau_min_relative_change=0.001,
        )
        for i in range(30):
            # Decreasing loss = good
            cb.on_step_end(step=i, metrics={"loss": 1.0 - i * 0.02})
        assert cb.should_abort is False

    def test_reset_on_train_start(self):
        cb = EarlyAbortCallback(nan_patience=1)
        cb.on_step_end(step=1, metrics={"loss": float("nan")})
        assert cb.should_abort is True

        cb.on_train_start()
        assert cb.should_abort is False
        assert cb.step_count == 0

    def test_ignores_missing_loss(self):
        cb = EarlyAbortCallback()
        cb.on_step_end(step=1, metrics={"reward": 0.5})  # No loss key
        cb.on_step_end(step=2, metrics={})
        cb.on_step_end(step=3)  # No metrics
        assert cb.should_abort is False

    def test_inf_triggers_abort(self):
        cb = EarlyAbortCallback(nan_patience=1)
        cb.on_step_end(step=1, metrics={"loss": float("inf")})
        assert cb.should_abort is True


# ---------------------------------------------------------------------------
# Adaptive perturbation proposer
# ---------------------------------------------------------------------------


class TestAdaptivePerturbationProposer:
    def test_adapts_factor_downward_on_success(self):
        space = create_quick_search_space()
        proposer = AdaptivePerturbationProposer(
            space, initial_factor=0.3, adaptation_window=3
        )
        current = {"learning_rate": 1e-5, "num_generations": 4,
                    "temperature": 0.7, "lora_r": 8}

        # History with high keep rate
        history = [
            {"params": {}, "objective": 0.5, "status": "keep"},
            {"params": {}, "objective": 0.6, "status": "keep"},
            {"params": {}, "objective": 0.7, "status": "keep"},
        ]
        proposer.propose(current, history)
        assert proposer.perturbation_factor < 0.3

    def test_adapts_factor_upward_on_failure(self):
        space = create_quick_search_space()
        proposer = AdaptivePerturbationProposer(
            space, initial_factor=0.2, adaptation_window=3
        )
        current = {"learning_rate": 1e-5, "num_generations": 4,
                    "temperature": 0.7, "lora_r": 8}

        history = [
            {"params": {}, "objective": 0.3, "status": "discard"},
            {"params": {}, "objective": 0.2, "status": "discard"},
            {"params": {}, "objective": 0.1, "status": "discard"},
        ]
        proposer.propose(current, history)
        assert proposer.perturbation_factor > 0.2

    def test_factor_clamped_to_bounds(self):
        space = create_quick_search_space()
        proposer = AdaptivePerturbationProposer(
            space, initial_factor=0.05, min_factor=0.05, max_factor=0.5
        )
        current = {"learning_rate": 1e-5, "num_generations": 4,
                    "temperature": 0.7, "lora_r": 8}

        # Many successes — factor tries to go below min
        history = [{"params": {}, "objective": 0.9, "status": "keep"}] * 20
        for _ in range(10):
            proposer.propose(current, history)
        assert proposer.perturbation_factor >= 0.05

    def test_create_proposer_adaptive(self):
        space = create_quick_search_space()
        p = create_proposer("adaptive", space)
        assert isinstance(p, AdaptivePerturbationProposer)


# ---------------------------------------------------------------------------
# Smart perturbation proposer
# ---------------------------------------------------------------------------


class TestSmartPerturbationProposer:
    def test_exploration_phase(self):
        """During exploration phase, behaves like standard perturbation."""
        space = create_quick_search_space()
        proposer = SmartPerturbationProposer(space, exploration_phase=10)
        current = {"learning_rate": 1e-5, "num_generations": 4,
                    "temperature": 0.7, "lora_r": 8}

        # Only 3 history items — should still be in exploration
        history = [
            {"params": {"learning_rate": 1e-5}, "objective": 0.5, "status": "keep"},
            {"params": {"learning_rate": 2e-5}, "objective": 0.6, "status": "keep"},
            {"params": {"learning_rate": 3e-5}, "objective": 0.4, "status": "discard"},
        ]
        params, desc = proposer.propose(current, history)
        assert "explore" in desc

    def test_smart_phase_after_exploration(self):
        """After exploration phase, uses importance-weighted selection."""
        space = create_quick_search_space()
        proposer = SmartPerturbationProposer(space, exploration_phase=3)
        current = {"learning_rate": 1e-5, "num_generations": 4,
                    "temperature": 0.7, "lora_r": 8}

        # Enough history to enter smart phase
        history = [
            {"params": {"learning_rate": 1e-5, "num_generations": 4, "temperature": 0.7, "lora_r": 8}, "objective": 0.5, "status": "keep"},
            {"params": {"learning_rate": 2e-5, "num_generations": 4, "temperature": 0.7, "lora_r": 8}, "objective": 0.6, "status": "keep"},
            {"params": {"learning_rate": 3e-5, "num_generations": 8, "temperature": 0.7, "lora_r": 8}, "objective": 0.7, "status": "keep"},
            {"params": {"learning_rate": 1e-5, "num_generations": 4, "temperature": 0.3, "lora_r": 16}, "objective": 0.4, "status": "discard"},
            {"params": {"learning_rate": 5e-5, "num_generations": 4, "temperature": 0.7, "lora_r": 8}, "objective": 0.65, "status": "keep"},
            {"params": {"learning_rate": 1e-6, "num_generations": 2, "temperature": 0.9, "lora_r": 4}, "objective": 0.3, "status": "discard"},
            {"params": {"learning_rate": 4e-5, "num_generations": 8, "temperature": 0.5, "lora_r": 8}, "objective": 0.75, "status": "keep"},
            {"params": {"learning_rate": 2e-5, "num_generations": 8, "temperature": 0.6, "lora_r": 16}, "objective": 0.55, "status": "discard"},
        ]
        params, desc = proposer.propose(current, history)
        assert "smart" in desc

    def test_create_proposer_smart(self):
        space = create_quick_search_space()
        p = create_proposer("smart", space)
        assert isinstance(p, SmartPerturbationProposer)


# ---------------------------------------------------------------------------
# Hyperparameter importance analysis
# ---------------------------------------------------------------------------


class TestParameterImportance:
    def _records(self):
        return [
            _make_record("e1", {"lr": 1e-5, "temp": 0.7, "lora_r": 8}, 0.5),
            _make_record("e2", {"lr": 2e-5, "temp": 0.7, "lora_r": 8}, 0.6),
            _make_record("e3", {"lr": 3e-5, "temp": 0.7, "lora_r": 16}, 0.7),
            _make_record("e4", {"lr": 1e-6, "temp": 0.3, "lora_r": 8}, 0.3, "discard"),
            _make_record("e5", {"lr": 5e-5, "temp": 0.5, "lora_r": 8}, 0.65),
            _make_record("e6", {"lr": 1e-6, "temp": 0.9, "lora_r": 4}, 0.35, "discard"),
            _make_record("e7", {"lr": 4e-5, "temp": 0.6, "lora_r": 16}, 0.75),
        ]

    def test_computes_importance(self):
        importance = compute_parameter_importance(self._records())
        assert len(importance) > 0
        # lr should be important (strong correlation with objective)
        assert "lr" in importance
        # All scores between 0 and 1
        for v in importance.values():
            assert 0.0 <= v <= 1.0

    def test_too_few_records_returns_empty(self):
        records = self._records()[:3]
        importance = compute_parameter_importance(records)
        assert importance == {}

    def test_convergence_curve(self):
        records = self._records()
        curve = compute_convergence_curve(records)
        assert len(curve) == len(records)
        # Running best should be monotonically non-decreasing
        for i in range(1, len(curve)):
            assert curve[i][1] >= curve[i - 1][1]

    def test_convergence_curve_minimize(self):
        records = self._records()
        curve = compute_convergence_curve(records, direction="minimize")
        # Running best should be monotonically non-increasing
        for i in range(1, len(curve)):
            assert curve[i][1] <= curve[i - 1][1]

    def test_convergence_curve_ignores_discarded_improvement(self):
        records = [
            _make_record("baseline", {"lr": 1e-5}, 0.5, "keep"),
            _make_record("exp_bad", {"lr": 2e-5}, 0.7, "discard"),
            _make_record("exp_good", {"lr": 3e-5}, 0.6, "keep"),
        ]
        curve = compute_convergence_curve(records)
        assert curve == [(0, 0.5), (1, 0.5), (2, 0.6)]

    def test_diminishing_returns(self):
        records = self._records()
        dr = compute_diminishing_returns(records, window=3)
        assert dr is not None
        assert 0.0 <= dr <= 1.0

    def test_diminishing_returns_too_few(self):
        records = self._records()[:3]
        dr = compute_diminishing_returns(records, window=5)
        assert dr is None


class TestAnalysisReport:
    def test_generate_report(self):
        records = [
            _make_record("baseline", {"lr": 1e-5, "temp": 0.7}, 0.5),
            _make_record("e1", {"lr": 2e-5, "temp": 0.7}, 0.6),
            _make_record("e2", {"lr": 3e-5, "temp": 0.5}, 0.7),
            _make_record("e3", {"lr": 1e-6, "temp": 0.3}, 0.3, "discard"),
            _make_record("e4", {"lr": 5e-5, "temp": 0.6}, 0.65),
            _make_record("e5", {"lr": 4e-5, "temp": 0.6}, 0.75),
            _make_record("e6", {"lr": 1e-6, "temp": 0.9}, 0.35, "discard"),
            _make_record("e7", {"lr": 3e-5, "temp": 0.7}, 0.72),
            _make_record("e8", {"lr": 4e-5, "temp": 0.5}, 0.68),
            _make_record("e9", {"lr": 5e-5, "temp": 0.6}, 0.76),
            _make_record("e10", {"lr": 2e-5, "temp": 0.8}, 0.60, "discard"),
        ]
        report = generate_report(records)
        assert "HYPERPARAMETER IMPORTANCE ANALYSIS" in report
        assert "Best:" in report
        assert "Worst:" in report
        assert "Parameter importance" in report
        assert "Convergence:" in report

    def test_generate_report_too_few(self):
        records = [
            _make_record("baseline", {}, 0.5),
            _make_record("e1", {}, 0.6),
        ]
        report = generate_report(records)
        assert "Not enough data" in report


# ---------------------------------------------------------------------------
# Integration: smart proposer in the loop
# ---------------------------------------------------------------------------


class TestSmartProposerInLoop:
    @pytest.mark.asyncio
    async def test_smart_proposer_runs_full_loop(self, tmp_path):
        agent = AsyncMock()
        agent.model = None
        agent.generate_response = AsyncMock(return_value="response")
        env = MagicMock()
        reward_fn = MagicMock()

        call_n = 0

        async def compute_reward(turns, context=None):
            nonlocal call_n
            call_n += 1
            return MagicMock(score=0.3 + call_n * 0.05)

        reward_fn.compute_reward = compute_reward

        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=12,
            proposer="smart",
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=1,
            save_checkpoints=False,
        )

        with patch.object(
            AutoResearchLoop, "_train_with_params", new_callable=AsyncMock
        ):
            tracker = await run_auto_research(
                agent=agent,
                environment=env,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=reward_fn,
                config=config,
                baseline_params={
                    "learning_rate": 1e-5,
                    "num_generations": 4,
                    "temperature": 0.7,
                    "lora_r": 8,
                },
            )

        assert tracker.num_experiments == 12
        # Analysis file should exist (>= 6 experiments)
        assert (tmp_path / "analysis.txt").exists()
