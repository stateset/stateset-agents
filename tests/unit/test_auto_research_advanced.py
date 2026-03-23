"""
Tests for advanced auto_research features:
- Early stopping / plateau detection
- Multi-algorithm search
- W&B integration (mocked)
- Reward calibration path
- Rich analysis and summary
- Duplicate experiment_id guard
- TSV header recovery
- OpenAI backend LLM proposer
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stateset_agents.training.auto_research.config import AutoResearchConfig
from stateset_agents.training.auto_research.experiment_loop import (
    AutoResearchLoop,
    run_auto_research,
)
from stateset_agents.training.auto_research.experiment_tracker import (
    ExperimentRecord,
    ExperimentTracker,
)
from stateset_agents.training.auto_research.proposer import ExperimentProposer
from stateset_agents.training.auto_research.search_spaces import (
    create_multi_algorithm_search_space,
    create_quick_search_space,
)


EVAL_SCENARIOS = [
    {
        "topic": "test",
        "context": "Test.",
        "user_responses": ["Hello"],
    },
]


class CountingProposer(ExperimentProposer):
    """Proposer that counts calls for testing."""

    def __init__(self):
        self._count = 0

    def propose(self, current_best, history):
        self._count += 1
        params = dict(current_best)
        params["temperature"] = 0.5 + self._count * 0.05
        return params, f"counting #{self._count}"


# ---------------------------------------------------------------------------
# Early stopping / plateau detection
# ---------------------------------------------------------------------------


class TestEarlyStoppingConfig:
    def test_patience_defaults_to_zero(self):
        config = AutoResearchConfig()
        assert config.improvement_patience == 0

    def test_negative_patience_warns(self):
        config = AutoResearchConfig(improvement_patience=-1)
        warnings = config.validate()
        assert any("improvement_patience" in w for w in warnings)


class TestEarlyStopping:
    @pytest.mark.asyncio
    async def test_stops_after_patience_exhausted(self, tmp_path):
        """Loop should stop when last N experiments all fail to improve."""
        agent = AsyncMock()
        agent.model = None
        agent.generate_response = AsyncMock(return_value="response")
        env = MagicMock()

        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=100,  # High limit — should stop via patience
            improvement_patience=3,  # Stop after 3 consecutive non-improvements
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=1,
            save_checkpoints=False,
        )

        # Reward that only improves on baseline, never after
        call_count = 0
        reward_fn = MagicMock()

        async def compute_reward(turns, context=None):
            nonlocal call_count
            call_count += 1
            # Baseline gets 0.5, everything else gets 0.3
            return MagicMock(score=0.5 if call_count == 1 else 0.3)

        reward_fn.compute_reward = compute_reward

        with patch.object(
            AutoResearchLoop, "_train_with_params", new_callable=AsyncMock
        ):
            tracker = await run_auto_research(
                agent=agent,
                environment=env,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=reward_fn,
                config=config,
                proposer=CountingProposer(),
            )

        # Should have stopped early — baseline + 3 discarded = 4
        assert tracker.num_experiments == 4
        assert tracker.num_kept == 1  # Only baseline


# ---------------------------------------------------------------------------
# Experiment tracker: duplicate ID guard
# ---------------------------------------------------------------------------


class TestDuplicateIdGuard:
    def test_duplicate_id_raises(self, tmp_path):
        tracker = ExperimentTracker(tmp_path)
        record = ExperimentRecord(
            experiment_id="exp_1", params={}, metrics={},
            objective_value=0.5, training_time=0, status="keep",
        )
        tracker.record(record)

        with pytest.raises(ValueError, match="already recorded"):
            tracker.record(record)

    def test_load_record_does_not_persist(self, tmp_path):
        tracker = ExperimentTracker(tmp_path)
        record = ExperimentRecord(
            experiment_id="exp_1", params={"lr": 0.001}, metrics={},
            objective_value=0.5, training_time=0, status="keep",
        )
        tracker.load_record(record)

        # JSONL should be empty (load_record doesn't persist)
        jsonl = tmp_path / "experiments.jsonl"
        assert not jsonl.exists() or jsonl.read_text().strip() == ""

    def test_load_then_record_new_works(self, tmp_path):
        tracker = ExperimentTracker(tmp_path)
        # Load old record
        tracker.load_record(ExperimentRecord(
            experiment_id="exp_1", params={}, metrics={},
            objective_value=0.5, training_time=0, status="keep",
        ))
        # Record new one — should work fine
        tracker.record(ExperimentRecord(
            experiment_id="exp_2", params={}, metrics={},
            objective_value=0.6, training_time=1, status="keep",
        ))
        assert tracker.num_experiments == 2

    def test_load_then_record_same_id_raises(self, tmp_path):
        tracker = ExperimentTracker(tmp_path)
        tracker.load_record(ExperimentRecord(
            experiment_id="exp_1", params={}, metrics={},
            objective_value=0.5, training_time=0, status="keep",
        ))
        with pytest.raises(ValueError, match="already recorded"):
            tracker.record(ExperimentRecord(
                experiment_id="exp_1", params={}, metrics={},
                objective_value=0.6, training_time=1, status="keep",
            ))


# ---------------------------------------------------------------------------
# TSV header recovery
# ---------------------------------------------------------------------------


class TestTSVHeaderRecovery:
    def test_tsv_header_recreated_if_missing(self, tmp_path):
        # Create JSONL (simulate resume scenario with deleted TSV)
        jsonl = tmp_path / "experiments.jsonl"
        jsonl.write_text('{"experiment_id": "exp_1"}\n')

        # Create tracker — should create TSV with header
        ExperimentTracker(tmp_path)
        tsv = (tmp_path / "results.tsv").read_text()
        assert "experiment_id" in tsv

    def test_tsv_header_preserved_if_exists(self, tmp_path):
        # Create existing TSV
        tsv_path = tmp_path / "results.tsv"
        tsv_path.write_text("experiment_id\tobjective\ttraining_time\tstatus\tdescription\n")

        ExperimentTracker(tmp_path)
        # Should not duplicate header
        lines = tsv_path.read_text().strip().split("\n")
        assert len(lines) == 1  # Just the header


# ---------------------------------------------------------------------------
# Rich analysis
# ---------------------------------------------------------------------------


class TestAnalysis:
    def test_get_analysis_basic(self, tmp_path):
        tracker = ExperimentTracker(tmp_path)
        tracker.record(ExperimentRecord(
            experiment_id="baseline", params={}, metrics={},
            objective_value=0.5, training_time=0, status="keep",
        ))
        tracker.record(ExperimentRecord(
            experiment_id="exp_1", params={"lr": 0.001}, metrics={},
            objective_value=0.7, training_time=10, status="keep",
        ))
        tracker.record(ExperimentRecord(
            experiment_id="exp_2", params={"lr": 0.01}, metrics={},
            objective_value=0.3, training_time=8, status="discard",
        ))

        analysis = tracker.get_analysis()
        assert analysis["total_experiments"] == 3
        assert analysis["kept"] == 2
        assert analysis["discarded"] == 1
        assert analysis["best_value"] == 0.7
        assert analysis["best_experiment_id"] == "exp_1"
        assert analysis["improvement_rate"] == 2 / 3
        assert analysis["avg_training_time"] > 0

    def test_get_analysis_empty(self, tmp_path):
        tracker = ExperimentTracker(tmp_path)
        analysis = tracker.get_analysis()
        assert analysis["total_experiments"] == 0
        assert analysis["best_value"] is None


# ---------------------------------------------------------------------------
# Multi-algorithm search space
# ---------------------------------------------------------------------------


class TestMultiAlgorithm:
    def test_search_space_has_algorithm_dim(self):
        space = create_multi_algorithm_search_space()
        dim_names = {d.name for d in space.dimensions}
        assert "algorithm" in dim_names
        algo_dim = next(d for d in space.dimensions if d.name == "algorithm")
        assert "gspo" in algo_dim.choices
        assert "grpo" in algo_dim.choices
        assert "dapo" in algo_dim.choices
        assert "vapo" in algo_dim.choices

    def test_auto_algorithm_config_validates(self):
        config = AutoResearchConfig(trainer_algorithm="auto")
        warnings = config.validate()
        assert not any("trainer_algorithm" in w for w in warnings)


# ---------------------------------------------------------------------------
# W&B integration (mocked)
# ---------------------------------------------------------------------------


class TestWandBIntegration:
    @pytest.mark.asyncio
    async def test_wandb_init_called(self, tmp_path):
        agent = AsyncMock()
        agent.model = None
        agent.generate_response = AsyncMock(return_value="response")
        env = MagicMock()
        reward_fn = MagicMock()

        async def compute_reward(turns, context=None):
            return MagicMock(score=0.5)

        reward_fn.compute_reward = compute_reward

        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=2,
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=1,
            save_checkpoints=False,
            log_to_wandb=True,
            wandb_project="test-project",
        )

        mock_wandb = MagicMock()
        mock_wandb.init = MagicMock()
        mock_wandb.log = MagicMock()
        mock_wandb.finish = MagicMock()
        mock_wandb.summary = MagicMock()

        with patch.object(
            AutoResearchLoop, "_train_with_params", new_callable=AsyncMock
        ), patch.dict("sys.modules", {"wandb": mock_wandb}):
            await run_auto_research(
                agent=agent,
                environment=env,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=reward_fn,
                config=config,
                proposer=CountingProposer(),
            )

        # wandb.init should have been called
        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args
        assert call_kwargs.kwargs.get("project") == "test-project"

        # wandb.log should have been called for each experiment
        assert mock_wandb.log.call_count >= 2  # baseline + 1 experiment

        # wandb.finish should have been called
        mock_wandb.finish.assert_called_once()

    @pytest.mark.asyncio
    async def test_wandb_init_failure_handled(self, tmp_path):
        """If wandb.init() raises, loop should continue without it."""
        agent = AsyncMock()
        agent.model = None
        agent.generate_response = AsyncMock(return_value="response")
        env = MagicMock()
        reward_fn = MagicMock()

        async def compute_reward(turns, context=None):
            return MagicMock(score=0.5)

        reward_fn.compute_reward = compute_reward

        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=2,
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=1,
            save_checkpoints=False,
            log_to_wandb=True,
        )

        # Mock wandb to raise on init
        mock_wandb = MagicMock()
        mock_wandb.init.side_effect = RuntimeError("wandb init failed")

        with patch.object(
            AutoResearchLoop, "_train_with_params", new_callable=AsyncMock
        ), patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker = await run_auto_research(
                agent=agent,
                environment=env,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=reward_fn,
                config=config,
                proposer=CountingProposer(),
            )
            # Should complete despite wandb failure
            assert tracker.num_experiments == 2


# ---------------------------------------------------------------------------
# Reward calibration path
# ---------------------------------------------------------------------------


class TestRewardCalibration:
    def test_calibration_config(self):
        config = AutoResearchConfig(
            calibrate_rewards=True,
            calibration_method="z_score",
        )
        assert config.calibrate_rewards is True
        assert config.calibration_method == "z_score"

    @pytest.mark.asyncio
    async def test_calibration_attempted_when_enabled(self, tmp_path):
        """When calibrate_rewards=True, the loop should try to wrap reward_fn."""
        agent = AsyncMock()
        agent.model = None
        agent.generate_response = AsyncMock(return_value="response")
        env = MagicMock()
        reward_fn = MagicMock()

        async def compute_reward(turns, context=None):
            return MagicMock(score=0.5)

        reward_fn.compute_reward = compute_reward

        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=1,
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=1,
            save_checkpoints=False,
            calibrate_rewards=True,
        )

        with patch.object(
            AutoResearchLoop, "_train_with_params", new_callable=AsyncMock
        ):
            loop = AutoResearchLoop(
                agent=agent,
                environment=env,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=reward_fn,
                config=config,
                proposer=CountingProposer(),
            )
            # Calibration was attempted (may succeed or fall back)
            # The _eval_reward_fn should be set
            assert loop._eval_reward_fn is not None


# ---------------------------------------------------------------------------
# LLM Proposer: OpenAI backend
# ---------------------------------------------------------------------------


class TestMultiAlgorithmIntegration:
    """Test that algorithm="auto" with a proposer selecting specific algorithms works."""

    @pytest.mark.asyncio
    async def test_proposer_selects_algorithm(self, tmp_path):
        """Verify that when trainer_algorithm='auto', the proposer's
        algorithm param is routed to the correct trainer."""
        agent = AsyncMock()
        agent.model = None
        agent.generate_response = AsyncMock(return_value="response")
        env = MagicMock()
        reward_fn = MagicMock()

        async def compute_reward(turns, context=None):
            return MagicMock(score=0.5)

        reward_fn.compute_reward = compute_reward

        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=3,
            trainer_algorithm="auto",
            output_dir=str(tmp_path),
            search_space_name="multi_algorithm",
            eval_episodes=1,
            save_checkpoints=False,
        )

        # Proposer that always picks "dapo"
        class DAPOProposer(ExperimentProposer):
            def __init__(self):
                self._n = 0

            def propose(self, current_best, history):
                self._n += 1
                return {
                    "algorithm": "dapo",
                    "learning_rate": 1e-5,
                    "temperature": 0.7,
                }, f"force dapo #{self._n}"

        train_calls = []

        async def capture_train(params):
            train_calls.append(params)

        with patch.object(
            AutoResearchLoop, "_train_with_params", side_effect=capture_train
        ):
            tracker = await run_auto_research(
                agent=agent,
                environment=env,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=reward_fn,
                config=config,
                proposer=DAPOProposer(),
            )

        # Should have trained with 2 experiments (baseline has no training)
        assert len(train_calls) == 2
        # Verify descriptions mention dapo
        exp_records = [r for r in tracker.records if r.experiment_id != "baseline"]
        assert all("dapo" in r.description for r in exp_records)

    @pytest.mark.asyncio
    async def test_auto_defaults_to_gspo(self, tmp_path):
        """When algorithm=auto and proposer doesn't specify, defaults to gspo."""
        agent = AsyncMock()
        agent.model = MagicMock()
        agent.model.parameters = MagicMock(return_value=[MagicMock()])
        agent.generate_response = AsyncMock(return_value="response")
        env = MagicMock()
        reward_fn = MagicMock()

        async def compute_reward(turns, context=None):
            return MagicMock(score=0.5)

        reward_fn.compute_reward = compute_reward

        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=2,
            trainer_algorithm="auto",
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=1,
            save_checkpoints=False,
        )

        # Proposer that does NOT include "algorithm" key
        class NoAlgoProposer(ExperimentProposer):
            def propose(self, current_best, history):
                return {"learning_rate": 1e-5}, "no algo specified"

        with patch.object(
            AutoResearchLoop, "_train_gspo", new_callable=AsyncMock
        ) as mock_gspo:
            await run_auto_research(
                agent=agent,
                environment=env,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=reward_fn,
                config=config,
                proposer=NoAlgoProposer(),
            )

        # Should have called GSPO (the default for "auto")
        assert mock_gspo.call_count == 1


class TestDAVOCallSignatures:
    """Verify DAPO and VAPO are called with correct signatures."""

    @pytest.mark.asyncio
    async def test_dapo_called_with_model_name(self, tmp_path):
        """DAPO entrypoint expects model_name, not config."""
        agent = AsyncMock()
        agent.model = MagicMock()
        # Give agent.model fake parameters so it's not detected as stub
        agent.model.parameters = MagicMock(return_value=[MagicMock()])
        agent.generate_response = AsyncMock(return_value="response")
        env = MagicMock()
        env.scenarios = [{"context": "Test scenario"}]
        reward_fn = MagicMock()

        async def compute_reward(turns, context=None):
            return MagicMock(score=0.5)

        reward_fn.compute_reward = compute_reward

        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=2,
            trainer_algorithm="dapo",
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=1,
            save_checkpoints=False,
            base_config_overrides={"model_name": "test-model"},
        )

        with patch(
            "stateset_agents.training.auto_research.experiment_loop."
            "AutoResearchLoop._train_dapo",
            new_callable=AsyncMock,
        ) as mock_dapo:
            await run_auto_research(
                agent=agent,
                environment=env,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=reward_fn,
                config=config,
                proposer=CountingProposer(),
            )

        # DAPO should have been called
        assert mock_dapo.call_count == 1

    @pytest.mark.asyncio
    async def test_vapo_called_with_model_name(self, tmp_path):
        """VAPO entrypoint expects model_name, not config."""
        agent = AsyncMock()
        agent.model = MagicMock()
        agent.model.parameters = MagicMock(return_value=[MagicMock()])
        agent.generate_response = AsyncMock(return_value="response")
        env = MagicMock()
        env.scenarios = [{"context": "Test scenario"}]
        reward_fn = MagicMock()

        async def compute_reward(turns, context=None):
            return MagicMock(score=0.5)

        reward_fn.compute_reward = compute_reward

        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=2,
            trainer_algorithm="vapo",
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=1,
            save_checkpoints=False,
            base_config_overrides={"model_name": "test-model"},
        )

        with patch(
            "stateset_agents.training.auto_research.experiment_loop."
            "AutoResearchLoop._train_vapo",
            new_callable=AsyncMock,
        ) as mock_vapo:
            await run_auto_research(
                agent=agent,
                environment=env,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=reward_fn,
                config=config,
                proposer=CountingProposer(),
            )

        assert mock_vapo.call_count == 1


class TestEarlyAbortWiring:
    """Verify EarlyAbortCallback is actually wired into GSPO training."""

    def test_gspo_entrypoint_accepts_callbacks(self):
        """The train_with_gspo function should accept a callbacks parameter."""
        import inspect

        from stateset_agents.training.gspo_entrypoints import train_with_gspo

        sig = inspect.signature(train_with_gspo)
        assert "callbacks" in sig.parameters

    def test_early_abort_callback_importable(self):
        from stateset_agents.training.auto_research import EarlyAbortCallback

        cb = EarlyAbortCallback()
        assert hasattr(cb, "on_step_end")
        assert hasattr(cb, "should_abort")
        assert hasattr(cb, "on_train_start")


class TestStubAgentDetection:
    """Test the stub agent detection logic."""

    @pytest.mark.asyncio
    async def test_stub_detected_with_no_model(self, tmp_path):
        config = AutoResearchConfig(
            time_budget=10, max_experiments=1,
            output_dir=str(tmp_path), search_space_name="quick",
            eval_episodes=1, save_checkpoints=False,
        )
        agent = AsyncMock()
        agent.model = None
        env = MagicMock()
        reward_fn = MagicMock()

        loop = AutoResearchLoop(
            agent=agent, environment=env,
            eval_scenarios=EVAL_SCENARIOS,
            reward_fn=reward_fn, config=config,
            proposer=CountingProposer(),
        )
        assert loop._is_stub_agent() is True

    @pytest.mark.asyncio
    async def test_real_model_not_detected_as_stub(self, tmp_path):
        config = AutoResearchConfig(
            time_budget=10, max_experiments=1,
            output_dir=str(tmp_path), search_space_name="quick",
            eval_episodes=1, save_checkpoints=False,
        )
        agent = AsyncMock()
        agent.model = MagicMock()
        agent.model.parameters = MagicMock(return_value=[MagicMock()])
        env = MagicMock()
        reward_fn = MagicMock()

        loop = AutoResearchLoop(
            agent=agent, environment=env,
            eval_scenarios=EVAL_SCENARIOS,
            reward_fn=reward_fn, config=config,
            proposer=CountingProposer(),
        )
        assert loop._is_stub_agent() is False


class TestGPUCleanup:
    """Test that GPU memory is cleaned up after crashes and timeouts."""

    @pytest.mark.asyncio
    async def test_cleanup_called_on_crash(self, tmp_path):
        agent = AsyncMock()
        agent.model = MagicMock()
        agent.model.parameters = MagicMock(return_value=[MagicMock()])
        agent.generate_response = AsyncMock(return_value="response")
        env = MagicMock()
        reward_fn = MagicMock()

        async def compute_reward(turns, context=None):
            return MagicMock(score=0.5)

        reward_fn.compute_reward = compute_reward

        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=3,
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=1,
            save_checkpoints=False,
        )

        async def crash_train(params):
            raise RuntimeError("OOM")

        with patch.object(
            AutoResearchLoop, "_train_with_params", side_effect=crash_train
        ), patch.object(
            AutoResearchLoop, "_cleanup_gpu"
        ) as mock_cleanup:
            await run_auto_research(
                agent=agent,
                environment=env,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=reward_fn,
                config=config,
                proposer=CountingProposer(),
            )

        # Cleanup should have been called for each crashed experiment
        assert mock_cleanup.call_count >= 2  # 2 experiments crash (baseline doesn't train)


class TestEnhancedAnalysis:
    """Test the enriched get_analysis() output."""

    def test_analysis_includes_convergence_curve(self, tmp_path):
        tracker = ExperimentTracker(tmp_path)
        for i in range(8):
            tracker.record(ExperimentRecord(
                experiment_id=f"exp_{i}",
                params={"lr": 1e-5 + i * 1e-6},
                metrics={},
                objective_value=0.5 + i * 0.02,
                training_time=10.0,
                status="keep" if i % 2 == 0 else "discard",
            ))

        analysis = tracker.get_analysis()
        assert "convergence_curve" in analysis
        assert len(analysis["convergence_curve"]) > 0
        # Running best should be monotonically non-decreasing
        curve = analysis["convergence_curve"]
        for i in range(1, len(curve)):
            assert curve[i][1] >= curve[i - 1][1]

    def test_analysis_includes_parameter_importance(self, tmp_path):
        tracker = ExperimentTracker(tmp_path)
        for i in range(8):
            tracker.record(ExperimentRecord(
                experiment_id=f"exp_{i}",
                params={"lr": 1e-5 * (i + 1), "temp": 0.5 + i * 0.05},
                metrics={},
                objective_value=0.3 + i * 0.05,
                training_time=10.0,
                status="keep",
            ))

        analysis = tracker.get_analysis()
        assert "parameter_importance" in analysis
        assert len(analysis["parameter_importance"]) > 0

    def test_analysis_includes_experiments_for_jupyter(self, tmp_path):
        tracker = ExperimentTracker(tmp_path)
        tracker.record(ExperimentRecord(
            experiment_id="baseline",
            params={"lr": 1e-5},
            metrics={"eval_reward": 0.5},
            objective_value=0.5,
            training_time=0,
            status="keep",
        ))

        analysis = tracker.get_analysis()
        assert "experiments" in analysis
        assert len(analysis["experiments"]) == 1
        exp = analysis["experiments"][0]
        assert exp["id"] == "baseline"
        assert exp["objective"] == 0.5
        assert exp["params"]["lr"] == 1e-5
        assert exp["metrics"]["eval_reward"] == 0.5


class TestTrackerLoadFromDisk:
    """Test loading a completed run from disk."""

    def test_load_from_jsonl(self, tmp_path):
        # Write a fake completed run
        jsonl = tmp_path / "experiments.jsonl"
        jsonl.write_text(
            json.dumps({"experiment_id": "baseline", "params": {"lr": 1e-5},
                        "metrics": {"eval_reward": 0.5}, "objective_value": 0.5,
                        "training_time": 0, "status": "keep", "description": "baseline"}) + "\n"
            + json.dumps({"experiment_id": "exp_0001", "params": {"lr": 2e-5},
                          "metrics": {"eval_reward": 0.6}, "objective_value": 0.6,
                          "training_time": 10, "status": "keep", "description": "lr up"}) + "\n"
            + json.dumps({"experiment_id": "exp_0002", "params": {"lr": 3e-5},
                          "metrics": {"eval_reward": 0.4}, "objective_value": 0.4,
                          "training_time": 8, "status": "discard", "description": "lr too high"}) + "\n"
        )

        tracker = ExperimentTracker.load(str(tmp_path))

        assert tracker.num_experiments == 3
        assert tracker.num_kept == 2
        assert tracker.num_discarded == 1
        assert tracker.best_value == 0.6
        assert tracker.best_record.experiment_id == "exp_0001"

    def test_load_empty_dir(self, tmp_path):
        tracker = ExperimentTracker.load(str(tmp_path))
        assert tracker.num_experiments == 0

    def test_load_then_get_analysis(self, tmp_path):
        jsonl = tmp_path / "experiments.jsonl"
        lines = []
        for i in range(8):
            lines.append(json.dumps({
                "experiment_id": f"exp_{i}",
                "params": {"lr": 1e-5 * (i + 1), "temp": 0.5 + i * 0.05},
                "metrics": {}, "objective_value": 0.3 + i * 0.05,
                "training_time": 10, "status": "keep",
            }))
        jsonl.write_text("\n".join(lines) + "\n")

        tracker = ExperimentTracker.load(str(tmp_path))
        analysis = tracker.get_analysis()

        assert analysis["total_experiments"] == 8
        assert "convergence_curve" in analysis
        assert "parameter_importance" in analysis
        assert "experiments" in analysis
        assert len(analysis["experiments"]) == 8


class TestConfigFromFile:
    """Test loading AutoResearchConfig from YAML/JSON files."""

    def test_from_json(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "time_budget": 120,
            "proposer": "smart",
            "search_space_name": "quick",
            "max_experiments": 50,
        }))

        config = AutoResearchConfig.from_file(config_path)
        assert config.time_budget == 120
        assert config.proposer == "smart"
        assert config.search_space_name == "quick"
        assert config.max_experiments == 50

    def test_from_json_nested(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "auto_research": {
                "time_budget": 60,
                "proposer": "bayesian",
            }
        }))

        config = AutoResearchConfig.from_file(config_path)
        assert config.time_budget == 60
        assert config.proposer == "bayesian"

    def test_aliases(self, tmp_path):
        """Common aliases like 'algorithm' and 'patience' should work."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "algorithm": "dapo",
            "patience": 15,
            "search_space": "model",
            "wandb": True,
        }))

        config = AutoResearchConfig.from_file(config_path)
        assert config.trainer_algorithm == "dapo"
        assert config.improvement_patience == 15
        assert config.search_space_name == "model"
        assert config.log_to_wandb is True

    def test_unknown_keys_ignored(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "time_budget": 120,
            "unknown_field": "should be ignored",
            "another_unknown": 42,
        }))

        config = AutoResearchConfig.from_file(config_path)
        assert config.time_budget == 120

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AutoResearchConfig.from_file(tmp_path / "nonexistent.json")

    def test_to_dict(self):
        config = AutoResearchConfig(time_budget=120, proposer="smart")
        d = config.to_dict()
        assert d["time_budget"] == 120
        assert d["proposer"] == "smart"
        assert isinstance(d, dict)

    def test_roundtrip_json(self, tmp_path):
        original = AutoResearchConfig(
            time_budget=180, proposer="adaptive",
            max_experiments=75, improvement_patience=8,
        )
        config_path = tmp_path / "roundtrip.json"
        config_path.write_text(json.dumps(original.to_dict()))

        loaded = AutoResearchConfig.from_file(config_path)
        assert loaded.time_budget == 180
        assert loaded.proposer == "adaptive"
        assert loaded.max_experiments == 75
        assert loaded.improvement_patience == 8


class TestCategoricalParameterImportance:
    """Test that categorical params are included in importance analysis."""

    def test_categorical_params_analyzed(self):
        from stateset_agents.training.auto_research.analysis import (
            compute_parameter_importance,
        )
        from stateset_agents.training.auto_research.experiment_tracker import (
            ExperimentRecord,
        )

        records = [
            ExperimentRecord("e1", {"lr": 1e-5, "algo": "gspo"}, {}, 0.7, 10, "keep"),
            ExperimentRecord("e2", {"lr": 2e-5, "algo": "gspo"}, {}, 0.75, 10, "keep"),
            ExperimentRecord("e3", {"lr": 1e-5, "algo": "grpo"}, {}, 0.4, 10, "discard"),
            ExperimentRecord("e4", {"lr": 3e-5, "algo": "gspo"}, {}, 0.65, 10, "keep"),
            ExperimentRecord("e5", {"lr": 1e-6, "algo": "grpo"}, {}, 0.35, 10, "discard"),
            ExperimentRecord("e6", {"lr": 2e-5, "algo": "gspo"}, {}, 0.72, 10, "keep"),
            ExperimentRecord("e7", {"lr": 5e-5, "algo": "grpo"}, {}, 0.5, 10, "keep"),
        ]

        importance = compute_parameter_importance(records)

        # Both numeric and categorical params should appear
        assert "lr" in importance
        assert "algo" in importance
        # "algo" should be important (gspo correlates with higher scores)
        assert importance["algo"] > 0

    def test_boolean_params_analyzed(self):
        from stateset_agents.training.auto_research.analysis import (
            compute_parameter_importance,
        )
        from stateset_agents.training.auto_research.experiment_tracker import (
            ExperimentRecord,
        )

        records = [
            ExperimentRecord("e1", {"lr": 1e-5, "use_lora": True}, {}, 0.7, 10, "keep"),
            ExperimentRecord("e2", {"lr": 2e-5, "use_lora": True}, {}, 0.75, 10, "keep"),
            ExperimentRecord("e3", {"lr": 1e-5, "use_lora": False}, {}, 0.4, 10, "discard"),
            ExperimentRecord("e4", {"lr": 3e-5, "use_lora": True}, {}, 0.65, 10, "keep"),
            ExperimentRecord("e5", {"lr": 1e-6, "use_lora": False}, {}, 0.35, 10, "discard"),
            ExperimentRecord("e6", {"lr": 2e-5, "use_lora": True}, {}, 0.72, 10, "keep"),
        ]

        importance = compute_parameter_importance(records)
        assert "use_lora" in importance


class TestToDataFrame:
    """Test converting ExperimentTracker to pandas DataFrame."""

    def test_to_dataframe_basic(self, tmp_path):
        tracker = ExperimentTracker(tmp_path)
        tracker.record(ExperimentRecord(
            experiment_id="baseline",
            params={"lr": 1e-5, "temp": 0.7},
            metrics={"eval_reward": 0.5, "eval_reward_std": 0.1},
            objective_value=0.5,
            training_time=0,
            status="keep",
            description="baseline",
        ))
        tracker.record(ExperimentRecord(
            experiment_id="exp_1",
            params={"lr": 2e-5, "temp": 0.6},
            metrics={"eval_reward": 0.6, "eval_reward_std": 0.08},
            objective_value=0.6,
            training_time=10,
            status="keep",
            description="lr up",
        ))

        df = tracker.to_dataframe()

        assert len(df) == 2
        assert "id" in df.columns
        assert "objective" in df.columns
        assert "status" in df.columns
        assert "running_best" in df.columns
        assert "param_lr" in df.columns
        assert "param_temp" in df.columns
        assert "metric_eval_reward" in df.columns

        # Running best should be monotonically non-decreasing
        assert df["running_best"].iloc[0] == 0.5
        assert df["running_best"].iloc[1] == 0.6

    def test_to_dataframe_running_best_ignores_discard(self, tmp_path):
        tracker = ExperimentTracker(tmp_path)
        tracker.record(ExperimentRecord(
            experiment_id="baseline",
            params={},
            metrics={"eval_reward": 0.5},
            objective_value=0.5,
            training_time=0,
            status="keep",
        ))
        tracker.record(ExperimentRecord(
            experiment_id="exp_rejected",
            params={},
            metrics={"eval_reward": 0.7},
            objective_value=0.7,
            training_time=0,
            status="discard",
        ))
        tracker.record(ExperimentRecord(
            experiment_id="exp_kept",
            params={},
            metrics={"eval_reward": 0.6},
            objective_value=0.6,
            training_time=0,
            status="keep",
        ))

        df = tracker.to_dataframe()
        assert list(df["running_best"]) == [0.5, 0.5, 0.6]

    def test_to_dataframe_empty(self, tmp_path):
        tracker = ExperimentTracker(tmp_path)
        df = tracker.to_dataframe()
        assert len(df) == 0

    def test_to_dataframe_from_loaded_tracker(self, tmp_path):
        """Load from disk then convert to DataFrame."""
        jsonl = tmp_path / "experiments.jsonl"
        jsonl.write_text(
            json.dumps({"experiment_id": "b", "params": {"lr": 1e-5},
                        "metrics": {"r": 0.5}, "objective_value": 0.5,
                        "training_time": 0, "status": "keep"}) + "\n"
            + json.dumps({"experiment_id": "e1", "params": {"lr": 2e-5},
                          "metrics": {"r": 0.7}, "objective_value": 0.7,
                          "training_time": 10, "status": "keep"}) + "\n"
        )

        tracker = ExperimentTracker.load(str(tmp_path))
        df = tracker.to_dataframe()
        assert len(df) == 2
        assert df["objective"].max() == 0.7
        assert "param_lr" in df.columns

    def test_to_dataframe_minimize_direction(self, tmp_path):
        tracker = ExperimentTracker(tmp_path, direction="minimize")
        tracker.record(ExperimentRecord(
            experiment_id="a", params={}, metrics={},
            objective_value=1.0, training_time=0, status="keep",
        ))
        tracker.record(ExperimentRecord(
            experiment_id="b", params={}, metrics={},
            objective_value=0.5, training_time=0, status="keep",
        ))

        df = tracker.to_dataframe()
        assert df["running_best"].iloc[0] == 1.0
        assert df["running_best"].iloc[1] == 0.5  # cummin


class TestLegacyTSVImport:
    """Test importing results from legacy autoresearch results.tsv."""

    def test_import_legacy_tsv(self, tmp_path):
        tsv_path = tmp_path / "results.tsv"
        tsv_path.write_text(
            "commit\tavg_reward\tmemory_gb\tstatus\tdescription\n"
            "f322238\t0.506406\t0.0\tkeep\tbaseline Qwen3.5-0.8B\n"
            "9341504\t0.507135\t0.0\tkeep\tdetailed system prompt\n"
            "dbe3c69\t0.523333\t0.0\tkeep\tLoRA r=4 alpha=8 + temp 0.5\n"
            "d868a30\t0.469010\t0.0\tdiscard\touter iterations 1->3\n"
        )

        tracker = ExperimentTracker.from_legacy_tsv(tsv_path)

        assert tracker.num_experiments == 4
        assert tracker.num_kept == 3
        assert tracker.num_discarded == 1
        assert tracker.best_value == pytest.approx(0.523333)
        assert tracker.best_record.experiment_id == "dbe3c69"

    def test_import_empty_tsv(self, tmp_path):
        tsv_path = tmp_path / "results.tsv"
        tsv_path.write_text("commit\tavg_reward\tmemory_gb\tstatus\tdescription\n")

        tracker = ExperimentTracker.from_legacy_tsv(tsv_path)
        assert tracker.num_experiments == 0

    def test_import_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ExperimentTracker.from_legacy_tsv(tmp_path / "nonexistent.tsv")

    def test_import_then_analyze(self, tmp_path):
        tsv_path = tmp_path / "results.tsv"
        lines = ["commit\tavg_reward\tmemory_gb\tstatus\tdescription"]
        for i in range(8):
            lines.append(f"abc{i:04d}\t{0.5 + i * 0.02:.6f}\t4.0\tkeep\texperiment {i}")
        tsv_path.write_text("\n".join(lines) + "\n")

        tracker = ExperimentTracker.from_legacy_tsv(tsv_path)
        analysis = tracker.get_analysis()
        assert analysis["total_experiments"] == 8
        assert analysis["best_value"] > 0.5


class TestCompareRuns:
    """Test comparing multiple auto-research runs."""

    def test_compare_two_runs(self, tmp_path):
        from stateset_agents.training.auto_research.analysis import compare_runs

        # Create two fake runs
        run_a = tmp_path / "run_perturbation"
        run_a.mkdir()
        (run_a / "experiments.jsonl").write_text(
            json.dumps({"experiment_id": "baseline", "params": {"lr": 1e-5}, "metrics": {}, "objective_value": 0.5, "training_time": 0, "status": "keep"}) + "\n"
            + json.dumps({"experiment_id": "exp_1", "params": {"lr": 2e-5}, "metrics": {}, "objective_value": 0.6, "training_time": 10, "status": "keep"}) + "\n"
        )

        run_b = tmp_path / "run_smart"
        run_b.mkdir()
        (run_b / "experiments.jsonl").write_text(
            json.dumps({"experiment_id": "baseline", "params": {"lr": 1e-5}, "metrics": {}, "objective_value": 0.5, "training_time": 0, "status": "keep"}) + "\n"
            + json.dumps({"experiment_id": "exp_1", "params": {"lr": 3e-5}, "metrics": {}, "objective_value": 0.7, "training_time": 12, "status": "keep"}) + "\n"
        )

        report = compare_runs(str(run_a), str(run_b))

        assert "RUN COMPARISON" in report
        assert "run_perturbation" in report
        assert "run_smart" in report
        assert "Winner: run_smart" in report
        assert "0.700000" in report

    def test_compare_single_run(self, tmp_path):
        from stateset_agents.training.auto_research.analysis import compare_runs

        run = tmp_path / "single_run"
        run.mkdir()
        (run / "experiments.jsonl").write_text(
            json.dumps({"experiment_id": "baseline", "params": {}, "metrics": {}, "objective_value": 0.5, "training_time": 0, "status": "keep"}) + "\n"
        )

        report = compare_runs(str(run))
        assert "Winner" in report

    def test_compare_empty(self):
        from stateset_agents.training.auto_research.analysis import compare_runs

        report = compare_runs()
        assert "No runs to compare" in report


class TestASCIIChart:
    def test_chart_renders(self):
        from stateset_agents.training.auto_research.analysis import _ascii_chart

        values = [0.5, 0.55, 0.6, 0.62, 0.65, 0.7, 0.72, 0.75, 0.8, 0.85]
        chart = _ascii_chart(values, width=20, height=5)
        assert chart is not None
        assert len(chart) == 5
        assert all(len(row) == 20 for row in chart)
        # Bottom-left should be filled (lower values)
        # Top-right should be filled (higher values)
        assert "█" in chart[0]  # Top row should have marks

    def test_chart_flat_line(self):
        from stateset_agents.training.auto_research.analysis import _ascii_chart

        values = [0.5] * 10
        chart = _ascii_chart(values, width=10, height=5)
        assert chart is not None
        # Flat line should have a horizontal line in the middle
        assert "─" in chart[2]  # Middle row

    def test_chart_too_few_values(self):
        from stateset_agents.training.auto_research.analysis import _ascii_chart

        assert _ascii_chart([0.5]) is None
        assert _ascii_chart([]) is None


class TestCLIConfigIntegration:
    """Verify the CLI auto-research command doesn't crash with --config."""

    def test_cli_with_config_file(self, tmp_path):
        from typer.testing import CliRunner
        from stateset_agents.cli import app

        config_path = tmp_path / "test.json"
        config_path.write_text(json.dumps({
            "auto_research": {
                "time_budget": 60,
                "proposer": "smart",
                "agent": {"model_name": "gpt2"},
                "environment": {
                    "scenarios": [{
                        "topic": "test",
                        "context": "Test",
                        "user_responses": ["Hi"],
                    }]
                }
            }
        }))

        runner = CliRunner()
        result = runner.invoke(app, [
            "auto-research", "--dry-run",
            "--config", str(config_path),
        ])
        assert result.exit_code == 0
        assert "smart" in result.output

    def test_cli_without_config_file(self):
        from typer.testing import CliRunner
        from stateset_agents.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "auto-research", "--dry-run",
            "--proposer", "adaptive",
        ])
        assert result.exit_code == 0
        assert "adaptive" in result.output


class TestPublicAPIExports:
    def test_multi_algorithm_exported_from_package(self):
        from stateset_agents.training.auto_research import (
            create_multi_algorithm_search_space,
        )

        space = create_multi_algorithm_search_space()
        assert len(space.dimensions) >= 5

    def test_validate_params_exported_from_package(self):
        from stateset_agents.training.auto_research import (
            validate_params_against_space,
        )

        space = create_quick_search_space()
        warnings = validate_params_against_space({"learning_rate": 1e-5}, space)
        assert warnings == []


class TestLLMProposerOpenAI:
    def test_openai_backend_mock(self):
        from stateset_agents.training.auto_research.llm_proposer import LLMProposer

        space = create_quick_search_space()
        proposer = LLMProposer(search_space=space, backend="openai")

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "params": {"learning_rate": 1e-4, "num_generations": 8,
                       "temperature": 0.5, "lora_r": 16},
            "description": "openai test"
        })
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        proposer._client = mock_client

        params, desc = proposer.propose(
            current_best={"learning_rate": 5e-6, "num_generations": 4},
            history=[],
        )

        assert params["learning_rate"] == 1e-4
        assert "llm:" in desc
        mock_client.chat.completions.create.assert_called_once()

    def test_unknown_backend_raises(self):
        from stateset_agents.training.auto_research.llm_proposer import LLMProposer

        space = create_quick_search_space()
        proposer = LLMProposer(search_space=space, backend="unknown")

        with pytest.raises(ValueError, match="Unknown LLM backend"):
            proposer._ensure_client()
