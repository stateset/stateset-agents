"""
Integration tests for the auto_research module.

These tests run the full autonomous research loop with a stub agent
to verify end-to-end behavior without requiring a GPU.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stateset_agents.training.auto_research.config import AutoResearchConfig
from stateset_agents.training.auto_research.experiment_loop import (
    AutoResearchLoop,
    run_auto_research,
)
from stateset_agents.training.auto_research.proposer import (
    ExperimentProposer,
)
from stateset_agents.training.auto_research.search_spaces import (
    create_quick_search_space,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EVAL_SCENARIOS = [
    {
        "topic": "test_scenario",
        "context": "Test evaluation scenario.",
        "user_responses": ["Hello", "Thanks"],
    },
]

TRAIN_SCENARIOS = [
    {
        "topic": "train_scenario",
        "context": "Test training scenario.",
        "user_responses": ["Hi", "Bye"],
    },
]


@pytest.fixture
def stub_agent():
    """Create a stub agent for testing."""
    agent = AsyncMock()
    agent.model = None
    agent.generate_response = AsyncMock(return_value="Test response.")
    agent.initialize = AsyncMock()
    return agent


@pytest.fixture
def stub_environment():
    """Create a stub environment for testing."""
    env = MagicMock()

    async def run_episode(agent_fn, scenario=None, max_turns=None):
        # Simulate a conversation episode
        trajectory = MagicMock()
        trajectory.turns = []
        trajectory.total_reward = 0.5
        trajectory.episode_length = 3
        return trajectory

    env.run_episode = run_episode
    env.scenarios = TRAIN_SCENARIOS
    env.clone = MagicMock(return_value=env)
    return env


@pytest.fixture
def stub_reward_fn():
    """Create a stub reward function."""
    reward_fn = MagicMock()

    async def compute_reward(turns, context=None):
        return MagicMock(score=0.5)

    reward_fn.compute_reward = compute_reward
    return reward_fn


class FixedProposer(ExperimentProposer):
    """Proposer that returns predetermined configs for testing."""

    def __init__(self, proposals: list[tuple[dict[str, Any], str]]):
        self._proposals = proposals
        self._index = 0

    def propose(
        self, current_best: dict[str, Any], history: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], str]:
        if self._index >= len(self._proposals):
            self._index = 0
        params, desc = self._proposals[self._index]
        self._index += 1
        return params, desc


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestAutoResearchLoopIntegration:
    """End-to-end tests of the autonomous research loop."""

    @pytest.mark.asyncio
    async def test_full_loop_with_fixed_proposer(
        self, stub_agent, stub_environment, stub_reward_fn, tmp_path
    ):
        """Run the full loop with a fixed proposer and verify results."""
        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=4,  # baseline + 3 experiments
            output_dir=str(tmp_path),
            proposer="perturbation",
            search_space_name="quick",
            eval_episodes=2,
            save_checkpoints=False,
        )

        proposer = FixedProposer([
            ({"learning_rate": 1e-5, "num_generations": 4}, "increase LR to 1e-5"),
            ({"learning_rate": 2e-5, "num_generations": 8}, "increase LR and generations"),
            ({"learning_rate": 5e-6, "num_generations": 2}, "decrease both"),
        ])

        # Mock the training to be a no-op (stub agent doesn't need real training)
        with patch.object(
            AutoResearchLoop, "_train_with_params", new_callable=AsyncMock
        ):
            tracker = await run_auto_research(
                agent=stub_agent,
                environment=stub_environment,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=stub_reward_fn,
                config=config,
                baseline_params={"learning_rate": 5e-6, "num_generations": 4},
                proposer=proposer,
            )

        # Verify results
        assert tracker.num_experiments == 4  # baseline + 3
        assert tracker.best_value is not None
        assert (tmp_path / "results.tsv").exists()
        assert (tmp_path / "experiments.jsonl").exists()

        # Verify TSV has correct number of lines (header + 4 records)
        tsv_lines = (tmp_path / "results.tsv").read_text().strip().split("\n")
        assert len(tsv_lines) == 5  # header + 4 records

    @pytest.mark.asyncio
    async def test_loop_stops_at_max_experiments(
        self, stub_agent, stub_environment, stub_reward_fn, tmp_path
    ):
        """Verify the loop respects max_experiments."""
        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=2,  # baseline + 1
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=1,
            save_checkpoints=False,
        )

        with patch.object(
            AutoResearchLoop, "_train_with_params", new_callable=AsyncMock
        ):
            tracker = await run_auto_research(
                agent=stub_agent,
                environment=stub_environment,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=stub_reward_fn,
                config=config,
            )

        assert tracker.num_experiments == 2

    @pytest.mark.asyncio
    async def test_crash_handling(
        self, stub_agent, stub_environment, stub_reward_fn, tmp_path
    ):
        """Verify that training crashes are handled gracefully."""
        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=3,
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=1,
            save_checkpoints=False,
        )

        call_count = 0

        async def _train_side_effect(params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("OOM: out of memory")
            # second call succeeds

        with patch.object(
            AutoResearchLoop,
            "_train_with_params",
            side_effect=_train_side_effect,
        ):
            tracker = await run_auto_research(
                agent=stub_agent,
                environment=stub_environment,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=stub_reward_fn,
                config=config,
            )

        # Should have baseline + 2 experiments (one crashed, one succeeded)
        assert tracker.num_experiments == 3
        assert tracker.num_crashed >= 1

    @pytest.mark.asyncio
    async def test_keep_and_discard_logic(
        self, stub_agent, stub_environment, tmp_path
    ):
        """Verify that better results are kept and worse ones discarded."""
        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=4,
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=1,
            save_checkpoints=False,
        )

        # Reward fn that returns different scores based on call count
        scores = iter([0.5, 0.7, 0.3, 0.8])  # baseline, exp1(better), exp2(worse), exp3(best)

        reward_fn = MagicMock()

        async def compute_reward(turns, context=None):
            return MagicMock(score=next(scores))

        reward_fn.compute_reward = compute_reward

        with patch.object(
            AutoResearchLoop, "_train_with_params", new_callable=AsyncMock
        ):
            tracker = await run_auto_research(
                agent=stub_agent,
                environment=stub_environment,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=reward_fn,
                config=config,
            )

        statuses = [r.status for r in tracker.records]
        # baseline is always kept; exp1 is better (keep); exp2 is worse (discard); exp3 is best (keep)
        assert statuses[0] == "keep"  # baseline
        assert tracker.num_kept >= 2  # at least baseline + one improvement

    @pytest.mark.asyncio
    async def test_resume_from_previous_run(
        self, stub_agent, stub_environment, stub_reward_fn, tmp_path
    ):
        """Verify that the loop can resume from a previous run."""
        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=3,
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=1,
            save_checkpoints=False,
        )

        # Run first batch
        with patch.object(
            AutoResearchLoop, "_train_with_params", new_callable=AsyncMock
        ):
            tracker1 = await run_auto_research(
                agent=stub_agent,
                environment=stub_environment,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=stub_reward_fn,
                config=config,
            )

        assert tracker1.num_experiments == 3

        # Run second batch — should resume
        config2 = AutoResearchConfig(
            time_budget=10,
            max_experiments=5,  # 3 from resume + 2 new
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=1,
            save_checkpoints=False,
        )

        with patch.object(
            AutoResearchLoop, "_train_with_params", new_callable=AsyncMock
        ):
            tracker2 = await run_auto_research(
                agent=stub_agent,
                environment=stub_environment,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=stub_reward_fn,
                config=config2,
            )

        # Should have resumed and run 2 more experiments
        assert tracker2.num_experiments == 5


    @pytest.mark.asyncio
    async def test_baseline_respects_wall_clock_budget(
        self, stub_agent, stub_environment, stub_reward_fn, tmp_path
    ):
        """Baseline evaluation should stop once the wall-clock budget is spent."""
        config = AutoResearchConfig(
            time_budget=10,
            max_wall_clock=1,
            max_experiments=1,
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=1,
            save_checkpoints=False,
        )

        async def slow_evaluate(self):
            await asyncio.sleep(1.2)
            return {"eval_reward": 0.5}

        with patch.object(AutoResearchLoop, "_evaluate", new=slow_evaluate):
            tracker = await run_auto_research(
                agent=stub_agent,
                environment=stub_environment,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=stub_reward_fn,
                config=config,
            )

        assert tracker.num_experiments == 1
        baseline = tracker.records[0]
        assert baseline.experiment_id == "baseline"
        assert baseline.status == "crash"
        assert "timeout" in baseline.description

    @pytest.mark.asyncio
    async def test_experiment_eval_respects_time_budget(
        self, stub_agent, stub_environment, stub_reward_fn, tmp_path
    ):
        """Experiment evaluation should be covered by the shared time budget."""
        config = AutoResearchConfig(
            time_budget=1,
            max_experiments=2,
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=1,
            save_checkpoints=False,
        )

        calls = 0

        async def evaluate_with_slow_second_call(self):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"eval_reward": 0.5}
            await asyncio.sleep(1.2)
            return {"eval_reward": 0.6}

        with patch.object(
            AutoResearchLoop, "_train_with_params", new_callable=AsyncMock
        ), patch.object(
            AutoResearchLoop, "_evaluate", new=evaluate_with_slow_second_call
        ):
            tracker = await run_auto_research(
                agent=stub_agent,
                environment=stub_environment,
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=stub_reward_fn,
                config=config,
            )

        assert tracker.num_experiments == 2
        experiment = tracker.records[1]
        assert experiment.experiment_id == "exp_0001"
        assert experiment.status == "crash"
        assert "timeout" in experiment.description


class TestSearchSpaces:
    """Test auto-research specific search spaces."""

    def test_auto_research_space(self):
        from stateset_agents.training.auto_research.search_spaces import (
            create_auto_research_search_space,
        )

        space = create_auto_research_search_space()
        assert len(space.dimensions) > 10  # Comprehensive

        # Check key dimensions exist
        names = {d.name for d in space.dimensions}
        assert "learning_rate" in names
        assert "lora_r" in names
        assert "temperature" in names
        assert "num_generations" in names

    def test_quick_space(self):
        space = create_quick_search_space()
        assert len(space.dimensions) == 4

    def test_list_spaces(self):
        from stateset_agents.training.auto_research.search_spaces import (
            list_auto_research_search_spaces,
        )

        spaces = list_auto_research_search_spaces()
        assert "auto_research" in spaces
        assert "quick" in spaces
        assert "reward" in spaces
        assert "model" in spaces

    def test_get_space(self):
        from stateset_agents.training.auto_research.search_spaces import (
            get_auto_research_search_space,
        )

        space = get_auto_research_search_space("quick")
        assert len(space.dimensions) == 4

        with pytest.raises(ValueError, match="Unknown"):
            get_auto_research_search_space("nonexistent")


class TestLLMProposer:
    """Test LLM proposer (mocked — no actual API calls)."""

    def test_format_functions(self):
        from stateset_agents.training.auto_research.llm_proposer import (
            _extract_json,
            _format_best_config,
            _format_history,
        )

        # Test JSON extraction
        assert _extract_json('{"a": 1}') == {"a": 1}
        assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
        assert _extract_json('```\n{"a": 1}\n```') == {"a": 1}

        # Test history formatting
        history = [
            {"params": {"lr": 0.001}, "objective": 0.5, "status": "keep"},
            {"params": {"lr": 0.01}, "objective": 0.3, "status": "discard"},
        ]
        formatted = _format_history(history)
        assert "0.500000" in formatted
        assert "[KEPT]" in formatted

        # Test empty history
        assert "(no experiments yet)" in _format_history([])

        # Test best config
        formatted = _format_best_config({"lr": 0.001})
        assert "lr" in formatted

    def test_llm_proposer_with_mock(self):
        from stateset_agents.training.auto_research.llm_proposer import LLMProposer

        space = create_quick_search_space()
        proposer = LLMProposer(search_space=space, backend="anthropic")

        # Mock the client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"params": {"learning_rate": 1e-4, "num_generations": 8, "temperature": 0.5, "lora_r": 16}, "description": "increase LR and generations for more exploration"}'
            )
        ]
        mock_client.messages.create.return_value = mock_response
        proposer._client = mock_client

        params, desc = proposer.propose(
            current_best={"learning_rate": 5e-6, "num_generations": 4},
            history=[],
        )

        assert params["learning_rate"] == 1e-4
        assert params["num_generations"] == 8
        assert "llm:" in desc

    def test_llm_proposer_clamps_bounds(self):
        from stateset_agents.training.auto_research.llm_proposer import LLMProposer

        space = create_quick_search_space()
        proposer = LLMProposer(search_space=space, backend="anthropic")

        mock_client = MagicMock()
        mock_response = MagicMock()
        # LLM proposes out-of-bounds values
        mock_response.content = [
            MagicMock(
                text='{"params": {"learning_rate": 999.0, "num_generations": 100, "temperature": -5.0, "lora_r": 4}, "description": "extreme values"}'
            )
        ]
        mock_client.messages.create.return_value = mock_response
        proposer._client = mock_client

        params, _ = proposer.propose(
            current_best={"learning_rate": 5e-6, "num_generations": 4},
            history=[],
        )

        # Values should be clamped
        assert params["learning_rate"] <= 1e-3  # max from search space
        assert params["num_generations"] <= 16
        assert params["temperature"] >= 0.1

    def test_llm_proposer_fallback_on_error(self):
        from stateset_agents.training.auto_research.llm_proposer import LLMProposer

        space = create_quick_search_space()
        proposer = LLMProposer(search_space=space, backend="anthropic")

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")
        proposer._client = mock_client

        params, desc = proposer.propose(
            current_best={"learning_rate": 5e-6, "num_generations": 4},
            history=[],
        )

        # Should fall back to perturbation
        assert "llm-fallback:" in desc
        assert isinstance(params, dict)
