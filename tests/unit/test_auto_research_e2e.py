"""
End-to-end test of auto_research using a real stub agent.

This test creates a real MultiTurnAgent in stub mode, a real
ConversationEnvironment, and a real reward function, then runs the
full autonomous research loop — verifying the entire pipeline works
end-to-end without a GPU.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import (
    CompositeReward,
    HelpfulnessReward,
    SafetyReward,
)
from stateset_agents.training.auto_research import (
    AutoResearchConfig,
    AutoResearchLoop,
    run_auto_research,
)
from stateset_agents.training.auto_research.proposer import ExperimentProposer


TRAIN_SCENARIOS = [
    {
        "topic": "general_inquiry",
        "context": "Customer has a general question.",
        "user_responses": ["Hi!", "Thanks."],
    },
]

EVAL_SCENARIOS = [
    {
        "topic": "order_status",
        "context": "Customer wants order status.",
        "user_responses": ["Where is my order?", "OK thanks."],
    },
    {
        "topic": "product_return",
        "context": "Customer wants to return a product.",
        "user_responses": ["I want to return this.", "It's broken."],
    },
]


class NoOpProposer(ExperimentProposer):
    """Proposer that returns slightly varied params (no real training changes)."""

    def __init__(self):
        self._call = 0

    def propose(self, current_best, history):
        self._call += 1
        params = dict(current_best)
        params["temperature"] = 0.5 + (self._call * 0.1)
        return params, f"noop #{self._call}: temp={params['temperature']:.1f}"


async def _make_stub_agent() -> MultiTurnAgent:
    config = AgentConfig(
        model_name="stub://auto-research-e2e",
        use_stub_model=True,
        stub_responses=[
            "I'd be happy to help you with that! Let me check.",
            "Thank you for reaching out. I understand your concern.",
            "I've looked into this and here's what I found.",
            "Is there anything else I can assist you with?",
        ],
        system_prompt="You are a helpful customer service agent.",
        temperature=0.7,
        max_new_tokens=64,
    )
    agent = MultiTurnAgent(config)
    await agent.initialize()
    return agent


def _make_env() -> ConversationEnvironment:
    return ConversationEnvironment(scenarios=TRAIN_SCENARIOS, max_turns=4)


def _make_reward() -> CompositeReward:
    return CompositeReward([
        HelpfulnessReward(weight=0.6),
        SafetyReward(weight=0.4),
    ])


class TestAutoResearchE2E:
    """End-to-end tests with real stub agent and environment."""

    @pytest.mark.asyncio
    async def test_full_loop_with_stub_agent(self, tmp_path):
        """Run the full loop with a real stub agent.

        Training is a no-op (stub agents can't be trained), but evaluation
        runs for real with the stub agent + real reward function.
        """
        agent = await _make_stub_agent()
        config = AutoResearchConfig(
            time_budget=30,
            max_experiments=4,
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=2,
            eval_seed=42,
            save_checkpoints=False,
        )

        with patch.object(
            AutoResearchLoop, "_train_with_params", new_callable=AsyncMock
        ):
            tracker = await run_auto_research(
                agent=agent,
                environment=_make_env(),
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=_make_reward(),
                config=config,
                baseline_params={"learning_rate": 5e-6, "temperature": 0.7},
                proposer=NoOpProposer(),
            )

        assert tracker.num_experiments == 4
        assert tracker.best_value is not None
        assert tracker.best_value > 0

        assert (tmp_path / "results.tsv").exists()
        assert (tmp_path / "experiments.jsonl").exists()

        records = []
        with open(tmp_path / "experiments.jsonl") as f:
            for line in f:
                records.append(json.loads(line))
        assert len(records) == 4
        assert records[0]["experiment_id"] == "baseline"
        assert records[0]["status"] == "keep"

    @pytest.mark.asyncio
    async def test_baseline_evaluation_produces_real_reward(self, tmp_path):
        """Verify the baseline evaluation produces a real reward score.

        Only runs the baseline (max_experiments=1), so no training is needed.
        """
        agent = await _make_stub_agent()
        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=1,
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=3,
            eval_seed=42,
            save_checkpoints=False,
        )

        with patch.object(
            AutoResearchLoop, "_train_with_params", new_callable=AsyncMock
        ):
            tracker = await run_auto_research(
                agent=agent,
                environment=_make_env(),
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=_make_reward(),
                config=config,
                proposer=NoOpProposer(),
            )

        baseline = tracker.records[0]
        assert baseline.experiment_id == "baseline"
        assert baseline.objective_value > 0
        assert "eval_reward" in baseline.metrics
        assert "eval_reward_std" in baseline.metrics
        assert "eval_success_rate" in baseline.metrics
        assert "eval_episode_length" in baseline.metrics

    @pytest.mark.asyncio
    async def test_resume_with_real_agent(self, tmp_path):
        """Run, stop, and resume with a real agent."""
        # First run: 3 experiments
        agent1 = await _make_stub_agent()
        config1 = AutoResearchConfig(
            time_budget=10,
            max_experiments=3,
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=2,
            save_checkpoints=False,
        )

        with patch.object(
            AutoResearchLoop, "_train_with_params", new_callable=AsyncMock
        ):
            tracker1 = await run_auto_research(
                agent=agent1,
                environment=_make_env(),
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=_make_reward(),
                config=config1,
                baseline_params={"learning_rate": 5e-6},
                proposer=NoOpProposer(),
            )
        assert tracker1.num_experiments == 3

        # Second run: resume and run 2 more
        agent2_config = AgentConfig(
            model_name="stub://auto-research-e2e-resume",
            use_stub_model=True,
            stub_responses=["Resumed response."],
            temperature=0.7,
            max_new_tokens=64,
        )
        agent2 = MultiTurnAgent(agent2_config)
        await agent2.initialize()

        config2 = AutoResearchConfig(
            time_budget=10,
            max_experiments=5,
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=2,
            save_checkpoints=False,
        )

        with patch.object(
            AutoResearchLoop, "_train_with_params", new_callable=AsyncMock
        ):
            tracker2 = await run_auto_research(
                agent=agent2,
                environment=_make_env(),
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=_make_reward(),
                config=config2,
                proposer=NoOpProposer(),
            )

        assert tracker2.num_experiments == 5

        records = []
        with open(tmp_path / "experiments.jsonl") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        assert len(records) == 5

    @pytest.mark.asyncio
    async def test_minimize_direction(self, tmp_path):
        """Verify minimize direction works correctly."""
        agent = await _make_stub_agent()
        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=3,
            direction="minimize",
            objective_metric="eval_reward_std",
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=2,
            save_checkpoints=False,
        )

        with patch.object(
            AutoResearchLoop, "_train_with_params", new_callable=AsyncMock
        ):
            tracker = await run_auto_research(
                agent=agent,
                environment=_make_env(),
                eval_scenarios=EVAL_SCENARIOS,
                reward_fn=_make_reward(),
                config=config,
                proposer=NoOpProposer(),
            )

        assert tracker.num_experiments == 3
        assert tracker.best_value is not None


class TestStubAgentWithoutMock:
    """Test that stub agents work end-to-end without mocking _train_with_params."""

    @pytest.mark.asyncio
    async def test_stub_agent_skips_training(self, tmp_path):
        """Stub agents should be detected and training should be skipped."""
        agent = await _make_stub_agent()
        config = AutoResearchConfig(
            time_budget=10,
            max_experiments=3,
            output_dir=str(tmp_path),
            search_space_name="quick",
            eval_episodes=2,
            save_checkpoints=False,
        )

        # NO mock on _train_with_params — the stub detection should skip it
        tracker = await run_auto_research(
            agent=agent,
            environment=_make_env(),
            eval_scenarios=EVAL_SCENARIOS,
            reward_fn=_make_reward(),
            config=config,
            baseline_params={"learning_rate": 5e-6, "temperature": 0.7},
            proposer=NoOpProposer(),
        )

        assert tracker.num_experiments == 3
        assert tracker.best_value is not None
        assert tracker.best_value > 0


class TestCLIDryRun:
    """Test the CLI auto-research command in dry-run mode."""

    def test_dry_run_shows_config(self):
        from typer.testing import CliRunner

        from stateset_agents.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "auto-research",
            "--dry-run",
            "--proposer", "bayesian",
            "--algorithm", "gspo",
            "--time-budget", "120",
            "--max-experiments", "10",
            "--search-space", "quick",
        ])
        assert result.exit_code == 0
        assert "bayesian" in result.output
        assert "gspo" in result.output
        assert "120s" in result.output
        assert "10" in result.output
        assert "quick" in result.output

    def test_dry_run_with_config_file(self, tmp_path):
        from typer.testing import CliRunner

        from stateset_agents.cli import app

        config_path = tmp_path / "test_config.json"
        config_path.write_text(json.dumps({
            "auto_research": {
                "time_budget": 60,
                "proposer": "grid",
                "algorithm": "grpo",
            }
        }))

        runner = CliRunner()
        result = runner.invoke(app, [
            "auto-research",
            "--dry-run",
            "--config", str(config_path),
        ])
        assert result.exit_code == 0
        assert "grid" in result.output
        assert "grpo" in result.output

    def test_dry_run_shows_resume_info(self, tmp_path):
        """If a previous run exists, dry-run should show resume info."""
        from typer.testing import CliRunner

        from stateset_agents.cli import app

        # Create fake previous run
        jsonl_path = tmp_path / "experiments.jsonl"
        jsonl_path.write_text(
            '{"experiment_id": "baseline"}\n'
            '{"experiment_id": "exp_0001"}\n'
            '{"experiment_id": "exp_0002"}\n'
        )

        runner = CliRunner()
        result = runner.invoke(app, [
            "auto-research",
            "--dry-run",
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
        assert "3 previous experiments found" in result.output
