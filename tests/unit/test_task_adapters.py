"""Unit tests for the Phase 0 runner's ``TaskAdapter`` interface.

Verifies that ``GSM8KAdapter`` and ``CustomerSupportAdapter`` honor the same
contract: ``load``, ``format_prompt``, ``score_response``, and ``max_new_tokens``
return sensible values on representative inputs.

These tests run in seconds, no GPU, no network — they exercise the small
amount of glue code the runner depends on, complementing the dataset-level
tests in ``test_gsm8k.py`` and ``test_customer_support_bench.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import run_phase0_benchmark as runner  # noqa: E402


class TestGSM8KAdapter:
    @pytest.fixture
    def adapter(self) -> "runner.GSM8KAdapter":
        return runner.GSM8KAdapter()

    def test_name_and_token_cap(self, adapter: "runner.GSM8KAdapter") -> None:
        assert adapter.name == "gsm8k"
        assert adapter.max_new_tokens > 0

    def test_format_prompt(self, adapter: "runner.GSM8KAdapter") -> None:
        from stateset_agents.data.gsm8k import GSM8KExample

        ex = GSM8KExample(
            question="If a train travels 60 mph for 2 hours, how far?",
            answer_text="60 * 2 = 120 #### 120",
            gold_answer=120.0,
        )
        prompt = adapter.format_prompt(ex)
        assert "60 mph for 2 hours" in prompt
        assert "step by step" in prompt.lower()
        assert "Answer:" in prompt

    def test_score_correct(self, adapter: "runner.GSM8KAdapter") -> None:
        from stateset_agents.data.gsm8k import GSM8KExample

        ex = GSM8KExample(question="Q?", answer_text="A. #### 42", gold_answer=42.0)
        score, parseable = adapter.score_response(ex, "Working it out... the answer is 42.")
        assert score == 1.0
        assert parseable is True

    def test_score_incorrect(self, adapter: "runner.GSM8KAdapter") -> None:
        from stateset_agents.data.gsm8k import GSM8KExample

        ex = GSM8KExample(question="Q?", answer_text="A. #### 42", gold_answer=42.0)
        score, parseable = adapter.score_response(ex, "The answer is 41.")
        assert score == 0.0
        assert parseable is True

    def test_score_unparseable(self, adapter: "runner.GSM8KAdapter") -> None:
        from stateset_agents.data.gsm8k import GSM8KExample

        ex = GSM8KExample(question="Q?", answer_text="A. #### 42", gold_answer=42.0)
        score, parseable = adapter.score_response(ex, "I'm not sure.")
        assert score == 0.0
        assert parseable is False


class TestCustomerSupportAdapter:
    @pytest.fixture
    def adapter(self) -> "runner.CustomerSupportAdapter":
        return runner.CustomerSupportAdapter()

    def test_name_and_token_cap(self, adapter: "runner.CustomerSupportAdapter") -> None:
        assert adapter.name == "customer_support"
        assert adapter.max_new_tokens > 0

    def test_load_respects_split(self, adapter: "runner.CustomerSupportAdapter") -> None:
        train, eval_ = adapter.load(n_train=16, n_eval=8)
        assert len(train) == 16
        assert len(eval_) == 8
        # No overlap.
        train_queries = {s.user_query for s in train}
        eval_queries = {s.user_query for s in eval_}
        assert not (train_queries & eval_queries)

    def test_load_caps_to_corpus(self, adapter: "runner.CustomerSupportAdapter") -> None:
        # Bundled corpus is 24; asking for 100 train + 100 eval should clamp.
        train, eval_ = adapter.load(n_train=100, n_eval=100)
        assert len(train) + len(eval_) <= 24

    def test_format_prompt(self, adapter: "runner.CustomerSupportAdapter") -> None:
        from stateset_agents.data.customer_support_bench import SupportScenario

        s = SupportScenario(
            intent="refund",
            user_query="I want my money back",
            must_acknowledge=["refund"],
        )
        prompt = adapter.format_prompt(s)
        assert "customer support agent" in prompt.lower()
        assert "I want my money back" in prompt
        assert "Agent:" in prompt

    def test_score_returns_composite_value(
        self, adapter: "runner.CustomerSupportAdapter"
    ) -> None:
        from stateset_agents.data.customer_support_bench import SupportScenario

        s = SupportScenario(
            intent="refund",
            user_query="I want a refund",
            must_acknowledge=["refund", "order"],
            must_avoid=["impossible"],
        )
        score, parseable = adapter.score_response(
            s,
            "I'm happy to help with your refund for that order. "
            "Could you share your order number please?",
        )
        assert 0.0 <= score <= 1.0
        assert parseable is True

    def test_score_safety_failure_zeros(
        self, adapter: "runner.CustomerSupportAdapter"
    ) -> None:
        from stateset_agents.data.customer_support_bench import SupportScenario

        s = SupportScenario(
            intent="refund",
            user_query="Refund please",
            must_acknowledge=["refund"],
        )
        # The "password is" red flag should zero the composite via safety gate.
        score, parseable = adapter.score_response(
            s, "Your password is hunter2. I'll refund."
        )
        assert score == 0.0


class TestToolCallingAdapter:
    @pytest.fixture
    def adapter(self) -> "runner.ToolCallingAdapter":
        return runner.ToolCallingAdapter()

    def test_name_and_token_cap(self, adapter: "runner.ToolCallingAdapter") -> None:
        assert adapter.name == "tool_calling"
        assert adapter.max_new_tokens > 0

    def test_load_splits_correctly(self, adapter: "runner.ToolCallingAdapter") -> None:
        train, eval_ = adapter.load(n_train=5, n_eval=3)
        assert len(train) == 5
        assert len(eval_) == 3
        train_queries = {s.user_query for s in train}
        eval_queries = {s.user_query for s in eval_}
        assert not (train_queries & eval_queries)

    def test_format_prompt(self, adapter: "runner.ToolCallingAdapter") -> None:
        from stateset_agents.data.tool_calling_bench import ToolCallScenario
        s = ToolCallScenario(
            user_query="What's the weather in NYC?",
            expected_tool="get_weather",
            expected_params={"city": "NYC"},
        )
        prompt = adapter.format_prompt(s)
        assert "What's the weather in NYC?" in prompt
        assert "```json" in prompt
        assert "tool" in prompt.lower()

    def test_score_correct_tool_call(self, adapter: "runner.ToolCallingAdapter") -> None:
        from stateset_agents.data.tool_calling_bench import ToolCallScenario
        s = ToolCallScenario(
            user_query="Compute 5 + 3",
            expected_tool="calculator",
            expected_params={"expression": "5 + 3"},
            expected_outcome="8",
        )
        response = (
            '```json\n{"tool": "calculator", "parameters": {"expression": "5 + 3"}}\n```\n'
            "Result: 8"
        )
        score, parseable = adapter.score_response(s, response)
        assert score == pytest.approx(1.0)
        assert parseable is True


class TestTaskRegistry:
    def test_registry_has_all_three_tasks(self) -> None:
        assert "gsm8k" in runner.TASKS
        assert "customer_support" in runner.TASKS
        assert "tool_calling" in runner.TASKS

    def test_registry_values_instantiate(self) -> None:
        for name, cls in runner.TASKS.items():
            adapter = cls()
            assert adapter.name == name

    def test_unknown_trainer_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown trainer"):
            runner.build_trainer_config("not-a-trainer")

    def test_trainer_config_includes_overrides(self) -> None:
        cfg = runner.build_trainer_config("gspo", model_name="some-model", learning_rate=1e-7)
        assert cfg["model_name"] == "some-model"
        assert cfg["learning_rate"] == 1e-7
        # GSPO-specific defaults still present.
        assert cfg["clip_range_left"] == 3e-4

    def test_get_git_commit_returns_string(self) -> None:
        commit = runner.get_git_commit()
        assert isinstance(commit, str)
        assert len(commit) > 0


class TestBuildEnvReward:
    """Verify the internal helper that wires task → env+reward for training."""

    def test_gsm8k(self) -> None:
        from stateset_agents.data.gsm8k import GSM8KExample
        adapter = runner.GSM8KAdapter()
        examples = [GSM8KExample(question="Q?", answer_text="A. #### 1", gold_answer=1.0)]
        env, reward, scenarios = runner._build_env_reward(adapter, examples)
        assert env is not None
        assert reward.name == "gsm8k"
        assert len(scenarios) == 1

    def test_customer_support(self) -> None:
        adapter = runner.CustomerSupportAdapter()
        from stateset_agents.data.customer_support_bench import load_support_scenarios
        examples = load_support_scenarios(limit=2)
        env, reward, scenarios = runner._build_env_reward(adapter, examples)
        assert reward.name == "support_composite"
        assert len(scenarios) == 2

    def test_tool_calling(self) -> None:
        adapter = runner.ToolCallingAdapter()
        from stateset_agents.data.tool_calling_bench import load_tool_call_scenarios
        examples = load_tool_call_scenarios(limit=3)
        env, reward, scenarios = runner._build_env_reward(adapter, examples)
        assert reward.name == "tool_call_composite"
        assert len(scenarios) == 3

    def test_unknown_task_raises(self) -> None:
        class BogusAdapter(runner.TaskAdapter):
            name = "bogus"
        with pytest.raises(ValueError, match="Unknown task"):
            runner._build_env_reward(BogusAdapter(), [])


class TestTrainWithTrainer:
    """The trainer dispatcher — exercise the recognition and error paths."""

    def test_unknown_trainer_returns_error(self) -> None:
        adapter = runner.GSM8KAdapter()
        agent, seconds, err = runner.train_with_trainer(
            trainer="not-a-trainer",
            model_name="stub://test",
            adapter=adapter,
            train_examples=[],
            seed=42,
            output_dir="/tmp/no-such-dir",
        )
        assert agent is None
        assert err is not None
        assert "not recognized" in err

    def test_returns_tuple_shape(self) -> None:
        # gspo path — will fail without GPU/torch fully wired, but the
        # function should always return the 3-tuple shape.
        adapter = runner.GSM8KAdapter()
        result = runner.train_with_trainer(
            trainer="gspo",
            model_name="stub://test",
            adapter=adapter,
            train_examples=[],
            seed=42,
            output_dir="/tmp/no-such-dir",
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        agent, seconds, err = result
        assert isinstance(seconds, float)


class TestEndToEndRunner:
    """Verify the runner's smoke-test path produces the expected output."""

    def test_smoke_test_emits_expected_log(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        import subprocess

        result = subprocess.run(
            [
                sys.executable, str(SCRIPT_DIR / "run_phase0_benchmark.py"),
                "--trainer", "gspo",
                "--task", "gsm8k",
                "--smoke-test",
                "--output", str(tmp_path / "smoke.json"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        # The runner uses logger.info — output lands on stderr by default.
        assert "Smoke test passed" in (result.stdout + result.stderr)

    def test_customer_support_smoke_test(self, tmp_path: Path) -> None:
        import subprocess

        result = subprocess.run(
            [
                sys.executable, str(SCRIPT_DIR / "run_phase0_benchmark.py"),
                "--trainer", "dapo",
                "--task", "customer_support",
                "--num-train-examples", "16",
                "--num-eval-examples", "8",
                "--smoke-test",
                "--output", str(tmp_path / "smoke.json"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "task=customer_support" in (result.stdout + result.stderr)
