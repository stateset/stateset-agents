"""GSPO trains on the real task prompts with the reward context the task needs.

Until this fix, ``train_with_gspo`` derived queries from ``scenario["context"]``
(falling back to the literal prompt "Hello") and only from the first
``generations_per_iteration`` scenarios, and passed the reward no scenario
fields. For GSM8K-shaped scenarios (``user_query``/``gold_answer``) that meant
training on "Hello" with a reward that never saw the gold answer: identically
zero reward, zero loss, no learning, for every retained native-GSPO run.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from stateset_agents.training import gspo_entrypoints as ep
from stateset_agents.training.trl_grpo_entrypoints import reward_history_from_log


def test_gsm8k_shaped_scenarios_become_prompt_plus_reward_context():
    scenarios = [
        {"user_query": "What is 2+2?", "gold_answer": 4.0, "answer_text": "4"},
        {"user_query": "What is 3+3?", "gold_answer": 6.0, "answer_text": "6"},
    ]
    queries = ep.queries_from_scenarios(scenarios)
    assert [q["prompt"] for q in queries] == ["What is 2+2?", "What is 3+3?"]
    assert queries[0]["context"]["gold_answer"] == 4.0
    assert queries[0]["context"]["answer_text"] == "4"
    assert queries[0]["context"]["scenario_index"] == 0
    assert "user_query" not in queries[0]["context"]


def test_multi_turn_scenarios_keep_context_as_prompt_and_merge_task_metadata():
    scenarios = [
        {
            "id": "s1",
            "context": "Customer needs help with an order",
            "user_responses": ["It has not arrived."],
            "task": {"expected_intent": "shipping"},
            "metadata": {"tier": "gold"},
        }
    ]
    (query,) = ep.queries_from_scenarios(scenarios)
    assert query["prompt"] == "Customer needs help with an order"
    ctx = query["context"]
    assert ctx["expected_intent"] == "shipping" and ctx["tier"] == "gold"
    assert ctx["id"] == "s1" and ctx["user_responses"] == ["It has not arrived."]
    assert "task" not in ctx and "metadata" not in ctx


def test_prompt_key_priority_and_missing_prompt_fails_closed():
    (q,) = ep.queries_from_scenarios(
        [{"prompt": "P", "user_query": "U", "context": "C"}]
    )
    assert q["prompt"] == "P"
    with pytest.raises(ValueError, match="no prompt text"):
        ep.queries_from_scenarios([{"gold_answer": 1.0}])
    with pytest.raises(ValueError, match="no prompt text"):
        ep.queries_from_scenarios([{"context": ""}])


def test_query_window_rotates_through_every_query_before_repeating():
    queries = list(range(10))
    assert ep.query_window(queries, 0, 4) == [0, 1, 2, 3]
    assert ep.query_window(queries, 1, 4) == [4, 5, 6, 7]
    assert ep.query_window(queries, 2, 4) == [8, 9, 0, 1]
    assert ep.query_window(queries, 0, 50) == queries
    assert ep.query_window([], 3, 4) == []
    assert ep.query_window(["only"], 7, 0) == ["only"]


def test_reward_identically_zero_detection():
    assert ep._reward_is_identically_zero({"average_reward": 0.0, "reward_std": 0.0})
    assert not ep._reward_is_identically_zero(
        {"average_reward": 0.0, "reward_std": 0.5}
    )
    assert not ep._reward_is_identically_zero(
        {"average_reward": 0.25, "reward_std": 0.0}
    )


def test_trl_reward_history_from_log_state():
    state = SimpleNamespace(
        log_history=[
            {"loss": 0.1, "reward": 0.25, "reward_std": 0.4},
            {"eval_loss": 0.2},
            {"loss": 0.05, "reward": 0.5, "reward_std": 0.3},
        ]
    )
    assert reward_history_from_log(state) == {
        "average_reward": [0.25, 0.5],
        "reward_std": [0.4, 0.3],
    }
    assert reward_history_from_log(None) == {"average_reward": [], "reward_std": []}


class _RecordingReward:
    """Reward that scores 1.0 only when it is handed the gold answer."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def compute_reward(self, turns, context=None):
        self.calls.append(dict(context or {}))
        score = 1.0 if (context or {}).get("gold_answer") is not None else 0.0
        return SimpleNamespace(total_reward=score, score=score)


@pytest.mark.asyncio
async def test_train_with_gspo_hands_the_reward_its_gold_answer(monkeypatch, tmp_path):
    """End to end on the stub backend: prompts are the task questions,
    rotate across iterations, and the reward sees the scenario fields."""
    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
    from stateset_agents.core.environment import ConversationEnvironment
    from stateset_agents.training.gspo_config import GSPOConfig

    seen: list[tuple[list[Any], int]] = []

    class _FakeTrainer:
        def __init__(self, **kwargs):
            self.reward_model = kwargs["reward_model"]
            self.training_metrics = {"average_reward": [], "reward_std": []}
            self.generator = SimpleNamespace()

        async def train_step(self, queries, num_groups=1):
            seen.append((list(queries), num_groups))
            rewards = []
            for q in queries:
                prompt = q["prompt"] if isinstance(q, dict) else q
                ctx = {
                    "user_query": prompt,
                    **(q.get("context", {}) if isinstance(q, dict) else {}),
                }
                info = await self.reward_model.compute_reward(turns=[], context=ctx)
                rewards.append(float(info.total_reward))
            mean = sum(rewards) / len(rewards)
            self.training_metrics["average_reward"].append(mean)
            self.training_metrics["reward_std"].append(0.0)
            return {"average_reward": mean, "reward_std": 0.0}

        def save_model(self, path):
            return None

    import stateset_agents.training.gspo_trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "GSPOTrainer", _FakeTrainer)
    scenarios = [
        {"user_query": f"Q{i}", "gold_answer": float(i), "answer_text": str(i)}
        for i in range(6)
    ]
    env = ConversationEnvironment(scenarios=scenarios, max_turns=1)
    agent = MultiTurnAgent(
        AgentConfig(model_name="stub://x", use_stub_model=True, stub_responses=["4"])
    )
    await agent.initialize()
    reward = _RecordingReward()
    cfg = GSPOConfig(
        model_name="stub://x",
        output_dir=str(tmp_path),
        report_to="none",
        num_outer_iterations=3,
        generations_per_iteration=4,
        save_steps=100,
    )
    trained = await ep.train_with_gspo(
        config=cfg, agent=agent, environment=env, reward_model=reward
    )
    prompts_per_iteration = [[q["prompt"] for q in qs] for qs, _ in seen]
    assert prompts_per_iteration == [
        ["Q0", "Q1", "Q2", "Q3"],
        ["Q4", "Q5", "Q0", "Q1"],
        ["Q2", "Q3", "Q4", "Q5"],
    ]
    assert all(n == 4 for _, n in seen)
    assert reward.calls and all(c["gold_answer"] is not None for c in reward.calls)
    assert trained._training_metrics["average_reward"] == [1.0, 1.0, 1.0]


@pytest.mark.asyncio
async def test_train_with_gspo_warns_when_reward_is_identically_zero(
    monkeypatch, tmp_path, caplog
):
    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
    from stateset_agents.core.environment import ConversationEnvironment
    from stateset_agents.training.gspo_config import GSPOConfig

    class _ZeroTrainer:
        def __init__(self, **kwargs):
            self.training_metrics = {"average_reward": [], "reward_std": []}
            self.generator = SimpleNamespace()

        async def train_step(self, queries, num_groups=1):
            self.training_metrics["average_reward"].append(0.0)
            self.training_metrics["reward_std"].append(0.0)
            return {"average_reward": 0.0, "reward_std": 0.0}

        def save_model(self, path):
            return None

    import stateset_agents.training.gspo_trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "GSPOTrainer", _ZeroTrainer)
    env = ConversationEnvironment(
        scenarios=[{"user_query": "Q", "gold_answer": 1.0}], max_turns=1
    )
    agent = MultiTurnAgent(
        AgentConfig(model_name="stub://x", use_stub_model=True, stub_responses=["x"])
    )
    await agent.initialize()
    cfg = GSPOConfig(
        model_name="stub://x",
        output_dir=str(tmp_path),
        report_to="none",
        num_outer_iterations=7,
        save_steps=100,
    )
    with caplog.at_level(logging.WARNING):
        await ep.train_with_gspo(
            config=cfg, agent=agent, environment=env, reward_model=_RecordingReward()
        )
    warnings = [r for r in caplog.records if "identically zero" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_train_with_gspo_token_derives_queries_and_rotates(monkeypatch, tmp_path):
    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
    from stateset_agents.core.environment import ConversationEnvironment
    from stateset_agents.training import gspo_token_trainer as tok
    from stateset_agents.training.gspo_config import GSPOConfig

    seen: list[list[Any]] = []

    class _FakeTokenTrainer:
        def __init__(self, **kwargs):
            self.training_metrics = {"average_reward": []}

        async def train_step_token_level(self, queries, num_groups=1):
            seen.append(list(queries))
            return {"average_reward": 1.0}

        def save_model(self, path):
            return None

    class _FakeManager:
        ref_model = None

        def __init__(self, config):
            pass

        def load_model_and_tokenizer(self):
            return object(), object()

    import stateset_agents.training.gspo_trainer as gspo_trainer_mod

    monkeypatch.setattr(tok, "GSPOTokenTrainer", _FakeTokenTrainer)
    monkeypatch.setattr(gspo_trainer_mod, "GSPOModelManager", _FakeManager)
    scenarios = [{"user_query": f"Q{i}", "gold_answer": float(i)} for i in range(5)]
    env = ConversationEnvironment(scenarios=scenarios, max_turns=1)
    agent = MultiTurnAgent(
        AgentConfig(model_name="stub://x", use_stub_model=True, stub_responses=["x"])
    )
    await agent.initialize()
    cfg = GSPOConfig(
        model_name="stub://x",
        output_dir=str(tmp_path),
        report_to="none",
        num_outer_iterations=2,
        generations_per_iteration=3,
        save_steps=100,
    )
    await tok.train_with_gspo_token(
        config=cfg, agent=agent, environment=env, reward_model=_RecordingReward()
    )
    assert [[q["prompt"] for q in qs] for qs in seen] == [
        ["Q0", "Q1", "Q2"],
        ["Q3", "Q4", "Q0"],
    ]
    assert seen[0][0]["context"]["gold_answer"] == 0.0


def test_experiment_loop_scenario_prompts_carry_reward_context():
    from stateset_agents.training.auto_research.experiment_loop import AutoResearchLoop

    loop = object.__new__(AutoResearchLoop)
    loop.environment = SimpleNamespace(
        scenarios=[
            {"user_query": "What is 2+2?", "gold_answer": 4.0},
            "plain prompt",
            {"context": "Customer needs help", "user_responses": ["x"]},
        ]
    )
    prompts, contexts = loop._scenario_prompts_and_contexts()
    assert prompts == ["What is 2+2?", "plain prompt", "Customer needs help"]
    assert contexts["What is 2+2?"]["gold_answer"] == 4.0
    assert contexts["What is 2+2?"]["scenario_index"] == 0
    assert "plain prompt" not in contexts
    assert contexts["Customer needs help"]["user_responses"] == ["x"]
