"""End-to-end: MultiTurn GRPO trains on a ToolUseEnvironment with a verifier.

A tiny random GPT-2 ``MultiTurnAgent`` plays calculator tool-use episodes
scored by ``MathAnswerVerifier`` through the trainer's real rollout path
(``generate_trajectories`` -> ``ToolUseEnvironment`` -> ``score_episode``) and
takes optimizer steps on the native token path (``training_step``).

* ``ForcedTurnAgent`` scripts the reply text (correct and wrong tool use, a
  malformed call, a direct guess) so rewards vary within each group, while the
  ids and log-probs the trainer consumes are the model's own;
* the plain sampling agent checks the run is deterministic per seed.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.core.environments.tool_use import ToolUseEnvironment  # noqa: E402
from stateset_agents.core.tool_registry import calculator_tool  # noqa: E402
from stateset_agents.rewards.math_verifier import MathAnswerVerifier  # noqa: E402
from stateset_agents.training.config import TrainingConfig  # noqa: E402
from stateset_agents.training.loss_computation import (  # noqa: E402
    _token_rows_for_group,
)
from stateset_agents.training.multi_turn_trainer import (  # noqa: E402
    MultiTurnGRPOTrainer,
)
from tests._tool_use_helpers import (  # noqa: E402
    SYSTEM_PROMPT,
    ForcedTurnAgent,
    calculator_script,
    make_agent,
    make_model,
    make_tokenizer,
    tool_call,
)

pytestmark = pytest.mark.integration

SCENARIOS = [
    {"id": "mul", "question": "What is 12*7?", "ground_truth": "84"},
    {"id": "add", "question": "What is 35+48?", "ground_truth": "83"},
]
STEPS = 3


def _script_for(expression: str, answer: str) -> list[list[str]]:
    wrong = str(int(answer) + 1)
    return [
        [tool_call(expression), f"The answer is \\boxed{{{answer}}}."],
        [tool_call(f"{expression} "), f"The answer is \\boxed{{{wrong}}}."],
        ['<tool_call>{"name": "calc"}</tool_call>', f"\\boxed{{{answer}}}"],
        [f"\\boxed{{{wrong}}}"],
    ]


SCRIPTS = {
    "mul": _script_for("12*7", "84"),
    "add": _script_for("35+48", "83"),
}


class _PerScenarioScript:
    """Route each episode to its scenario's script by the question asked."""

    def __init__(self) -> None:
        self.scripts = {
            sid: calculator_script(replies) for sid, replies in SCRIPTS.items()
        }

    def __call__(self, history: list[dict[str, Any]], call: int) -> str:
        question = next(m["content"] for m in history if m["role"] == "user")
        sid = next(s["id"] for s in SCENARIOS if s["question"] == question)
        return self.scripts[sid](history, call)


def _config(seed: int) -> TrainingConfig:
    return TrainingConfig(
        num_episodes=STEPS,
        num_generations=4,
        learning_rate=1e-2,
        weight_decay=0.0,
        gradient_accumulation_steps=1,
        max_prompt_length=512,
        max_completion_length=64,
        beta=0.0,
        zero_signal_patience=0,
        bf16=False,
        fp16=False,
        report_to="none",
        seed=seed,
        lr_scheduler_type="constant",
    )


def _train(agent: Any, seed: int) -> dict[str, Any]:
    environment = ToolUseEnvironment(
        SCENARIOS,
        [calculator_tool()],
        max_tool_calls=2,
        system_prompt=SYSTEM_PROMPT,
    )
    trainer = MultiTurnGRPOTrainer(
        agent=agent,
        environment=environment,
        reward_fn=MathAnswerVerifier(),
        config=_config(seed),
    )
    before = [p.detach().clone() for p in agent.model.parameters()]
    rewards: list[list[float]] = []
    metrics: list[dict[str, Any]] = []
    groups_seen: list[Any] = []

    async def run() -> None:
        await trainer.initialize()
        torch.manual_seed(seed)
        for _ in range(STEPS):
            groups = await trainer.generate_trajectories(SCENARIOS, num_generations=4)
            groups_seen.extend(groups)
            rewards.append([t.total_reward for g in groups for t in g.trajectories])
            metrics.append(await trainer.training_step(groups))

    asyncio.run(run())
    delta = sum(
        float((p.detach() - b).abs().sum())
        for p, b in zip(agent.model.parameters(), before, strict=True)
    )
    texts = [
        [t.content for t in traj.turns] for g in groups_seen for traj in g.trajectories
    ]
    return {
        "rewards": rewards,
        "metrics": metrics,
        "delta": delta,
        "groups": groups_seen,
        "texts": texts,
    }


def _forced(seed: int) -> ForcedTurnAgent:
    tokenizer = make_tokenizer()
    return ForcedTurnAgent(
        make_agent(make_model(seed), tokenizer), _PerScenarioScript()
    )


@pytest.fixture(scope="module")
def forced_run() -> dict[str, Any]:
    return _train(_forced(0), seed=0)


def test_verifier_rewards_flow_into_token_path_updates(forced_run):
    for step_rewards in forced_run["rewards"]:
        assert step_rewards == [1.0, 0.0, 1.0, 0.0] * 2
    for metrics in forced_run["metrics"]:
        assert metrics["path"] == "token"
        assert metrics["optimizer_step"] is True
        assert metrics["nonzero_advantage_fraction"] == 1.0
    assert forced_run["delta"] > 0.0


def test_loss_counts_only_the_policys_tokens(forced_run):
    for group in forced_run["groups"]:
        rows = _token_rows_for_group(group)
        assert rows is not None
        observations = [
            t.content
            for traj in group.trajectories
            for t in traj.turns
            if t.role == "tool"
        ]
        assert len(observations) == 3
        replies = [
            t
            for traj in group.trajectories
            for t in traj.turns
            if t.role == "assistant"
        ]
        # Each row's response is exactly a reply's sampled ids.
        assert [list(r) for _, r, _ in rows] == [
            t.metadata["token_ids"] for t in replies
        ]
    # The loss's token count is the replies' tokens, no observation token.
    for step, metrics in enumerate(forced_run["metrics"]):
        groups = forced_run["groups"][2 * step : 2 * step + 2]
        reply_tokens = sum(
            len(t.metadata["token_ids"])
            for g in groups
            for traj in g.trajectories
            for t in traj.turns
            if t.role == "assistant"
        )
        assert metrics["num_tokens"] == reply_tokens


def test_forced_runs_are_deterministic_per_seed(forced_run):
    again = _train(_forced(0), seed=0)
    assert again["rewards"] == forced_run["rewards"]
    # Same rewards and texts exactly; float sums to rounding (kernels may
    # reorder reductions between runs in one process).
    assert again["texts"] == forced_run["texts"]
    assert again["delta"] == pytest.approx(forced_run["delta"], rel=1e-6)
    for a, b in zip(again["metrics"], forced_run["metrics"], strict=True):
        assert float(a["total_loss"]) == pytest.approx(float(b["total_loss"]), rel=1e-5)
    other = _train(_forced(1), seed=1)
    assert other["delta"] != pytest.approx(forced_run["delta"], rel=1e-3)


def _sampled(seed: int) -> Any:
    return make_agent(make_model(seed), make_tokenizer(), max_new_tokens=24)


def test_sampled_rollouts_are_deterministic_per_seed():
    first = _train(_sampled(0), seed=0)
    second = _train(_sampled(0), seed=0)
    assert first["texts"] == second["texts"]
    assert first["delta"] == pytest.approx(second["delta"], rel=1e-6)
    assert all(m["path"] == "token" for m in first["metrics"])
    # A random policy's text is parsed without errors; every episode ended.
    for group in first["groups"]:
        for trajectory in group.trajectories:
            assert trajectory.final_state["termination"] in {
                "final_answer",
                "budget_exhausted",
                "max_turns",
            }
    third = _train(_sampled(1), seed=1)
    assert third["texts"] != first["texts"]
