"""The same reply earns the same reward on every training path.

A one-turn episode in ``MultiTurnGRPOTrainer`` is the reference. The
single-response paths (``SingleTurnGRPOTrainer``, ``GSPOTrainer``,
``GSPOTokenTrainer``, the TRL reward bridge in both prompt formats,
``ComputationalGRPOEngine``) and ``Environment.get_reward`` must score the same
reply to the same request identically, for every shipped reward factory, the
core rewards and a model judge. A meta-test pins that these consumers score
through the shared helpers in ``core.reward_context`` instead of calling
``compute_reward`` with their own turns and context.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateset_agents.core.basic_rewards import (
    CorrectnessReward,
    EngagementReward,
    HelpfulnessReward,
    SafetyReward,
    TaskCompletionReward,
)
from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.reward_base import RewardResult, RewardType
from stateset_agents.core.reward_context import score_response
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

from .test_reward_gaming import ALL_FACTORIES

PROMPT = "I was double charged for order #1182, can I get a refund of $42.10?"
REPLY = (
    "Your refund of $42.10 for order #1182 was issued to your Visa ending 4421 "
    "today. It usually posts within 5-7 business days."
)
SCENARIO = {
    "id": "refund",
    "user_query": PROMPT,
    "user_turns": [PROMPT],
    "ground_truth": "$42.10",
}
UNLABELED = {"id": "refund", "user_query": PROMPT, "user_turns": [PROMPT]}


class Judge:
    """An offline model judge that records what it was asked."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return '{"score": 0.8}'


def _judge_reward():
    from stateset_agents.rewards.multi_objective_components import (
        ModelBasedRewardComponent,
    )
    from stateset_agents.rewards.multi_objective_reward import (
        MultiObjectiveRewardFunction,
    )

    return MultiObjectiveRewardFunction(
        [ModelBasedRewardComponent(api_client=Judge(), provider="local", strict=True)]
    )


REWARDS = {
    "CorrectnessReward": CorrectnessReward,
    "HelpfulnessReward": HelpfulnessReward,
    "SafetyReward": SafetyReward,
    "EngagementReward": EngagementReward,
    "TaskCompletionReward": lambda: TaskCompletionReward(
        task_criteria={"goal_keywords": ["refund issued"]}
    ),
    "model_judge": _judge_reward,
    **{
        name.rsplit(".", 1)[1]
        + "@"
        + name.rsplit(".", 2)[-2]: (
            lambda factory=factory, build=build: build(factory, "refund")
        )
        for name, (factory, build) in ALL_FACTORIES.items()
    },
}


def _reply_agent():
    async def respond(history, context=None):
        return REPLY

    return SimpleNamespace(generate_response=respond)


async def _multi_turn(reward, scenario):
    trainer = MultiTurnGRPOTrainer(
        agent=_reply_agent(),
        environment=ConversationEnvironment([scenario], max_turns=1),
        reward_fn=reward,
        config=TrainingConfig(num_generations=1, batched_rollouts=False),
    )
    [group] = await trainer.generate_trajectories([scenario], num_generations=1)
    [trajectory] = group.trajectories
    return trajectory.total_reward


async def _environment_attached(reward, scenario):
    trainer = MultiTurnGRPOTrainer(
        agent=_reply_agent(),
        environment=ConversationEnvironment([scenario], max_turns=1, reward_fn=reward),
        reward_fn=None,
        config=TrainingConfig(num_generations=1, batched_rollouts=False),
    )
    [group] = await trainer.generate_trajectories([scenario], num_generations=1)
    return group.trajectories[0].total_reward


async def _single_turn(reward, scenario):
    from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer

    async def answer(prompt):
        return REPLY

    trainer = SingleTurnGRPOTrainer.__new__(SingleTurnGRPOTrainer)
    trainer.agent = SimpleNamespace(generate_response=answer)
    trainer.reward_fn = reward
    trainer.environment = ConversationEnvironment([])
    group, _ = await trainer._generate_trajectory_group(
        PROMPT, num_generations=1, context=dict(scenario)
    )
    return group.trajectories[0].total_reward


async def _gspo(reward, scenario):
    from stateset_agents.training.gspo_trainer import GSPOTrainer

    trainer = GSPOTrainer.__new__(GSPOTrainer)
    trainer.reward_model = reward
    return await trainer._reward_for_response(PROMPT, dict(scenario), REPLY)


async def _gspo_token(reward, scenario):
    from stateset_agents.training.gspo_token_trainer import GSPOTokenTrainer

    trainer = GSPOTokenTrainer.__new__(GSPOTokenTrainer)
    trainer.reward_model = reward
    return await trainer._reward_for_response(PROMPT, dict(scenario), REPLY)


async def _trl_plain(reward, scenario):
    from stateset_agents.training.trl_grpo_trainer import TRLGRPORewardFunction

    bridge = TRLGRPORewardFunction(
        reward, agent=None, environment=SimpleNamespace(scenarios=[scenario])
    )
    [score] = await bridge.compute_rewards([REPLY], [f"User: {PROMPT}\nAssistant:"])
    return score


async def _trl_conversational(reward, scenario):
    from stateset_agents.training.trl_grpo_trainer import TRLGRPORewardFunction

    bridge = TRLGRPORewardFunction(
        reward, agent=None, environment=SimpleNamespace(scenarios=[scenario])
    )
    [score] = await bridge.compute_rewards(
        [[{"role": "assistant", "content": REPLY}]],
        [[{"role": "user", "content": PROMPT}]],
    )
    return score


async def _engine(reward, scenario):
    from stateset_agents.core.computational_engine import ComputationalGRPOEngine

    async def answer(prompt):
        return REPLY

    engine = ComputationalGRPOEngine(
        agent=SimpleNamespace(generate_response=answer),
        environment=ConversationEnvironment([]),
        reward_function=reward,
        num_workers=1,
    )
    [trajectory] = await engine.generate_trajectory_batch([PROMPT])
    return trajectory.learned_reward


async def _get_reward(reward, scenario):
    environment = ConversationEnvironment([scenario], max_turns=1, reward_fn=reward)

    async def respond(history, context):
        return REPLY

    trajectory = await environment.run_episode(respond, scenario)
    return await environment.get_reward(trajectory)


LABELED_PATHS = {
    "environment_attached": _environment_attached,
    "single_turn": _single_turn,
    "gspo": _gspo,
    "gspo_token": _gspo_token,
    "trl_plain": _trl_plain,
    "trl_conversational": _trl_conversational,
    "get_reward": _get_reward,
}


@pytest.mark.parametrize("path", sorted(LABELED_PATHS))
@pytest.mark.parametrize("reward_name", sorted(REWARDS))
async def test_every_path_scores_a_reply_like_a_one_turn_episode(reward_name, path):
    if path.startswith("trl"):
        pytest.importorskip("trl")
    make = REWARDS[reward_name]
    expected = await _multi_turn(make(), SCENARIO)
    got = await LABELED_PATHS[path](make(), SCENARIO)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12), (
        reward_name,
        path,
        got,
        expected,
    )


@pytest.mark.parametrize("reward_name", sorted(REWARDS))
async def test_engine_scores_a_reply_like_a_one_turn_episode(reward_name):
    # The engine has prompts but no scenario labels.
    make = REWARDS[reward_name]
    expected = await _multi_turn(make(), UNLABELED)
    assert await _engine(make(), UNLABELED) == pytest.approx(
        expected, rel=1e-12, abs=1e-12
    )


async def test_parity_cases_discriminate():
    """The parity set is not vacuous: labels and the request change scores."""
    right = await _gspo(CorrectnessReward(), SCENARIO)
    wrong = await _gspo(CorrectnessReward(), {**SCENARIO, "ground_truth": "$9.99"})
    assert (right, wrong) == (1.0, 0.0)
    judge = _judge_reward()
    assert await _gspo(judge, SCENARIO) == pytest.approx(0.8)
    [component] = judge.components
    assert PROMPT in component.api_client.prompts[0]


async def test_immediate_rewards_get_the_step_scope_on_single_response_paths():
    seen = []

    class Scoped:
        reward_type = RewardType.IMMEDIATE

        async def compute_reward(self, turns, context=None):
            seen.append(((context or {}).get("reward_scope"), [t.role for t in turns]))
            return RewardResult(score=0.5)

    await score_response(Scoped(), PROMPT, REPLY, SCENARIO)
    await _multi_turn(Scoped(), SCENARIO)
    assert seen[0] == seen[1] == ("latest_turn", ["user", "assistant"])


# --- Meta-test: consumers score through the shared helpers ------------------

ROOT = Path(__file__).resolve().parents[2] / "stateset_agents"
REWARD_CALLS = {"compute_reward", "compute_turn_reward"}
#: Direct reward calls that are not a consumer scoring a reply.
ALLOWED = {
    # The environment applies its own reward step by step; score_episode
    # replays exactly those calls, pinned by the reward contract tests.
    ("core/environment_base.py", "Environment._apply_step_reward"),
    ("core/environment_base.py", "Environment._finish_episode"),
    # Reward wrappers delegating to the reward they wrap.
    ("training/reward_calibration.py", "CalibratedRewardFunction.compute_reward"),
    # Keeps the result (for its reward-fallback flag) while score_response
    # scores the reply.
    ("training/spine/scoring.py", "_ResultRecorder.compute_reward"),
    ("training/train.py", "UncertaintyWeightedReward.compute_reward"),
    # Calibration statistics over caller-supplied episodes and contexts.
    ("training/reward_calibration.py", "MultiRewardCalibrator.calibrate"),
    # Data curation (an NSR gate on harvested samples and reward-filtered
    # SFT), not an RL training reward.
    ("training/harvest.py", "nsr_gate_passes"),
    ("training/sft.py", "judge_completion"),
}
CONSUMERS = [
    *sorted((ROOT / "training").rglob("*.py")),
    ROOT / "core" / "computational_engine.py",
    ROOT / "core" / "environment_base.py",
    # Tool-use episodes are scored by the shared environment/trainer path.
    ROOT / "core" / "environments" / "tool_use.py",
]


def _direct_reward_calls(path: Path) -> list[tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, str]] = []

    def visit(node: ast.AST, scope: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, [*scope, child.name])
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in REWARD_CALLS
            ):
                found.append((path.relative_to(ROOT).as_posix(), ".".join(scope)))
            visit(child, scope)

    visit(tree, [])
    return found


def test_consumers_score_through_the_shared_helpers():
    offenders = []
    for path in CONSUMERS:
        for relative, scope in _direct_reward_calls(path):
            if (relative, scope) in ALLOWED:
                continue
            offenders.append(f"{relative}:{scope}")
    assert not offenders, (
        "score replies with reward_context.score_response / score_trajectory / "
        f"score_episode instead of calling the reward directly: {offenders}"
    )


def test_each_scoring_consumer_uses_a_shared_helper():
    # The trainers score through the spine's scorers, which are built only on
    # the reward_context helpers (no parallel scoring path).
    helpers = {"score_response", "score_trajectory", "score_episode"}
    scorers = {"ResponseScorer": "score_response", "EpisodeScorer": "score_episode"}
    spine = (ROOT / "training/spine/scoring.py").read_text(encoding="utf-8")
    for helper in scorers.values():
        assert f"{helper}(" in spine, helper
    consumers = {
        "training/single_turn_trainer.py": "ResponseScorer",
        "training/gspo_trainer.py": "ResponseScorer",
        "training/trl_grpo_trainer.py": "ResponseScorer",
        "training/auto_research/experiment_loop.py": "score_response",
        "core/computational_engine.py": "score_response",
        "core/environment_base.py": "score_trajectory",
        "training/multi_turn_trainer.py": "EpisodeScorer",
        "training/multi_turn_evaluation.py": "score_episode",
        "training/distributed_trainer.py": "EpisodeScorer",
    }
    for relative, helper in consumers.items():
        assert helper in helpers or helper in scorers
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert f"{helper}(" in source, (relative, helper)
    # GSPO-token samples and scores with GSPOTrainer's shared step (its
    # _reward_for_response) instead of a scoring loop of its own.
    token = (ROOT / "training/gspo_token_trainer.py").read_text(encoding="utf-8")
    assert "self._gspo_step(" in token
    assert "_reward_for_response" not in token


async def test_judge_component_never_stands_in_zero_for_a_missing_user_turn():
    from stateset_agents.rewards.multi_objective_components import (
        ModelBasedRewardComponent,
    )

    judge = Judge()
    component = ModelBasedRewardComponent(
        api_client=judge, provider="local", strict=True
    )
    score = await component.compute_score(
        [{"role": "assistant", "content": REPLY}], {"user_query": PROMPT}
    )
    assert score == pytest.approx(0.8)
    assert PROMPT in judge.prompts[0] and REPLY in judge.prompts[0]


async def test_get_reward_scores_evidence_rewards_like_score_episode():
    from stateset_agents.core.reward_context import score_episode
    from stateset_agents.core.task_environment import TaskEnvironment
    from stateset_agents.rewards.task_outcome import TaskOutcomeReward

    task = {
        "id": "t",
        "required_actions": [{"name": "close"}],
        "expected_response": "done",
    }

    async def agent(history, context):
        return "done"

    reward = TaskOutcomeReward()
    environment = TaskEnvironment(
        [task], success_criteria=lambda t, c: True, max_turns=1, reward_fn=reward
    )
    trajectory = await environment.run_episode(agent, task)
    unattached = TaskEnvironment(
        [task], success_criteria=lambda t, c: True, max_turns=1
    )
    held = await unattached.run_episode(agent, task)
    expected = await score_episode(held, TaskOutcomeReward(), None, task)
    assert await environment.get_reward(trajectory) == pytest.approx(expected)
    assert trajectory.total_reward == pytest.approx(expected)
