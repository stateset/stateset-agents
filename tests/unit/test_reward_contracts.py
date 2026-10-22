"""The contract every shipped reward keeps (see docs/REWARD_CONTRACT.md).

Rewards and reward components are discovered from ``stateset_agents.core.reward``
and every ``stateset_agents.rewards`` submodule, and factories are the
``create_*_reward`` functions there, so a new reward is covered as soon as it
ships. A reward whose constructor needs arguments must be registered in
``CONSTRUCTORS`` (or skipped in ``SKIPPED`` with a reason).

For representative inputs a reward's score must be finite, inside its declared
range (``score_range`` attribute, default ``[0, 1]``), the same on a repeated
call, and computed without mutating the turns or context. A reward may raise
only when it declares ``fail_on_error``; composers must never turn a failing or
NaN component into a NaN total.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import math
import pkgutil
import re
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import stateset_agents.rewards as rewards_package
from stateset_agents.core.basic_rewards import (
    ConcisenessReward,
    HelpfulnessReward,
    SafetyReward,
)
from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.reward_base import (
    CompositeReward,
    RewardFunction,
    RewardResult,
    RewardType,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards.multi_objective_components import (
    BaseRewardComponent,
    EmpathyRewardComponent,
    LengthRewardComponent,
)
from stateset_agents.rewards.multi_objective_reward import (
    MultiObjectiveRewardFunction,
)

DEFAULT_RANGE = (0.0, 1.0)


def _reward_modules() -> list[ModuleType]:
    modules = [importlib.import_module("stateset_agents.core.reward")]
    modules.append(rewards_package)
    for info in pkgutil.iter_modules(
        rewards_package.__path__, f"{rewards_package.__name__}."
    ):
        try:
            modules.append(importlib.import_module(info.name))
        except ImportError:  # pragma: no cover - optional dependency missing
            continue
    return modules


def _qualname(obj: Any) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def _discover() -> tuple[dict[str, type], dict[str, Callable[..., Any]]]:
    classes: dict[str, type] = {}
    factories: dict[str, Callable[..., Any]] = {}
    for module in _reward_modules():
        for value in vars(module).values():
            if inspect.isclass(value):
                try:
                    is_reward = issubclass(value, (RewardFunction, BaseRewardComponent))
                except TypeError:
                    # Python 3.10 reports some generic aliases (list[int]) as
                    # classes that issubclass() then rejects.
                    continue
                if (
                    is_reward
                    and not inspect.isabstract(value)
                    and value.__module__.startswith("stateset_agents.")
                ):
                    classes[_qualname(value)] = value
            elif inspect.isfunction(value) and re.fullmatch(
                r"create_\w*reward", value.__name__
            ):
                if value.__module__.startswith("stateset_agents."):
                    factories[_qualname(value)] = value
    return classes, factories


class _FakeJudge:
    """Deterministic stand-in for an LLM judge (no network)."""

    config = SimpleNamespace(model_name="fake-judge")

    async def evaluate(self, query: str, response: str) -> float:
        return 0.25 + 0.5 * (len(response) % 2)


async def _fake_nsr(payload: dict[str, Any]) -> dict[str, Any]:
    # A real verdict word, so the "verdict" inputs score both 1.0 and 0.0.
    return {"decision": "approved", "decision_id": "d-1", "confidence": 0.9}


#: Constructors for rewards that need arguments, keyed by qualified name.
CONSTRUCTORS: dict[str, Callable[[Any], Any]] = {
    "stateset_agents.core.reward_base.CompositeReward": lambda cls: cls(
        [HelpfulnessReward(weight=0.5), SafetyReward(weight=0.5)]
    ),
    "stateset_agents.rewards.llm_judge_adapter.LLMJudgeReward": lambda cls: cls(
        judge=_FakeJudge()
    ),
    "stateset_agents.rewards.nsr_verifier.NSRVerifierReward": lambda cls: cls(
        client=_fake_nsr
    ),
    "stateset_agents.rewards.multi_objective_reward.MultiObjectiveRewardFunction": (
        lambda cls: cls([EmpathyRewardComponent(), LengthRewardComponent()])
    ),
    "stateset_agents.core.reward_factories.create_adaptive_reward": lambda f: f(
        [HelpfulnessReward(), ConcisenessReward()],
        {"Where is my order?": "It ships tomorrow."},
    ),
    "stateset_agents.core.reward_factories.create_task_agent_reward": lambda f: f(
        {"required_actions": ["lookup_order"], "goal_keywords": ["ships"]}
    ),
    "stateset_agents.core.reward_factories.create_domain_reward": lambda f: f(
        "customer_service"
    ),
    "stateset_agents.rewards.multi_objective_reward.create_domain_reward": lambda f: f(
        "customer_service"
    ),
}

#: Rewards that cannot run offline, with the reason and where they are tested.
_RULER_REASON = (
    "calls an LLM judge through litellm; id mapping and range validation are "
    "covered in test_reward_environment_integrity.py"
)
SKIPPED: dict[str, str] = {
    "stateset_agents.rewards.llm_reward.RulerRewardFunction": _RULER_REASON,
    "stateset_agents.rewards.ruler_reward.RulerRewardFunction": _RULER_REASON,
    "stateset_agents.rewards.support_judge_reward.SupportJudgeReward": (
        "needs a whole-conversation judge model; covered by its own tests"
    ),
    "stateset_agents.rewards.llm_judge.create_llm_judge_reward": (
        "builds a networked LLM judge client from provider credentials"
    ),
    "stateset_agents.rewards.llm_judge_adapter.create_rlaif_reward": (
        "builds a networked LLM judge client from provider credentials"
    ),
}

CLASSES, FACTORIES = _discover()
CASES = sorted({**CLASSES, **FACTORIES})

LONG_TEXT = "Your refund is on its way. " * 4000
INPUTS: dict[str, list[ConversationTurn]] = {
    "empty": [],
    "single_assistant": [
        ConversationTurn(role="assistant", content="I can help with that today.")
    ],
    "multi_turn": [
        ConversationTurn(role="user", content="Where is my order?"),
        ConversationTurn(role="assistant", content="Let me check that for you."),
        ConversationTurn(role="user", content="Thanks"),
        ConversationTurn(role="assistant", content="It ships tomorrow. Anything else?"),
    ],
    "unicode": [
        ConversationTurn(role="user", content="退款在哪里？"),
        ConversationTurn(
            role="assistant",
            content="Je comprends \U0001f60a — 我们会尽快处理您的退款。",
        ),
    ],
    "very_long": [
        ConversationTurn(role="user", content="Status?"),
        ConversationTurn(role="assistant", content=LONG_TEXT),
    ],
    "blank_reply": [
        ConversationTurn(role="user", content="Hello?"),
        ConversationTurn(role="assistant", content=""),
    ],
    "verdict_approve": [
        ConversationTurn(role="user", content="Can I get a refund for #A1?"),
        ConversationTurn(role="assistant", content="Approved: your refund is issued."),
    ],
    "verdict_deny": [
        ConversationTurn(role="user", content="Can I get a refund for #A1?"),
        ConversationTurn(role="assistant", content="Denied: the item is past 90 days."),
    ],
}
CONTEXTS: dict[str, dict[str, Any] | None] = {
    "no_context": None,
    "empty_context": {},
    "scenario_context": {
        "scenario": {
            "id": "s1",
            "expected_response": "It ships tomorrow. Anything else?",
        },
        "user_query": "Where is my order?",
        "ground_truth": {"answer": "tomorrow"},
    },
}


def _build(name: str) -> Any:
    if name in SKIPPED:
        pytest.skip(SKIPPED[name])
    target = CLASSES.get(name) or FACTORIES[name]
    if name in CONSTRUCTORS:
        return CONSTRUCTORS[name](target)
    try:
        return target()
    except TypeError as error:
        pytest.fail(
            f"{name} needs constructor arguments: register it in CONSTRUCTORS "
            f"(or in SKIPPED with a reason) so the reward contract covers it. "
            f"({error})"
        )


def _snapshot(turns: list[Any]) -> list[Any]:
    return [
        (
            (t.role, t.content, dict(t.metadata), t.turn_id)
            if isinstance(t, ConversationTurn)
            else copy.deepcopy(t)
        )
        for t in turns
    ]


async def _score(reward: Any, turns: list[Any], context: Any) -> float:
    if isinstance(reward, BaseRewardComponent):
        return float(await reward.compute_score(turns, context))
    result = await reward.compute_reward(turns, context)
    assert isinstance(result, RewardResult), type(result)
    return float(result.score)


def _turns_for(reward: Any, name: str) -> list[Any]:
    turns = copy.deepcopy(INPUTS[name])
    if isinstance(reward, BaseRewardComponent):
        return [{"role": t.role, "content": t.content} for t in turns]
    return turns


def test_discovery_finds_the_shipped_rewards():
    # Guard against discovery silently covering nothing.
    assert len(CLASSES) >= 25 and len(FACTORIES) >= 10
    for expected in (
        "stateset_agents.core.basic_rewards.SafetyReward",
        "stateset_agents.rewards.multi_objective_reward.MultiObjectiveRewardFunction",
        "stateset_agents.rewards.task_outcome.TaskOutcomeReward",
        "stateset_agents.rewards.multi_objective_components.LengthRewardComponent",
        "stateset_agents.core.reward_factories.create_helpful_agent_reward",
    ):
        assert expected in CASES, expected
    unknown = (set(CONSTRUCTORS) | set(SKIPPED)) - set(CASES)
    assert not unknown, f"stale registry entries: {sorted(unknown)}"


@pytest.mark.parametrize("input_name", sorted(INPUTS))
@pytest.mark.parametrize("name", CASES)
async def test_reward_contract(name: str, input_name: str) -> None:
    reward = _build(name)
    low, high = getattr(reward, "score_range", DEFAULT_RANGE)
    for context_name, context in CONTEXTS.items():
        where = f"{name} on {input_name}/{context_name}"
        turns = _turns_for(reward, input_name)
        context = copy.deepcopy(context)
        turns_before, context_before = _snapshot(turns), copy.deepcopy(context)
        try:
            first = await _score(reward, turns, context)
        except Exception as error:  # noqa: BLE001 - the contract inspects it
            assert getattr(
                reward, "fail_on_error", False
            ), f"{where} raised {error!r} without declaring fail_on_error"
            continue
        second = await _score(
            reward, _turns_for(reward, input_name), copy.deepcopy(context)
        )
        assert math.isfinite(first), f"{where} scored {first}"
        assert low <= first <= high, f"{where} scored {first} outside {low, high}"
        assert first == second, f"{where} is not deterministic: {first} != {second}"
        assert _snapshot(turns) == turns_before, f"{where} mutated its turns"
        assert context == context_before, f"{where} mutated its context"


# --- Composers flag or raise failures; they never emit NaN --------------------


class _Scripted(RewardFunction):
    def __init__(self, score: float, *, critical: bool = False) -> None:
        super().__init__(reward_type=RewardType.CUMULATIVE, name=f"scripted{score}")
        self.score = score
        self.fail_on_error = critical

    async def compute_reward(self, turns, context=None):
        return RewardResult(score=self.score)


async def test_composite_raises_for_a_nan_component():
    # A NaN is a bug or garbage output, never an outage to reweight around.
    composite = CompositeReward([_Scripted(math.nan), _Scripted(0.5)])
    with pytest.raises(ValueError, match="non-finite"):
        await composite.compute_reward(INPUTS["multi_turn"])


async def test_composite_raises_for_a_critical_nan_component():
    composite = CompositeReward([_Scripted(math.nan, critical=True), _Scripted(0.5)])
    with pytest.raises(ValueError, match="non-finite"):
        await composite.compute_reward(INPUTS["multi_turn"])


async def test_multi_objective_raises_for_a_nan_component():
    class NaNComponent(EmpathyRewardComponent):
        async def compute_score(self, turns, context=None):
            return math.nan

    reward = MultiObjectiveRewardFunction(
        [EmpathyRewardComponent(weight=0.5), NaNComponent(name="nan", weight=0.5)]
    )
    with pytest.raises(ValueError, match="non-finite"):
        await reward.compute_reward(
            [{"role": "assistant", "content": "Sorry, I understand."}]
        )


@pytest.mark.parametrize("reward_type", [RewardType.IMMEDIATE, RewardType.SPARSE])
async def test_environment_flags_a_nan_reward(reward_type):
    reward = _Scripted(math.nan)
    reward.reward_type = reward_type
    scenario = {"id": "s", "user_turns": ["hi", "and?"]}
    environment = ConversationEnvironment([scenario], max_turns=3, reward_fn=reward)

    async def respond(history, context):
        return "Happy to help."

    trajectory = await environment.run_episode(respond, scenario)
    assert math.isfinite(trajectory.total_reward)
    assert all(math.isfinite(r) for r in trajectory.turn_rewards)
    assert "finite" in trajectory.metadata["reward_error"]


async def test_trainer_drops_an_episode_with_a_nan_reward():
    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    async def respond(history, context):
        return "Happy to help."

    scenario = {"id": "s", "user_turns": ["hi"]}
    trainer = MultiTurnGRPOTrainer(
        agent=SimpleNamespace(generate_response=respond),
        environment=ConversationEnvironment([scenario], max_turns=2),
        reward_fn=_Scripted(math.nan),
        config=TrainingConfig(num_generations=2),
    )
    assert await trainer.generate_trajectories([scenario], num_generations=2) == []


async def test_nsr_contract_inputs_exercise_both_verdicts():
    from stateset_agents.rewards.nsr_verifier import NSRVerifierReward

    reward = NSRVerifierReward(client=_fake_nsr)
    approve = await reward.compute_reward(INPUTS["verdict_approve"], {})
    deny = await reward.compute_reward(INPUTS["verdict_deny"], {})
    assert (approve.score, deny.score) == (1.0, 0.0)
