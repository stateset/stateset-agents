"""Reward failures fail closed: no neutral or valid-looking stand-in scores.

A judge that returns no verdict, a composite in which no component scored, and
an environment asked for a reward it cannot compute all raise, so composers
redistribute the weight, environments flag the episode and trainers drop it.
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import pytest

from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.reward_base import (
    CompositeReward,
    RewardFunction,
    RewardResult,
    RewardType,
    RewardUnavailableError,
)
from stateset_agents.core.trajectory import ConversationTurn, MultiTurnTrajectory
from stateset_agents.rewards.llm_judge import (
    JudgeConfig,
    JudgeProvider,
    JudgeUnavailableError,
    LLMJudge,
)
from stateset_agents.rewards.llm_judge_adapter import (
    LLMJudgeReward,
    LLMJudgeRewardComponent,
    LLMJudgeRewardWithFallback,
)
from stateset_agents.rewards.multi_objective_components import (
    EmpathyRewardComponent,
    ModelBasedRewardComponent,
)
from stateset_agents.rewards.multi_objective_reward import (
    MultiObjectiveRewardFunction,
)

TURNS = [
    ConversationTurn(role="user", content="Where is my refund?"),
    ConversationTurn(role="assistant", content="Sorry, I understand. Let me help."),
]
DICT_TURNS = [{"role": t.role, "content": t.content} for t in TURNS]


class Boom(RewardFunction):
    def __init__(self, reward_type: RewardType = RewardType.IMMEDIATE) -> None:
        super().__init__(reward_type=reward_type, name="boom")

    async def compute_reward(self, turns, context=None):
        raise RewardUnavailableError("judge down")


class NaNReward(RewardFunction):
    async def compute_reward(self, turns, context=None):
        return RewardResult(score=math.nan)


class DownJudge:
    config = SimpleNamespace(model_name="down")

    async def evaluate(self, query: str, response: str) -> float:
        raise JudgeUnavailableError("judge timeout")


def _replies(*replies: str):
    remaining = iter(replies)

    async def respond(history, context):
        return next(remaining)

    return respond


# --- Judges ------------------------------------------------------------------


async def test_llm_judge_without_a_verdict_raises_instead_of_scoring_neutral():
    judge = LLMJudge(JudgeConfig(provider=JudgeProvider.LOCAL))
    with pytest.raises(JudgeUnavailableError, match="no score"):
        await judge.evaluate("q", "r")
    with pytest.raises(JudgeUnavailableError):
        await judge.evaluate_batch(["q"], ["r"])


async def test_llm_judge_reward_propagates_judge_failures():
    with pytest.raises(JudgeUnavailableError):
        await LLMJudgeReward(judge=DownJudge()).compute_reward(TURNS)


async def test_judge_component_failure_is_redistributed_not_scored_half():
    # Renormalizing is an opt-in (round 6); by default the outage raises.
    reward = MultiObjectiveRewardFunction(
        [
            LLMJudgeRewardComponent(DownJudge(), weight=0.5),
            EmpathyRewardComponent(weight=0.5),
        ],
        outage_policy="renormalize",
    )
    result = await reward.compute_reward(DICT_TURNS)
    assert result.metadata["failed_components"] == ["llm_judge"]
    assert result.score == pytest.approx(result.components["empathy"])


async def test_model_based_judge_failure_is_redistributed_not_scored_half():
    class BadClient:
        async def generate(self, prompt):
            raise RuntimeError("judge timeout")

    reward = MultiObjectiveRewardFunction(
        [
            ModelBasedRewardComponent(api_client=BadClient(), provider="generate"),
            EmpathyRewardComponent(weight=0.5),
        ],
        outage_policy="renormalize",
    )
    result = await reward.compute_reward(DICT_TURNS)
    assert result.metadata["failed_components"] == ["model_judge"]
    assert "model_judge" not in result.components
    assert result.score == pytest.approx(result.components["empathy"])


async def test_model_based_judge_reply_without_a_score_is_a_failure():
    class Chatty:
        async def generate(self, prompt):
            return "I think it is quite good."

    component = ModelBasedRewardComponent(api_client=Chatty(), provider="generate")
    with pytest.raises(RewardUnavailableError, match="not strict JSON"):
        await component.compute_score(DICT_TURNS)


async def test_fallback_reward_flags_the_judge_failure_it_falls_back_from():
    reward = LLMJudgeRewardWithFallback(judge=DownJudge())
    result = await reward.compute_reward(TURNS)
    assert result.metadata["mode"] == "heuristic_fallback"
    assert "judge timeout" in result.metadata["judge_error"]


# --- Composites with nothing scored ------------------------------------------


async def test_composite_with_no_scored_component_raises():
    with pytest.raises(RewardUnavailableError, match="unavailable"):
        await CompositeReward([Boom()]).compute_reward(TURNS)
    renormalizing = CompositeReward([Boom()])
    renormalizing.outage_policy = "renormalize"
    with pytest.raises(RewardUnavailableError, match="every reward function failed"):
        await renormalizing.compute_reward(TURNS)
    with pytest.raises(ValueError, match="no reward functions"):
        await CompositeReward([]).compute_reward(TURNS)


async def test_multi_objective_with_no_scored_component_raises():
    class Down(EmpathyRewardComponent):
        async def compute_score(self, turns, context=None):
            raise RewardUnavailableError("down")

    with pytest.raises(RewardUnavailableError, match="unavailable"):
        await MultiObjectiveRewardFunction([Down()]).compute_reward(DICT_TURNS)
    renormalizing = MultiObjectiveRewardFunction([Down()], outage_policy="renormalize")
    with pytest.raises(RewardUnavailableError, match="every component failed"):
        await renormalizing.compute_reward(DICT_TURNS)
    with pytest.raises(ValueError, match="no components"):
        await MultiObjectiveRewardFunction([]).compute_reward(DICT_TURNS)


async def test_an_all_failed_composite_is_flagged_and_never_trained_on():
    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    scenario = {"id": "s", "user_turns": ["hi", "more"]}
    reward = CompositeReward([Boom()])
    environment = ConversationEnvironment([scenario], max_turns=2, reward_fn=reward)
    trajectory = await environment.run_episode(_replies("hello", "sure"), scenario)
    assert "unavailable" in trajectory.metadata["reward_error"]

    trainer = MultiTurnGRPOTrainer(
        agent=SimpleNamespace(generate_response=_replies(*["ok"] * 8)),
        environment=environment,
        reward_fn=reward,
        config=TrainingConfig(num_generations=2),
    )
    with pytest.raises(RewardUnavailableError, match="No episode got a reward"):
        await trainer.generate_trajectories([scenario], num_generations=2)


# --- Environment.get_reward --------------------------------------------------

TRAJECTORY = MultiTurnTrajectory(
    turns=[ConversationTurn(role="assistant", content="x")]
)


async def test_get_reward_raises_for_failed_and_non_finite_rewards():
    with pytest.raises(RuntimeError, match="judge down"):
        await ConversationEnvironment([], reward_fn=Boom()).get_reward(TRAJECTORY)
    with pytest.raises(ValueError, match="finite"):
        await ConversationEnvironment([], reward_fn=NaNReward()).get_reward(TRAJECTORY)


async def test_get_reward_without_a_reward_function_is_explicit():
    with pytest.raises(ValueError, match="no reward_fn"):
        await ConversationEnvironment([]).get_reward(TRAJECTORY)


# --- Consumers of rewards ----------------------------------------------------


def _engine(environment, reward_function, **kwargs):
    from stateset_agents.core.computational_engine import ComputationalGRPOEngine

    async def answer(prompt):
        return "fine"

    agent = SimpleNamespace(generate_response=answer)
    return ComputationalGRPOEngine(
        agent=agent,
        environment=environment,
        reward_function=reward_function,
        num_workers=1,
        **kwargs,
    )


async def test_engine_drops_trajectories_whose_reward_failed(caplog):
    class Half(RewardFunction):
        calls = 0

        async def compute_reward(self, turns, context=None):
            Half.calls += 1
            if Half.calls % 2:
                raise RewardUnavailableError("judge down")
            return RewardResult(score=0.75)

    engine = _engine(ConversationEnvironment([]), Half())
    with caplog.at_level(logging.WARNING):
        batch = await engine.generate_trajectory_batch(["a", "b", "c", "d"])
    assert [t.learned_reward for t in batch] == [0.75, 0.75]
    assert all(t.raw_reward_signal is None for t in batch)  # no env reward_fn
    assert engine.metrics["failed_trajectories"] == 2.0


async def test_engine_iteration_with_no_rewarded_trajectory_raises():
    engine = _engine(ConversationEnvironment([], reward_fn=Boom()), Boom())
    with pytest.raises(RuntimeError, match="reward computations failed"):
        await engine.train_iteration(["a", "b"])


async def test_single_turn_trainer_drops_generations_without_a_reward():
    from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer

    class Flaky(RewardFunction):
        calls = 0

        async def compute_reward(self, turns, context=None):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RewardUnavailableError("judge 503")
            return RewardResult(score=0.25)

    async def answer(prompt):
        return "answer"

    trainer = SingleTurnGRPOTrainer.__new__(SingleTurnGRPOTrainer)
    trainer.agent = SimpleNamespace(generate_response=answer)
    trainer.reward_fn = Flaky()
    trainer.environment = ConversationEnvironment([])
    group, _ = await trainer._generate_trajectory_group("q", num_generations=3)
    assert [t.total_reward for t in group.trajectories] == [0.25, 0.25]

    trainer.reward_fn = Boom()  # every generation's reward is unavailable
    with pytest.raises(RewardUnavailableError, match="failed to get a reward"):
        await trainer._generate_trajectory_group("q", num_generations=2)

    trainer.reward_fn = None  # and the environment has no reward: a setup bug
    with pytest.raises(ValueError, match="no reward_fn"):
        await trainer._generate_trajectory_group("q", num_generations=2)


# --- Only outages are absorbed; bugs propagate -------------------------------


class Buggy(RewardFunction):
    def __init__(self, error: Exception) -> None:
        super().__init__(reward_type=RewardType.IMMEDIATE, name="buggy")
        self.error = error

    async def compute_reward(self, turns, context=None):
        raise self.error


class Half(RewardFunction):
    async def compute_reward(self, turns, context=None):
        return RewardResult(score=0.5)


@pytest.mark.parametrize(
    "error", [TypeError("bad call"), AttributeError("no .get"), KeyError("k")]
)
async def test_composite_reraises_programming_errors(error):
    composite = CompositeReward([Buggy(error), Half()])
    with pytest.raises(type(error)):
        await composite.compute_reward(TURNS)


async def test_composite_redistributes_only_an_unavailable_component():
    composite = CompositeReward([Buggy(RewardUnavailableError("judge 503")), Half()])
    # By default an outage makes the whole composite unavailable...
    with pytest.raises(RewardUnavailableError, match="judge 503"):
        await composite.compute_reward(TURNS)
    # ...and renormalizing over the survivors is an explicit opt-in.
    composite.outage_policy = "renormalize"
    result = await composite.compute_reward(TURNS)
    assert result.metadata["failed_components"] == ["buggy"]
    assert result.score == pytest.approx(1.0)  # Half's 0.5 x the whole weight 2


async def test_multi_objective_reraises_programming_errors():
    class Broken(EmpathyRewardComponent):
        async def compute_score(self, turns, context=None):
            raise AttributeError("'str' object has no attribute 'get'")

    reward = MultiObjectiveRewardFunction(
        [Broken(name="broken"), EmpathyRewardComponent()]
    )
    with pytest.raises(AttributeError):
        await reward.compute_reward(DICT_TURNS)


async def test_tutoring_reward_scores_a_string_ground_truth():
    from stateset_agents.core.reward_context import score_episode
    from stateset_agents.core.reward_factories import create_tutoring_reward

    scenario = {"id": "m", "user_turns": ["What is 2+2?"], "ground_truth": "4"}
    totals = {}
    for reply in ("The answer is 4.", "The answer is 5."):
        reward = create_tutoring_reward()
        environment = ConversationEnvironment([scenario], max_turns=1, reward_fn=reward)
        trajectory = await environment.run_episode(_replies(reply), scenario)
        assert "reward_error" not in trajectory.metadata
        totals[reply] = await score_episode(trajectory, reward, environment, scenario)
    assert totals["The answer is 4."] > totals["The answer is 5."]


# --- Correctness: a correct answer is a full score, unlabeled is flagged -----


async def test_correct_answer_scores_full_marks_not_the_neutral_value():
    from stateset_agents.core.basic_rewards import CorrectnessReward

    reward = CorrectnessReward(ground_truth={"answer": "4"})
    right = await reward.compute_reward(_turns("The answer is 4."))
    wrong = await reward.compute_reward(_turns("The answer is 5."))
    assert (right.score, wrong.score) == (1.0, 0.0)
    assert "no_ground_truth" not in right.metadata


async def test_unlabeled_correctness_is_neutral_and_flagged():
    from stateset_agents.core.basic_rewards import CorrectnessReward

    result = await CorrectnessReward().compute_reward(_turns("The answer is 4."))
    assert result.score == 0.5
    assert result.metadata["no_ground_truth"] is True


async def test_evaluation_counts_correct_answers_as_successes_not_neutrals():
    from stateset_agents.core.basic_rewards import CorrectnessReward
    from stateset_agents.training.evaluation import EvaluationConfig, evaluate_agent

    labeled = {"id": "m", "user_turns": ["What is 2+2?"], "ground_truth": "4"}
    unlabeled = {"id": "u", "user_turns": ["What is 2+2?"]}

    async def answer(history, context):
        return "The answer is 4."

    for scenario, expected in ((labeled, 1.0), (unlabeled, 0.0)):
        metrics = await evaluate_agent(
            agent=SimpleNamespace(generate_response=answer),
            environment=ConversationEnvironment(
                [scenario], max_turns=1, reward_fn=CorrectnessReward()
            ),
            scenarios=[scenario],
            config=EvaluationConfig(num_episodes=1, concurrency=1),
        )
        assert metrics["eval_success_rate"] == expected


def _turns(reply: str) -> list[ConversationTurn]:
    return [
        ConversationTurn(role="user", content="What is 2+2?"),
        ConversationTurn(role="assistant", content=reply),
    ]


# --- Network rewards fail closed unless a fallback is opted into -------------


async def _unreachable(payload):
    raise ConnectionError("nsr unreachable")


def _verdict_turns(reply: str) -> list[ConversationTurn]:
    return [
        ConversationTurn(role="user", content="Refund for #A1?"),
        ConversationTurn(role="assistant", content=reply),
    ]


async def test_nsr_outage_is_unavailable_by_default_and_redistributed():
    from stateset_agents.rewards.nsr_verifier import (
        NSRVerifierConfig,
        NSRVerifierReward,
    )

    nsr = NSRVerifierReward(config=NSRVerifierConfig(), client=_unreachable)
    with pytest.raises(RewardUnavailableError, match="nsr unreachable"):
        await nsr.compute_reward(_verdict_turns("Decision: approve"), {})
    composite = CompositeReward([nsr, Half()])
    with pytest.raises(RewardUnavailableError, match="nsr unreachable"):
        await composite.compute_reward(_verdict_turns("Decision: approve"), {})
    composite.outage_policy = "renormalize"
    result = await composite.compute_reward(_verdict_turns("Decision: approve"), {})
    assert result.metadata["failed_components"] == ["NSRVerifierReward"]


async def test_opted_in_nsr_fallback_is_flagged_on_the_episode():
    from stateset_agents.core.reward_context import score_episode
    from stateset_agents.rewards.nsr_verifier import (
        NSRVerifierConfig,
        NSRVerifierReward,
    )

    nsr = NSRVerifierReward(
        config=NSRVerifierConfig(error_score=0.0), client=_unreachable
    )
    nsr.reward_type = RewardType.SPARSE
    scenario = {"id": "r", "user_turns": ["Refund for #A1?"]}
    for attach in (True, False):
        environment = ConversationEnvironment(
            [scenario], max_turns=1, reward_fn=nsr if attach else None
        )
        trajectory = await environment.run_episode(
            _replies("Decision: approve"), scenario
        )
        assert await score_episode(trajectory, nsr, environment, scenario) == 0.0
        assert trajectory.metadata["reward_fallback"] == "nsr_error"


async def test_ruler_failure_is_unavailable_unless_fallback_is_enabled(monkeypatch):
    import stateset_agents.rewards.llm_reward as lr

    monkeypatch.setattr(lr, "LITELLM_AVAILABLE", True)

    async def broken(messages):
        raise RuntimeError("judge 503")

    strict = lr.RulerRewardFunction(model="judge", max_retries=1)
    strict._make_completion_call = broken
    with pytest.raises(RewardUnavailableError, match="judge 503"):
        await strict.compute_reward(TURNS)

    lenient = lr.RulerRewardFunction(
        model="judge", max_retries=1, fallback_enabled=True
    )
    lenient._make_completion_call = broken
    result = await lenient.compute_reward(TURNS)
    assert result.metadata["reward_fallback"] == "ruler_error"
    composite = await CompositeReward([lenient, Half()]).compute_reward(TURNS)
    assert "ruler_error" in composite.metadata["reward_fallback"]


# --- Trainers absorb only outages (RewardUnavailableError) -------------------


class _Returns(RewardFunction):
    """A reward that returns a fixed (possibly invalid) value."""

    def __init__(self, value: object) -> None:
        super().__init__(reward_type=RewardType.CUMULATIVE, name="returns")
        self.value = value

    async def compute_reward(self, turns, context=None):
        return self.value


_BUGS = [
    pytest.param(lambda: Buggy(TypeError("bad arg")), TypeError, id="TypeError"),
    pytest.param(
        lambda: Buggy(AttributeError("no attr")), AttributeError, id="AttributeError"
    ),
    pytest.param(lambda: Buggy(KeyError("k")), KeyError, id="KeyError"),
    pytest.param(lambda: Buggy(ValueError("bad data")), ValueError, id="ValueError"),
    pytest.param(lambda: _Returns(None), TypeError, id="None"),
    pytest.param(lambda: _Returns(math.nan), ValueError, id="NaN"),
]


class _BatchAgent:
    model = SimpleNamespace(eval=lambda: None)

    async def generate_turns_batch(self, histories, contexts):
        return ["ok"] * len(histories)

    async def generate_response(self, history, context):
        return "ok"


@pytest.mark.parametrize("batched", [False, True], ids=["sequential", "batched"])
@pytest.mark.parametrize("attached", [False, True], ids=["trainer", "environment"])
@pytest.mark.parametrize(("make", "error"), _BUGS)
async def test_multi_turn_trainer_propagates_reward_bugs(
    make, error, attached, batched
):
    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    scenario = {"id": "s", "user_turns": ["hi"]}
    reward = make()
    environment = ConversationEnvironment(
        [scenario], max_turns=1, reward_fn=reward if attached else None
    )
    trainer = MultiTurnGRPOTrainer(
        agent=_BatchAgent(),
        environment=environment,
        reward_fn=None if attached else reward,
        config=TrainingConfig(num_generations=2, batched_rollouts=batched),
    )
    with pytest.raises(error):
        await trainer.generate_trajectories([scenario], num_generations=2)


async def test_multi_turn_trainer_drops_only_the_episode_whose_reward_is_down():
    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    class DownOnce(RewardFunction):
        calls = 0

        async def compute_reward(self, turns, context=None):
            DownOnce.calls += 1
            if DownOnce.calls == 1:
                raise RewardUnavailableError("judge 503")
            return RewardResult(score=0.5)

    scenario = {"id": "s", "user_turns": ["hi"]}
    trainer = MultiTurnGRPOTrainer(
        agent=_BatchAgent(),
        environment=ConversationEnvironment([scenario], max_turns=1),
        reward_fn=DownOnce(),
        config=TrainingConfig(num_generations=3, batched_rollouts=False),
    )
    [group] = await trainer.generate_trajectories([scenario], num_generations=3)
    assert len(group.trajectories) == 2


@pytest.mark.parametrize(("make", "error"), _BUGS)
async def test_single_turn_trainer_propagates_reward_bugs(make, error):
    from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer

    async def answer(prompt):
        return "answer"

    trainer = SingleTurnGRPOTrainer.__new__(SingleTurnGRPOTrainer)
    trainer.agent = SimpleNamespace(generate_response=answer)
    trainer.reward_fn = make()
    trainer.environment = ConversationEnvironment([])
    with pytest.raises(error):
        await trainer._generate_trajectory_group("q", num_generations=2)


@pytest.mark.parametrize(("make", "error"), _BUGS)
async def test_engine_propagates_reward_bugs(make, error):
    engine = _engine(ConversationEnvironment([]), make())
    with pytest.raises(error):
        await engine.generate_trajectory_batch(["a", "b"])


@pytest.mark.parametrize(("make", "error"), _BUGS)
async def test_post_training_evaluation_propagates_reward_bugs(make, error):
    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    scenario = {"id": "s", "user_turns": ["hi"]}
    trainer = MultiTurnGRPOTrainer(
        agent=_BatchAgent(),
        environment=ConversationEnvironment([scenario], max_turns=1),
        reward_fn=make(),
        config=TrainingConfig(num_generations=2),
    )
    with pytest.raises(error):
        await trainer.run_post_training_evaluation([scenario], num_samples=1)


@pytest.mark.parametrize(("make", "error"), _BUGS)
async def test_legacy_reward_coercion_never_reads_a_failure_as_zero(make, error):
    from stateset_agents.training.multi_turn_evaluation import coerce_reward_result

    with pytest.raises(error):
        await coerce_reward_result(make().compute_reward(TURNS), (Exception,))
    assert await coerce_reward_result(
        RewardResult(score=0.25, breakdown={"a": 0.25}), ()
    ) == (0.25, {"a": 0.25})


# --- Judges accept only strict JSON with an in-range score -------------------

NOT_STRICT = [
    "0.8",
    "On a scale of 0 to 1, I'd give it 0.8",
    "85/100",
    "7/10",
    "-0.4",
    "Score: 0.9",
    'Score: {"score": 0.8}',
    # One fenced JSON verdict is accepted (round 6); see test_judge_round6.py.
    '{"score": 1.5}',
    '{"score": -0.1}',
    '{"score": "0.8"}',
    '{"score": true}',
    '{"rating": 0.8}',
    "[0.8]",
    "",
]


class _Says:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def generate(self, prompt: str) -> str:
        return self.reply


@pytest.mark.parametrize("reply", NOT_STRICT)
async def test_model_judge_component_rejects_anything_but_strict_json(reply):
    from stateset_agents.rewards.multi_objective_components import (
        ModelBasedRewardComponent,
    )

    component = ModelBasedRewardComponent(api_client=_Says(reply), provider="local")
    with pytest.raises(RewardUnavailableError):
        await component.compute_score(DICT_TURNS)


@pytest.mark.parametrize(
    ("reply", "score"), [('{"score": 0.8}', 0.8), (' {"score": 1} ', 1.0)]
)
async def test_model_judge_component_reads_a_strict_json_score(reply, score):
    from stateset_agents.rewards.multi_objective_components import (
        ModelBasedRewardComponent,
    )

    component = ModelBasedRewardComponent(api_client=_Says(reply), provider="local")
    assert await component.compute_score(DICT_TURNS) == pytest.approx(score)
    assert '{"score"' in component.judge_prompt_template.format(
        user_query="q", assistant_response="r"
    )


def _llm_judge(reply: str, **config):
    from stateset_agents.rewards.llm_judge import JudgeConfig, JudgeProvider, LLMJudge

    judge = LLMJudge(JudgeConfig(provider=JudgeProvider.LOCAL, **config))
    prompts: list = []

    async def call(messages):
        prompts.append(messages[-1]["content"])
        return reply

    judge.backend._call_api = call
    return judge, prompts


@pytest.mark.parametrize("reply", NOT_STRICT)
async def test_llm_judge_rejects_anything_but_strict_json(reply):
    from stateset_agents.rewards.llm_judge import JudgeUnavailableError

    judge, _ = _llm_judge(reply)
    with pytest.raises(JudgeUnavailableError):
        await judge.evaluate("q", "r")


async def test_llm_judge_reads_a_strict_json_score():
    judge, _ = _llm_judge(
        '{"score": 7, "reasoning": "ok"}', score_min=0.0, score_max=10.0
    )
    assert await judge.evaluate("q", "r") == pytest.approx(0.7)


@pytest.mark.parametrize("winner", ["C", "maybe", None])
async def test_llm_judge_pairwise_needs_a_known_winner(winner):
    import json

    from stateset_agents.rewards.llm_judge import JudgeUnavailableError

    judge, _ = _llm_judge(json.dumps({"winner": winner}))
    with pytest.raises(JudgeUnavailableError):
        await judge.compare_pairwise("q", "a", "b")


async def test_llm_judge_shows_the_reference_and_keys_the_cache_on_it():
    judge, prompts = _llm_judge('{"score": 0.5}')
    await judge.evaluate("q", "r", reference="GOLD-ANSWER-1")
    assert "GOLD-ANSWER-1" in prompts[-1]
    await judge.evaluate("q", "r", reference="GOLD-ANSWER-2")
    assert len(prompts) == 2 and "GOLD-ANSWER-2" in prompts[-1]  # not a cache hit
    await judge.evaluate("q", "r", reference="GOLD-ANSWER-1")
    assert len(prompts) == 2  # same reference: cached
    await judge.evaluate("q", "r")
    assert len(prompts) == 3 and "Reference" not in prompts[-1]


@pytest.mark.parametrize(
    "reply",
    [
        {"status": "ok", "verdict": "approved"},
        {"decision": None},
        {"decision": "maybe"},
        {},
    ],
)
async def test_nsr_reply_without_a_decision_is_not_a_verified_zero(reply):
    from stateset_agents.rewards.nsr_verifier import (
        NSRVerifierConfig,
        NSRVerifierReward,
    )

    async def client(payload):
        return reply

    turns = [
        ConversationTurn(role="user", content="Can I get a refund for #A1?"),
        ConversationTurn(role="assistant", content="Approved: your refund is issued."),
    ]
    with pytest.raises(RewardUnavailableError, match="no decision"):
        await NSRVerifierReward(NSRVerifierConfig(), client=client).compute_reward(
            turns
        )
    flagged = await NSRVerifierReward(
        NSRVerifierConfig(error_score=0.0), client=client
    ).compute_reward(turns)
    assert flagged.metadata["mode"] == "nsr_error"
    assert flagged.metadata["reward_fallback"] == "nsr_error"


async def test_nsr_reply_with_a_decision_is_verified():
    from stateset_agents.rewards.nsr_verifier import (
        NSRVerifierConfig,
        NSRVerifierReward,
    )

    async def client(payload):
        return {"decision": "APPROVED", "decision_id": "d1"}

    turns = [
        ConversationTurn(role="user", content="Can I get a refund for #A1?"),
        ConversationTurn(role="assistant", content="Approved: your refund is issued."),
    ]
    result = await NSRVerifierReward(NSRVerifierConfig(), client=client).compute_reward(
        turns
    )
    assert (result.score, result.metadata["mode"]) == (1.0, "verified")
