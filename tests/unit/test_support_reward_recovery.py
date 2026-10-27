"""Reward regression diagnostics and independent checkpoint selection."""

import copy
import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.evaluation.support_judge import (
    ALL_CHECKS,
    JudgeResult,
    rubric_revision,
)
from stateset_agents.rewards.support_judge_reward import SupportJudgeReward
from stateset_agents.rewards.support_reward_audit import audit_support_rewards
from stateset_agents.training.callbacks import (
    notify_optimizer_step,
    notify_training_end,
    notify_training_start,
)
from stateset_agents.training.judge_selection import (
    JudgeCheckpointSelector,
    support_judge_evaluator,
)


def result(score=0.9, identity="validation", **kwargs):
    checks = dict.fromkeys(ALL_CHECKS, True)
    return replace(
        JudgeResult(
            id=identity,
            score=score,
            pass_scores=(score, score),
            criteria=({}, {}),
            parse_failures=0,
            disagreement=0.0,
            checks=(checks, checks),
        ),
        **kwargs,
    )


class Judge:
    def __init__(self, value=None):
        self.value = value or result()
        self.items = []

    def score(self, items):
        self.items.extend(items)
        return [replace(self.value, id=item.id) for item in items]


@pytest.fixture
def selection(tmp_path):
    model = torch.nn.Linear(1, 1)
    model.train()
    config = SimpleNamespace(output_dir=str(tmp_path), seed=0)
    trainer = SimpleNamespace(
        config=config, agent=SimpleNamespace(model=model), global_step=0
    )
    scores = {0: 0.9, 10: 0.8, 20: 0.95, 25: 0.94}
    calls = []

    async def evaluate(current, cases):
        assert not current.agent.model.training
        calls.append(copy.deepcopy(cases))
        random.random()
        return [result(scores[current.global_step], case["id"]) for case in cases]

    async def save(*, checkpoint_name):
        root = tmp_path / checkpoint_name
        root.mkdir()
        (root / "model.safetensors").write_bytes(str(trainer.global_step).encode())

    trainer.save_checkpoint = save
    kwargs = {
        "evaluate": evaluate,
        "train_cases": [{"id": "train", "query": "train"}],
        "validation_cases": [{"id": "validation", "query": "validate"}],
        "test_cases": [{"id": "test", "query": "test"}],
        "judge_model": "google/gemma-4-31B-it",
        "judge_revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
        "rubric": rubric_revision(),
        "every_updates": 10,
    }
    return trainer, JudgeCheckpointSelector(**kwargs), scores, calls, kwargs


@pytest.mark.asyncio
async def test_baseline_and_best_judged_checkpoint_survive_reward_regression(selection):
    trainer, selector, scores, calls, _ = selection
    state = random.getstate()
    await notify_training_start([selector], trainer=trainer, config=trainer.config)
    assert random.getstate() == state
    assert trainer.agent.model.training
    trainer.global_step = 10
    await notify_optimizer_step([selector], step=10, metrics={"training_reward": 1.0})
    assert selector.selected_checkpoint().name == "judge-selected-step-0"
    trainer.global_step = 20
    await notify_optimizer_step([selector], step=20, metrics={"training_reward": 0.1})
    trainer.global_step = 25
    await notify_training_end([selector], metrics={"final_step": 25})
    assert selector.selected_checkpoint().name == "judge-selected-step-20"
    report = json.loads(
        (Path(trainer.config.output_dir) / "judge-selection.json").read_text()
    )
    assert [row["step"] for row in report["evaluations"]] == [0, 10, 20, 25]
    assert report["production_approved"] is False
    assert {case["id"] for group in calls for case in group} == {"validation"}
    assert random.getstate() == state


@pytest.mark.asyncio
async def test_equal_scores_keep_baseline_and_detect_artifact_changes(selection):
    trainer, selector, scores, _, _ = selection
    await selector.on_training_start(trainer, trainer.config)
    scores[10] = scores[0]
    trainer.global_step = 10
    await selector.on_optimizer_step(10, {})
    assert selector.selected_checkpoint().name == "judge-selected-step-0"
    (selector.selected_checkpoint() / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        selector.selected_checkpoint()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        result(parse_failures=1),
        result(score=float("nan")),
        result(checks=({}, {})),
        result(pass_scores=(0.2, 0.8), score=0.5, disagreement=0.6),
    ],
)
async def test_unusable_judge_never_replaces_selected_checkpoint(selection, bad):
    trainer, selector, _, _, _ = selection
    await selector.on_training_start(trainer, trainer.config)
    selector.evaluate = lambda *_: [bad]
    trainer.global_step = 10
    with pytest.raises(ValueError):
        await notify_optimizer_step([selector], step=10, metrics={})
    assert selector.selected_checkpoint().name == "judge-selected-step-0"


@pytest.mark.asyncio
async def test_failed_save_keeps_previous_receipt(selection):
    trainer, selector, _, _, _ = selection
    await selector.on_training_start(trainer, trainer.config)
    before = (Path(trainer.config.output_dir) / "judge-selection.json").read_bytes()

    async def fail(**_):
        raise OSError("disk full")

    trainer.save_checkpoint = fail
    trainer.global_step = 20
    with pytest.raises(OSError, match="disk full"):
        await selector.on_optimizer_step(20, {})
    assert (
        Path(trainer.config.output_dir) / "judge-selection.json"
    ).read_bytes() == before
    assert selector.selected_checkpoint().name == "judge-selected-step-0"


@pytest.mark.parametrize("duplicate", ["id", "content"])
def test_train_validation_test_overlap_rejected(selection, duplicate):
    *_, kwargs = selection
    kwargs["test_cases"] = [dict(kwargs["validation_cases"][0])]
    if duplicate == "content":
        kwargs["test_cases"][0]["id"] = "renamed"
    with pytest.raises(ValueError, match="unique|repeated"):
        JudgeCheckpointSelector(**kwargs)


@pytest.mark.asyncio
async def test_selection_budget_is_explicit_and_fail_closed(selection):
    trainer, selector, _, _, _ = selection
    selector.max_evaluations = 1
    await selector.on_training_start(trainer, trainer.config)
    trainer.global_step = 10
    with pytest.raises(RuntimeError, match="budget exhausted"):
        await selector.on_optimizer_step(10, {})


@pytest.mark.asyncio
async def test_distributed_and_resumed_runs_fail_before_judging(selection):
    trainer, selector, _, calls, _ = selection
    trainer.world_size = 2
    with pytest.raises(ValueError, match="single-process"):
        await selector.on_training_start(trainer, trainer.config)
    trainer.world_size = 1
    trainer.global_step = 1
    with pytest.raises(ValueError, match="fresh"):
        await selector.on_training_start(trainer, trainer.config)
    assert calls == []


@pytest.mark.asyncio
async def test_full_conversation_judge_and_hard_veto():
    judge = Judge()
    reward = SupportJudgeReward(judge, max_judged_episodes=1)
    turns = [
        ConversationTurn(role=role, content=text)
        for role, text in [
            ("user", "My order number is 12345."),
            ("assistant", "What happened?"),
            ("user", "It was damaged."),
            ("assistant", "I can help replace the damaged item."),
        ]
    ]
    value = await reward.compute_reward(turns, {"id": "case"})
    assert value.score == 0.9
    assert len(judge.items[0].transcript) == 4
    with pytest.raises(RuntimeError, match="budget exhausted"):
        await reward.compute_reward(turns, {"id": "case"})
    value = await reward.compute_reward(
        turns, {"must_avoid": ["replace the damaged item"]}
    )
    assert value.score == 0
    assert len(judge.items) == 1


@pytest.mark.asyncio
async def test_judge_errors_and_parse_failures_do_not_become_neutral_reward():
    judge = Judge(result(parse_failures=1))
    reward = SupportJudgeReward(judge, max_judged_episodes=2, judge_weight=0.25)
    turns = [
        ConversationTurn(role="assistant", content="The store policy permits a return.")
    ]
    with pytest.raises(ValueError, match="parse"):
        await reward.compute_reward(turns)
    assert reward.judged_episodes == 1


@pytest.mark.asyncio
async def test_one_unsafe_turn_cannot_be_diluted_by_safe_turns_and_high_judge_score():
    judge = Judge(result(1.0))
    reward = SupportJudgeReward(judge, max_judged_episodes=1)
    value = await reward.compute_reward(
        [
            ConversationTurn(role="user", content="Please help."),
            ConversationTurn(role="assistant", content="Your password is public."),
            ConversationTurn(role="user", content="What next?"),
            ConversationTurn(
                role="assistant", content="I can help with the store's return policy."
            ),
        ]
    )
    assert value.score == 0
    assert judge.items == []


@pytest.mark.asyncio
async def test_rule_only_blend_does_not_spend_judge_budget():
    judge = Judge()
    reward = SupportJudgeReward(judge, max_judged_episodes=1, judge_fraction=0)
    value = await reward.compute_reward(
        [
            ConversationTurn(
                role="assistant", content="The store policy permits a return."
            )
        ]
    )
    assert value.breakdown["judge_sampled"] == 0
    assert reward.judged_episodes == 0
    assert judge.items == []


@pytest.mark.asyncio
async def test_real_fixture_audit_is_diagnostic_not_a_200_case_quality_claim():
    fixture = (
        Path(__file__).parents[1]
        / "fixtures/support_conversation_reward/preflight5_regressions.json"
    )
    conversations = json.loads(fixture.read_text())["conversations"]
    report = await audit_support_rewards(conversations, expected_pairs=3)
    assert report["rewards"]["v3"]["direction_agreements"] == 3
    assert report["rewards"]["v2"]["direction_agreements"] < 3
    assert report["quality_gain_established"] is False
    with pytest.raises(ValueError, match="exactly"):
        await audit_support_rewards(conversations, expected_pairs=200)


@pytest.mark.asyncio
async def test_audit_reports_ties_and_rejects_duplicate_pairs():
    item = {
        "id": "tie",
        "context": {},
        "judge": {"base": 0.5, "trained": 0.5},
        "base": [["assistant", "Returns are allowed within the policy window."]],
        "trained": [["assistant", "Returns are allowed within the policy window."]],
    }
    report = await audit_support_rewards([item], expected_pairs=1)
    assert report["rewards"]["v3"]["judge_ties"] == 1
    assert report["rewards"]["v3"]["direction_agreements"] == 0
    assert report["rewards"]["v3"]["spearman_all_transcripts"] is None
    with pytest.raises(ValueError, match="unique"):
        await audit_support_rewards([item, item], expected_pairs=2)


@pytest.mark.asyncio
async def test_evaluator_uses_current_policy_fresh_clones_and_full_conversation(
    selection,
):
    trainer, selector, _, _, _ = selection
    calls = []

    class Environment:
        def clone(self):
            return Environment()

        async def run_episode(self, agent_fn, *, scenario, max_turns):
            response = await agent_fn(
                [{"role": "user", "content": scenario["query"]}], {}
            )
            calls.append(response)
            return SimpleNamespace(
                turns=[
                    ConversationTurn(role="user", content=scenario["query"]),
                    ConversationTurn(role="assistant", content=response),
                ]
            )

    async def generate(*_):
        return f"Current policy at step {trainer.global_step}."

    trainer.agent.generate_response = generate
    judge = Judge()
    selector.evaluate = support_judge_evaluator(judge, Environment())
    await selector.on_training_start(trainer, trainer.config)
    trainer.global_step = 10
    await selector.on_optimizer_step(10, {})
    assert calls == ["Current policy at step 0.", "Current policy at step 10."]
    assert [item.id for item in judge.items] == ["validation", "validation"]
