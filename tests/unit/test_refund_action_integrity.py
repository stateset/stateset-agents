"""Adversarial actions cannot earn rewards or contaminate verified SFT data."""

import json
import re
import sys

import pytest

from stateset_agents.core.environment_base import EpisodeStatus
from stateset_agents.core.environments.refund_environment import (
    RefundEnvironment,
    refund_benchmark,
)
from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
    refund_policy_benchmark,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.data.refund_demonstrations import (
    RejectedTrajectory,
    reference_refund_demonstration,
    replay_refund_candidate,
)


def action(tool, **args):
    return ConversationTurn(
        role="assistant", content=json.dumps({"tool": tool, "args": args})
    )


@pytest.fixture(params=[RefundEnvironment, RefundPolicyEnvironment])
def sandbox(request):
    return request.param(), {
        "order_id": "A",
        "eligible": True,
        "amount_cents": 100,
        "paid_cents": 100,
        "refunded_cents": 0,
        "days_since_delivery": 0,
        "return_window_days": 30,
        "status": "delivered",
        "chargeback_open": False,
        "customer_note": "Please refund.",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        '{"tool":"deny","tool":"refund","args":{"order_id":"A","amount_cents":100}}',
        '{"tool":"refund","args":{"order_id":"B","order_id":"A","amount_cents":100}}',
        '{"tool":"refund","args":{"order_id":"A","amount_cents":999,"amount_cents":100}}',
        '{"tool":"refund","args":{"order_id":"A","amount_cents":100,"amount_cents":100}}',
        '{"tool":"refund","args":{"order_id":"A","amount_cents":999,"amou\\u006et_cents":100}}',
        '{"tool":"refund","args":{"order_id":"A","amount_cents":NaN}}',
        '{"tool":"refund","args":{"order_id":"A","amount_cents":Infinity}}',
        '{"tool":"refund","args":{"order_id":"A","amount_cents":1e999}}',
        '{"tool":"refund","args":{"order_id":"A","amount_cents":true}}',
        '{"tool":"refund","args":{"order_id":"A","amount_cents":100.0}}',
        "[]",
        "null",
        '{"tool":[],"args":{}}',
        "[" * (sys.getrecursionlimit() + 100)
        + "0"
        + "]" * (sys.getrecursionlimit() + 100),
    ],
)
async def test_malformed_actions_are_scored_failures_without_ledger_mutation(
    sandbox, bad
):
    env, row = sandbox
    state = await env.reset(row)
    state, _, _, _ = await env.step(state, action("lookup_order", order_id="A"))
    state, reward, done, info = await env.step(
        state, ConversationTurn(role="assistant", content=bad)
    )
    assert not done and reward == 0
    assert state.context["refunds"] == []
    assert info["metrics"]["policy_violations"] == 1
    # A subsequent correct action does not erase the violation.
    state, _, _, _ = await env.step(
        state, action("refund", order_id="A", amount_cents=100)
    )
    state, reward, done, info = await env.step(state, action("finish"))
    assert done and reward == -1 and info["metrics"]["task_success"] == 0
    assert state.context["refunds"] == [100]


@pytest.mark.asyncio
@pytest.mark.parametrize("spoof", ["role", "tool_calls"])
async def test_nonassistant_or_multiple_action_channels_cannot_apply_a_refund(
    sandbox, spoof
):
    env, row = sandbox
    state = await env.reset(row)
    state, _, _, _ = await env.step(state, action("lookup_order", order_id="A"))
    turn = action("refund", order_id="A", amount_cents=100)
    if spoof == "role":
        turn.role = "user"
    else:
        turn.tool_calls = [{"name": "refund", "arguments": {"amount_cents": 999}}]
    state, _, _, info = await env.step(state, turn)
    assert state.context["refunds"] == []
    assert info["metrics"]["policy_violations"] == 1


@pytest.mark.asyncio
async def test_exhausted_turns_after_valid_refund_require_explicit_finish(sandbox):
    env, row = sandbox
    state = await env.reset(row)
    for turn in (
        action("lookup_order", order_id="A"),
        action("refund", order_id="A", amount_cents=100),
        action("lookup_order", order_id="A"),
        action("lookup_order", order_id="A"),
    ):
        state, reward, done, info = await env.step(state, turn)
    assert done and state.status == EpisodeStatus.TIMEOUT
    assert reward == -1 and info["metrics"]["task_success"] == 0
    assert state.context["refunds"] == [100]
    with pytest.raises(ValueError, match="completed"):
        await env.step(state, action("finish"))


@pytest.mark.asyncio
async def test_duplicate_keys_are_rejected_from_otherwise_verified_demonstrations():
    row = refund_policy_benchmark("train", 8, 42)[0]
    candidate = await reference_refund_demonstration(row)
    candidate["messages"][-1]["content"] = '{"tool":"refund","tool":"finish","args":{}}'
    with pytest.raises(RejectedTrajectory):
        await replay_refund_candidate(row, candidate)


@pytest.mark.asyncio
async def test_evaluator_metadata_cannot_change_model_observations_or_reward(sandbox):
    env, facts = sandbox
    transcripts = []
    for label in ("train", "validation", "test"):
        row = {
            **facts,
            "split": label,
            "family": f"private-family-{label}",
            "expected_action": f"private-answer-{label}",
            "reviewer_notes": {"answer": f"private-review-{label}"},
        }
        state = await env.reset(row)
        messages = list(state.context["messages"])
        state, _, _, info = await env.step(state, action("lookup_order", order_id="A"))
        observation = json.loads(info["messages"][0]["content"])
        assert (
            not {"split", "family", "expected_action", "reviewer_notes"}
            & observation.keys()
        )
        messages.extend(info["messages"])
        state, _, _, info = await env.step(
            state, action("refund", order_id="A", amount_cents=100)
        )
        messages.extend(info["messages"])
        state, reward, done, info = await env.step(state, action("finish"))
        assert done and reward == 1 and info["metrics"]["task_success"] == 1
        transcripts.append(messages)
        # Private evidence remains available to the harness, independently of
        # the public observations used to choose actions.
        assert state.context["scenario"]["split"] == label
    assert transcripts[0] == transcripts[1] == transcripts[2]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "environment,generator,prefix",
    [
        (RefundEnvironment, refund_benchmark, "refund"),
        (RefundPolicyEnvironment, refund_policy_benchmark, "policy"),
    ],
)
async def test_model_visible_ids_and_initial_prompts_do_not_name_the_split(
    environment, generator, prefix
):
    seen = set()
    prompts = set()
    for seed in (42, 43):
        for split in ("train", "validation", "test"):
            rows = generator(split, 16, seed)
            assert rows == generator(split, 16, seed)
            for row in rows:
                identity = row["order_id"]
                assert re.fullmatch(prefix + r"-[0-9a-f]{24}", identity)
                assert identity not in seen
                seen.add(identity)
                env = environment()
                state = await env.reset(row)
                prompt = state.context["messages"][0]["content"]
                assert identity in prompt
                assert not re.search(r"\b(train|validation|test)\b", prompt)
                prompts.add(prompt.replace(identity, "<order>"))
                _, _, _, info = await env.step(
                    state, action("lookup_order", order_id=identity)
                )
                observed = json.loads(info["messages"][0]["content"])
                assert observed["order_id"] == identity
                assert not {"family", "split", "expected_action"} & observed.keys()
                assert observed == {
                    key: value for key, value in row.items() if key != "family"
                }
    assert len(prompts) == 1
    assert len(seen) == 96


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_default_benchmark_hides_counter_parity_and_shuffles_balanced_labels(split):
    rows = refund_benchmark(split, 512, 42)
    assert sum(row["eligible"] for row in rows) == 256
    assert rows == refund_benchmark(split, 512, 42)
    assert rows != refund_benchmark(split, 512, 43)
    assert all(len(row["order_id"].rsplit("-", 1)[1]) == 24 for row in rows)
    for predicted in (
        [i % 2 == 0 for i in range(len(rows))],
        [int(row["order_id"][-1], 16) % 2 == 0 for row in rows],
    ):
        accuracy = sum(
            label == row["eligible"] for label, row in zip(predicted, rows, strict=True)
        ) / len(rows)
        assert 0.35 < accuracy < 0.65


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": True},
        {"count": 1.5},
        {"count": 0},
        {"seed": True},
        {"seed": -1},
        {"seed": "42"},
    ],
)
def test_default_benchmark_rejects_ambiguous_seed_and_count(kwargs):
    with pytest.raises(ValueError):
        refund_benchmark("test", **kwargs)
