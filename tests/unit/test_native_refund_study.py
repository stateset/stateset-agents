"""Reviewed cases and real environment observations define study semantics."""

import json
from types import SimpleNamespace

import pytest

from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
    _resolution,
    refund_policy_benchmark,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.training.native_refund_study import evaluate, load_cases


def test_reviewed_partitions_reject_renamed_duplicate_cases(tmp_path):
    data = {
        "provenance": {"source": "reviewed", "reviewer": "test-reviewer"},
        **{
            split: refund_policy_benchmark(split, 32, 0)
            for split in ("train", "validation", "test")
        },
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    assert load_cases(path)["provenance"]["source"] == "reviewed"
    duplicate = dict(data["train"][0], order_id="renamed-duplicate")
    data["test"][0] = duplicate
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="disjoint"):
        load_cases(path)


@pytest.mark.asyncio
async def test_native_episode_delivers_tool_observations_and_private_labels_stay_private():
    environment = RefundPolicyEnvironment()
    case = refund_policy_benchmark("train", 8, 0)[0]
    case["expected_response"] = "private-reference"
    messages_seen = []

    async def policy(messages, context):
        messages_seen.append(list(messages))
        assert "scenario" not in context
        if len(messages_seen) == 1:
            tool, arguments = "lookup_order", {"order_id": case["order_id"]}
        elif len(messages_seen) == 2:
            tool, value = _resolution(case)
            arguments = {
                "order_id": case["order_id"],
                "amount_cents" if tool == "refund" else "reason": value,
            }
            assert case["order_id"] in messages[-1]["content"]
        else:
            tool, arguments = "finish", {}
        return ConversationTurn(
            role="assistant", content=json.dumps({"tool": tool, "args": arguments})
        )

    result = await environment.run_episode(policy, case)
    assert len(messages_seen) == 3
    assert messages_seen[0][0]["role"] == "user"
    assert result.total_reward == 1.0


@pytest.mark.asyncio
async def test_evaluation_binds_code_identity_and_retains_failed_outcomes():
    from stateset_agents.evaluation.implementation import package_implementation

    async def policy(messages, context):
        return ConversationTurn(role="assistant", content='{"tool":"finish","args":{}}')

    agent = SimpleNamespace(
        config=SimpleNamespace(model_name="test-model"), generate_turn=policy
    )
    identity = package_implementation()
    report = await evaluate(
        agent,
        refund_policy_benchmark("test", 8, 0)[:1],
        seed=42,
        settings={"test": True},
        checkpoint="initial",
        implementation=identity,
    )
    assert report["implementation"] == identity
    assert report["total"] == 1 and report["passed"] == 0
    assistant = next(
        turn for turn in report["outcomes"][0]["trace"] if turn["role"] == "assistant"
    )
    assert assistant["content"] == '{"tool":"finish","args":{}}'
