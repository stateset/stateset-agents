"""Industry replay checks declared outcomes against fresh fictional sandboxes."""

import copy
import json
import subprocess
import sys

import pytest

from stateset_agents.core.environments.industry_support_environment import (
    POLICIES,
    IndustrySupportEnvironment,
    support_cases,
    support_resolution,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.environment_trace import (
    replay_environment_trace,
    replay_industry_trace,
)


async def make_trace(domain, case, mode="success"):
    env = IndustrySupportEnvironment(domain)
    state = await env.reset(case)
    trace = {
        "schema_version": 1,
        "case_id": case["record_id"],
        "case_hash": content_hash(case),
        "initial_messages": copy.deepcopy(state.context["messages"]),
        "steps": [],
    }
    actions = [
        ("lookup_record", {"record_id": case["record_id"]}),
        (
            "resolve",
            {
                "record_id": case["record_id"],
                "resolution": support_resolution(domain, case),
            },
        ),
        ("finish", {}),
    ]
    if mode == "wrong_resolution":
        actions[1][1]["resolution"] = "unapproved_override"
    elif mode == "early_finish":
        actions = actions[-1:]
    elif mode == "duplicate":
        actions.insert(2, copy.deepcopy(actions[1]))
    elif mode == "timeout":
        actions = actions[:1] * 4
    elif mode == "external_stop":
        actions = actions[:1]
    total = 0.0
    for tool, args in actions:
        action = {
            "role": "assistant",
            "content": json.dumps({"tool": tool, "args": args}),
        }
        state, reward, done, info = await env.step(state, ConversationTurn(**action))
        total += reward
        trace["steps"].append(
            {
                "action": action,
                "reward": reward,
                "done": done,
                "status": state.status.value,
                "observations": info["messages"],
                "metrics": info["metrics"],
            }
        )
    trace["terminal"] = {
        "reward": -1.0 if mode == "external_stop" else total,
        "status": "timeout" if mode == "external_stop" else state.status.value,
        "truncated": (
            "generated_tokens"
            if mode == "external_stop"
            else "environment_timeout" if mode == "timeout" else None
        ),
    }
    return trace


@pytest.mark.asyncio
@pytest.mark.parametrize("domain", POLICIES)
@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "wrong_resolution",
        "early_finish",
        "duplicate",
        "timeout",
        "external_stop",
    ],
)
async def test_every_industry_family_replays_success_failure_and_truncation(
    domain, mode
):
    for case in support_cases(domain, "train", 8, 42):
        trace = await make_trace(domain, case, mode)
        before = copy.deepcopy((case, trace))
        result = await replay_industry_trace(domain, case, trace)
        assert result["success"] is (mode == "success")
        assert result["reward"] == (1.0 if mode == "success" else -1.0)
        assert result["tool_calls"] == len(trace["steps"])
        assert result["external_stop_declared"] is (mode == "external_stop")
        assert result["policy_violations"] == int(
            mode in {"wrong_resolution", "duplicate"}
        )
        assert (case, trace) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "case",
        "case_id",
        "prompt",
        "observation",
        "action",
        "metrics",
        "done",
        "terminal",
        "extra_action",
        "schema",
        "nan",
    ],
)
async def test_industry_replay_rejects_fabricated_evidence(mutation):
    case = support_cases("travel", "train", 8, 42)[0]
    trace = await make_trace("travel", case)
    if mutation == "case":
        case["quoted_price_cents"] += 1
    elif mutation == "case_id":
        trace["case_id"] = "another-case"
    elif mutation == "prompt":
        trace["initial_messages"][0]["content"] = "forged policy"
    elif mutation == "observation":
        trace["steps"][0]["observations"] = []
    elif mutation == "action":
        trace["steps"][1]["action"]["content"] = "I completed everything."
    elif mutation == "metrics":
        trace["steps"][0]["metrics"]["tool_calls"] = 900
    elif mutation == "done":
        trace["steps"][0]["done"] = True
    elif mutation == "terminal":
        trace["terminal"]["reward"] = 10
    elif mutation == "extra_action":
        trace["steps"].append(copy.deepcopy(trace["steps"][-1]))
    elif mutation == "schema":
        trace["schema_version"] = True
    else:
        trace["steps"][0]["reward"] = float("nan")
    with pytest.raises(ValueError):
        await replay_industry_trace("travel", case, trace)


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", [None, "", "environment_timeout", "missing"])
async def test_incomplete_industry_trace_requires_explicit_external_stop(cause):
    case = support_cases("services", "train", 8, 42)[0]
    trace = await make_trace("services", case, "external_stop")
    trace["terminal"]["truncated"] = cause
    with pytest.raises(ValueError, match="external stop"):
        await replay_industry_trace("services", case, trace)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", [None, "reset", "replay"])
async def test_generic_replay_closes_its_sandbox_on_success_and_failure(
    asynchronous, failure
):
    case = support_cases("saas", "train", 8, 42)[0]
    trace = await make_trace("saas", case)
    closed = []

    class Tracked(IndustrySupportEnvironment):
        async def reset(self, scenario=None):
            if failure == "reset":
                raise ConnectionError("offline reset failure")
            return await super().reset(scenario)

        def close(self):
            closed.append(True)

    if asynchronous:

        async def close(self):
            closed.append(True)

        Tracked.close = close
    if failure == "replay":
        trace["initial_messages"] = []
    invocation = replay_environment_trace(
        lambda: Tracked("saas"), case, trace, case_id=case["record_id"]
    )
    if failure:
        with pytest.raises(ConnectionError if failure == "reset" else ValueError):
            await invocation
    else:
        assert (await invocation)["success"]
    assert closed == [True]


def test_offline_replay_import_does_not_load_training_or_provider_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from stateset_agents.evaluation.environment_trace import replay_industry_trace; assert not any(name in sys.modules for name in ('torch', 'transformers', 'river_client'))",
        ],
        check=True,
        timeout=20,
    )
