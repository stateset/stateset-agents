"""ToolUseEnvironment: parsing, the typed registry, budgets and reward paths."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from stateset_agents.core.environment_base import EpisodeStatus
from stateset_agents.core.environments.tool_use import ToolUseEnvironment
from stateset_agents.core.errors import ConfigurationError
from stateset_agents.core.reward_base import RewardType
from stateset_agents.core.reward_context import score_episode
from stateset_agents.core.tool_calls import (
    ToolCallError,
    ToolCallRequest,
    parse_tool_calls,
    strip_tool_calls,
)
from stateset_agents.core.tool_registry import (
    ToolRegistry,
    ToolSpec,
    calculator_tool,
    validate_arguments,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards.math_verifier import MathAnswerVerifier

CALL = (
    '<tool_call>{"name": "calculator", "arguments": {"expression": "12*7"}}</tool_call>'
)
SCENARIO = {"id": "mul", "question": "What is 12*7?", "ground_truth": "84"}


def _scripted(replies: list[str]):
    remaining = iter(replies)
    seen: list[dict[str, Any]] = []

    async def respond(history: Any, context: Any) -> str:
        seen.append(context)
        return next(remaining)

    respond.contexts = seen  # type: ignore[attr-defined]
    return respond


def _environment(**kwargs: Any) -> ToolUseEnvironment:
    kwargs.setdefault("max_tool_calls", 2)
    return ToolUseEnvironment([SCENARIO], [calculator_tool()], **kwargs)


# --- parsing ---------------------------------------------------------------


def test_parses_chat_template_calls_in_order_with_surrounding_text():
    text = (
        'Let me check. <tool_call>{"name": "calculator", "arguments": '
        '{"expression": "1+1"}}</tool_call> and <tool_call>{"name": "search", '
        '"arguments": "{\\"q\\": \\"x\\"}"}</tool_call> done'
    )
    parsed = parse_tool_calls(text)
    assert [c.name for c in parsed.calls] == ["calculator", "search"]
    assert parsed.calls[1].arguments == {"q": "x"}  # arguments as a JSON string
    assert parsed.text == "Let me check.  and  done"
    assert parsed.calls[0].source == "chat_template"


def test_parses_the_repo_json_action_form():
    parsed = parse_tool_calls('{"tool": "calculator", "args": {"expression": "2"}}')
    assert parsed.calls == (
        ToolCallRequest("calculator", {"expression": "2"}, "call_0", "json_action"),
    )
    assert parse_tool_calls('{"tool": "x"}', json_actions=True).errors
    assert not parse_tool_calls('{"tool": "x"}', json_actions=False).attempted


def test_parses_structured_openai_tool_calls():
    turn = ConversationTurn(
        role="assistant",
        content="",
        tool_calls=[
            {"id": "a1", "function": {"name": "calculator", "arguments": '{"e": 1}'}},
            {"function": {"name": "calculator", "arguments": {"e": 2}}},
        ],
    )
    parsed = parse_tool_calls(turn)
    assert [(c.call_id, c.arguments) for c in parsed.calls] == [
        ("a1", {"e": 1}),
        ("call_1", {"e": 2}),
    ]


@pytest.mark.parametrize(
    "reply,message",
    [
        ("<tool_call>{not json}</tool_call>", "invalid JSON"),
        ('<tool_call>{"name": "a", "name": "b"}</tool_call>', "duplicate"),
        (
            '<tool_call>{"name": "a", "arguments": {"x": NaN}}</tool_call>',
            "invalid JSON",
        ),
        ('<tool_call>{"arguments": {}}</tool_call>', "missing tool name"),
        ('<tool_call>{"name": "a", "args": {}}</tool_call>', "unexpected keys"),
        ('<tool_call>{"name": "a", "arguments": [1]}</tool_call>', "JSON object"),
        ('<tool_call>{"name": "a", "arguments": {}}', "unterminated"),
        ("stray </tool_call>", "unterminated"),
        ('<tool_call>["calculator"]</tool_call>', "must be a JSON object"),
        ('{"tool": "a", "args": {}, "extra": 1}', "exactly tool and args"),
        ("[" * 5000 + "]" * 5000, None),
    ],
)
def test_malformed_calls_are_errors_not_exceptions(reply, message):
    parsed = parse_tool_calls(reply)
    if message is None:  # a JSON array is not an action: it is an answer
        assert not parsed.attempted
        return
    assert parsed.errors and not parsed.calls
    assert message in parsed.errors[0].message
    assert isinstance(parsed.errors[0], ToolCallError)


def test_strip_tool_calls_leaves_only_answer_text():
    assert strip_tool_calls(f"{CALL} The answer is 84.") == "The answer is 84."
    assert strip_tool_calls('{"tool": "calculator", "args": {}}') == ""
    assert strip_tool_calls("plain answer") == "plain answer"


# --- registry ---------------------------------------------------------------


def test_tool_spec_validation_and_schema():
    spec = calculator_tool()
    assert spec.schema()["function"]["name"] == "calculator"
    assert spec.schema()["function"]["parameters"]["required"] == ["expression"]
    with pytest.raises(ConfigurationError):
        ToolSpec("bad name!", {"type": "object"}, lambda: 1)
    with pytest.raises(ConfigurationError):
        ToolSpec("t", {"type": "array"}, lambda: 1)
    with pytest.raises(ConfigurationError):
        ToolSpec("t", {"type": "object"}, lambda: 1, timeout_seconds=0)
    with pytest.raises(ConfigurationError):
        ToolRegistry([spec, calculator_tool()])


@pytest.mark.parametrize(
    "arguments,problem",
    [
        ({}, "missing required argument 'n'"),
        ({"n": 1, "x": 2}, "unexpected arguments ['x']"),
        ({"n": "1"}, "must be of type integer"),
        ({"n": True}, "must be of type integer"),
        ({"n": 1.5}, "must be of type integer"),
        ({"n": 1, "mode": "z"}, "must be one of"),
        ({"n": 1, "mode": "a"}, None),
    ],
)
def test_argument_validation(arguments, problem):
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}, "mode": {"enum": ["a", "b"]}},
        "required": ["n"],
        "additionalProperties": False,
    }
    result = validate_arguments(schema, arguments)
    assert (result is None) if problem is None else (problem in result)


def _run(registry: ToolRegistry, name: str, arguments: dict[str, Any]):
    return asyncio.run(registry.execute(ToolCallRequest(name, arguments, "c", "test")))


def test_execution_errors_are_typed_observations():
    def boom(x: int) -> int:
        raise RuntimeError("kaput")

    def slow(x: int) -> int:
        time.sleep(1.0)
        return x

    async def slow_async(x: int) -> int:
        await asyncio.sleep(5)
        return x

    params = {"type": "object", "properties": {"x": {"type": "integer"}}}
    registry = ToolRegistry(
        [
            ToolSpec("boom", params, boom),
            ToolSpec("slow", params, slow, timeout_seconds=0.05),
            ToolSpec("slow_async", params, slow_async, timeout_seconds=0.05),
            ToolSpec("objects", params, lambda x: object()),
            ToolSpec("nan", params, lambda x: float("nan")),
        ]
    )
    cases = {
        "missing": "unknown_tool",
        "boom": "tool_error",
        "slow": "timeout",
        "slow_async": "timeout",
        "objects": "unserializable_result",
        "nan": "unserializable_result",
    }
    for name, error_type in cases.items():
        observation = _run(registry, name, {"x": 1})
        assert not observation.ok and observation.error_type == error_type, name
        payload = json.loads(observation.content())
        assert payload["error"]["type"] == error_type


def test_calculator_is_exact_and_never_evaluates_python():
    registry = ToolRegistry([calculator_tool()])
    assert _run(registry, "calculator", {"expression": "12*7"}).result == 84
    assert _run(registry, "calculator", {"expression": "1/3 + 1/6"}).result == "1/2"
    assert _run(registry, "calculator", {"expression": "2^10"}).result == 1024
    for expression in ("__import__('os')", "9^9^9", "x + 1", "1/0"):
        observation = _run(registry, "calculator", {"expression": expression})
        assert observation.error_type == "tool_error", expression


def test_tools_cannot_mutate_the_logged_call():
    def mutate(items: list) -> int:
        items.append(99)
        return len(items)

    spec = ToolSpec(
        "mutate", {"type": "object", "properties": {"items": {"type": "array"}}}, mutate
    )
    call = ToolCallRequest("mutate", {"items": [1]}, "c", "test")
    observation = asyncio.run(ToolRegistry([spec]).execute(call))
    assert observation.result == 2 and call.arguments == {"items": [1]}


def test_long_observations_are_truncated_but_stay_json():
    spec = ToolSpec("big", {"type": "object"}, lambda: "x" * 10_000)
    observation = _run(ToolRegistry([spec]), "big", {})
    content = observation.content(max_chars=300)
    assert len(content) < 400 and json.loads(content)["truncated"] is True


# --- environment ------------------------------------------------------------


def test_episode_runs_tools_then_ends_on_the_final_answer():
    agent = _scripted([CALL, "The answer is \\boxed{84}."])
    trajectory = asyncio.run(_environment().run_episode(agent, SCENARIO))
    roles = [t.role for t in trajectory.turns]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert json.loads(trajectory.turns[3].content) == {
        "name": "calculator",
        "result": 84,
    }
    state = trajectory.final_state
    assert state["termination"] == "final_answer"
    assert state["final_answer"] == "The answer is \\boxed{84}."
    assert state["tool_calls_used"] == 1
    assert state["tool_log"][0]["arguments"] == {"expression": "12*7"}
    assert trajectory.metadata["status"] == "completed"


def test_malformed_calls_spend_budget_and_exhaustion_ends_the_episode():
    agent = _scripted(["<tool_call>{oops</tool_call>", CALL, CALL])
    trajectory = asyncio.run(_environment().run_episode(agent, SCENARIO))
    observations = [json.loads(t.content) for t in trajectory.turns if t.role == "tool"]
    assert [o.get("error", {}).get("type") for o in observations] == [
        "malformed_call",
        None,
        "budget_exhausted",
    ]
    assert trajectory.final_state["termination"] == "budget_exhausted"
    assert trajectory.final_state["final_answer"] is None
    assert trajectory.metadata["status"] == "failed"


def test_one_turn_with_more_calls_than_budget_stops_at_the_budget():
    agent = _scripted([CALL * 3])
    trajectory = asyncio.run(
        _environment(max_tool_calls=2).run_episode(agent, SCENARIO)
    )
    log = trajectory.final_state["tool_log"]
    assert [entry["id"] for entry in log] == ["call_0", "call_1"]
    assert trajectory.final_state["termination"] == "budget_exhausted"


def test_max_turns_bounds_the_episode():
    env = _environment(max_tool_calls=5, max_turns=2)
    trajectory = asyncio.run(env.run_episode(_scripted([CALL, CALL]), SCENARIO))
    assert trajectory.final_state["termination"] == "max_turns"
    assert trajectory.metadata["status"] == "timeout"


def test_reference_data_is_hidden_from_the_policy():
    agent = _scripted(["\\boxed{84}"])
    scenario = {**SCENARIO, "tests": [{"args": [1], "expected": 2}], "solution": "s"}
    asyncio.run(_environment().run_episode(agent, scenario))
    [context] = agent.contexts
    flat = json.dumps(context, default=str)
    assert "ground_truth" not in flat and "solution" not in flat and "tests" not in flat
    assert context["scenario"]["question"] == SCENARIO["question"]


def test_scenarios_need_a_question_and_tools_are_required():
    with pytest.raises(ConfigurationError):
        ToolUseEnvironment([{"id": "x"}], [calculator_tool()])
    with pytest.raises(ConfigurationError):
        ToolUseEnvironment([SCENARIO], [])
    with pytest.raises(ConfigurationError):
        ToolUseEnvironment([SCENARIO], [calculator_tool()], max_tool_calls=-1)


def test_system_prompt_lists_tools_and_both_call_forms():
    prompt = asyncio.run(_environment().get_initial_prompt(SCENARIO))
    assert "<tool_call>" in prompt and '{"tool"' in prompt and "calculator" in prompt
    custom = asyncio.run(_environment(system_prompt="Be brief.").get_initial_prompt({}))
    assert custom == "Be brief."


@pytest.mark.parametrize("attach", [True, False])
def test_episode_reward_is_the_same_wherever_the_verifier_is_held(attach):
    verifier = MathAnswerVerifier()
    assert verifier.reward_type == RewardType.SPARSE
    env = _environment(reward_fn=verifier if attach else None)
    trajectory = asyncio.run(
        env.run_episode(_scripted([CALL, "The answer is \\boxed{84}."]), SCENARIO)
    )
    total = asyncio.run(score_episode(trajectory, verifier, env, SCENARIO))
    assert total == 1.0 and trajectory.turn_rewards == [0.0, 1.0]


def test_verifier_reads_the_final_answer_not_the_tool_call():
    """A budget-exhausted episode whose calls hold the answer earns nothing."""
    env = _environment(max_tool_calls=1)
    hidden = CALL.replace("12*7", "84")
    trajectory = asyncio.run(env.run_episode(_scripted([hidden, hidden]), SCENARIO))
    total = asyncio.run(score_episode(trajectory, MathAnswerVerifier(), env, SCENARIO))
    assert total == 0.0


def test_batched_episodes_match_sequential_ones():
    env = _environment()
    replies = {
        "a": [CALL, "\\boxed{84}"],
        "b": ["\\boxed{80}"],
    }

    def reply_for(history: list[dict[str, Any]], script: list[str]) -> str:
        return script[sum(1 for m in history if m["role"] == "assistant")]

    scenarios = [dict(SCENARIO, id="a"), dict(SCENARIO, id="b")]

    async def batch(histories, contexts):
        return [
            reply_for(h, replies[c["scenario"]["id"]])
            for h, c in zip(histories, contexts, strict=True)
        ]

    batched = asyncio.run(env.run_episodes_batched(batch, scenarios))
    sequential = [
        asyncio.run(env.run_episode(_scripted(replies[s["id"]]), s)) for s in scenarios
    ]
    for one, other in zip(batched, sequential, strict=True):
        assert [(t.role, t.content) for t in one.turns] == [
            (t.role, t.content) for t in other.turns
        ]


def test_clone_and_reset_are_independent():
    env = _environment(seed=3)
    clone = env.clone()
    assert clone.max_tool_calls == env.max_tool_calls and clone.registry is env.registry
    state = asyncio.run(env.reset())
    assert state.status == EpisodeStatus.ONGOING
    assert state.context["messages"] == [
        {"role": "user", "content": SCENARIO["question"]}
    ]


@pytest.mark.parametrize("attach", [True, False])
def test_per_step_rewards_replay_identically_and_state_is_json(attach):
    """An IMMEDIATE reward is averaged over the episode's steps (tool steps
    included, which is why verifiers are SPARSE) and scores the same whether
    the environment or the trainer holds it."""
    from stateset_agents.core.basic_rewards import CorrectnessReward

    reward = CorrectnessReward()
    env = _environment(reward_fn=reward if attach else None)
    trajectory = asyncio.run(env.run_episode(_scripted([CALL, "It is 84."]), SCENARIO))
    total = asyncio.run(score_episode(trajectory, reward, env, SCENARIO))
    assert total == pytest.approx(0.5)
    json.dumps(trajectory.final_state)
    assert [m["role"] for m in trajectory.final_state["transcript"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


# --- an unanswered episode earns nothing (round 5) ---------------------------

#: The right answer stated beside another call, every turn: the episode runs
#: out of budget and never gives a final answer.
ANSWER_AND_CALL = "The answer is 84. " + CALL


@pytest.mark.parametrize("attach", [True, False])
def test_an_episode_that_ran_out_of_budget_earns_nothing(attach):
    """docs/TOOL_USE_RL.md: an episode that ran out of budget earns nothing,
    even when its last turn states the right answer next to a call; on the
    environment path and the trainer path alike."""
    from stateset_agents.core.reward_context import score_trajectory

    verifier = MathAnswerVerifier()
    env = _environment(reward_fn=verifier if attach else None)
    trajectory = asyncio.run(
        env.run_episode(_scripted([ANSWER_AND_CALL] * 3), SCENARIO)
    )
    assert trajectory.final_state["termination"] == "budget_exhausted"
    assert trajectory.final_state["final_answer"] is None
    if attach:
        assert trajectory.total_reward == 0.0
    total = asyncio.run(score_episode(trajectory, verifier, env, SCENARIO))
    assert total == 0.0
    assert asyncio.run(score_trajectory(MathAnswerVerifier(), trajectory)) == 0.0


def test_an_episode_cut_off_by_max_turns_earns_nothing():
    env = _environment(max_tool_calls=5, max_turns=2)
    trajectory = asyncio.run(
        env.run_episode(_scripted([ANSWER_AND_CALL] * 2), SCENARIO)
    )
    assert trajectory.final_state["termination"] == "max_turns"
    total = asyncio.run(score_episode(trajectory, MathAnswerVerifier(), env, SCENARIO))
    assert total == 0.0


def test_the_code_verifier_also_needs_a_final_answer():
    from stateset_agents.rewards import code_verifier
    from stateset_agents.rewards.code_verifier import SandboxedCodeVerifier

    if not code_verifier.sandbox_available():
        pytest.skip("the sandbox needs POSIX rlimits")
    code = "```python\ndef double(n):\n    return 2 * n\n```\n"
    scenario = {
        "id": "code",
        "question": "Write double(n).",
        "tests": [{"args": [2], "expected": 4}],
        "entry_point": "double",
    }
    env = ToolUseEnvironment([scenario], [calculator_tool()], max_tool_calls=1)
    exhausted = asyncio.run(env.run_episode(_scripted([code + CALL] * 2), scenario))
    assert exhausted.final_state["termination"] == "budget_exhausted"
    verifier = SandboxedCodeVerifier()
    assert asyncio.run(score_episode(exhausted, verifier, env, scenario)) == 0.0
    answered = asyncio.run(env.run_episode(_scripted([code]), scenario))
    assert asyncio.run(score_episode(answered, verifier, env, scenario)) == 1.0


def test_verifiers_without_episode_state_still_score_replies():
    """Single-reply paths carry no termination; nothing changes for them."""
    from stateset_agents.core.reward_context import score_response

    score = asyncio.run(
        score_response(MathAnswerVerifier(), "What is 12*7?", "\\boxed{84}", SCENARIO)
    )
    assert score == 1.0


# --- executed calls feed TaskOutcomeReward (round 5) --------------------------


@pytest.mark.parametrize("attach", [True, False])
def test_executed_tool_calls_are_recorded_for_outcome_rewards(attach):
    from stateset_agents.rewards.task_outcome import TaskOutcomeReward

    scenario = {
        **SCENARIO,
        "expected_response": "The answer is 84.",
        "expected_tool_calls": [{"tool": "calculator", "args": {"expression": "12*7"}}],
    }
    reward = TaskOutcomeReward()
    env = _environment(reward_fn=reward if attach else None)
    replies = ["<tool_call>{oops</tool_call>", CALL, "The answer is 84."]
    trajectory = asyncio.run(env.run_episode(_scripted(replies), scenario))
    # The malformed call never executed, so only the calculator call counts.
    assert trajectory.final_state["executed_tool_calls"] == [
        {"tool": "calculator", "args": {"expression": "12*7"}}
    ]
    assert asyncio.run(score_episode(trajectory, reward, env, scenario)) == 1.0
