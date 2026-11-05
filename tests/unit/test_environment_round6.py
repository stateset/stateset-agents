"""Round-6 environment defects: what the policy sees, tool-call ids, tool
timeouts under load, observation bounds, scenario-forged termination and
JSON-only final answers.

E1  Reference data nested inside an allowlisted value (a scenario's
    ``"context"``, a task's ``"instructions"``) reached the policy, through
    its context and the rendered system prompt. Every test here runs real
    episodes with an ``agent_fn`` that records every argument it receives
    (history, rendered system prompt, context) and checks the secret appears
    nowhere in it, while the reward still sees it.
E2  Tool turns answered ids the environment invented (``call_N``) instead of
    the structured call's own id, and the history dropped ``tool_call_id``,
    ``name`` and ``tool_calls``.
E4  Healthy synchronous tools timed out when the shared executor was busy.
E7  Observations overran their bound; a scenario's ``termination`` could
    make an unanswered episode score; JSON-only final answers spent budget.
"""

from __future__ import annotations

import asyncio
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.environments.refund_delegation_environment import (
    RefundDelegationEnvironment,
)
from stateset_agents.core.environments.refund_environment import RefundEnvironment
from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
)
from stateset_agents.core.environments.tool_use import ToolUseEnvironment
from stateset_agents.core.errors import ConfigurationError
from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.reward_context import (
    build_reward_context,
    score_episode,
    trajectory_reward_context,
    unanswered_termination,
)
from stateset_agents.core.task_environment import TaskEnvironment
from stateset_agents.core.tool_calls import ToolCallRequest, parse_tool_calls
from stateset_agents.core.tool_registry import (
    ToolObservation,
    ToolRegistry,
    ToolSpec,
    calculator_tool,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards.math_verifier import MathAnswerVerifier

SECRET = "SECRET_84"
#: A scenario context mixing customer facts the policy needs with reference
#: data it must never see, at several depths.
CONTEXT: dict[str, Any] = {
    "expected_refund": SECRET,
    "customer_tier": "gold",
    "order_id": "A1",
    "account": {
        "status": "active",
        "goldAnswer": SECRET,
        "notes": [{"text": "prefers email", "label": SECRET}],
    },
    "grading": {"rubric": SECRET},
    "history": [{"ground_truth": SECRET, "event": "delivered"}],
}
CALL = (
    '<tool_call>{"name": "calculator", "arguments": {"expression": "12*7"}}</tool_call>'
)


class _Recorder:
    """A scripted ``agent_fn`` that records every argument it receives."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, history: Any, context: Any) -> Any:
        self.calls.append(
            {"history": copy.deepcopy(history), "context": copy.deepcopy(context)}
        )
        return self.replies[min(len(self.calls), len(self.replies)) - 1]

    async def batch(self, histories: Any, contexts: Any) -> list[Any]:
        return [await self(h, c) for h, c in zip(histories, contexts, strict=True)]

    def seen(self) -> str:
        return json.dumps(self.calls, default=str)


class _ContextRecorder(RewardFunction):
    """A SPARSE reward that records the context it is scored with."""

    def __init__(self) -> None:
        super().__init__(reward_type=RewardType.SPARSE)
        self.contexts: list[dict[str, Any]] = []

    async def compute_reward(self, turns, context=None):
        self.contexts.append(copy.deepcopy(dict(context or {})))
        return RewardResult(score=0.0)


def _policy_view(env: Any, scenario: dict[str, Any], recorder: _Recorder) -> str:
    """Everything the policy received: every agent_fn argument (the system
    prompt is the history's first turn) and the rendered initial prompt."""
    prompt = asyncio.run(env.get_initial_prompt(scenario))
    return recorder.seen() + json.dumps(prompt)


def _assert_reward_still_sees_secret(reward: _ContextRecorder, trajectory) -> None:
    assert reward.contexts, "the reward was never applied"
    assert SECRET in json.dumps(reward.contexts, default=str)
    assert SECRET in json.dumps(trajectory_reward_context(trajectory), default=str)


# --- E1: the recursive allowlist --------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "expected_refund",
        "expectedRefund",
        "gold",
        "gold_answer",
        "goldAnswer",
        "golden_output",
        "answer",
        "answers",
        "final_answer",
        "ground_truth",
        "groundTruth",
        "groundtruth",
        "reference",
        "reference_solution",
        "references",
        "solution",
        "label",
        "labels",
        "correct_facts",
        "correctAction",
        "target",
        "target_expression",
        "rubric",
        "grading",
        "grade",
        "tests",
        "test_cases",
        "unit_tests",
        "hidden",
        "hidden_tests",
        "hiddenTests",
        "secret",
        "secret_code",
        "required_actions",
        "expected_tool_calls",
        "success_criteria",
        "oracle_output",
        "judge_notes",
        "ref_answer",
        "gt",
        "Expected-Outcome",
    ],
)
def test_reference_like_nested_keys_are_recognized(key):
    from stateset_agents.core.environment_base import is_reference_key

    assert is_reference_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "customer_tier",
        "order_id",
        "status",
        "notes",
        "text",
        "event",
        "refund_amount",
        "contest_id",
        "latest_order",
        "background",
        "role",
        "content",
        "tool_call_id",
        "name",
        "arguments",
        "result",
        "error_type",
    ],
)
def test_ordinary_nested_keys_stay_visible(key):
    from stateset_agents.core.environment_base import is_reference_key

    assert not is_reference_key(key)


def test_policy_visible_filters_every_depth_and_honours_opt_in():
    from stateset_agents.core.environment_base import policy_visible

    visible = policy_visible(CONTEXT)
    assert SECRET not in json.dumps(visible)
    assert visible["customer_tier"] == "gold" and visible["order_id"] == "A1"
    assert visible["account"] == {
        "status": "active",
        "notes": [{"text": "prefers email"}],
    }
    assert visible["history"] == [{"event": "delivered"}]
    assert "grading" not in visible
    shown = policy_visible(CONTEXT, ("expected_refund",))
    assert shown["expected_refund"] == SECRET  # an explicit opt-in
    assert policy_visible(("a", {"answer": 1, "b": 2})) == ("a", {"b": 2})
    assert policy_visible("free text is shown as written") == (
        "free text is shown as written"
    )


def test_conversation_environment_never_shows_nested_reference_data():
    scenario = {
        "id": "c1",
        "user_query": "Where is my refund?",
        "topic": "refunds",
        "context": CONTEXT,
        "user_turns": ["Where is my refund?", "Order A1, please."],
    }
    reward = _ContextRecorder()
    env = ConversationEnvironment([scenario], max_turns=3, reward_fn=reward)
    agent = _Recorder(["Let me check order A1.", "It is on its way."])
    trajectory = asyncio.run(env.run_episode(agent, scenario))
    asyncio.run(env.run_episodes_batched(agent.batch, [scenario, scenario]))
    seen = _policy_view(env, scenario, agent)
    assert seen.count(SECRET) == 0
    # The rendered system prompt and the context keep the customer facts.
    system = agent.calls[0]["history"][0]
    assert system["role"] == "system" and "'customer_tier': 'gold'" in system["content"]
    assert "'order_id': 'A1'" in system["content"]
    assert agent.calls[0]["context"]["scenario"]["context"]["customer_tier"] == "gold"
    _assert_reward_still_sees_secret(reward, trajectory)


def test_structured_user_turns_never_carry_reference_data():
    scenario = {
        "id": "c2",
        "user_turns": [
            "Where is my refund?",
            {"text": "Order A1.", "expected_reply": SECRET},
        ],
    }
    env = ConversationEnvironment([scenario], max_turns=3)
    agent = _Recorder(["Which order?", "Refunded."])
    trajectory = asyncio.run(env.run_episode(agent, scenario))
    assert SECRET not in agent.seen()
    assert "Order A1." in agent.seen()
    unscripted = {"id": "c3", "user_query": "hi", "user_responses": [scenario]}
    env = ConversationEnvironment([unscripted], max_turns=2, seed=0)
    agent = _Recorder(["hello", "bye now"])
    asyncio.run(env.run_episode(agent, unscripted))
    assert SECRET not in agent.seen() and "Order A1." in agent.seen()
    assert SECRET in json.dumps(trajectory_reward_context(trajectory), default=str)


def test_context_made_only_of_reference_data_renders_no_system_turn():
    env = ConversationEnvironment(
        [{"id": "c", "user_query": "hi", "context": {"expected_answer": SECRET}}],
        max_turns=1,
    )
    assert asyncio.run(env.get_initial_prompt(env.scenarios[0])) == ""


def test_task_environment_never_shows_nested_reference_data():
    task = {
        "id": "t1",
        "description": {"summary": "Refund order A1", "expected_result": SECRET},
        "instructions": {"steps": ["look up", "refund"], "answer": SECRET},
        "goal": {"outcome": "customer refunded", "gold": SECRET},
        "context": CONTEXT,
        "required_actions": ["lookup", "refund"],
    }
    reward = _ContextRecorder()
    env = TaskEnvironment(
        [task], success_criteria=lambda turns, ctx: False, max_turns=2, reward_fn=reward
    )
    agent = _Recorder(['{"action": "lookup"}', '{"action": "refund"}'])
    trajectory = asyncio.run(env.run_episode(agent, task))
    seen = _policy_view(env, task, agent)
    assert seen.count(SECRET) == 0
    system = agent.calls[0]["history"][0]["content"]
    assert "Refund order A1" in system and "'look up', 'refund'" in system
    assert agent.calls[0]["context"]["task"]["context"]["order_id"] == "A1"
    assert agent.calls[0]["context"]["task_goal"] == {"outcome": "customer refunded"}
    _assert_reward_still_sees_secret(reward, trajectory)


def test_tool_use_environment_never_shows_nested_reference_data():
    scenario = {
        "id": "q1",
        "question": "What is 12*7?",
        "ground_truth": "84",
        "context": CONTEXT,
        "examples": [{"input": "2*2", "expected_output": SECRET, "note": "shown"}],
    }
    reward = _ContextRecorder()
    env = ToolUseEnvironment(
        [scenario], [calculator_tool()], max_tool_calls=2, reward_fn=reward
    )
    agent = _Recorder([CALL, "The answer is 84."])
    trajectory = asyncio.run(env.run_episode(agent, scenario))
    asyncio.run(env.run_episodes_batched(agent.batch, [scenario, scenario]))
    seen = _policy_view(env, scenario, agent)
    assert seen.count(SECRET) == 0
    context = agent.calls[0]["context"]["scenario"]
    assert context["context"]["customer_tier"] == "gold"
    assert context["examples"] == [{"input": "2*2", "note": "shown"}]
    _assert_reward_still_sees_secret(reward, trajectory)


@pytest.mark.parametrize(
    "build,row,replies",
    [
        (
            RefundEnvironment,
            {"order_id": "o-1", "amount_cents": 500, "eligible": True},
            [
                '{"tool": "lookup_order", "args": {"order_id": "o-1"}}',
                '{"tool": "refund", "args": {"order_id": "o-1", "amount_cents": 500}}',
                '{"tool": "finish", "args": {}}',
            ],
        ),
        (
            RefundPolicyEnvironment,
            {
                "order_id": "p-1",
                "paid_cents": 900,
                "refunded_cents": 0,
                "days_since_delivery": 3,
                "return_window_days": 30,
                "status": "delivered",
                "chargeback_open": False,
                "customer_note": "Please refund.",
            },
            [
                '{"tool": "lookup_order", "args": {"order_id": "p-1"}}',
                '{"tool": "finish", "args": {}}',
            ],
        ),
        (
            RefundDelegationEnvironment,
            {
                "order_id": "d-1",
                "paid_cents": 900,
                "refunded_cents": 0,
                "days_since_delivery": 3,
                "return_window_days": 30,
                "status": "delivered",
                "chargeback_open": False,
                "customer_note": "Please refund.",
            },
            [
                '{"tool": "resolve_refund", "args": {"intent": "resolve_refund"}}',
                '{"tool": "finish", "args": {}}',
            ],
        ),
    ],
)
def test_refund_environments_never_show_nested_reference_data(build, row, replies):
    scenario = {**row, "context": CONTEXT}
    env = build()
    reward = _ContextRecorder()
    env.reward_fn = reward
    agent = _Recorder(replies)
    trajectory = asyncio.run(env.run_episode(agent, scenario))
    seen = _policy_view(env, scenario, agent)
    assert len(agent.calls) == len(replies)
    assert seen.count(SECRET) == 0
    _assert_reward_still_sees_secret(reward, trajectory)
    if build is RefundEnvironment:  # the one refund sandbox that shows a context
        shown = agent.calls[0]["context"]["scenario"]["context"]
        assert shown["customer_tier"] == "gold" and shown["order_id"] == "A1"


@pytest.mark.parametrize("environment", ["conversation", "tool_use"])
def test_a_real_agent_prompt_never_contains_nested_reference_data(environment):
    """The prompt a MultiTurnAgent (stub backend) actually renders."""
    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

    agent = MultiTurnAgent(
        AgentConfig(model_name="stub://test", use_stub_model=True, max_new_tokens=16)
    )
    asyncio.run(agent.initialize())
    prompts: list[str] = []
    render = agent._render_prompt

    def spy(messages: Any, context: Any = None) -> Any:
        prompt, rendered = render(messages, context)
        prompts.append(prompt)
        return prompt, rendered

    agent._render_prompt = spy  # type: ignore[method-assign]
    if environment == "conversation":
        scenario: dict[str, Any] = {"id": "c", "user_query": "hi", "context": CONTEXT}
        env: Any = ConversationEnvironment([scenario], max_turns=2)
    else:
        scenario = {"id": "q", "question": "12*7?", "context": CONTEXT}
        env = ToolUseEnvironment([scenario], [calculator_tool()])
    asyncio.run(env.run_episode(agent.generate_response, scenario))
    assert prompts and not any(SECRET in prompt for prompt in prompts)
    if environment == "conversation":
        assert "customer_tier" in prompts[0] and "A1" in prompts[0]


def test_opting_a_nested_key_in_shows_it():
    scenario = {"id": "c", "user_query": "hi", "context": CONTEXT}
    env = ConversationEnvironment(
        [scenario], max_turns=1, visible_context_keys=("expected_refund",)
    )
    agent = _Recorder(["ok"])
    asyncio.run(env.run_episode(agent, scenario))
    assert agent.calls[0]["context"]["scenario"]["context"]["expected_refund"] == (
        SECRET
    )
    assert "goldAnswer" not in agent.seen()


# --- E2: tool-call ids round-trip -------------------------------------------


def _structured(call_id: str | None, expression: str) -> dict[str, Any]:
    call: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "calculator",
            "arguments": json.dumps({"expression": expression}),
        },
    }
    if call_id is not None:
        call["id"] = call_id
    return call


def test_parser_keeps_a_structured_calls_own_id():
    turn = ConversationTurn(
        role="assistant",
        content="",
        tool_calls=[_structured("call_abc123", "1"), _structured(None, "2")],
    )
    first, second = parse_tool_calls(turn).calls
    assert (first.call_id, first.has_own_id) == ("call_abc123", True)
    assert (second.call_id, second.has_own_id) == ("call_1", False)
    [template] = parse_tool_calls(CALL).calls
    assert not template.has_own_id


@pytest.mark.parametrize("batched", [False, True])
def test_structured_tool_call_ids_round_trip_through_history(batched):
    scenario = {"id": "q", "question": "What is 12*7?"}
    env = ToolUseEnvironment([scenario], [calculator_tool()], max_tool_calls=3)
    calls = [_structured("call_abc123", "12*7"), _structured(None, "1+1")]
    reply = ConversationTurn(role="assistant", content="", tool_calls=calls)
    agent = _Recorder([reply, "The answer is 84."])
    if batched:
        [trajectory] = asyncio.run(env.run_episodes_batched(agent.batch, [scenario]))
    else:
        trajectory = asyncio.run(env.run_episode(agent, scenario))
    state = trajectory.final_state
    assert [entry["id"] for entry in state["tool_log"]] == ["call_abc123", "call_1"]
    tool_turns = [t for t in trajectory.turns if t.role == "tool"]
    assert [t.metadata["tool_call_id"] for t in tool_turns] == [
        "call_abc123",
        "call_1",
    ]
    assert {t.metadata["name"] for t in tool_turns} == {"calculator"}
    # The history the policy answered from pairs each observation with its call.
    history = agent.calls[-1]["history"]
    assistant = [m for m in history if m["role"] == "assistant"][0]
    assert assistant["tool_calls"] == calls
    observations = [m for m in history if m["role"] == "tool"]
    assert [(m["tool_call_id"], m["name"]) for m in observations] == [
        ("call_abc123", "calculator"),
        ("call_1", "calculator"),
    ]
    assert json.loads(observations[0]["content"]) == {
        "name": "calculator",
        "result": 84,
    }
    # The JSON transcript keeps the same links.
    transcript = state["transcript"]
    assert transcript[1]["tool_calls"] == calls
    assert [m["tool_call_id"] for m in transcript if m["role"] == "tool"] == [
        "call_abc123",
        "call_1",
    ]
    json.dumps(state)


def test_generated_ids_never_collide_with_a_calls_own_id():
    scenario = {"id": "q", "question": "What is 12*7?"}
    env = ToolUseEnvironment([scenario], [calculator_tool()], max_tool_calls=3)
    own = ConversationTurn(
        role="assistant", content="", tool_calls=[_structured("call_1", "2")]
    )
    agent = _Recorder([own, CALL, "84"])
    trajectory = asyncio.run(env.run_episode(agent, scenario))
    ids = [entry["id"] for entry in trajectory.final_state["tool_log"]]
    assert ids == ["call_1", "call_2"]
    history = agent.calls[-1]["history"]
    assert [m["tool_call_id"] for m in history if m["role"] == "tool"] == ids


# --- E4: synchronous tools under load ----------------------------------------


def _napping_registry(seconds: float, timeout: float) -> ToolRegistry:
    def nap(x: int) -> int:
        time.sleep(seconds)
        return x

    return ToolRegistry(
        [ToolSpec("nap", {"type": "object"}, nap, timeout_seconds=timeout)]
    )


async def _concurrent_calls(registry: ToolRegistry, count: int) -> list[Any]:
    return await asyncio.gather(
        *(
            registry.execute(ToolCallRequest("nap", {"x": i}, f"c{i}", "test"))
            for i in range(count)
        )
    )


def test_concurrent_sync_calls_never_time_out_behind_a_saturated_executor():
    """64 healthy calls (0.05s each, timeout 2s) while every worker of the
    loop's shared executor is busy: none may time out."""
    release = threading.Event()

    async def scenario() -> list[Any]:
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=4)
        loop.set_default_executor(executor)
        for _ in range(4):  # other work holding every shared worker
            loop.run_in_executor(None, release.wait, 30)
        try:
            return await _concurrent_calls(_napping_registry(0.05, 2.0), 64)
        finally:
            release.set()

    observations = asyncio.run(scenario())
    assert [o.error_type for o in observations if not o.ok] == []
    assert sorted(o.result for o in observations) == list(range(64))


def test_concurrent_sync_calls_never_queue_into_timeouts():
    """64 calls of 0.2s with a 2s timeout and a 4-worker shared executor
    (16 waves, 3.2s if queued): every call succeeds."""

    async def scenario() -> list[Any]:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(4))
        return await _concurrent_calls(_napping_registry(0.2, 2.0), 64)

    observations = asyncio.run(scenario())
    assert sum(o.error_type == "timeout" for o in observations) == 0
    assert all(o.ok for o in observations)


def test_a_stuck_sync_tool_still_times_out_and_does_not_block_others():
    release = threading.Event()

    def stuck() -> int:
        release.wait(30)
        return 1

    def quick() -> int:
        return 2

    registry = ToolRegistry(
        [
            ToolSpec("stuck", {"type": "object"}, stuck, timeout_seconds=0.2),
            ToolSpec("quick", {"type": "object"}, quick, timeout_seconds=2.0),
        ]
    )

    async def scenario() -> tuple[Any, list[Any], float]:
        started = time.monotonic()
        stuck_calls = [
            registry.execute(ToolCallRequest("stuck", {}, f"s{i}", "test"))
            for i in range(40)
        ]
        quick_calls = [
            registry.execute(ToolCallRequest("quick", {}, f"q{i}", "test"))
            for i in range(8)
        ]
        results = await asyncio.gather(*stuck_calls, *quick_calls)
        return results[:40], results[40:], time.monotonic() - started

    try:
        stuck_results, quick_results, elapsed = asyncio.run(scenario())
    finally:
        release.set()
    assert {o.error_type for o in stuck_results} == {"timeout"}
    assert all(o.ok and o.result == 2 for o in quick_results)
    assert elapsed < 10.0  # bounded by the 0.2s timeout, not the 30s wait


def test_async_tools_are_awaited_and_cancelled_at_their_timeout():
    cancelled = threading.Event()

    async def fast(x: int) -> int:
        await asyncio.sleep(0)
        return x * 2

    async def slow(x: int) -> int:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return x

    params = {"type": "object", "properties": {"x": {"type": "integer"}}}
    registry = ToolRegistry(
        [
            ToolSpec("fast", params, fast),
            ToolSpec("slow", params, slow, timeout_seconds=0.1),
        ]
    )

    async def scenario() -> tuple[Any, Any]:
        return (
            await registry.execute(ToolCallRequest("fast", {"x": 21}, "a", "t")),
            await registry.execute(ToolCallRequest("slow", {"x": 1}, "b", "t")),
        )

    ok, timed_out = asyncio.run(scenario())
    assert ok.ok and ok.result == 42
    assert timed_out.error_type == "timeout" and cancelled.is_set()


def test_sync_tool_errors_and_awaitable_results_are_unchanged():
    async def later() -> int:
        return 7

    def boom() -> int:
        raise ValueError("bad input")

    registry = ToolRegistry(
        [
            ToolSpec("boom", {"type": "object"}, boom),
            ToolSpec("later", {"type": "object"}, later),
            ToolSpec("deferred", {"type": "object"}, lambda: later()),
        ]
    )

    async def scenario() -> list[Any]:
        return [
            await registry.execute(ToolCallRequest(name, {}, name, "t"))
            for name in ("boom", "later", "deferred")
        ]

    boom_obs, later_obs, deferred_obs = asyncio.run(scenario())
    assert boom_obs.error_type == "tool_error"
    assert boom_obs.error_message == "ValueError: bad input"
    assert later_obs.result == 7 and deferred_obs.result == 7


# --- E7(a): observations never exceed their bound ------------------------------

ADVERSARIAL_RESULTS: list[Any] = [
    '"' * 5000,
    "\\" * 5000,
    '"\\' * 3000,
    "中文😀" * 2000,
    " \x00\x1f\x7f" * 1500,
    {"a": [{"b": '"\\' * 50, "c": "😀" * 20, "d": [None, True, 1.5]}] * 100},
    [[["\\u0000", '"']]] * 800,
    [1.5] * 3000,
]
# Short ids: pytest exports each test id in PYTEST_CURRENT_TEST, and Windows
# rejects environment values longer than 32767 characters.
ADVERSARIAL_IDS = [
    "quotes",
    "backslashes",
    "quote-backslash",
    "cjk-emoji",
    "control-chars",
    "nested-dict",
    "nested-lists",
    "float-list",
]


@pytest.mark.parametrize("bound", [128, 300, 2000])
@pytest.mark.parametrize("result", ADVERSARIAL_RESULTS, ids=ADVERSARIAL_IDS)
def test_truncated_results_respect_the_bound_and_stay_json(bound, result):
    observation = ToolObservation("big", "c", True, result)
    content = observation.content(bound)
    assert len(content) <= bound
    payload = json.loads(content)
    assert payload["truncated"] is True and payload["name"] == "big"
    full = json.dumps(result, sort_keys=True)
    assert full.startswith(payload["result"])
    assert len(payload["result"]) > 0


@pytest.mark.parametrize("bound", [128, 300, 2000])
def test_truncated_errors_keep_their_type_within_the_bound(bound):
    observation = ToolObservation.failure(
        "x" * 5000, "c", "unserializable_result", '"\\😀' * 3000
    )
    content = observation.content(bound)
    assert len(content) <= bound
    payload = json.loads(content)
    assert payload["truncated"] is True
    assert payload["error"]["type"] == "unserializable_result"
    assert ("x" * 5000).startswith(payload["name"])


def test_small_observations_are_unchanged_and_tiny_bounds_rejected():
    observation = ToolObservation("calculator", "c", True, 84)
    assert observation.content() == '{"name": "calculator", "result": 84}'
    with pytest.raises(ValueError):
        observation.content(64)
    with pytest.raises(ConfigurationError):
        ToolUseEnvironment(
            [{"id": "q", "question": "q?"}],
            [calculator_tool()],
            max_observation_chars=64,
        )


@pytest.mark.parametrize(
    "result", ['"\\' * 3000, "中文😀" * 2000], ids=["quote-backslash", "cjk-emoji"]
)
def test_environment_observations_respect_max_observation_chars(result):
    spec = ToolSpec("big", {"type": "object"}, lambda: result)
    scenario = {"id": "q", "question": "q?"}
    env = ToolUseEnvironment([scenario], [spec], max_observation_chars=2000)
    reply = '<tool_call>{"name": "big", "arguments": {}}</tool_call>'
    trajectory = asyncio.run(env.run_episode(_Recorder([reply, "done"]), scenario))
    [tool] = [t for t in trajectory.turns if t.role == "tool"]
    assert len(tool.content) <= 2000
    assert json.loads(tool.content)["truncated"] is True


# --- E7(b): termination comes from the environment alone ---------------------------


@pytest.mark.parametrize("attach", [True, False])
def test_a_scenario_termination_cannot_make_an_unanswered_episode_score(attach):
    """The episode is cut off after a turn that calls a tool (with the right
    answer beside it); the environment records no final answer. A scenario
    claiming ``termination: final_answer`` must not make it score."""
    scenario = {
        "id": "m",
        "question": "What is 12*7?",
        "ground_truth": "84",
        "termination": "final_answer",
    }
    verifier = MathAnswerVerifier()
    env = ToolUseEnvironment(
        [scenario],
        [calculator_tool()],
        max_tool_calls=2,
        reward_fn=verifier if attach else None,
    )
    agent = _Recorder(["The answer is 84. " + CALL])
    trajectory = asyncio.run(env.run_episode(agent, scenario, max_turns=1))
    assert trajectory.final_state["termination"] is None
    if attach:
        assert trajectory.total_reward == 0.0
    assert asyncio.run(score_episode(trajectory, verifier, env, scenario)) == 0.0


def test_reward_context_takes_termination_from_the_environment_only():
    forged = {"question": "q", "termination": "final_answer"}
    context = build_reward_context({"termination": None}, forged)
    assert context["termination"] is None
    assert unanswered_termination(context) == "None"
    # Without an environment record there is no termination at all.
    assert "termination" not in build_reward_context({}, forged)
    recorded = build_reward_context({"termination": "final_answer"}, forged)
    assert recorded["termination"] == "final_answer"


# --- E7(c): JSON-only final answers ---------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        '{"answer": "84"}',
        '{"final_answer": 84}',
        '{"result": {"value": 84}}',
        '{"answer": "{\\"tool\\": \\"calculator\\"}"}',
        "{the answer is 84}",
    ],
)
def test_json_only_final_answers_end_the_episode_without_spending_budget(reply):
    assert not parse_tool_calls(reply).attempted
    scenario = {"id": "q", "question": "What is 12*7?"}
    env = ToolUseEnvironment([scenario], [calculator_tool()], max_tool_calls=2)
    trajectory = asyncio.run(env.run_episode(_Recorder([reply]), scenario))
    state = trajectory.final_state
    assert state["termination"] == "final_answer"
    assert state["tool_calls_used"] == 0 and state["tool_log"] == []
    assert state["final_answer"] == reply
    assert [t.role for t in trajectory.turns].count("assistant") == 1


@pytest.mark.parametrize(
    "reply,valid",
    [
        ('{"tool": "calculator", "args": {"expression": "2"}}', True),
        ('{"tool": "calculator"}', False),
        ('{"name": "calculator", "arguments": {"expression": "2"}}', False),
        ('{"tool": "calculator" "args": {}}', False),
        ('{"tool": "a", "args": {}, "tool": "b"}', False),
    ],
)
def test_json_objects_that_name_a_tool_are_still_calls(reply, valid):
    parsed = parse_tool_calls(reply)
    assert parsed.attempted and parsed.text == ""
    assert bool(parsed.calls) is valid and bool(parsed.errors) is not valid
