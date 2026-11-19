"""Planning commands and callbacks cannot replay or overwrite retired state."""

import asyncio
import copy
import pickle
import threading

import pytest

from stateset_agents.core.agent import AgentConfig
from stateset_agents.core.agent import MultiTurnAgent as ModelAgent
from stateset_agents.core.conversation import ConversationResetError
from stateset_agents.core.multiturn_agent import (
    ConversationClosedError,
    ConversationConflict,
    MultiTurnAgent,
)
from stateset_agents.experimental.long_term_planning import (
    HeuristicPlanner,
    PlanningConfig,
    PlanningManager,
    PlanStatus,
)


def manager():
    return PlanningManager(PlanningConfig(enabled=True))


def build(planning, identifier="test", **context):
    return planning.build_plan_message(
        [{"role": "user", "content": "Draft report, review it, then send it"}],
        {"conversation_id": identifier, **context},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "cancel"])
async def test_failed_turn_consumes_advance_without_automatically_advancing_again(
    failure,
):
    planning = manager()
    agent = MultiTurnAgent(
        {}, planning_manager=planning, response_backend=lambda prompt: "ok"
    )
    context = await agent.start_conversation()
    await agent.continue_conversation(
        context.conversation_id, "Draft report, review it, then send it"
    )

    async def fail(prompt):
        if failure == "cancel":
            raise asyncio.CancelledError
        raise LookupError("generation failed")

    agent.response_backend = fail
    with pytest.raises((LookupError, asyncio.CancelledError)):
        await agent.continue_conversation(
            context.conversation_id,
            "done",
            context_update={"plan_update": {"action": "advance"}},
        )
    assert "plan_update" not in context.metadata
    assert planning.get_plan(context.conversation_id).progress() == pytest.approx(1 / 3)
    agent.response_backend = lambda prompt: "ok"
    await agent.continue_conversation(context.conversation_id, "what is next?")
    assert planning.get_plan(context.conversation_id).progress() == pytest.approx(1 / 3)


def test_failed_revision_does_not_publish_partial_plan_or_counters():
    planning = manager()
    build(planning)
    before = copy.deepcopy(planning.get_plan("test"))
    count = planning.get_status("test")["turn_count"]

    class Fails(HeuristicPlanner):
        def revise_plan(self, plan, goal, config):
            plan.steps[0].notes = "must not leak"
            raise OSError("planner response lost")

    planning.planner = Fails()
    with pytest.raises(OSError):
        build(planning, plan_goal="Different goal", plan_update={"action": "advance"})
    assert planning.get_plan("test") == before
    assert planning.get_status("test")["turn_count"] == count
    receipt = planning.get_status("test")["last_finished"]
    assert receipt["state"] == "failed"
    assert receipt["callback_status"] == "unconfirmed"


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["clear", "update", "direct"])
async def test_late_planner_cannot_overwrite_current_state(mutation):
    planning = manager()
    build(planning)
    entered, release = threading.Event(), threading.Event()

    class Blocked(HeuristicPlanner):
        def revise_plan(self, plan, goal, config):
            entered.set()
            assert release.wait(5)
            return self.create_plan(goal, config)

    planning.planner = Blocked()
    task = asyncio.create_task(
        asyncio.to_thread(build, planning, plan_goal="Replacement plan")
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        with pytest.raises(ConversationConflict):
            build(planning)
        with pytest.raises(RuntimeError, match="Cannot copy"):
            copy.deepcopy(planning)
        assert build(planning, "unrelated") is not None
        if mutation == "clear":
            planning.clear_conversation("test")
        elif mutation == "update":
            planning.apply_update("test", {"action": "advance"})
        else:
            planning.get_plan("test").steps[0].notes = "external edit"
    finally:
        release.set()
    with pytest.raises(ConversationConflict):
        await task
    current = planning.get_plan("test")
    if mutation == "clear":
        assert current is None
    elif mutation == "update":
        assert current.steps[0].status is PlanStatus.DONE
    else:
        assert current.steps[0].notes == "external edit"
    assert planning.get_status("test")["active"] is None


@pytest.mark.asyncio
async def test_end_during_planner_does_not_resurrect_plan_or_commit_reply():
    planning = manager()
    entered, release = threading.Event(), threading.Event()

    class Blocked(HeuristicPlanner):
        def create_plan(self, goal, config):
            entered.set()
            assert release.wait(5)
            return super().create_plan(goal, config)

    planning.planner = Blocked()
    calls = []
    agent = MultiTurnAgent(
        {},
        planning_manager=planning,
        response_backend=lambda prompt: calls.append(prompt) or "late",
    )
    context = await agent.start_conversation()

    def turn():
        return asyncio.run(
            agent.continue_conversation(context.conversation_id, "Draft a report")
        )

    task = asyncio.create_task(asyncio.to_thread(turn))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert agent.end_conversation(context.conversation_id) is context
    finally:
        release.set()
    with pytest.raises(ConversationClosedError):
        await task
    assert planning.get_plan(context.conversation_id) is None
    assert calls == []
    assert len(context.history) == 1


def test_inspection_is_detached_and_idle_copy_restores_locks():
    planning = manager()
    build(planning)
    status = planning.get_status("test")
    status["last_finished"]["state"] = "mutated"
    assert planning.get_status("test")["last_finished"]["state"] == "completed"
    for clone in (copy.deepcopy(planning), pickle.loads(pickle.dumps(planning))):
        clone.apply_update("test", {"action": "advance"})
        assert clone.get_plan("test").progress() > 0
    assert planning.get_plan("test").progress() == 0


def test_retired_extractor_cannot_start_a_planner_callback():
    planning = manager()
    calls = []

    def extract(messages, context):
        planning.clear_conversation("test")
        return "A newly extracted goal"

    class Recorder(HeuristicPlanner):
        def create_plan(self, goal, config):
            calls.append(goal)
            return super().create_plan(goal, config)

    planning.goal_extractor = extract
    planning.planner = Recorder()
    with pytest.raises(ConversationConflict):
        build(planning)
    assert calls == []
    assert planning.get_plan("test") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["named", "response", "stream"])
@pytest.mark.parametrize("existing_plan", [False, True])
async def test_agent_reset_fences_plan_publication_without_erasing_existing_plan(
    mode, existing_plan
):
    planning = manager()
    if mode == "named":
        agent = MultiTurnAgent(
            {}, planning_manager=planning, response_backend=lambda prompt: "late"
        )
        context = await agent.start_conversation()
        identifier = context.conversation_id
    else:
        agent = ModelAgent(
            AgentConfig(model_name="stub://test"), planning_manager=planning
        )
        await agent.initialize()
        identifier = "test"
    if existing_plan:
        build(planning, identifier)
    original = copy.deepcopy(planning.get_plan(identifier))
    entered, release = threading.Event(), threading.Event()

    class Blocked(HeuristicPlanner):
        def create_plan(self, goal, config):
            entered.set()
            assert release.wait(5)
            return super().create_plan(goal, config)

        def revise_plan(self, plan, goal, config):
            return self.create_plan(goal, config)

    planning.planner = Blocked()

    async def generate():
        metadata = {
            "conversation_id": identifier,
            "plan_goal": "Prepare a different report",
        }
        if mode == "named":
            return await agent.continue_conversation(
                identifier, "new goal", context_update=metadata
            )
        messages = [{"role": "user", "content": "new goal"}]
        if mode == "response":
            return await agent.generate_response(messages, context=metadata)
        return [
            chunk
            async for chunk in agent.generate_response_stream(
                messages, context=metadata
            )
        ]

    task = asyncio.create_task(asyncio.to_thread(lambda: asyncio.run(generate())))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        await agent.reset()
    finally:
        release.set()
    with pytest.raises(ConversationResetError):
        await task
    assert planning.get_plan(identifier) == original
    assert planning.get_status(identifier)["active"] is None
    assert (
        planning.get_status(identifier)["last_finished"]["error_type"]
        == "ConversationResetError"
    )


@pytest.mark.asyncio
async def test_context_change_rejects_late_plan_as_well_as_late_response():
    planning = manager()
    agent = MultiTurnAgent(
        {}, planning_manager=planning, response_backend=lambda prompt: "late"
    )
    context = await agent.start_conversation()
    entered, release = threading.Event(), threading.Event()

    class Blocked(HeuristicPlanner):
        def create_plan(self, goal, config):
            entered.set()
            assert release.wait(5)
            return super().create_plan(goal, config)

    planning.planner = Blocked()
    task = asyncio.create_task(
        asyncio.to_thread(
            lambda: asyncio.run(
                agent.continue_conversation(context.conversation_id, "Draft a report")
            )
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        agent._apply_context_update(context, {"topic": "external update"})
    finally:
        release.set()
    with pytest.raises(ConversationConflict, match="changed during planning"):
        await task
    assert planning.get_plan(context.conversation_id) is None
    assert context.topic == "external update"
    assert len(context.history) == 1
