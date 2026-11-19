"""Canonical gateway conversations own requests through retirement and cleanup."""

import asyncio
import copy
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, Request
from httpx import ASGITransport, AsyncClient

from stateset_agents.api.routers import agents as agent_routes
from stateset_agents.api.routers import v1
from stateset_agents.api.schemas import ConversationRequest
from stateset_agents.api.services.agent_service import AgentService
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.conversation import ConversationResetError, conversation_call
from stateset_agents.experimental.long_term_planning import (
    HeuristicPlanner,
    PlanningConfig,
    PlanningManager,
)
from stateset_agents.utils.security import SecurityMonitor


def request_for(app):
    return Request({"type": "http", "app": app, "state": {"request_id": "test"}})


def identity(user="owner"):
    return SimpleNamespace(user=SimpleNamespace(user_id=user))


class Backend:
    config = SimpleNamespace(default_model="stub://test")
    is_stub = True

    def __init__(self, callback):
        self.callback = callback
        self.calls = []

    async def create_openai_response(self, request):
        self.calls.append(request)
        response = await self.callback()
        return {
            "choices": [{"message": {"content": response}}],
            "usage": {"total_tokens": 3},
        }


async def chat(app, backend, identifier="test", user="owner", message="question"):
    return await v1.chat_v1(
        request_for(app),
        v1.ConversationRequestV1(message=message, conversation_id=identifier),
        backend,
        identity(user),
    )


@pytest.mark.asyncio
async def test_v1_overlap_rejected_before_provider_and_status_is_detached():
    app = FastAPI()
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked():
        entered.set()
        await release.wait()
        return "answer"

    backend = Backend(blocked)
    task = asyncio.create_task(chat(app, backend))
    try:
        await entered.wait()
        with pytest.raises(HTTPException) as error:
            await chat(app, backend)
        assert error.value.status_code == 409
        assert len(backend.calls) == 1
        status = await v1.get_conversation_status_v1(
            request_for(app), "test", identity()
        )
        assert status["active"]["generation_status"] == "unconfirmed"
        assert status["active"]["user_recorded"] is False
        status["active"]["state"] = "mutated"
        assert (
            app.state.v1_conversations["test"]["_lifecycle"]["active"]["state"]
            == "running"
        )
        with pytest.raises(HTTPException) as denied:
            await v1.get_conversation_status_v1(
                request_for(app), "test", identity("other")
            )
        assert denied.value.status_code == 404
    finally:
        release.set()
    assert (await task).response == "answer"
    assert [
        turn["role"] for turn in app.state.v1_conversations["test"]["messages"]
    ] == ["user", "assistant"]


@pytest.mark.asyncio
@pytest.mark.parametrize("retire", ["delete", "expire", "evict", "replace"])
async def test_v1_retired_record_cannot_receive_reply_or_refresh_access(
    retire, monkeypatch
):
    app = FastAPI()
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked():
        entered.set()
        await release.wait()
        return "late answer"

    task = asyncio.create_task(chat(app, Backend(blocked)))
    await entered.wait()
    old = app.state.v1_conversations["test"]
    if retire == "delete":
        await v1.end_conversation_v1(request_for(app), "test", identity())
    elif retire == "expire":
        app.state.v1_conversations_access["test"] = (
            time.time() - v1.V1_CONVERSATION_TTL_SECONDS - 10
        )
        with pytest.raises(HTTPException):
            await v1.get_conversation_v1(request_for(app), "test", identity())
    elif retire == "evict":
        app.state.v1_conversations["new"] = {"messages": [], "user_id": "owner"}
        app.state.v1_conversations_access["new"] = time.time() + 1
        monkeypatch.setattr(v1, "V1_MAX_CONVERSATIONS", 1)
        await v1.get_conversation_v1(request_for(app), "new", identity())
    else:
        app.state.v1_conversations["test"] = {"messages": [], "user_id": "new-owner"}
    before_access = copy.deepcopy(app.state.v1_conversations_access)
    release.set()
    with pytest.raises(HTTPException) as error:
        await task
    assert error.value.status_code == 409
    assert old["messages"] == []
    assert old["_lifecycle"]["last_finished"]["generation_status"] == "returned"
    assert old["_lifecycle"]["last_finished"]["response_recorded"] is False
    assert app.state.v1_conversations_access == before_access


@pytest.mark.asyncio
async def test_v1_cancelled_request_keeps_ownership_until_provider_cleanup_finishes():
    app = FastAPI()
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def blocked():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            return "ignored cancellation"

    backend = Backend(blocked)
    task = asyncio.create_task(chat(app, backend))
    try:
        await entered.wait()
        task.cancel()
        await cleaning.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        with pytest.raises(HTTPException) as conflict:
            await chat(app, backend)
        assert conflict.value.status_code == 409
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    record = app.state.v1_conversations["test"]
    assert record["messages"] == []
    receipt = record["_lifecycle"]["last_finished"]
    assert receipt["state"] == "cancelled"
    assert receipt["generation_status"] == "returned"
    assert receipt["cancellation_requested"] is True


@pytest.mark.asyncio
async def test_v1_independent_conversations_progress_and_invalid_input_creates_nothing():
    app = FastAPI()
    entered, release = asyncio.Event(), asyncio.Event()
    count = 0

    async def backend_call():
        nonlocal count
        count += 1
        if count == 1:
            entered.set()
            await release.wait()
        return "answer"

    backend = Backend(backend_call)
    with pytest.raises(HTTPException) as invalid:
        await chat(app, backend, message="  ")
    assert invalid.value.status_code == 422
    assert not hasattr(app.state, "v1_conversations")
    task = asyncio.create_task(chat(app, backend, "first"))
    try:
        await entered.wait()
        assert (await chat(app, backend, "second")).response == "answer"
    finally:
        release.set()
        await task


class Agent:
    tokenizer = None
    planning_manager = None

    def __init__(self, callback):
        self.callback = callback
        self.calls = []

    async def generate_response(self, messages, context):
        self.calls.append((messages, context))
        return await self.callback()


def service_request(identifier="test", user="owner", **kwargs):
    return ConversationRequest(
        conversation_id=identifier,
        user_id=user,
        messages=[{"role": "user", "content": "question"}],
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("retire", ["conversation", "agent", "replace_agent"])
async def test_service_retirement_rejects_late_reply_but_keeps_accepted_user(retire):
    service = AgentService(SecurityMonitor())
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked():
        entered.set()
        await release.wait()
        return "late"

    service.agents["agent"] = Agent(blocked)
    task = asyncio.create_task(
        service.get_conversation_response("agent", service_request())
    )
    await entered.wait()
    record = service.conversations["agent"][0]
    if retire == "conversation":
        assert service.delete_conversation("test", user_id="owner")
    elif retire == "agent":
        assert service.delete_agent("agent")
    else:
        service.agents["agent"] = Agent(blocked)
    release.set()
    with pytest.raises(HTTPException) as error:
        await task
    assert error.value.status_code == 409
    assert [turn["role"] for turn in record["messages"]] == ["user"]
    assert record["total_tokens"] == 0
    assert record["_lifecycle"]["last_finished"]["user_recorded"] is True


@pytest.mark.asyncio
async def test_service_owns_inputs_and_blocks_overlap_without_reassigning_principal():
    service = AgentService(SecurityMonitor())
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked():
        entered.set()
        await release.wait()
        return "answer"

    agent = Agent(blocked)
    service.agents["agent"] = agent
    payload = service_request(
        context={
            "conversation_id": "victim",
            "user_id": "victim",
            "nested": ["initial"],
        }
    )
    task = asyncio.create_task(service.get_conversation_response("agent", payload))
    try:
        await entered.wait()
        messages, context = agent.calls[0]
        messages[0]["content"] = "provider mutation"
        context["nested"].append("provider mutation")
        assert context["conversation_id"] == "test"
        assert context["user_id"] == "owner"
        assert service.conversations["agent"][0]["messages"][0]["content"] == "question"
        with pytest.raises(HTTPException) as conflict:
            await service.get_conversation_response("agent", service_request())
        assert conflict.value.status_code == 409
        with pytest.raises(HTTPException) as denied:
            service.get_conversation_status("test", "other")
        assert denied.value.status_code == 404
    finally:
        release.set()
    await task
    assert service.conversations["agent"][0]["messages"][0]["content"] == "question"
    assert payload.context["nested"] == ["initial"]
    assert service.conversations["agent"][0]["metadata"]["context"]["nested"] == [
        "initial"
    ]


@pytest.mark.asyncio
async def test_http_v1_conflict_and_status_routes_use_normal_access_rules():
    app = FastAPI()
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked():
        entered.set()
        await release.wait()
        return "answer"

    backend = Backend(blocked)
    app.include_router(v1.router)
    app.dependency_overrides[v1.get_inference_service] = lambda: backend

    async def user(request: Request):
        return identity(request.headers.get("x-user", "owner"))

    app.dependency_overrides[v1.get_request_context] = user
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        task = asyncio.create_task(
            client.post(
                "/api/v1/conversations",
                json={"conversation_id": "test", "message": "question"},
            )
        )
        try:
            await entered.wait()
            conflict = await client.post(
                "/api/v1/conversations",
                json={"conversation_id": "test", "message": "second"},
            )
            assert conflict.status_code == 409
            status = await client.get("/api/v1/conversations/test/status")
            assert status.status_code == 200
            assert status.json()["active"]["state"] == "running"
            assert (
                await client.get(
                    "/api/v1/conversations/test/status", headers={"x-user": "other"}
                )
            ).status_code == 404
        finally:
            release.set()
        assert (await task).status_code == 200


@pytest.mark.asyncio
async def test_service_guard_fences_real_agent_cleanup_after_cancellation():
    service = AgentService(SecurityMonitor())
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    effects = []

    class Managed(MultiTurnAgent):
        @conversation_call
        async def generate_response(self, messages, context=None):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleaning.set()
                await release.wait()
                self._conversation_owner().check()
                effects.append("late effect")
                return "late"

    agent = Managed(AgentConfig(model_name="stub://test"))
    service.agents["agent"] = agent
    task = asyncio.create_task(
        service.get_conversation_response("agent", service_request())
    )
    try:
        await entered.wait()
        task.cancel()
        await cleaning.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert effects == []
    assert len(service.conversations["agent"][0]["messages"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["v1", "service"])
async def test_failed_provider_retains_policy_and_truthful_outcome(surface):
    async def fail():
        raise OSError("private provider error text")

    if surface == "v1":
        app = FastAPI()
        with pytest.raises(HTTPException) as error:
            await chat(app, Backend(fail))
        record = app.state.v1_conversations["test"]
        assert record["messages"] == []
    else:
        service = AgentService(SecurityMonitor())
        service.agents["agent"] = Agent(fail)
        with pytest.raises(HTTPException) as error:
            await service.get_conversation_response("agent", service_request())
        record = service.conversations["agent"][0]
        assert len(record["messages"]) == 1
    assert error.value.status_code == 500
    receipt = record["_lifecycle"]["last_finished"]
    assert receipt["state"] == "failed"
    assert receipt["generation_status"] == "unconfirmed"
    assert receipt["work_error_type"] == "OSError"
    assert "private provider" not in repr(receipt)


@pytest.mark.asyncio
async def test_service_ending_conversation_fences_real_planner_publication():
    entered, release = threading.Event(), threading.Event()

    class Blocked(HeuristicPlanner):
        def create_plan(self, goal, config):
            entered.set()
            assert release.wait(5)
            return super().create_plan(goal, config)

    planning = PlanningManager(PlanningConfig(enabled=True), planner=Blocked())
    agent = MultiTurnAgent(
        AgentConfig(model_name="stub://test"), planning_manager=planning
    )
    await agent.initialize()
    service = AgentService(SecurityMonitor())
    service.agents["agent"] = agent
    payload = service_request(context={"plan_goal": "Draft a project report"})
    task = asyncio.create_task(
        asyncio.to_thread(
            lambda: asyncio.run(service.get_conversation_response("agent", payload))
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert service.delete_conversation("test", "owner")
    finally:
        release.set()
    with pytest.raises(HTTPException) as error:
        await task
    assert error.value.status_code == 409
    assert planning.get_plan("test") is None
    assert agent.get_history() == []


@pytest.mark.asyncio
async def test_service_external_history_edit_is_not_overwritten_by_late_reply():
    service = AgentService(SecurityMonitor())
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked():
        entered.set()
        await release.wait()
        return "late"

    service.agents["agent"] = Agent(blocked)
    task = asyncio.create_task(
        service.get_conversation_response("agent", service_request())
    )
    await entered.wait()
    record = service.conversations["agent"][0]
    record["messages"].append({"role": "system", "content": "external edit"})
    release.set()
    with pytest.raises(HTTPException) as error:
        await task
    assert error.value.status_code == 409
    assert [turn["role"] for turn in record["messages"]] == ["user", "system"]


@pytest.mark.asyncio
async def test_service_failed_agent_cleanup_keeps_handle_and_blocks_new_work():
    service = AgentService(SecurityMonitor())

    async def respond():
        return "answer"

    agent = Agent(respond)
    service.agents["agent"] = agent
    await service.get_conversation_response("agent", service_request())

    class FailingCleanup:
        def clear_conversation(self, identifier):
            raise OSError("cleanup failed")

    agent.planning_manager = FailingCleanup()
    with pytest.raises(OSError):
        service.delete_agent("agent")
    assert service.agents["agent"] is agent
    with pytest.raises(HTTPException) as error:
        await service.get_conversation_response("agent", service_request("new"))
    assert error.value.status_code == 409
    assert len(agent.calls) == 1


@pytest.mark.asyncio
async def test_v1_history_inspection_returns_detached_data():
    app = FastAPI()

    async def respond():
        return "answer"

    await chat(app, Backend(respond))
    result = await v1.get_conversation_v1(request_for(app), "test", identity())
    result["messages"][0]["content"] = "caller edit"
    assert app.state.v1_conversations["test"]["messages"][0]["content"] == "question"


@pytest.mark.asyncio
async def test_service_status_route_preserves_owner_and_admin_visibility():
    service = AgentService(SecurityMonitor())

    async def respond():
        return "answer"

    service.agents["agent"] = Agent(respond)
    await service.get_conversation_response("agent", service_request())
    app = FastAPI()
    app.include_router(agent_routes.conversation_router)
    app.dependency_overrides[agent_routes.get_agent_service] = lambda: service

    async def user(request: Request):
        return SimpleNamespace(
            user_id=request.headers.get("x-user", "owner"),
            roles=["admin"] if request.headers.get("x-admin") else ["user"],
        )

    app.dependency_overrides[agent_routes.get_current_user] = user
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        status = await client.get("/conversations/test/status")
        assert status.status_code == 200
        assert status.json()["last_finished"]["response_recorded"] is True
        assert (
            await client.get("/conversations/test/status", headers={"x-user": "other"})
        ).status_code == 404
        assert (
            await client.get(
                "/conversations/test/status",
                headers={"x-user": "other", "x-admin": "yes"},
            )
        ).status_code == 200


@pytest.mark.asyncio
async def test_escaped_old_request_cannot_retire_a_new_owner():
    service = AgentService(SecurityMonitor())
    escaped_release, entered, release = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    children = []
    calls = 0

    class Managed(MultiTurnAgent):
        @conversation_call
        async def generate_response(self, messages, context=None):
            nonlocal calls
            calls += 1
            if calls == 1:

                async def escaped():
                    await escaped_release.wait()
                    self._conversation_owner().check()

                children.append(asyncio.create_task(escaped()))
            else:
                entered.set()
                await release.wait()
            return "answer"

    service.agents["agent"] = Managed(AgentConfig(model_name="stub://test"))
    await service.get_conversation_response("agent", service_request())
    task = asyncio.create_task(
        service.get_conversation_response("agent", service_request())
    )
    try:
        await entered.wait()
        escaped_release.set()
        with pytest.raises(ConversationResetError):
            await children[0]
        status = service.get_conversation_status("test", "owner")
        assert status["state"] == "active"
        assert status["active"]["state"] == "running"
    finally:
        release.set()
    assert (await task).response == "answer"
    assert service.get_conversation_status("test", "owner")["state"] == "active"


@pytest.mark.asyncio
async def test_concurrent_deletion_cannot_clear_a_reused_conversation_id():
    service = AgentService(SecurityMonitor())
    entered, release = threading.Event(), threading.Event()
    clears = []

    async def respond():
        return "answer"

    agent = Agent(respond)
    service.agents["agent"] = agent
    await service.get_conversation_response("agent", service_request())

    def clear(identifier):
        clears.append(identifier)
        entered.set()
        if len(clears) == 1:
            assert release.wait(5)

    agent.planning_manager = SimpleNamespace(clear_conversation=clear)
    deleting = asyncio.create_task(
        asyncio.to_thread(service.delete_conversation, "test", "owner")
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        with pytest.raises(HTTPException) as error:
            service.delete_conversation("test", "owner")
        assert error.value.status_code == 409
        assert clears == ["test"]
        assert service.get_conversation_status("test", "owner")["state"] == "ended"
    finally:
        release.set()
        await deleting
    agent.planning_manager = None
    assert (
        await service.get_conversation_response("agent", service_request())
    ).response == "answer"


@pytest.mark.asyncio
async def test_failed_conversation_cleanup_can_be_retried_without_admitting_work():
    service = AgentService(SecurityMonitor())

    async def respond():
        return "answer"

    agent = Agent(respond)
    service.agents["agent"] = agent
    await service.get_conversation_response("agent", service_request())
    clears = []

    def clear(identifier):
        clears.append(identifier)
        if len(clears) == 1:
            raise RuntimeError("cleanup failed")

    agent.planning_manager = SimpleNamespace(clear_conversation=clear)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        service.delete_conversation("test", "owner")
    with pytest.raises(HTTPException) as error:
        await service.get_conversation_response("agent", service_request())
    assert error.value.status_code == 409
    assert service.delete_conversation("test", "owner") is True
    assert clears == ["test", "test"]
    assert service.conversations["agent"] == []
