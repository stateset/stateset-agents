"""Durable hosted conversations preserve authorization and uncertain outcomes."""

import asyncio
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, Request
from httpx import ASGITransport, AsyncClient

from stateset_agents.api.auth import get_request_context
from stateset_agents.api.conversation_store import (
    ConversationStore,
    ConversationStoreUnavailable,
)
from stateset_agents.api.dependencies import get_inference_service
from stateset_agents.api.routers import v1


def request(app):
    return Request({"type": "http", "app": app, "state": {"request_id": "test"}})


def context(user="owner"):
    return SimpleNamespace(user=SimpleNamespace(user_id=user))


class Backend:
    config = SimpleNamespace(default_model="stub://test")
    is_stub = True

    def __init__(self, callback=None):
        self.callback = callback
        self.calls = []

    async def create_openai_response(self, payload):
        self.calls.append(payload)
        result = await self.callback() if self.callback else "answer"
        return {
            "choices": [{"message": {"content": result}}],
            "usage": {"total_tokens": 3},
        }


def application(path, **kwargs):
    app = FastAPI()
    app.state.v1_conversation_store = ConversationStore(path, **kwargs)
    return app


async def chat(app, backend, identifier="test", user="owner", message="question"):
    return await v1.chat_v1(
        request(app),
        v1.ConversationRequestV1(message=message, conversation_id=identifier),
        backend,
        context(user),
    )


async def status(app, identifier="test", user="owner"):
    return await v1.get_conversation_status_v1(request(app), identifier, context(user))


@pytest.mark.asyncio
async def test_restart_preserves_history_owner_and_receipts(tmp_path):
    path = tmp_path / "conversations.db"
    first = application(path)
    await chat(first, Backend())
    before = await status(first)
    assert before["last_finished"]["response_recorded"] is True
    await first.state.v1_conversation_store.aclose()
    second = application(path)
    try:
        assert await status(second) == before
        with pytest.raises(HTTPException) as denied:
            await status(second, user="other")
        assert denied.value.status_code == 404
        backend = Backend()
        await chat(second, backend, message="follow up")
        assert [message.content for message in backend.calls[0].messages] == [
            "question",
            "answer",
            "follow up",
        ]
        history = await v1.get_conversation_v1(request(second), "test", context())
        history["messages"][0]["content"] = "edited"
        assert (await v1.get_conversation_v1(request(second), "test", context()))[
            "messages"
        ][0]["content"] == "question"
    finally:
        await second.state.v1_conversation_store.aclose()


@pytest.mark.asyncio
async def test_two_workers_share_request_ownership_without_blocking_other_ids(tmp_path):
    path = tmp_path / "conversations.db"
    first, second = application(path), application(path)
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked():
        entered.set()
        await release.wait()
        return "answer"

    active = asyncio.create_task(chat(first, Backend(blocked)))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        backend = Backend()
        with pytest.raises(HTTPException) as conflict:
            await chat(second, backend)
        assert conflict.value.status_code == 409
        assert not backend.calls
        assert (await status(second))["active"]["state"] == "running"
        with pytest.raises(HTTPException) as denied:
            await chat(second, backend, user="other")
        assert denied.value.status_code == 404
        assert (
            await chat(second, backend, identifier="independent")
        ).response == "answer"
    finally:
        release.set()
        await active
        await first.state.v1_conversation_store.aclose()
        await second.state.v1_conversation_store.aclose()


@pytest.mark.asyncio
async def test_durable_delete_fences_late_reply_and_survives_restart(tmp_path):
    path = tmp_path / "conversations.db"
    first, second = application(path), application(path)
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked():
        entered.set()
        await release.wait()
        return "late reply"

    active = asyncio.create_task(chat(first, Backend(blocked)))
    await asyncio.wait_for(entered.wait(), 2)
    await v1.end_conversation_v1(request(second), "test", context())
    release.set()
    with pytest.raises(HTTPException) as conflict:
        await active
    assert conflict.value.status_code == 409
    await first.state.v1_conversation_store.aclose()
    await second.state.v1_conversation_store.aclose()
    third = application(path)
    try:
        with pytest.raises(HTTPException) as missing:
            await status(third)
        assert missing.value.status_code == 404
        assert (await chat(third, Backend(), user="new-owner")).response == "answer"
    finally:
        await third.state.v1_conversation_store.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary", ["admitted", "dispatched", "returned", "published"]
)
async def test_process_exit_recovers_without_replaying_generation(tmp_path, boundary):
    path = tmp_path / "crash.db"
    script = """
import asyncio
import os
import sys
from stateset_agents.api.conversation_store import ConversationStore
async def main():
    store = ConversationStore(sys.argv[1])
    with store.operation("test", "owner", lambda: True) as operation:
        if sys.argv[2] == "admitted":
            os._exit(0)
        async def provider():
            if sys.argv[2] == "dispatched":
                os._exit(0)
            return "answer"
        await operation.run(provider)
        if sys.argv[2] == "returned":
            os._exit(0)
        operation.publish({"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}, response=True)
        os._exit(0)
asyncio.run(main())
"""
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", script, str(path), boundary],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    app = application(path)
    try:
        observed = await status(app)
        receipt = observed["last_finished"]
        assert observed["active"] is None
        assert receipt["state"] == "interrupted"
        assert observed["recovery_required"] is (boundary in {"dispatched", "returned"})
        assert receipt["response_recorded"] is (boundary == "published")
        backend = Backend()
        if boundary in {"dispatched", "returned"}:
            with pytest.raises(HTTPException) as conflict:
                await chat(app, backend)
            assert conflict.value.status_code == 409
            assert not backend.calls
            for user, operation_id, expected in [
                ("other", receipt["id"], 404),
                ("owner", "stale", 409),
            ]:
                with pytest.raises(HTTPException) as rejected:
                    await v1.acknowledge_conversation_v1(
                        request(app),
                        "test",
                        v1.ConversationAcknowledgementV1(operation_id=operation_id),
                        context(user),
                    )
                assert rejected.value.status_code == expected
            acknowledged = await v1.acknowledge_conversation_v1(
                request(app),
                "test",
                v1.ConversationAcknowledgementV1(operation_id=receipt["id"]),
                context(),
            )
            assert acknowledged["recovery_required"] is False
            assert not backend.calls
        await chat(app, backend, message="continue")
        assert len(backend.calls) == 1
    finally:
        await app.state.v1_conversation_store.aclose()


@pytest.mark.asyncio
async def test_live_process_is_not_misclassified_as_interrupted(tmp_path):
    path = tmp_path / "live.db"
    script = """
import asyncio
import sys
from stateset_agents.api.conversation_store import ConversationStore
async def main():
    store = ConversationStore(sys.argv[1])
    with store.operation("test", "owner", lambda: True) as operation:
        async def provider():
            print("dispatched", flush=True)
            await asyncio.Event().wait()
        await operation.run(provider)
asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        str(path),
        cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert await asyncio.wait_for(process.stdout.readline(), 10) == b"dispatched\n"
        app = application(path)
        try:
            observed = await status(app)
            assert observed["active"]["generation_status"] == "unconfirmed"
            assert observed["last_finished"] is None
            with pytest.raises(HTTPException) as conflict:
                await chat(app, Backend())
            assert conflict.value.status_code == 409
            process.terminate()
            await asyncio.wait_for(process.wait(), 10)
            recovered = await status(app)
            assert recovered["last_finished"]["state"] == "interrupted"
            assert recovered["recovery_required"] is True
        finally:
            await app.state.v1_conversation_store.aclose()
    finally:
        if process.returncode is None:
            process.kill()
        await process.communicate()


@pytest.mark.asyncio
async def test_publication_failure_stops_before_provider_and_never_falls_back(
    tmp_path, monkeypatch
):
    app = application(tmp_path / "failed.db")
    store = app.state.v1_conversation_store
    original = store._write

    def fail(*args, **kwargs):
        raise ConversationStoreUnavailable("disk failure")

    monkeypatch.setattr(store, "_write", fail)
    backend = Backend()
    with pytest.raises(HTTPException) as error:
        await chat(app, backend)
    assert error.value.status_code == 503
    assert not backend.calls
    monkeypatch.setattr(store, "_write", original)
    await chat(app, backend)
    assert len(backend.calls) == 1
    await store.aclose()


@pytest.mark.asyncio
async def test_shutdown_drains_cancelled_callback_before_releasing_ownership(tmp_path):
    app = application(tmp_path / "shutdown.db")
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def blocked():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            return "late"

    request_task = asyncio.create_task(chat(app, Backend(blocked)))
    await asyncio.wait_for(entered.wait(), 2)
    closing = asyncio.create_task(app.state.v1_conversation_store.aclose())
    try:
        await asyncio.wait_for(cleaning.wait(), 2)
        for _ in range(3):
            closing.cancel()
            await asyncio.sleep(0)
            assert not closing.done()
        assert (await status(app))["active"]["state"] == "running"
        assert (await status(app))["active"]["cancellation_requested"] is True
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await request_task
    with pytest.raises(asyncio.CancelledError):
        await closing
    await app.state.v1_conversation_store.aclose()
    restarted = application(tmp_path / "shutdown.db")
    try:
        receipt = (await status(restarted))["last_finished"]
        assert receipt["state"] == "cancelled"
        assert receipt["generation_status"] == "returned"
        assert receipt["response_recorded"] is False
    finally:
        await restarted.state.v1_conversation_store.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("retire", ["expiry", "capacity"])
async def test_durable_retention_prevents_late_publication(tmp_path, retire):
    path = tmp_path / "retention.db"
    app = application(path, max_items=1)
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked():
        entered.set()
        await release.wait()
        return "late"

    task = asyncio.create_task(chat(app, Backend(blocked)))
    await asyncio.wait_for(entered.wait(), 2)
    if retire == "expiry":
        with sqlite3.connect(path) as connection:
            connection.execute(
                "UPDATE gateway_conversations SET accessed = 0 WHERE id = 'test'"
            )
    else:
        await chat(app, Backend(), identifier="other")
    release.set()
    with pytest.raises(HTTPException) as stale:
        await task
    assert stale.value.status_code == 409
    with pytest.raises(HTTPException) as missing:
        await status(app)
    assert missing.value.status_code == 404
    await app.state.v1_conversation_store.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt", ["version", "principal", "receipt"])
async def test_unsupported_or_corrupt_records_fail_closed(tmp_path, corrupt):
    path = tmp_path / "schema.db"
    app = application(path)
    await chat(app, Backend())
    with sqlite3.connect(path) as connection:
        data = json.loads(
            connection.execute(
                "SELECT data FROM gateway_conversations WHERE id = 'test'"
            ).fetchone()[0]
        )
        if corrupt == "version":
            data["version"] = 99
        elif corrupt == "principal":
            data["record"]["user_id"] = "other"
        else:
            data["record"]["_lifecycle"]["last_finished"]["response_recorded"] = "yes"
        connection.execute(
            "UPDATE gateway_conversations SET data = ?", (json.dumps(data),)
        )
    backend = Backend()
    with pytest.raises(HTTPException) as error:
        await chat(app, backend)
    assert error.value.status_code == 503
    assert not backend.calls
    with sqlite3.connect(path) as connection:
        assert (
            json.loads(
                connection.execute("SELECT data FROM gateway_conversations").fetchone()[
                    0
                ]
            )
            == data
        )
    await app.state.v1_conversation_store.aclose()


@pytest.mark.asyncio
async def test_http_routes_recover_and_preserve_authorization(tmp_path):
    app = application(tmp_path / "http.db")
    app.include_router(v1.router)
    backend = Backend()
    app.dependency_overrides[get_inference_service] = lambda: backend

    async def principal(request: Request):
        return context(request.headers.get("x-user", "owner"))

    app.dependency_overrides[get_request_context] = principal
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/conversations",
            json={"conversation_id": "test", "message": "hello"},
        )
        assert response.status_code == 200
        assert (await client.get("/api/v1/conversations/test/status")).json()[
            "last_finished"
        ]["state"] == "completed"
        for method, suffix, payload in [
            ("get", "/status", None),
            ("get", "", None),
            ("delete", "", None),
            ("post", "/acknowledge", {"operation_id": "unknown"}),
        ]:
            result = await client.request(
                method,
                "/api/v1/conversations/test" + suffix,
                headers={"x-user": "other"},
                json=payload,
            )
            assert result.status_code == 404
        conflict = await client.post(
            "/api/v1/conversations/test/acknowledge", json={"operation_id": "unknown"}
        )
        assert conflict.status_code == 409
    await app.state.v1_conversation_store.aclose()


@pytest.mark.asyncio
async def test_callback_cannot_close_its_own_store(tmp_path):
    from stateset_agents.core.multiturn_context import ConversationConflict

    app = application(tmp_path / "self-close.db")

    async def respond():
        with pytest.raises(ConversationConflict, match="own store"):
            await app.state.v1_conversation_store.aclose()
        return "answer"

    assert (await asyncio.wait_for(chat(app, Backend(respond)), 2)).response == "answer"
    await app.state.v1_conversation_store.aclose()


@pytest.mark.asyncio
async def test_cancellation_publication_failure_still_drains_callback(
    tmp_path, monkeypatch
):
    app = application(tmp_path / "cancel-write.db")
    store = app.state.v1_conversation_store
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def blocked():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.set()

    task = asyncio.create_task(chat(app, Backend(blocked)))
    await asyncio.wait_for(entered.wait(), 2)
    original = store._write

    def fail(*args, **kwargs):
        raise ConversationStoreUnavailable("disk unavailable")

    monkeypatch.setattr(store, "_write", fail)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()
    monkeypatch.setattr(store, "_write", original)
    observed = await status(app)
    assert observed["last_finished"]["state"] == "interrupted"
    assert observed["recovery_required"] is True
    await store.aclose()


@pytest.mark.asyncio
async def test_recorded_answer_survives_lost_final_acknowledgement(
    tmp_path, monkeypatch
):
    app = application(tmp_path / "answer.db")
    store = app.state.v1_conversation_store
    original = store._write

    def fail_final(connection, identifier, principal, revision, record):
        if record["_lifecycle"]["active"] is None:
            raise ConversationStoreUnavailable("final acknowledgement unavailable")
        return original(connection, identifier, principal, revision, record)

    monkeypatch.setattr(store, "_write", fail_final)
    backend = Backend()
    with pytest.raises(HTTPException) as failed:
        await chat(app, backend)
    assert failed.value.status_code == 503
    assert len(backend.calls) == 1
    monkeypatch.setattr(store, "_write", original)
    observed = await status(app)
    assert observed["last_finished"]["state"] == "interrupted"
    assert observed["last_finished"]["response_recorded"] is True
    assert observed["recovery_required"] is False
    history = await v1.get_conversation_v1(request(app), "test", context())
    assert [item["content"] for item in history["messages"]] == ["question", "answer"]
    await store.aclose()


def install_lifespan_dependencies(monkeypatch, path):
    from stateset_agents.api import distributed_cache, main, persistence
    from stateset_agents.api.config import APIConfig

    config = APIConfig(v1_conversation_db_path=str(path))
    monkeypatch.setattr(main, "get_config", lambda: config)
    monkeypatch.delenv("STATESET_DEFAULT_CHECKPOINT", raising=False)

    async def noop(*args, **kwargs):
        pass

    async def acquire(*args):
        return SimpleNamespace(aclose=noop)

    monkeypatch.setattr(distributed_cache, "acquire_cache", acquire)
    monkeypatch.setattr(persistence, "acquire_database", acquire)
    app = FastAPI()
    app.state.config = config
    app.state.agent_service = object()
    app.state.training_service = SimpleNamespace(aclose=noop)
    backend = Backend()
    backend.check_health = noop
    backend.aclose = noop
    app.state.inference_service = backend
    return app, backend


@pytest.mark.asyncio
async def test_configured_lifespan_restores_history_and_closes_journal_first(
    tmp_path, monkeypatch
):
    from stateset_agents.api import main

    path = tmp_path / "lifespan.db"
    first, backend = install_lifespan_dependencies(monkeypatch, path)
    closed = []

    async def close_backend():
        assert first.state.v1_conversation_store._closed
        closed.append("inference")

    backend.aclose = close_backend
    async with main.lifespan(first):
        assert first.state.v1_conversation_store.path == path
        await chat(first, backend)
    assert closed == ["inference"]
    second, other_backend = install_lifespan_dependencies(monkeypatch, path)
    async with main.lifespan(second):
        assert (await status(second))["last_finished"]["response_recorded"] is True
        await chat(second, other_backend, message="after restart")
        assert [message.content for message in other_backend.calls[0].messages] == [
            "question",
            "answer",
            "after restart",
        ]


@pytest.mark.asyncio
async def test_explicit_journal_failure_aborts_development_startup(
    tmp_path, monkeypatch
):
    from stateset_agents.api import main

    path = tmp_path / "unsupported.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE gateway_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO gateway_metadata VALUES ('schema_version', '99')"
        )
    app, backend = install_lifespan_dependencies(monkeypatch, path)
    closed = []

    async def close_backend():
        closed.append("inference")

    backend.aclose = close_backend
    with pytest.raises(ConversationStoreUnavailable, match="Unsupported"):
        async with main.lifespan(app):
            pytest.fail("Durable startup silently fell back to memory")
    assert not hasattr(app.state, "v1_conversation_store")
    assert closed == ["inference"]
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT value FROM gateway_metadata").fetchone() == (
            "99",
        )


def test_journal_path_is_explicit_configuration(monkeypatch, tmp_path):
    from stateset_agents.api.config import APIConfig

    monkeypatch.delenv("API_V1_CONVERSATION_DB", raising=False)
    assert APIConfig.from_env().v1_conversation_db_path is None
    path = str(tmp_path / "durable.db")
    monkeypatch.setenv("API_V1_CONVERSATION_DB", path)
    assert APIConfig.from_env().v1_conversation_db_path == path


@pytest.mark.asyncio
async def test_lifespan_drains_conversation_before_closing_inference(
    tmp_path, monkeypatch
):
    from stateset_agents.api import main

    app, backend = install_lifespan_dependencies(monkeypatch, tmp_path / "drain.db")
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events = []

    async def blocked():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            events.append("callback cleaned")
            return "late"

    async def close_backend():
        events.append("inference closed")

    backend.callback = blocked
    backend.aclose = close_backend
    lifespan = main.lifespan(app)
    await lifespan.__aenter__()
    active = asyncio.create_task(chat(app, backend))
    await asyncio.wait_for(entered.wait(), 2)
    closing = asyncio.create_task(lifespan.__aexit__(None, None, None))
    try:
        await asyncio.wait_for(cleaning.wait(), 2)
        closing.cancel()
        await asyncio.sleep(0)
        assert not events and not closing.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await active
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert events == ["callback cleaned", "inference closed"]


@pytest.mark.asyncio
async def test_finished_operation_cannot_checkpoint_over_a_reused_identifier(tmp_path):
    from stateset_agents.core.conversation import ConversationResetError

    store = ConversationStore(tmp_path / "reuse.db")

    async def generate():
        return "answer"

    with store.operation("test", "owner", lambda: True) as old:
        await old.run(generate)
        old.publish(
            {"messages": [{"role": "assistant", "content": "old"}]}, response=True
        )
    store.delete("test", "owner")
    with store.operation("test", "owner", lambda: True) as new:
        await new.run(generate)
        new.publish(
            {"messages": [{"role": "assistant", "content": "new"}]}, response=True
        )
    with pytest.raises(ConversationResetError):
        old.checkpoint()
    assert store.read("test", "owner")["messages"] == [
        {"role": "assistant", "content": "new"}
    ]
    await store.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["dispatch", "answer"])
async def test_lost_commit_acknowledgement_is_reconciled_without_replay(
    tmp_path, monkeypatch, boundary
):
    from contextlib import contextmanager

    app = application(tmp_path / "lost-ack.db")
    store = app.state.v1_conversation_store
    original = store._transaction
    failed = False

    @contextmanager
    def transaction(*, write=True):
        nonlocal failed
        fail_after_commit = False
        with original(write=write) as connection:
            yield connection
            if write and not failed:
                row = connection.execute(
                    "SELECT data FROM gateway_conversations WHERE id = 'test'"
                ).fetchone()
                receipt = (
                    json.loads(row[0])["record"]["_lifecycle"]["active"]
                    if row
                    else None
                )
                if receipt:
                    fail_after_commit = (
                        receipt["generation_status"] == "unconfirmed"
                        if boundary == "dispatch"
                        else receipt["response_recorded"]
                    )
        if fail_after_commit:
            failed = True
            raise ConversationStoreUnavailable("commit acknowledgement lost")

    monkeypatch.setattr(store, "_transaction", transaction)
    backend = Backend()
    with pytest.raises(HTTPException) as error:
        await chat(app, backend)
    assert failed and error.value.status_code == 503
    assert len(backend.calls) == (0 if boundary == "dispatch" else 1)
    observed = await status(app)
    assert observed["last_finished"]["state"] == "interrupted"
    assert observed["last_finished"]["response_recorded"] is (boundary == "answer")
    assert observed["recovery_required"] is (boundary == "dispatch")
    await store.aclose()


@pytest.mark.asyncio
async def test_lifespan_reentry_replaces_closed_journal_owner(tmp_path, monkeypatch):
    from stateset_agents.api import main

    app, backend = install_lifespan_dependencies(monkeypatch, tmp_path / "reentry.db")
    async with main.lifespan(app):
        await chat(app, backend)
        old = app.state.v1_conversation_store
        old_health = app.state.health_checker
    assert old.closed

    async def noop(*args):
        pass

    def inference_factory():
        fresh = Backend()
        fresh.check_health = noop
        fresh.aclose = noop
        return fresh

    # Custom owners opt into replacement; the built-in health checker and
    # journal are recreated automatically after their confirmed cleanup.
    app.state.lifecycle_factories = {
        "inference_service": inference_factory,
        "training_service": lambda: SimpleNamespace(aclose=noop),
    }
    async with main.lifespan(app):
        assert app.state.health_checker is not old_health
        assert app.state.v1_conversation_store is not old
        assert not app.state.v1_conversation_store.closed
        assert (await status(app))["last_finished"]["response_recorded"] is True
