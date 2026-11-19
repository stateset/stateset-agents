"""Persistence boundaries isolate records and commit related writes together."""

import asyncio
import copy
from datetime import datetime

import pytest

from stateset_agents.api.persistence import (
    Agent,
    Conversation,
    DatabaseBackend,
    DatabaseConfig,
    UnitOfWork,
)


@pytest.fixture(params=[DatabaseBackend.MEMORY, DatabaseBackend.SQLITE])
async def database(request, tmp_path):
    database = UnitOfWork(
        DatabaseConfig(backend=request.param, connection_url=str(tmp_path / "data.db"))
    )
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_connect_preserves_repository_identity_and_data(database):
    repository = database.agents
    await repository.create(Agent(id="existing"))
    await database.connect()
    assert database.agents is repository
    assert await database.agents.get("existing") is not None


@pytest.mark.asyncio
async def test_inputs_and_reads_are_detached_from_persisted_records(database):
    agent = Agent(id="agent", config={"nested": ["original"]})
    created = await database.agents.create(agent)
    agent.config["nested"].append("input edit")
    created.config["nested"].append("result edit")
    fetched = await database.agents.get(agent.id)
    assert fetched.config == {"nested": ["original"]}
    fetched.config["nested"].append("read edit")
    listed = await database.agents.list()
    listed[0].config["nested"].append("list edit")
    assert (await database.agents.get(agent.id)).config == {"nested": ["original"]}


@pytest.mark.asyncio
async def test_transaction_rolls_back_all_repositories_on_failure(database):
    await database.agents.create(Agent(id="existing", name="before"))
    with pytest.raises(ValueError, match="stop"):
        async with database.transaction():
            await database.agents.update(Agent(id="existing", name="after"))
            await database.conversations.create(Conversation(id="new"))
            raise ValueError("stop")
    assert (await database.agents.get("existing")).name == "before"
    assert await database.conversations.get("new") is None
    status = database.get_status()["last_transaction"]
    assert status["state"] == "rolled_back"
    assert status["error_type"] == "ValueError"


@pytest.mark.asyncio
async def test_transaction_commits_all_repositories_and_retains_receipt(database):
    async with database.transaction() as transaction:
        assert transaction is database
        await database.agents.create(Agent(id="agent"))
        await database.conversations.create(
            Conversation(id="conversation", agent_id="agent")
        )
        assert await database.agents.get("agent") is not None
        assert database.get_status()["active_transaction"]["state"] == "active"
    assert await database.agents.count() == 1
    assert await database.conversations.count() == 1
    status = database.get_status()
    assert status["active_transaction"] is None
    assert status["last_transaction"]["state"] == "committed"
    assert status["last_transaction"]["commit_status"] == "acknowledged"
    status["last_transaction"]["state"] = "caller edit"
    assert database.get_status()["last_transaction"]["state"] == "committed"


@pytest.mark.asyncio
async def test_cancellation_rolls_back_instead_of_publishing_partial_work(database):
    entered = asyncio.Event()

    async def write():
        async with database.transaction():
            await database.agents.create(Agent(id="pending"))
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(write())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await database.agents.get("pending") is None
    assert database.get_status()["last_transaction"]["error_type"] == "CancelledError"


@pytest.mark.asyncio
async def test_filtering_happens_before_pagination(database):
    for identifier, status in [
        ("old", "active"),
        ("middle", "active"),
        ("new", "inactive"),
    ]:
        await database.agents.create(Agent(id=identifier, status=status))
    selected = await database.agents.list(
        limit=1, offset=1, filters={"status": "active"}
    )
    assert [item.id for item in selected] == ["old"]
    assert await database.agents.count(filters={"status": "active"}) == 2


@pytest.mark.asyncio
async def test_entity_timestamps_round_trip_without_mutating_input(database):
    when = datetime(2024, 1, 2, 3, 4, 5)
    record = Conversation(id="conversation", last_message_at=when)
    data = record.to_dict()
    before = copy.deepcopy(data)
    restored = Conversation.from_dict(data)
    assert data == before
    assert restored.last_message_at == when
    await database.conversations.create(record)
    assert (await database.conversations.get(record.id)).last_message_at == when


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["read", "write", "transaction", "close"])
async def test_competing_and_inherited_tasks_cannot_join_transaction(
    database, operation
):
    from stateset_agents.api.persistence import TransactionConflict

    entered = asyncio.Event()

    async def competing():
        await entered.wait()
        with pytest.raises(TransactionConflict):
            if operation == "read":
                await database.agents.get("pending")
            elif operation == "write":
                await database.agents.create(Agent(id="foreign"))
            elif operation == "close":
                await database.close()
            else:
                async with database.transaction():
                    pytest.fail("competing transaction admitted")

    outsider = asyncio.create_task(competing())
    async with database.transaction():
        await database.agents.create(Agent(id="pending"))
        inherited = asyncio.create_task(competing())
        entered.set()
        await asyncio.gather(outsider, inherited)
        assert await database.agents.get("pending") is not None
    assert await database.agents.get("foreign") is None
    assert database.get_status()["last_transaction"]["state"] == "committed"


@pytest.mark.asyncio
async def test_nested_scope_rejected_without_poisoning_outer_scope(database):
    from stateset_agents.api.persistence import TransactionConflict

    async with database.transaction():
        with pytest.raises(TransactionConflict, match="Nested"):
            async with database.transaction():
                pytest.fail("nested transaction admitted")
        await database.agents.create(Agent(id="outer"))
    assert await database.agents.get("outer") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["read", "close"])
async def test_finished_scope_cannot_affect_a_new_transaction(database, operation):
    from stateset_agents.api.persistence import TransactionConflict

    release = asyncio.Event()

    async def escaped():
        await release.wait()
        with pytest.raises(TransactionConflict):
            if operation == "close":
                await database.close()
            else:
                await database.agents.get("new")

    async with database.transaction():
        child = asyncio.create_task(escaped())
    # Reject the escaped scope even when no other transaction currently owns it.
    release.set()
    await child
    async with database.transaction():
        await database.agents.create(Agent(id="new"))
    assert database.get_status()["last_transaction"]["state"] == "committed"


@pytest.mark.asyncio
async def test_caught_operation_error_requires_rollback(database):
    from stateset_agents.api.persistence import TransactionAborted

    with pytest.raises(TransactionAborted):
        async with database.transaction():
            await database.agents.create(Agent(id="partial"))
            with pytest.raises(ValueError, match="not found"):
                await database.conversations.update(Conversation(id="missing"))
            with pytest.raises(TransactionAborted):
                await database.agents.create(Agent(id="after-error"))
    assert await database.agents.count() == 0
    receipt = database.get_status()["last_transaction"]
    assert receipt["operation_error_type"] == "ValueError"
    assert receipt["rollback_status"] == "acknowledged"


@pytest.mark.asyncio
async def test_rollback_failure_retains_original_error_and_closes_admission(
    database, monkeypatch
):
    failure = ValueError("body failure")

    def fail_rollback():
        raise OSError("rollback failure")

    monkeypatch.setattr(database, "_rollback", fail_rollback)
    with pytest.raises(ValueError) as caught:
        async with database.transaction():
            await database.agents.create(Agent(id="uncertain"))
            raise failure
    assert caught.value is failure
    receipt = database.get_status()["last_transaction"]
    assert receipt["error_type"] == "ValueError"
    assert receipt["rollback_error_type"] == "OSError"
    assert receipt["rollback_status"] == "unconfirmed"
    with pytest.raises(RuntimeError, match="closed"):
        await database.agents.get("uncertain")


@pytest.mark.asyncio
async def test_sqlite_cross_connection_isolation_contention_and_retry(tmp_path):
    import sqlite3

    config = DatabaseConfig(
        backend=DatabaseBackend.SQLITE, connection_url=str(tmp_path / "isolation.db")
    )
    async with UnitOfWork(config) as first, UnitOfWork(config) as second:
        async with first.transaction():
            await first.agents.create(Agent(id="private"))
            assert await second.agents.get("private") is None
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                async with second.transaction():
                    pytest.fail("second writer admitted")
            assert (
                second.get_status()["last_transaction"]["commit_status"]
                == "not_started"
            )
        assert await second.agents.get("private") is not None
        async with second.transaction():
            await second.conversations.create(Conversation(id="later"))
        assert await first.conversations.get("later") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("after_commit", [False, True])
async def test_sqlite_commit_error_does_not_invent_an_acknowledgement(
    tmp_path, monkeypatch, after_commit
):
    config = DatabaseConfig(
        backend=DatabaseBackend.SQLITE, connection_url=str(tmp_path / "commit.db")
    )
    async with UnitOfWork(config) as database:
        commit = database._commit
        calls = []

        def fail_commit():
            calls.append("commit")
            if after_commit:
                commit()
            raise OSError("commit acknowledgement lost")

        monkeypatch.setattr(database, "_commit", fail_commit)
        with pytest.raises(OSError, match="acknowledgement"):
            async with database.transaction():
                await database.agents.create(Agent(id="candidate"))
        receipt = database.get_status()["last_transaction"]
        assert receipt["commit_status"] == "unconfirmed"
        assert receipt["rollback_status"] == (
            "not_active" if after_commit else "acknowledged"
        )
        assert receipt["state"] == ("failed" if after_commit else "rolled_back")
        assert calls == ["commit"]
    async with UnitOfWork(config) as reopened:
        assert (await reopened.agents.get("candidate") is not None) is after_commit


@pytest.mark.asyncio
async def test_sqlite_memory_database_shares_connection_across_repositories():
    config = DatabaseConfig(backend=DatabaseBackend.SQLITE, connection_url=":memory:")
    async with UnitOfWork(config) as database:
        async with database.transaction():
            await database.agents.create(Agent(id="agent"))
            await database.conversations.create(Conversation(id="conversation"))
        assert await database.agents.count() == 1
        assert await database.conversations.count() == 1
    with pytest.raises(RuntimeError, match="closed"):
        await database.agents.count()


@pytest.mark.asyncio
async def test_close_retains_cleanup_through_repeated_cancellation():
    from types import SimpleNamespace

    database = UnitOfWork(DatabaseConfig())
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def close(name):
        calls.append(name)
        if name == "agents":
            entered.set()
            await release.wait()

    for name in ("agents", "conversations", "training_jobs", "api_keys"):
        setattr(
            database, "_" + name, SimpleNamespace(close=lambda name=name: close(name))
        )
    closing = asyncio.create_task(database.close())
    await asyncio.wait_for(entered.wait(), 2)
    try:
        for _ in range(3):
            closing.cancel()
            await asyncio.sleep(0)
            assert not closing.done()
        assert database.get_status()["state"] == "closing"
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    await database.close()
    assert calls == ["agents", "conversations", "training_jobs", "api_keys"]
    assert database.get_status()["state"] == "closed"


@pytest.mark.asyncio
async def test_old_global_close_cannot_detach_replacement(monkeypatch):
    from types import SimpleNamespace

    from stateset_agents.api import persistence

    entered, release = asyncio.Event(), asyncio.Event()

    async def close():
        entered.set()
        await release.wait()

    monkeypatch.setattr(persistence, "_database", SimpleNamespace(close=close))
    task = asyncio.create_task(persistence.close_database())
    await asyncio.wait_for(entered.wait(), 2)
    replacement = object()
    monkeypatch.setattr(persistence, "_database", replacement)
    release.set()
    await task
    assert persistence.get_database() is replacement


@pytest.mark.asyncio
async def test_sqlite_process_exit_preserves_only_committed_groups(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    path = tmp_path / "process.db"
    script = """
import asyncio
import os
import sys
from stateset_agents.api.persistence import Agent, Conversation, DatabaseBackend, DatabaseConfig, UnitOfWork
async def main():
    database = UnitOfWork(DatabaseConfig(backend=DatabaseBackend.SQLITE, connection_url=sys.argv[1]))
    async with database.transaction():
        await database.agents.create(Agent(id="committed"))
        await database.conversations.create(Conversation(id="committed", agent_id="committed"))
    async with database.transaction():
        await database.agents.create(Agent(id="interrupted"))
        await database.conversations.create(Conversation(id="interrupted", agent_id="interrupted"))
        os._exit(0)
asyncio.run(main())
"""
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", script, str(path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    async with UnitOfWork(
        DatabaseConfig(backend=DatabaseBackend.SQLITE, connection_url=str(path))
    ) as database:
        assert [item.id for item in await database.agents.list()] == ["committed"]
        assert [item.id for item in await database.conversations.list()] == [
            "committed"
        ]


@pytest.mark.asyncio
async def test_failed_context_entry_closes_partial_sqlite_connection(
    tmp_path, monkeypatch
):
    from stateset_agents.api.persistence import SQLiteRepository

    database = UnitOfWork(
        DatabaseConfig(
            backend=DatabaseBackend.SQLITE, connection_url=str(tmp_path / "partial.db")
        )
    )
    original = SQLiteRepository.connect
    failure = RuntimeError("schema failed")
    calls = []

    async def connect(repository):
        calls.append(repository)
        if len(calls) == 2:
            assert database.get_status()["state"] == "new"
            with pytest.raises(RuntimeError, match="not connected"):
                _ = database.agents
            raise failure
        await original(repository)

    monkeypatch.setattr(SQLiteRepository, "connect", connect)
    with pytest.raises(RuntimeError) as caught:
        async with database:
            pytest.fail("partial initialization published")
    assert caught.value is failure
    assert database._store.closed
    assert database.get_status()["state"] == "closed"


@pytest.mark.asyncio
async def test_failed_cleanup_is_retained_and_not_replayed():
    from types import SimpleNamespace

    database = UnitOfWork(DatabaseConfig())
    calls = []
    failure = OSError("close failed")

    async def close(name):
        calls.append(name)
        if name == "agents":
            raise failure

    for name in ("agents", "conversations", "training_jobs", "api_keys"):
        setattr(
            database, "_" + name, SimpleNamespace(close=lambda name=name: close(name))
        )
    for _ in range(2):
        with pytest.raises(OSError) as caught:
            await database.close()
        assert caught.value is failure
    assert calls == ["agents", "conversations", "training_jobs", "api_keys"]
    assert database.get_status()["state"] == "failed"


@pytest.mark.asyncio
async def test_finished_task_cannot_access_a_later_transaction(database):
    from stateset_agents.api.persistence import TransactionConflict

    release = asyncio.Event()

    async def escaped():
        await release.wait()
        with pytest.raises(TransactionConflict):
            await database.agents.create(Agent(id="escaped"))

    async with database.transaction():
        child = asyncio.create_task(escaped())
    async with database.transaction():
        await database.agents.create(Agent(id="new"))
        release.set()
        await child
        assert database.get_status()["active_transaction"]["state"] == "active"
    assert [item.id for item in await database.agents.list()] == ["new"]


@pytest.mark.asyncio
async def test_duplicate_create_and_missing_update_never_succeed_silently(database):
    import sqlite3

    await database.agents.create(Agent(id="duplicate", name="original"))
    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        await database.agents.create(Agent(id="duplicate", name="replacement"))
    assert (await database.agents.get("duplicate")).name == "original"
    with pytest.raises(ValueError, match="not found"):
        await database.agents.update(Agent(id="missing"))
    assert await database.agents.count() == 1


@pytest.mark.asyncio
async def test_resource_scope_preserves_autocommit_when_body_raises(tmp_path):
    config = DatabaseConfig(
        backend=DatabaseBackend.SQLITE, connection_url=str(tmp_path / "autocommit.db")
    )
    with pytest.raises(ValueError):
        async with UnitOfWork(config) as database:
            await database.agents.create(Agent(id="published"))
            raise ValueError("ordinary resource scope, no transaction")
    async with UnitOfWork(config) as reopened:
        assert await reopened.agents.get("published") is not None


@pytest.mark.asyncio
async def test_nonfinite_json_failure_rolls_back_previous_sqlite_write(tmp_path):
    config = DatabaseConfig(
        backend=DatabaseBackend.SQLITE, connection_url=str(tmp_path / "finite.db")
    )
    async with UnitOfWork(config) as database:
        with pytest.raises(ValueError):
            async with database.transaction():
                await database.agents.create(Agent(id="first"))
                await database.conversations.create(
                    Conversation(id="invalid", metadata={"value": float("nan")})
                )
        assert await database.agents.count() == 0
        assert await database.conversations.count() == 0


@pytest.mark.asyncio
async def test_resource_cleanup_error_does_not_replace_body_error(
    tmp_path, monkeypatch
):
    database = UnitOfWork(
        DatabaseConfig(
            backend=DatabaseBackend.SQLITE, connection_url=str(tmp_path / "cleanup.db")
        )
    )
    await database.connect()
    original = database._store.close
    failure = ValueError("body error")

    def close():
        original()
        raise OSError("cleanup acknowledgement failed")

    monkeypatch.setattr(database._store, "close", close)
    with pytest.raises(ValueError) as caught:
        async with database:
            raise failure
    assert caught.value is failure
    assert database.get_status()["state"] == "failed"
    with pytest.raises(OSError):
        await database.close()


@pytest.mark.asyncio
async def test_shared_sqlite_connection_cleanup_is_attempted_once(
    tmp_path, monkeypatch
):
    database = UnitOfWork(
        DatabaseConfig(
            backend=DatabaseBackend.SQLITE,
            connection_url=str(tmp_path / "shared-close.db"),
        )
    )
    await database.connect()
    original = database._store.close
    calls = []
    failure = OSError("close acknowledgement unavailable")

    def close():
        calls.append("close")
        raise failure

    monkeypatch.setattr(database._store, "close", close)
    try:
        for _ in range(2):
            with pytest.raises(OSError) as caught:
                await database.close()
            assert caught.value is failure
        assert calls == ["close"]
        assert database.get_status()["state"] == "failed"
    finally:
        original()
