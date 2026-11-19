"""State history must remain stable across nested mutations and repeated restore."""

import asyncio
from dataclasses import dataclass, field

import pytest

from stateset_agents.core.enhanced_state_management import (
    ConversationStateManager,
    StateManager,
)


class Uncopyable:
    def __deepcopy__(self, memo):
        raise RuntimeError("resource cannot be cloned")


@dataclass
class Payload:
    items: list[str] = field(default_factory=list)


@pytest.mark.asyncio
async def test_snapshot_survives_input_and_cached_value_mutation_and_repeated_restore():
    manager = StateManager()
    original = {"turns": [{"content": "before", "tags": ["saved"]}]}
    await manager.set("conversation", original, create_snapshot=True)
    snapshot_id = manager.get_snapshots()[0].snapshot_id

    original["turns"][0]["content"] = "after"
    original["turns"].append({"content": "later"})
    await manager.set("temporary", [1])

    for _ in range(2):
        assert await manager.restore_snapshot(snapshot_id)
        assert not await manager.exists("temporary")
        restored = await manager.get("conversation")
        assert restored == {"turns": [{"content": "before", "tags": ["saved"]}]}
        assert restored is not original
        restored["turns"][0]["tags"].append("mutated after restore")
        await manager.set("temporary", [2])

    assert manager.get_snapshots()[0].state_data == {
        "conversation": {"turns": [{"content": "before", "tags": ["saved"]}]}
    }


@pytest.mark.asyncio
async def test_snapshot_inspection_returns_detached_data_and_metadata():
    manager = StateManager()
    await manager.set("key", {"values": [1]}, create_snapshot=True)
    observed = manager.get_snapshots()
    snapshot_id = observed[0].snapshot_id
    observed[0].state_data["key"]["values"].append(2)
    observed[0].metadata["version"] = 999
    observed[0].snapshot_id = "changed"
    observed.clear()

    assert await manager.restore_snapshot(snapshot_id)
    assert await manager.get("key") == {"values": [1]}
    assert manager.state_version == 1
    assert manager.get_snapshots()[0].snapshot_id == snapshot_id


@pytest.mark.asyncio
async def test_snapshot_preserves_cycles_aliases_and_python_value_types():
    manager = StateManager()
    shared = []
    cycle = {"self": None}
    cycle["self"] = cycle
    payload = Payload(["before"])
    await manager.set("left", shared)
    await manager.set("right", shared)
    await manager.set("cycle", cycle)
    await manager.set("payload", payload)
    await manager.set("types", ({1, 2}, b"bytes"), create_snapshot=True)
    snapshot_id = manager.get_snapshots()[0].snapshot_id
    shared.append("after")
    payload.items.append("after")

    assert await manager.restore_snapshot(snapshot_id)
    state = manager.state
    assert state["left"] == []
    assert state["left"] is state["right"]
    assert state["left"] is not shared
    assert state["cycle"]["self"] is state["cycle"]
    assert isinstance(state["payload"], Payload)
    assert state["payload"].items == ["before"]
    assert state["types"] == ({1, 2}, b"bytes")
    assert isinstance(state["types"], tuple)


@pytest.mark.asyncio
async def test_conversation_updates_do_not_rewrite_saved_history():
    manager = StateManager()
    conversations = ConversationStateManager(manager)
    await conversations.create_conversation("chat", "user", {"status": "before"})
    await conversations.add_turn("chat", "user", "hello")
    snapshot_id = await manager._create_snapshot()
    await conversations.add_turn("chat", "assistant", "later")
    await conversations.update_context("chat", {"status": "after"})

    assert await manager.restore_snapshot(snapshot_id)
    conversation = await conversations.get_conversation("chat")
    assert [turn["content"] for turn in conversation["turns"]] == ["hello"]
    assert conversation["context"] == {"status": "before"}


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_existing_value", [False, True])
async def test_snapshot_copy_failure_rejects_write_before_side_effects(
    bad_existing_value,
):
    manager = StateManager()
    await manager.set("key", "original")
    bad_value = Uncopyable()
    if bad_existing_value:
        await manager.set("resource", bad_value)
    manager.dirty_keys.clear()
    before_log = manager.get_change_log()
    events = []
    await manager.watch("key", lambda *args: events.append(args))

    with pytest.raises(TypeError, match="deepcopy") as error:
        await manager.set(
            "key",
            "replacement" if bad_existing_value else bad_value,
            create_snapshot=True,
        )
    assert isinstance(error.value.__cause__, RuntimeError)
    assert await manager.get("key", use_cache=False) == "original"
    assert await manager.get("key") == "original"
    assert manager.get_change_log() == before_log
    assert not events
    assert not manager.dirty_keys
    assert manager.state_version == 1
    assert manager.get_snapshots() == []


@pytest.mark.asyncio
async def test_direct_snapshot_copy_failure_does_not_publish_history():
    manager = StateManager()
    await manager.set("resource", Uncopyable())
    with pytest.raises(TypeError, match="deepcopy"):
        await manager._create_snapshot()
    assert manager.get_snapshots() == []
    assert manager.state_version == 1


@pytest.mark.asyncio
async def test_failed_restore_copy_preserves_current_state_and_cache():
    manager = StateManager()
    await manager.set("key", ["saved"], create_snapshot=True)
    snapshot_id = manager.get_snapshots()[0].snapshot_id
    await manager.set("key", ["current"])
    await manager.set("extra", [1])
    # Public legacy internals can still be edited directly; even corrupt history
    # must not clear live state before the restore has been prepared.
    manager.snapshots[0].state_data["resource"] = Uncopyable()
    before_log = manager.get_change_log()
    before_dirty = manager.dirty_keys.copy()
    with pytest.raises(TypeError, match="deepcopy"):
        await manager.restore_snapshot(snapshot_id)
    assert await manager.get("key", use_cache=False) == ["current"]
    assert await manager.get("key") == ["current"]
    assert await manager.get("extra") == [1]
    assert manager.state_version == 2
    assert manager.get_change_log() == before_log
    assert manager.dirty_keys == before_dirty


@pytest.mark.asyncio
async def test_disabled_versioning_keeps_accepting_uncopyable_values():
    manager = StateManager(enable_versioning=False)
    value = Uncopyable()
    assert await manager.set("key", value, create_snapshot=True)
    assert await manager.get("key") is value
    assert await manager._create_snapshot() == ""
    assert not await manager.restore_snapshot("missing")
    assert manager.get_snapshots() == []


@pytest.mark.asyncio
async def test_snapshot_retention_and_version_contract_are_preserved():
    manager = StateManager(max_snapshots=2)
    ids = []
    for value in range(3):
        await manager.set("value", [value], create_snapshot=True)
        ids.append(manager.get_snapshots()[-1].snapshot_id)
    assert manager.state_version == 4
    assert [snapshot.snapshot_id for snapshot in manager.get_snapshots()] == ids[1:]
    assert not await manager.restore_snapshot(ids[0])
    assert await manager.restore_snapshot(ids[1])
    assert manager.state_version == 2
    assert await manager.get("value") == [1]


@pytest.mark.asyncio
async def test_snapshot_captures_write_before_cache_or_watcher_can_mutate_it(
    monkeypatch,
):
    manager = StateManager()
    entered, release = asyncio.Event(), asyncio.Event()
    cache_set = manager.cache_backend.set

    async def delayed_set(key, value, ttl=None):
        if key == "saved":
            entered.set()
            await release.wait()
        return await cache_set(key, value, ttl)

    def watcher(key, old, new):
        new.append("watcher")

    monkeypatch.setattr(manager.cache_backend, "set", delayed_set)
    await manager.watch("saved", watcher)
    value = ["before"]
    writer = asyncio.create_task(manager.set("saved", value, create_snapshot=True))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert manager.get_snapshots() == []
        value.append("during cache write")
        await manager.set("later", [1])
        release.set()
        assert await writer
        snapshot = manager.get_snapshots()[0]
        assert snapshot.state_data == {"saved": ["before"]}
        assert value == ["before", "during cache write", "watcher"]
    finally:
        release.set()
        await asyncio.gather(writer, return_exceptions=True)
