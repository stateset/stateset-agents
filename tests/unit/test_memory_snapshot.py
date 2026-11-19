"""Snapshot recovery under corruption, concurrent mutation, cancellation and crashes."""

import asyncio
import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import threading
from datetime import datetime, timezone

import numpy as np
import pytest

from stateset_agents.core.enhanced import MemorySnapshotConflict, VectorMemory
from stateset_agents.core.enhanced import memory_snapshot as snapshots
from stateset_agents.core.enhanced.vector_memory import MemoryEntry, _MemoryBackend

from .test_vector_memory_lifecycle import encoders as encoders


async def saved(tmp_path):
    memory = VectorMemory()
    await memory.add_entry(
        "refund café", {"nested": [None, True, 1, 2.5, {"x": "值"}]}, 0.7
    )
    await memory.add_entry("shipping", {"id": 2}, -0.3)
    memory.entries[0].timestamp = datetime(2025, 1, 2, 3, 4, tzinfo=timezone.utc)
    await memory.retrieve_relevant("refund")
    path = tmp_path / "memory.json"
    receipt = await memory.save_snapshot(path)
    return memory, path, receipt


def last(memory):
    return memory.get_snapshot_status()["last_finished"]


def rewrite(path, mutate):
    document = json.loads(path.read_text())
    mutate(document)
    document.pop("sha256")
    document["sha256"] = hashlib.sha256(snapshots._canonical(document)).hexdigest()
    path.write_bytes(snapshots._canonical(document))


@pytest.mark.asyncio
async def test_round_trip_preserves_data_and_retrieval_without_runtime_objects(
    tmp_path,
):
    source, path, receipt = await saved(tmp_path)
    assert receipt["state"] == "completed"
    assert receipt["publication"] == "published"
    assert receipt["file_synced"] is True
    assert receipt["directory_synced"] is (os.name == "posix")
    raw = json.loads(path.read_text())
    assert "embedding" not in raw["payload"]["entries"][0]
    target = VectorMemory("destination-model", max_entries=10)
    await target.add_entry("old")
    generation = target._owner().generation
    result = await target.restore_snapshot(path, expected_revision=receipt["revision"])
    assert result["publication"] == "published"
    assert result["file_synced"] is False  # Restore performs no file publication.
    assert target._owner().generation == generation + 1
    assert target.entries == source.entries
    assert target.entries is not source.entries
    assert target.embedding_model == "destination-model"
    assert target.max_entries == 10
    assert (await target.retrieve_relevant("refund"))[0].content == "refund café"
    assert source.entries[0].access_count == 1
    metadata = await VectorMemory.inspect_snapshot(path)
    assert metadata["revision"] == receipt["revision"]
    assert metadata["entry_count"] == 2
    assert "entries" not in metadata
    detached = target.get_snapshot_status()
    detached["last_finished"]["state"] = "oops"
    assert last(target)["state"] == "completed"
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_revision_checks_prevent_lost_updates_and_aba(tmp_path):
    source, path, first = await saved(tmp_path)
    with pytest.raises(MemorySnapshotConflict):
        await source.save_snapshot(path)
    assert last(source)["observed_revision"] == first["revision"]
    second = await source.save_snapshot(path, expected_revision=first["revision"])
    assert second["revision"] != first["revision"]
    other = VectorMemory()
    with pytest.raises(MemorySnapshotConflict):
        await other.save_snapshot(path, expected_revision=first["revision"])
    with pytest.raises(MemorySnapshotConflict):
        await other.restore_snapshot(path, expected_revision=first["revision"])
    assert other.entries == []
    assert (await other.inspect_snapshot(path))["revision"] == second["revision"]
    with pytest.raises(MemorySnapshotConflict):
        await other.save_snapshot(
            tmp_path / "absent.json", expected_revision=first["revision"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad", [object(), (1, 2), {1: "key"}, float("nan"), float("inf")]
)
async def test_nonportable_context_rejected_before_file_effect(tmp_path, bad):
    memory = VectorMemory()
    memory.entries = [MemoryEntry("text", context={"bad": bad})]
    with pytest.raises(ValueError):
        await memory.save_snapshot(tmp_path / "memory.json")
    assert not list(tmp_path.iterdir())
    assert last(memory)["publication"] == "not_started"


@pytest.mark.asyncio
async def test_cycles_and_extended_entries_are_not_silently_coerced(tmp_path):
    memory = VectorMemory()
    cycle = {}
    cycle["self"] = cycle
    memory.entries = [MemoryEntry("text", context=cycle)]
    with pytest.raises(ValueError, match="cycles"):
        await memory.save_snapshot(tmp_path / "memory.json")

    class ExtendedEntry(MemoryEntry):
        custom_state = "must not silently disappear"

    memory.entries = [ExtendedEntry("text")]
    with pytest.raises(ValueError, match="canonical"):
        await memory.save_snapshot(tmp_path / "memory.json")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(schema_version=2),
        lambda d: d.update(schema_version=True),
        lambda d: d.update(kind="other"),
        lambda d: d.update(extra="unknown"),
        lambda d: d.update(revision="bad"),
        lambda d: d["payload"].update(max_entries=1),
        lambda d: d["payload"]["entries"][0].update(access_count=True),
        lambda d: d["payload"]["entries"][0].update(timestamp="2025-01-02"),
        lambda d: d["payload"]["entries"][0].update(importance="high"),
        lambda d: d["payload"]["entries"][0].update(context=[]),
    ],
)
async def test_strict_schema_rejects_even_rehashed_invalid_data(tmp_path, mutation):
    source, path, _ = await saved(tmp_path)
    rewrite(path, mutation)
    original = source.entries
    with pytest.raises(ValueError):
        await source.restore_snapshot(path)
    assert source.entries is original
    with pytest.raises(ValueError):
        await VectorMemory.inspect_snapshot(path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b"[]", b"\xff"]
)
async def test_invalid_files_never_replace_current_entries_or_get_overwritten(
    tmp_path, raw
):
    source, path, receipt = await saved(tmp_path)
    original = source.entries
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        await source.restore_snapshot(path)
    with pytest.raises(ValueError):
        await source.save_snapshot(path, expected_revision=receipt["revision"])
    assert source.entries is original
    assert path.read_bytes() == raw


@pytest.mark.asyncio
async def test_checksum_limits_and_capacity_are_enforced(tmp_path):
    source, path, _ = await saved(tmp_path)
    original = source.entries
    for limit in (True, 0, -1, 10):
        with pytest.raises(ValueError):
            await source.restore_snapshot(path, max_bytes=limit)
        with pytest.raises(ValueError):
            await source.save_snapshot(tmp_path / "small.json", max_bytes=limit)
    with pytest.raises(ValueError, match="capacity"):
        await VectorMemory(max_entries=1).restore_snapshot(path)
    assert source.entries is original
    path.write_text(path.read_text().replace("shipping", "tampered"))
    with pytest.raises(ValueError, match="checksum"):
        await source.restore_snapshot(path)


@pytest.mark.asyncio
async def test_reindex_uses_destination_only_and_is_atomic(tmp_path, monkeypatch):
    _, path, _ = await saved(tmp_path)
    target = VectorMemory("local-model")
    target.model = object()
    target.entries = [MemoryEntry("old", embedding=np.array([1.0, 0.0]))]
    original = target.entries
    calls = []

    async def encode(text):
        calls.append(text)
        return (
            np.array([1.0, 0.0, 0.0]) if text.startswith("refund") else np.array([1.0])
        )

    monkeypatch.setattr(target, "_generate_embedding", encode)
    with pytest.raises(ValueError, match="inconsistent"):
        await target.restore_snapshot(path)
    assert target.entries is original
    assert calls == ["refund café", "shipping"]

    async def good(text):
        return np.array([1.0, 0.0, 0.0])

    monkeypatch.setattr(target, "_generate_embedding", good)
    await target.restore_snapshot(path)
    assert all(entry.embedding.shape == (3,) for entry in target.entries)
    assert target.embedding_model == "local-model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "effect", ["insert", "retrieve", "nested", "replace", "capacity", "backend"]
)
async def test_restore_cannot_overwrite_newer_memory(tmp_path, monkeypatch, effect):
    _, path, _ = await saved(tmp_path)
    target = VectorMemory()
    target.model = object()
    target.entries = [
        MemoryEntry("original", context={"nested": []}, embedding=np.ones(2))
    ]
    entered, release = asyncio.Event(), asyncio.Event()

    async def encode(text):
        if text == "refund café":
            entered.set()
            await release.wait()
        return np.ones(2)

    monkeypatch.setattr(target, "_generate_embedding", encode)
    task = asyncio.create_task(target.restore_snapshot(path))
    await asyncio.wait_for(entered.wait(), 5)
    if effect == "insert":
        await target.add_entry("newer")
    elif effect == "retrieve":
        await target.retrieve_relevant("original")
    elif effect == "nested":
        target.entries[0].context["nested"].append("newer")
    elif effect == "replace":
        target.entries = list(target.entries)
    elif effect == "capacity":
        target.max_entries += 1
    else:
        target._owner().generation += 1
    current = target.entries
    release.set()
    with pytest.raises(MemorySnapshotConflict, match="Live memory"):
        await task
    assert target.entries is current
    assert target.entries[0].content == "original"
    assert last(target)["publication"] == "not_started"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["add", "retrieve"])
async def test_restore_fences_older_inflight_operations(tmp_path, monkeypatch, action):
    _, path, _ = await saved(tmp_path)
    target = VectorMemory()
    target.model = object()
    target.entries = [MemoryEntry("old", embedding=np.ones(2))]
    entered, release = asyncio.Event(), asyncio.Event()

    async def encode(text):
        if text == "late":
            entered.set()
            await release.wait()
        return np.ones(2)

    monkeypatch.setattr(target, "_generate_embedding", encode)
    task = asyncio.create_task(
        target.add_entry("late")
        if action == "add"
        else target.retrieve_relevant("late")
    )
    await asyncio.wait_for(entered.wait(), 5)
    await target.restore_snapshot(path)
    release.set()
    with pytest.raises(RuntimeError, match="index restored"):
        await task
    assert [entry.content for entry in target.entries] == ["refund café", "shipping"]
    assert target.entries[0].access_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("suppress", [False, True])
async def test_restore_cancellation_owns_cleanup_and_never_commits(
    tmp_path, monkeypatch, suppress
):
    _, path, _ = await saved(tmp_path)
    target = VectorMemory()
    target.model = object()
    original = target.entries
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def encode(text):
        calls.append(text)
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            if not suppress:
                raise
            return np.ones(2)

    monkeypatch.setattr(target, "_generate_embedding", encode)
    task = asyncio.create_task(target.restore_snapshot(path))
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    await asyncio.wait_for(cleaning.wait(), 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    with pytest.raises(MemorySnapshotConflict):
        await target.restore_snapshot(path)
    with pytest.raises(RuntimeError, match="snapshot"):
        copy.deepcopy(target)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert target.entries is original
    assert calls == ["refund café"]
    assert last(target)["state"] == "cancelled"


@pytest.mark.asyncio
async def test_save_cancellation_drains_worker_and_reports_actual_publication(
    tmp_path, monkeypatch
):
    source, path, first = await saved(tmp_path)
    original = snapshots.os.replace
    entered, release = threading.Event(), threading.Event()

    def replace(src, dst):
        entered.set()
        assert release.wait(5)
        return original(src, dst)

    monkeypatch.setattr(snapshots.os, "replace", replace)
    task = asyncio.create_task(
        source.save_snapshot(path, expected_revision=first["revision"])
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert source.get_snapshot_status()["active"]["publication"] == "unconfirmed"
        with pytest.raises(RuntimeError, match="snapshot"):
            pickle.dumps(source)
        with pytest.raises(MemorySnapshotConflict):
            await VectorMemory().save_snapshot(
                path, expected_revision=first["revision"]
            )
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    receipt = last(source)
    assert receipt["state"] == "cancelled"
    assert receipt["publication"] == "published"
    assert receipt["directory_synced"] is (os.name == "posix")
    assert (await source.inspect_snapshot(path))["revision"] == receipt["revision"]
    assert source.get_snapshot_status()["active"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["fsync_file", "before_replace", "after_replace", "fsync_directory"]
)
async def test_file_failures_preserve_truthful_publication_receipts(
    tmp_path, monkeypatch, failure
):
    source, path, first = await saved(tmp_path)
    old_bytes = path.read_bytes()
    original_replace = snapshots.os.replace

    def fail(*args):
        raise OSError("injected failure")

    def replace(src, dst):
        if failure == "after_replace":
            original_replace(src, dst)
        fail()

    if failure == "fsync_file":
        monkeypatch.setattr(snapshots.os, "fsync", fail)
    elif failure == "fsync_directory":
        monkeypatch.setattr(snapshots, "_sync_directory", fail)
    else:
        monkeypatch.setattr(snapshots.os, "replace", replace)
    with pytest.raises(OSError):
        await source.save_snapshot(path, expected_revision=first["revision"])
    receipt = last(source)
    assert receipt["state"] == "failed"
    assert receipt["work_error_type"] == "OSError"
    if failure in ("fsync_file", "before_replace"):
        assert path.read_bytes() == old_bytes
    else:
        assert (await source.inspect_snapshot(path))["revision"] == receipt["revision"]
    assert (
        receipt["publication"]
        == {
            "fsync_file": "not_started",
            "before_replace": "unconfirmed",
            "after_replace": "unconfirmed",
            "fsync_directory": "published",
        }[failure]
    )
    assert receipt["directory_synced"] is False
    assert not list(tmp_path.glob("*.stage-*"))


@pytest.mark.asyncio
async def test_special_paths_and_missing_parent_are_rejected(tmp_path):
    source, path, _ = await saved(tmp_path)
    link = tmp_path / "symlink.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        await source.restore_snapshot(link)
    with pytest.raises(ValueError, match="symlink"):
        await source.save_snapshot(link)
    with pytest.raises(FileNotFoundError):
        await source.save_snapshot(tmp_path / "absent" / "file.json")
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "pipe"
        os.mkfifo(fifo)
        with pytest.raises(ValueError, match="regular"):
            await source.restore_snapshot(fifo)


@pytest.mark.asyncio
async def test_idle_copy_pickle_and_legacy_backend_state(tmp_path):
    memory, _, _ = await saved(tmp_path)
    for clone in (copy.deepcopy(memory), pickle.loads(pickle.dumps(memory))):
        assert clone.entries == memory.entries
        assert clone.get_snapshot_status() == memory.get_snapshot_status()
        assert clone._owner().lock is not memory._owner().lock
    owner = _MemoryBackend()
    legacy = {
        key: value
        for key, value in owner.__getstate__().items()
        if key
        not in (
            "mutation_revision",
            "snapshot_operation",
            "last_snapshot_operation",
        )
    }
    owner.__setstate__(legacy)
    assert owner.mutation_revision == 0
    assert owner.snapshot_operation is None


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before", "after"])
async def test_process_death_preserves_one_complete_file_and_releases_writer_lock(
    tmp_path, phase
):
    _, path, first = await saved(tmp_path)
    # The subprocess holds the real OS writer lock until it is killed at replace.
    script = r"""
import asyncio, sys
from stateset_agents.core.enhanced import VectorMemory
from stateset_agents.core.enhanced import memory_snapshot as snapshots
original = snapshots.os.replace
def replace(src, dst):
    if sys.argv[3] == "after":
        original(src, dst)
    print("READY", flush=True)
    sys.stdin.read(1)
    if sys.argv[3] == "before":
        original(src, dst)
snapshots.os.replace = replace
async def main():
    memory = VectorMemory()
    await memory.add_entry("new process data")
    await memory.save_snapshot(sys.argv[1], expected_revision=sys.argv[2])
asyncio.run(main())
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(path), first["revision"], phase],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert (
            await asyncio.wait_for(asyncio.to_thread(process.stdout.readline), 20)
            == "READY\n"
        )
        with pytest.raises(MemorySnapshotConflict, match="writer"):
            await VectorMemory().save_snapshot(path)
    finally:
        process.kill()
        await asyncio.to_thread(process.communicate, timeout=10)
    expected = (
        ["refund café", "shipping"] if phase == "before" else ["new process data"]
    )
    recovery = r"""
import asyncio, json, sys
from stateset_agents.core.enhanced import VectorMemory
async def main():
    memory = VectorMemory()
    await memory.restore_snapshot(sys.argv[1])
    print(json.dumps([entry.content for entry in memory.entries]))
asyncio.run(main())
"""
    recovered = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", recovery, str(path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert json.loads(recovered.stdout) == expected
    info = await VectorMemory.inspect_snapshot(path)
    assert (info["revision"] == first["revision"]) is (phase == "before")
    # Orphan staging files are never loaded/promoted. Lock ownership died with process.
    stages = list(tmp_path.glob("*.stage-*"))
    assert bool(stages) is (phase == "before")
    await VectorMemory().save_snapshot(path, expected_revision=info["revision"])
    assert list(tmp_path.glob("*.stage-*")) == stages


@pytest.mark.asyncio
async def test_restore_invalidates_deferred_enhanced_candidate(tmp_path, encoders):
    from stateset_agents.core.agent import AgentConfig
    from stateset_agents.core.enhanced import EnhancedMultiTurnAgent

    memory = VectorMemory("first")
    agent = EnhancedMultiTurnAgent(
        AgentConfig(model_name="stub://snapshot", stub_responses=["answer"]),
        memory_system=memory,
        reasoning_enabled=False,
    )
    await agent.initialize()
    await memory.add_entry("refund")
    path = tmp_path / "agent-memory.json"
    await memory.save_snapshot(path)
    candidate = (await agent.generate_turns("refund", 1))[0]
    await memory.restore_snapshot(path)
    status = agent.get_agent_status()["memory_snapshot"]
    assert status["last_finished"]["action"] == "restore"
    assert status["last_finished"]["state"] == "completed"
    status["last_finished"]["state"] = "changed"
    assert last(memory)["state"] == "completed"
    with pytest.raises(RuntimeError, match="memory changed"):
        await agent.remember_candidate(candidate)
    assert [entry.content for entry in memory.entries] == ["refund"]


@pytest.mark.asyncio
async def test_failed_backend_can_be_backed_up_without_running_source_model(
    tmp_path, encoders
):
    memory = VectorMemory("first")
    await memory.initialize()
    await memory.add_entry("refund")
    _, failures = encoders
    failures["to"] = True
    with pytest.raises(RuntimeError):
        await memory.initialize(force=True)
    path = tmp_path / "failed-backend.json"
    await memory.save_snapshot(path)
    # Restoring entry data never attempts the failed source encoder configuration.
    target = VectorMemory("unloaded-destination")
    await target.restore_snapshot(path)
    assert (await target.retrieve_relevant("refund"))[0].content == "refund"


@pytest.mark.asyncio
async def test_save_is_a_detached_point_in_time_snapshot(tmp_path, monkeypatch):
    memory, path, first = await saved(tmp_path)
    original = snapshots._publish
    entered, release = threading.Event(), threading.Event()

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(snapshots, "_publish", blocked)
    task = asyncio.create_task(
        memory.save_snapshot(path, expected_revision=first["revision"])
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        await memory.add_entry("later")
        memory.entries[0].context["nested"].append("later")
    finally:
        release.set()
    receipt = await task
    restored = VectorMemory()
    await restored.restore_snapshot(path)
    assert len(restored.entries) == receipt["entry_count"] == 2
    assert "later" not in restored.entries[0].context["nested"]
    assert memory._owner().mutation_revision > receipt["captured_mutation_revision"]


@pytest.mark.asyncio
async def test_cancelled_read_retains_ownership_until_worker_failure(
    tmp_path, monkeypatch
):
    memory, path, _ = await saved(tmp_path)
    original = memory.entries
    entered, release = threading.Event(), threading.Event()

    def fail_read(*args):
        entered.set()
        assert release.wait(5)
        raise OSError("read failed during cancellation")

    monkeypatch.setattr(snapshots, "_read", fail_read)
    task = asyncio.create_task(memory.restore_snapshot(path))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        with pytest.raises(MemorySnapshotConflict):
            await memory.save_snapshot(tmp_path / "other.json")
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert memory.entries is original
    assert last(memory)["state"] == "cancelled"
    assert last(memory)["work_error_type"] == "OSError"
    assert last(memory)["error_type"] == "CancelledError"


@pytest.mark.asyncio
async def test_cleanup_failure_after_publication_does_not_hide_published_file(
    tmp_path, monkeypatch
):
    memory, path, first = await saved(tmp_path)
    unlink = snapshots.Path.unlink

    def fail_stage_cleanup(self, *args, **kwargs):
        if self.name.startswith(".memory.json.stage-"):
            raise PermissionError("cleanup failure")
        return unlink(self, *args, **kwargs)

    monkeypatch.setattr(snapshots.Path, "unlink", fail_stage_cleanup)
    with pytest.raises(PermissionError):
        await memory.save_snapshot(path, expected_revision=first["revision"])
    receipt = last(memory)
    assert receipt["publication"] == "published"
    assert receipt["cleanup_error_type"] == "PermissionError"
    assert (await VectorMemory.inspect_snapshot(path))["revision"] == receipt[
        "revision"
    ]
