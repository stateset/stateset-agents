"""Portable named conversation recovery without callback replay or lost updates."""

import asyncio
import copy
import hashlib
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone

import pytest

from stateset_agents.core import conversation_snapshot as snapshots
from stateset_agents.core.conversation import ConversationResetError
from stateset_agents.core.enhanced import VectorMemory
from stateset_agents.core.multiturn_agent import ConversationConflict, MultiTurnAgent
from stateset_agents.utils.snapshot_files import canonical_json, writer_lock


async def saved(tmp_path):
    agent = MultiTurnAgent({}, response_backend=lambda prompt: "first answer")
    context = await agent.start_conversation(
        user_id="owner",
        initial_context={
            "topic": "shipping",
            "intent": "track",
            "entities": {"order": [12]},
            "metadata": {"locale": "fr", "nested": [None, True, 2.5, "café"]},
        },
    )
    context.started_at = datetime(2025, 1, 2, tzinfo=timezone.utc)
    await agent.continue_conversation(context.conversation_id, "my order")
    path = tmp_path / "conversation.json"
    receipt = await agent.save_conversation(context.conversation_id, path)
    return agent, context, path, receipt


def last(agent):
    return agent.get_conversation_snapshot_status()["last_finished"]


def rewrite(path, mutate):
    document = json.loads(path.read_text())
    mutate(document)
    document.pop("sha256")
    document["sha256"] = hashlib.sha256(canonical_json(document)).hexdigest()
    path.write_bytes(canonical_json(document))


@pytest.mark.asyncio
async def test_roundtrip_preserves_context_and_continues_with_destination_backend(
    tmp_path,
):
    source, context, path, receipt = await saved(tmp_path)
    prompts = []
    target = MultiTurnAgent(
        {"model_name": "stub://destination"},
        response_backend=lambda prompt: prompts.append(prompt) or "second answer",
    )
    restored = await target.restore_conversation(
        path, expected_revision=receipt["revision"]
    )
    assert restored == context
    assert restored is not context
    assert prompts == []
    assert restored.get_status()["revision"] > context.get_status()["revision"]
    assert (
        restored.get_status()["last_finished"] == context.get_status()["last_finished"]
    )
    assert receipt["publication"] == "published"
    assert receipt["file_synced"] is True
    assert receipt["directory_synced"] is (os.name == "posix")
    assert last(target)["file_synced"] is False
    assert target.config.model_name == "stub://destination"
    answer = await target.continue_conversation(
        restored.conversation_id, "when will it arrive?"
    )
    assert answer[-1]["content"] == "second answer"
    assert "User: my order" in prompts[0]
    assert "Assistant: first answer" in prompts[0]
    assert "when will it arrive?" in prompts[0]
    restored.entities["order"].append(99)
    assert context.entities["order"] == [12]
    assert len(context.history) == 2
    metadata = await source.inspect_conversation_snapshot(path)
    assert metadata["turn_count"] == 2
    assert metadata["revision"] == receipt["revision"]
    assert "conversation" not in metadata
    status = target.get_conversation_snapshot_status()
    status["last_finished"]["state"] = "mutated"
    assert last(target)["state"] == "completed"
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_fresh_process_restores_and_uses_history(tmp_path):
    _, context, path, _ = await saved(tmp_path)
    program = """
import asyncio, json, sys
from stateset_agents.core.multiturn_agent import MultiTurnAgent
async def main():
    seen = []
    agent = MultiTurnAgent({}, response_backend=lambda prompt: seen.append(prompt) or "recovered")
    context = await agent.restore_conversation(sys.argv[1])
    await agent.continue_conversation(context.conversation_id, "next question")
    print(json.dumps({"id": context.conversation_id, "prompt": seen[0], "turns": len(context.history)}))
asyncio.run(main())
"""
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", program, str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    recovered = json.loads(result.stdout)
    assert recovered["id"] == context.conversation_id
    assert recovered["turns"] == 4
    assert "first answer" in recovered["prompt"]


@pytest.mark.asyncio
async def test_revisions_create_only_restore_collision_and_cross_kind_lock(tmp_path):
    agent, context, path, first = await saved(tmp_path)
    with pytest.raises(ConversationConflict):
        await agent.save_conversation(context.conversation_id, path)
    assert last(agent)["observed_revision"] == first["revision"]
    second = await agent.save_conversation(
        context.conversation_id, path, expected_revision=first["revision"]
    )
    assert second["revision"] != first["revision"]
    with pytest.raises(ConversationConflict):
        await agent.save_conversation(
            context.conversation_id, path, expected_revision=first["revision"]
        )
    destination = MultiTurnAgent({})
    with pytest.raises(ConversationConflict):
        await destination.restore_conversation(
            path, expected_revision=first["revision"]
        )
    assert destination.active_conversations == {}
    with pytest.raises(ConversationConflict, match="already exists"):
        await agent.restore_conversation(path)
    assert agent.active_conversations[context.conversation_id] is context
    with writer_lock(path, RuntimeError):
        with pytest.raises(ConversationConflict, match="Another writer"):
            await agent.save_conversation(
                context.conversation_id, path, expected_revision=second["revision"]
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
async def test_incomplete_turn_receipt_survives_without_replay(tmp_path, outcome):
    agent = MultiTurnAgent({})
    context = await agent.start_conversation()

    async def incomplete(candidate):
        if outcome == "cancelled":
            raise asyncio.CancelledError
        raise LookupError("provider outcome unknown")

    agent.strategies["incomplete"] = incomplete
    with pytest.raises((LookupError, asyncio.CancelledError)):
        await agent.continue_conversation(
            context.conversation_id, "accepted", strategy="incomplete"
        )
    path = tmp_path / "incomplete.json"
    await agent.save_conversation(context.conversation_id, path)
    calls = []
    target = MultiTurnAgent(
        {}, response_backend=lambda prompt: calls.append(prompt) or "unexpected"
    )
    recovered = await target.restore_conversation(path)
    assert calls == []
    assert [turn["content"] for turn in recovered.history] == ["accepted"]
    assert recovered.get_status()["last_finished"]["state"] == outcome
    assert recovered.get_status()["last_finished"]["generation_status"] == "unconfirmed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad", [object(), (1, 2), {1: "key"}, float("nan"), float("inf")]
)
async def test_nonportable_data_rejected_without_file_effects(tmp_path, bad):
    agent = MultiTurnAgent({})
    context = await agent.start_conversation(initial_context={"metadata": {"bad": bad}})
    with pytest.raises(ValueError):
        await agent.save_conversation(
            context.conversation_id, tmp_path / "invalid.json"
        )
    assert not list(tmp_path.iterdir())
    assert last(agent)["publication"] == "not_started"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(schema_version=True),
        lambda d: d.update(schema_version=2),
        lambda d: d.update(kind="other"),
        lambda d: d.update(extra="unknown"),
        lambda d: d.update(revision="bad"),
        lambda d: d["payload"].update(context_revision=True),
        lambda d: d["payload"]["conversation"].update(started_at="2025-01-02"),
        lambda d: d["payload"]["conversation"].update(entities=[]),
        lambda d: d["payload"]["conversation"].update(
            history=[{"content": "missing role"}]
        ),
        lambda d: d["payload"]["last_turn"].update(state="running"),
        lambda d: d["payload"]["last_turn"].update(user_recorded=False),
        lambda d: d["payload"]["last_turn"].update(cancellation_requested=True),
    ],
)
async def test_invalid_schema_never_attaches_context(tmp_path, mutation):
    _, _, path, _ = await saved(tmp_path)
    rewrite(path, mutation)
    target = MultiTurnAgent({})
    with pytest.raises(ValueError):
        await target.restore_conversation(path)
    assert target.active_conversations == {}
    assert last(target)["publication"] == "not_started"
    with pytest.raises(ValueError):
        await target.inspect_conversation_snapshot(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b"not JSON"])
async def test_invalid_json_is_rejected(tmp_path, raw):
    path = tmp_path / "invalid.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        await MultiTurnAgent({}).restore_conversation(path)


@pytest.mark.asyncio
async def test_checksum_size_and_symlink_rejected(tmp_path):
    agent, context, path, _ = await saved(tmp_path)
    with pytest.raises(ValueError, match="max_bytes"):
        await MultiTurnAgent({}).restore_conversation(path, max_bytes=10)
    with pytest.raises(ValueError, match="max_bytes"):
        await agent.save_conversation(
            context.conversation_id, tmp_path / "small.json", max_bytes=10
        )
    assert not (tmp_path / "small.json").exists()
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        await MultiTurnAgent({}).restore_conversation(link)
    document = json.loads(path.read_text())
    document["payload"]["conversation"]["topic"] = "tampered"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="checksum"):
        await MultiTurnAgent({}).restore_conversation(path)


@pytest.mark.asyncio
async def test_planning_state_is_not_silently_discarded(tmp_path):
    agent = MultiTurnAgent({}, planning_manager=object())
    context = await agent.start_conversation()
    path = tmp_path / "plan.json"
    with pytest.raises(ValueError, match="planning-manager"):
        await agent.save_conversation(context.conversation_id, path)
    with pytest.raises(ValueError, match="planning-manager"):
        await agent.restore_conversation(path)
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_active_turn_cannot_be_snapshotted(tmp_path):
    agent = MultiTurnAgent({})
    context = await agent.start_conversation()
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(candidate):
        entered.set()
        await release.wait()
        return "answer"

    agent.strategies["blocked"] = blocked
    task = asyncio.create_task(
        agent.continue_conversation(
            context.conversation_id, "question", strategy="blocked"
        )
    )
    try:
        await entered.wait()
        with pytest.raises(ConversationConflict, match="active turn"):
            await agent.save_conversation(
                context.conversation_id, tmp_path / "active.json"
            )
    finally:
        release.set()
        await task
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_snapshot_captures_one_revision_while_later_turns_continue(
    tmp_path, monkeypatch
):
    agent, context, path, first = await saved(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = snapshots.publish_bytes

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(snapshots, "publish_bytes", blocked)
    task = asyncio.create_task(
        agent.save_conversation(
            context.conversation_id, path, expected_revision=first["revision"]
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        await agent.continue_conversation(context.conversation_id, "later")
        with pytest.raises(ConversationConflict):
            await agent.save_conversation(
                context.conversation_id, tmp_path / "other.json"
            )
    finally:
        release.set()
    receipt = await task
    recovered = await MultiTurnAgent({}).restore_conversation(path)
    assert len(recovered.history) == 2
    assert len(context.history) == 4
    assert receipt["context_revision"] < context.get_status()["revision"]


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["cancel", "reset"])
async def test_interrupted_restore_drains_read_without_attaching(
    tmp_path, monkeypatch, interrupt
):
    _, _, path, _ = await saved(tmp_path)
    target = MultiTurnAgent({})
    entered, release = threading.Event(), threading.Event()
    original = snapshots._read

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(snapshots, "_read", blocked)
    task = asyncio.create_task(target.restore_conversation(path))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        if interrupt == "cancel":
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
        else:
            await target.reset()
        await asyncio.sleep(0)
        assert not task.done()
        with pytest.raises(RuntimeError, match="Cannot copy"):
            copy.deepcopy(target)
    finally:
        release.set()
    with pytest.raises(
        asyncio.CancelledError if interrupt == "cancel" else ConversationResetError
    ):
        await task
    assert target.active_conversations == {}
    assert last(target)["publication"] == "not_started"
    assert target.get_conversation_snapshot_status()["active"] == []


@pytest.mark.asyncio
async def test_cancelled_save_can_publish_and_retains_truthful_receipt(
    tmp_path, monkeypatch
):
    agent, context, path, first = await saved(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = snapshots.publish_bytes

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(snapshots, "publish_bytes", blocked)
    task = asyncio.create_task(
        agent.save_conversation(
            context.conversation_id, path, expected_revision=first["revision"]
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        with pytest.raises(RuntimeError, match="Cannot copy"):
            copy.deepcopy(context)
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    receipt = last(agent)
    assert receipt["state"] == "cancelled"
    assert receipt["publication"] == "published"
    assert (await agent.inspect_conversation_snapshot(path))["revision"] == receipt[
        "revision"
    ]
    assert context.get_status()["snapshot_active"] is None


@pytest.mark.asyncio
async def test_other_snapshot_kind_is_not_overwritten(tmp_path):
    agent = MultiTurnAgent({})
    context = await agent.start_conversation()
    memory = VectorMemory()
    await memory.add_entry("remember this")
    path = tmp_path / "memory.json"
    receipt = await memory.save_snapshot(path)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        await agent.save_conversation(
            context.conversation_id, path, expected_revision=receipt["revision"]
        )
    assert path.read_bytes() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before", "after"])
async def test_process_death_leaves_complete_conversation_and_releases_lock(
    tmp_path, phase
):
    agent, context, path, first = await saved(tmp_path)
    script = """
import asyncio, sys
from stateset_agents.core.multiturn_agent import MultiTurnAgent
from stateset_agents.utils import snapshot_files
original = snapshot_files.os.replace
def replace(src, dst):
    if sys.argv[3] == "after":
        original(src, dst)
    print("READY", flush=True)
    sys.stdin.read(1)
    if sys.argv[3] == "before":
        original(src, dst)
snapshot_files.os.replace = replace
async def main():
    agent = MultiTurnAgent({}, response_backend=lambda prompt: "new answer")
    context = await agent.restore_conversation(sys.argv[1])
    await agent.continue_conversation(context.conversation_id, "new question")
    await agent.save_conversation(context.conversation_id, sys.argv[1], expected_revision=sys.argv[2])
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
        with pytest.raises(ConversationConflict, match="writer"):
            await agent.save_conversation(
                context.conversation_id, path, expected_revision=first["revision"]
            )
    finally:
        process.kill()
        await asyncio.to_thread(process.communicate, timeout=10)
    destination = MultiTurnAgent({})
    recovered = await destination.restore_conversation(path)
    assert recovered.conversation_id == context.conversation_id
    assert len(recovered.history) == (2 if phase == "before" else 4)
    revision = (await destination.inspect_conversation_snapshot(path))["revision"]
    # A dead process releases the advisory lock; orphan stages are not adopted.
    receipt = await destination.save_conversation(
        recovered.conversation_id, path, expected_revision=revision
    )
    assert receipt["state"] == "completed"
