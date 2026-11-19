"""Conversation and planning snapshots restore together without executing planners."""

import asyncio
import copy
import hashlib
import json
import subprocess
import sys
import threading

import pytest

from stateset_agents.core import conversation_snapshot as snapshots
from stateset_agents.core.multiturn_agent import ConversationConflict, MultiTurnAgent
from stateset_agents.experimental.long_term_planning import (
    HeuristicPlanner,
    PlanningConfig,
    PlanningManager,
)
from stateset_agents.utils.snapshot_files import canonical_json


def make_agent(planning=None):
    return MultiTurnAgent(
        {},
        planning_manager=planning or PlanningManager(PlanningConfig(enabled=True)),
        response_backend=lambda prompt: "ok",
    )


async def saved(tmp_path):
    agent = make_agent()
    context = await agent.start_conversation()
    await agent.continue_conversation(
        context.conversation_id, "Draft report, review it, then send it"
    )
    await agent.continue_conversation(
        context.conversation_id,
        "draft done",
        context_update={"plan_update": {"action": "advance"}},
    )
    agent.planning_manager.apply_update(
        context.conversation_id,
        {"step_id": "step_2", "status": "blocked", "notes": "Waiting for review"},
    )
    agent.planning_manager.get_plan(context.conversation_id).metadata["nested"] = [
        "retained"
    ]
    path = tmp_path / "planned.json"
    receipt = await agent.save_conversation(context.conversation_id, path)
    return agent, context, path, receipt


@pytest.mark.asyncio
async def test_roundtrip_restores_plan_steps_and_counters_before_next_prompt(tmp_path):
    source, context, path, receipt = await saved(tmp_path)

    class NoReplay(HeuristicPlanner):
        def create_plan(self, *args):
            raise AssertionError("Restore/continuation must reuse the plan")

        def revise_plan(self, *args):
            raise AssertionError("A consumed advance must not trigger revision")

    planning = PlanningManager(
        PlanningConfig(enabled=True, update_interval=7, plan_prefix="Recovered plan"),
        planner=NoReplay(),
    )
    destination = make_agent(planning)
    restored = await destination.restore_conversation(
        path, expected_revision=receipt["revision"]
    )
    plan = planning.get_plan(restored.conversation_id)
    assert restored == context
    assert plan == source.planning_manager.get_plan(context.conversation_id)
    assert planning.get_status(restored.conversation_id)["turn_count"] == 2
    assert planning.get_status(restored.conversation_id)["last_update"] == 2
    assert planning.config.update_interval == 7
    prompts = []
    destination.response_backend = lambda prompt: prompts.append(prompt) or "continued"
    await destination.continue_conversation(restored.conversation_id, "status please")
    assert "Recovered plan" in prompts[0]
    assert "[x] Draft report" in prompts[0]
    assert "[!] review it" in prompts[0]
    assert planning.get_plan(restored.conversation_id).progress() == pytest.approx(
        1 / 3
    )
    plan.metadata["nested"].append("local mutation")
    assert source.planning_manager.get_plan(context.conversation_id).metadata[
        "nested"
    ] == ["retained"]
    metadata = await destination.inspect_conversation_snapshot(path)
    assert metadata["schema_version"] == 2
    assert metadata["has_planning"] is True


@pytest.mark.asyncio
async def test_fresh_process_recovers_plan_and_uses_it_without_callbacks(tmp_path):
    _, context, path, _ = await saved(tmp_path)
    script = """
import asyncio, json, sys
from stateset_agents.core.multiturn_agent import MultiTurnAgent
from stateset_agents.experimental.long_term_planning import PlanningManager, PlanningConfig
class NoReplay:
    def create_plan(self, *args): raise AssertionError("replayed")
    def revise_plan(self, *args): raise AssertionError("replayed")
async def main():
    prompts = []
    manager = PlanningManager(PlanningConfig(enabled=True), planner=NoReplay())
    agent = MultiTurnAgent({}, planning_manager=manager, response_backend=lambda prompt: prompts.append(prompt) or "ok")
    context = await agent.restore_conversation(sys.argv[1])
    await agent.continue_conversation(context.conversation_id, "status")
    print(json.dumps({"id": context.conversation_id, "prompt": prompts[0], "progress": manager.get_plan(context.conversation_id).progress()}))
asyncio.run(main())
"""
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", script, str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    data = json.loads(result.stdout)
    assert data["id"] == context.conversation_id
    assert data["progress"] == pytest.approx(1 / 3)
    assert "[!] review it" in data["prompt"]


@pytest.mark.asyncio
async def test_legacy_snapshot_restores_into_planning_agent_without_inventing_plan(
    tmp_path,
):
    source = MultiTurnAgent({})
    context = await source.start_conversation()
    path = tmp_path / "legacy.json"
    await source.save_conversation(context.conversation_id, path)
    assert json.loads(path.read_text())["schema_version"] == 1
    destination = make_agent()
    restored = await destination.restore_conversation(path)
    assert restored == context
    assert destination.planning_manager.get_plan(context.conversation_id) is None


@pytest.mark.asyncio
async def test_plan_cannot_be_discarded_or_overwrite_orphaned_destination_state(
    tmp_path,
):
    _, context, path, _ = await saved(tmp_path)
    without_planning = MultiTurnAgent({})
    with pytest.raises(ValueError, match="requires a built-in planning manager"):
        await without_planning.restore_conversation(path)
    assert without_planning.active_conversations == {}
    destination = make_agent()
    manager = destination.planning_manager
    manager.build_plan_message(
        [],
        {
            "conversation_id": context.conversation_id,
            "goal": "Existing destination plan",
        },
    )
    original = copy.deepcopy(manager.get_plan(context.conversation_id))
    with pytest.raises(ConversationConflict, match="Planning state already exists"):
        await destination.restore_conversation(path)
    assert destination.active_conversations == {}
    assert manager.get_plan(context.conversation_id) == original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(turn_count=True),
        lambda p: p.update(last_update=999),
        lambda p: p.update(extra="unknown"),
        lambda p: p["plan"].update(updated_at="2025-01-02"),
        lambda p: p["plan"]["steps"][0].update(status="invented"),
        lambda p: p["plan"]["steps"][0].update(notes={"not": "text"}),
        lambda p: p["last_operation"].update(state="running"),
        lambda p: p["last_operation"].update(callback_status="success"),
    ],
)
async def test_invalid_planning_snapshot_never_attaches_either_owner(
    tmp_path, mutation
):
    _, _, path, _ = await saved(tmp_path)
    document = json.loads(path.read_text())
    mutation(document["payload"]["planning"])
    document.pop("sha256")
    document["sha256"] = hashlib.sha256(canonical_json(document)).hexdigest()
    path.write_bytes(canonical_json(document))
    destination = make_agent()
    with pytest.raises(ValueError):
        await destination.restore_conversation(path)
    assert destination.active_conversations == {}
    assert destination.planning_manager._plans == {}


@pytest.mark.asyncio
async def test_failed_planner_command_is_consumed_and_receipt_restores_without_replay(
    tmp_path,
):
    source = make_agent()
    context = await source.start_conversation()
    await source.continue_conversation(
        context.conversation_id, "Draft report and send it"
    )

    class Fails(HeuristicPlanner):
        def revise_plan(self, *args):
            raise OSError("unknown planner outcome")

    source.planning_manager.planner = Fails()
    with pytest.raises(OSError):
        await source.continue_conversation(
            context.conversation_id,
            "switch",
            context_update={"plan_goal": "New report goal"},
        )
    assert "plan_goal" not in context.metadata
    path = tmp_path / "failed.json"
    await source.save_conversation(context.conversation_id, path)
    destination = make_agent()
    destination.planning_manager.planner = Fails()
    restored = await destination.restore_conversation(path)
    receipt = destination.planning_manager.get_status(restored.conversation_id)[
        "last_finished"
    ]
    assert receipt["state"] == "failed"
    assert receipt["callback_status"] == "unconfirmed"
    await destination.continue_conversation(restored.conversation_id, "status")
    assert (
        destination.planning_manager.get_plan(restored.conversation_id).goal
        == "Draft report and send it"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["cancel", "replace_manager"])
async def test_interrupted_read_cannot_attach_a_partial_planning_session(
    tmp_path, monkeypatch, interrupt
):
    _, _, path, _ = await saved(tmp_path)
    destination = make_agent()
    manager = destination.planning_manager
    entered, release = threading.Event(), threading.Event()
    read = snapshots._read

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return read(*args)

    monkeypatch.setattr(snapshots, "_read", blocked)
    task = asyncio.create_task(destination.restore_conversation(path))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        if interrupt == "cancel":
            task.cancel()
        else:
            destination.planning_manager = PlanningManager(PlanningConfig(enabled=True))
    finally:
        release.set()
    with pytest.raises(
        asyncio.CancelledError if interrupt == "cancel" else ConversationConflict
    ):
        await task
    assert destination.active_conversations == {}
    assert manager._plans == destination.planning_manager._plans == {}


@pytest.mark.asyncio
async def test_legacy_file_can_be_explicitly_upgraded_with_its_revision(tmp_path):
    source = MultiTurnAgent({})
    context = await source.start_conversation()
    path = tmp_path / "upgrade.json"
    first = await source.save_conversation(context.conversation_id, path)
    destination = make_agent()
    recovered = await destination.restore_conversation(path)
    await destination.continue_conversation(
        recovered.conversation_id, "Draft report and send it"
    )
    second = await destination.save_conversation(
        recovered.conversation_id, path, expected_revision=first["revision"]
    )
    assert second["revision"] != first["revision"]
    assert json.loads(path.read_text())["schema_version"] == 2
    restored_again = make_agent()
    await restored_again.restore_conversation(path)
    assert restored_again.planning_manager.get_plan(
        context.conversation_id
    ) == destination.planning_manager.get_plan(context.conversation_id)


@pytest.mark.asyncio
async def test_active_standalone_planner_cannot_be_snapshotted(tmp_path):
    source = make_agent()
    context = await source.start_conversation()
    entered, release = threading.Event(), threading.Event()

    class Blocked(HeuristicPlanner):
        def create_plan(self, goal, config):
            entered.set()
            assert release.wait(5)
            return super().create_plan(goal, config)

    manager = source.planning_manager
    manager.planner = Blocked()
    task = asyncio.create_task(
        asyncio.to_thread(
            manager.build_plan_message,
            [],
            {
                "conversation_id": context.conversation_id,
                "plan_goal": "An independent planning operation",
            },
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        with pytest.raises(ConversationConflict, match="still active"):
            await source.save_conversation(
                context.conversation_id, tmp_path / "active.json"
            )
    finally:
        release.set()
        await task
    assert not list(tmp_path.iterdir())
    assert context.get_status()["snapshot_active"] is None


@pytest.mark.asyncio
async def test_nonportable_plan_state_fails_before_file_publication(tmp_path):
    source, context, _, _ = await saved(tmp_path)
    source.planning_manager.get_plan(context.conversation_id).metadata[
        "unsupported"
    ] = object()
    path = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="JSON"):
        await source.save_conversation(context.conversation_id, path)
    assert not path.exists()
