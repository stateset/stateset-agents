"""Portable built-in planning state; runtime callbacks remain destination-owned."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, cast

from stateset_agents.core.multiturn_context import ConversationConflict
from stateset_agents.utils.snapshot_files import json_value

from .long_term_planning import Plan, PlanningManager, PlanStatus, PlanStep


def _fields(value: Any, names: set[str]) -> None:
    if type(value) is not dict or set(value) != names:
        raise ValueError("Invalid planning snapshot fields")


def _string(value: Any, optional: bool = False) -> None:
    if optional and value is None:
        return
    if type(value) is not str:
        raise ValueError("Planning snapshot text must be a string")


def _time(value: Any) -> datetime:
    if type(value) is not str:
        raise ValueError("Planning timestamp must be an ISO string")
    result = datetime.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError("Planning timestamp must use datetime.isoformat()")
    return result


def _encode_time(value: Any) -> str:
    if type(value) is not datetime:
        raise ValueError("Planning timestamps must be canonical datetimes")
    return value.isoformat()


def prepare_planning(data: Any) -> dict[str, Any]:
    """Validate portable state and prepare detached runtime values without effects."""
    data = json_value(data)
    _fields(data, {"revision", "turn_count", "last_update", "plan", "last_operation"})
    for key in ("revision", "turn_count", "last_update"):
        if type(data[key]) is not int or data[key] < 0:
            raise ValueError("Planning counters must be nonnegative integers")
    if data["last_update"] > data["turn_count"]:
        raise ValueError("Planning update counter exceeds turn count")
    receipt = data["last_operation"]
    if receipt is not None:
        _fields(receipt, {"id", "state", "phase", "callback_status", "error_type"})
        if (
            type(receipt["id"]) is not str
            or re.fullmatch(r"[0-9a-f]{32}", receipt["id"]) is None
        ):
            raise ValueError("Invalid planning operation identifier")
        if receipt["state"] not in ("completed", "failed", "cancelled"):
            raise ValueError("Planning snapshots require settled operations")
        if receipt["phase"] not in ("preparing", "planning", "committing", "finished"):
            raise ValueError("Unknown planning operation phase")
        if receipt["callback_status"] not in ("not_started", "unconfirmed", "returned"):
            raise ValueError("Unknown planning callback status")
        if receipt["state"] == "completed":
            if receipt["phase"] != "finished" or receipt["error_type"] is not None:
                raise ValueError("Invalid completed planning operation")
        elif type(receipt["error_type"]) is not str or not receipt["error_type"]:
            raise ValueError("Failed planning operations require an error type")
    plan = data["plan"]
    if plan is not None:
        _fields(plan, {"goal", "steps", "created_at", "updated_at", "metadata"})
        _string(plan["goal"])
        if type(plan["steps"]) is not list or type(plan["metadata"]) is not dict:
            raise ValueError("Invalid planning steps or metadata")
        steps = []
        for step in plan["steps"]:
            _fields(step, {"description", "status", "step_id", "notes", "updated_at"})
            for key in ("description", "status", "step_id"):
                _string(step[key])
            _string(step["notes"], optional=True)
            steps.append(
                PlanStep(
                    description=step["description"],
                    status=PlanStatus(step["status"]),
                    step_id=step["step_id"],
                    notes=step["notes"],
                    updated_at=_time(step["updated_at"]),
                )
            )
        data["plan"] = Plan(
            goal=plan["goal"],
            steps=steps,
            created_at=_time(plan["created_at"]),
            updated_at=_time(plan["updated_at"]),
            metadata=plan["metadata"],
        )
    return cast(dict[str, Any], data)


def capture_planning(manager: PlanningManager, conversation_id: str) -> dict[str, Any]:
    """Capture settled canonical planning data under its manager lock."""
    with manager._lock:
        if conversation_id in manager._operations:
            raise ConversationConflict("Planning operation is still active")
        plan = manager._plans.get(conversation_id)
        encoded = None
        if plan is not None:
            if type(plan) is not Plan or any(
                type(step) is not PlanStep for step in plan.steps
            ):
                raise ValueError(
                    "Snapshots require canonical Plan and PlanStep objects"
                )
            if any(type(step.status) is not PlanStatus for step in plan.steps):
                raise ValueError("Snapshots require canonical PlanStatus values")
            encoded = {
                "goal": plan.goal,
                "created_at": _encode_time(plan.created_at),
                "updated_at": _encode_time(plan.updated_at),
                "metadata": plan.metadata,
                "steps": [
                    {
                        "description": step.description,
                        "status": step.status.value,
                        "step_id": step.step_id,
                        "notes": step.notes,
                        "updated_at": _encode_time(step.updated_at),
                    }
                    for step in plan.steps
                ],
            }
        data: dict[str, Any] = json_value(
            {
                "revision": manager._revisions.get(conversation_id, 0),
                "turn_count": manager._turn_counters.get(conversation_id, 0),
                "last_update": manager._last_updates.get(conversation_id, 0),
                "plan": encoded,
                "last_operation": manager._last_operations.get(conversation_id),
            }
        )
        prepare_planning(data)
        return data


def install_planning(
    manager: PlanningManager, conversation_id: str, prepared: dict[str, Any]
) -> None:
    """Attach prepared state to an empty ID; caller holds the manager lock."""
    if any(
        conversation_id in registry
        for registry in (
            manager._plans,
            manager._turn_counters,
            manager._last_updates,
            manager._last_operations,
            manager._operations,
        )
    ):
        raise ConversationConflict(
            "Planning state already exists for this conversation"
        )
    if prepared["plan"] is not None:
        manager._plans[conversation_id] = prepared["plan"]
    manager._turn_counters[conversation_id] = prepared["turn_count"]
    manager._last_updates[conversation_id] = prepared["last_update"]
    manager._revisions[conversation_id] = (
        max(manager._revisions.get(conversation_id, 0), prepared["revision"]) + 1
    )
    if prepared["last_operation"] is not None:
        manager._last_operations[conversation_id] = prepared["last_operation"]
