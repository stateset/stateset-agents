"""Portable checkpoints for the named-conversation API, without replaying work."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..utils.async_calls import run_sync_owned
from ..utils.snapshot_files import (
    canonical_json,
    json_value,
    open_regular,
    publish_bytes,
)
from .multiturn_context import ConversationConflict, ConversationContext, context_data

if TYPE_CHECKING:
    from .multiturn_agent import MultiTurnAgent


def _keys(value: Any, keys: set[str]) -> None:
    if type(value) is not dict or set(value) != keys:
        raise ValueError("Invalid conversation snapshot fields")


def _identifier(value: Any) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise ValueError("Snapshot revision must be a 32-character lowercase hex ID")


def _timestamp(value: Any) -> datetime:
    if type(value) is not str:
        raise ValueError("Snapshot timestamp must be an ISO datetime string")
    parsed = datetime.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("Snapshot timestamp must use canonical datetime.isoformat()")
    return parsed


def _text(value: Any, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if type(value) is not str or not value.strip():
        raise ValueError("Snapshot text fields must be nonempty strings")


def _validate_payload(payload: Any, version: int = 1) -> None:
    fields = {"conversation", "context_revision", "last_turn", "source_model"}
    if version == 2:
        fields.add("planning")
    _keys(payload, fields)
    if version == 2:
        from ..experimental.planning_snapshot import prepare_planning

        prepare_planning(payload["planning"])
    if type(payload["context_revision"]) is not int or payload["context_revision"] < 0:
        raise ValueError("Context revision must be a nonnegative integer")
    _text(payload["source_model"])
    context = payload["conversation"]
    _keys(context, set(ConversationContext.__dataclass_fields__))
    _text(context["conversation_id"])
    for key in ("user_id", "topic", "intent"):
        if context[key] is not None and type(context[key]) is not str:
            raise ValueError("Optional context fields must be strings or null")
    _timestamp(context["started_at"])
    if type(context["entities"]) is not dict or type(context["metadata"]) is not dict:
        raise ValueError("Entities and metadata must be JSON objects")
    if type(context["history"]) is not list:
        raise ValueError("Conversation history must be a list")
    for turn in context["history"]:
        if type(turn) is not dict or "content" not in turn:
            raise ValueError("Conversation turns require role and content fields")
        _text(turn.get("role"))
    json_value(context)
    receipt = payload["last_turn"]
    if receipt is None:
        return
    _keys(
        receipt,
        {
            "id",
            "state",
            "phase",
            "strategy",
            "user_recorded",
            "generation_status",
            "tool_status",
            "error_type",
            "cancellation_requested",
        },
    )
    _identifier(receipt["id"])
    _text(receipt["strategy"])
    if receipt["state"] not in ("completed", "failed", "cancelled"):
        raise ValueError("Only settled turn receipts can be restored")
    if receipt["phase"] not in (
        "preparing",
        "compressing",
        "analyzing",
        "generating",
        "tools",
        "committing",
        "finished",
    ):
        raise ValueError("Unknown conversation turn phase")
    if type(receipt["user_recorded"]) is not bool:
        raise ValueError("user_recorded must be boolean")
    if type(receipt["cancellation_requested"]) is not bool:
        raise ValueError("cancellation_requested must be boolean")
    for key in ("generation_status", "tool_status"):
        if receipt[key] not in ("not_started", "unconfirmed", "returned"):
            raise ValueError("Unknown conversation callback outcome")
    if receipt["state"] == "completed":
        if (
            receipt["phase"] != "finished"
            or not receipt["user_recorded"]
            or receipt["generation_status"] != "returned"
            or receipt["error_type"] is not None
            or receipt["cancellation_requested"]
        ):
            raise ValueError("Inconsistent completed turn receipt")
    else:
        _text(receipt["error_type"])


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate conversation snapshot JSON key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"Invalid JSON constant: {value}")


def _read(path: Path, max_bytes: int) -> dict[str, Any]:
    with os.fdopen(open_regular(path, os.O_RDONLY), "rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("Conversation snapshot exceeds max_bytes")
    document: dict[str, Any] = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_invalid_constant
    )
    _keys(
        document,
        {"schema_version", "kind", "revision", "created_at", "payload", "sha256"},
    )
    if type(document["schema_version"]) is not int or document[
        "schema_version"
    ] not in (1, 2):
        raise ValueError("Unsupported conversation snapshot schema version")
    if document["kind"] != "stateset_named_conversation":
        raise ValueError("Invalid conversation snapshot kind")
    _identifier(document["revision"])
    _timestamp(document["created_at"])
    _validate_payload(document["payload"], document["schema_version"])
    unsigned = {key: value for key, value in document.items() if key != "sha256"}
    if document["sha256"] != hashlib.sha256(canonical_json(unsigned)).hexdigest():
        raise ValueError("Conversation snapshot checksum mismatch")
    return document


def _arguments(max_bytes: int, expected_revision: str | None) -> None:
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if expected_revision is not None:
        _identifier(expected_revision)


def _ledger(agent: MultiTurnAgent) -> dict[str, Any]:
    if not hasattr(agent, "_conversation_snapshots"):
        agent._conversation_snapshots = {"active": {}, "last_finished": None}
    return agent._conversation_snapshots


def get_snapshot_status(agent: MultiTurnAgent) -> dict[str, Any]:
    """Return detached file-operation receipts for this named-conversation agent."""
    with agent._conversation_owner().lock:
        ledger = _ledger(agent)
        return copy.deepcopy(
            {
                "active": list(ledger["active"].values()),
                "last_finished": ledger["last_finished"],
            }
        )


def _admit(
    agent: MultiTurnAgent, path: Path, action: str, context: ConversationContext | None
) -> dict[str, Any]:
    if agent.planning_manager is not None:
        from ..experimental.long_term_planning import PlanningManager

        if type(agent.planning_manager) is not PlanningManager:
            raise ValueError("Snapshots require the built-in planning-manager type")
    operation = {
        "id": uuid4().hex,
        "action": action,
        "path": str(path),
        "state": "running",
        "phase": "capturing" if action == "save" else "reading",
        "conversation_id": context.conversation_id if context is not None else None,
        "revision": None,
        "observed_revision": None,
        "context_revision": None,
        "publication": "not_started",
        "file_synced": False,
        "directory_synced": False,
        "error_type": None,
        "work_error_type": None,
        "cleanup_error_type": None,
    }
    with agent._conversation_owner().lock:
        agent._conversation_owner().check()
        if context is not None:
            with context._owner().lock:
                if (
                    agent.active_conversations.get(context.conversation_id)
                    is not context
                    or context._owner().state != "active"
                ):
                    raise ConversationConflict("Conversation is no longer active")
                if (
                    context._owner().operation is not None
                    or context._owner().snapshot_operation is not None
                ):
                    raise ConversationConflict(
                        "Conversation has an active turn or snapshot operation"
                    )
                context._owner().snapshot_operation = operation
        _ledger(agent)["active"][operation["id"]] = operation
    return operation


def _update(
    agent: MultiTurnAgent,
    operation: dict[str, Any],
    context: ConversationContext | None,
    fields: dict[str, Any],
) -> None:
    with agent._conversation_owner().lock:
        if context is None:
            operation.update(fields)
        else:
            with context._owner().lock:
                operation.update(fields)


def _finish(
    agent: MultiTurnAgent,
    operation: dict[str, Any],
    context: ConversationContext | None,
    error: BaseException | None,
) -> dict[str, Any]:
    with agent._conversation_owner().lock:
        fields = {
            "state": (
                "completed"
                if error is None
                else (
                    "cancelled"
                    if isinstance(error, asyncio.CancelledError)
                    else (
                        "conflict"
                        if isinstance(error, ConversationConflict)
                        else "failed"
                    )
                )
            ),
            "error_type": type(error).__name__ if error is not None else None,
        }
        _update(agent, operation, context, fields)
        ledger = _ledger(agent)
        ledger["active"].pop(operation["id"])
        ledger["last_finished"] = copy.deepcopy(operation)
        if context is not None:
            with context._owner().lock:
                if context._owner().snapshot_operation is operation:
                    context._owner().snapshot_operation = None
                context._owner().last_snapshot = copy.deepcopy(operation)
        return copy.deepcopy(operation)


async def save_conversation(
    agent: MultiTurnAgent,
    conversation_id: str,
    path: str | os.PathLike[str],
    expected_revision: str | None,
    max_bytes: int,
) -> dict[str, Any]:
    """Capture a settled context once, then drain file publication on cancellation."""
    _arguments(max_bytes, expected_revision)
    target = Path(path).absolute()
    with agent._conversation_owner().lock:
        context = agent.active_conversations.get(conversation_id)
        if context is None:
            raise ValueError(f"Conversation {conversation_id} not found")
    operation = _admit(agent, target, "save", context)
    try:
        with agent._conversation_owner().lock, context._owner().lock:
            agent._conversation_owner().check()
            if (
                agent.active_conversations.get(conversation_id) is not context
                or context._owner().state != "active"
            ):
                raise ConversationConflict("Conversation ended before snapshot capture")
            if type(context) is not ConversationContext:
                raise ValueError(
                    "Snapshots require canonical ConversationContext objects"
                )
            if context._owner().operation is not None:
                raise ConversationConflict(
                    "Conversation turn started before snapshot capture"
                )
            data = context_data(context)
            if type(data["started_at"]) is not datetime:
                raise ValueError("Conversation started_at must be a datetime")
            data["started_at"] = data["started_at"].isoformat()
            payload = {
                "conversation": json_value(data),
                "context_revision": context._owner().revision,
                "last_turn": copy.deepcopy(context._owner().last_operation),
                "source_model": agent.config.model_name,
            }
            version = 1
            if agent.planning_manager is not None:
                from ..experimental.planning_snapshot import capture_planning

                payload["planning"] = capture_planning(
                    agent.planning_manager, conversation_id
                )
                version = 2
            _validate_payload(payload, version)
            operation["context_revision"] = payload["context_revision"]
        document = {
            "schema_version": version,
            "kind": "stateset_named_conversation",
            "revision": uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        document["sha256"] = hashlib.sha256(canonical_json(document)).hexdigest()
        raw = canonical_json(document)
        if len(raw) > max_bytes:
            raise ValueError("Conversation snapshot exceeds max_bytes")
        _update(agent, operation, context, {"revision": document["revision"]})

        def write() -> None:
            try:
                publish_bytes(
                    target,
                    raw,
                    expected_revision=expected_revision,
                    read_revision=lambda: _read(target, max_bytes)["revision"],
                    status=lambda fields: _update(agent, operation, context, fields),
                    conflict=ConversationConflict,
                )
            except BaseException as error:
                _update(
                    agent, operation, context, {"work_error_type": type(error).__name__}
                )
                raise

        await run_sync_owned(write)
        agent._conversation_owner().check()
    except BaseException as error:
        _finish(agent, operation, context, error)
        raise
    return _finish(agent, operation, context, None)


async def restore_conversation(
    agent: MultiTurnAgent,
    path: str | os.PathLike[str],
    expected_revision: str | None,
    max_bytes: int,
) -> ConversationContext:
    """Attach one restored context; never replace an existing ID or invoke a model."""
    _arguments(max_bytes, expected_revision)
    target = Path(path).absolute()
    operation = _admit(agent, target, "restore", None)
    manager = agent.planning_manager
    restored = None
    try:

        def read() -> dict[str, Any]:
            try:
                return _read(target, max_bytes)
            except BaseException as error:
                _update(
                    agent, operation, None, {"work_error_type": type(error).__name__}
                )
                raise

        document = await run_sync_owned(read)
        _update(
            agent,
            operation,
            None,
            {
                "revision": document["revision"],
                "observed_revision": document["revision"],
                "phase": "validating",
            },
        )
        if expected_revision is not None and expected_revision != document["revision"]:
            raise ConversationConflict(
                "Snapshot revision does not match the requested revision"
            )
        payload = document["payload"]
        data = payload["conversation"]
        data["started_at"] = _timestamp(data["started_at"])
        candidate = ConversationContext(**data)
        candidate._owner().revision = payload["context_revision"] + 1
        candidate._owner().last_operation = payload["last_turn"]
        planning = None
        if manager is not None:
            from ..experimental.planning_snapshot import prepare_planning

            planning = prepare_planning(
                payload.get(
                    "planning",
                    {
                        "revision": 0,
                        "turn_count": 0,
                        "last_update": 0,
                        "plan": None,
                        "last_operation": None,
                    },
                )
            )
        elif "planning" in payload:
            raise ValueError(
                "Restoring planning state requires a built-in planning manager"
            )
        with (
            agent._conversation_owner().lock,
            manager._lock if manager is not None else nullcontext(),
        ):
            agent._conversation_owner().check()
            if agent.planning_manager is not manager:
                raise ConversationConflict("Planning manager changed during restore")
            operation.update(
                phase="committing",
                conversation_id=candidate.conversation_id,
                context_revision=candidate._owner().revision,
            )
            if candidate.conversation_id in agent.active_conversations:
                raise ConversationConflict(
                    "Conversation ID already exists; restore into a fresh registry"
                )
            if manager is not None:
                from ..experimental.planning_snapshot import install_planning

                assert planning is not None
                install_planning(manager, candidate.conversation_id, planning)
            agent.active_conversations[candidate.conversation_id] = candidate
            restored = candidate
            operation.update(publication="published", phase="finished")
    except BaseException as error:
        _finish(agent, operation, None, error)
        raise
    _finish(agent, operation, restored, None)
    return restored


async def inspect_snapshot(
    path: str | os.PathLike[str], max_bytes: int
) -> dict[str, Any]:
    """Validate metadata without attaching a conversation or executing callbacks."""
    _arguments(max_bytes, None)
    document = await run_sync_owned(_read, Path(path).absolute(), max_bytes)
    payload = document["payload"]
    return {
        "schema_version": document["schema_version"],
        "has_planning": "planning" in payload,
        "revision": document["revision"],
        "created_at": document["created_at"],
        "conversation_id": payload["conversation"]["conversation_id"],
        "context_revision": payload["context_revision"],
        "turn_count": len(payload["conversation"]["history"]),
        "source_model": payload["source_model"],
        "last_turn_state": (
            payload["last_turn"]["state"] if payload["last_turn"] else None
        ),
    }
