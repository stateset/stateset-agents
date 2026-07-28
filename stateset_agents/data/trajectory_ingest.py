"""Ingest conversation trajectories logged by agents built elsewhere.

Lets users bring conversation logs from agents built with any framework —
raw OpenAI chat-completions message lists, or LangChain/LangGraph message
dumps — into stateset-agents' grade -> curate -> retrain loop without
rewriting their agent.

Supported source formats
-------------------------
* **OpenAI chat-completions** (:func:`from_openai_messages`,
  :func:`from_openai_jsonl`): a list of ``{"role", "content", ...}`` dicts,
  matching the shape returned by ``client.chat.completions.create`` /
  stored in ``messages=``. ``content`` may be a plain string or a list of
  multimodal parts (as used by the Responses/vision APIs); text parts are
  concatenated and non-text parts are recorded (but skipped) in the turn's
  metadata under ``skipped_parts``. ``tool_calls`` (on assistant turns) and
  ``tool_call_id`` (on tool turns) are preserved verbatim in
  ``ConversationTurn.tool_calls`` / ``ConversationTurn.metadata``.

* **LangChain message dumps** (:func:`from_langchain_json`): supports the
  two common shapes seen in the wild —

  1. Flat dicts with a ``"type"`` key: ``{"type": "human"|"ai"|"system"|
     "tool", "data": {"content": ..., "tool_calls": ..., ...}}`` or the
     data fields inlined directly on the object (some exporters omit the
     ``data`` wrapper).
  2. LangChain's ``dumpd``/``dumps`` serialized form:
     ``{"lc": 1, "type": "constructor", "id": [..., "HumanMessage"],
     "kwargs": {"content": ..., ...}}``. The message class is taken from
     the last element of ``id`` (``HumanMessage`` -> ``user``,
     ``AIMessage`` -> ``assistant``, ``SystemMessage`` -> ``system``,
     ``ToolMessage``/``FunctionMessage`` -> ``tool``).

  Other LangChain export shapes are not covered; contributions welcome.

Every loader accepts an optional per-conversation reward: if the source
object carries a top-level ``"reward"`` or ``"score"`` field (or a nested
``"metadata": {"reward": ...}``), it is attached to
``MultiTurnTrajectory.total_reward``. When
absent, the trajectory is left unscored for ``scripts/grade_transcript.py``
(or any other reward function) to fill in.

:func:`to_grading_history` is the other half of the round trip: it emits the
plain ``{"role", "content"}`` dicts that ``scripts/grade_transcript.py``'s
``load_transcript`` reads from a JSONL history file, so ingested logs plug
straight into the existing grade -> curate loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stateset_agents.core.trajectory import ConversationTurn, MultiTurnTrajectory

__all__ = [
    "from_openai_messages",
    "from_openai_jsonl",
    "from_langchain_json",
    "to_grading_history",
]


_LC_TYPE_TO_ROLE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "chat": "assistant",
    "generic": "assistant",
}

_LC_CLASS_TO_ROLE = {
    "HumanMessage": "user",
    "AIMessage": "assistant",
    "AIMessageChunk": "assistant",
    "SystemMessage": "system",
    "ToolMessage": "tool",
    "FunctionMessage": "tool",
    "ChatMessage": "assistant",
}


def _extract_reward(obj: dict[str, Any]) -> float | None:
    """Pull an optional per-conversation reward/score out of a source dict."""
    for key in ("reward", "score"):
        if key in obj and obj[key] is not None:
            try:
                return float(obj[key])
            except (TypeError, ValueError):
                pass
    metadata = obj.get("metadata")
    if isinstance(metadata, dict):
        for key in ("reward", "score"):
            if key in metadata and metadata[key] is not None:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError):
                    pass
    return None


def _normalize_content(content: Any) -> tuple[str, list[Any]]:
    """Normalize OpenAI-style ``content`` (str or list-of-parts) to text.

    Returns ``(text, skipped_parts)``. String content passes through
    unchanged. List content (multimodal parts, e.g.
    ``[{"type": "text", "text": "..."}, {"type": "image_url", ...}]``) has
    its text parts concatenated with newlines; non-text parts are returned
    in ``skipped_parts`` for the caller to stash in turn metadata.
    """
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if isinstance(content, list):
        text_pieces: list[str] = []
        skipped: list[Any] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in (
                "text",
                "input_text",
                "output_text",
            ):
                text_pieces.append(str(part.get("text", "")))
            elif isinstance(part, str):
                text_pieces.append(part)
            else:
                skipped.append(part)
        return "\n".join(text_pieces), skipped
    # Unknown scalar type — stringify rather than dropping data.
    return str(content), []


def from_openai_messages(
    messages: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
    reward: float | None = None,
) -> MultiTurnTrajectory:
    """Build a :class:`MultiTurnTrajectory` from OpenAI chat-completions messages.

    Args:
        messages: list of ``{"role", "content", ...}`` dicts, e.g. the
            ``messages`` list passed to/returned from
            ``client.chat.completions.create``.
        metadata: extra metadata merged onto the trajectory's ``metadata``.
        reward: optional total reward to attach; if not given, this is left
            unset (``None`` turn rewards -> distributed as 0.0) so the
            grading loop can fill it in.

    Raises:
        ValueError: if ``messages`` is empty or a message is missing
            ``"role"``.
    """
    if not messages:
        raise ValueError("from_openai_messages requires at least one message")

    turns: list[ConversationTurn] = []
    for i, msg in enumerate(messages):
        if "role" not in msg:
            raise ValueError(f"message at index {i} is missing 'role'")
        role = msg["role"]
        text, skipped = _normalize_content(msg.get("content"))

        turn_metadata: dict[str, Any] = {}
        if skipped:
            turn_metadata["skipped_parts"] = skipped
        if "name" in msg:
            turn_metadata["name"] = msg["name"]
        if "tool_call_id" in msg:
            turn_metadata["tool_call_id"] = msg["tool_call_id"]
        # Preserve any other unrecognized keys verbatim so nothing is lost.
        for key, value in msg.items():
            if key in ("role", "content", "tool_calls", "name", "tool_call_id"):
                continue
            turn_metadata[key] = value

        turns.append(
            ConversationTurn(
                role=role,
                content=text,
                metadata=turn_metadata,
                tool_calls=msg.get("tool_calls"),
            )
        )

    traj_metadata = dict(metadata or {})
    total_reward = reward if reward is not None else 0.0
    turn_rewards = None
    if reward is not None:
        # Score only the last turn — the conversation-level reward isn't
        # naturally divisible across turns; downstream training code that
        # cares about per-turn attribution should re-grade with a reward fn.
        turn_rewards = [0.0] * (len(turns) - 1) + [float(reward)]

    return MultiTurnTrajectory(
        turns=turns,
        total_reward=total_reward,
        turn_rewards=turn_rewards,
        metadata=traj_metadata,
    )


def from_openai_jsonl(path: str | Path) -> list[MultiTurnTrajectory]:
    """Load one conversation per line from a JSONL file.

    Each line is either ``{"messages": [...], "reward"?, "score"?,
    "metadata"?}`` or a bare list of messages ``[{"role": ..., ...}, ...]``.
    """
    path = Path(path)
    trajectories: list[MultiTurnTrajectory] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON — {exc}") from exc

            if isinstance(obj, list):
                messages = obj
                reward = None
                metadata: dict[str, Any] = {}
            elif isinstance(obj, dict):
                messages = obj.get("messages")
                if messages is None:
                    raise ValueError(
                        f"{path}:{lineno}: expected a 'messages' key or a bare list"
                    )
                reward = _extract_reward(obj)
                metadata = {
                    k: v
                    for k, v in obj.items()
                    if k not in ("messages", "reward", "score")
                }
            else:
                raise ValueError(f"{path}:{lineno}: expected a JSON object or array")

            trajectories.append(
                from_openai_messages(messages, metadata=metadata, reward=reward)
            )
    return trajectories


def _langchain_role_and_content(
    item: dict[str, Any],
) -> tuple[str, Any, dict[str, Any]]:
    """Resolve (role, content, kwargs) from either supported LangChain shape."""
    if item.get("lc") is not None and "kwargs" in item:
        # dumpd/dumps serialized form: {"lc": 1, "type": "constructor",
        # "id": [..., "HumanMessage"], "kwargs": {...}}
        class_name = ""
        id_path = item.get("id")
        if isinstance(id_path, list) and id_path:
            class_name = str(id_path[-1])
        role = _LC_CLASS_TO_ROLE.get(class_name)
        kwargs = item.get("kwargs") or {}
        if role is None:
            role = _LC_TYPE_TO_ROLE.get(
                str(kwargs.get("type", "")).lower(), "assistant"
            )
        return role, kwargs.get("content"), kwargs

    # Flat shape: {"type": "human"|"ai"|..., "data": {...}} or fields inlined.
    lc_type = str(item.get("type", "")).lower()
    role = _LC_TYPE_TO_ROLE.get(lc_type, lc_type or "assistant")
    data = item.get("data") if isinstance(item.get("data"), dict) else item
    return role, data.get("content"), data


def _langchain_messages_to_trajectory(
    messages: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None,
    reward: float | None,
) -> MultiTurnTrajectory:
    turns: list[ConversationTurn] = []
    for msg in messages:
        role, raw_content, kwargs = _langchain_role_and_content(msg)
        text, skipped = _normalize_content(raw_content)

        turn_metadata: dict[str, Any] = {}
        if skipped:
            turn_metadata["skipped_parts"] = skipped
        for key, value in kwargs.items():
            if key in ("content", "type"):
                continue
            turn_metadata[key] = value

        tool_calls = kwargs.get("tool_calls") or kwargs.get(
            "additional_kwargs", {}
        ).get("tool_calls")

        turns.append(
            ConversationTurn(
                role=role,
                content=text,
                metadata=turn_metadata,
                tool_calls=tool_calls,
            )
        )

    traj_metadata = dict(metadata or {})
    total_reward = reward if reward is not None else 0.0
    turn_rewards = None
    if reward is not None:
        turn_rewards = [0.0] * (len(turns) - 1) + [float(reward)]

    return MultiTurnTrajectory(
        turns=turns,
        total_reward=total_reward,
        turn_rewards=turn_rewards,
        metadata=traj_metadata,
    )


def from_langchain_json(
    obj_or_path: str | Path | dict[str, Any] | list[Any],
) -> list[MultiTurnTrajectory]:
    """Load LangChain/LangGraph message-dump conversations.

    Accepts a path to a JSON file, or an already-parsed object, in one of:

    * A bare list of messages -> single conversation.
    * A list of conversations, each a list of messages, or a dict
      ``{"messages": [...], "reward"?, "score"?, "metadata"?}``.
    * A single conversation dict ``{"messages": [...], ...}``.

    See the module docstring for the two supported per-message shapes.
    """
    if isinstance(obj_or_path, (str, Path)) and not isinstance(
        obj_or_path, (dict, list)
    ):
        path = Path(obj_or_path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = obj_or_path

    def _is_message_list(value: Any) -> bool:
        return isinstance(value, list) and all(isinstance(v, dict) for v in value)

    conversations: list[
        tuple[list[dict[str, Any]], dict[str, Any] | None, float | None]
    ] = []

    if isinstance(data, dict) and "messages" in data:
        messages = data["messages"]
        reward = _extract_reward(data)
        meta = {
            k: v for k, v in data.items() if k not in ("messages", "reward", "score")
        }
        conversations.append((messages, meta, reward))
    elif isinstance(data, list):
        if _is_message_list(data) and data and ("type" in data[0] or "lc" in data[0]):
            # Bare list of messages == single conversation.
            conversations.append((data, None, None))
        else:
            for item in data:
                if isinstance(item, dict) and "messages" in item:
                    messages = item["messages"]
                    reward = _extract_reward(item)
                    meta = {
                        k: v
                        for k, v in item.items()
                        if k not in ("messages", "reward", "score")
                    }
                    conversations.append((messages, meta, reward))
                elif isinstance(item, list):
                    conversations.append((item, None, None))
                else:
                    raise ValueError(
                        "from_langchain_json: expected each item to be a message list "
                        "or a {'messages': [...]} dict"
                    )
    else:
        raise ValueError(
            "from_langchain_json: expected a list of messages, a list of "
            "conversations, or a {'messages': [...]} dict"
        )

    return [
        _langchain_messages_to_trajectory(messages, metadata=meta, reward=reward)
        for messages, meta, reward in conversations
    ]


def to_grading_history(trajectory: MultiTurnTrajectory) -> list[dict[str, str]]:
    """Emit ``{"role", "content"}`` dicts for ``scripts/grade_transcript.py``.

    The result is exactly what ``load_transcript`` in that script expects:
    one dict per turn with plain string ``role``/``content`` (JSONL-ready —
    write each dict as its own line with ``json.dumps``).
    """
    return [
        {"role": turn.role or "", "content": turn.content or ""}
        for turn in trajectory.turns
    ]
