"""Budget-enforced MCP tools that let an LLM act as the weight programmer.

Registered on the StateSet Agents MCP server only when it is started with
``stateset-agents mcp --weight-space``. Tools are plain functions over a
process-local session registry, so they are testable without an MCP client.

The tool surface is deliberately structured rather than a free Python
kernel: every way of running the network goes through the session, so the
forward-pass budget cannot be bypassed. Reading module shapes, editing and
reverting are free; capturing activations and evaluating spend budget.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .deltas import direction_ablation
from .session import BudgetExhausted, WeightSpaceSession, load_splits

if TYPE_CHECKING:
    import torch

__all__ = [
    "TOOLS",
    "register_session",
    "ws_ablate",
    "ws_capture_direction",
    "ws_close",
    "ws_evaluate",
    "ws_export",
    "ws_modules",
    "ws_open",
    "ws_revert",
]


@dataclass
class _Handle:
    session: WeightSpaceSession
    directions: dict[str, torch.Tensor] = field(default_factory=dict)


_SESSIONS: dict[str, _Handle] = {}


class _ToolError(Exception):
    """A caller mistake reported back to the LLM as ``{"error": ...}``."""


def _err(message: str) -> dict[str, Any]:
    return {"error": message}


def _handle(session_id: str) -> _Handle:
    try:
        return _SESSIONS[session_id]
    except KeyError:
        raise _ToolError(f"unknown session '{session_id}'") from None


def register_session(session: WeightSpaceSession) -> str:
    """Track ``session`` and return its id."""
    session_id = uuid.uuid4().hex[:12]
    _SESSIONS[session_id] = _Handle(session=session)
    return session_id


def ws_open(
    model_id: str, budget: int, splits_path: str, dtype: str = "float32"
) -> dict[str, Any]:
    """Load ``model_id`` for weight-space editing with a forward-pass ``budget``.

    ``splits_path`` is a JSON file ``{split: [{"prompt", "continuation"}, ...]}``
    of PRACTICE data; ``ws_evaluate`` reports the mean per-token log-prob of
    each split's continuations. Keep held-out data out of the session.
    """
    try:
        session = WeightSpaceSession.from_pretrained(
            model_id, budget=budget, splits=load_splits(splits_path), dtype=dtype
        )
    except (OSError, ValueError) as exc:
        return _err(f"{type(exc).__name__}: {exc}")
    session_id = register_session(session)
    return {
        "session_id": session_id,
        "base_model": model_id,
        "splits": sorted(session.splits),
        "linear_modules": len(session.linear_modules()),
        "budget": session.budget.as_dict(),
    }


def ws_modules(session_id: str, contains: str | None = None) -> dict[str, Any]:
    """List editable Linear modules as ``name -> [out, in]`` (free)."""
    try:
        session = _handle(session_id).session
    except _ToolError as exc:
        return _err(str(exc))
    return {"modules": session.linear_modules(contains)}


def ws_capture_direction(
    session_id: str,
    module: str,
    positive_prompts: list[str],
    negative_prompts: list[str],
    name: str,
    position: Literal["last", "mean"] = "last",
    top_k: int = 8,
) -> dict[str, Any]:
    """Store ``mean(out | positive) - mean(out | negative)`` at ``module`` as ``name``.

    Spends ONE forward pass for both prompt sets. Returns the direction's
    norm, its cosine with previously stored directions, and the ``top_k``
    input columns of ``module`` that write most along it.
    """
    import torch

    try:
        handle = _handle(session_id)
        if not positive_prompts or not negative_prompts:
            raise _ToolError("need at least one positive and one negative prompt")
        outs = handle.session.capture(
            module,
            [*positive_prompts, *negative_prompts],
            position=position,
            reason=f"capture:{name}",
        )
    except (_ToolError, BudgetExhausted, ValueError, AttributeError) as exc:
        return _err(str(exc))

    n_pos = len(positive_prompts)
    direction = outs[:n_pos].mean(0) - outs[n_pos:].mean(0)
    unit = direction / direction.norm().clamp_min(1e-12)
    cosines = {
        other: round(float(torch.nn.functional.cosine_similarity(unit, d, dim=0)), 4)
        for other, d in handle.directions.items()
    }
    handle.directions[name] = direction
    writes = unit @ handle.session.weight(module)
    top = writes.abs().topk(min(top_k, writes.shape[0])).indices
    return {
        "direction": name,
        "module": module,
        "norm": round(float(direction.norm()), 6),
        "cosine_with": cosines,
        "top_columns": [int(i) for i in top],
        "budget": handle.session.budget.as_dict(),
    }


def ws_ablate(
    session_id: str,
    module: str,
    direction: str,
    k: int,
    alpha: float = 1.0,
    note: str = "",
) -> dict[str, Any]:
    """Remove stored ``direction`` from the ``k`` columns of ``module`` writing most along it (free)."""
    try:
        handle = _handle(session_id)
        if direction not in handle.directions:
            raise _ToolError(
                f"unknown direction '{direction}' (stored: {sorted(handle.directions)})"
            )
        delta = direction_ablation(
            handle.session.weight(module),
            handle.directions[direction],
            k=k,
            alpha=alpha,
        )
        edit_id = handle.session.apply(
            module, delta, note=note or f"ablate {direction} k={k} alpha={alpha}"
        )
    except (_ToolError, ValueError, AttributeError) as exc:
        return _err(str(exc))
    return {
        "edit_id": edit_id,
        "module": module,
        "rank": delta.rank,
        "changed_weights": int((delta.dense() != 0).sum()),
        "active_edits": handle.session.active_edit_ids,
    }


def ws_evaluate(session_id: str) -> dict[str, Any]:
    """Score the practice splits with the current edits (spends ONE forward pass)."""
    try:
        session = _handle(session_id).session
        metrics = session.evaluate()
    except (_ToolError, BudgetExhausted, ValueError) as exc:
        return _err(str(exc))
    baseline = next(
        (e["metrics"] for e in session.evaluations if not e["active_edits"]), None
    )
    return {
        "metrics": metrics,
        "baseline": baseline,
        "delta": (
            {k: round(metrics[k] - baseline[k], 6) for k in metrics}
            if baseline
            else None
        ),
        "active_edits": session.active_edit_ids,
        "budget": session.budget.as_dict(),
    }


def ws_revert(session_id: str) -> dict[str, Any]:
    """Undo the most recent edit exactly (free)."""
    try:
        session = _handle(session_id).session
        edit = session.revert()
    except (_ToolError, IndexError, ValueError) as exc:
        return _err(str(exc))
    return {
        "reverted": edit.edit_id,
        "module": edit.module,
        "active_edits": session.active_edit_ids,
    }


def ws_export(session_id: str, output_dir: str) -> dict[str, Any]:
    """Write the active edits as a PEFT LoRA adapter + provenance manifest."""
    try:
        session = _handle(session_id).session
        path = session.export(output_dir)
    except (_ToolError, ValueError, OSError) as exc:
        return _err(str(exc))
    return {
        "path": str(path),
        "edits": len(session.active_edit_ids),
        "budget": session.budget.as_dict(),
    }


def ws_close(session_id: str) -> dict[str, Any]:
    """Drop the session and free its model."""
    if _SESSIONS.pop(session_id, None) is None:
        return _err(f"unknown session '{session_id}'")
    return {"closed": session_id}


#: Registration order for the MCP server.
TOOLS = (
    ws_open,
    ws_modules,
    ws_capture_direction,
    ws_ablate,
    ws_evaluate,
    ws_revert,
    ws_export,
    ws_close,
)
