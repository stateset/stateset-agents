"""Explicitly synthetic, offline refund fixtures for the private workflow tutorial."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from stateset_agents.core.environments.refund_delegation_environment import (
    RefundDelegationEnvironment,
)
from stateset_agents.core.environments.refund_policy_environment import (
    refund_policy_benchmark,
)
from stateset_agents.core.trajectory import ConversationTurn

from .datasets import prepare_dataset
from .storage import private_directory, write_bytes


def create_synthetic_fixture(root: Path, *, count: int = 32) -> Path:
    """Execute fictional teacher actions offline and prepare isolated temporal splits.

    This produces no customer evidence, provider calls, model samples or measured
    learning. The generated teacher transcripts stay in the private workspace.
    """
    if type(count) is not int or count < 8 or count % 8:
        raise ValueError("Fixture count must be a positive multiple of eight")
    root = private_directory(root)
    rows = asyncio.run(_rows(count))
    payload = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    )
    path = root / "synthetic-source.jsonl"
    write_bytes(path, payload.encode())
    destination = root / "dataset"
    prepare_dataset(
        path,
        destination,
        validation_after="2026-02-01T00:00:00Z",
        test_after="2026-03-01T00:00:00Z",
        synthetic=True,
    )
    return destination


async def _rows(count: int) -> list[dict[str, Any]]:
    rows = []
    for month, split in enumerate(("train", "validation", "test"), 1):
        for case in refund_policy_benchmark(split, count=count, seed=123):
            environment = RefundDelegationEnvironment()
            state = await environment.reset(case)
            messages = list(state.context["messages"])
            for action in (
                {"tool": "resolve_refund", "args": {"intent": "resolve_refund"}},
                {"tool": "finish", "args": {}},
            ):
                text = json.dumps(action)
                state, _, _, info = await environment.step(
                    state, ConversationTurn(role="assistant", content=text)
                )
                messages.append({"role": "assistant", "content": text})
                messages.extend(info["messages"])
            if info["metrics"]["task_success"] != 1:
                raise ValueError("Synthetic teacher failed its policy fixture")
            rows.append(
                {
                    "record_id": case["order_id"],
                    "source_id": f'synthetic-{split}-{case["order_id"]}',
                    "source_kind": "synthetic",
                    "timestamp": f"2026-0{month}-01T00:00:00Z",
                    "messages": messages,
                    "evaluation_case": case,
                    "reward": 1.0,
                }
            )
    return rows
