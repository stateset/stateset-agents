"""Weight-space programming: an LLM edits weights directly, no optimizer.

Experimental (see :mod:`stateset_agents.experimental`). Inspired by
"Sparks of Weight-Space Programming" (Tsuchida et al.,
https://rikiyat.github.io/blog/weight-space-programming/): a model reads a
network's weights, makes targeted edits, checks them against a small budget
of forward passes, and keeps only what wins.

This package is the harness, not the programmer:

- :class:`WeightSpaceSession` — budgeted read/edit/measure loop with exact,
  LIFO-revertible low-rank edits and export to a PEFT LoRA adapter.
- :func:`direction_ablation` — the rank-1 "remove a direction from the
  columns that write it" edit.
- :mod:`.mcp_tools` — the same loop as MCP tools, so an MCP client such as
  Claude Code can act as the programmer
  (``stateset-agents mcp --weight-space``).

Everything here imports ``torch`` lazily.
"""

from __future__ import annotations

from .deltas import LowRankDelta, direction_ablation
from .session import (
    AppliedEdit,
    BudgetExhausted,
    ForwardBudget,
    WeightSpaceSession,
    load_splits,
)

__all__ = [
    "AppliedEdit",
    "BudgetExhausted",
    "ForwardBudget",
    "LowRankDelta",
    "WeightSpaceSession",
    "direction_ablation",
    "load_splits",
]
