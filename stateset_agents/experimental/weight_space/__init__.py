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
- :class:`EditSpec` / :mod:`.checkpoints` — serializable token-row and
  neuron-column edits written into a checkpoint, verified tensor by tensor
  against the base, with structure-matched random controls
  (``stateset-agents weight-edit``).
- :mod:`.atlas` — weight-only reads: which MLP neurons write toward a
  token's logit direction and what each neuron promotes (candidates to test).
- :mod:`.protocol` — paired statistics (exact McNemar, bootstrap CI) for
  judging an edit against the base on identical problems.

Everything here imports ``torch`` lazily.
"""

from __future__ import annotations

from .atlas import (
    NeuronScore,
    logit_direction,
    neuron_readout,
    token_neighbours,
    top_writers,
)
from .deltas import LowRankDelta, direction_ablation
from .edits import (
    EditSpec,
    NeuronColumnScale,
    SpecMismatch,
    TensorLayout,
    TokenRowScale,
    apply_spec,
    matched_control,
    verify_spec,
)
from .protocol import (
    bootstrap_ci,
    clean_finish,
    mcnemar_exact,
    paired_compare,
    strict_rows,
)
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
    "EditSpec",
    "ForwardBudget",
    "LowRankDelta",
    "NeuronColumnScale",
    "NeuronScore",
    "SpecMismatch",
    "TensorLayout",
    "TokenRowScale",
    "WeightSpaceSession",
    "apply_spec",
    "bootstrap_ci",
    "clean_finish",
    "direction_ablation",
    "load_splits",
    "logit_direction",
    "matched_control",
    "mcnemar_exact",
    "neuron_readout",
    "paired_compare",
    "strict_rows",
    "token_neighbours",
    "top_writers",
    "verify_spec",
]
