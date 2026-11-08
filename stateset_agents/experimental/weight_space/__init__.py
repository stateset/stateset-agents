"""Weight-space programming: an LLM edits weights directly, no optimizer.

Experimental (see :mod:`stateset_agents.experimental`). Inspired by
"Sparks of Weight-Space Programming" (Takehi et al.,
https://rikiyat.github.io/blog/weight-space-programming/): a model reads a
network's weights, makes targeted edits, checks them against a small budget
of forward passes, and keeps only what wins.

This package is the harness, not the programmer:

- :class:`WeightSpaceSession` — budgeted read/edit/measure loop with exact,
  LIFO-revertible low-rank edits and export to a PEFT LoRA adapter.
- :func:`direction_ablation` — the rank-1 "remove a direction from the
  columns that write it" edit.
- :func:`constrained_edit` — a minimum-change dense linear edit satisfying
  target activation mappings while preserving a declared reference span,
  with explicit numerical validation and optional update-norm limits.
- :mod:`.mcp_tools` — the same loop as MCP tools, so an MCP client such as
  Claude Code can act as the programmer
  (``stateset-agents mcp --weight-space``).
- :class:`EditSpec` / :mod:`.checkpoints` — serializable token-row and
  neuron-column edits written into a checkpoint, verified tensor by tensor
  against the base, with structure-matched random controls
  (``stateset-agents weight-edit``).
- :mod:`.atlas` — weight-only reads: which MLP neurons write toward a
  token's logit direction and what each neuron promotes (candidates to test).
- :mod:`.circuits` — design a conditional MLP neuron ("when X, write Y")
  from captured activations: ridge detector as the gate, a near-constant
  unit as the multiplier, the least-used neuron as the slot; emitted as a
  :class:`NeuronRewrite` edit.
- :mod:`.protocol` — paired statistics (exact McNemar, bootstrap CI) for
  judging an edit against the base on identical problems, strict scoring,
  and token-budget curves from one fixed-seed run.
- :mod:`.handwritten_decoder` — a complete 100,616-parameter ReLU network
  whose weights implement error correction directly, with no fitting.
- :mod:`.handwritten_graph` — a 114,984-parameter recurrent ReLU network
  whose handwritten weights compute exact directed shortest-path distances.
- :mod:`.handwritten_weighted_graph` — an exactly 2,000,000-parameter
  Linear/ReLU network for minimum costs on weighted seventeen-node graphs;
  all weights are analytic, with active neuron splitting to match the budget.
- :mod:`.handwritten_routes` — an analytic Linear/ReLU network that composes
  distances, deterministic next-hop selection, and traversal into full routes.
- :mod:`.relu_program` — compile supplied affine-plus-hinge arithmetic into an
  existing ReLU MLP, with an explicit input domain and scalar edit ledger.
  An explicit output index preserves other outputs by using only hidden rows
  outside their dependency paths; insufficient disconnected capacity is rejected.

Everything here imports ``torch`` lazily.
"""

from __future__ import annotations

from .atlas import (
    NeuronScore,
    logit_direction,
    neuron_readout,
    protected_write,
    quiet_direction,
    token_neighbours,
    top_writers,
)
from .circuits import (
    ConditionalNeuron,
    auc,
    design_conditional_neuron,
    least_used_neuron,
    lookback_query,
    near_constant_unit,
    rewrite_activation,
    ridge_detector,
    shift_threshold,
    silence_on_flag,
)
from .constrained import ConstrainedEdit, EditDiagnostics, constrained_edit
from .deltas import LowRankDelta, direction_ablation
from .edits import (
    EditSpec,
    HeadRewrite,
    NeuronColumnScale,
    NeuronRewrite,
    SpecMismatch,
    TensorLayout,
    TokenRowScale,
    apply_spec,
    matched_control,
    verify_spec,
)
from .protocol import (
    at_budget,
    bootstrap_ci,
    budget_curve,
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
    "ConditionalNeuron",
    "ConstrainedEdit",
    "EditDiagnostics",
    "EditSpec",
    "ForwardBudget",
    "HeadRewrite",
    "LowRankDelta",
    "NeuronColumnScale",
    "NeuronRewrite",
    "NeuronScore",
    "SpecMismatch",
    "TensorLayout",
    "TokenRowScale",
    "WeightSpaceSession",
    "apply_spec",
    "at_budget",
    "auc",
    "bootstrap_ci",
    "budget_curve",
    "clean_finish",
    "constrained_edit",
    "design_conditional_neuron",
    "direction_ablation",
    "least_used_neuron",
    "load_splits",
    "logit_direction",
    "lookback_query",
    "matched_control",
    "mcnemar_exact",
    "near_constant_unit",
    "neuron_readout",
    "paired_compare",
    "protected_write",
    "quiet_direction",
    "rewrite_activation",
    "ridge_detector",
    "shift_threshold",
    "silence_on_flag",
    "strict_rows",
    "token_neighbours",
    "top_writers",
    "verify_spec",
]
