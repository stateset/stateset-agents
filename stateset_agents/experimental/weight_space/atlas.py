"""Weight-only atlas: read which weights correspond to which tokens, no forward passes.

The final hidden state reaches the logits through the last RMSNorm gain and
the unembedding, so the logit direction of token ``t`` is ``gain * E[t]``.
An MLP neuron ``j`` of layer ``L`` writes its output along column ``j`` of
that layer's ``down_proj``; the cosine between that column and a logit
direction says how directly the neuron votes for (or against) the token.

These are *candidates*, not causes: on Qwen3.5-0.8B every atlas neuron
edited alone was null or harmful on held-out math, while token-row edits of
the same tokens moved accuracy by tens of points (see ``benchmarks/weight_space``).
Use the atlas to choose what to test, then test it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    "NeuronScore",
    "logit_direction",
    "neuron_readout",
    "protected_write",
    "quiet_direction",
    "token_neighbours",
    "top_writers",
]


@dataclass(frozen=True)
class NeuronScore:
    """One MLP neuron and how closely its output column aligns with a direction."""

    layer: int
    neuron: int
    alignment: float


def _gain(norm_weight: torch.Tensor, zero_centered: bool) -> torch.Tensor:
    w = norm_weight.float()
    return 1.0 + w if zero_centered else w


def logit_direction(
    embed: torch.Tensor,
    ids: Sequence[int],
    norm_weight: torch.Tensor,
    *,
    zero_centered: bool = True,
) -> torch.Tensor:
    """Residual-stream direction that raises the logits of ``ids`` (mean over ids).

    ``zero_centered`` selects the RMSNorm convention: ``x * (1 + w)`` (Qwen3.5,
    Gemma) or ``x * w`` (Llama, Qwen3). ``embed`` is the unembedding matrix
    (the tied embedding, or ``lm_head``).
    """
    return embed[list(ids)].float().mean(0) * _gain(norm_weight, zero_centered)


def protected_write(
    embed: torch.Tensor,
    target: int,
    norm_weight: torch.Tensor,
    *,
    protect: Sequence[int],
    zero_centered: bool = True,
) -> torch.Tensor:
    """Unit residual write that moves ``target``'s logit and no ``protect`` logit.

    Through the final norm a write ``v`` changes token ``t``'s logit by
    ``(e_t * gain) . v`` over the shared ``rms(x)``. So ``v`` is ``target``'s
    readout with the protected tokens' readouts projected out: their logits
    move by exactly zero (to first order) while ``target``'s moves. A plain
    :func:`logit_direction` write also moves every token whose readout
    overlaps (on Qwen3.5-0.8B a ``</think>`` write moves ``<|im_end|>`` by 23%
    as much). Raises ValueError if ``target`` is protected or the protected
    readouts leave nothing of the target's.
    """
    import torch

    if target in set(protect):
        raise ValueError("the target token cannot also be protected")
    gain = _gain(norm_weight, zero_centered).double()
    r = embed[target].double() * gain
    v = r
    if protect:
        q = embed[list(protect)].double() * gain  # [k, hidden]
        coef = torch.linalg.lstsq(q.T, r.unsqueeze(1)).solution.squeeze(1)
        v = r - q.T @ coef
    if float(v.norm()) < 1e-6 * float(r.norm()):
        raise ValueError(
            f"protecting {len(protect)} tokens leaves no write that moves token {target}"
        )
    out: torch.Tensor = (v / v.norm()).float()
    return out


def quiet_direction(
    readout: torch.Tensor, residuals: torch.Tensor, *, k: int
) -> torch.Tensor:
    """A unit residual direction the unembedding barely sees and the model barely uses.

    ``readout`` is ``[vocab, hidden]`` (embedding times the final norm's gain);
    ``residuals`` ``[n, hidden]`` captured residual-stream vectors at the layer
    that will read the direction. Among the ``k`` directions with the smallest
    total logit effect (bottom eigenvectors of ``readout^T readout``), returns
    the one with the smallest second moment of ``residual / rms(residual)``. A
    designed unit can write a private signal there for a later unit to read,
    moving logits little and colliding little with what the residual already
    carries.
    """
    import torch

    hidden = readout.shape[1]
    if not 0 < k <= hidden:
        raise ValueError(f"k must be in 1..{hidden}, got {k}")
    r = readout.double()
    _, vecs = torch.linalg.eigh(r.T @ r)
    low = vecs[:, :k]
    x = residuals.double()
    xn = x / x.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
    _, sub = torch.linalg.eigh(low.T @ (xn.T @ xn / len(xn)) @ low)
    w = low @ sub[:, 0]
    out: torch.Tensor = (w / w.norm()).float()
    return out


def top_writers(
    down_proj: Mapping[int, torch.Tensor], direction: torch.Tensor, *, k: int = 10
) -> list[NeuronScore]:
    """The ``k`` neurons (over all layers) whose output columns best align with ``direction``.

    ``down_proj`` maps layer -> ``[hidden, intermediate]`` weight. Alignment is
    the cosine between each column and ``direction``; pass ``-direction`` for
    neurons that write against it.
    """
    import torch

    d = direction.float() / direction.float().norm().clamp_min(1e-12)
    scores: list[NeuronScore] = []
    for layer, weight in down_proj.items():
        cols = weight.float()
        cos = (d @ cols) / cols.norm(dim=0).clamp_min(1e-12)
        values, idx = torch.topk(cos, min(k, cos.numel()))
        scores += [
            NeuronScore(layer=int(layer), neuron=int(j), alignment=float(v))
            for v, j in zip(values.tolist(), idx.tolist(), strict=True)
        ]
    scores.sort(key=lambda s: s.alignment, reverse=True)
    return scores[:k]


def neuron_readout(
    column: torch.Tensor,
    embed: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    k: int = 10,
    zero_centered: bool = True,
) -> list[int]:
    """Token ids whose logits a neuron's output column raises the most."""
    logits = embed.float() @ (column.float() * _gain(norm_weight, zero_centered))
    return [int(i) for i in logits.topk(min(k, logits.numel())).indices.tolist()]


def token_neighbours(
    embed: torch.Tensor, ids: Sequence[int], *, k: int = 10
) -> list[int]:
    """Nearest tokens (cosine, embedding rows) to the mean of ``ids``, excluding ``ids``."""
    rows = embed.float()
    unit = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
    query = unit[list(ids)].mean(0)
    sims = unit @ query
    sims[list(ids)] = float("-inf")
    k = min(k, sims.numel() - len(set(ids)))
    return [int(i) for i in sims.topk(k).indices.tolist()]
