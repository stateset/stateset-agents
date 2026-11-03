"""Design a conditional MLP neuron from captured activations: "when X, write Y".

The recipe that worked on Qwen3.5-0.8B (``benchmarks/weight_space``): a linear
read-write (``down_proj += B (x) A``) cannot threshold, so a detector that is
merely *lower* on negatives still writes there. A gated MLP neuron can:
``act = silu(g . x) * (u . x)`` is ~0 when ``g . x`` is negative. So repurpose
one neuron:

- **gate row** ``g = gain * (c @ W_gate)``, where ``c`` is a ridge detector fit
  on the layer's *captured gate pre-activations* (``W_gate x``). The new
  pre-activation is then exactly ``gain * (c . W_gate x)``, computable from the
  captures, so its behaviour can be checked before any weight is written.
- **up row** ``u = s * W_up[k]`` for an up unit ``k`` whose activation is
  nearly constant across contexts (scaled to ~1): a multiplier, not a signal.
- **down column** along the direction to write (e.g. a token's logit direction
  from :func:`.atlas.logit_direction`), sized so a firing neuron writes about
  ``write_norm``.
- **which neuron**: the least-used unit of the layer (mean |activation| times
  output-column norm), excluding the up unit it copies.

Where the detector fires matters as much as how well it separates: on
Qwen3.5-0.8B a conclude unit trained on the answer *token* (AUC 0.99) emitted
``</think>`` mid-sentence, while one trained on the following *line break*
(AUC 0.85) concluded cleanly and lifted held-out accuracy to 42%.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .edits import NeuronRewrite

if TYPE_CHECKING:
    import torch

__all__ = [
    "ConditionalNeuron",
    "auc",
    "design_conditional_neuron",
    "least_used_neuron",
    "near_constant_unit",
    "ridge_detector",
]


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Probability a random positive outscores a random negative (ties count half)."""

    p, n = pos.float().flatten(), neg.float().flatten()
    if not len(p) or not len(n):
        raise ValueError("auc needs at least one positive and one negative")
    greater = (p[:, None] > n[None, :]).float().sum()
    ties = (p[:, None] == n[None, :]).float().sum()
    return float((greater + 0.5 * ties) / (len(p) * len(n)))


def ridge_detector(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    lam: float,
    train: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weights ``c`` with ``features @ c ~ labels`` (ridge, no intercept).

    ``labels`` are +1 (fire) / -1 (stay quiet); ``train`` optionally masks the
    rows used for fitting. Intercepts come from near-constant features, which
    real activations always have.
    """
    import torch

    x = features.float()
    y = labels.float()
    if train is not None:
        x, y = x[train], y[train]
    eye = torch.eye(x.shape[1], dtype=x.dtype)
    weights: torch.Tensor = torch.linalg.solve(x.T @ x + lam * eye, x.T @ y)
    return weights


def near_constant_unit(acts: torch.Tensor) -> tuple[int, float]:
    """The unit whose activation varies least relative to its mean, and 1/mean.

    Multiplying that unit's activation by the returned scale gives ~1.0 in
    every context.
    """
    a = acts.float()
    mean, std = a.mean(0), a.std(0) + 1e-6
    unit = int((mean.abs() / std).argmax())
    return unit, float(1.0 / mean[unit])


def least_used_neuron(
    acts: torch.Tensor, down_weight: torch.Tensor, *, exclude: Collection[int] = ()
) -> int:
    """The neuron contributing least: mean |activation| times output-column norm."""
    use = acts.float().abs().mean(0) * down_weight.float().norm(dim=0)
    for j in exclude:
        use[j] = float("inf")
    return int(use.argmin())


@dataclass(frozen=True)
class ConditionalNeuron:
    """A designed neuron and how it behaved on the captured contexts."""

    rewrite: NeuronRewrite
    report: dict[str, Any] = field(default_factory=dict)


def design_conditional_neuron(
    *,
    layer: int,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_acts: torch.Tensor,
    up_acts: torch.Tensor,
    labels: torch.Tensor,
    write_direction: torch.Tensor,
    write_norm: float,
    gain: float = 4.0,
    lam: float = 100.0,
    neuron: int | None = None,
    train: torch.Tensor | None = None,
    fire_threshold: float = 0.5,
    activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> ConditionalNeuron:
    """Rewrite one neuron of ``layer`` to write ``write_direction`` when ``labels`` > 0.

    ``gate_acts`` / ``up_acts`` are the layer's captured ``gate_proj`` /
    ``up_proj`` outputs (pre-activation) at N contexts, ``labels`` +1/-1 per
    context. ``gate_weight`` / ``up_weight`` are ``[intermediate, hidden]`` and
    ``down_weight`` ``[hidden, intermediate]``. By default the detector is fit
    on even-indexed contexts and the report measures it on the odd ones.
    """
    import torch

    act_fn = activation or torch.nn.functional.silu
    n = len(labels)
    if train is None:
        train = torch.arange(n) % 2 == 0
    heldout = ~train
    labels = labels.float()

    c = ridge_detector(gate_acts, labels, lam=lam, train=train)
    up_unit, up_scale = near_constant_unit(up_acts)
    current = act_fn(gate_acts.float()) * up_acts.float()
    v = (
        neuron
        if neuron is not None
        else least_used_neuron(current, down_weight, exclude={up_unit})
    )

    act = act_fn(gain * (gate_acts.float() @ c)) * (
        up_scale * up_acts.float()[:, up_unit]
    )
    pos, neg = labels > 0, labels < 0
    firing = act[pos & (act > fire_threshold)]
    act_on = float(firing.mean()) if len(firing) else float(act[pos].mean())
    if act_on <= 0:
        raise ValueError("the designed neuron never fires on the positive contexts")
    unit_dir = write_direction.float() / write_direction.float().norm()

    gate_row = gain * (c @ gate_weight.float())
    up_row = up_scale * up_weight.float()[up_unit]
    down_col = (write_norm / act_on) * unit_dir
    rewrite = NeuronRewrite(
        layer=layer,
        neuron=v,
        gate=tuple(float(x) for x in gate_row),
        up=tuple(float(x) for x in up_row),
        down=tuple(float(x) for x in down_col),
    )
    hp, hn = heldout & pos, heldout & neg
    report: dict[str, Any] = {
        "layer": layer,
        "neuron": v,
        "up_unit": up_unit,
        "gain": gain,
        "lam": lam,
        "write_norm": write_norm,
        "activation_when_firing": round(act_on, 4),
        "heldout_auc": (
            round(auc(act[hp], act[hn]), 4) if hp.any() and hn.any() else None
        ),
        "heldout_fire_rate": {
            "positive": round(float((act[hp] > fire_threshold).float().mean()), 4),
            "negative": round(float((act[hn] > fire_threshold).float().mean()), 4),
        },
        "heldout_mean_activation": {
            "positive": round(float(act[hp].mean()), 4),
            "negative": round(float(act[hn].mean()), 4),
        },
    }
    return ConditionalNeuron(rewrite=rewrite, report=report)
