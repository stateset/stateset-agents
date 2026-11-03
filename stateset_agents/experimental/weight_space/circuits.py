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
    "rewrite_activation",
    "ridge_detector",
    "shift_threshold",
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
    """A designed neuron and how it behaved on the captured contexts.

    ``detector`` is the ridge readout over the layer's gate pre-activations;
    with ``up_unit`` / ``up_scale`` it lets :meth:`activation` compute what the
    new neuron would do at any captured context, before writing a weight.
    """

    rewrite: NeuronRewrite
    report: dict[str, Any] = field(default_factory=dict)
    detector: torch.Tensor | None = None
    up_unit: int = 0
    up_scale: float = 1.0
    gain: float = 1.0
    act_fn: Callable[[torch.Tensor], torch.Tensor] | None = None

    def activation(
        self, gate_acts: torch.Tensor, up_acts: torch.Tensor
    ) -> torch.Tensor:
        """The designed neuron's activation at contexts with these captures."""
        import torch

        if self.detector is None:
            raise ValueError("this design carries no detector")
        fn = self.act_fn or torch.nn.functional.silu
        return fn(self.gain * (gate_acts.float() @ self.detector)) * (
            self.up_scale * up_acts.float()[:, self.up_unit]
        )


def _read_through(
    row: torch.Tensor, weight: torch.Tensor, acts: torch.Tensor, what: str
) -> torch.Tensor:
    """``x . row`` at each captured context, from ``acts = x @ weight.T``.

    Solves ``weight.T @ c = row`` (minimum norm) so ``acts @ c == x . row``.
    The solve runs in float64: float32 least squares truncates weakly read
    directions, and on some CPU builds returns residuals that differ from run
    to run on identical inputs (1e-6 to 2.5e-2 on Qwen3.5-0.8B layer 23).
    """
    import torch

    w = weight.double().T  # [hidden, intermediate]
    r = row.double()
    c = torch.linalg.lstsq(w, r.unsqueeze(1)).solution.squeeze(1)
    miss = float((w @ c - r).norm() / r.norm().clamp_min(1e-12))
    if miss > 1e-3:
        raise ValueError(
            f"the {what} row is not in the span of the layer's {what} rows "
            f"(relative residual {miss:.2e}); capture the MLP input instead"
        )
    read: torch.Tensor = (acts.double() @ c).float()
    return read


def rewrite_activation(
    rewrite: NeuronRewrite,
    *,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    gate_acts: torch.Tensor,
    up_acts: torch.Tensor,
    activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """What a stored :class:`NeuronRewrite` would output at captured contexts.

    ``gate_acts`` / ``up_acts`` are the *base* layer's ``gate_proj`` /
    ``up_proj`` outputs at N contexts and ``gate_weight`` / ``up_weight`` its
    ``[intermediate, hidden]`` weights. The rewrite's rows are read through
    those captures, so an earlier spec (e.g. a previous design) can be scored
    on new contexts without loading the edited model. Raises ``ValueError``
    when a row is outside the span the captures can express.
    """
    import torch

    act_fn = activation or torch.nn.functional.silu
    gate = _read_through(torch.tensor(rewrite.gate), gate_weight, gate_acts, "gate")
    up = _read_through(torch.tensor(rewrite.up), up_weight, up_acts, "up")
    return act_fn(gate) * up


def shift_threshold(
    rewrite: NeuronRewrite, bias: float, *, write_scale: float = 1.0
) -> NeuronRewrite:
    """Give a designed neuron an explicit firing threshold and write strength.

    The ridge detector has no intercept, so a designed neuron's threshold is
    whatever the fit's shrinkage left. This returns the same neuron with gate
    row ``gate - bias * up``: its pre-activation becomes
    ``gate . x - bias * (up . x)``. For an up row that reads ~1.0 in every
    context (as :func:`design_conditional_neuron` builds it) that is
    ``gate . x - bias``, so a positive ``bias`` needs that much more detector
    evidence to fire. ``write_scale`` multiplies the down column.
    """
    import math

    if not math.isfinite(bias):
        raise ValueError(f"bias must be finite, got {bias}")
    if not (math.isfinite(write_scale) and write_scale > 0):
        raise ValueError(
            f"write_scale must be a positive finite number, got {write_scale}"
        )
    if bias == 0.0 and write_scale == 1.0:
        return rewrite
    return NeuronRewrite(
        layer=rewrite.layer,
        neuron=rewrite.neuron,
        gate=tuple(g - bias * u for g, u in zip(rewrite.gate, rewrite.up, strict=True)),
        up=rewrite.up,
        down=tuple(write_scale * v for v in rewrite.down),
    )


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
    up_unit: int | None = None,
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
    ``neuron`` / ``up_unit`` pin the slot and the multiplier unit instead of
    choosing them from the captures (e.g. to reproduce an earlier design).
    """
    import torch

    act_fn = activation or torch.nn.functional.silu
    n = len(labels)
    if train is None:
        train = torch.arange(n) % 2 == 0
    heldout = ~train
    labels = labels.float()

    c = ridge_detector(gate_acts, labels, lam=lam, train=train)
    if up_unit is None:
        up_unit, up_scale = near_constant_unit(up_acts)
    else:
        up_scale = float(1.0 / up_acts.float()[:, up_unit].mean())
    current = act_fn(gate_acts.float()) * up_acts.float()
    v = (
        neuron
        if neuron is not None
        else least_used_neuron(current, down_weight, exclude={up_unit})
    )

    score = (
        gate_acts.float() @ c
    )  # the detector; the new pre-activation is gain * score
    act = act_fn(gain * score) * (up_scale * up_acts.float()[:, up_unit])
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
    both = bool(hp.any() and hn.any())
    report: dict[str, Any] = {
        "layer": layer,
        "neuron": v,
        "up_unit": up_unit,
        "gain": gain,
        "lam": lam,
        "write_norm": write_norm,
        "activation_when_firing": round(act_on, 4),
        # Separation by the detector score (what the gate reads). The activation
        # squashes every non-firing context near zero, so its AUC mostly counts
        # ties among them and understates the detector.
        "heldout_detector_auc": round(auc(score[hp], score[hn]), 4) if both else None,
        "heldout_activation_auc": round(auc(act[hp], act[hn]), 4) if both else None,
        "heldout_fire_rate": {
            "positive": round(float((act[hp] > fire_threshold).float().mean()), 4),
            "negative": round(float((act[hn] > fire_threshold).float().mean()), 4),
        },
        "heldout_mean_activation": {
            "positive": round(float(act[hp].mean()), 4),
            "negative": round(float(act[hn].mean()), 4),
        },
    }
    return ConditionalNeuron(
        rewrite=rewrite,
        report=report,
        detector=c,
        up_unit=up_unit,
        up_scale=up_scale,
        gain=gain,
        act_fn=act_fn,
    )
