"""Tests for designing a conditional MLP neuron from captured activations."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.experimental.weight_space.circuits import (  # noqa: E402
    auc,
    design_conditional_neuron,
    least_used_neuron,
    near_constant_unit,
    ridge_detector,
)
from stateset_agents.experimental.weight_space.edits import NeuronRewrite  # noqa: E402

HIDDEN, INTER, N = 8, 16, 400


def synthetic(seed: int = 0) -> dict:
    """Residuals carrying a +/-1 concept on dim 0 and a constant on dim 7.

    gate/up pre-activations are exact linear reads of the residual, as in a real
    MLP; up unit 5 reads only the constant dimension, so it is ~constant.
    """
    g = torch.Generator().manual_seed(seed)
    concept = torch.where(torch.rand(N, generator=g) < 0.4, 1.0, -1.0)
    x = torch.randn(N, HIDDEN, generator=g) * 0.5
    x[:, 0] += 1.5 * concept
    x[:, 7] = 3.0 + 0.05 * torch.randn(N, generator=g)
    wg = torch.randn(INTER, HIDDEN, generator=g)
    wu = torch.randn(INTER, HIDDEN, generator=g)
    wu[5] = 0.0
    wu[5, 7] = 1.0
    wd = torch.randn(HIDDEN, INTER, generator=g)
    wd[:, 11] *= 0.01  # a nearly unused neuron
    return {
        "x": x,
        "labels": concept,
        "gate_weight": wg,
        "up_weight": wu,
        "down_weight": wd,
        "gate_acts": x @ wg.T,
        "up_acts": x @ wu.T,
    }


def test_auc_counts_correctly_ordered_pairs() -> None:
    assert auc(torch.tensor([2.0, 3.0]), torch.tensor([0.0, 1.0])) == 1.0
    assert auc(torch.tensor([0.0]), torch.tensor([1.0])) == 0.0
    assert auc(torch.tensor([1.0, 0.0]), torch.tensor([0.5])) == 0.5


def test_ridge_detector_recovers_a_linear_concept() -> None:
    d = synthetic()
    c = ridge_detector(d["gate_acts"], d["labels"], lam=1.0)
    score = d["gate_acts"] @ c
    assert auc(score[d["labels"] > 0], score[d["labels"] < 0]) > 0.97


def test_near_constant_unit_finds_the_bias_unit() -> None:
    d = synthetic()
    unit, scale = near_constant_unit(d["up_acts"])
    assert unit == 5
    assert float((d["up_acts"][:, unit] * scale).mean()) == pytest.approx(1.0, abs=0.01)


def test_least_used_neuron_weights_activation_by_column_norm() -> None:
    d = synthetic()
    acts = torch.nn.functional.silu(d["gate_acts"]) * d["up_acts"]
    assert least_used_neuron(acts, d["down_weight"]) == 11
    assert least_used_neuron(acts, d["down_weight"], exclude={11}) != 11


def test_designed_neuron_fires_on_the_concept_and_writes_the_direction() -> None:
    d = synthetic()
    direction = torch.zeros(HIDDEN)
    direction[3] = 2.0
    design = design_conditional_neuron(
        layer=4,
        gate_weight=d["gate_weight"],
        up_weight=d["up_weight"],
        down_weight=d["down_weight"],
        gate_acts=d["gate_acts"],
        up_acts=d["up_acts"],
        labels=d["labels"],
        write_direction=direction,
        write_norm=2.5,
        lam=1.0,
    )
    rw = design.rewrite
    assert isinstance(rw, NeuronRewrite)
    assert (rw.layer, rw.neuron) == (4, 11)  # the least-used unit, not the bias unit
    assert len(rw.gate) == len(rw.up) == len(rw.down) == HIDDEN

    # Recompute the new neuron's activation from the residuals and its new rows.
    x = d["x"]
    act = torch.nn.functional.silu(x @ torch.tensor(rw.gate)) * (
        x @ torch.tensor(rw.up)
    )
    heldout = torch.arange(N) % 2 == 1
    pos, neg = act[heldout & (d["labels"] > 0)], act[heldout & (d["labels"] < 0)]
    assert auc(pos, neg) > 0.97
    assert float(neg.abs().mean()) < 0.2 * float(pos.mean())

    # It writes along the requested direction, sized so firing writes ~write_norm.
    down = torch.tensor(rw.down)
    assert torch.allclose(down / down.norm(), direction / direction.norm(), atol=1e-6)
    write = act[d["labels"] > 0].unsqueeze(1) * down
    assert float(write.norm(dim=1)[act[d["labels"] > 0] > 0.5].mean()) == pytest.approx(
        2.5, rel=0.05
    )
    report = design.report
    assert report["heldout_detector_auc"] > 0.97
    assert report["up_unit"] == 5
    assert report["neuron"] == 11


def test_design_can_pin_the_neuron_and_the_multiplier_unit() -> None:
    d = synthetic()
    design = design_conditional_neuron(
        layer=0,
        gate_weight=d["gate_weight"],
        up_weight=d["up_weight"],
        down_weight=d["down_weight"],
        gate_acts=d["gate_acts"],
        up_acts=d["up_acts"],
        labels=d["labels"],
        write_direction=torch.ones(HIDDEN),
        write_norm=1.0,
        lam=1.0,
        neuron=3,
        up_unit=5,
    )
    assert (design.rewrite.neuron, design.report["up_unit"]) == (3, 5)
    scale = 1.0 / float(d["up_acts"][:, 5].mean())
    assert design.rewrite.up == pytest.approx(
        tuple(float(v) * scale for v in d["up_weight"][5]), rel=1e-5
    )


def test_report_measures_the_detector_score_not_the_squashed_activation() -> None:
    d = synthetic()
    design = design_conditional_neuron(
        layer=0,
        gate_weight=d["gate_weight"],
        up_weight=d["up_weight"],
        down_weight=d["down_weight"],
        gate_acts=d["gate_acts"],
        up_acts=d["up_acts"],
        labels=d["labels"],
        write_direction=torch.ones(HIDDEN),
        write_norm=1.0,
        lam=1.0,
    )
    held = torch.arange(N) % 2 == 1
    score = d["x"] @ torch.tensor(design.rewrite.gate)  # gain x detector score
    pos, neg = held & (d["labels"] > 0), held & (d["labels"] < 0)
    assert design.report["heldout_detector_auc"] == pytest.approx(
        auc(score[pos], score[neg]), abs=1e-3
    )
    assert "heldout_activation_auc" in design.report


def test_design_exposes_the_detector_and_its_scores() -> None:
    d = synthetic()
    design = design_conditional_neuron(
        layer=0,
        gate_weight=d["gate_weight"],
        up_weight=d["up_weight"],
        down_weight=d["down_weight"],
        gate_acts=d["gate_acts"],
        up_acts=d["up_acts"],
        labels=d["labels"],
        write_direction=torch.ones(HIDDEN),
        write_norm=1.0,
        lam=1.0,
    )
    assert design.detector is not None and design.detector.shape == (INTER,)
    # The new neuron's pre-activation is gain x (captured gate outputs . detector).
    gate_row = torch.tensor(design.rewrite.gate)
    assert torch.allclose(
        d["x"] @ gate_row,
        design.report["gain"] * (d["gate_acts"] @ design.detector),
        atol=1e-3,
    )
    acts = design.activation(d["gate_acts"], d["up_acts"])
    assert acts.shape == (N,)
    assert float(acts[d["labels"] > 0].mean()) > float(acts[d["labels"] < 0].mean())
