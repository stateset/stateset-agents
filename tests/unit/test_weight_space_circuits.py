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


def test_rewrite_activation_scores_any_stored_rewrite_from_the_captures() -> None:
    from stateset_agents.experimental.weight_space.circuits import rewrite_activation

    d = synthetic()
    g = torch.Generator().manual_seed(7)
    rw = NeuronRewrite(
        layer=0,
        neuron=2,
        gate=tuple(float(v) for v in torch.randn(HIDDEN, generator=g)),
        up=tuple(float(v) for v in torch.randn(HIDDEN, generator=g)),
        down=tuple(0.0 for _ in range(HIDDEN)),
    )
    x = d["x"]
    want = torch.nn.functional.silu(x @ torch.tensor(rw.gate)) * (
        x @ torch.tensor(rw.up)
    )
    got = rewrite_activation(
        rw,
        gate_weight=d["gate_weight"],
        up_weight=d["up_weight"],
        gate_acts=d["gate_acts"],
        up_acts=d["up_acts"],
    )
    assert torch.allclose(got, want, atol=1e-3)


def test_rewrite_activation_refuses_a_row_the_captures_cannot_express() -> None:
    from stateset_agents.experimental.weight_space.circuits import rewrite_activation

    d = synthetic()
    wg = d["gate_weight"].clone()
    wg[:, 2] = 0.0  # no gate row reads residual dim 2
    gate = [0.0] * HIDDEN
    gate[2] = 1.0
    rw = NeuronRewrite(
        layer=0,
        neuron=0,
        gate=tuple(gate),
        up=tuple(float(v) for v in d["up_weight"][5]),
        down=tuple(0.0 for _ in range(HIDDEN)),
    )
    with pytest.raises(ValueError, match="span"):
        rewrite_activation(
            rw,
            gate_weight=wg,
            up_weight=d["up_weight"],
            gate_acts=d["x"] @ wg.T,
            up_acts=d["up_acts"],
        )


def _designed(d: dict):
    direction = torch.zeros(HIDDEN)
    direction[3] = 2.0
    return design_conditional_neuron(
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
    ).rewrite


def test_shift_threshold_lowers_the_pre_activation_by_bias_times_the_up_read() -> None:
    from stateset_agents.experimental.weight_space.circuits import shift_threshold

    d = synthetic()
    rw = _designed(d)
    shifted = shift_threshold(rw, 1.5, write_scale=0.5)
    x = d["x"]
    gate, up = x @ torch.tensor(rw.gate), x @ torch.tensor(rw.up)
    assert torch.allclose(x @ torch.tensor(shifted.gate), gate - 1.5 * up, atol=1e-4)
    # The designer's up row reads ~1.0 everywhere, so the shift is ~-bias.
    assert float((gate - x @ torch.tensor(shifted.gate)).mean()) == pytest.approx(
        1.5, rel=0.02
    )
    assert shifted.up == rw.up
    assert torch.allclose(torch.tensor(shifted.down), 0.5 * torch.tensor(rw.down))
    assert (shifted.layer, shifted.neuron) == (rw.layer, rw.neuron)


def test_a_higher_threshold_fires_less_on_both_classes() -> None:
    from stateset_agents.experimental.weight_space.circuits import shift_threshold

    d = synthetic()
    rw = _designed(d)
    x, pos = d["x"], d["labels"] > 0

    def fire(r: NeuronRewrite) -> tuple[float, float]:
        act = torch.nn.functional.silu(x @ torch.tensor(r.gate)) * (
            x @ torch.tensor(r.up)
        )
        on = act > 0.5
        return float(on[pos].float().mean()), float(on[~pos].float().mean())

    low, mid, high = (fire(shift_threshold(rw, b)) for b in (-2.0, 0.0, 4.0))
    assert low[0] >= mid[0] >= high[0] and low[1] >= mid[1] >= high[1]
    assert high[0] < mid[0]
    assert shift_threshold(rw, 0.0) == rw


@pytest.mark.parametrize(
    ("bias", "write_scale"),
    [
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (0.0, 0.0),
        (0.0, -1.0),
        (0.0, float("nan")),
    ],
)
def test_shift_threshold_refuses_non_finite_bias_and_non_positive_write(
    bias: float, write_scale: float
) -> None:
    from stateset_agents.experimental.weight_space.circuits import shift_threshold

    with pytest.raises(ValueError):
        shift_threshold(_designed(synthetic()), bias, write_scale=write_scale)


def test_rewrite_activation_reads_a_weakly_read_direction_exactly() -> None:
    """The read-through solve is float64: float32 least squares truncates weak
    directions (and on some CPU builds returns run-to-run different residuals)."""
    from stateset_agents.experimental.weight_space.circuits import rewrite_activation

    g = torch.Generator().manual_seed(3)
    u, _ = torch.linalg.qr(torch.randn(INTER, HIDDEN, generator=g))
    v, _ = torch.linalg.qr(torch.randn(HIDDEN, HIDDEN, generator=g))
    s = torch.logspace(0, -6, HIDDEN)  # the weakest read is 1e-6 of the strongest
    wg = (u * s) @ v.T
    wu = torch.randn(INTER, HIDDEN, generator=g)
    gate_row = v[:, -1].clone()  # exactly the weakest-read direction
    x = torch.randn(N, HIDDEN, generator=g)
    rw = NeuronRewrite(
        layer=0,
        neuron=0,
        gate=tuple(float(t) for t in gate_row),
        up=tuple(float(t) for t in wu[0]),
        down=tuple(0.0 for _ in range(HIDDEN)),
    )
    got = rewrite_activation(
        rw,
        gate_weight=wg,
        up_weight=wu,
        gate_acts=(x.double() @ wg.double().T),
        up_acts=x @ wu.T,
    )
    want = torch.nn.functional.silu(x @ gate_row) * (x @ wu[0])
    assert torch.allclose(got.float(), want, atol=1e-3)


def _rms_norm(x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    return x / x.pow(2).mean().sqrt() * gain


def test_lookback_query_points_along_the_target_key_without_rotary_dims() -> None:
    from stateset_agents.experimental.weight_space.circuits import lookback_query

    g = torch.Generator().manual_seed(5)
    hd, rot = 16, 4
    base = torch.randn(hd, generator=g)
    target_keys = base + 0.05 * torch.randn(6, hd, generator=g)
    gain = 1.0 + 0.3 * torch.rand(hd, generator=g)
    q_out = lookback_query(target_keys, gain, rotary_dims=rot)
    assert torch.equal(q_out[:rot], torch.zeros(rot))
    u = target_keys[:, rot:].mean(0)
    u = u / u.norm()
    for c in (0.01, 1.0, 300.0):  # any positive read gives the same query
        q = _rms_norm(c * q_out, gain)
        assert float(q[:rot].abs().max()) == 0.0
        assert float((q[rot:] / q[rot:].norm()) @ u) == pytest.approx(1.0, abs=1e-5)
    # It scores the target keys above unrelated keys.
    others = torch.randn(200, hd, generator=g)
    q = _rms_norm(q_out, gain)
    assert float((target_keys @ q).min()) > float((others @ q).max())


@pytest.mark.parametrize(("rot", "gain_fill"), [(16, 1.0), (4, 0.0)])
def test_lookback_query_refuses_no_free_dims_or_a_dead_gain(
    rot: int, gain_fill: float
) -> None:
    from stateset_agents.experimental.weight_space.circuits import lookback_query

    keys = torch.ones(3, 16)
    with pytest.raises(ValueError):
        lookback_query(keys, torch.full((16,), gain_fill), rotary_dims=rot)


def test_silence_on_flag_cancels_the_multiplier_and_lowers_the_gate() -> None:
    from stateset_agents.experimental.weight_space.circuits import silence_on_flag

    g = torch.Generator().manual_seed(9)
    hidden = 8
    gain = 1.0 + 0.2 * torch.rand(hidden, generator=g)
    flag = torch.zeros(hidden)
    flag[7] = 1.0
    rw = NeuronRewrite(
        layer=3,
        neuron=5,
        gate=tuple(float(v) for v in torch.randn(hidden, generator=g)),
        up=tuple(float(v) for v in torch.randn(hidden, generator=g)),
        down=tuple(float(v) for v in torch.randn(hidden, generator=g)),
    )
    out = silence_on_flag(rw, flag, norm_gain=gain, flag_read=4.0, gate_weight=1.5)
    base = torch.randn(hidden, generator=g)
    base[7] = 0.0  # a normalised input that does not use the flag direction
    flagged = base + 4.0 * flag * gain  # the residual carries the flag at read 4
    gate, up = torch.tensor(rw.gate), torch.tensor(rw.up)
    gate2, up2 = torch.tensor(out.gate), torch.tensor(out.up)
    assert float(up2 @ flagged) == pytest.approx(float(up @ flagged) - 1.0, abs=1e-5)
    assert float(gate2 @ flagged) == pytest.approx(
        float(gate @ flagged) - 6.0, abs=1e-5
    )
    assert float(up2 @ base) == pytest.approx(
        float(up @ base), abs=1e-6
    )  # no flag: unchanged
    assert float(gate2 @ base) == pytest.approx(float(gate @ base), abs=1e-6)
    assert out.down == rw.down and (out.layer, out.neuron) == (3, 5)


def test_silence_on_flag_refuses_a_non_positive_read() -> None:
    from stateset_agents.experimental.weight_space.circuits import silence_on_flag

    rw = NeuronRewrite(
        layer=0, neuron=0, gate=(1.0, 0.0), up=(1.0, 0.0), down=(0.0, 0.0)
    )
    with pytest.raises(ValueError):
        silence_on_flag(
            rw,
            torch.tensor([0.0, 1.0]),
            norm_gain=torch.ones(2),
            flag_read=0.0,
            gate_weight=1.0,
        )
