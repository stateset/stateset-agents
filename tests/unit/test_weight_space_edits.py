"""Tests for weight-space edit specs, checkpoint patching, verification, and protocol stats."""

from __future__ import annotations

import json
import random

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.experimental.weight_space.edits import (  # noqa: E402
    EditSpec,
    NeuronColumnScale,
    SpecMismatch,
    TensorLayout,
    TokenRowScale,
    apply_spec,
    matched_control,
    verify_spec,
)
from stateset_agents.experimental.weight_space.protocol import (  # noqa: E402
    bootstrap_ci,
    mcnemar_exact,
    paired_compare,
)

EMBED = "model.embed_tokens.weight"
DOWN = "model.layers.{layer}.mlp.down_proj.weight"
LAYOUT = TensorLayout(token_keys=(EMBED,), down_fmt=DOWN)


def state() -> dict:
    return {
        EMBED: torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        DOWN.format(layer=1): torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    }


SPEC = EditSpec(
    name="w5t15",
    edits=(
        TokenRowScale(ids=(1, 3), scale=0.5),
        TokenRowScale(ids=(2,), scale=1.5),
        NeuronColumnScale(layer=1, neurons=(2,), scale=0.0),
    ),
)


# --------------------------------------------------------------- apply / verify


def test_apply_spec_scales_exactly_the_named_rows_and_columns() -> None:
    tensors = state()
    changed = apply_spec(SPEC, tensors, LAYOUT)
    assert torch.equal(
        tensors[EMBED], torch.tensor([[1.0, 2.0], [1.5, 2.0], [7.5, 9.0], [3.5, 4.0]])
    )
    assert torch.equal(
        tensors[DOWN.format(layer=1)], torch.tensor([[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]])
    )
    # rows 1, 2, 3 (2 weights each) + one column (2 weights)
    assert changed == 8


def test_verify_spec_accepts_a_faithful_patch_and_rejects_a_stray_change() -> None:
    base, edited = state(), state()
    apply_spec(SPEC, edited, LAYOUT)
    verify_spec(SPEC, base, edited, LAYOUT)  # no raise
    edited[EMBED][0, 0] += 1.0  # a row the spec never names
    with pytest.raises(SpecMismatch, match="row 0"):
        verify_spec(SPEC, base, edited, LAYOUT)


def test_verify_spec_rejects_a_wrong_scale() -> None:
    base, edited = state(), state()
    edited[EMBED][[1, 3]] *= 0.25  # spec says 0.5
    edited[EMBED][2] *= 1.5
    edited[DOWN.format(layer=1)][:, 2] = 0.0
    with pytest.raises(SpecMismatch, match="scale"):
        verify_spec(SPEC, base, edited, LAYOUT)


def test_apply_spec_rejects_out_of_range_indices() -> None:
    bad = EditSpec(name="bad", edits=(TokenRowScale(ids=(9,), scale=0.5),))
    with pytest.raises(ValueError, match="row 9"):
        apply_spec(bad, state(), LAYOUT)


# --------------------------------------------------------------- spec identity


def test_spec_json_round_trip_preserves_content_hash() -> None:
    again = EditSpec.from_json(SPEC.to_json())
    assert again == SPEC
    assert again.content_hash() == SPEC.content_hash()
    assert json.loads(SPEC.to_json())["edits"][0] == {
        "kind": "token_rows",
        "ids": [1, 3],
        "scale": 0.5,
    }


def test_content_hash_changes_with_any_parameter() -> None:
    tweaked = EditSpec(
        name="w5t15", edits=(TokenRowScale(ids=(1, 3), scale=0.6),) + SPEC.edits[1:]
    )
    assert tweaked.content_hash() != SPEC.content_hash()


# --------------------------------------------------------------- matched controls


def test_matched_control_keeps_structure_but_moves_targets() -> None:
    control = matched_control(
        SPEC, rng=random.Random(0), vocab_size=4, n_neurons=3, token_pool=(0,)
    )
    kinds = [type(e).__name__ for e in control.edits]
    assert kinds == ["TokenRowScale", "TokenRowScale", "NeuronColumnScale"]
    assert [e.scale for e in control.edits] == [0.5, 1.5, 0.0]
    # every token target replaced from the pool; neuron targets avoid the originals
    assert control.edits[0].ids == (0, 0) and control.edits[1].ids == (0,)
    assert control.edits[2].layer == 1 and 2 not in control.edits[2].neurons
    assert control.name == "control:w5t15"


# --------------------------------------------------------------- paired statistics


def test_mcnemar_exact_matches_hand_values() -> None:
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)  # 2 * (1/2)^5
    assert mcnemar_exact(1, 9) == pytest.approx(
        2 * 11 / 1024
    )  # 2 * (C(10,0)+C(10,1)) / 2^10
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 0) == 1.0


def test_bootstrap_ci_brackets_the_mean() -> None:
    lo, hi = bootstrap_ci([1] * 50 + [0] * 50, iters=4000, seed=0)
    assert 0.38 < lo < 0.42 and 0.58 < hi < 0.62


def test_paired_compare_counts_flips_by_problem_id() -> None:
    base = [{"id": str(i), "correct": i < 2} for i in range(10)]
    edit = [
        {"id": str(i), "correct": i < 6} for i in reversed(range(10))
    ]  # order must not matter
    out = paired_compare(base, edit)
    assert (out["n"], out["gained"], out["lost"]) == (10, 4, 0)
    assert out["delta"] == pytest.approx(0.4)
    assert out["mcnemar_p"] == pytest.approx(0.125)


# --------------------------------------------------------------- tensor layouts


def test_layout_detects_tied_multimodal_names() -> None:
    names = [
        "model.language_model.embed_tokens.weight",
        "model.visual.patch_embed.proj.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        *(f"model.language_model.layers.{i}.mlp.down_proj.weight" for i in range(3)),
    ]
    layout = TensorLayout.detect(names)
    assert layout.token_keys == ("model.language_model.embed_tokens.weight",)
    assert layout.down_fmt == "model.language_model.layers.{layer}.mlp.down_proj.weight"


def test_layout_includes_a_stored_lm_head() -> None:
    names = [EMBED, "lm_head.weight", DOWN.format(layer=0)]
    layout = TensorLayout.detect(names)
    assert layout.token_keys == ("lm_head.weight", EMBED)
    assert layout.down_fmt == DOWN


def test_layout_detection_fails_loudly_without_an_embedding() -> None:
    with pytest.raises(ValueError, match="embed_tokens"):
        TensorLayout.detect([DOWN.format(layer=0)])


def test_token_edit_scales_every_token_matrix() -> None:
    tensors = state()
    tensors["lm_head.weight"] = tensors[EMBED].clone()
    layout = TensorLayout(token_keys=("lm_head.weight", EMBED), down_fmt=DOWN)
    spec = EditSpec(name="t", edits=(TokenRowScale(ids=(2,), scale=2.0),))
    base = {k: v.clone() for k, v in tensors.items()}
    assert apply_spec(spec, tensors, layout) == 4
    assert torch.equal(tensors["lm_head.weight"][2], torch.tensor([10.0, 12.0]))
    assert torch.equal(tensors[EMBED][2], torch.tensor([10.0, 12.0]))
    verify_spec(spec, base, tensors, layout)
