"""Tests for the NeuronRewrite edit kind: a whole MLP neuron written by hand."""

from __future__ import annotations

import json
import math
import random

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.experimental.weight_space.edits import (  # noqa: E402
    EditSpec,
    NeuronColumnScale,
    NeuronRewrite,
    SpecMismatch,
    TensorLayout,
    TokenRowScale,
    apply_spec,
    matched_control,
    verify_spec,
)

EMBED = "model.embed_tokens.weight"
DOWN = "model.layers.{layer}.mlp.down_proj.weight"
GATE = "model.layers.1.mlp.gate_proj.weight"
UP = "model.layers.1.mlp.up_proj.weight"
LAYOUT = TensorLayout(token_keys=(EMBED,), down_fmt=DOWN)


def state() -> dict:
    """hidden = 2, intermediate = 3 (gate/up: [3, 2]; down: [2, 3])."""
    return {
        EMBED: torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        GATE: torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        UP: torch.tensor([[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]),
        DOWN.format(layer=1): torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    }


REWRITE = NeuronRewrite(
    layer=1, neuron=2, gate=(0.5, -0.5), up=(0.25, 0.25), down=(3.0, 4.0)
)
SPEC = EditSpec(
    name="answer-neuron",
    edits=(TokenRowScale(ids=(0,), scale=0.5), REWRITE),
)


def test_layout_derives_gate_and_up_names_from_down() -> None:
    assert LAYOUT.gate_fmt == "model.layers.{layer}.mlp.gate_proj.weight"
    assert LAYOUT.up_fmt == "model.layers.{layer}.mlp.up_proj.weight"


def test_rewrite_sets_the_neurons_rows_and_column_only() -> None:
    tensors = state()
    changed = apply_spec(SPEC, tensors, LAYOUT)
    assert torch.equal(tensors[GATE][2], torch.tensor([0.5, -0.5]))
    assert torch.equal(tensors[UP][2], torch.tensor([0.25, 0.25]))
    assert torch.equal(tensors[DOWN.format(layer=1)][:, 2], torch.tensor([3.0, 4.0]))
    # other neurons untouched
    assert torch.equal(tensors[GATE][:2], state()[GATE][:2])
    assert torch.equal(
        tensors[DOWN.format(layer=1)][:, :2], state()[DOWN.format(layer=1)][:, :2]
    )
    # Counts weights whose value actually changed: 2 embedding + 2 gate + 2 up
    # + 1 down (down[0, 2] was already 3.0).
    assert changed == 7


def test_verify_accepts_the_rewrite_and_rejects_a_wrong_value() -> None:
    base, edited = state(), state()
    apply_spec(SPEC, edited, LAYOUT)
    verify_spec(SPEC, base, edited, LAYOUT)  # no raise
    edited[UP][2, 0] = 9.0
    with pytest.raises(SpecMismatch, match="up_proj"):
        verify_spec(SPEC, base, edited, LAYOUT)


def test_verify_rejects_a_change_to_another_neuron() -> None:
    base, edited = state(), state()
    apply_spec(SPEC, edited, LAYOUT)
    edited[GATE][0, 0] = 7.0
    with pytest.raises(SpecMismatch, match="row 0"):
        verify_spec(SPEC, base, edited, LAYOUT)


def test_rewrite_round_trips_json_and_hash_tracks_values() -> None:
    again = EditSpec.from_json(SPEC.to_json())
    assert again == SPEC
    assert again.content_hash() == SPEC.content_hash()
    assert json.loads(SPEC.to_json())["edits"][1]["kind"] == "neuron_rewrite"
    nudged = NeuronRewrite(
        layer=1, neuron=2, gate=(0.5, -0.5), up=(0.25, 0.25), down=(3.0, 4.000001)
    )
    other = EditSpec(name="answer-neuron", edits=(SPEC.edits[0], nudged))
    assert other.content_hash() != SPEC.content_hash()


def test_rewrite_rejects_vectors_of_the_wrong_length() -> None:
    bad = EditSpec(
        name="bad",
        edits=(
            NeuronRewrite(
                layer=1, neuron=2, gate=(1.0,), up=(1.0, 1.0), down=(1.0, 1.0)
            ),
        ),
    )
    with pytest.raises(ValueError, match="gate"):
        apply_spec(bad, state(), LAYOUT)


def test_rewrite_cannot_be_combined_with_a_scale_of_the_same_neuron() -> None:
    clash = EditSpec(
        name="clash",
        edits=(REWRITE, NeuronColumnScale(layer=1, neurons=(2,), scale=2.0)),
    )
    with pytest.raises(ValueError, match="neuron 2"):
        apply_spec(clash, state(), LAYOUT)


def test_matched_control_keeps_the_read_and_randomizes_the_write() -> None:
    control = matched_control(SPEC, rng=random.Random(0), vocab_size=2, n_neurons=3)
    rewrite = control.edits[1]
    assert isinstance(rewrite, NeuronRewrite)
    assert (rewrite.layer, rewrite.neuron) == (1, 2)
    assert rewrite.gate == REWRITE.gate and rewrite.up == REWRITE.up
    assert rewrite.down != REWRITE.down
    assert math.dist(rewrite.down, (0.0, 0.0)) == pytest.approx(5.0)  # same norm


def test_rewrite_patches_a_checkpoint_whose_neuron_spans_shards(tmp_path) -> None:
    st = pytest.importorskip("safetensors.torch")
    from stateset_agents.experimental.weight_space.checkpoints import (
        apply_spec_to_checkpoint,
        verify_checkpoint,
    )

    base = tmp_path / "base"
    base.mkdir()
    full = state()
    # gate/up in one shard, down (and the embedding) in another
    st.save_file(
        {GATE: full[GATE], UP: full[UP]}, str(base / "model-00001-of-00002.safetensors")
    )
    st.save_file(
        {EMBED: full[EMBED], DOWN.format(layer=1): full[DOWN.format(layer=1)]},
        str(base / "model-00002-of-00002.safetensors"),
    )
    out = tmp_path / "edited"
    manifest = apply_spec_to_checkpoint(SPEC, base, out)
    assert manifest["changed_weights"] == 7
    verify_checkpoint(SPEC, base, out)
    patched = st.load_file(str(out / "model-00001-of-00002.safetensors"))
    assert torch.equal(patched[GATE][2], torch.tensor([0.5, -0.5]))
