"""Tests for the HeadRewrite edit kind: one gated attention query head written by hand."""

from __future__ import annotations

import json
import math
import random

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.experimental.weight_space.edits import (  # noqa: E402
    EditSpec,
    HeadRewrite,
    SpecMismatch,
    TensorLayout,
    TokenRowScale,
    apply_spec,
    matched_control,
    verify_spec,
)

EMBED = "model.embed_tokens.weight"
DOWN = "model.layers.{layer}.mlp.down_proj.weight"
Q_PROJ = "model.layers.1.self_attn.q_proj.weight"
O_PROJ = "model.layers.1.self_attn.o_proj.weight"
LAYOUT = TensorLayout(token_keys=(EMBED,), down_fmt=DOWN)


def state() -> dict:
    """hidden = 2, 2 heads of head_dim 2, gated: q_proj [2*2*2, 2], o_proj [2, 2*2].

    q_proj rows per head h: query rows 4h, 4h+1 then gate rows 4h+2, 4h+3.
    """
    return {
        EMBED: torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        Q_PROJ: torch.arange(16, dtype=torch.float32).view(8, 2),
        O_PROJ: torch.arange(8, dtype=torch.float32).view(2, 4) + 100.0,
        DOWN.format(layer=1): torch.ones(2, 3),
    }


REWRITE = HeadRewrite(
    layer=1,
    head=1,
    query_out=(2.0, -1.0),
    query_in=(0.5, 0.25),
    gate_out=(3.0, 3.0),
    gate_in=(1.0, 0.0),
    out_write=(-4.0, 2.0),
    out_read=(1.0, 0.5),
)
SPEC = EditSpec(name="guard", edits=(TokenRowScale(ids=(0,), scale=0.5), REWRITE))


def test_layout_derives_attention_names_from_down() -> None:
    assert LAYOUT.q_fmt == "model.layers.{layer}.self_attn.q_proj.weight"
    assert LAYOUT.o_fmt == "model.layers.{layer}.self_attn.o_proj.weight"


def test_head_rewrite_sets_its_query_and_gate_rows_and_output_columns_only() -> None:
    tensors = state()
    apply_spec(SPEC, tensors, LAYOUT)
    q, o = tensors[Q_PROJ], tensors[O_PROJ]
    # head 1's query rows (4, 5) = query_out (x) query_in; gate rows (6, 7) likewise
    assert torch.equal(q[4], torch.tensor([1.0, 0.5]))
    assert torch.equal(q[5], torch.tensor([-0.5, -0.25]))
    assert torch.equal(q[6], torch.tensor([3.0, 0.0]))
    assert torch.equal(q[7], torch.tensor([3.0, 0.0]))
    # head 1's output columns (2, 3) = out_write (x) out_read
    assert torch.equal(o[:, 2], torch.tensor([-4.0, 2.0]))
    assert torch.equal(o[:, 3], torch.tensor([-2.0, 1.0]))
    # head 0 untouched
    assert torch.equal(q[:4], state()[Q_PROJ][:4])
    assert torch.equal(o[:, :2], state()[O_PROJ][:, :2])


def test_verify_accepts_the_head_and_rejects_changes_outside_it() -> None:
    base, edited = state(), state()
    apply_spec(SPEC, edited, LAYOUT)
    verify_spec(SPEC, base, edited, LAYOUT)  # no raise
    wrong = {k: v.clone() for k, v in edited.items()}
    wrong[Q_PROJ][5, 0] = 9.0
    with pytest.raises(SpecMismatch, match="q_proj"):
        verify_spec(SPEC, base, wrong, LAYOUT)
    other = {k: v.clone() for k, v in edited.items()}
    other[Q_PROJ][0, 0] = 9.0  # head 0
    with pytest.raises(SpecMismatch, match="row 0"):
        verify_spec(SPEC, base, other, LAYOUT)


def test_head_rewrite_round_trips_json() -> None:
    again = EditSpec.from_json(SPEC.to_json())
    assert again == SPEC
    assert again.content_hash() == SPEC.content_hash()
    assert json.loads(SPEC.to_json())["edits"][1]["kind"] == "head_rewrite"


def test_head_rewrite_rejects_mismatched_lengths() -> None:
    bad = HeadRewrite(
        layer=1,
        head=1,
        query_out=(1.0, 1.0),
        query_in=(1.0, 1.0),
        gate_out=(1.0,),  # head_dim is 2
        gate_in=(1.0, 1.0),
        out_write=(1.0, 1.0),
        out_read=(1.0, 1.0),
    )
    with pytest.raises(ValueError, match="head_dim"):
        apply_spec(EditSpec(name="bad", edits=(bad,)), state(), LAYOUT)


def test_head_rewrite_refuses_an_ungated_attention_layer() -> None:
    tensors = state()
    tensors[Q_PROJ] = torch.zeros(4, 2)  # 2 heads x head_dim 2: no gate rows
    with pytest.raises(ValueError, match="gated"):
        apply_spec(EditSpec(name="ungated", edits=(REWRITE,)), tensors, LAYOUT)


def test_head_out_of_range_is_refused() -> None:
    far = HeadRewrite(
        layer=1,
        head=2,
        query_out=REWRITE.query_out,
        query_in=REWRITE.query_in,
        gate_out=REWRITE.gate_out,
        gate_in=REWRITE.gate_in,
        out_write=REWRITE.out_write,
        out_read=REWRITE.out_read,
    )
    with pytest.raises(ValueError, match="out of range"):
        apply_spec(EditSpec(name="far", edits=(far,)), state(), LAYOUT)


def test_a_head_cannot_be_rewritten_twice() -> None:
    with pytest.raises(ValueError, match="head 1"):
        apply_spec(EditSpec(name="twice", edits=(REWRITE, REWRITE)), state(), LAYOUT)


def test_matched_control_keeps_the_reads_and_randomizes_the_write() -> None:
    control = matched_control(SPEC, rng=random.Random(0), vocab_size=2, n_neurons=3)
    head = control.edits[1]
    assert isinstance(head, HeadRewrite)
    assert (head.layer, head.head) == (1, 1)
    assert (head.query_out, head.query_in) == (REWRITE.query_out, REWRITE.query_in)
    assert (head.gate_out, head.gate_in) == (REWRITE.gate_out, REWRITE.gate_in)
    assert head.out_read == REWRITE.out_read
    assert head.out_write != REWRITE.out_write
    assert math.dist(head.out_write, (0.0, 0.0)) == pytest.approx(math.hypot(4.0, 2.0))


def test_head_rewrite_patches_a_sharded_checkpoint(tmp_path) -> None:
    st = pytest.importorskip("safetensors.torch")
    from stateset_agents.experimental.weight_space.checkpoints import (
        apply_spec_to_checkpoint,
        verify_checkpoint,
    )

    base = tmp_path / "base"
    base.mkdir()
    full = state()
    st.save_file({Q_PROJ: full[Q_PROJ]}, str(base / "model-00001-of-00002.safetensors"))
    st.save_file(
        {
            EMBED: full[EMBED],
            O_PROJ: full[O_PROJ],
            DOWN.format(layer=1): full[DOWN.format(layer=1)],
        },
        str(base / "model-00002-of-00002.safetensors"),
    )
    out = tmp_path / "edited"
    apply_spec_to_checkpoint(SPEC, base, out)
    verify_checkpoint(SPEC, base, out)
    q = st.load_file(str(out / "model-00001-of-00002.safetensors"))[Q_PROJ]
    o = st.load_file(str(out / "model-00002-of-00002.safetensors"))[O_PROJ]
    assert torch.equal(q[4], torch.tensor([1.0, 0.5]))
    assert torch.equal(o[:, 2], torch.tensor([-4.0, 2.0]))
