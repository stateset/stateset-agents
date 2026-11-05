"""Tests for ``stateset_agents.experimental.weight_space``.

Weight-space programming: an LLM edits a network's weights directly (no
optimizer), spending a scarce budget of forward passes to check its edits,
and ships the kept edit as an ordinary LoRA adapter. These tests pin the
harness contracts that make that safe to run: edits are exact and exactly
revertible, the forward-pass budget cannot be bypassed through the harness,
and the exported adapter reproduces the in-place edit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from stateset_agents.experimental.weight_space import mcp_tools  # noqa: E402
from stateset_agents.experimental.weight_space import (  # noqa: E402
    BudgetExhausted,
    LowRankDelta,
    WeightSpaceSession,
    direction_ablation,
    load_splits,
)

VOCAB = 64
MLP_OUT = "model.layers.1.mlp.down_proj"  # Linear(32 -> 16)
ATTN_OUT = "model.layers.0.self_attn.o_proj"  # Linear(16 -> 16)


def encode(text: str) -> list[int]:
    """Deterministic char-level tokenizer for the tiny model."""
    return [ord(ch) % VOCAB for ch in text]


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    model = transformers.Qwen3ForCausalLM(config).eval()
    # Default init (std 0.02) makes every logit ~0; widen it so edits show.
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() == 2:
                param.normal_(0.0, 0.3)
    return model


SPLITS = {
    "loop": [("ab ab ab", " ab"), ("xyz xyz", " xyz")],
    "fresh": [("count: one", " two three"), ("hi", " there")],
}


def make_session(model, budget: int = 10) -> WeightSpaceSession:
    return WeightSpaceSession(
        model,
        encode,
        budget=budget,
        splits=SPLITS,
        base_model="tiny-qwen3",
        batch_size=2,
    )


def logits(model, text: str):
    ids = torch.tensor([encode(text)])
    with torch.no_grad():
        return model(input_ids=ids).logits[0]


# --------------------------------------------------------------- deltas


@pytest.mark.parametrize("direction", [[1.0, 0.0], [5.0, 0.0]])
def test_direction_ablation_removes_direction_from_top_k_columns(direction) -> None:
    weight = torch.tensor([[3.0, -1.0, 2.0], [4.0, 5.0, 6.0]])
    delta = direction_ablation(weight, torch.tensor(direction), k=2)
    # Columns 0 and 2 write most along e0 (|3|, |2| > |-1|); their e0
    # component is removed, column 1 and the e1 row are untouched. An
    # unnormalised direction must give the same edit.
    assert torch.equal(
        weight + delta.dense(), torch.tensor([[0.0, -1.0, 0.0], [4.0, 5.0, 6.0]])
    )
    assert delta.rank == 1


def test_direction_ablation_alpha_scales_the_removed_component() -> None:
    weight = torch.tensor([[3.0, -1.0, 2.0], [4.0, 5.0, 6.0]])
    delta = direction_ablation(weight, torch.tensor([1.0, 0.0]), k=2, alpha=0.5)
    assert torch.equal(
        weight + delta.dense(), torch.tensor([[1.5, -1.0, 1.0], [4.0, 5.0, 6.0]])
    )


def test_direction_ablation_follows_the_weights_device() -> None:
    # Regression (live GPU run): a CPU direction against an accelerator-resident
    # weight raised "Expected all tensors to be on the same device". The meta
    # device stands in for "a device other than the direction's".
    weight = torch.empty(4, 6, device="meta")
    delta = direction_ablation(weight, torch.tensor([1.0, 0.0, 0.0, 0.0]), k=2)
    assert delta.B.device.type == "meta" and delta.A.device.type == "meta"


@pytest.mark.parametrize(
    ("direction", "k", "message"),
    [
        ([0.0, 0.0], 1, "zero"),
        ([1.0, 0.0, 0.0], 1, "length"),
        ([1.0, 0.0], 4, "k="),
        ([1.0, 0.0], 0, "k="),
    ],
)
def test_direction_ablation_rejects_bad_inputs(direction, k, message) -> None:
    weight = torch.ones(2, 3)
    with pytest.raises(ValueError, match=message):
        direction_ablation(weight, torch.tensor(direction), k=k)


# --------------------------------------------------------------- edits


def test_apply_changes_model_output_and_revert_restores_bit_exactly(tiny_model) -> None:
    session = make_session(tiny_model)
    original = tiny_model.get_submodule(MLP_OUT).weight.detach().clone()
    before = logits(tiny_model, "ab ab")

    direction = torch.randn(16)
    session.apply(MLP_OUT, direction_ablation(original, direction, k=8), note="e1")
    assert not torch.allclose(logits(tiny_model, "ab ab"), before)

    reverted = session.revert()
    assert reverted.note == "e1"
    assert torch.equal(tiny_model.get_submodule(MLP_OUT).weight, original)
    assert torch.equal(logits(tiny_model, "ab ab"), before)
    assert session.edits == []


def test_stacked_edits_on_one_module_revert_last_in_first_out(tiny_model) -> None:
    session = make_session(tiny_model)
    weight = tiny_model.get_submodule(MLP_OUT).weight
    original = weight.detach().clone()

    session.apply(MLP_OUT, direction_ablation(original, torch.randn(16), k=4))
    after_first = weight.detach().clone()
    session.apply(MLP_OUT, direction_ablation(after_first, torch.randn(16), k=4))

    session.revert()
    assert torch.equal(weight, after_first)
    session.revert()
    assert torch.equal(weight, original)
    with pytest.raises(IndexError):
        session.revert()


def test_apply_rejects_delta_with_wrong_shape(tiny_model) -> None:
    session = make_session(tiny_model)
    wrong = LowRankDelta(B=torch.ones(16, 1), A=torch.ones(1, 16))  # o_proj shape
    with pytest.raises(ValueError, match="shape"):
        session.apply(MLP_OUT, wrong)
    assert session.edits == []


# --------------------------------------------------------------- budget


def test_evaluate_spends_budget_and_refuses_when_exhausted(tiny_model) -> None:
    session = make_session(tiny_model, budget=2)
    session.evaluate()
    session.capture(MLP_OUT, ["ab ab"])
    assert session.budget.used == 2
    with pytest.raises(BudgetExhausted):
        session.evaluate()
    with pytest.raises(BudgetExhausted):
        session.capture(MLP_OUT, ["ab"])
    assert session.budget.used == 2


def test_evaluate_scores_mean_continuation_logprob_per_split(tiny_model) -> None:
    session = make_session(tiny_model)
    metrics = session.evaluate()

    def unbatched(prompt: str, continuation: str) -> float:
        ids = encode(prompt) + encode(continuation)
        logp = logits(tiny_model, prompt + continuation).log_softmax(-1)
        start = len(encode(prompt))
        picks = [logp[t - 1, ids[t]].item() for t in range(start, len(ids))]
        return sum(picks) / len(picks)

    for split, pairs in SPLITS.items():
        want = sum(unbatched(p, c) for p, c in pairs) / len(pairs)
        assert metrics[split] == pytest.approx(want, abs=1e-5)


def test_capture_returns_last_token_output_of_module(tiny_model) -> None:
    session = make_session(tiny_model)
    seen = {}
    handle = tiny_model.get_submodule(MLP_OUT).register_forward_hook(
        lambda _m, _i, out: seen.setdefault("out", out.detach().clone())
    )
    with torch.no_grad():
        tiny_model(input_ids=torch.tensor([encode("xyz")]))
    handle.remove()

    captured = session.capture(MLP_OUT, ["ab ab ab", "xyz"])
    assert captured.shape == (2, 16)
    assert torch.allclose(captured[1], seen["out"][0, -1], atol=1e-6)


def test_capture_does_not_retain_earlier_batches(tiny_model) -> None:
    # Regression (live GPU OOM): capture kept row *views* into every batch's full
    # (batch x seq x out) activation, so all batches stayed alive until return.
    import weakref

    class Tracked(torch.Tensor):
        live: list = []  # weakrefs (WeakSet would compare tensors with ==)

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            out = super().__torch_function__(func, types, args, kwargs or {})
            for t in out if isinstance(out, (list, tuple)) else [out]:
                if isinstance(t, Tracked) and t.dim() == 3:  # full activations only
                    cls.live.append(weakref.ref(t))
            return out

    alive_at_batch_start: list[int] = []

    def tag(_m, _i, out):
        alive_at_batch_start.append(sum(r() is not None for r in Tracked.live))
        tracked = out.as_subclass(Tracked)
        Tracked.live.append(weakref.ref(tracked))
        return tracked

    handle = tiny_model.get_submodule(MLP_OUT).register_forward_hook(tag)
    try:
        session = make_session(tiny_model)  # batch_size=2 -> 3 batches below
        session.capture(MLP_OUT, ["ab ab", "xyz", "count", "hi there", "abc", "de"])
    finally:
        handle.remove()
    assert alive_at_batch_start == [0, 0, 0], alive_at_batch_start


def test_capture_many_reads_several_modules_in_one_forward_pass(tiny_model) -> None:
    session = make_session(tiny_model, budget=3)
    prompts = ["ab ab ab", "xyz"]
    many = session.capture_many([MLP_OUT, ATTN_OUT], prompts)
    assert session.budget.used == 1
    assert set(many) == {MLP_OUT, ATTN_OUT}
    # Same rows the (independently hook-checked) single-module capture returns.
    assert torch.allclose(many[MLP_OUT], session.capture(MLP_OUT, prompts))
    assert torch.allclose(many[ATTN_OUT], session.capture(ATTN_OUT, prompts))
    with pytest.raises(ValueError, match="at least one module"):
        session.capture_many([], prompts)


# --------------------------------------------------------------- export


def test_exported_adapter_reproduces_the_in_place_edit(
    tiny_model, tmp_path: Path
) -> None:
    peft = pytest.importorskip("peft")
    session = make_session(tiny_model)
    mlp_w = tiny_model.get_submodule(MLP_OUT).weight.detach().clone()
    session.apply(MLP_OUT, direction_ablation(mlp_w, torch.randn(16), k=8))
    # A rank-2 edit on another module forces rank padding in the export.
    session.apply(
        ATTN_OUT, LowRankDelta(B=torch.randn(16, 2) * 0.3, A=torch.randn(2, 16))
    )
    edited = {
        name: tiny_model.get_submodule(name).weight.detach().clone()
        for name in (MLP_OUT, ATTN_OUT)
    }
    want = logits(tiny_model, "ab ab ab")

    out = session.export(tmp_path / "adapter")
    session.revert()
    session.revert()

    model = peft.PeftModel.from_pretrained(tiny_model, out)
    with torch.no_grad():
        got = model(input_ids=torch.tensor([encode("ab ab ab")])).logits[0]
    assert torch.allclose(got, want, atol=1e-5)

    merged = model.merge_and_unload()
    for name, weight in edited.items():
        assert torch.allclose(merged.get_submodule(name).weight, weight, atol=1e-6)


def test_export_manifest_lists_only_active_edits_and_budget(
    tiny_model, tmp_path: Path
) -> None:
    session = make_session(tiny_model, budget=5)
    w = tiny_model.get_submodule(MLP_OUT).weight.detach().clone()
    session.apply(MLP_OUT, direction_ablation(w, torch.randn(16), k=8), note="kept")
    session.apply(
        ATTN_OUT, LowRankDelta(B=torch.ones(16, 1), A=torch.ones(1, 16)), note="dropped"
    )
    session.revert()
    session.evaluate()

    out = session.export(tmp_path / "adapter")
    manifest = json.loads((out / "weight_space_manifest.json").read_text())
    assert [e["note"] for e in manifest["edits"]] == ["kept"]
    assert manifest["edits"][0]["module"] == MLP_OUT
    # direction_ablation with k=8 rewrites 8 columns of a 16-row weight.
    assert manifest["edits"][0]["changed_weights"] == 8 * 16
    assert manifest["budget"] == {"limit": 5, "used": 1, "ledger": ["evaluate"]}
    assert manifest["base_model"] == "tiny-qwen3"


def test_export_without_edits_is_refused(tiny_model, tmp_path: Path) -> None:
    session = make_session(tiny_model)
    with pytest.raises(ValueError, match="no active edits"):
        session.export(tmp_path / "adapter")


# --------------------------------------------------------------- MCP tool functions


def test_mcp_tool_flow_enforces_budget_and_exports(tiny_model, tmp_path: Path) -> None:
    sid = mcp_tools.register_session(make_session(tiny_model, budget=3))
    try:
        cap = mcp_tools.ws_capture_direction(
            sid, MLP_OUT, ["ab ab ab", "xyz xyz"], ["count: one", "hi"], name="loop"
        )
        assert cap["direction"] == "loop"
        assert len(cap["top_columns"]) == 8
        assert cap["budget"]["used"] == 1

        edit = mcp_tools.ws_ablate(sid, MLP_OUT, "loop", k=8, alpha=1.0)
        assert edit["changed_weights"] == 8 * 16

        metrics = mcp_tools.ws_evaluate(sid)
        assert set(metrics["metrics"]) == set(SPLITS)
        assert metrics["active_edits"] == [edit["edit_id"]]

        exported = mcp_tools.ws_export(sid, str(tmp_path / "adapter"))
        assert (Path(exported["path"]) / "adapter_model.safetensors").exists()

        mcp_tools.ws_evaluate(sid)  # third and last forward pass
        refused = mcp_tools.ws_evaluate(sid)
        assert "budget" in refused["error"]

        assert mcp_tools.ws_revert(sid)["reverted"] == edit["edit_id"]
    finally:
        mcp_tools.ws_close(sid)


def test_mcp_tools_report_errors_instead_of_raising(tiny_model) -> None:
    assert "unknown session" in mcp_tools.ws_evaluate("nope")["error"]
    sid = mcp_tools.register_session(make_session(tiny_model))
    try:
        assert (
            "unknown direction"
            in mcp_tools.ws_ablate(sid, MLP_OUT, "missing", k=4)["error"]
        )
        assert "no edits" in mcp_tools.ws_revert(sid)["error"]
    finally:
        mcp_tools.ws_close(sid)


def test_ws_close_frees_the_session(tiny_model) -> None:
    sid = mcp_tools.register_session(make_session(tiny_model))
    assert mcp_tools.ws_close(sid) == {"closed": sid}
    assert "unknown session" in mcp_tools.ws_modules(sid)["error"]


# --------------------------------------------------------------- splits files


def test_load_splits_reads_prompt_continuation_rows(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps({"loop": [{"prompt": "a a", "continuation": " a"}], "ok": []})
    )
    assert load_splits(path) == {"loop": [("a a", " a")], "ok": []}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"loop": [{"prompt": "a a"}]}, "loop\\[0\\]"),
        ({"loop": [{"prompt": "a", "continuation": ""}]}, "empty"),
        ({"loop": [{"prompt": "", "continuation": "a"}]}, "empty"),
        ({"loop": "not a list"}, "list"),
        (["not", "a", "mapping"], "object"),
    ],
)
def test_load_splits_rejects_malformed_files(tmp_path: Path, payload, message) -> None:
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        load_splits(path)
