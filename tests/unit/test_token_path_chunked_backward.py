"""Chunked backward on the batched per-token GRPO path.

The token path backpropagates one row chunk at a time (``backward_fn``), so
peak activation memory is bounded by one chunk instead of every chunk of every
group. These tests pin that the chunked gradients equal the gradients of the
original single-backward loss (one full-group forward, ``policy_loss`` on the
whole group, population reduction, one ``backward``) for every aggregation, with
KL and entropy on and off, and that only one chunk's graph is alive at once.
"""

from __future__ import annotations

import copy
import weakref
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.core.trajectory import (  # noqa: E402
    ConversationTurn,
    MultiTurnTrajectory,
    TrajectoryGroup,
)
from stateset_agents.training import loss_computation as lc  # noqa: E402
from stateset_agents.training import objectives as O  # noqa: E402
from stateset_agents.training import rl_losses  # noqa: E402

VOCAB = 211


def _tiny_model(seed: int = 0, dtype=torch.float64):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_embd=32,
            n_layer=2,
            n_head=2,
            vocab_size=VOCAB,
            n_positions=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    return model.to(dtype)


def _traj(reward, seed, n_assistant):
    g = torch.Generator().manual_seed(seed)
    turns = []
    for k in range(n_assistant):
        turns.append(ConversationTurn(role="user", content=f"q{k}"))
        plen = 3 + int(torch.randint(0, 5, (1,), generator=g))
        rlen = 2 + int(torch.randint(0, 6, (1,), generator=g))
        turns.append(
            ConversationTurn(
                role="assistant",
                content="x",
                metadata={
                    "prompt_token_ids": torch.randint(
                        0, VOCAB, (plen,), generator=g
                    ).tolist(),
                    "token_ids": torch.randint(0, VOCAB, (rlen,), generator=g).tolist(),
                },
            )
        )
    return MultiTurnTrajectory(turns=turns, total_reward=reward)


def _groups():
    """Two groups of different sizes and turn counts (7 and 4 rows)."""
    first = TrajectoryGroup(
        scenario_id="a",
        trajectories=[
            _traj(r, seed=10 + i, n_assistant=1 + i % 2)
            for i, r in enumerate((1.0, 0.0, 0.5, 0.25, 0.9))
        ],
    )
    second = TrajectoryGroup(
        scenario_id="b",
        trajectories=[
            _traj(r, seed=40 + i, n_assistant=2) for i, r in enumerate((0.3, 0.8))
        ],
    )
    return [first, second]


def _config(**over):
    base = {
        "max_prompt_length": 64,
        "max_completion_length": 16,
        "clip_ratio": 0.2,
        "seq_clip_ratio": 3e-4,
        "entropy_coef": 0.0,
        "advantage_normalization": True,
        "baseline_type": "group_mean",
        "reward_clip": None,
        "generation_batch_size": 0,
        "objective": None,
        "objective_overrides": None,
        "old_logprobs_source": "recompute",
    }
    base.update(over)
    return SimpleNamespace(**base)


def _pad(rows):
    width = max(len(p) + len(r) for p, r, _ in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    attn = torch.zeros(len(rows), width, dtype=torch.long)
    resp = torch.zeros(len(rows), width)
    for i, (p, r, _) in enumerate(rows):
        seq = p + r
        ids[i, : len(seq)] = torch.tensor(seq)
        attn[i, : len(seq)] = 1
        resp[i, len(p) : len(seq)] = 1
    return ids, attn, resp


def _single_backward_reference(groups, config, model, *, beta, ref, old_logprobs):
    """The pre-chunking algorithm: one graph for the whole batch, one backward.

    fp32 models use ``rl_losses.gather_token_logprobs`` exactly as before;
    fp64 models gather in fp64 so the comparison isolates the chunking.
    """
    gather_dtype = None if model.dtype == torch.float32 else model.dtype
    objective = lc._token_objective(config)
    entropy_coef = float(config.entropy_coef)
    use_kl = beta > 0 and ref is not None
    if use_kl:
        objective = lc._token_kl_objective(objective, config, beta=beta)
    losses = []
    totals = dict.fromkeys(
        ("kl_penalty", "entropy", "ratio_mean", "clip_fraction"), 0.0
    )
    n_trajectories = sum(len(group.trajectories) for group in groups)
    n_tokens = sum(
        len(reply)
        for group in groups
        for _, reply, _ in lc._token_rows_for_group(group)
    )
    for gi, group in enumerate(groups):
        rows = lc._token_rows_for_group(group)
        rewards = torch.tensor([float(t.total_reward) for t in group.trajectories])
        adv = lc._token_advantages(rewards, config, objective)
        row_adv = adv[torch.tensor([ti for _, _, ti in rows])]
        ids, attn, resp = _pad(rows)
        logits = model(input_ids=ids, attention_mask=attn).logits
        lp, mask = rl_losses.gather_token_logprobs(
            logits, ids, resp, dtype=gather_dtype
        )
        entropy = None
        if entropy_coef > 0:
            lsm = torch.log_softmax(
                logits[:, :-1, :].to(gather_dtype or torch.float32), dim=-1
            )
            entropy = -(lsm.exp() * lsm).sum(-1)
        lp_ref = None
        if use_kl:
            with torch.no_grad():
                ref_logits = ref(input_ids=ids, attention_mask=attn).logits
                lp_ref, _ = rl_losses.gather_token_logprobs(
                    ref_logits, ids, resp, dtype=gather_dtype
                )
        result = O.policy_loss(
            logp_cur=lp,
            mask=mask,
            advantages=row_adv,
            objective=objective.with_(entropy_coef=entropy_coef),
            logp_old=None if old_logprobs is None else old_logprobs[gi],
            logp_ref=lp_ref,
            entropy=entropy,
            total_rows=n_trajectories,
            total_tokens=n_tokens,
        )
        losses.append(result.loss)
        token_count = int(mask.count_nonzero())
        kl_weight = len(rows) if objective.kl == "k3_sequence" else token_count
        totals["kl_penalty"] += result.metrics["kl"] * kl_weight
        totals["entropy"] += result.metrics["entropy"] * token_count
        totals["ratio_mean"] += result.metrics["ratio_mean"] * token_count
        totals["clip_fraction"] += result.metrics["clip_fraction"] * token_count
    total = torch.stack(losses).sum()
    total.backward()
    metrics = {
        key: value
        / (
            n_trajectories
            if key == "kl_penalty" and objective.kl == "k3_sequence"
            else n_tokens
        )
        for key, value in totals.items()
    }
    return total.detach(), metrics


def _chunked(groups, config, model, *, beta, ref, old_logprobs, backward_fn):
    agent = SimpleNamespace(model=model, tokenizer=None)
    if beta > 0 or ref is not None:
        return lc.compute_enhanced_grpo_loss(
            groups,
            beta,
            config,
            agent,
            reference_model=ref,
            old_logprobs=old_logprobs,
            backward_fn=backward_fn,
        )
    return lc.compute_grpo_loss(
        groups,
        config,
        agent,
        0.0,
        0,
        None,
        old_logprobs=old_logprobs,
        backward_fn=backward_fn,
    )


def _grads(model):
    return [
        (
            p.grad.detach().clone()
            if p.grad is not None
            else torch.zeros_like(p, memory_format=torch.preserve_format)
        )
        for p in model.parameters()
    ]


def _max_abs_diff(a, b):
    return max(float((x - y).abs().max()) for x, y in zip(a, b, strict=True))


def _old_policy_snapshot(groups, config, dtype=torch.float64):
    """Per-token log-probs of a perturbed policy, so ratios leave 1 and clip."""
    stale = _tiny_model(0, dtype)
    with torch.no_grad():
        for p in stale.parameters():
            p.add_(0.05 * torch.randn_like(p))
    agent = SimpleNamespace(model=stale, tokenizer=None)
    return lc.compute_token_old_logprobs(groups, config, agent)


def _compare(objective, kl, entropy, chunk, off_policy=False, dtype=torch.float64):
    config = _config(
        objective=objective,
        entropy_coef=0.05 if entropy else 0.0,
        generation_batch_size=chunk,
    )
    groups = _groups()
    old = _old_policy_snapshot(groups, config, dtype) if off_policy else None
    beta = 0.07 if kl else 0.0
    ref = _tiny_model(1, dtype) if kl else None

    reference_model = _tiny_model(0, dtype)
    want_loss, want_metrics = _single_backward_reference(
        groups, config, reference_model, beta=beta, ref=ref, old_logprobs=old
    )
    want = _grads(reference_model)

    model = _tiny_model(0, dtype)
    calls = []

    def backward_fn(part):
        calls.append(float(part.detach()))
        part.backward()

    out = _chunked(
        groups,
        config,
        model,
        beta=beta,
        ref=ref,
        old_logprobs=old,
        backward_fn=backward_fn,
    )
    got = _grads(model)
    return out, want_loss, want_metrics, want, got, calls


OBJECTIVE_CASES = [None, "grpo", "gspo", "gspo_token", "dapo", "dr_grpo", "cispo"]


@pytest.mark.parametrize("chunk", [1, 3, 0], ids=["chunk1", "chunk3", "all"])
@pytest.mark.parametrize("entropy", [False, True], ids=["noent", "ent"])
@pytest.mark.parametrize("kl", [False, True], ids=["nokl", "kl"])
@pytest.mark.parametrize("objective", OBJECTIVE_CASES, ids=str)
def test_chunked_backward_gradients_equal_single_backward(
    objective, kl, entropy, chunk
):
    out, want_loss, want_metrics, want, got, calls = _compare(
        objective, kl, entropy, chunk
    )
    assert out["path"] == "token"
    assert out["backward_done"] is True
    assert not out["total_loss"].requires_grad
    rows = (7, 4)
    size = chunk or max(rows)
    assert len(calls) == sum(-(-n // size) for n in rows)
    assert _max_abs_diff(got, want) < 1e-10
    torch.testing.assert_close(out["total_loss"], want_loss, atol=1e-10, rtol=0)
    assert sum(calls) == pytest.approx(float(want_loss), abs=1e-10)
    for key, value in want_metrics.items():
        assert float(out[key]) == pytest.approx(value, abs=1e-9), key


@pytest.mark.parametrize("chunk", [1, 3, 0], ids=["chunk1", "chunk3", "all"])
@pytest.mark.parametrize("objective", OBJECTIVE_CASES, ids=str)
def test_chunked_backward_matches_off_policy_inner_updates(objective, chunk):
    out, want_loss, want_metrics, want, got, _ = _compare(
        objective, kl=True, entropy=True, chunk=chunk, off_policy=True
    )
    assert out["off_policy"] is True
    assert want_metrics["clip_fraction"] > 0 or objective in ("cispo",)
    assert _max_abs_diff(got, want) < 1e-10
    torch.testing.assert_close(out["total_loss"], want_loss, atol=1e-10, rtol=0)
    for key, value in want_metrics.items():
        assert float(out[key]) == pytest.approx(value, abs=1e-9), key


@pytest.mark.parametrize("chunk", [1, 3, 0], ids=["chunk1", "chunk3", "all"])
@pytest.mark.parametrize("objective", OBJECTIVE_CASES, ids=str)
def test_fp32_chunked_backward_matches_the_original_fp32_loss(objective, chunk):
    """Against the unmodified fp32 reference (``rl_losses`` gather included)."""
    out, want_loss, _, want, got, _ = _compare(
        objective,
        kl=True,
        entropy=True,
        chunk=chunk,
        off_policy=True,
        dtype=torch.float32,
    )
    scale = max(float(w.abs().max()) for w in want)
    assert _max_abs_diff(got, want) <= 1e-5 * max(scale, 1.0)
    torch.testing.assert_close(out["total_loss"], want_loss, atol=1e-5, rtol=1e-5)


def test_without_backward_fn_the_loss_keeps_one_graph_with_the_same_gradient():
    groups = _groups()
    config = _config(objective="gspo", entropy_coef=0.05, generation_batch_size=2)
    reference_model = _tiny_model(0)
    _single_backward_reference(
        groups, config, reference_model, beta=0.0, ref=None, old_logprobs=None
    )
    model = _tiny_model(0)
    out = _chunked(
        groups,
        config,
        model,
        beta=0.0,
        ref=None,
        old_logprobs=None,
        backward_fn=None,
    )
    assert out.get("backward_done", False) is False
    assert out["total_loss"].requires_grad
    out["total_loss"].backward()
    assert _max_abs_diff(_grads(model), _grads(reference_model)) < 1e-10


def test_backward_fn_scaling_is_linear():
    """The trainer divides each chunk by the accumulation steps; the summed
    gradient must be the full loss's gradient divided the same way."""
    groups = _groups()
    config = _config(objective="dapo", entropy_coef=0.05, generation_batch_size=2)
    reference_model = _tiny_model(0)
    _single_backward_reference(
        groups, config, reference_model, beta=0.0, ref=None, old_logprobs=None
    )
    model = _tiny_model(0)
    _chunked(
        groups,
        config,
        model,
        beta=0.0,
        ref=None,
        old_logprobs=None,
        backward_fn=lambda part: (part / 4).backward(),
    )
    want = [g / 4 for g in _grads(reference_model)]
    assert _max_abs_diff(_grads(model), want) < 1e-10


# --- the gather -----------------------------------------------------------------


def test_response_logprobs_match_rl_losses_gather_and_full_softmax_entropy():
    model = _tiny_model(0, torch.float32)
    rows = lc._token_rows_for_group(_groups()[0])
    ids, attn, resp = _pad(rows)
    logits = model(input_ids=ids, attention_mask=attn).logits
    want_lp, want_mask = rl_losses.gather_token_logprobs(logits, ids, resp)
    lsm = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    want_ent = -(lsm.exp() * lsm).sum(-1) * want_mask

    for with_entropy in (False, True):
        lp, mask, ent = lc._response_token_logprobs(
            logits, ids, resp, want_entropy=with_entropy
        )
        torch.testing.assert_close(mask, want_mask)
        torch.testing.assert_close(lp, want_lp, atol=1e-5, rtol=1e-5)
        if with_entropy:
            torch.testing.assert_close(ent * mask, want_ent, atol=1e-5, rtol=1e-5)
        else:
            assert ent is None
    # gradients agree as well
    g_want = torch.autograd.grad(want_lp.sum(), logits, retain_graph=True)[0]
    lp, _, _ = lc._response_token_logprobs(logits, ids, resp, want_entropy=False)
    g_got = torch.autograd.grad(lp.sum(), logits, retain_graph=True)[0]
    torch.testing.assert_close(g_got, g_want, atol=1e-6, rtol=1e-5)


def test_response_logprobs_handle_bf16_logits_in_fp32():
    logits = torch.randn(2, 6, VOCAB).to(torch.bfloat16).requires_grad_()
    ids = torch.randint(0, VOCAB, (2, 6))
    resp = torch.zeros(2, 6)
    resp[0, 3:] = 1
    resp[1, 2:5] = 1
    want, _ = rl_losses.gather_token_logprobs(logits, ids, resp)
    got, _, _ = lc._response_token_logprobs(logits, ids, resp, want_entropy=False)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_response_logprobs_never_materialize_full_width_fp32_logits():
    """Only response positions are upcast: the largest fp32 tensor saved for
    backward has one row per response token, not one per padded position."""
    logits = torch.randn(3, 40, VOCAB, requires_grad=True).to(torch.bfloat16)
    ids = torch.randint(0, VOCAB, (3, 40))
    resp = torch.zeros(3, 40)
    resp[:, 35:] = 1  # 5 response tokens per row, long prompts
    biggest = []

    def pack(t):
        if t.dtype == torch.float32 and t.dim() == 2:
            biggest.append(t.shape[0])
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        lc._response_token_logprobs(logits, ids, resp, want_entropy=True)
    assert biggest and max(biggest) <= 3 * 5


# --- memory bound ---------------------------------------------------------------


class _Holder:
    __slots__ = ("tensor", "__weakref__")

    def __init__(self, tensor):
        self.tensor = tensor


class _SavedTensorMeter:
    """Bytes of tensors autograd holds for backward, live and peak."""

    def __init__(self):
        self.live = 0
        self.peak = 0

    def _release(self, size):
        self.live -= size

    def pack(self, tensor):
        holder = _Holder(tensor)
        size = tensor.numel() * tensor.element_size()
        self.live += size
        self.peak = max(self.peak, self.live)
        weakref.finalize(holder, self._release, size)
        return holder

    @staticmethod
    def unpack(holder):
        return holder.tensor


def _peak_saved_bytes(backward_fn, chunk=1):
    groups = _groups()
    config = _config(objective="gspo", entropy_coef=0.05, generation_batch_size=chunk)
    model = _tiny_model(0)
    meter = _SavedTensorMeter()
    with torch.autograd.graph.saved_tensors_hooks(meter.pack, meter.unpack):
        out = _chunked(
            groups,
            config,
            model,
            beta=0.07,
            ref=_tiny_model(1),
            old_logprobs=None,
            backward_fn=backward_fn,
        )
        if backward_fn is None:
            out["total_loss"].backward()
    return meter.peak


def test_peak_graph_memory_is_bounded_by_one_chunk():
    retained = _peak_saved_bytes(None)
    chunked = _peak_saved_bytes(lambda part: part.backward())
    # 11 one-row chunks: the retained graph holds all of them at once.
    assert chunked * 4 < retained, (chunked, retained)


@pytest.mark.asyncio
@pytest.mark.parametrize("steps", [1, 2])
async def test_native_streaming_releases_chunks_and_counts_each_batch_once(
    tmp_path, steps
):
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    groups = _groups()
    config = _config(
        objective="gspo",
        entropy_coef=0.05,
        generation_batch_size=1,
        gradient_accumulation_steps=steps,
        max_grad_norm=100000.0,
        bf16=False,
        fp16=False,
        beta=0.0,
        zero_signal_patience=0,
        output_dir=str(tmp_path),
    )
    reference = _tiny_model(0)
    _single_backward_reference(
        groups, config, reference, beta=0, ref=None, old_logprobs=None
    )
    model = _tiny_model(0)
    trainer = MultiTurnGRPOTrainer(
        SimpleNamespace(model=model), object(), config=config
    )
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    meter = _SavedTensorMeter()
    with torch.autograd.graph.saved_tensors_hooks(meter.pack, meter.unpack):
        await trainer.training_step(groups)
    if steps > 1:
        state = trainer._grpo_accumulation
        assert state["policy_count"] == 7
        assert state["entropy_count"] == sum(
            len(reply)
            for group in groups
            for _, reply, _ in lc._token_rows_for_group(group)
        )
        trainer._flush_accumulation(torch)
    for actual, expected in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(
            actual, expected - 0.01 * expected.grad, atol=1e-10, rtol=0
        )
    assert trainer.global_step == 1
    retained = _peak_saved_bytes(None)
    assert meter.peak * 4 < retained, (meter.peak, retained)


def test_backward_runs_after_every_chunk_forward():
    groups = _groups()
    config = _config(objective="dapo", entropy_coef=0.05, generation_batch_size=3)
    model = _tiny_model(0)
    events = []
    orig = model.forward

    def spy(*args, **kwargs):
        events.append("F")
        return orig(*args, **kwargs)

    model.forward = spy
    _chunked(
        groups,
        config,
        model,
        beta=0.0,
        ref=None,
        old_logprobs=None,
        backward_fn=lambda part: (events.append("B"), part.backward()),
    )
    # 7 rows -> chunks of 3,3,1; 4 rows -> 3,1
    assert "".join(events) == "FB" * 5


def test_a_non_finite_chunk_stops_before_later_chunks():
    groups = _groups()
    config = _config(generation_batch_size=2)
    model = _tiny_model(0)
    seen = []

    def backward_fn(part):
        seen.append(part)
        raise ValueError("GRPO loss must be a finite differentiable scalar")

    with pytest.raises(ValueError):
        _chunked(
            groups,
            config,
            model,
            beta=0.0,
            ref=None,
            old_logprobs=None,
            backward_fn=backward_fn,
        )
    assert len(seen) == 1


def test_group_expectation_ratio_is_rejected_with_backward_fn():
    with pytest.raises(ValueError):
        _compare("gepo", kl=False, entropy=False, chunk=2)


def test_seeded_models_are_identical():
    """The comparisons rely on equal seeds giving equal starting weights."""
    a, b = _tiny_model(0), copy.deepcopy(_tiny_model(0))
    assert all(
        torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters(), strict=True)
    )
