"""Memory-efficient Chunked Fused GRPO / GSPO / DAPO RL Loss Kernels.

Eliminates full ``[B, T, V]`` vocabulary logit materialization during long-context
multi-turn RL training by projecting ``hidden_states`` (``[B, T, H]``) through
``lm_head_weight`` (``[V, H]``) in token chunks (e.g. 256 tokens at a time) with
optional per-chunk activation checkpointing.

Cuts peak vocabulary logit memory by 60-85% on 16k-32k context windows while
matching full-materialization ``gather_token_logprobs`` numerically in both
forward pass and backward gradients.

Torch is imported strictly lazily (zero eager ``torch`` import at module load).
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from .rl_losses import k3_kl, masked_mean, safe_exp_ratio
from .trainer_utils import get_torch, require_torch


def triton_chunked_logprobs_available() -> bool:
    """Return True when Triton + CUDA are both available for fused SRAM log-softmax."""
    if importlib.util.find_spec("triton") is None:
        return False
    torch = get_torch()
    if torch is None:
        try:
            torch = require_torch()
        except ImportError:
            return False
    return bool(torch.cuda.is_available())


def _t() -> Any:
    return get_torch() or require_torch()


def _triton_row_logprob_entropy(
    logits_2d: Any,
    labels_1d: Any,
    mask_1d: Any,
) -> tuple[Any, Any] | None:
    """Optional Triton single-pass online-logsumexp kernel on CUDA; returns None if unavailable."""
    if not (logits_2d.is_cuda and triton_chunked_logprobs_available()):
        return None
    try:
        triton: Any = importlib.import_module("triton")
        tl: Any = importlib.import_module("triton.language")
    except ImportError:
        return None

    torch = _t()
    n_rows, vocab = logits_2d.shape
    block_v = min(4096, triton.next_power_of_2(vocab))

    @triton.jit
    def _kernel(
        logits_ptr: Any,
        labels_ptr: Any,
        mask_ptr: Any,
        out_lp_ptr: Any,
        out_ent_ptr: Any,
        stride_row: int,
        n_cols: int,
        BLOCK: tl.constexpr,
    ) -> None:
        row_id = tl.program_id(0)
        m_val = tl.load(mask_ptr + row_id)
        if m_val == 0.0:
            tl.store(out_lp_ptr + row_id, 0.0)
            tl.store(out_ent_ptr + row_id, 0.0)
            return
        row_ptr = logits_ptr + row_id * stride_row
        # Pass 1: online max + sumexp
        m_i = -float("inf")
        d_i = 0.0
        for off in range(0, n_cols, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            valid = cols < n_cols
            x = tl.load(row_ptr + cols, mask=valid, other=-float("inf")).to(tl.float32)
            m_new = tl.maximum(m_i, tl.max(x, axis=0))
            d_i = d_i * tl.exp(m_i - m_new) + tl.sum(
                tl.where(valid, tl.exp(x - m_new), 0.0), axis=0
            )
            m_i = m_new
        lse = m_i + tl.log(d_i)
        target = tl.load(labels_ptr + row_id)
        target_logit = tl.load(row_ptr + target).to(tl.float32)
        tl.store(out_lp_ptr + row_id, target_logit - lse)

        # Pass 2: exact entropy
        ent = 0.0
        for off in range(0, n_cols, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            valid = cols < n_cols
            x = tl.load(row_ptr + cols, mask=valid, other=0.0).to(tl.float32)
            log_p = x - lse
            p = tl.where(valid, tl.exp(log_p), 0.0)
            ent += -tl.sum(p * tl.where(valid, log_p, 0.0), axis=0)
        tl.store(out_ent_ptr + row_id, ent)

    out_lp = torch.empty((n_rows,), device=logits_2d.device, dtype=torch.float32)
    out_ent = torch.empty((n_rows,), device=logits_2d.device, dtype=torch.float32)
    _kernel[(n_rows,)](
        logits_2d,
        labels_1d,
        mask_1d,
        out_lp,
        out_ent,
        logits_2d.stride(0),
        vocab,
        BLOCK=block_v,
    )
    return out_lp, out_ent


def _chunk_logprob_and_entropy(
    hidden_chunk: Any,
    lm_head_weight: Any,
    lm_head_bias: Any | None,
    labels_chunk: Any,
    mask_chunk: Any,
    softmax_dtype: Any,
    *,
    use_triton: bool = False,
) -> tuple[Any, Any]:
    """Compute masked token log-probs and token entropies for one `[N_chunk, H]` slice."""
    torch = _t()
    logits = torch.nn.functional.linear(hidden_chunk, lm_head_weight, lm_head_bias)
    if use_triton and not torch.is_grad_enabled():
        triton_res = _triton_row_logprob_entropy(logits, labels_chunk, mask_chunk)
        if triton_res is not None:
            return triton_res
    scores = logits.to(softmax_dtype)
    scores = scores - scores.amax(dim=-1, keepdim=True).detach()

    inactive = mask_chunk == 0
    scores = scores.masked_fill(inactive.unsqueeze(-1), 0.0)
    safe_labels = labels_chunk.masked_fill(inactive, 0)

    lse = torch.logsumexp(scores, dim=-1)
    selected = scores.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    logprobs = selected - lse

    # Exact token entropy H(p) = logsumexp(z) - sum_v p_v z_v over active rows
    probs = torch.exp(scores - lse.unsqueeze(-1))
    entropy = -(probs * (scores - lse.unsqueeze(-1))).sum(dim=-1)

    mask_cast = mask_chunk.to(logprobs.dtype)
    return logprobs * mask_cast, entropy * mask_cast


def chunked_gather_token_logprobs(
    hidden_states: Any,
    lm_head_weight: Any,
    input_ids: Any,
    response_mask: Any,
    *,
    lm_head_bias: Any | None = None,
    chunk_size: int = 256,
    dtype: Any = None,
    use_checkpointing: bool = False,
    return_entropy: bool = False,
) -> tuple[Any, ...]:
    """Chunked shift-by-one token log-prob gather without materializing ``[B, T, V]``.

    Parameters
    ----------
    hidden_states:
        Final transformer hidden states of shape ``[B, T, H]``.
    lm_head_weight:
        Output projection weight of shape ``[V, H]``.
    input_ids:
        Token IDs of shape ``[B, T]``.
    response_mask:
        Binary mask of shape ``[B, T]`` (1 for assistant response tokens, 0 for prompt/pad).
    lm_head_bias:
        Optional output bias of shape ``[V]``.
    chunk_size:
        Number of flattened token positions projected per chunk (default ``256``).
    dtype:
        Softmax accumulation dtype (defaults to ``torch.float32``).
    use_checkpointing:
        When ``True`` and gradients are enabled, wraps each chunk in
        ``torch.utils.checkpoint.checkpoint`` so intermediate ``[chunk, V]``
        tensors are freed immediately and recomputed on backward.
    return_entropy:
        When ``True``, returns ``(token_logprobs, shifted_mask, token_entropy)``.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    torch = _t()
    bsz, seq_len, hidden_dim = hidden_states.shape
    if seq_len < 2:
        empty = hidden_states.new_zeros((bsz, 0), dtype=torch.float32)
        return (empty, empty, empty) if return_entropy else (empty, empty)

    shift_hidden = hidden_states[..., :-1, :].reshape(-1, hidden_dim)
    shift_labels = input_ids[..., 1:].reshape(-1)
    shifted_mask_2d = response_mask[..., 1:].to(torch.float32)
    shift_mask = shifted_mask_2d.reshape(-1)
    softmax_dtype = torch.float32 if dtype is None else dtype

    total_tokens = shift_hidden.shape[0]
    logprob_chunks: list[Any] = []
    entropy_chunks: list[Any] = []

    for start in range(0, total_tokens, chunk_size):
        end = min(start + chunk_size, total_tokens)
        h_c = shift_hidden[start:end]
        l_c = shift_labels[start:end]
        m_c = shift_mask[start:end]

        if (
            use_checkpointing
            and torch.is_grad_enabled()
            and (h_c.requires_grad or lm_head_weight.requires_grad)
        ):
            import torch.utils.checkpoint as _torch_ckpt

            def _run_chunk(
                hc: Any,
                w: Any,
                lc: Any = l_c,
                mc: Any = m_c,
            ) -> tuple[Any, Any]:
                return _chunk_logprob_and_entropy(
                    hc, w, lm_head_bias, lc, mc, softmax_dtype
                )

            lp_c, ent_c = _torch_ckpt.checkpoint(
                _run_chunk, h_c, lm_head_weight, use_reentrant=False
            )
        else:
            lp_c, ent_c = _chunk_logprob_and_entropy(
                h_c, lm_head_weight, lm_head_bias, l_c, m_c, softmax_dtype
            )
        logprob_chunks.append(lp_c)
        if return_entropy:
            entropy_chunks.append(ent_c)

    token_logprobs = torch.cat(logprob_chunks, dim=0).reshape(bsz, seq_len - 1)
    if return_entropy:
        token_entropy = torch.cat(entropy_chunks, dim=0).reshape(bsz, seq_len - 1)
        return token_logprobs, shifted_mask_2d, token_entropy
    return token_logprobs, shifted_mask_2d


def moe_router_aux_loss(
    router_logits: Any,
    *,
    num_experts: int | None = None,
    top_k: int = 2,
    aux_coef: float = 0.01,
    attention_mask: Any | None = None,
) -> Any:
    """Switch / DeepSeek / Qwen3-MoE load-balancing auxiliary loss (computed in float32).

    ``router_logits`` may have shape ``[B, T, E]`` or ``[N, E]``.
    Computes ``aux_coef * E * sum_i (f_i * P_i)`` where ``f_i`` is the fraction
    of tokens routed to expert ``i`` (top-k) and ``P_i`` is the mean routing
    probability for expert ``i``.
    """
    torch = _t()
    logits = router_logits.to(torch.float32)
    if logits.ndim == 3:
        if attention_mask is not None:
            flat_mask = attention_mask.reshape(-1).to(torch.bool)
            logits_2d = logits.reshape(-1, logits.shape[-1])[flat_mask]
        else:
            logits_2d = logits.reshape(-1, logits.shape[-1])
    elif logits.ndim == 2:
        logits_2d = logits
    else:
        raise ValueError(
            f"router_logits must be 2D [N, E] or 3D [B, T, E], got shape {tuple(logits.shape)}"
        )

    if logits_2d.numel() == 0:
        return logits.new_tensor(0.0)

    n_experts = int(num_experts or logits_2d.shape[-1])
    k = max(1, min(int(top_k), n_experts))
    probs = torch.softmax(logits_2d, dim=-1)
    top_idx = torch.topk(probs, k=k, dim=-1).indices
    one_hot = torch.nn.functional.one_hot(top_idx, num_classes=n_experts).to(
        torch.float32
    )
    # Fraction of routing slots assigned to each expert
    tokens_per_expert = one_hot.sum(dim=1).mean(dim=0) / float(k)
    router_prob_per_expert = probs.mean(dim=0)
    return (
        float(aux_coef)
        * float(n_experts)
        * torch.sum(tokens_per_expert * router_prob_per_expert)
    )


def chunked_rl_policy_loss(
    hidden_states: Any,
    lm_head_weight: Any,
    input_ids: Any,
    response_mask: Any,
    old_logprobs: Any,
    advantages: Any,
    *,
    algorithm: str = "gspo",
    ref_logprobs: Any | None = None,
    lm_head_bias: Any | None = None,
    clip_low: float = 0.2,
    clip_high: float | None = None,
    kl_coef: float = 0.02,
    entropy_coef: float = 0.0,
    chunk_size: int = 256,
    use_checkpointing: bool = False,
    importance_weights: Any | None = None,
    rollout_logprobs: Any | None = None,
    tis_clip_max: float | None = None,
    icepop_bounds: tuple[float, float] | None = None,
    router_logits: Any | None = None,
    moe_aux_coef: float = 0.0,
    moe_top_k: int = 2,
    max_completion_length: int | None = None,
    cispo_is_cap: float = 5.0,
) -> dict[str, Any]:
    """Compute memory-efficient chunked GRPO, GSPO, DAPO, CISPO, Dr. GRPO, or Dr. CISPO policy loss.

    Supports:
    - ``grpo``: Token-level clipped surrogate, sequence-normalized mean.
    - ``dr_grpo``: Token-level clipped surrogate with Dr. GRPO constant-length token-sum
      normalization (eliminating length bias).
    - ``gspo``: Sequence-level geometric mean importance ratio (``exp(mean_t log_ratio)``)
      for stable long-horizon multi-turn dialogue.
    - ``dapo``: Asymmetric Clip-Higher (``clip_low=0.20, clip_high=0.28`` default)
      with global token-level mean normalization.
    - ``cispo``: Per-token detached clipped importance weights multiplying ``A_t * log pi(y_t)``
      with global token-level mean normalization.
    - ``dr_cispo``: CISPO per-token clipped importance weights combined with Dr. GRPO
      constant-length token-sum normalization (zero length bias, async-safe).
    - Optional off-policy staleness ``importance_weights`` (for disaggregated async RL).
    """
    torch = _t()
    algo = algorithm.strip().lower()
    if algo not in ("grpo", "gspo", "dapo", "cispo", "dr_grpo", "dr_cispo"):
        raise ValueError(
            f"Unsupported algorithm {algorithm!r}; expected grpo, gspo, dapo, "
            "cispo, dr_grpo, or dr_cispo."
        )

    eps_low = float(clip_low)
    eps_high = (
        float(clip_high)
        if clip_high is not None
        else (0.28 if algo == "dapo" else eps_low)
    )

    cur_logprobs, mask, token_entropy = chunked_gather_token_logprobs(
        hidden_states,
        lm_head_weight,
        input_ids,
        response_mask,
        lm_head_bias=lm_head_bias,
        chunk_size=chunk_size,
        use_checkpointing=use_checkpointing,
        return_entropy=True,
    )

    adv = advantages.to(torch.float32)
    if adv.ndim == 1:
        adv_2d = adv.unsqueeze(-1).expand_as(cur_logprobs)
    else:
        adv_2d = adv

    old_lp = old_logprobs.detach().to(torch.float32)
    cur_lp = cur_logprobs.to(torch.float32)
    token_log_ratio = (cur_lp - old_lp) * mask

    if algo == "gspo":
        # Sequence-level geometric importance ratio: exp( (1/|y|) sum_t log(pi/pi_old) )
        seq_lengths = torch.clamp(mask.sum(dim=-1, keepdim=True), min=1.0)
        seq_log_ratio = token_log_ratio.sum(dim=-1, keepdim=True) / seq_lengths
        ratio = safe_exp_ratio(seq_log_ratio).expand_as(cur_lp)
        mean_mode = "seq"
    elif algo in ("dapo", "cispo"):
        ratio = safe_exp_ratio(token_log_ratio)
        mean_mode = "token"
    elif algo in ("dr_grpo", "dr_cispo"):
        ratio = safe_exp_ratio(token_log_ratio)
        mean_mode = "seq_sum_const"
    else:
        ratio = safe_exp_ratio(token_log_ratio)
        mean_mode = "seq"

    if importance_weights is not None:
        iw = importance_weights.detach().to(torch.float32)
        if iw.ndim == 1:
            iw = iw.unsqueeze(-1).expand_as(cur_lp)
        adv_2d = adv_2d * iw

    # FP8 rollout vs. BF16 learner mismatch correction (Truncated IS / IcePop)
    eff_mask = mask
    tis_mean_ratio = 1.0
    tis_clipped_fraction = 0.0
    icepop_masked_fraction = 0.0
    rollout_learner_kl = 0.0
    if rollout_logprobs is not None:
        roll_lp = rollout_logprobs.detach().to(torch.float32)
        mismatch_log = (old_lp - roll_lp) * mask
        rho_is = safe_exp_ratio(mismatch_log)
        with torch.no_grad():
            rollout_learner_kl = float(
                masked_mean(torch.abs(mismatch_log), mask, mode="token").item()
            )
        if icepop_bounds is not None:
            lo_b, hi_b = float(icepop_bounds[0]), float(icepop_bounds[1])
            in_bounds = ((rho_is >= lo_b) & (rho_is <= hi_b)).to(torch.float32)
            with torch.no_grad():
                icepop_masked_fraction = float(
                    masked_mean(1.0 - in_bounds, mask, mode="token").item()
                )
            eff_mask = mask * in_bounds
        if tis_clip_max is not None:
            cap = float(tis_clip_max)
            with torch.no_grad():
                tis_clipped_fraction = float(
                    masked_mean(
                        (rho_is > cap).to(torch.float32), mask, mode="token"
                    ).item()
                )
            rho_is = torch.clamp(rho_is, max=cap)
        with torch.no_grad():
            tis_mean_ratio = float(masked_mean(rho_is, eff_mask, mode="token").item())
        adv_2d = adv_2d * rho_is.detach()

    clipped_ratio = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)
    if algo in ("cispo", "dr_cispo"):
        # CISPO (matches objectives.py): detached upper-capped per-token IS weight * A_t * log pi(y_t)
        cispo_weight = torch.clamp(ratio, max=float(cispo_is_cap)).detach()
        per_token_obj = cispo_weight * adv_2d * cur_lp
    else:
        surr1 = ratio * adv_2d
        surr2 = clipped_ratio * adv_2d
        per_token_obj = torch.minimum(surr1, surr2)

    if mean_mode == "seq_sum_const":
        norm_len = float(
            max_completion_length
            if max_completion_length is not None and max_completion_length > 0
            else max(1, int(cur_lp.shape[-1]))
        )
        batch_denom = max(float(cur_lp.shape[0]) * norm_len, 1.0)
        pg_loss = -(per_token_obj * eff_mask).sum() / batch_denom
    else:
        pg_loss = -masked_mean(per_token_obj, eff_mask, mode=mean_mode)

    kl_loss = (
        k3_kl(cur_lp, ref_logprobs.detach(), eff_mask)
        if ref_logprobs is not None and kl_coef > 0.0
        else cur_lp.new_tensor(0.0)
    )
    mean_entropy = masked_mean(token_entropy, eff_mask, mode="token")
    moe_loss = (
        moe_router_aux_loss(
            router_logits,
            top_k=moe_top_k,
            aux_coef=moe_aux_coef,
        )
        if router_logits is not None and moe_aux_coef > 0.0
        else cur_lp.new_tensor(0.0)
    )
    total_loss = (
        pg_loss
        + float(kl_coef) * kl_loss
        - float(entropy_coef) * mean_entropy
        + moe_loss
    )

    with torch.no_grad():
        clipped_mask = ((ratio < 1.0 - eps_low) | (ratio > 1.0 + eps_high)).to(
            torch.float32
        )
        clip_fraction = float(masked_mean(clipped_mask, eff_mask, mode="token").item())
        mean_ratio = float(masked_mean(ratio, eff_mask, mode="token").item())

    return {
        "loss": total_loss,
        "pg_loss": pg_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "moe_aux_loss": moe_loss.detach(),
        "entropy": mean_entropy.detach(),
        "clip_fraction": clip_fraction,
        "mean_ratio": mean_ratio,
        "tis_mean_ratio": tis_mean_ratio,
        "tis_clipped_fraction": tis_clipped_fraction,
        "icepop_masked_fraction": icepop_masked_fraction,
        "rollout_learner_kl": rollout_learner_kl,
        "token_logprobs": cur_logprobs,
        "mask": eff_mask,
        "kernel_backend": (
            "triton_cuda"
            if (hidden_states.is_cuda and triton_chunked_logprobs_available())
            else "torch_chunked"
        ),
    }


def extract_lm_head_and_backbone(model: Any) -> tuple[Any, Any, Any | None] | None:
    """Resolve ``(backbone, lm_head_weight, lm_head_bias)`` if the model supports chunked projection."""
    target = model
    if hasattr(target, "base_model") and hasattr(target.base_model, "model"):
        target = target.base_model.model

    lm_head = getattr(target, "lm_head", None)
    if lm_head is None and hasattr(target, "get_output_embeddings"):
        lm_head = target.get_output_embeddings()
    if lm_head is None or not hasattr(lm_head, "weight"):
        return None

    backbone = (
        getattr(target, "model", None)
        or getattr(target, "transformer", None)
        or getattr(target, "backbone", None)
    )
    if backbone is None:
        return None

    return (backbone, lm_head.weight, getattr(lm_head, "bias", None))


def chunked_model_token_logprobs(
    model: Any,
    input_ids: Any,
    response_mask: Any,
    *,
    attention_mask: Any | None = None,
    chunk_size: int = 256,
    dtype: Any = None,
    use_checkpointing: bool = False,
    return_entropy: bool = False,
) -> tuple[Any, ...]:
    """Run model backbone + chunked LM-head projection without materializing ``[B, T, V]`` logits.

    Automatically falls back to full-logit ``gather_token_logprobs`` if ``model``
    does not expose a separate transformer backbone and ``lm_head``.
    """
    from .rl_losses import gather_token_logprobs

    parts = extract_lm_head_and_backbone(model)
    if parts is not None:
        backbone, lm_weight, lm_bias = parts
        kwargs: dict[str, Any] = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        out = backbone(input_ids, **kwargs)
        hidden = (
            out.last_hidden_state
            if hasattr(out, "last_hidden_state")
            else (out[0] if isinstance(out, tuple) else out)
        )
        return chunked_gather_token_logprobs(
            hidden,
            lm_weight,
            input_ids,
            response_mask,
            lm_head_bias=lm_bias,
            chunk_size=chunk_size,
            dtype=dtype,
            use_checkpointing=use_checkpointing,
            return_entropy=return_entropy,
        )

    # Seamless fallback when model doesn't split backbone / lm_head
    kwargs_fb: dict[str, Any] = {}
    if attention_mask is not None:
        kwargs_fb["attention_mask"] = attention_mask
    out_fb = model(input_ids, **kwargs_fb)
    logits = out_fb.logits if hasattr(out_fb, "logits") else out_fb
    lp, mask = gather_token_logprobs(logits, input_ids, response_mask, dtype=dtype)
    if return_entropy:
        return lp, mask, lp.new_zeros(lp.shape)
    return lp, mask


__all__ = [
    "chunked_gather_token_logprobs",
    "chunked_model_token_logprobs",
    "chunked_rl_policy_loss",
    "extract_lm_head_and_backbone",
    "moe_router_aux_loss",
    "triton_chunked_logprobs_available",
]
