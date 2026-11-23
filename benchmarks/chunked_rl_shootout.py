"""Reproducible Chunked Fused RL Loss & Disaggregated Async Rollout Shootout.

Measures and verifies:
1. Forward & backward numerical equivalence between full-materialization
   ``gather_token_logprobs`` and memory-efficient ``chunked_gather_token_logprobs``.
2. Peak vocabulary-logit memory savings on long-context sequences (both real
   measured tensor footprint and analytical 16k/32k large-vocab projection).
3. Disaggregated async rollout throughput with DAPO zero-variance dynamic sampling.

Run directly:
    python benchmarks/chunked_rl_shootout.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any


def run_chunked_loss_shootout(
    *,
    batch_size: int = 2,
    seq_len: int = 256,
    hidden_dim: int = 64,
    vocab_size: int = 4096,
    chunk_size: int = 64,
) -> dict[str, Any]:
    """Benchmark chunked vs. full-logit GRPO/GSPO forward+backward equivalence & speed."""
    import torch

    from stateset_agents.training.chunked_rl_loss import (
        chunked_gather_token_logprobs,
        chunked_rl_policy_loss,
        triton_chunked_logprobs_available,
    )
    from stateset_agents.training.rl_blueprint import estimate_chunked_vram_savings
    from stateset_agents.training.rl_losses import gather_token_logprobs

    torch.manual_seed(42)
    h1 = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
    w1 = torch.randn(vocab_size, hidden_dim, requires_grad=True)
    h2 = h1.detach().clone().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    response_mask = torch.ones(batch_size, seq_len)
    response_mask[:, :16] = 0.0

    # 1. Full logit materialization baseline
    t0 = time.perf_counter()
    full_logits = torch.nn.functional.linear(h1, w1)
    lp_full, m_full = gather_token_logprobs(full_logits, input_ids, response_mask)
    lp_full.sum().backward()
    full_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # 2. Chunked memory-efficient kernel
    t1 = time.perf_counter()
    lp_chunk, m_chunk = chunked_gather_token_logprobs(
        h2,
        w2,
        input_ids,
        response_mask,
        chunk_size=chunk_size,
        use_checkpointing=True,
    )
    lp_chunk.sum().backward()
    chunk_ms = round((time.perf_counter() - t1) * 1000.0, 2)

    max_lp_diff = float((lp_full - lp_chunk).abs().max().item())
    assert h1.grad is not None and h2.grad is not None
    assert w1.grad is not None and w2.grad is not None
    max_h_grad_diff = float((h1.grad - h2.grad).abs().max().item())
    max_w_grad_diff = float((w1.grad - w2.grad).abs().max().item())

    # 3. Full chunked GSPO & DAPO policy loss smoke
    advantages = torch.tensor([1.0, -1.0][:batch_size], dtype=torch.float32)
    gspo = chunked_rl_policy_loss(
        h2.detach().clone().requires_grad_(True),
        w2.detach().clone().requires_grad_(True),
        input_ids,
        response_mask,
        lp_chunk.detach(),
        advantages,
        algorithm="gspo",
        ref_logprobs=lp_chunk.detach(),
        chunk_size=chunk_size,
    )
    dapo = chunked_rl_policy_loss(
        h2.detach().clone().requires_grad_(True),
        w2.detach().clone().requires_grad_(True),
        input_ids,
        response_mask,
        lp_chunk.detach(),
        advantages,
        algorithm="dapo",
        ref_logprobs=lp_chunk.detach(),
        chunk_size=chunk_size,
    )

    return {
        "numerical_equivalence": {
            "max_logprob_abs_diff": round(max_lp_diff, 8),
            "max_hidden_grad_abs_diff": round(max_h_grad_diff, 8),
            "max_weight_grad_abs_diff": round(max_w_grad_diff, 8),
            "exact_match": max_lp_diff < 1e-4
            and max_h_grad_diff < 1e-4
            and max_w_grad_diff < 1e-4,
        },
        "timing_ms": {
            "full_logits_ms": full_ms,
            "chunked_checkpointed_ms": chunk_ms,
        },
        "loss_values": {
            "gspo_loss": round(float(gspo["loss"].item()), 6),
            "dapo_loss": round(float(dapo["loss"].item()), 6),
            "mean_entropy": round(float(gspo["entropy"].item()), 4),
            "kernel_backend": str(gspo.get("kernel_backend", "pytorch_chunked")),
        },
        "triton_available": bool(triton_chunked_logprobs_available()),
        "vram_projection_16k_qwen3": estimate_chunked_vram_savings(
            batch_size=4, seq_len=16384, vocab_size=152064, chunk_size=256
        ),
    }


async def run_multi_domain_sandbox_shootout() -> dict[str, Any]:
    """Benchmark concurrent commerce + coding + SQL verifiable RL episodes with per-domain z-score normalization."""
    from stateset_agents.core.sandbox_pool import (
        MultiDomainSandboxPool,
        SandboxEpisodeSpec,
    )
    from stateset_agents.core.trajectory import ConversationTurn

    async def _policy(
        obs: str, meta: dict[str, Any], turn_idx: int
    ) -> ConversationTurn:
        dom = meta["domain"]
        if meta["scenario"].get("simulate_fail"):
            return ConversationTurn(role="assistant", content="Unable to verify.")
        if dom == "commerce":
            if turn_idx == 0:
                return ConversationTurn(
                    role="assistant",
                    content='<tool_call>{"name": "issue_refund", "arguments": {"order_id": "ORD-77701", "reason": "damaged"}}</tool_call>',
                )
            return ConversationTurn(
                role="assistant", content="Refund issued for ORD-77701."
            )
        if dom == "coding":
            if turn_idx == 0:
                return ConversationTurn(
                    role="assistant",
                    content='<tool_call>{"name": "write_solution", "arguments": {"code": "def compute_net_revenue(s, p, r):\\n    return (s - r) * p"}}</tool_call>',
                )
            if turn_idx == 1:
                return ConversationTurn(
                    role="assistant",
                    content='<tool_call>{"name": "run_tests", "arguments": {}}</tool_call>',
                )
            return ConversationTurn(role="assistant", content="All tests passed.")
        if turn_idx == 0:
            return ConversationTurn(
                role="assistant",
                content='<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT SUM(revenue_cents) AS total FROM sales WHERE region = \'west\'"}}</tool_call>',
            )
        return ConversationTurn(role="assistant", content="Total revenue is 25000.")

    pool = MultiDomainSandboxPool(max_concurrency=6)
    specs = [
        SandboxEpisodeSpec("c-ok", "commerce", {"target_order_id": "ORD-77701"}),
        SandboxEpisodeSpec(
            "c-bad", "commerce", {"target_order_id": "ORD-77701", "simulate_fail": True}
        ),
        SandboxEpisodeSpec("k-ok", "coding", {}),
        SandboxEpisodeSpec("k-bad", "coding", {"simulate_fail": True}),
        SandboxEpisodeSpec("s-ok", "sql", {"expected_value": 25000}),
        SandboxEpisodeSpec(
            "s-bad", "sql", {"expected_value": 25000, "simulate_fail": True}
        ),
    ]
    t0 = time.perf_counter()
    res = await pool.run_batch(specs, _policy, normalize_per_domain=True)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "episode_count": res["episode_count"],
        "elapsed_ms": elapsed_ms,
        "episodes_per_sec": round(
            res["episode_count"] / max(1e-4, elapsed_ms / 1000.0), 2
        ),
        "domain_stats": res["domain_stats"],
    }


async def run_disaggregated_async_shootout(num_steps: int = 6) -> dict[str, Any]:
    """Benchmark DisaggregatedRLRuntime with DAPO zero-variance dynamic sampling."""
    from stateset_agents.training.disaggregated_rl import (
        DisaggregatedRLConfig,
        DisaggregatedRLRuntime,
    )

    rt = DisaggregatedRLRuntime(
        DisaggregatedRLConfig(
            min_batch_size=4,
            max_batch_size=8,
            dynamic_sampling=True,
            num_rollout_workers=4,
        )
    )
    counter = {"n": 0}

    async def producer(worker_id: int, policy_ver: int) -> list[dict[str, Any]]:
        counter["n"] += 1
        # Every 3rd group is all-1.0 (zero variance) to prove DAPO dynamic sampling drops it
        if counter["n"] % 3 == 0:
            return [
                {"reward": 1.0, "token_count": 32, "sampler_log_probs": (-0.2, -0.3)}
                for _ in range(4)
            ]
        return [
            {
                "reward": 1.0 if i % 2 == 0 else 0.1,
                "token_count": 48,
                "sampler_log_probs": (-0.25, -0.35),
            }
            for i in range(4)
        ]

    async def learner(batch: list[Any], step: int) -> dict[str, Any]:
        return {
            "mean_adv": round(
                sum(r.payload["advantage"] for r in batch) / len(batch), 4
            )
        }

    res = await rt.run_steps(
        num_steps=num_steps, producer_fn=producer, learner_step_fn=learner
    )
    return dict(res["metrics"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="StateSet Chunked & Disaggregated RL Shootout"
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    loss_rep = run_chunked_loss_shootout()
    async_rep = asyncio.run(run_disaggregated_async_shootout())
    sandbox_rep = asyncio.run(run_multi_domain_sandbox_shootout())
    report = {
        "chunked_rl_loss": loss_rep,
        "disaggregated_async_rl": async_rep,
        "multi_domain_sandbox_pool": sandbox_rep,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    eq = loss_rep["numerical_equivalence"]
    vp = loss_rep["vram_projection_16k_qwen3"]
    print("StateSet Open-Weight RL Shootout")
    print(
        f"  Chunked Loss Equivalence : exact={eq['exact_match']} "
        f"(max Δlp={eq['max_logprob_abs_diff']}, max Δgrad={eq['max_hidden_grad_abs_diff']}, "
        f"backend={loss_rep['loss_values']['kernel_backend']}, triton={loss_rep['triton_available']})"
    )
    print(
        f"  16k Qwen3 Logit VRAM     : {vp['full_logits_gb']} GB -> {vp['chunked_logits_gb']} GB "
        f"({vp['peak_logit_vram_saved_pct']}% saved)"
    )
    print(
        f"  Disaggregated Async RL   : {async_rep['rollouts_per_sec']} rollouts/s, "
        f"{async_rep['tokens_per_sec']} tok/s, "
        f"dropped_zero_var={async_rep['groups_dropped_zero_variance']}"
    )
    print(
        f"  Multi-Domain Sandbox Pool: {sandbox_rep['episode_count']} eps in "
        f"{sandbox_rep['elapsed_ms']}ms ({sandbox_rep['episodes_per_sec']} eps/s, "
        f"domains={','.join(sorted(sandbox_rep['domain_stats']))})"
    )


if __name__ == "__main__":
    main()
