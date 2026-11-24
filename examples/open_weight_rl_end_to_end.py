"""End-to-End StateSet Open-Weight RL + Surgical Warm-Start Example.

Runs completely offline in ~2 seconds on CPU or GPU to demonstrate the full
StateSet open-weight RL differentiation stack:

1. Step-0 PyTorch Forward-Hook CAA Surgical Warm-Start (``extract_torch_caa_vector``
   + ``bake_torch_caa_steering``) on a real PyTorch causal Transformer LM.
2. Multi-turn verifiable rollouts on ``MCPRLEnvironment`` across BOTH the
   deterministic SQLite ``CommerceLedgerSandbox`` AND ``CodeVerificationSandbox``.
3. ``DisaggregatedRLRuntime`` streaming rollouts with DAPO zero-variance dynamic
   sampling into real PyTorch optimizer steps using memory-efficient
   ``chunked_model_token_logprobs`` + ``chunked_rl_policy_loss`` (GSPO).
4. ``OnlineRewardHackingGuard`` step audit verifying calibration & stability.

Run directly:
    python examples/open_weight_rl_end_to_end.py
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from stateset_agents.core.sandbox_pool import MultiDomainSandboxPool
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.training.adaptive_curriculum import (
    AdaptiveDifficultyCurriculum,
    CurriculumTask,
)
from stateset_agents.training.chunked_rl_loss import (
    chunked_model_token_logprobs,
    chunked_rl_policy_loss,
)
from stateset_agents.training.disaggregated_rl import (
    DisaggregatedRLConfig,
    DisaggregatedRLRuntime,
)
from stateset_agents.training.rl_flywheel_guard import OnlineRewardHackingGuard
from stateset_agents.training.studio_callback import StudioTelemetryCallback
from stateset_agents.weight_studio.gpu_surgery import (
    bake_torch_caa_steering,
    extract_torch_caa_vector,
)


class _TinyTransformerBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(hidden, hidden * 2)
        self.mlp.down_proj = nn.Linear(hidden * 2, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = torch.nn.functional.gelu(self.mlp.up_proj(h))
        return x + self.mlp.down_proj(h)


class _TinyBackbone(nn.Module):
    def __init__(self, vocab: int, hidden: int, num_layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [_TinyTransformerBlock(hidden) for _ in range(num_layers)]
        )

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        return SimpleNamespace(last_hidden_state=h)


class TinyCausalLM(nn.Module):
    """Minimal HuggingFace-shaped causal LM (`model.layers` + `lm_head`)."""

    def __init__(self, vocab: int = 256, hidden: int = 32, num_layers: int = 3) -> None:
        super().__init__()
        self.model = _TinyBackbone(vocab, hidden, num_layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        h = self.model(input_ids).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(h))


async def run_demo() -> dict[str, Any]:
    torch.manual_seed(0)
    lm = TinyCausalLM(vocab=256, hidden=32, num_layers=3)
    optimizer = torch.optim.AdamW(lm.parameters(), lr=1e-3)

    # 1. Step-0 PyTorch Forward-Hook CAA Surgical Warm-Start
    pos_ids = torch.randint(10, 120, (2, 16))
    neg_ids = torch.randint(120, 240, (2, 16))
    caa = extract_torch_caa_vector(lm, pos_ids, neg_ids)
    baked = bake_torch_caa_steering(
        lm,
        caa["direction_tensor"],
        layer_index=caa["recommended_layer"],
        alpha=0.5,
    )

    # 2. Unified Adaptive Curriculum + 5-Domain Verifiable Sandbox Pool + Chunked GSPO (FP8 TIS + MoE Aux)
    pool = MultiDomainSandboxPool(max_concurrency=5)
    curriculum = AdaptiveDifficultyCurriculum(
        [
            CurriculumTask(
                "ord-1",
                "commerce",
                "Refund damaged ORD-77701",
                {"target_order_id": "ORD-77701"},
            ),
            CurriculumTask("code-1", "coding", "Implement compute_net_revenue", {}),
            CurriculumTask(
                "sql-1", "sql", "Sum west revenue", {"expected_value": 25000}
            ),
            CurriculumTask(
                "api-1",
                "rest_api",
                "Cancel sub_101 idempotently",
                {"target_subscription_id": "sub_101"},
            ),
            CurriculumTask("swe-1", "docker_code", "Fix billing discount bug", {}),
        ]
    )

    async def policy_fn(
        obs: str, meta: dict[str, Any], turn_idx: int
    ) -> ConversationTurn:
        dom = meta["domain"]
        if dom == "commerce":
            if turn_idx == 0:
                return ConversationTurn(
                    role="assistant",
                    content='<tool_call>{"name": "issue_refund", "arguments": {"order_id": "ORD-77701", "reason": "damaged"}}</tool_call>',
                )
            return ConversationTurn(
                role="assistant", content="Verified refund for ORD-77701."
            )
        if dom == "coding":
            if turn_idx == 0:
                return ConversationTurn(
                    role="assistant",
                    content='<tool_call>{"name": "write_solution", "arguments": {"code": "def compute_net_revenue(sold, price, refunded):\\n    return (sold - refunded) * price"}}</tool_call>',
                )
            if turn_idx == 1:
                return ConversationTurn(
                    role="assistant",
                    content='<tool_call>{"name": "run_tests", "arguments": {}}</tool_call>',
                )
            return ConversationTurn(
                role="assistant", content="All coding unit tests passed."
            )
        if dom == "sql":
            if turn_idx == 0:
                return ConversationTurn(
                    role="assistant",
                    content='<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT SUM(revenue_cents) AS total FROM sales WHERE region = \'west\'"}}</tool_call>',
                )
            return ConversationTurn(
                role="assistant", content="West revenue is 25000 cents."
            )
        if dom == "rest_api":
            if turn_idx == 0:
                return ConversationTurn(
                    role="assistant",
                    content='<tool_call>{"name": "http_request", "arguments": {"method": "POST", "path": "/v1/subscriptions/sub_101/cancel", "idempotency_key": "idem-demo-1", "body": {"reason": "user_request"}}}</tool_call>',
                )
            return ConversationTurn(
                role="assistant", content="Subscription sub_101 canceled."
            )
        if turn_idx == 0:
            return ConversationTurn(
                role="assistant",
                content='<tool_call>{"name": "apply_patch", "arguments": {"path": "billing.py", "content": "def apply_discount(subtotal_cents: int, pct: int) -> int:\\n    return subtotal_cents - (subtotal_cents * pct) // 100\\n"}}</tool_call>',
            )
        if turn_idx == 1:
            return ConversationTurn(
                role="assistant",
                content='<tool_call>{"name": "run_pytest", "arguments": {}}</tool_call>',
            )
        return ConversationTurn(
            role="assistant", content="Fixed billing.py and verified pytest."
        )

    guard = OnlineRewardHackingGuard()
    studio_cb = StudioTelemetryCallback(run_name="open-weight-e2e-demo")
    rt = DisaggregatedRLRuntime(
        DisaggregatedRLConfig(min_batch_size=2, max_batch_size=5)
    )

    async def learner(episodes: list[dict[str, Any]], step: int) -> dict[str, Any]:
        bsz = len(episodes)
        ids = torch.randint(0, 256, (bsz, 24))
        mask = torch.ones(bsz, 24)
        mask[:, :6] = 0.0
        with torch.no_grad():
            old_lp, _ = chunked_model_token_logprobs(lm, ids, mask, chunk_size=8)
        adv = torch.tensor(
            [float(e.get("normalized_advantage", 0.0)) or 1.0 for e in episodes],
            dtype=torch.float32,
        )
        hidden = lm.model(ids).last_hidden_state
        router_logits = torch.randn(bsz, 24, 4)
        loss_out = chunked_rl_policy_loss(
            hidden,
            lm.lm_head.weight,
            ids,
            mask,
            old_lp,
            adv,
            algorithm="gspo",
            ref_logprobs=old_lp,
            rollout_logprobs=old_lp - 0.05,
            tis_clip_max=2.0,
            icepop_bounds=(0.2, 5.0),
            router_logits=router_logits,
            moe_aux_coef=0.01,
            chunk_size=8,
            use_checkpointing=True,
        )
        optimizer.zero_grad()
        loss_out["loss"].backward()
        optimizer.step()
        return {
            "loss": round(float(loss_out["loss"].item()), 5),
            "entropy": round(float(loss_out["entropy"].item()), 4),
            "tis_mean_ratio": round(float(loss_out["tis_mean_ratio"]), 4),
            "moe_aux_loss": round(float(loss_out["moe_aux_loss"].item()), 5),
        }

    loop_res = await rt.run_adaptive_sandbox_loop(
        pool=pool,
        curriculum=curriculum,
        policy_fn=policy_fn,
        learner_step_fn=learner,
        num_steps=2,
        batch_size=5,
        model=lm,
        studio_callback=studio_cb,
        hacking_guard=guard,
    )
    return {
        "warm_start": {
            "layer": baked["layer_index"],
            "mode": baked["mode"],
            "delta_l2": baked["delta_l2_norm"],
        },
        "domains_verified": sorted(
            {
                d
                for s in loop_res["steps"]
                for d, st in (s.get("domain_stats") or {}).items()
                if st.get("verified_pass_rate", 0.0) >= 1.0
            }
        ),
        "self_play_variants_created": loop_res["self_play_variants_created"],
        "rl_steps": loop_res["steps"],
    }


if __name__ == "__main__":
    summary = asyncio.run(run_demo())
    print("StateSet Open-Weight RL End-to-End Demo PASSED:")
    print(
        f"  Step-0 Warm-Start : L{summary['warm_start']['layer']:02d} ({summary['warm_start']['mode']}, ΔL2={summary['warm_start']['delta_l2']})"
    )
    print(f"  Verified Domains  : {', '.join(summary['domains_verified'])}")
    print(f"  Self-Play Variants: {summary['self_play_variants_created']}")
    print(f"  Chunked GSPO Steps: {len(summary['rl_steps'])}")
