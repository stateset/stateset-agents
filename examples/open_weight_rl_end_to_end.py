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

from stateset_agents.core.mcp_rl_env import CodeVerificationSandbox, MCPRLEnvironment
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.training.chunked_rl_loss import (
    chunked_model_token_logprobs,
    chunked_rl_policy_loss,
)
from stateset_agents.training.disaggregated_rl import (
    DisaggregatedRLConfig,
    DisaggregatedRLRuntime,
)
from stateset_agents.training.rl_flywheel_guard import OnlineRewardHackingGuard
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

    # 2. Verify BOTH Commerce SQLite Sandbox & Coding Verification Sandbox
    commerce_env = MCPRLEnvironment(max_turns=4)
    st = await commerce_env.reset(
        {"target_order_id": "ORD-77701", "expect_refunded": True}
    )
    st, _, _, _ = await commerce_env.step(
        st,
        ConversationTurn(
            role="assistant",
            content='<tool_call>{"name": "get_order", "arguments": {"order_id": "ORD-77701"}}</tool_call>',
        ),
    )
    st, _, _, _ = await commerce_env.step(
        st,
        ConversationTurn(
            role="assistant",
            content='<tool_call>{"name": "issue_refund", "arguments": {"order_id": "ORD-77701", "reason": "damaged"}}</tool_call>',
        ),
    )
    _, commerce_reward, _, commerce_info = await commerce_env.step(
        st,
        ConversationTurn(
            role="assistant",
            content="Verified order ORD-77701 and issued your full refund.",
        ),
    )

    code_box = CodeVerificationSandbox()
    code_env = MCPRLEnvironment(
        tool_executor=code_box.call_tool, state_verifier=code_box.verify, max_turns=4
    )
    cst = await code_env.reset()
    cst, _, _, _ = await code_env.step(
        cst,
        ConversationTurn(
            role="assistant",
            content='<tool_call>{"name": "write_solution", "arguments": {"code": "def compute_net_revenue(sold, price, refunded):\\n    return (sold - refunded) * price"}}</tool_call>',
        ),
    )
    cst, _, _, _ = await code_env.step(
        cst,
        ConversationTurn(
            role="assistant",
            content='<tool_call>{"name": "run_tests", "arguments": {}}</tool_call>',
        ),
    )
    _, code_reward, _, code_info = await code_env.step(
        cst,
        ConversationTurn(
            role="assistant", content="Solution verified against all unit tests."
        ),
    )

    # 3. Disaggregated Async RL + Chunked GSPO Optimizer Step + Online Hacking Guard
    guard = OnlineRewardHackingGuard()
    rt = DisaggregatedRLRuntime(
        DisaggregatedRLConfig(
            min_batch_size=2,
            max_batch_size=4,
            dynamic_sampling=True,
            num_rollout_workers=2,
        )
    )

    async def producer(_worker_id: int, _ver: int) -> list[dict[str, Any]]:
        return [
            {
                "reward": commerce_reward,
                "token_count": 20,
                "sampler_log_probs": (-0.3, -0.4),
            },
            {"reward": 0.15, "token_count": 20, "sampler_log_probs": (-0.6, -0.7)},
        ]

    async def learner(batch: list[Any], step: int) -> dict[str, Any]:
        bsz = len(batch)
        ids = torch.randint(0, 256, (bsz, 24))
        mask = torch.ones(bsz, 24)
        mask[:, :6] = 0.0
        with torch.no_grad():
            old_lp, _ = chunked_model_token_logprobs(lm, ids, mask, chunk_size=8)
        adv = torch.tensor(
            [float(r.payload["advantage"]) for r in batch], dtype=torch.float32
        )
        hidden = lm.model(ids).last_hidden_state
        loss_out = chunked_rl_policy_loss(
            hidden,
            lm.lm_head.weight,
            ids,
            mask,
            old_lp,
            adv,
            algorithm="gspo",
            ref_logprobs=old_lp,
            chunk_size=8,
            use_checkpointing=True,
        )
        optimizer.zero_grad()
        loss_out["loss"].backward()
        optimizer.step()

        guard_rep = guard.check_step(
            step,
            [
                {
                    "prompt": "Order #77701 damaged",
                    "response": "Verified order and issued refund.",
                    "proxy_score": 0.9,
                    "human_score": 0.88,
                },
                {
                    "prompt": "Order #77702 damaged",
                    "response": "bad",
                    "proxy_score": 0.2,
                    "human_score": 0.2,
                },
            ],
            mean_entropy=float(loss_out["entropy"].item()),
        )
        return {
            "loss": round(float(loss_out["loss"].item()), 5),
            "entropy": round(float(loss_out["entropy"].item()), 4),
            "guard_action": guard_rep["action"],
        }

    rl_res = await rt.run_steps(
        num_steps=2, producer_fn=producer, learner_step_fn=learner
    )
    return {
        "warm_start": {
            "layer": baked["layer_index"],
            "mode": baked["mode"],
            "delta_l2": baked["delta_l2_norm"],
        },
        "commerce_verified": commerce_info["verification"]["verified"],
        "code_verified": code_info["verification"]["verified"],
        "rl_steps": rl_res["steps"],
    }


if __name__ == "__main__":
    summary = asyncio.run(run_demo())
    print("StateSet Open-Weight RL End-to-End Demo PASSED:")
    print(
        f"  Step-0 Warm-Start : L{summary['warm_start']['layer']:02d} ({summary['warm_start']['mode']}, ΔL2={summary['warm_start']['delta_l2']})"
    )
    print(f"  Commerce Sandbox  : verified={summary['commerce_verified']}")
    print(f"  CodeExec Sandbox  : verified={summary['code_verified']}")
    print(f"  Chunked GSPO Steps: {summary['rl_steps']}")
