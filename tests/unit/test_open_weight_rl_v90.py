"""Unit tests for the v0.90 Open-Weight RL Leadership Suite.

Covers:
1. Chunked Fused GRPO / GSPO / DAPO loss numerical & gradient equivalence vs full logits.
2. Disaggregated Async Rollout <-> Learner Runtime with DAPO dynamic sampling & live LoRA sync.
3. MCP-Native Verifiable Multi-Turn RL Environment + deterministic SQLite CommerceLedgerSandbox.
4. Online Reward-Hacking Circuit-Breaker (hacking risk, entropy collapse, length explosion)
   and Production-to-RL Curriculum Builder.
5. Declarative RL Blueprint + Cloud Launch Plans + `stateset-agents rl` CLI + Shootout benchmark.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


def test_chunked_rl_loss_matches_full_logits_forward_and_backward() -> None:
    torch = pytest.importorskip("torch")

    from stateset_agents.training.chunked_rl_loss import (
        chunked_gather_token_logprobs,
        chunked_rl_policy_loss,
    )
    from stateset_agents.training.rl_losses import gather_token_logprobs

    torch.manual_seed(7)
    bsz, seq_len, hidden_dim, vocab_size = 2, 32, 16, 128
    h1 = torch.randn(bsz, seq_len, hidden_dim, requires_grad=True)
    w1 = torch.randn(vocab_size, hidden_dim, requires_grad=True)
    b1 = torch.randn(vocab_size, requires_grad=True)

    h2 = h1.detach().clone().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)
    b2 = b1.detach().clone().requires_grad_(True)

    input_ids = torch.randint(0, vocab_size, (bsz, seq_len))
    response_mask = torch.ones(bsz, seq_len)
    response_mask[:, :8] = 0.0

    # Full logit materialization
    full_logits = torch.nn.functional.linear(h1, w1, b1)
    lp_full, mask_full = gather_token_logprobs(full_logits, input_ids, response_mask)
    lp_full.sum().backward()

    # Chunked checkpointed kernel (chunk_size=7 to exercise uneven final chunk)
    lp_chunk, mask_chunk, entropy = chunked_gather_token_logprobs(
        h2,
        w2,
        input_ids,
        response_mask,
        lm_head_bias=b2,
        chunk_size=7,
        use_checkpointing=True,
        return_entropy=True,
    )
    lp_chunk.sum().backward()

    assert torch.allclose(lp_full, lp_chunk, atol=1e-5, rtol=1e-4)
    assert torch.allclose(mask_full, mask_chunk)
    assert h1.grad is not None and h2.grad is not None
    assert w1.grad is not None and w2.grad is not None
    assert b1.grad is not None and b2.grad is not None
    assert torch.allclose(h1.grad, h2.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(w1.grad, w2.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(b1.grad, b2.grad, atol=1e-4, rtol=1e-4)
    assert float(entropy.mean().item()) > 0.0

    # Exercise all 3 algorithms (GRPO, GSPO, DAPO) with KL + off-policy importance weights
    adv = torch.tensor([1.0, -1.0], dtype=torch.float32)
    iw = torch.tensor([1.1, 0.9], dtype=torch.float32)
    for algo in ("grpo", "gspo", "dapo"):
        h_test = h1.detach().clone().requires_grad_(True)
        w_test = w1.detach().clone().requires_grad_(True)
        out = chunked_rl_policy_loss(
            h_test,
            w_test,
            input_ids,
            response_mask,
            lp_chunk.detach(),
            adv,
            algorithm=algo,
            ref_logprobs=lp_chunk.detach(),
            kl_coef=0.05,
            entropy_coef=0.01,
            chunk_size=11,
            importance_weights=iw,
        )
        out["loss"].backward()
        assert torch.isfinite(out["loss"])
        assert h_test.grad is not None and torch.isfinite(h_test.grad).all()


def test_disaggregated_async_rl_dynamic_sampling_and_live_lora_sync(
    tmp_path: Path,
) -> None:
    from stateset_agents.training.disaggregated_rl import (
        DisaggregatedRLConfig,
        DisaggregatedRLRuntime,
        truncated_importance_weight,
    )
    from stateset_agents.weight_studio.live_serving import LiveServingBridge

    lora_dir = tmp_path / "outputs" / "rl_lora"
    lora_dir.mkdir(parents=True)
    (lora_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    synced: list[dict] = []

    def fake_http(
        url: str, payload: dict, headers: dict, timeout: float
    ) -> tuple[int, dict]:
        synced.append({"url": url, "payload": payload})
        return 200, {"status": "loaded"}

    bridge = LiveServingBridge(tmp_path, http_caller=fake_http)
    rt = DisaggregatedRLRuntime(
        DisaggregatedRLConfig(
            min_batch_size=4,
            max_batch_size=8,
            dynamic_sampling=True,
            num_rollout_workers=2,
            weight_sync_interval=2,
            live_vllm_endpoint="http://127.0.0.1:8000/v1",
            live_lora_name="rl-step-lora",
        ),
        workspace_root=tmp_path,
        serving_bridge=bridge,
    )

    calls = {"n": 0}

    async def producer(worker_id: int, policy_ver: int) -> list[dict]:
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            # Zero-variance group -> dropped by DAPO dynamic sampling
            return [
                {"reward": 1.0, "token_count": 20, "sampler_log_probs": (-0.2,)}
            ] * 4
        return [
            {
                "reward": 1.0 if i < 2 else 0.0,
                "token_count": 30,
                "sampler_log_probs": (-0.3, -0.4),
            }
            for i in range(4)
        ]

    async def learner(batch: list, step: int) -> dict:
        return {"lora_path": str(lora_dir), "step_seen": step}

    res = asyncio.run(
        rt.run_steps(num_steps=4, producer_fn=producer, learner_step_fn=learner)
    )
    metrics = res["metrics"]
    assert metrics["steps_completed"] == 4
    assert metrics["groups_dropped_zero_variance"] >= 1
    assert metrics["weight_syncs_completed"] == 2
    assert len(synced) == 2
    assert (
        truncated_importance_weight((-1.0, -1.0), (-0.1, -0.1), max_weight=2.0) == 2.0
    )


def test_mcp_native_rl_environment_commerce_sandbox_and_http_bridge() -> None:
    from stateset_agents.core.mcp_rl_env import MCPRLEnvironment
    from stateset_agents.core.trajectory import ConversationTurn

    async def _run() -> None:
        env = MCPRLEnvironment(max_turns=4, process_reward_weight=0.25)
        state = await env.reset(
            {"target_order_id": "ORD-77701", "expect_refunded": True}
        )

        # Turn 1: inspect order via <tool_call>
        state, r1, done1, info1 = await env.step(
            state,
            ConversationTurn(
                role="assistant",
                content='<tool_call>{"name": "get_order", "arguments": {"order_id": "ORD-77701"}}</tool_call>',
            ),
        )
        assert not done1
        assert r1 > 0
        assert info1["tool_results"][0]["is_error"] is False

        # Turn 2: issue eligible refund
        state, r2, done2, info2 = await env.step(
            state,
            ConversationTurn(
                role="assistant",
                content='<tool_call>{"name": "issue_refund", "arguments": {"order_id": "ORD-77701", "reason": "damaged"}}</tool_call>',
            ),
        )
        assert not done2
        assert info2["tool_results"][0]["is_error"] is False

        # Turn 3: final answer -> triggers verifiable SQLite ledger check
        state, r3, done3, info3 = await env.step(
            state,
            ConversationTurn(
                role="assistant",
                content="I verified order ORD-77701 (9 days since delivery) and issued your full refund.",
            ),
        )
        assert done3
        assert info3["verification"]["verified"] is True
        assert r3 >= 0.9

        # Policy-violation scenario: 45-day order ORD-99102 must NOT be refunded
        s_old = await env.reset(
            {"target_order_id": "ORD-99102", "expect_refunded": False}
        )
        s_old, _, _, info_old = await env.step(
            s_old,
            ConversationTurn(
                role="assistant",
                content='<tool_call>{"name": "issue_refund", "arguments": {"order_id": "ORD-99102"}}</tool_call>',
            ),
        )
        assert info_old["tool_results"][0]["is_error"] is True

    asyncio.run(_run())


def test_online_reward_hacking_guard_curriculum_and_rl_blueprint_cli(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from stateset_agents.cli import app as cli_app
    from stateset_agents.training.rl_flywheel_guard import (
        OnlineRewardHackingGuard,
        ProductionTraceCurriculumBuilder,
    )
    from stateset_agents.weight_studio.telemetry_ingest import LiveTraceCollector

    # 1. OnlineRewardHackingGuard catches entropy collapse, length explosion, and reward hacking
    guard = OnlineRewardHackingGuard(workspace_root=tmp_path)
    ok_rep = guard.check_step(
        1,
        [
            {
                "prompt": "p1",
                "response": "short grounded answer",
                "proxy_score": 0.9,
                "human_score": 0.88,
            },
            {
                "prompt": "p2",
                "response": "bad answer",
                "proxy_score": 0.2,
                "human_score": 0.18,
            },
        ],
        current_kl_coef=0.02,
        mean_entropy=0.45,
    )
    assert ok_rep["action"] == "CONTINUE"

    ent_rep = guard.check_step(
        2,
        [{"prompt": "p1", "response": " ".join(["word"] * 40), "proxy_score": 0.8}],
        current_kl_coef=0.02,
        mean_entropy=0.02,
    )
    assert ent_rep["action"] == "INCREASE_KL"
    assert ent_rep["recommended_kl_coef"] > 0.02

    hack_rep = guard.check_step(
        3,
        [
            {
                "prompt": "p1",
                "response": "hacked 1",
                "proxy_score": 0.95,
                "human_score": 0.10,
            },
            {
                "prompt": "p2",
                "response": "hacked 2",
                "proxy_score": 0.90,
                "human_score": 0.15,
            },
            {
                "prompt": "p3",
                "response": "good",
                "proxy_score": 0.30,
                "human_score": 0.90,
            },
        ],
        current_kl_coef=0.02,
        mean_entropy=0.4,
    )
    assert hack_rep["action"] == "HALT_AND_ROLLBACK"

    # 2. Seed failing OTLP traces -> ProductionTraceCurriculumBuilder -> `stateset-agents rl` CLI
    collector = LiveTraceCollector(tmp_path)
    collector.ingest_payload(
        {
            "traces": [
                {
                    "trace_id": "f1",
                    "prompt": "Order #101 broken",
                    "response": "Sorry [NAME]. Sorry [NAME].",
                },
                {
                    "trace_id": "f2",
                    "prompt": "Order #102 missing",
                    "response": "ok",
                },
            ]
        }
    )
    cur = ProductionTraceCurriculumBuilder(tmp_path).build_curriculum(max_episodes=10)
    assert cur["episode_count"] == 2
    assert Path(cur["curriculum_path"]).is_file()

    # 3. `stateset-agents rl --cloud modal --curriculum-from-traces --warm-start --json`
    cli_res = CliRunner().invoke(
        cli_app,
        [
            "rl",
            "--model",
            "Qwen/Qwen3-8B",
            "--algorithm",
            "dapo",
            "--cloud",
            "modal",
            "--gpus",
            "4",
            "--curriculum-from-traces",
            "--warm-start",
            "--workspace",
            str(tmp_path),
            "--json",
        ],
    )
    assert cli_res.exit_code == 0, cli_res.output
    plan = json.loads(cli_res.output)
    assert plan["dispatch"]["provider"] == "modal"
    assert plan["vram_estimate"]["peak_logit_vram_saved_pct"] > 90.0
    assert plan["curriculum"]["episode_count"] == 2
    assert plan["warm_start_result"]["warm_started"] is True


def test_v91_gpu_torch_caa_warm_start_chunked_model_wrapper_and_code_sandbox() -> None:
    """Verify real PyTorch hook CAA warm-start + rollback, chunked_model_token_logprobs & CodeVerificationSandbox."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from types import SimpleNamespace

    from stateset_agents.core.mcp_rl_env import (
        CodeVerificationSandbox,
        MCPRLEnvironment,
    )
    from stateset_agents.core.trajectory import ConversationTurn
    from stateset_agents.training.chunked_rl_loss import chunked_model_token_logprobs
    from stateset_agents.training.dapo_config import DAPOConfig
    from stateset_agents.training.gspo_config import GSPOConfig
    from stateset_agents.training.rl_losses import gather_token_logprobs
    from stateset_agents.weight_studio.gpu_surgery import (
        bake_torch_caa_steering,
        extract_torch_caa_vector,
    )

    class _Block(nn.Module):
        def __init__(self, hidden: int, bias: bool) -> None:
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.up_proj = nn.Linear(hidden, hidden)
            self.mlp.down_proj = nn.Linear(hidden, hidden, bias=bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.mlp.down_proj(torch.relu(self.mlp.up_proj(x)))

    class _LM(nn.Module):
        def __init__(self, *, bias: bool) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.embed = nn.Embedding(64, 16)
            self.model.layers = nn.ModuleList([_Block(16, bias) for _ in range(3)])
            self.lm_head = nn.Linear(16, 64, bias=False)

            def _backbone_fwd(input_ids: torch.Tensor) -> SimpleNamespace:
                h = self.model.embed(input_ids)
                for layer in self.model.layers:
                    h = layer(h)
                return SimpleNamespace(last_hidden_state=h)

            self.model.forward = _backbone_fwd  # type: ignore[method-assign]

        def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(
                logits=self.lm_head(self.model(input_ids).last_hidden_state)
            )

    torch.manual_seed(11)
    for use_bias, expected_mode in ((True, "bias_shift"), (False, "rank1_weight_bake")):
        lm = _LM(bias=use_bias)
        pos_ids = torch.randint(0, 32, (2, 10))
        neg_ids = torch.randint(32, 64, (2, 10))

        # Verify chunked_model_token_logprobs matches full logits exactly
        mask = torch.ones(2, 10)
        mask[:, :3] = 0.0
        lp_chunk, _ = chunked_model_token_logprobs(lm, pos_ids, mask, chunk_size=5)
        lp_full, _ = gather_token_logprobs(lm(pos_ids).logits, pos_ids, mask)
        assert torch.allclose(lp_chunk, lp_full, atol=1e-5)

        # Verify PyTorch hook CAA extraction + bake + exact rollback
        caa = extract_torch_caa_vector(lm, pos_ids, neg_ids)
        assert caa["num_layers"] == 3
        assert caa["peak_separation"] > 0.0
        before_logits = lm(pos_ids).logits.detach().clone()

        baked = bake_torch_caa_steering(
            lm, caa["direction_tensor"], layer_index=caa["recommended_layer"], alpha=0.6
        )
        assert baked["mode"] == expected_mode
        assert baked["delta_l2_norm"] > 0.0
        after_logits = lm(pos_ids).logits.detach().clone()
        assert not torch.allclose(before_logits, after_logits)

        baked["rollback"]()
        restored_logits = lm(pos_ids).logits.detach().clone()
        assert torch.allclose(before_logits, restored_logits, atol=1e-6)

    # Verify CodeVerificationSandbox inside MCPRLEnvironment
    async def _run_code_env() -> None:
        sb = CodeVerificationSandbox()
        env = MCPRLEnvironment(
            tool_executor=sb.call_tool, state_verifier=sb.verify, max_turns=4
        )
        s = await env.reset()
        s, _, _, _ = await env.step(
            s,
            ConversationTurn(
                role="assistant",
                content='<tool_call>{"name": "write_solution", "arguments": {"code": "def compute_net_revenue(s, p, r):\\n    return (s - r) * p"}}</tool_call>',
            ),
        )
        s, _, _, info_t = await env.step(
            s,
            ConversationTurn(
                role="assistant",
                content='<tool_call>{"name": "run_tests", "arguments": {}}</tool_call>',
            ),
        )
        assert info_t["tool_results"][0]["result"]["passed"] is True
        _, reward, done, info_end = await env.step(
            s, ConversationTurn(role="assistant", content="All assertions passed.")
        )
        assert done and info_end["verification"]["verified"] is True and reward >= 0.9

    asyncio.run(_run_code_env())

    # Verify GSPOConfig & DAPOConfig expose opt-in chunked loss fields cleanly
    g_cfg = GSPOConfig(chunked_loss_chunk_size=128, use_loss_checkpointing=True)
    d_cfg = DAPOConfig(chunked_loss_chunk_size=128, use_loss_checkpointing=True)
    assert g_cfg.chunked_loss_chunk_size == 128
    assert d_cfg.use_loss_checkpointing is True
