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


def test_v92_trainer_live_chunked_wiring_and_turn_level_credit_assignment() -> None:
    """Verify DAPOTrainer & GSPOTrainer execute live chunked logprob path + Turn-Level PRM Credit Assignment."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from types import SimpleNamespace

    from stateset_agents.training.dapo_config import DAPOConfig
    from stateset_agents.training.dapo_trainer import DAPOTrainer
    from stateset_agents.training.gspo_config import GSPOConfig
    from stateset_agents.training.gspo_trainer import GSPOTrainer
    from stateset_agents.training.turn_credit import (
        compute_turn_level_advantages,
        expand_turn_advantages_to_tokens,
    )

    class _SplitLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.embed = nn.Embedding(64, 16)
            self.lm_head = nn.Linear(16, 64, bias=False)
            self.full_forward_calls = 0

            def _backbone(input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(last_hidden_state=self.model.embed(input_ids))

            self.model.forward = _backbone  # type: ignore[method-assign]

        def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
            self.full_forward_calls += 1
            return SimpleNamespace(
                logits=self.lm_head(self.model(input_ids, **kwargs).last_hidden_state)
            )

    torch.manual_seed(99)
    lm = _SplitLM()
    input_ids = torch.randint(0, 64, (2, 16))
    attn_mask = torch.ones_like(input_ids)
    resp_mask = torch.ones_like(input_ids)
    resp_mask[:, :5] = 0

    # 1. Verify DAPOTrainer.compute_token_log_probs chunked vs full match identically
    dapo_full = object.__new__(DAPOTrainer)
    dapo_full.model = lm
    dapo_full.config = DAPOConfig(chunked_loss_chunk_size=None)
    dapo_full._logprob_dtype = None

    dapo_chunk = object.__new__(DAPOTrainer)
    dapo_chunk.model = lm
    dapo_chunk.config = DAPOConfig(
        chunked_loss_chunk_size=5, use_loss_checkpointing=True
    )
    dapo_chunk._logprob_dtype = None

    lp_f, cnt_f = dapo_full.compute_token_log_probs(input_ids, attn_mask, resp_mask)
    calls_after_full = lm.full_forward_calls
    lp_c, cnt_c = dapo_chunk.compute_token_log_probs(input_ids, attn_mask, resp_mask)
    assert lm.full_forward_calls == calls_after_full  # chunked bypassed full lm_head!
    assert torch.allclose(lp_f, lp_c, atol=1e-5)
    assert torch.allclose(cnt_f, cnt_c)

    # 2. Verify GSPOTrainer._compute_group_sequence_log_probs chunked vs full match identically
    class _StubTok:
        padding_side = "right"

        def __call__(self, texts: object, **kwargs: object) -> dict[str, torch.Tensor]:
            if isinstance(texts, str):
                return {"input_ids": torch.tensor([[1, 2, 3, 4]])}
            return {"input_ids": input_ids.clone(), "attention_mask": attn_mask.clone()}

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [1, 2, 3, 4]

    gspo_full = object.__new__(GSPOTrainer)
    gspo_full.model = lm
    gspo_full.tokenizer = _StubTok()
    gspo_full.config = GSPOConfig(chunked_loss_chunk_size=None)

    gspo_chunk = object.__new__(GSPOTrainer)
    gspo_chunk.model = lm
    gspo_chunk.tokenizer = _StubTok()
    gspo_chunk.config = GSPOConfig(
        chunked_loss_chunk_size=6, use_loss_checkpointing=True
    )

    seq_f, len_f = gspo_full._compute_group_sequence_log_probs("hi", ["a", "b"])
    calls_before_gspo_chunk = lm.full_forward_calls
    seq_c, len_c = gspo_chunk._compute_group_sequence_log_probs("hi", ["a", "b"])
    assert lm.full_forward_calls == calls_before_gspo_chunk
    assert torch.allclose(seq_f, seq_c, atol=1e-5)
    assert torch.allclose(len_f, len_c)

    # 3. Verify Turn-Level Process Reward Credit Assignment differentiates good vs bad turns
    credits = compute_turn_level_advantages(
        [
            {"turn_process_rewards": [1.0, 1.0, 1.0], "terminal_reward": 1.0},
            {"turn_process_rewards": [1.0, 0.0, 0.1], "terminal_reward": 0.0},
        ],
        gamma=0.95,
        prm_weight=0.4,
    )
    assert len(credits) == 2
    assert credits[0].turn_advantages[0] > 0.0
    # In episode 1, turn 0 succeeded (prm=1.0) while turn 1 failed (prm=0.0)
    assert credits[1].turn_advantages[0] > credits[1].turn_advantages[1]

    tok_adv = expand_turn_advantages_to_tokens(
        [c.turn_advantages for c in credits],
        turn_token_spans=[[(2, 5), (6, 9), (10, 13)], [(2, 5), (6, 9), (10, 13)]],
        seq_len=16,
        as_torch=True,
    )
    assert tuple(tok_adv.shape) == (2, 16)
    assert float(tok_adv[0, 3].item()) == pytest.approx(credits[0].turn_advantages[0])
    assert float(tok_adv[1, 7].item()) == pytest.approx(credits[1].turn_advantages[1])
    assert float(tok_adv[0, 0].item()) == 0.0


def test_v93_gepo_vapo_chunked_parity_sql_sandbox_telemetry_callback_and_cli_shootout(
    tmp_path: Path,
) -> None:
    """Verify GEPO/VAPO live chunked wiring, SQLAnalyticsSandbox, StudioTelemetryCallback & CLI --run-shootout."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from types import SimpleNamespace
    from typer.testing import CliRunner

    from stateset_agents.cli import app as cli_app
    from stateset_agents.core.mcp_rl_env import MCPRLEnvironment, SQLAnalyticsSandbox
    from stateset_agents.core.trajectory import ConversationTurn
    from stateset_agents.training.gepo_trainer import GEPOConfig, GEPOTrainer
    from stateset_agents.training.studio_callback import StudioTelemetryCallback
    from stateset_agents.training.vapo_config import VAPOConfig
    from stateset_agents.training.vapo_trainer import VAPOTrainer

    class _SplitLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.embed = nn.Embedding(64, 16)
            self.lm_head = nn.Linear(16, 64, bias=False)
            self.full_forward_calls = 0

            def _backbone(input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(last_hidden_state=self.model.embed(input_ids))

            self.model.forward = _backbone  # type: ignore[method-assign]

        def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
            self.full_forward_calls += 1
            return SimpleNamespace(
                logits=self.lm_head(self.model(input_ids, **kwargs).last_hidden_state)
            )

    torch.manual_seed(123)
    lm = _SplitLM()
    ids = torch.randint(0, 64, (2, 16))
    attn = torch.ones_like(ids)
    resp = torch.ones_like(ids)
    resp[:, :5] = 0

    # 1. VAPOTrainer live chunked parity
    v_full = object.__new__(VAPOTrainer)
    v_full.model = lm
    v_full.config = VAPOConfig(chunked_loss_chunk_size=None)
    v_full._logprob_dtype = None

    v_chunk = object.__new__(VAPOTrainer)
    v_chunk.model = lm
    v_chunk.config = VAPOConfig(chunked_loss_chunk_size=5, use_loss_checkpointing=True)
    v_chunk._logprob_dtype = None

    vlp_f = v_full.compute_token_log_probs(ids, attn, resp)
    calls_before = lm.full_forward_calls
    vlp_c = v_chunk.compute_token_log_probs(ids, attn, resp)
    assert lm.full_forward_calls == calls_before
    assert torch.allclose(vlp_f, vlp_c, atol=1e-5)

    # 2. GEPOTrainer live chunked parity
    g_full = object.__new__(GEPOTrainer)
    g_full.model = lm
    g_full.config = GEPOConfig(chunked_loss_chunk_size=None)

    g_chunk = object.__new__(GEPOTrainer)
    g_chunk.model = lm
    g_chunk.config = GEPOConfig(chunked_loss_chunk_size=5, use_loss_checkpointing=True)

    _, gseq_f = g_full.compute_sequence_log_probs(ids, attn, 5)
    calls_before_g = lm.full_forward_calls
    _, gseq_c = g_chunk.compute_sequence_log_probs(ids, attn, 5)
    assert lm.full_forward_calls == calls_before_g
    assert torch.allclose(gseq_f, gseq_c, atol=1e-5)

    # 3. SQLAnalyticsSandbox inside MCPRLEnvironment (read-only enforced + exact result verified)
    async def _run_sql() -> None:
        sb = SQLAnalyticsSandbox()
        env = MCPRLEnvironment(
            tool_executor=sb.call_tool, state_verifier=sb.verify, max_turns=4
        )
        st = await env.reset({"expected_value": 25000})
        st, _, _, bad_inf = await env.step(
            st,
            ConversationTurn(
                role="assistant",
                content='<tool_call>{"name": "run_sql", "arguments": {"sql": "DROP TABLE sales"}}</tool_call>',
            ),
        )
        assert bad_inf["tool_results"][0]["is_error"] is True
        st, _, _, ok_inf = await env.step(
            st,
            ConversationTurn(
                role="assistant",
                content='<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT SUM(revenue_cents) AS total FROM sales WHERE region = \'west\'"}}</tool_call>',
            ),
        )
        assert ok_inf["tool_results"][0]["is_error"] is False
        _, rew, done, end_inf = await env.step(
            st,
            ConversationTurn(
                role="assistant", content="Total west revenue is 25000 cents."
            ),
        )
        assert done and end_inf["verification"]["verified"] is True and rew >= 0.8

    asyncio.run(_run_sql())

    # 4. StudioTelemetryCallback durable JSONL + live HTTP push
    pushed_payloads: list[dict] = []

    def fake_post(
        url: str, payload: dict, headers: dict, timeout: float
    ) -> tuple[int, dict]:
        pushed_payloads.append({"url": url, "payload": payload, "headers": headers})
        return 200, {"ingested_count": 1}

    cb = StudioTelemetryCallback(
        run_name="v93-test-run",
        workspace_root=tmp_path,
        studio_url="http://127.0.0.1:8765",
        studio_token="secret-tok",
        http_caller=fake_post,
    )
    step_out = cb.on_step(
        1,
        {"loss": -0.42, "reward": 0.88},
        guard_report={"action": "CONTINUE", "alerts": []},
        sample_traces=[{"prompt": "hi", "response": "hello"}],
    )
    assert step_out["logged"] is True
    assert step_out["traces_pushed"] == 1
    assert Path(step_out["metrics_path"]).is_file()
    assert pushed_payloads[0]["headers"]["Authorization"] == "Bearer secret-tok"

    # 5. CLI `stateset-agents rl --run-shootout --json`
    res = CliRunner().invoke(
        cli_app,
        ["rl", "--run-shootout", "--workspace", str(tmp_path), "--json"],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["shootout"]["disaggregated_async_rl"]["steps_completed"] == 3


def test_v94_multi_domain_sandbox_pool_sleep_wake_sync_rl_cockpit_and_triton_probe(
    tmp_path: Path,
) -> None:
    """Verify MultiDomainSandboxPool, vLLM sleep/wake sync, Live RL Cockpit API/SSE & Triton backend detection."""
    from fastapi.testclient import TestClient

    from stateset_agents.core.sandbox_pool import (
        MultiDomainSandboxPool,
        SandboxEpisodeSpec,
    )
    from stateset_agents.core.trajectory import ConversationTurn
    from stateset_agents.training.chunked_rl_loss import (
        triton_chunked_logprobs_available,
    )
    from stateset_agents.training.disaggregated_rl import (
        DisaggregatedRLConfig,
        DisaggregatedRLRuntime,
    )
    from stateset_agents.training.studio_callback import StudioTelemetryCallback
    from stateset_agents.weight_studio import create_studio_app
    from stateset_agents.weight_studio.live_serving import LiveServingBridge

    # 1. Triton availability probe never crashes on CPU or GPU
    assert isinstance(triton_chunked_logprobs_available(), bool)

    # 2. MultiDomainSandboxPool running commerce + coding + sql concurrently with per-domain normalization
    async def policy_fn(obs: str, meta: dict, turn_idx: int) -> ConversationTurn:
        dom = meta["domain"]
        fail_mode = bool(meta["scenario"].get("simulate_fail", False))
        if fail_mode:
            return ConversationTurn(role="assistant", content="I am unsure.")
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
            return ConversationTurn(
                role="assistant", content="All coding tests passed."
            )
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
    batch_res = asyncio.run(pool.run_batch(specs, policy_fn, normalize_per_domain=True))
    assert batch_res["episode_count"] == 6
    assert set(batch_res["domain_stats"]) == {"commerce", "coding", "sql"}
    for dom in ("commerce", "coding", "sql"):
        assert batch_res["domain_stats"][dom]["verified_pass_rate"] == 0.5
    ok_advs = [
        e["normalized_advantage"] for e in batch_res["episodes"] if e["verified"]
    ]
    bad_advs = [
        e["normalized_advantage"] for e in batch_res["episodes"] if not e["verified"]
    ]
    assert all(a > 0 for a in ok_advs)
    assert all(a < 0 for a in bad_advs)

    # 3. DisaggregatedRLRuntime vLLM Level-1 Sleep -> Weight Push -> Wake-Up sync
    endpoints_hit: list[str] = []

    def fake_vllm_http(
        url: str, payload: dict, headers: dict, timeout: float
    ) -> tuple[int, dict]:
        endpoints_hit.append(url)
        return 200, {"ok": True}

    bridge = LiveServingBridge(tmp_path, http_caller=fake_vllm_http)
    drt = DisaggregatedRLRuntime(
        DisaggregatedRLConfig(live_vllm_endpoint="http://127.0.0.1:8000/v1"),
        workspace_root=tmp_path,
        serving_bridge=bridge,
    )
    sw = drt.sync_vllm_sleep_wake(
        level=1, weight_push_fn=lambda: {"nccl_broadcast": "ok"}
    )
    assert sw is not None and sw["synced"] is True
    assert endpoints_hit == [
        "http://127.0.0.1:8000/sleep",
        "http://127.0.0.1:8000/wake_up",
    ]
    assert drt.metrics.weight_syncs_completed == 1

    # 4. Live 3D Weight Studio RL Cockpit API (`POST/GET /api/weight-studio/rl/telemetry`)
    app = create_studio_app(workspace_root=tmp_path)
    client = TestClient(app)

    def studio_http_adapter(
        url: str, payload: dict, headers: dict, timeout: float
    ) -> tuple[int, dict]:
        path = url.replace("http://studio.local", "")
        resp = client.post(path, json=payload, headers=headers)
        return resp.status_code, resp.json()

    cb = StudioTelemetryCallback(
        run_name="v94-live-cockpit",
        workspace_root=tmp_path,
        studio_url="http://studio.local",
        http_caller=studio_http_adapter,
    )
    out = cb.on_step(
        1,
        {"reward": 0.91, "loss": -0.35, "entropy": 1.22},
        guard_report={"action": "CONTINUE", "alerts": []},
        sample_traces=[{"prompt": "Order #100", "response": "Verified order #100."}],
    )
    assert out["telemetry_pushed"] is True
    assert out["traces_pushed"] == 1

    r_get = client.get("/api/weight-studio/rl/telemetry")
    assert r_get.status_code == 200
    assert r_get.json()["count"] == 1
    assert r_get.json()["latest"]["run_name"] == "v94-live-cockpit"
    assert r_get.json()["latest"]["guard_action"] == "CONTINUE"


def test_v95_rl_mcp_tools_cockpit_controls_and_upgraded_shootout(
    tmp_path: Path,
) -> None:
    """Verify v0.95.0 RL MCP tools (HTTP + stdio parity), RL Cockpit Pause/Rollback controls, Plan/Pool REST APIs & upgraded shootout."""
    from fastapi.testclient import TestClient

    from stateset_agents.mcp_server import (
        studio_rl_plan,
        studio_rl_telemetry,
        studio_run_sandbox_pool,
    )
    from stateset_agents.weight_studio import create_studio_app
    from stateset_agents.weight_studio.mcp_bridge import (
        get_canonical_studio_tool_names,
    )

    # 1. Canonical MCP tool list includes all 3 new Open-Weight RL tools
    canonical = set(get_canonical_studio_tool_names())
    assert {
        "studio_rl_telemetry",
        "studio_rl_plan",
        "studio_run_sandbox_pool",
    }.issubset(canonical)

    # 2. REST API: /rl/control (pause/resume/rollback), /rl/plan, /rl/sandbox-pool
    app = create_studio_app(workspace_root=tmp_path)
    client = TestClient(app)

    r_pause = client.post(
        "/api/weight-studio/rl/control",
        json={"action": "pause", "run_name": "gspo-v95", "reason": "operator pause"},
    )
    assert r_pause.status_code == 200
    assert r_pause.json()["control"]["status"] == "paused"

    r_tel = client.get("/api/weight-studio/rl/telemetry")
    assert r_tel.status_code == 200
    assert r_tel.json()["control"]["status"] == "paused"
    assert isinstance(r_tel.json()["triton_available"], bool)

    r_rb = client.post(
        "/api/weight-studio/rl/control",
        json={"action": "rollback", "run_name": "gspo-v95", "reason": "hacking alert"},
    )
    assert r_rb.status_code == 200
    assert r_rb.json()["control"]["status"] == "rollback_requested"

    r_plan = client.post(
        "/api/weight-studio/rl/plan",
        json={
            "model": "Qwen/Qwen3-8B",
            "algorithm": "dapo",
            "cloud": "modal",
            "gpus": 4,
        },
    )
    assert r_plan.status_code == 200
    assert r_plan.json()["dispatch"]["provider"] == "modal"
    assert r_plan.json()["vram_estimate"]["peak_logit_vram_saved_pct"] > 90.0

    r_pool = client.post(
        "/api/weight-studio/rl/sandbox-pool",
        json={"domains": ["commerce", "coding", "sql"], "include_failure_pairs": True},
    )
    assert r_pool.status_code == 200
    assert r_pool.json()["episode_count"] == 6
    assert set(r_pool.json()["domain_stats"]) == {"commerce", "coding", "sql"}

    # 3. Streamable HTTP MCP JSON-RPC calls for all 3 RL tools
    for tool_name, args in (
        ("studio_rl_telemetry", {"limit": 10}),
        ("studio_rl_plan", {"algorithm": "gspo", "cloud": "runpod", "gpus": 2}),
        ("studio_run_sandbox_pool", {"domains": ["commerce", "sql"]}),
    ):
        resp = client.post(
            "/api/weight-studio/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": args},
            },
        )
        assert resp.status_code == 200
        res_obj = resp.json()["result"]
        assert res_obj["isError"] is False

    # 4. Direct stdio MCP functions parity
    tel_stdio = studio_rl_telemetry(limit=5, workspace=str(tmp_path))
    assert "triton_available" in tel_stdio
    plan_stdio = studio_rl_plan(
        algorithm="dapo", cloud="modal", workspace=str(tmp_path)
    )
    assert plan_stdio["dispatch"]["provider"] == "modal"
    pool_stdio = studio_run_sandbox_pool(
        domains=["coding", "sql"], workspace=str(tmp_path)
    )
    assert pool_stdio["episode_count"] == 4


def test_v96_bidirectional_rl_control_sqlite_rl_ledger_and_multi_run_filtering(
    tmp_path: Path,
) -> None:
    """Verify persisted SQLite RL run ledger across restarts + bi-directional Pause/Rollback directives into StudioTelemetryCallback."""
    from fastapi.testclient import TestClient

    from stateset_agents.training.studio_callback import StudioTelemetryCallback
    from stateset_agents.weight_studio import create_studio_app

    # 1. Stream two distinct RL runs into App #1 and verify SQLite WAL persistence
    app1 = create_studio_app(workspace_root=tmp_path)
    c1 = TestClient(app1)
    signals: list[tuple[str, dict]] = []

    def _post_adapter(
        url: str, payload: dict, headers: dict, timeout: float
    ) -> tuple[int, dict]:
        resp = c1.post(
            url.replace("http://studio.local", ""), json=payload, headers=headers
        )
        return resp.status_code, resp.json()

    def _get_adapter(url: str, headers: dict, timeout: float) -> tuple[int, dict]:
        resp = c1.get(url.replace("http://studio.local", ""), headers=headers)
        return resp.status_code, resp.json()

    cb_a = StudioTelemetryCallback(
        run_name="run-alpha-gspo",
        workspace_root=tmp_path,
        studio_url="http://studio.local",
        http_caller=_post_adapter,
        http_get_caller=_get_adapter,
        on_control_signal=lambda act, ctrl: signals.append((act, ctrl)),
    )
    cb_b = StudioTelemetryCallback(
        run_name="run-beta-dapo",
        workspace_root=tmp_path,
        studio_url="http://studio.local",
        http_caller=_post_adapter,
        http_get_caller=_get_adapter,
    )

    s1 = cb_a.on_step(1, {"reward": 0.72, "loss": -0.15, "entropy": 1.35})
    assert s1["telemetry_pushed"] is True
    assert s1["should_pause"] is False
    assert s1["should_rollback"] is False

    cb_b.on_step(1, {"reward": 0.85, "loss": -0.22, "entropy": 1.18})

    # 2. Operator clicks PAUSE in 3D Weight Studio -> next trainer step receives should_pause=True & triggers signal callback
    r_pause = c1.post(
        "/api/weight-studio/rl/control",
        json={
            "action": "pause",
            "run_name": "run-alpha-gspo",
            "reason": "Inspect reward spike",
        },
    )
    assert r_pause.status_code == 200
    s2 = cb_a.on_step(2, {"reward": 0.99, "loss": -0.40, "entropy": 0.42})
    assert s2["should_pause"] is True
    assert s2["should_rollback"] is False
    assert len(signals) == 1 and signals[0][0] == "pause"

    # Operator requests ROLLBACK -> poll_control() sees rollback_requested immediately
    c1.post(
        "/api/weight-studio/rl/control",
        json={
            "action": "rollback",
            "run_name": "run-alpha-gspo",
            "reason": "Hacking guard",
        },
    )
    polled = cb_a.poll_control()
    assert polled["status"] == "rollback_requested"
    s3 = cb_a.on_step(3, {"reward": 0.98, "loss": -0.45, "entropy": 0.35})
    assert s3["should_rollback"] is True

    # 3. Fresh Studio App #2 (simulating server restart) reloads persisted SQLite RL runs & filters by run_name
    app2 = create_studio_app(workspace_root=tmp_path)
    c2 = TestClient(app2)

    r_stats = c2.get("/api/weight-studio/store/stats")
    assert r_stats.status_code == 200
    assert r_stats.json()["rl_step_count"] == 4

    r_all = c2.get("/api/weight-studio/rl/telemetry?limit=50")
    assert r_all.status_code == 200
    body_all = r_all.json()
    assert body_all["count"] == 4
    run_names = {r["run_name"] for r in body_all["runs"]}
    assert run_names == {"run-alpha-gspo", "run-beta-dapo"}

    r_beta = c2.get("/api/weight-studio/rl/telemetry?run_name=run-beta-dapo")
    assert r_beta.status_code == 200
    assert r_beta.json()["count"] == 1
    assert r_beta.json()["latest"]["run_name"] == "run-beta-dapo"
    assert r_beta.json()["latest"]["reward"] == 0.85


def test_v97_training_runtime_auto_pause_and_weight_rollback_enforcement(
    tmp_path: Path,
) -> None:
    """Verify TrainingRuntime.attach_studio_callback auto-snapshots best weights and enforces Studio Pause/Rollback in place."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from fastapi.testclient import TestClient

    from stateset_agents.training.gspo_config import GSPOConfig
    from stateset_agents.training.gspo_trainer import GSPOTrainer
    from stateset_agents.training.studio_callback import StudioTelemetryCallback
    from stateset_agents.weight_studio import create_studio_app

    app = create_studio_app(workspace_root=tmp_path)
    client = TestClient(app)

    def _post(
        url: str, payload: dict, headers: dict, timeout: float
    ) -> tuple[int, dict]:
        resp = client.post(
            url.replace("http://studio.local", ""), json=payload, headers=headers
        )
        return resp.status_code, resp.json()

    def _get(url: str, headers: dict, timeout: float) -> tuple[int, dict]:
        resp = client.get(url.replace("http://studio.local", ""), headers=headers)
        return resp.status_code, resp.json()

    torch.manual_seed(7)
    model = nn.Linear(8, 4)
    initial_w = model.weight.detach().clone()

    trainer = object.__new__(GSPOTrainer)
    trainer.config = GSPOConfig()
    trainer.model = model

    cb = StudioTelemetryCallback(
        run_name="v97-auto-enforce",
        workspace_root=tmp_path,
        studio_url="http://studio.local",
        http_caller=_post,
        http_get_caller=_get,
    )
    trainer.attach_studio_callback(
        cb, auto_snapshot=True, pause_poll_interval=0.01, max_pause_wait_sec=0.05
    )

    # Step 1 improves reward -> snapshot updated to good_w
    with torch.no_grad():
        model.weight.add_(0.5)
    good_w = model.weight.detach().clone()
    res1: dict = {"policy_loss": -0.2, "average_reward": 0.92}
    trainer.enforce_studio_step(1, res1)
    assert res1["studio_rolled_back"] is False
    assert cb.summary()["best_snapshot_step"] == 1

    # Corrupt model weights (simulating reward-hacking collapse) and request Studio ROLLBACK
    with torch.no_grad():
        model.weight.add_(10.0)
    assert not torch.allclose(model.weight, good_w)

    client.post(
        "/api/weight-studio/rl/control",
        json={
            "action": "rollback",
            "run_name": "v97-auto-enforce",
            "reason": "collapse",
        },
    )
    res2: dict = {"policy_loss": 5.0, "average_reward": 0.12}
    trainer.enforce_studio_step(2, res2)
    assert res2["studio_rolled_back"] is True
    assert torch.allclose(model.weight, good_w, atol=1e-6)
    assert not torch.allclose(model.weight, initial_w)
    assert cb.summary()["rollbacks_executed"] == 1

    # Operator requests PAUSE -> cooperative wait executes and returns cleanly on timeout/resume
    client.post(
        "/api/weight-studio/rl/control",
        json={"action": "pause", "run_name": "v97-auto-enforce", "reason": "inspect"},
    )
    res3: dict = {"policy_loss": -0.1, "average_reward": 0.93}
    trainer.enforce_studio_step(3, res3)
    assert res3["studio_pause_waited_sec"] > 0.0
    assert cb.summary()["pause_events_handled"] == 1

    # Verify all 4 flagship trainers expose StudioControlMixin parity
    from stateset_agents.training.dapo_trainer import DAPOTrainer
    from stateset_agents.training.gepo_trainer import GEPOTrainer
    from stateset_agents.training.vapo_trainer import VAPOTrainer

    for cls in (GSPOTrainer, DAPOTrainer, GEPOTrainer, VAPOTrainer):
        assert hasattr(cls, "attach_studio_callback")
        assert hasattr(cls, "enforce_studio_step")


def test_v98_moe_fp8_tis_adaptive_curriculum_and_trajectory_credit_inspector(
    tmp_path: Path,
) -> None:
    """Verify FP8 rollout TIS/IcePop + MoE router aux loss, AdaptiveDifficultyCurriculum & SQLite Trajectory Credit Inspector."""
    torch = pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from stateset_agents.mcp_server import studio_query_rl_trajectories
    from stateset_agents.training.adaptive_curriculum import (
        AdaptiveDifficultyCurriculum,
        CurriculumTask,
    )
    from stateset_agents.training.chunked_rl_loss import (
        chunked_rl_policy_loss,
        moe_router_aux_loss,
    )
    from stateset_agents.weight_studio import create_studio_app

    # 1. MoE router load-balancing aux loss + FP8 rollout TIS / IcePop mismatch correction
    torch.manual_seed(99)
    h = torch.randn(2, 12, 16, requires_grad=True)
    w = torch.randn(64, 16, requires_grad=True)
    ids = torch.randint(0, 64, (2, 12))
    mask = torch.ones(2, 12)
    mask[:, :3] = 0.0
    old_lp = torch.full((2, 11), -2.5)
    # Simulate FP8 rollout logprobs with one extreme outlier token
    rollout_lp = old_lp.clone()
    rollout_lp[0, 5] = -8.0
    adv = torch.tensor([1.0, -1.0])
    router_logits = torch.randn(2, 12, 8, requires_grad=True)

    aux_only = moe_router_aux_loss(router_logits, top_k=2, aux_coef=0.02)
    assert float(aux_only.item()) > 0.0

    loss_out = chunked_rl_policy_loss(
        h,
        w,
        ids,
        mask,
        old_lp,
        adv,
        algorithm="dapo",
        chunk_size=6,
        rollout_logprobs=rollout_lp,
        tis_clip_max=2.0,
        icepop_bounds=(0.2, 5.0),
        router_logits=router_logits,
        moe_aux_coef=0.02,
    )
    loss_out["loss"].backward()
    assert router_logits.grad is not None and float(router_logits.grad.abs().sum()) > 0
    assert loss_out["icepop_masked_fraction"] > 0.0
    assert loss_out["tis_clipped_fraction"] > 0.0
    assert loss_out["rollout_learner_kl"] > 0.0
    assert float(loss_out["moe_aux_loss"].item()) > 0.0

    # 2. AdaptiveDifficultyCurriculum: frontier focus, production trace promotion & self-play synthesis
    cur = AdaptiveDifficultyCurriculum(
        [
            CurriculumTask("easy-1", "commerce", "Check order status"),
            CurriculumTask("front-1", "commerce", "Handle partial refund"),
            CurriculumTask("hard-1", "coding", "Fix race condition"),
        ]
    )
    for _ in range(3):
        cur.record_outcome("easy-1", passed=True, reward=0.95)
    cur.record_outcome("front-1", passed=True, reward=0.9)
    cur.record_outcome("front-1", passed=False, reward=0.1)
    for _ in range(3):
        cur.record_outcome("hard-1", passed=False, reward=0.05)

    assert cur.stats()["bands"]["frontier"] == 1
    assert cur.stats()["bands"]["too_easy"] == 1
    batch = cur.sample_batch(1, frontier_ratio=1.0, unseen_ratio=0.0)
    assert batch[0].task_id == "front-1"

    promoted = cur.promote_production_traces(
        [
            {
                "trace_id": "tr-1",
                "prompt": "Fix duplicate charge #991",
                "proxy_score": 0.1,
            }
        ]
    )
    assert len(promoted) == 1 and promoted[0].source == "production_trace"

    sp = cur.synthesize_harder_variants(max_new=3)
    assert len(sp) == 1 and sp[0].parent_task_id == "easy-1" and sp[0].difficulty == 2

    # 3. Verifiable Trajectory Credit Inspector (SQLite + REST + MCP parity)
    app = create_studio_app(workspace_root=tmp_path)
    client = TestClient(app)

    r_post = client.post(
        "/api/weight-studio/rl/trajectories",
        json={
            "trajectory_id": "traj-98-1",
            "run_name": "gspo-v98",
            "domain": "commerce",
            "verified": False,
            "terminal_reward": 0.15,
            "prompt": "Refund ORD-991",
            "response": "Could not locate order.",
            "turns": [
                {"turn": 1, "tool": "lookup_order", "process_reward": 0.9},
                {"turn": 2, "tool": "issue_refund", "process_reward": 0.0},
            ],
        },
    )
    assert r_post.status_code == 200
    saved_turns = r_post.json()["trajectory"]["turns"]
    assert "credit" in saved_turns[0] and "credit" in saved_turns[1]

    # Sandbox pool also records its episodes into persisted rl_trajectories
    r_pool = client.post(
        "/api/weight-studio/rl/sandbox-pool",
        json={"domains": ["commerce", "sql"], "include_failure_pairs": True},
    )
    assert r_pool.status_code == 200

    r_get = client.get("/api/weight-studio/rl/trajectories?failed_only=true")
    assert r_get.status_code == 200
    assert r_get.json()["count"] >= 3

    # MCP stdio & HTTP parity for studio_query_rl_trajectories
    stdio_res = studio_query_rl_trajectories(
        domain="commerce", failed_only=True, workspace=str(tmp_path)
    )
    assert stdio_res["count"] >= 2
    assert stdio_res["stats"]["rl_trajectory_count"] >= 5
