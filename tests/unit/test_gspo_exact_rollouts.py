"""Native GSPO samples its group in one batched generate and takes its
old-policy log-probs from the same batched scorer as the current ones.

Before this, each response was generated sequentially and its old log-prob
came from a separate unbatched forward; the kernel/padding noise between that
and the batched current pass (~4e-3 per token) exceeded GSPO's clip band
(3e-4 / 4e-4), so ~90% of on-policy samples were gated to zero gradient.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent  # noqa: E402
from stateset_agents.core.trajectory import ConversationTurn  # noqa: E402
from stateset_agents.training import loss_computation as lc  # noqa: E402


def _tiny_hf_agent(max_new_tokens: int = 6):
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_embd=32,
            n_layer=2,
            n_head=2,
            vocab_size=50257,
            n_positions=128,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    agent = MultiTurnAgent(
        AgentConfig(
            model_name="gpt2",
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            do_sample=True,
            use_chat_template=False,
        )
    )
    agent.model = model
    agent.tokenizer = tokenizer
    agent.generation_config = agent._build_generation_config()
    return agent


# --- MultiTurnAgent.generate_turns -------------------------------------------------


@pytest.mark.asyncio
async def test_generate_turns_returns_n_turns_with_exact_token_metadata():
    agent = _tiny_hf_agent()
    turns = await agent.generate_turns([{"role": "user", "content": "hello there"}], 3)
    assert len(turns) == 3 and all(isinstance(t, ConversationTurn) for t in turns)
    for t in turns:
        md = t.metadata
        assert md["prompt_token_ids"] and md["token_ids"]
        assert len(md["sampler_log_probs"]) == len(md["token_ids"])
        assert all(x <= 0.0 for x in md["sampler_log_probs"])
    # every prompt is the same rendered prompt
    assert len({tuple(t.metadata["prompt_token_ids"]) for t in turns}) == 1


@pytest.mark.asyncio
async def test_generate_turns_sampler_logprobs_match_a_forward_pass():
    agent = _tiny_hf_agent()
    turns = await agent.generate_turns("hello there", 2)
    rows = [
        (list(t.metadata["prompt_token_ids"]), list(t.metadata["token_ids"]), i)
        for i, t in enumerate(turns)
    ]
    lp, mask, _ = lc._forward_token_rows(
        agent.model, rows, torch.device("cpu"), 2, grad=False
    )
    for (_p, r, _), row_lp, row_mask, turn in zip(rows, lp, mask, turns, strict=True):
        got = row_lp[row_mask.bool()].tolist()
        assert len(got) == len(r)
        assert got == pytest.approx(turn.metadata["sampler_log_probs"], abs=2e-3)


@pytest.mark.asyncio
async def test_generate_turns_zero_and_stub_fallback():
    agent = _tiny_hf_agent()
    assert await agent.generate_turns("hi", 0) == []
    stub = MultiTurnAgent(
        AgentConfig(
            model_name="stub://x", use_stub_model=True, stub_responses=["a", "b"]
        )
    )
    await stub.initialize()
    turns = await stub.generate_turns("hi", 2)
    assert [t.content for t in turns] == ["a", "b"]
    assert all("token_ids" not in t.metadata for t in turns)


@pytest.mark.asyncio
async def test_generate_turns_uses_the_rollout_backend_as_one_batch():
    class Backend:
        def __init__(self):
            self.calls = []

        async def generate_with_logprobs(self, prompts, **kw):
            self.calls.append(list(prompts))
            return [
                SimpleNamespace(
                    response=f"r{i}",
                    prompt_token_ids=[1, 2],
                    response_token_ids=[3 + i],
                    token_logprobs=[-0.1 * (i + 1)],
                )
                for i, _ in enumerate(prompts)
            ]

    stub = MultiTurnAgent(
        AgentConfig(model_name="stub://x", use_stub_model=True, stub_responses=["x"])
    )
    await stub.initialize()
    backend = Backend()
    stub.set_rollout_backend(backend)
    turns = await stub.generate_turns("hi", 3)
    assert len(backend.calls) == 1 and len(backend.calls[0]) == 3
    assert [t.content for t in turns] == ["r0", "r1", "r2"]
    assert turns[2].metadata["sampler_log_probs"] == [pytest.approx(-0.3)]


# --- GSPO generation through generate_turns ------------------------------------------


@pytest.mark.asyncio
async def test_gspo_hf_generation_batches_the_group_and_skips_rescoring(monkeypatch):
    from stateset_agents.training import gspo_generation as gg
    from stateset_agents.training.gspo_config import GSPOConfig

    class FakeAgent:
        tokenizer = None
        model = None
        config = SimpleNamespace(system_prompt=None)
        memory_window = 0

        def __init__(self):
            self.turn_calls = []
            self.response_calls = 0

        async def generate_turns(self, messages, n, context=None):
            self.turn_calls.append((messages, n))
            return [
                ConversationTurn(
                    role="assistant",
                    content=f"resp{i}",
                    metadata={
                        "token_ids": [1, 2],
                        "sampler_log_probs": [-0.5, -0.25 * i],
                    },
                )
                for i in range(n)
            ]

        async def generate_response(self, messages):
            self.response_calls += 1
            return "seq"

    gen = gg.GSPOTrajectoryGenerator.__new__(gg.GSPOTrajectoryGenerator)
    gen.config = GSPOConfig(model_name="fake", use_vllm=False)
    gen.agent = FakeAgent()
    gen.environment = None
    gen.vllm_generator = None
    gen.sampling_params = None
    gen._vllm_initialized = False
    rescored = []

    async def no_rescoring(self, prompt_text, response):
        rescored.append(response)
        return -9.0

    monkeypatch.setattr(
        gg.GSPOTrajectoryGenerator, "_compute_sequence_log_prob", no_rescoring
    )
    out = await gen._generate_with_hf("What is 2+2?", 4)
    assert [r for r, _ in out] == ["resp0", "resp1", "resp2", "resp3"]
    assert [lp for _, lp in out] == pytest.approx([-0.5, -0.75, -1.0, -1.25])
    assert gen.agent.turn_calls == [([{"role": "user", "content": "What is 2+2?"}], 4)]
    assert gen.agent.response_calls == 0 and rescored == []


# --- trainer: old log-probs from the same batched scorer -----------------------------


def _gspo_trainer(rescore: bool):
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

    from stateset_agents.training.gspo_config import GSPOConfig
    from stateset_agents.training.gspo_trainer import GSPOTrainer

    torch.manual_seed(1)
    model = GPT2LMHeadModel(
        GPT2Config(n_embd=32, n_layer=2, n_head=2, vocab_size=50257, n_positions=128)
    )
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    class Reward:
        async def compute_reward(self, turns, context=None):
            score = 1.0 if turns[0].content == "ok" else 0.0
            return SimpleNamespace(total_reward=score, score=score)

    cfg = GSPOConfig(
        model_name="gpt2",
        num_generations=2,
        num_outer_iterations=1,
        num_iterations=1,
        max_prompt_length=32,
        max_completion_length=32,
        rescore_old_log_probs=rescore,
    )
    trainer = GSPOTrainer(
        config=cfg,
        model=model,
        tokenizer=tokenizer,
        agent=None,
        environment=None,
        reward_model=Reward(),
        ref_model=None,
    )

    async def fake_group(prompt, n):
        # deliberately wrong old log-probs, as an unbatched scorer would give
        return [("ok", -3.0), ("nope", -30.0)]

    trainer.generator.generate_group_responses = fake_group  # type: ignore[method-assign]
    return trainer


@pytest.mark.asyncio
async def test_first_update_is_exactly_on_policy_when_rescoring():
    trainer = _gspo_trainer(rescore=True)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    metrics = await trainer.train_step(["hello"], num_groups=1)
    assert metrics["sequence_importance_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["clipping_fraction"] == 0.0
    assert metrics["generation_log_prob_gap"] > 0.0  # the generator's numbers were off
    # ratio 1 makes the loss value the (zero-mean) advantage, but the gate is
    # open so the reward's gradient moves the policy
    assert metrics["policy_loss"] == pytest.approx(0.0, abs=1e-6)
    moved = any(
        not torch.equal(b, a.detach())
        for b, a in zip(before, trainer.model.parameters(), strict=True)
    )
    assert moved


@pytest.mark.asyncio
async def test_generator_log_probs_are_used_when_rescoring_is_off():
    trainer = _gspo_trainer(rescore=False)
    metrics = await trainer.train_step(["hello"], num_groups=1)
    assert metrics["sequence_importance_ratio"] != pytest.approx(1.0, abs=1e-3)
    assert metrics["clipping_fraction"] == 1.0
    assert metrics["generation_log_prob_gap"] == 0.0
