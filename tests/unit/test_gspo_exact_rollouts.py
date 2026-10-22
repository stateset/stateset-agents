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
from tests._tiny_tokenizer import tiny_tokenizer  # noqa: E402


def _tiny_hf_agent(max_new_tokens: int = 6, stop_sequences: list[str] | None = None):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_embd=32,
            n_layer=2,
            n_head=2,
            vocab_size=256,
            n_positions=128,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    tokenizer = tiny_tokenizer()
    agent = MultiTurnAgent(
        AgentConfig(
            model_name="gpt2",
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            do_sample=True,
            use_chat_template=False,
            stop_sequences=list(stop_sequences or []),
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
    from transformers import GPT2Config, GPT2LMHeadModel

    from stateset_agents.training.gspo_config import GSPOConfig
    from stateset_agents.training.gspo_trainer import GSPOTrainer

    torch.manual_seed(1)
    model = GPT2LMHeadModel(
        GPT2Config(n_embd=32, n_layer=2, n_head=2, vocab_size=256, n_positions=128)
    )
    tokenizer = tiny_tokenizer()

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


# --- per-row stopping in batched generation ----------------------------------------


def test_stop_criteria_is_per_row():
    """HF ORs a scalar across the batch: row 0 hitting a stop string used to
    end every sibling, and a stop string in any other row was ignored."""
    from stateset_agents.core.agent import StopOnSpecialTokens

    tok = tiny_tokenizer()
    stop = tok.encode("\n\n", add_special_tokens=False)
    criteria = StopOnSpecialTokens(["\n\n"], tok)
    ids = torch.tensor([[5, 6, *stop], [5, 6, 7, 8][: 2 + len(stop)]])
    assert criteria(ids, None).tolist() == [True, False]
    ids = torch.tensor([[5, 6, 7, 8][: 2 + len(stop)], [5, 6, *stop]])
    assert criteria(ids, None).tolist() == [False, True]


@pytest.mark.asyncio
async def test_batched_rows_stop_independently_and_never_record_padding(monkeypatch):
    agent = _tiny_hf_agent(max_new_tokens=8, stop_sequences=["\n\n"])
    tok = agent.tokenizer
    stop = tok.encode("\n\n", add_special_tokens=False)
    eos = tok.eos_token_id
    monkeypatch.setattr(tok, "pad_token_id", eos, raising=False)
    prompt = "User: hi\nAssistant: "
    prompt_ids = tok(prompt, return_tensors="pt")["input_ids"]
    plen = prompt_ids.shape[1]
    vocab = agent.model.config.vocab_size
    # Row 0 stops on "\n\n" (then HF pads it with pad == eos); row 1 keeps
    # going and samples a real EOS later.
    rows = [[9, *stop, eos, eos, eos], [9, 10, 11, 12, eos, eos]]
    width = max(len(r) for r in rows)
    rows = [r + [eos] * (width - len(r)) for r in rows]

    def fake_generate(**kwargs):
        criteria = kwargs["stopping_criteria"]
        seq = torch.cat([prompt_ids.repeat(2, 1), torch.tensor(rows)], dim=1)
        # The criterion must not stop row 1 when row 0 hits the stop string.
        upto = plen + 1 + len(stop)
        assert criteria(seq[:, :upto], None).tolist() == [True, False]
        logits = tuple(torch.zeros(2, vocab) for _ in range(width))
        return SimpleNamespace(sequences=seq, logits=logits)

    monkeypatch.setattr(agent.model, "generate", fake_generate)
    batch = await agent._generate_with_model_details_batch(prompt, 2)
    assert batch[0]["token_ids"] == [9, *stop]  # padding is not a sampled EOS
    assert batch[1]["token_ids"] == [9, 10, 11, 12, eos]


def test_overlong_prompt_keeps_latest_turn_and_assistant_cue():
    agent = _tiny_hf_agent(max_new_tokens=8)
    tok = agent.tokenizer
    original = tok.truncation_side
    tok.model_max_length = 40
    prompt = "User: " + "old " * 50 + "\nUser: LATEST\nAssistant: "
    ids = agent._tokenize_prompt(prompt)["input_ids"][0]
    assert ids.shape[0] == 32
    assert tok.decode(ids).endswith("LATEST\nAssistant: ")
    assert tok.truncation_side == original


@pytest.mark.asyncio
async def test_train_step_metrics_are_builtin_floats_with_tensor_lr(monkeypatch):
    """``train_step`` promises ``dict[str, float]``: numpy means and a
    tensor-valued scheduler LR must not leak into logged metrics."""
    trainer = _gspo_trainer(rescore=True)
    monkeypatch.setattr(trainer.scheduler, "get_last_lr", lambda: [torch.tensor(2e-5)])
    metrics = await trainer.train_step(["hello"], num_groups=1)
    assert {k: type(v) for k, v in metrics.items() if type(v) is not float} == {}
    assert metrics["learning_rate"] == pytest.approx(2e-5)
