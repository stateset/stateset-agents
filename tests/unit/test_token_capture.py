"""Agents can return the generated tokens alongside the text.

``generate_turn`` yields a ``ConversationTurn`` whose metadata carries the
exact prompt token ids, the sampled response token ids, and the sampler's
per-token log-probs, so GRPO can train on what was actually generated instead
of re-tokenising text.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent  # noqa: E402
from stateset_agents.core.trajectory import ConversationTurn  # noqa: E402
from tests._tiny_tokenizer import tiny_tokenizer  # noqa: E402


def _tiny_hf_agent():
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
    config = AgentConfig(
        model_name="gpt2",
        max_new_tokens=6,
        temperature=1.0,
        do_sample=True,
        use_chat_template=False,
    )
    agent = MultiTurnAgent(config)
    agent.model = model
    agent.tokenizer = tokenizer
    agent.generation_config = agent._build_generation_config()
    return agent


@pytest.mark.asyncio
async def test_generate_turn_carries_token_ids_and_sampler_logprobs():
    agent = _tiny_hf_agent()
    turn = await agent.generate_turn([{"role": "user", "content": "hello there"}])
    assert isinstance(turn, ConversationTurn)
    assert turn.role == "assistant"
    assert isinstance(turn.content, str)
    md = turn.metadata
    prompt_ids, token_ids, lps = (
        md["prompt_token_ids"],
        md["token_ids"],
        md["sampler_log_probs"],
    )
    assert len(prompt_ids) > 0 and len(token_ids) > 0
    assert len(lps) == len(token_ids)
    assert all(isinstance(i, int) for i in prompt_ids + token_ids)
    assert all(isinstance(x, float) and x <= 0.0 for x in lps)


@pytest.mark.asyncio
async def test_generate_turn_tokens_are_the_generated_text():
    agent = _tiny_hf_agent()
    turn = await agent.generate_turn("hello there")
    raw = agent.tokenizer.decode(turn.metadata["token_ids"], skip_special_tokens=True)
    # content is the cleaned response; the raw decode must contain it
    assert turn.content.strip() in raw or raw.strip().startswith(turn.content.strip())


@pytest.mark.asyncio
async def test_generate_turn_logprobs_match_model_forward():
    """A forward pass on the sampled IDs reproduces raw model log probabilities."""
    agent = _tiny_hf_agent()
    turn = await agent.generate_turn("hello there")
    ids = torch.tensor([turn.metadata["prompt_token_ids"] + turn.metadata["token_ids"]])
    with torch.no_grad():
        logits = agent.model(input_ids=ids).logits
    lp = torch.log_softmax(logits[0, :-1].float(), -1)
    n_prompt = len(turn.metadata["prompt_token_ids"])
    got = lp[n_prompt - 1 :].gather(-1, ids[0, n_prompt:].unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(
        got, torch.tensor(turn.metadata["model_log_probs"]), atol=1e-4, rtol=1e-4
    )


@pytest.mark.asyncio
async def test_generate_response_is_unchanged_by_generate_turn():
    agent = _tiny_hf_agent()
    torch.manual_seed(1)
    text = await agent.generate_response("hello there")
    torch.manual_seed(1)
    turn = await agent.generate_turn("hello there")
    assert turn.content == text


@pytest.mark.asyncio
async def test_stub_agent_generate_turn_has_no_token_metadata():
    config = AgentConfig(
        model_name="stub://x", use_stub_model=True, stub_responses=["ok"]
    )
    agent = MultiTurnAgent(config)
    await agent.initialize()
    turn = await agent.generate_turn([{"role": "user", "content": "hi"}])
    assert turn.role == "assistant" and turn.content == "ok"
    assert "token_ids" not in turn.metadata


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.asyncio
async def test_agent_scores_the_actual_draws(batch_size, monkeypatch):
    agent = _tiny_hf_agent()
    agent.generation_config.top_p = 0.7
    agent.generation_config.top_k = 6
    agent.generation_config.temperature = 0.6
    original = torch.multinomial
    actual = []

    def trace(probabilities, *args, **kwargs):
        actions = original(probabilities, *args, **kwargs)
        actual.append(
            (actions.squeeze(1), probabilities.gather(1, actions).squeeze(1).log())
        )
        return actions

    monkeypatch.setattr(torch, "multinomial", trace)
    turns = (
        [await agent.generate_turn("hello")]
        if batch_size == 1
        else await agent.generate_turns("hello", batch_size)
    )
    for row, turn in enumerate(turns):
        evidence = turn.metadata
        assert evidence["sampler_log_probs_source"] == "generation"
        ids = evidence["token_ids"]
        assert ids == [int(a[row]) for a, _ in actual[: len(ids)]]
        expected = [float(p[row]) for _, p in actual[: len(ids)]]
        assert evidence["sampler_log_probs"] == pytest.approx(expected, abs=5e-7)
        assert (
            max(
                abs(a - b)
                for a, b in zip(expected, evidence["model_log_probs"], strict=True)
            )
            > 1
        )


@pytest.mark.asyncio
async def test_greedy_actions_have_unit_behavior_probability():
    agent = _tiny_hf_agent()
    agent.generation_config.do_sample = False
    turn = await agent.generate_turn("hello")
    assert turn.metadata["sampler_log_probs"] == [0.0] * len(turn.metadata["token_ids"])
    assert all(x < 0 for x in turn.metadata["model_log_probs"])


def test_a_real_pad_token_and_a_stop_suffix_crossing_the_prompt():
    from types import SimpleNamespace

    from stateset_agents.core.generation_evidence import hf_turn_evidence

    seq = torch.tensor([[4, 3, 9, 9], [4, 1, 0, 9], [4, 6, 1, 9]])
    scores = [torch.zeros((3, 10)) for _ in range(3)]
    rows = hf_turn_evidence(
        SimpleNamespace(sequences=seq, logits=scores, scores=scores),
        seq[0, :1],
        SimpleNamespace(do_sample=True, num_beams=1),
        [7, 8],
        [[4, 3], [1, 0]],
    )
    assert [r["token_ids"] for r in rows] == [[3], [1, 0], [6, 1, 9]]
    assert [len(r["sampler_log_probs"]) for r in rows] == [1, 2, 3]


def test_archive_without_behavior_provenance_requires_rescore():
    from stateset_agents.training.rollout_contract import SampledTurn

    md = {"prompt_token_ids": [1], "token_ids": [2], "sampler_log_probs": [-1.0]}
    assert SampledTurn.from_metadata(md).sampler_log_probs == (-1.0,)
    legacy = dict(md, generated=True)
    assert SampledTurn.from_metadata(legacy).sampler_log_probs is None
    assert (
        SampledTurn.from_metadata(
            dict(legacy, rollout_backend="VLLMGenerator")
        ).sampler_log_probs
        is None
    )
    assert SampledTurn.from_metadata(
        dict(legacy, sampler_log_probs_source="generation")
    ).sampler_log_probs == (-1.0,)


@pytest.mark.parametrize(
    "dtype,gap", [(torch.float16, 17.0), (torch.float32, 103.0), (torch.float64, 744.0)]
)
def test_sampler_log_evidence_uses_finite_multinomial_weights(dtype, gap):
    from decimal import Decimal, localcontext
    from types import SimpleNamespace

    from stateset_agents.core.generation_evidence import hf_turn_evidence

    # Exercise either possible draw, including one with subnormal weight.
    scores = torch.tensor([[0.0, -gap], [0.0, -gap]], dtype=dtype)
    weights = scores.softmax(-1)[0].tolist()
    assert weights[1] > 0
    with localcontext() as ctx:
        ctx.prec = 400
        w = list(map(Decimal.from_float, weights))
        expected = [float((v / sum(w)).ln()) for v in w]
    raw = scores.float() if dtype == torch.float16 else scores
    rows = hf_turn_evidence(
        SimpleNamespace(
            sequences=torch.tensor([[1, 0], [1, 1]]), scores=[scores], logits=[scores]
        ),
        torch.tensor([1]),
        SimpleNamespace(do_sample=True, num_beams=1),
        None,
        [],
    )
    for i, row in enumerate(rows):
        assert row["sampler_log_probs"][0] == pytest.approx(
            expected[i], rel=2e-12, abs=0
        )
        assert row["model_log_probs"][0] == float(raw.log_softmax(-1)[i, i])
