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


def _tiny_hf_agent():
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
    """Sampler log-probs must be the model's own log-probs of the sampled
    tokens (temperature 1, no truncation), so a training-time forward pass on
    the stored ids reproduces them."""
    agent = _tiny_hf_agent()
    turn = await agent.generate_turn("hello there")
    ids = torch.tensor([turn.metadata["prompt_token_ids"] + turn.metadata["token_ids"]])
    with torch.no_grad():
        logits = agent.model(input_ids=ids).logits
    lp = torch.log_softmax(logits[0, :-1].float(), -1)
    n_prompt = len(turn.metadata["prompt_token_ids"])
    got = lp[n_prompt - 1 :].gather(-1, ids[0, n_prompt:].unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(
        got, torch.tensor(turn.metadata["sampler_log_probs"]), atol=1e-4, rtol=1e-4
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
