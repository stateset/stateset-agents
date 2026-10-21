"""Preserve complete sampled actions while rendering configured stop strings.

Without a stop string, ordinary reply text is the decode of sampled token IDs.
With a stop string, displayed text ends before it and the retained action still
includes the complete token that triggered stopping, plus its likelihood.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent  # noqa: E402
from stateset_agents.core.agent_config import ConfigValidationError  # noqa: E402
from tests.learning._toy import make_tokenizer  # noqa: E402

MERGED_TOKENS = ["\n\n", "ok\n\nmore", "ereUser: no"]


@pytest.fixture(autouse=True)
def _single_threaded_torch():
    """Tiny models run fastest single-threaded (and -n 8 oversubscribes)."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _tokenizer() -> Any:
    tok = make_tokenizer()
    tok.add_tokens(MERGED_TOKENS)
    return tok


def _gpt2(vocab_size: int, seed: int = 0) -> Any:
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    return GPT2LMHeadModel(
        GPT2Config(
            n_embd=32,
            n_layer=1,
            n_head=2,
            vocab_size=vocab_size,
            n_positions=256,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )


def _scripted_gpt2(vocab_size: int, script: list[int]) -> Any:
    """A real GPT-2 whose logits are pushed onto ``script`` step by step."""
    from transformers import GPT2LMHeadModel

    class ScriptedGPT2(GPT2LMHeadModel):
        prompt_len: int | None = None

        def generate(self, *args: Any, **kwargs: Any) -> Any:
            self.prompt_len = None
            return super().generate(*args, **kwargs)

        def forward(self, input_ids: Any = None, **kwargs: Any) -> Any:
            out = super().forward(input_ids=input_ids, **kwargs)
            if self.prompt_len is None:
                self.prompt_len = int(input_ids.shape[1])
            step = int(input_ids.shape[1]) - self.prompt_len
            if step < len(script):
                out.logits[:, -1, script[step]] += 1e4
            return out

    base = _gpt2(vocab_size)
    model = ScriptedGPT2(base.config)
    model.load_state_dict(base.state_dict())
    return model


def _agent(model: Any, tok: Any, *, max_new_tokens: int = 12, **config: Any) -> Any:
    agent = MultiTurnAgent(
        AgentConfig(
            model_name="gpt2",
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            do_sample=True,
            use_chat_template=False,
            **config,
        )
    )
    agent.model = model
    agent.tokenizer = tok
    agent.generation_config = agent._build_generation_config()
    if hasattr(model, "prompt_len"):
        agent.generation_config.use_cache = False  # scripted model reads full ids
    return agent


def _ids(tok: Any, pieces: list[str]) -> list[int]:
    return [int(tok.convert_tokens_to_ids(p)) for p in pieces]


def _decoded(tok: Any, turn: Any) -> str:
    return str(tok.decode(turn.metadata["token_ids"], skip_special_tokens=True))


def _assert_faithful(tok: Any, turn: Any) -> None:
    md = turn.metadata
    assert md["token_ids"], "a real model must record its sampled ids"
    assert len(md["sampler_log_probs"]) == len(md["token_ids"])
    assert turn.content == _decoded(tok, turn)


USER = [{"role": "user", "content": "hello"}]


# --- the invariant on sampled (unscripted) rollouts -----------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [0, 1, 2])
async def test_sampled_turn_text_equals_decoded_token_ids(seed: int) -> None:
    tok = _tokenizer()
    agent = _agent(_gpt2(len(tok), seed), tok, max_new_tokens=16)
    torch.manual_seed(seed)
    turns = await agent.generate_turns(USER, 8)
    for turn in turns:
        _assert_faithful(tok, turn)
    torch.manual_seed(seed)
    _assert_faithful(tok, await agent.generate_turn(USER))


# --- scripted generations -------------------------------------------------------


def _script(tok: Any, pieces: list[str], *, eos: bool = True) -> list[int]:
    ids = _ids(tok, pieces)
    return ids + ([int(tok.eos_token_id)] if eos else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("batched", [False, True])
async def test_multi_paragraph_reply_is_not_truncated_by_default(batched: bool) -> None:
    tok = _tokenizer()
    pieces = [*"Para one.", "\n", "\n", *"Para two.", "\n\n", *"Three"]
    script = _script(tok, pieces)
    agent = _agent(_scripted_gpt2(len(tok), script), tok, max_new_tokens=40)
    turns = (
        await agent.generate_turns(USER, 2)
        if batched
        else [await agent.generate_turn(USER)]
    )
    for turn in turns:
        assert turn.content == "Para one.\n\nPara two.\n\nThree"
        assert turn.metadata["token_ids"] == script
        _assert_faithful(tok, turn)


@pytest.mark.asyncio
@pytest.mark.parametrize("batched", [False, True])
async def test_merged_token_containing_blank_line_is_kept_verbatim(
    batched: bool,
) -> None:
    tok = _tokenizer()
    script = _script(tok, ["ok\n\nmore", *" fine "])
    agent = _agent(_scripted_gpt2(len(tok), script), tok, max_new_tokens=20)
    turns = (
        await agent.generate_turns(USER, 2)
        if batched
        else [await agent.generate_turn(USER)]
    )
    for turn in turns:
        assert turn.content == "ok\n\nmore fine "
        _assert_faithful(tok, turn)


@pytest.mark.asyncio
@pytest.mark.parametrize("batched", [False, True])
async def test_stop_sequence_on_token_boundary_preserves_sampled_stop_tokens(
    batched: bool,
) -> None:
    tok = _tokenizer()
    script = _script(tok, [*"Hello\nUser: hi there"])
    agent = _agent(
        _scripted_gpt2(len(tok), script),
        tok,
        max_new_tokens=30,
        stop_sequences=["User:"],
    )
    turns = (
        await agent.generate_turns(USER, 2)
        if batched
        else [await agent.generate_turn(USER)]
    )
    for turn in turns:
        assert turn.content == "Hello\n"
        assert turn.metadata["token_ids"] == _ids(tok, [*"Hello\nUser:"])
        assert len(turn.metadata["sampler_log_probs"]) == len(
            turn.metadata["token_ids"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("batched", [False, True])
async def test_stop_sequence_inside_a_merged_token_preserves_the_complete_action(
    batched: bool,
) -> None:
    tok = _tokenizer()
    script = _script(tok, [*"Hi th", "ereUser: no", *"more"])
    agent = _agent(
        _scripted_gpt2(len(tok), script),
        tok,
        max_new_tokens=30,
        stop_sequences=["User:"],
    )
    turns = (
        await agent.generate_turns(USER, 2)
        if batched
        else [await agent.generate_turn(USER)]
    )
    for turn in turns:
        # Display can end within a token; policy loss must retain that token.
        assert turn.content == "Hi there"
        assert turn.metadata["token_ids"] == _ids(tok, [*"Hi th", "ereUser: no"])
        assert len(turn.metadata["sampler_log_probs"]) == len(
            turn.metadata["token_ids"]
        )


@pytest.mark.asyncio
async def test_stop_inside_merged_token_also_ends_generation() -> None:
    tok = _tokenizer()
    script = _script(tok, [*"Hi th", "ereUser: no", *"more"])
    model = _scripted_gpt2(len(tok), script)
    agent = _agent(model, tok, max_new_tokens=30, stop_sequences=["User:"])
    calls: list[int] = []
    original = model.forward

    def counting_forward(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    model.forward = counting_forward  # type: ignore[method-assign]
    await agent.generate_turn(USER)
    # 5 single chars + the merged token: generation stops right after it.
    assert len(calls) == 6


def test_stop_sequences_default_to_empty() -> None:
    assert list(AgentConfig(model_name="gpt2").stop_sequences) == []


# --- canned text injection ------------------------------------------------------


BILLING = [{"role": "user", "content": "I see an unexpected charge on my bill"}]


@pytest.mark.asyncio
async def test_no_canned_billing_text_by_default() -> None:
    tok = _tokenizer()
    agent = _agent(_scripted_gpt2(len(tok), _script(tok, [*"ok"])), tok)
    assert await agent.generate_response(BILLING) == "ok"
    turn = await agent.generate_turn(BILLING)
    assert turn.content == "ok"


@pytest.mark.asyncio
async def test_persona_hints_are_opt_in_and_never_reach_rollout_turns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tok = _tokenizer()
    agent = _agent(
        _scripted_gpt2(len(tok), _script(tok, [*"ok"])),
        tok,
        enable_persona_hints=True,
    )
    assert "billing" in (await agent.generate_response(BILLING)).lower()
    with caplog.at_level(logging.WARNING, logger="stateset_agents.core.agent"):
        turn = await agent.generate_turn(BILLING)
        turns = await agent.generate_turns(BILLING, 2)
    assert turn.content == "ok"
    assert [t.content for t in turns] == ["ok", "ok"]
    assert any("enable_persona_hints" in r.getMessage() for r in caplog.records)


# --- top_k ---------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, None])
def test_top_k_zero_or_none_means_disabled(value: Any) -> None:
    tok = _tokenizer()
    agent = _agent(_gpt2(len(tok)), tok, top_k=value)
    assert agent.generation_config.top_k == 0


def test_top_k_default_and_bounds_are_unchanged() -> None:
    assert AgentConfig(model_name="gpt2").top_k == 50
    for bad in (-1, 1001, 2.5, True):
        with pytest.raises(ConfigValidationError):
            AgentConfig(model_name="gpt2", top_k=bad)


@pytest.mark.asyncio
async def test_rollout_backend_receives_the_configured_top_k() -> None:
    tok = _tokenizer()
    seen: list[dict[str, Any]] = []

    class Backend:
        async def generate_with_logprobs(self, prompts: Any, **kwargs: Any) -> Any:
            from types import SimpleNamespace

            seen.append(kwargs)
            ids = [int(t) for t in tok(prompts[0])["input_ids"]]
            return [
                SimpleNamespace(
                    prompt=p,
                    response="ok",
                    prompt_token_ids=ids,
                    response_token_ids=_ids(tok, ["o", "k"]),
                    token_logprobs=[-0.1, -0.2],
                )
                for p in prompts
            ]

    for top_k, expected in ((0, -1), (None, -1), (7, 7)):
        agent = _agent(_gpt2(len(tok)), tok, top_k=top_k)
        agent.set_rollout_backend(Backend())
        await agent.generate_turn(USER)
        assert seen[-1]["top_k"] == expected


@pytest.mark.asyncio
async def test_rollout_backend_stop_display_preserves_every_sampled_token() -> None:
    from types import SimpleNamespace

    tok = _tokenizer()
    action_ids = _ids(tok, [*"Hi th", "ereUser: no"])
    probabilities = [-0.1] * len(action_ids)

    class Backend:
        async def generate_with_logprobs(self, prompts: Any, **kwargs: Any) -> Any:
            return [
                SimpleNamespace(
                    prompt=prompt,
                    response="Hi thereUser: no",
                    prompt_token_ids=list(tok(prompt)["input_ids"]),
                    response_token_ids=list(action_ids),
                    token_logprobs=list(probabilities),
                )
                for prompt in prompts
            ]

    agent = _agent(_gpt2(len(tok)), tok, stop_sequences=["User:"])
    agent.set_rollout_backend(Backend())
    for turn in [
        await agent.generate_turn(USER),
        *(await agent.generate_turns(USER, 2)),
    ]:
        assert turn.content == "Hi there"
        assert turn.metadata["token_ids"] == action_ids
        assert turn.metadata["sampler_log_probs"] == probabilities


@pytest.mark.parametrize("bad", ["User:", [""], [3], None])
def test_stop_sequences_must_be_a_list_of_non_empty_strings(bad: Any) -> None:
    with pytest.raises(ConfigValidationError):
        AgentConfig(model_name="gpt2", stop_sequences=bad)


def test_legacy_stop_sequences_are_exported_for_opt_in() -> None:
    from stateset_agents.core.agent import LEGACY_STOP_SEQUENCES

    cfg = AgentConfig(model_name="gpt2", stop_sequences=list(LEGACY_STOP_SEQUENCES))
    assert cfg.stop_sequences == ["User:", "System:", "\n\n"]
