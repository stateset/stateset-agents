"""GSPO scoring must not count the tail of a long prompt as response tokens.

The rendered prompt used to be measured with truncation at
``max_prompt_length`` while the full prompt+response text was only truncated
at ``max_prompt_length + max_completion_length``; any prompt longer than the
limit leaked its tail into the response mask (and with a very long prompt the
response was cut entirely)."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.training.gspo_config import GSPOConfig  # noqa: E402
from stateset_agents.training.gspo_generation import (  # noqa: E402
    render_prompt_for_scoring,
)
from stateset_agents.training.gspo_trainer import GSPOTrainer  # noqa: E402
from tests.unit.test_loss_computation_real_tokenizer_masks import (  # noqa: E402
    CHATML,
    _byte_tokenizer,
)


class _ZeroLM(torch.nn.Module):
    def __init__(self, vocab: int):
        super().__init__()
        self.vocab = vocab
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch, length = input_ids.shape
        return types.SimpleNamespace(
            logits=torch.zeros(batch, length, self.vocab) + self.w
        )


@pytest.mark.parametrize("system_chars", [10, 400, 2000])
def test_response_lengths_ignore_prompt_length(system_chars):
    tok = _byte_tokenizer(CHATML)
    trainer = GSPOTrainer.__new__(GSPOTrainer)
    trainer.config = GSPOConfig(max_prompt_length=64, max_completion_length=64)
    trainer.tokenizer = tok
    trainer.model = _ZeroLM(len(tok))
    system_prompt = "s" * system_chars
    trainer.agent = types.SimpleNamespace(
        config=types.SimpleNamespace(system_prompt=system_prompt)
    )
    responses = ["Shipped.", "Let me check that."]
    rendered = render_prompt_for_scoring(tok, "Where?", system_prompt)
    prompt_len = len(tok(rendered, add_special_tokens=False)["input_ids"])
    expected = [
        len(tok(rendered + r, add_special_tokens=False)["input_ids"]) - prompt_len
        for r in responses
    ]

    _, lengths = trainer._compute_group_sequence_log_probs("Where?", responses)
    assert lengths.tolist() == expected


def _scheduler_lrs(steps: int, **config_kwargs) -> list[float]:
    trainer = GSPOTrainer(
        config=GSPOConfig(model_name="gpt2", learning_rate=1.0, **config_kwargs),
        model=_ZeroLM(4),
        tokenizer=None,
        agent=None,
        environment=None,
        reward_model=None,
        ref_model=None,
    )
    lrs = []
    for _ in range(steps):
        trainer.optimizer.step()
        trainer.scheduler.step()
        lrs.append(trainer.scheduler.get_last_lr()[0])
    return lrs


def test_schedule_is_sized_by_optimizer_steps_taken():
    # train_step takes num_iterations optimizer steps per outer iteration, so
    # the cosine spans outer * iterations steps: it decays to zero exactly at
    # the end of training, not after the first third of it.
    lrs = _scheduler_lrs(12, num_outer_iterations=4, num_iterations=3, warmup_ratio=0)
    assert lrs[3] > 0.5 * lrs[0]
    assert lrs[-1] == pytest.approx(0.0, abs=1e-9)


def test_schedule_does_not_oscillate_past_its_end():
    # HF's cosine is periodic beyond num_training_steps: cos(pi*k) flipped the
    # LR between full and zero on alternate steps.
    lrs = _scheduler_lrs(6, num_outer_iterations=2, warmup_ratio=0)
    assert lrs[2:] == pytest.approx([lrs[1]] * 4)


def _generator_with_turn(token_ids, sampler, content, rescore=False):
    from stateset_agents.training.gspo_generation import GSPOTrajectoryGenerator

    tok = _byte_tokenizer(CHATML)
    tok.eos_token = "<|im_end|>"
    rendered = render_prompt_for_scoring(tok, "Where?", None)
    prompt_ids = tok(rendered, add_special_tokens=False)["input_ids"]

    async def generate_turns(messages, n):
        return [
            types.SimpleNamespace(
                content=content,
                metadata={
                    "prompt_token_ids": prompt_ids,
                    "token_ids": token_ids(tok),
                    "sampler_log_probs": sampler,
                },
            )
        ] * n

    agent = types.SimpleNamespace(
        tokenizer=tok,
        model=_ZeroLM(len(tok)),
        config=types.SimpleNamespace(system_prompt=None),
        generate_turns=generate_turns,
        memory_window=0,
    )
    config = GSPOConfig(
        model_name="gpt2", use_vllm=False, rescore_old_log_probs=rescore
    )
    return GSPOTrajectoryGenerator(config, agent, None), tok


@pytest.mark.asyncio
async def test_sampler_log_prob_excludes_eos_that_scoring_never_sees(monkeypatch):
    """GSPO scores decoded text (no EOS); the old log-prob must cover exactly
    those tokens, or the sequence ratio is biased by log p(EOS) / |y|."""

    def ids(tok):
        return tok("Hi", add_special_tokens=False)["input_ids"] + [tok.eos_token_id]

    gen, tok = _generator_with_turn(ids, [-0.5, -0.25, -3.0], "Hi")
    rescored = []

    async def no_rescore(*args):
        rescored.append(args)
        return 0.0

    monkeypatch.setattr(gen, "_compute_sequence_log_prob", no_rescore)
    out = await gen._generate_with_hf("Where?", 2)
    assert [lp for _, lp in out] == [pytest.approx(-0.75)] * 2
    assert rescored == []


@pytest.mark.asyncio
async def test_sampler_log_prob_rescored_when_text_does_not_round_trip(monkeypatch):
    # The cleaned content differs from the sampled tokens: rescore instead.
    def ids(tok):
        return tok("  Hi  ", add_special_tokens=False)["input_ids"]

    gen, tok = _generator_with_turn(ids, [-0.1] * 6, "Hi")

    async def rescore(prompt_text, response):
        return -9.0

    monkeypatch.setattr(gen, "_compute_sequence_log_prob", rescore)
    out = await gen._generate_with_hf("Where?", 1)
    assert out == [("Hi", -9.0)]


@pytest.mark.asyncio
async def test_vllm_old_log_prob_drops_eos_and_checks_round_trip(monkeypatch):
    gen, tok = _generator_with_turn(lambda t: [], [], "Hi")
    rendered = render_prompt_for_scoring(tok, "Where?", None)
    hi = tok("Hi", add_special_tokens=False)["input_ids"]

    async def rescore(prompt_text, response):
        return -9.0

    monkeypatch.setattr(gen, "_compute_sequence_log_prob", rescore)
    exact = types.SimpleNamespace(
        response="Hi",
        token_logprobs=[-0.5, -0.25, -3.0],
        response_token_ids=hi + [tok.eos_token_id],
        prompt_token_ids=None,
        cumulative_logprob=-3.75,
    )
    assert await gen._vllm_old_log_prob(rendered, exact) == pytest.approx(-0.75)
    mismatched = types.SimpleNamespace(**{**vars(exact), "response": "Hello"})
    assert await gen._vllm_old_log_prob(rendered, mismatched) == -9.0
