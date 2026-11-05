"""GSPO and GEPO score with the shared gather, matching the full-vocab path.

``rl_losses.gather_token_logprobs`` (``logit[y] - logsumexp(logits)`` in
fp32) replaced ``log_softmax(logits).gather(y)`` in GSPO's sequence scoring
(current and reference) and GEPO's ``compute_sequence_log_probs``. In fp32
both are the same quantity up to rounding: per-token log-probs agree to 1e-6
absolute; sequence sums (tens of nats in fp32) to 1e-6 relative. bf16
policies are now scored in fp32.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch.nn.functional as F  # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from stateset_agents.training.gepo_trainer import GEPOConfig, GEPOTrainer  # noqa: E402
from stateset_agents.training.gspo_config import GSPOConfig  # noqa: E402
from stateset_agents.training.gspo_generation import (  # noqa: E402
    build_scoring_text,
    render_prompt_for_scoring,
    scoring_prompt_length,
)
from stateset_agents.training.gspo_trainer import GSPOTrainer  # noqa: E402
from tests._tiny_tokenizer import tiny_tokenizer  # noqa: E402

ATOL = 1e-6  # per-token log-probs (|log p| ~ 5)
SUM_TOL = {"rtol": 1e-6, "atol": 1e-6}  # sums of ~20 tokens: ~1 fp32 ulp
RESPONSES = ["12 apples", "ok", "a much longer answer"]


def _model(dtype: Any = torch.float32) -> Any:
    torch.manual_seed(0)
    return GPT2LMHeadModel(
        GPT2Config(
            n_embd=32,
            n_layer=2,
            n_head=2,
            vocab_size=128,
            n_positions=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    ).to(dtype)


def _gspo(model: Any, ref_model: Any = None) -> GSPOTrainer:
    return GSPOTrainer(
        config=GSPOConfig(
            model_name="gpt2",
            num_generations=len(RESPONSES),
            max_prompt_length=32,
            max_completion_length=32,
            bf16=False,
            report_to="none",
        ),
        model=model,
        tokenizer=tiny_tokenizer(128),
        agent=None,
        environment=None,
        reward_model=None,
        ref_model=ref_model,
    )


def _full_vocab_sequence_log_probs(
    model: Any, tokenizer: Any, prompt: str, responses: list[str]
) -> tuple[Any, Any]:
    """The pre-gather GSPO scoring, verbatim: full log-softmax then gather."""
    rendered = render_prompt_for_scoring(tokenizer, prompt, None)
    prompt_length = scoring_prompt_length(tokenizer, rendered)
    tokenizer.padding_side = "right"
    inputs = tokenizer(
        [build_scoring_text(rendered, r) for r in responses],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=prompt_length + 32,
        add_special_tokens=False,
    )
    logits = model(**inputs).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = inputs["input_ids"][:, 1:]
    shift_mask = inputs["attention_mask"][:, 1:]
    token_log_probs = (
        F.log_softmax(shift_logits, dim=-1)
        .gather(-1, shift_labels.unsqueeze(-1))
        .squeeze(-1)
    )
    response_mask = torch.zeros_like(shift_mask)
    start = max(prompt_length - 1, 0)
    response_mask[:, start:] = shift_mask[:, start:]
    lengths = response_mask.sum(-1).to(torch.float32).clamp(min=1.0)
    return (token_log_probs * response_mask).sum(-1), lengths


def test_gathered_token_log_probs_match_full_vocab_log_softmax() -> None:
    from stateset_agents.training import rl_losses

    model = _model()
    torch.manual_seed(1)
    input_ids = torch.randint(0, 128, (4, 24))
    logits = model(input_ids=input_ids).logits
    for scale in (1.0, 30.0):  # also sharply peaked distributions
        scaled = logits * scale
        got, _ = rl_losses.gather_token_logprobs(
            scaled, input_ids, torch.ones_like(input_ids, dtype=torch.float32)
        )
        want = (
            F.log_softmax(scaled[:, :-1, :], dim=-1)
            .gather(-1, input_ids[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )
        torch.testing.assert_close(got, want, rtol=0, atol=ATOL * scale)


def test_gspo_current_scoring_matches_full_vocab_values_and_gradients() -> None:
    model = _model()
    trainer = _gspo(model)
    got, got_lengths = trainer._compute_group_sequence_log_probs("hi", RESPONSES)
    want, want_lengths = _full_vocab_sequence_log_probs(
        model, trainer.tokenizer, "hi", RESPONSES
    )
    torch.testing.assert_close(got, want, **SUM_TOL)
    assert torch.equal(got_lengths, want_lengths)

    got.sum().backward()
    gathered = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    want.sum().backward()
    for left, right in zip(gathered, (p.grad for p in model.parameters()), strict=True):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=ATOL)


def test_gspo_reference_scoring_matches_full_vocab() -> None:
    ref = _model()
    trainer = _gspo(_model(), ref_model=ref)
    batch = trainer._compute_batch_ref_log_probs("hi", RESPONSES)
    with torch.no_grad():
        want, _ = _full_vocab_sequence_log_probs(
            ref, trainer.tokenizer, "hi", RESPONSES
        )
    assert not batch.requires_grad
    torch.testing.assert_close(batch, want, **SUM_TOL)


@pytest.mark.asyncio
async def test_gspo_single_reference_score_equals_its_batch_row() -> None:
    trainer = _gspo(_model(), ref_model=_model())
    batch = trainer._compute_batch_ref_log_probs("hi", RESPONSES)
    for index, response in enumerate(RESPONSES):
        single = await trainer._compute_ref_log_prob("hi", response)
        assert single == pytest.approx(float(batch[index]), rel=1e-6, abs=ATOL)


def test_gepo_sequence_log_probs_match_full_vocab() -> None:
    model = _model()
    trainer = GEPOTrainer(
        config=GEPOConfig(model_name="gpt2", group_size=2, report_to="none"),
        model=model,
        tokenizer=None,
        reward_fn=lambda prompt, response: 0.0,
    )
    input_ids = torch.tensor([[5, 6, 7, 17, 18, 19], [5, 6, 7, 20, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
    tokens, sequences = trainer.compute_sequence_log_probs(
        input_ids, attention_mask, response_start_idx=3
    )

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    want_tokens = (
        F.log_softmax(logits[:, :-1, :], dim=-1)
        .gather(-1, input_ids[:, 1:].unsqueeze(-1))
        .squeeze(-1)
    )
    mask = GEPOTrainer.build_response_mask(attention_mask, 3)
    torch.testing.assert_close(tokens, want_tokens, rtol=0, atol=ATOL)
    torch.testing.assert_close(sequences, (want_tokens * mask).sum(-1), **SUM_TOL)


def test_bf16_policies_are_scored_in_fp32() -> None:
    trainer = _gspo(_model(torch.bfloat16))
    log_probs, _ = trainer._compute_group_sequence_log_probs("hi", RESPONSES)
    assert log_probs.dtype == torch.float32

    gepo = GEPOTrainer(
        config=GEPOConfig(model_name="gpt2", group_size=2, report_to="none"),
        model=_model(torch.bfloat16),
        tokenizer=None,
        reward_fn=lambda prompt, response: 0.0,
    )
    tokens, sequences = gepo.compute_sequence_log_probs(
        torch.tensor([[5, 6, 7, 8]]), torch.ones(1, 4, dtype=torch.long), 2
    )
    assert tokens.dtype == sequences.dtype == torch.float32
