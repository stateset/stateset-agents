"""
Tests for GSPO scoring consistency: shared tokenization convention between
generation and scoring, loss normalization by group count, and removal of
fake-parameter injection for parameterless models.
"""

from __future__ import annotations

import logging

import pytest
import torch

pytest.importorskip("transformers")


def test_build_scoring_text_no_space_join():
    """The text scored must be exactly prompt_text + response, no injected space."""
    from stateset_agents.training import gspo_generation as gg

    assert gg.build_scoring_text("<user>hi<assistant>", "there") == (
        "<user>hi<assistant>there"
    )
    # Explicitly not the old " ".join-style behavior.
    assert gg.build_scoring_text("prompt", "response") != "prompt response"
    assert gg.build_scoring_text("prompt", "response") == "promptresponse"


@pytest.mark.asyncio
async def test_generate_with_hf_uses_rendered_chat_template_for_scoring(monkeypatch):
    """When the tokenizer exposes a chat template, generation and scoring must
    share the exact same rendered prompt text (no re-derivation / mismatch)."""
    from stateset_agents.training import gspo_generation as gg
    from stateset_agents.training.gspo_config import GSPOConfig

    captured_prompts = []

    class FakeTokenizer:
        chat_template = "{{ messages }}"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "<user>" + messages[0]["content"] + "<assistant>"

        def __call__(self, text, **kwargs):
            captured_prompts.append(text)
            n = max(len(text.split()), 1)
            return {
                "input_ids": torch.ones(1, n, dtype=torch.long),
                "attention_mask": torch.ones(1, n, dtype=torch.long),
            }

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def parameters(self, recurse=True):
            return super().parameters(recurse)

        def forward(self, input_ids, attention_mask=None):
            batch, seq_len = input_ids.shape
            vocab = 5
            logits = torch.zeros(batch, seq_len, vocab)

            class Output:
                pass

            out = Output()
            out.logits = logits
            return out

    class FakeAgent:
        def __init__(self):
            self.tokenizer = FakeTokenizer()
            self.model = FakeModel()

        async def generate_response(self, messages):
            return "there"

    config = GSPOConfig(model_name="fake-model", use_vllm=False)
    generator = gg.GSPOTrajectoryGenerator.__new__(gg.GSPOTrajectoryGenerator)
    generator.config = config
    generator.agent = FakeAgent()
    generator.environment = None
    generator.vllm_generator = None
    generator.sampling_params = None
    generator._vllm_initialized = False

    responses = await generator._generate_with_hf("hi", 1)

    assert len(responses) == 1
    response, _log_prob = responses[0]
    assert response == "there"

    # The rendered prompt (with chat template) must be what gets scored,
    # concatenated directly with the response (no injected separator).
    assert "<user>hi<assistant>there" in captured_prompts


def test_normalize_total_loss_independent_of_group_count():
    from stateset_agents.training.gspo_trainer import normalize_total_loss

    one = normalize_total_loss(torch.tensor(6.0), num_groups=1)
    three = normalize_total_loss(torch.tensor(18.0), num_groups=3)
    assert torch.allclose(one, three)


def test_normalize_total_loss_guards_zero_groups():
    from stateset_agents.training.gspo_trainer import normalize_total_loss

    result = normalize_total_loss(torch.tensor(5.0), num_groups=0)
    assert torch.allclose(result, torch.tensor(5.0))


def test_no_fake_parameter_injection():
    import inspect

    from stateset_agents.training import gspo_trainer

    src = inspect.getsource(gspo_trainer.GSPOTrainer.__init__)
    assert "nn.Parameter(torch.zeros" not in src
    assert "_stub_param" not in src


def test_rescore_old_log_probs_default_true():
    from stateset_agents.training.gspo_config import GSPOConfig

    config = GSPOConfig(model_name="fake-model")
    assert config.rescore_old_log_probs is True


class _ChatTemplateTokenizer:
    """Minimal tokenizer stub exposing a chat template, shared by the
    trainer-parity and batch-rescoring tests below."""

    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = ""
        for m in messages:
            rendered += f"<{m['role']}>{m['content']}"
        return rendered + "<assistant>"

    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            texts = text
        else:
            texts = [text]
        n = max(max((len(t.split()) for t in texts), default=1), 1)
        return {
            "input_ids": torch.ones(len(texts), n, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), n, dtype=torch.long),
        }


def test_trainer_scoring_text_matches_generation_scoring_text():
    """The trainer's current-log-prob scoring text
    (`_compute_group_sequence_log_probs`) must equal the generation-side
    scoring text (`render_prompt_for_scoring` + `build_scoring_text`) for a
    chat-template tokenizer, so the importance-ratio numerator and
    denominator are tokenized identically."""
    from stateset_agents.training import gspo_generation as gg

    tokenizer = _ChatTemplateTokenizer()
    prompt = "hi"
    response = "there"

    # Generation-side convention (used to build old_log_probs).
    rendered_prompt = gg.render_prompt_for_scoring(tokenizer, prompt, None)
    generation_scoring_text = gg.build_scoring_text(rendered_prompt, response)

    # Trainer-side convention (used to build current_log_probs), replicated
    # exactly as `_compute_group_sequence_log_probs` computes it.
    from stateset_agents.training.gspo_trainer import build_scoring_text as trainer_bst
    from stateset_agents.training.gspo_trainer import (
        render_prompt_for_scoring as trainer_rps,
    )

    trainer_rendered_prompt = trainer_rps(tokenizer, prompt, None)
    trainer_scoring_text = trainer_bst(trainer_rendered_prompt, response)

    assert trainer_scoring_text == generation_scoring_text
    assert trainer_scoring_text == "<user>hi<assistant>there"


def test_render_prompt_for_scoring_includes_system_prompt():
    from stateset_agents.training import gspo_generation as gg

    tokenizer = _ChatTemplateTokenizer()
    rendered = gg.render_prompt_for_scoring(tokenizer, "hi", "be nice")
    assert rendered == "<system>be nice<user>hi<assistant>"

    rendered_no_system = gg.render_prompt_for_scoring(tokenizer, "hi", None)
    assert rendered_no_system == "<user>hi<assistant>"


@pytest.mark.asyncio
async def test_generate_batch_groups_vllm_rescores_when_enabled(monkeypatch):
    """vLLM's batch generation path must rescore rollouts with the HF forward
    pass (same as the single-prompt path) when rescore_old_log_probs is
    enabled, instead of returning raw cumulative_logprob unconditionally."""
    from stateset_agents.training import gspo_generation as gg
    from stateset_agents.training.gspo_config import GSPOConfig

    class FakeResult:
        def __init__(self, response, cumulative_logprob):
            self.response = response
            self.cumulative_logprob = cumulative_logprob

    class FakeVLLMGenerator:
        async def generate_groups(self, prompts, num_generations_per_prompt):
            return {p: [FakeResult("there", -99.0)] for p in prompts}

    class FakeAgent:
        def __init__(self):
            self.tokenizer = _ChatTemplateTokenizer()
            self.model = None

    config = GSPOConfig(model_name="fake-model", use_vllm=False, rescore_old_log_probs=True)
    generator = gg.GSPOTrajectoryGenerator.__new__(gg.GSPOTrajectoryGenerator)
    generator.config = config
    generator.agent = FakeAgent()
    generator.environment = None
    generator.vllm_generator = FakeVLLMGenerator()
    generator.sampling_params = None
    generator._vllm_initialized = True
    generator._temperature_bias_warned = False

    rescored_calls = []

    async def fake_compute_sequence_log_prob(self, prompt_text, response):
        rescored_calls.append((prompt_text, response))
        return -1.0

    monkeypatch.setattr(
        gg.GSPOTrajectoryGenerator,
        "_compute_sequence_log_prob",
        fake_compute_sequence_log_prob,
    )

    results = await generator.generate_batch_groups(["hi"], 1)

    assert results["hi"] == [("there", -1.0)]
    # Must NOT return the raw, unrescored cumulative_logprob (-99.0).
    assert rescored_calls == [("<user>hi<assistant>", "there")]


@pytest.mark.asyncio
async def test_generate_batch_groups_vllm_returns_raw_logprob_when_rescore_disabled():
    from stateset_agents.training import gspo_generation as gg
    from stateset_agents.training.gspo_config import GSPOConfig

    class FakeResult:
        def __init__(self, response, cumulative_logprob):
            self.response = response
            self.cumulative_logprob = cumulative_logprob

    class FakeVLLMGenerator:
        async def generate_groups(self, prompts, num_generations_per_prompt):
            return {p: [FakeResult("there", -42.0)] for p in prompts}

    class FakeAgent:
        def __init__(self):
            self.tokenizer = _ChatTemplateTokenizer()
            self.model = None

    config = GSPOConfig(
        model_name="fake-model", use_vllm=False, rescore_old_log_probs=False, temperature=1.0
    )
    generator = gg.GSPOTrajectoryGenerator.__new__(gg.GSPOTrajectoryGenerator)
    generator.config = config
    generator.agent = FakeAgent()
    generator.environment = None
    generator.vllm_generator = FakeVLLMGenerator()
    generator.sampling_params = None
    generator._vllm_initialized = True
    generator._temperature_bias_warned = False

    results = await generator.generate_batch_groups(["hi"], 1)

    assert results["hi"] == [("there", -42.0)]


def test_vllm_temperature_warning_when_rescore_disabled(caplog):
    from stateset_agents.training import gspo_generation as gg
    from stateset_agents.training.gspo_config import GSPOConfig

    config = GSPOConfig(
        model_name="fake-model",
        temperature=0.7,
        rescore_old_log_probs=False,
    )
    generator = gg.GSPOTrajectoryGenerator.__new__(gg.GSPOTrajectoryGenerator)
    generator.config = config
    generator.agent = None
    generator.environment = None
    generator.vllm_generator = None
    generator.sampling_params = None
    generator._vllm_initialized = False
    generator._temperature_bias_warned = False

    with caplog.at_level(logging.WARNING):
        generator._warn_vllm_temperature_bias_once()

    assert any(
        "old-policy" in r.message.lower() or "temperature" in r.message.lower()
        for r in caplog.records
    )
