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
