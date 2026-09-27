"""The HuggingFace fallback preserves the vLLM generation contract."""

import sys
from types import SimpleNamespace

import pytest
import torch

from stateset_agents.training.vllm_backend import (
    HuggingFaceGeneratorFallback,
    VLLMConfig,
    create_generator,
)


class ModelInputs(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, _prompt, **_kwargs):
        return ModelInputs(input_ids=torch.tensor([[1]]))

    def decode(self, _tokens, **_kwargs):
        return "answer"


class FakeModel:
    device = "cpu"

    def __init__(self, scores):
        self.scores = scores
        self.generation_kwargs = None

    def to(self, _device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        return SimpleNamespace(sequences=torch.tensor([[1, 2]]), scores=self.scores)


@pytest.mark.asyncio
async def test_factory_fallback_preserves_model_loading_config(monkeypatch):
    calls = []
    tokenizer = FakeTokenizer()
    model = FakeModel([torch.tensor([[0.0, 0.0, 1.0]])])

    def load_tokenizer(name, **kwargs):
        calls.append(("tokenizer", name, kwargs))
        return tokenizer

    def load_model(name, **kwargs):
        calls.append(("model", name, kwargs))
        return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        ),
    )
    config = VLLMConfig(
        model_name="model",
        tokenizer_name="other-tokenizer",
        revision="abc123",
        trust_remote_code=False,
    )
    fallback = create_generator(config, prefer_vllm=False)

    assert isinstance(fallback, HuggingFaceGeneratorFallback)
    assert await fallback.initialize()
    assert calls[0][0:2] == ("tokenizer", "other-tokenizer")
    assert calls[1][0:2] == ("model", "model")
    assert all(kwargs["revision"] == "abc123" for _, _, kwargs in calls)
    assert all(kwargs["trust_remote_code"] is False for _, _, kwargs in calls)


@pytest.mark.asyncio
async def test_fallback_uses_sampling_defaults_and_exact_token_scores():
    config = VLLMConfig(
        model_name="model",
        temperature=0.4,
        top_p=0.8,
        top_k=12,
        max_tokens=7,
    )
    fallback = HuggingFaceGeneratorFallback(config, device="cpu")
    fallback.tokenizer = FakeTokenizer()
    model = FakeModel([torch.tensor([[0.0, 0.0, 1.0]])])
    fallback.model = model
    fallback._initialized = True

    result = (await fallback.generate("prompt"))[0]

    assert model.generation_kwargs["temperature"] == 0.4
    assert model.generation_kwargs["top_p"] == 0.8
    assert model.generation_kwargs["top_k"] == 12
    assert model.generation_kwargs["max_new_tokens"] == 7
    assert result.response_token_ids == [2]
    assert len(result.token_logprobs) == 1
    assert result.cumulative_logprob == pytest.approx(result.token_logprobs[0])


@pytest.mark.asyncio
async def test_fallback_rejects_missing_sample_score():
    fallback = HuggingFaceGeneratorFallback("model", device="cpu")
    fallback.tokenizer = FakeTokenizer()
    fallback.model = FakeModel([])
    fallback._initialized = True

    with pytest.raises(ValueError, match="misaligned sampled-token"):
        await fallback.generate("prompt")


@pytest.mark.asyncio
async def test_fallback_honors_explicit_greedy_generation_override():
    fallback = HuggingFaceGeneratorFallback("model", device="cpu")
    fallback.tokenizer = FakeTokenizer()
    model = FakeModel([torch.tensor([[0.0, 0.0, 1.0]])])
    fallback.model = model
    fallback._initialized = True

    await fallback.generate("prompt", temperature=0, max_tokens=2, top_k=-1)

    assert model.generation_kwargs["do_sample"] is False
    assert model.generation_kwargs["temperature"] == 1.0
    assert model.generation_kwargs["max_new_tokens"] == 2
    assert model.generation_kwargs["top_k"] == 0
