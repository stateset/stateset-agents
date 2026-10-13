"""Exact token log-probability contracts for the optional vLLM adapter."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from stateset_agents.training.vllm_backend import (
    HuggingFaceGeneratorFallback,
    VLLMGenerator,
    quick_generate,
)


def logprob(value: float) -> SimpleNamespace:
    return SimpleNamespace(logprob=value)


class FakeEngine:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def generate(self, prompts, params):
        self.calls.append((list(prompts), params))
        return self.outputs


def generator(outputs) -> tuple[VLLMGenerator, FakeEngine]:
    engine = FakeEngine(outputs)
    adapter = object.__new__(VLLMGenerator)
    adapter._initialized = True
    adapter.engine = engine
    adapter.tokenizer = SimpleNamespace(encode=lambda text: [1] if text == "P" else [])
    adapter.create_sampling_params = lambda **kwargs: kwargs
    return adapter, engine


def completion_output(*, logprobs=None, prompt="P"):
    if logprobs is None:
        logprobs = [{10: logprob(-0.2)}, {11: logprob(-0.3)}]
    return SimpleNamespace(
        prompt=prompt,
        prompt_token_ids=[1],
        outputs=[
            SimpleNamespace(
                text="answer",
                token_ids=[10, 11],
                logprobs=logprobs,
                finish_reason="stop",
            )
        ],
    )


def test_generation_uses_only_the_sampled_tokens_logprobs():
    adapter, engine = generator([completion_output()])

    result = adapter.generate_sync(["P"])

    assert result[0].response_token_ids == [10, 11]
    assert result[0].token_logprobs == [-0.2, -0.3]
    assert result[0].cumulative_logprob == pytest.approx(-0.5)
    assert engine.calls[0][0] == ["P"]


def test_generation_allows_an_empty_completion_without_logprobs():
    output = completion_output()
    output.outputs[0].token_ids = []
    output.outputs[0].logprobs = None
    adapter, _ = generator([output])

    result = adapter.generate_sync(["P"])

    assert result[0].token_logprobs == []
    assert result[0].sequence_length == 0


def test_generation_rejects_missing_logprob_list_for_sampled_tokens():
    output = completion_output()
    output.outputs[0].logprobs = None
    adapter, _ = generator([output])

    with pytest.raises(ValueError, match="sampled-token log-probabilities"):
        adapter.generate_sync(["P"])


@pytest.mark.parametrize(
    ("logprobs", "message"),
    [
        ([{99: logprob(-0.1)}, {11: logprob(-0.3)}], "sampled token"),
        ([None, {11: logprob(-0.3)}], "sampled token"),
        ([{10: logprob(-0.2)}], "sampled-token"),
        ([{10: logprob(float("nan"))}, {11: logprob(-0.3)}], "invalid"),
    ],
)
def test_generation_rejects_missing_or_invalid_sample_logprobs(logprobs, message):
    adapter, _ = generator([completion_output(logprobs=logprobs)])

    with pytest.raises(ValueError, match=message):
        adapter.generate_sync(["P"])


def test_generation_rejects_wrong_output_count_and_prompt_order():
    adapter, _ = generator([])
    with pytest.raises(ValueError, match="wrong number"):
        adapter.generate_sync(["P"])

    adapter, _ = generator([completion_output(prompt="other")])
    with pytest.raises(ValueError, match="prompt order"):
        adapter.generate_sync(["P"])


def scoring_output(*, token_ids=None, prompt_logprobs=None, prompt="PR"):
    return SimpleNamespace(
        prompt=prompt,
        prompt_token_ids=[1, 2] if token_ids is None else token_ids,
        prompt_logprobs=(
            [None, {2: logprob(-0.4)}] if prompt_logprobs is None else prompt_logprobs
        ),
    )


def test_sequence_scoring_uses_exact_response_token_logprob():
    adapter, engine = generator([scoring_output()])

    assert adapter.compute_log_probs_for_sequences(["P"], ["R"]) == [(-0.4, [-0.4])]
    assert engine.calls[0][0] == ["PR"]


def test_sequence_scoring_rejects_pair_count_and_boundary_change():
    adapter, engine = generator([scoring_output()])
    with pytest.raises(ValueError, match="same length"):
        adapter.compute_log_probs_for_sequences(["P"], [])
    assert not engine.calls

    adapter, _ = generator([scoring_output(token_ids=[3, 2])])
    with pytest.raises(ValueError, match="prompt prefix"):
        adapter.compute_log_probs_for_sequences(["P"], ["R"])


def test_sequence_scoring_rejects_wrong_result_count():
    adapter, _ = generator([])
    with pytest.raises(ValueError, match="wrong number"):
        adapter.compute_log_probs_for_sequences(["P"], ["R"])


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (scoring_output(prompt_logprobs=[None, {99: logprob(-0.4)}]), "response token"),
        (scoring_output(prompt_logprobs=[None]), "misaligned"),
        (scoring_output(prompt_logprobs=[None, {2: logprob(float("inf"))}]), "invalid"),
        (scoring_output(prompt="other"), "prompt order"),
    ],
)
def test_sequence_scoring_rejects_incomplete_evidence(output, message):
    adapter, _ = generator([output])
    with pytest.raises(ValueError, match=message):
        adapter.compute_log_probs_for_sequences(["P"], ["R"])


@pytest.mark.asyncio
async def test_group_generation_preserves_every_requested_sample():
    adapter, _ = generator([])
    adapter.generate = AsyncMock(
        side_effect=lambda prompts, *_args, **_kwargs: [
            SimpleNamespace(prompt=prompt) for prompt in prompts
        ]
    )

    groups = await adapter.generate_groups(["A", "B"], 2)

    assert len(groups["A"]) == len(groups["B"]) == 2
    assert adapter.generate.await_args.args[0] == ["A", "A", "B", "B"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompts", "count", "message"),
    [(["A", "A"], 2, "unique prompts"), (["A"], 0, "positive")],
)
async def test_group_generation_rejects_unrepresentable_requests(
    prompts, count, message
):
    adapter, _ = generator([])
    adapter.generate = AsyncMock()
    with pytest.raises(ValueError, match=message):
        await adapter.generate_groups(prompts, count)
    adapter.generate.assert_not_awaited()

    fallback = object.__new__(HuggingFaceGeneratorFallback)
    fallback.generate = AsyncMock()
    with pytest.raises(ValueError, match=message):
        await fallback.generate_groups(prompts, count)
    fallback.generate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("returned_prompts", [["A"], ["B", "A"]])
async def test_group_generation_rejects_incomplete_or_misattributed_results(
    returned_prompts,
):
    adapter, _ = generator([])
    adapter.generate = AsyncMock(
        return_value=[SimpleNamespace(prompt=prompt) for prompt in returned_prompts]
    )
    with pytest.raises(ValueError, match="incomplete|wrong prompt"):
        await adapter.generate_groups(["A"], 2)


@pytest.mark.asyncio
async def test_quick_generate_rejects_invalid_group_before_initialization(monkeypatch):
    from stateset_agents.training import vllm_backend

    create = AsyncMock()
    monkeypatch.setattr(vllm_backend, "create_generator", create)
    with pytest.raises(ValueError, match="unique prompts"):
        await quick_generate("model", ["A", "A"], num_generations=2)
    create.assert_not_called()


@pytest.mark.asyncio
async def test_gspo_batch_api_rejects_duplicate_prompt_keys():
    from stateset_agents.training.gspo_generation import GSPOTrajectoryGenerator

    generator = object.__new__(GSPOTrajectoryGenerator)
    with pytest.raises(ValueError, match="unique prompts"):
        await generator.generate_batch_groups(["A", "A"], 2)


@pytest.mark.asyncio
async def test_gspo_rejects_short_group_before_scoring():
    from stateset_agents.training.gspo_generation import GSPOTrajectoryGenerator

    generator = object.__new__(GSPOTrajectoryGenerator)
    generator.vllm_generator = SimpleNamespace(
        generate_groups=AsyncMock(
            return_value={"P": [SimpleNamespace(prompt="P", response="partial")]}
        )
    )
    generator._generate_with_hf = AsyncMock(return_value=[("native", -0.5)])

    assert await generator._generate_with_vllm("P", 2) == [("native", -0.5)]
    generator._generate_with_hf.assert_awaited_once_with("P", 2)


@pytest.mark.asyncio
async def test_dapo_rejects_short_group_before_building_tensors():
    from stateset_agents.training.dapo_trainer import DAPOTrainer

    trainer = object.__new__(DAPOTrainer)
    trainer.config = SimpleNamespace(group_size=2)
    trainer.vllm_generator = SimpleNamespace(
        generate_groups=AsyncMock(
            return_value={"P": [SimpleNamespace(prompt="P", response="partial")]}
        )
    )
    trainer._generate_with_hf = AsyncMock(return_value=[{"response": "native"}])

    assert await trainer._generate_with_vllm("P") == [{"response": "native"}]
    trainer._generate_with_hf.assert_awaited_once_with("P")
