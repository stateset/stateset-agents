"""Exact token log-probability contracts for the optional vLLM adapter."""

from types import SimpleNamespace

import pytest

from stateset_agents.training.vllm_backend import VLLMGenerator


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
