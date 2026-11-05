"""``spine.rollouts``: prompt-group rollouts and the shared vLLM engine setup."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.spine import (
    PromptGroupRollouts,
    VLLMEngineMixin,
    vllm_dtype,
    vllm_engine_config,
)


def _generator(groups: dict[str, list[Any]]) -> tuple[Any, list[str]]:
    calls: list[str] = []

    async def generate(prompt: str) -> list[Any]:
        calls.append(prompt)
        return groups[prompt]

    return generate, calls


@pytest.mark.asyncio
async def test_collect_builds_items_in_prompt_order() -> None:
    generate, calls = _generator({"a": [1, 2], "b": [3, 4]})

    async def build(prompt: str, responses: list[Any]) -> tuple[str, int]:
        return prompt, sum(responses)

    rollouts = PromptGroupRollouts(generate)
    assert await rollouts.collect(["a", "b"], build) == [("a", 3), ("b", 7)]
    assert calls == ["a", "b"]
    assert rollouts.prompts_processed == 2 and rollouts.groups_dropped == 0


@pytest.mark.asyncio
async def test_small_and_rejected_groups_are_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    generate, _ = _generator({"short": [1], "drop": [1, 2], "keep": [3, 4]})
    built: list[str] = []

    async def build(prompt: str, responses: list[Any]) -> str | None:
        built.append(prompt)
        return None if prompt == "drop" else prompt

    rollouts = PromptGroupRollouts(generate, min_group_size=2, label="GEPO")
    with caplog.at_level(logging.WARNING):
        items = await rollouts.collect(["short", "drop", "keep"], build)
    assert items == ["keep"]
    assert built == ["drop", "keep"]  # a too-small group is never built
    assert "GEPO" in caplog.text
    assert rollouts.prompts_processed == 3 and rollouts.groups_dropped == 2


@pytest.mark.asyncio
async def test_limit_stops_sampling_once_enough_items_were_built() -> None:
    generate, calls = _generator({p: [1, 2] for p in "abcd"})

    async def build(prompt: str, responses: list[Any]) -> str | None:
        return None if prompt == "a" else prompt

    rollouts = PromptGroupRollouts(generate)
    assert await rollouts.collect("abcd", build, limit=2) == ["b", "c"]
    assert calls == ["a", "b", "c"]
    assert rollouts.prompts_processed == 3 and rollouts.groups_dropped == 1
    # Counters describe the last collect only.
    assert await rollouts.collect("d", build, limit=0) == []
    assert rollouts.prompts_processed == 0


@pytest.mark.parametrize(
    ("fp16", "bf16", "dtype"),
    [(True, False, "float16"), (False, True, "bfloat16"), (False, False, "auto")],
)
def test_vllm_dtype(fp16: bool, bf16: bool, dtype: str) -> None:
    assert vllm_dtype(SimpleNamespace(fp16=fp16, bf16=bf16)) == dtype


def test_vllm_engine_config_honours_every_vllm_field() -> None:
    config = TrainingConfig(
        model_name="org/model",
        model_revision="abc123",
        report_to="none",
        bf16=True,
        vllm_gpu_memory_utilization=0.5,
        vllm_tensor_parallel_size=2,
        vllm_enable_prefix_caching=False,
        vllm_max_model_len=4096,
        vllm_quantization="awq",
        vllm_enable_chunked_prefill=False,
        max_completion_length=77,
        temperature=0.3,
        top_p=0.8,
    )
    kwargs = vllm_engine_config(config, lambda **kw: kw)
    assert kwargs == {
        "model_name": "org/model",
        "revision": "abc123",
        "gpu_memory_utilization": 0.5,
        "tensor_parallel_size": 2,
        "enable_prefix_caching": False,
        "max_model_len": 4096,
        "quantization": "awq",
        "enable_chunked_prefill": False,
        "max_tokens": 77,
        "temperature": 0.3,
        "top_p": 0.8,
        "dtype": "bfloat16",
    }


class _Host(VLLMEngineMixin):
    _vllm_label = "Host"
    _vllm_init_exceptions = (RuntimeError,)

    def __init__(self) -> None:
        self.config = TrainingConfig(report_to="none", bf16=False)
        self._reset_vllm_state()


class _Engine:
    def __init__(self, config: Any, *, result: Any = True) -> None:
        self.config = config
        self.result = result
        self.calls = 0

    async def initialize(self) -> Any:
        self.calls += 1
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


@pytest.mark.asyncio
async def test_engine_lifecycle() -> None:
    host = _Host()
    assert host.vllm_generator is None and host.using_vllm is False
    assert await host.initialize_vllm() is False

    host._install_vllm_generator(lambda **kw: kw, _Engine)
    engine = host.vllm_generator
    assert engine.config["dtype"] == "auto"
    assert await host.initialize_vllm() is True
    assert await host.initialize_vllm() is True  # idempotent
    assert engine.calls == 1 and host.using_vllm is True


@pytest.mark.asyncio
async def test_engine_failures_fall_back_to_huggingface(
    caplog: pytest.LogCaptureFixture,
) -> None:
    host = _Host()
    host._install_vllm_generator(None, _Engine)  # backend symbols missing
    assert host.vllm_generator is None

    host.vllm_generator = _Engine({}, result=RuntimeError("no GPU"))
    with caplog.at_level(logging.WARNING):
        assert await host.initialize_vllm() is False
    assert "Host" in caplog.text and "no GPU" in caplog.text
    assert host.using_vllm is False

    host.vllm_generator = _Engine({}, result=KeyError("unexpected"))
    with pytest.raises(KeyError):  # only the declared failures fall back
        await host.initialize_vllm()
