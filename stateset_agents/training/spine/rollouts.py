"""Rollout sources shared by the prompt/response trainers.

:class:`PromptGroupRollouts` is the rollout phase of DAPO, GEPO and VAPO: one
sampled group per prompt, turned by the trainer's ``build`` callback (scoring,
advantages, frozen old-policy log-probs) into the items an
:class:`~stateset_agents.training.spine.update.UpdateLoop` runs over.
Generation itself stays the trainer's overridable ``generate_group_responses``
(HF: ``trainer_runtime.hf_generate_group``).

:class:`VLLMEngineMixin` and :func:`vllm_engine_config` are the one vLLM
rollout-engine setup used by DAPO and the GSPO/base trajectory generators.

This module never imports ``torch``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

_Item = TypeVar("_Item")


class PromptGroupRollouts:
    """Sample one response group per prompt and build an update item from it.

    ``generate(prompt)`` returns the group (the trainer's own
    ``generate_group_responses``, so tests and subclasses can replace it).
    Groups smaller than ``min_group_size`` are skipped with a warning (a
    group-relative advantage needs at least two samples). ``build(prompt,
    responses)`` scores the group and returns its item, or ``None`` to drop it
    (e.g. DAPO's dynamic sampling). :meth:`collect` stops once ``limit`` items
    were built.
    """

    def __init__(
        self,
        generate: Callable[[str], Awaitable[Sequence[Any]]],
        *,
        min_group_size: int = 1,
        label: str = "rollout",
    ) -> None:
        self.generate = generate
        self.min_group_size = int(min_group_size)
        self.label = label
        #: Prompts sampled by the last :meth:`collect`.
        self.prompts_processed = 0
        #: Groups the last :meth:`collect` skipped (too small or dropped).
        self.groups_dropped = 0

    async def collect(
        self,
        prompts: Iterable[str],
        build: Callable[[str, list[Any]], Awaitable[_Item | None]],
        *,
        limit: int | None = None,
    ) -> list[_Item]:
        """Return the built items, in prompt order."""
        self.prompts_processed = 0
        self.groups_dropped = 0
        items: list[_Item] = []
        for prompt in prompts:
            if limit is not None and len(items) >= limit:
                break
            responses = list(await self.generate(prompt))
            self.prompts_processed += 1
            if len(responses) < self.min_group_size:
                logger.warning(
                    "%s: %d response(s) for a prompt, need %d; skipping it",
                    self.label,
                    len(responses),
                    self.min_group_size,
                )
                self.groups_dropped += 1
                continue
            item = await build(prompt, responses)
            if item is None:
                self.groups_dropped += 1
            else:
                items.append(item)
        return items


def vllm_dtype(config: Any) -> str:
    """vLLM engine dtype for a training config: fp16, bf16, else checkpoint."""
    if getattr(config, "fp16", False):
        return "float16"
    if getattr(config, "bf16", False):
        return "bfloat16"
    return "auto"


def vllm_engine_config(config: Any, config_cls: Any) -> Any:
    """Build a ``VLLMConfig`` (``config_cls``) from a trainer config.

    Every ``vllm_*`` training field is honoured; absent fields fall back to
    the ``TrainingConfig`` defaults.
    """
    return config_cls(
        model_name=config.model_name,
        revision=getattr(config, "model_revision", None),
        gpu_memory_utilization=getattr(config, "vllm_gpu_memory_utilization", 0.85),
        tensor_parallel_size=getattr(config, "vllm_tensor_parallel_size", 1),
        enable_prefix_caching=getattr(config, "vllm_enable_prefix_caching", True),
        max_model_len=getattr(config, "vllm_max_model_len", None),
        quantization=getattr(config, "vllm_quantization", None),
        enable_chunked_prefill=getattr(config, "vllm_enable_chunked_prefill", True),
        max_tokens=config.max_completion_length,
        temperature=config.temperature,
        top_p=config.top_p,
        dtype=vllm_dtype(config),
    )


class VLLMEngineMixin:
    """vLLM rollout-engine lifecycle: create, initialize, report readiness.

    The host sets ``config``, calls :meth:`_reset_vllm_state` in ``__init__``
    and, after loading its (patchable) backend symbols, calls
    :meth:`_install_vllm_generator`. ``_vllm_label`` names the host in logs;
    ``_vllm_init_exceptions`` are the initialization failures that fall back
    to HuggingFace generation.
    """

    config: Any
    vllm_generator: Any | None
    _vllm_initialized: bool
    _vllm_label: str = "vLLM rollouts"
    _vllm_init_exceptions: tuple[type[BaseException], ...] = (
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        OSError,
    )

    def _reset_vllm_state(self) -> None:
        self.vllm_generator = None
        self._vllm_initialized = False

    def _install_vllm_generator(self, config_cls: Any, generator_cls: Any) -> None:
        """Create the engine from the loaded backend classes (lazy start)."""
        if config_cls is None or generator_cls is None:
            logger.warning(
                "%s: vLLM backend unavailable, using HuggingFace generation",
                self._vllm_label,
            )
            return
        self.vllm_generator = generator_cls(vllm_engine_config(self.config, config_cls))

    async def initialize_vllm(self) -> bool:
        """Start the vLLM engine; False (HuggingFace fallback) on failure."""
        generator = self.vllm_generator
        if generator is None:
            return False
        if self._vllm_initialized:
            return True
        try:
            success = bool(await generator.initialize())
        except self._vllm_init_exceptions as error:
            logger.warning(
                "%s: failed to initialize vLLM (%s); using HuggingFace generation",
                self._vllm_label,
                error,
            )
            self._vllm_initialized = False
            return False
        self._vllm_initialized = success
        if success:
            logger.info("%s: vLLM engine initialized", self._vllm_label)
        return success

    @property
    def using_vllm(self) -> bool:
        """Whether generation goes through an initialized vLLM engine."""
        return bool(self._vllm_initialized and self.vllm_generator is not None)


__all__ = [
    "PromptGroupRollouts",
    "VLLMEngineMixin",
    "vllm_dtype",
    "vllm_engine_config",
]
