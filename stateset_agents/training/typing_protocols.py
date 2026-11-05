"""Structural types for the RL numerics (``loss_computation``, ``objectives``,
``rl_losses``).

The losses are duck-typed on purpose: trainers pass Hugging Face models and
tokenizers, tests pass small fakes, and configs are read field by field with
defaults. These Protocols name exactly the surface the numerics touch, so the
loss signatures say what they need instead of ``Any``. Everything torch-typed
is imported only for type checking; importing this module never needs torch.

The ``Any`` that remains lives here, at the boundary with third-party objects
whose own types are ``Any`` (``nn.Module.__call__``, ``BatchEncoding``).
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

if TYPE_CHECKING:  # torch stays a lazy runtime dependency
    import torch
    from torch import Tensor

__all__ = [
    "AgentLike",
    "BackwardFn",
    "CausalLMOutput",
    "ChatTokenizerLike",
    "DeviceLike",
    "LossConfig",
    "PolicyModel",
    "ScoredTrajectory",
    "TensorLike",
    "TokenRow",
    "TokenizerLike",
    "TrajectoryGroupLike",
]

#: A torch device or anything torch accepts in its place.
DeviceLike: TypeAlias = "torch.device | str | int"

#: Values torch can turn into a tensor (``torch.as_tensor``).
TensorLike: TypeAlias = "Tensor | Sequence[float]"

#: One assistant turn on the native token path:
#: ``(prompt_token_ids, response_token_ids, trajectory_index)``.
TokenRow: TypeAlias = tuple[list[int], list[int], int]

#: Per-chunk backward callback. It may also expose ``prepare(...)`` and
#: ``backward_normalized(NormalizedLoss)``; the loss looks those up with
#: ``getattr`` (see ``grpo_accumulation``).
BackwardFn: TypeAlias = "Callable[[Tensor], None]"


class LossConfig(Protocol):
    """A training config as the losses read it.

    Every field is optional: the losses read each one with ``getattr`` and a
    documented default, so ``TrainingConfig``, a trainer-specific config, a
    ``SimpleNamespace`` in a test and ``{}`` all qualify. The fields read are
    ``entropy_coef``, ``clip_ratio``/``clip_epsilon``, ``seq_clip_ratio``,
    ``max_prompt_length``, ``max_completion_length``, ``objective``,
    ``objective_overrides``, ``reward_clip``, ``baseline_type``,
    ``advantage_normalization``, ``generation_batch_size``,
    ``old_logprobs_source``, ``bf16``/``fp16`` and the explicit-fallback
    switches ``allow_sequence_fallback``, ``allow_missing_sampler_logprobs``
    and ``allow_unmasked_labels``. Declaring no members keeps direct
    attribute access a type error, so a new read must state its default.
    """


class CausalLMOutput(Protocol):
    """What the losses read from a causal-LM forward pass."""

    @property
    def logits(self) -> Tensor: ...

    @property
    def loss(self) -> Tensor: ...


class PolicyModel(Protocol):
    """A causal LM called with keyword tensors (``input_ids``, ``labels``...).

    ``nn.Module.__call__`` is typed ``Callable[..., Any]``, so every torch
    model satisfies this. Optional attributes (``device``, ``training``,
    ``eval``/``train``, the DDP markers) are read with ``getattr``.
    """

    def __call__(self, **inputs: Tensor | None) -> CausalLMOutput: ...


class TokenizerLike(Protocol):
    """A Hugging Face style tokenizer call returning a mutable encoding.

    Chat-template support (``apply_chat_template``/``chat_template``) is
    optional and probed with ``hasattr``.
    """

    def __call__(self, text: str, **kwargs: Any) -> MutableMapping[str, Any]: ...


class ChatTokenizerLike(TokenizerLike, Protocol):
    """A tokenizer with a chat template (``apply_chat_template``)."""

    chat_template: str | None

    def apply_chat_template(
        self, conversation: list[dict[str, str]], /, **kwargs: Any
    ) -> Any: ...


class AgentLike(Protocol):
    """The agent attributes the losses read; both may be unset (``None``)."""

    @property
    def model(self) -> PolicyModel | None: ...

    @property
    def tokenizer(self) -> TokenizerLike | None: ...


class ScoredTrajectory(Protocol):
    """A rollout with a scalar reward (``MultiTurnTrajectory``)."""

    @property
    def total_reward(self) -> float: ...


class TrajectoryGroupLike(Protocol):
    """One prompt's group of rollouts.

    Trajectories are typed ``object`` because the losses read them
    defensively (``turns``, ``metadata``, ``log_probs`` via ``getattr``);
    only the reward is required, via :class:`ScoredTrajectory`.
    """

    @property
    def trajectories(self) -> Sequence[object]: ...
