"""Validated sampled tokens shared by native trainers and rollout engines."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SampledTurn:
    """Exact conditioning and sampled ids; optional behavior-policy likelihoods."""

    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    sampler_log_probs: tuple[float, ...] | None = None
    policy_version: int | None = None

    def __post_init__(self) -> None:
        for name in ("prompt_token_ids", "token_ids"):
            values = getattr(self, name)
            if not values or any(type(x) is not int or x < 0 for x in values):
                raise ValueError(f"{name} must contain nonnegative integer token ids")
        if self.sampler_log_probs is not None:
            if len(self.sampler_log_probs) != len(self.token_ids) or any(
                type(x) not in (int, float) or not math.isfinite(x) or x > 0
                for x in self.sampler_log_probs
            ):
                raise ValueError(
                    "sampler_log_probs must be finite aligned log probabilities"
                )
        if self.policy_version is not None and (
            type(self.policy_version) is not int or self.policy_version < 0
        ):
            raise ValueError("policy_version must be a nonnegative integer")

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> SampledTurn:
        """Validate metadata without coercing corrupt ids or likelihoods."""
        scores = metadata.get("sampler_log_probs")
        return cls(
            tuple(metadata.get("prompt_token_ids", ())),
            tuple(metadata.get("token_ids", ())),
            None if scores is None else tuple(scores),
            metadata.get("policy_version"),
        )


class RolloutContractError(ValueError):
    """A rollout cannot be scored without silently changing the estimator.

    Loss loops must propagate this instead of skipping the trajectory: a
    per-row skip would hide a configuration error behind a smaller batch.
    """


def require_explicit_fallback(config: Any, option: str, reason: str) -> None:
    """Reject a changed estimator unless the caller opted into compatibility."""
    if not getattr(config, option, False):
        raise RolloutContractError(
            f"{reason}. Set {option}=True to explicitly allow this fallback"
        )
