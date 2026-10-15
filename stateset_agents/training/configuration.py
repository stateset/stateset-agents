"""Torch-free resolution of CLI, API and Python training configuration."""

from __future__ import annotations

import difflib
from typing import Any

from .config import TrainingConfig, TrainingProfile


def resolve_training_config(
    *,
    profile: str,
    num_episodes: int,
    config_overrides: dict[str, Any] | None,
    resume_from_checkpoint: str | None,
    objective: str | None,
    objective_overrides: dict[str, Any] | None,
    extra_config: dict[str, Any],
) -> TrainingConfig:
    """Merge every ``train()`` config source, rejecting unknown or conflicting keys."""
    field_names = set(TrainingConfig.__dataclass_fields__)
    unknown = sorted(set(extra_config) - field_names)
    if unknown:
        hints = []
        for name in unknown:
            close = difflib.get_close_matches(name, field_names, n=1)
            hints.append(
                f"{name!r}" + (f" (did you mean {close[0]!r}?)" if close else "")
            )
        raise TypeError(
            "train() got unexpected keyword argument(s): " + ", ".join(hints)
        )

    merged = dict(config_overrides or {})
    named = {
        "resume_from_checkpoint": resume_from_checkpoint,
        "objective": objective,
        "objective_overrides": objective_overrides,
    }
    sources = [
        (name, value) for name, value in named.items() if value is not None
    ] + list(extra_config.items())
    for name, value in sources:
        if name in merged and merged[name] != value:
            raise TypeError(
                f"train() got conflicting values for {name!r}: "
                f"{merged[name]!r} and {value!r}"
            )
        merged[name] = value
    if "num_episodes" in merged:
        raise TypeError(
            "pass num_episodes as train()'s own argument, not as a config override"
        )
    if merged.get("objective") is not None:
        from .objectives import OBJECTIVES

        if merged["objective"] not in OBJECTIVES:
            raise ValueError(
                f"objective {merged['objective']!r} is not a known preset; "
                f"expected one of: {', '.join(sorted(OBJECTIVES))}"
            )

    config = TrainingConfig.from_profile(
        profile=TrainingProfile(profile),
        num_episodes=num_episodes,
        **merged,
    )
    config.validate_for_training()
    return config
