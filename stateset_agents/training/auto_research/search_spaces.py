"""
Auto-research specific search spaces.

These are tailored for the autonomous research loop — they define the
hyperparameter dimensions that the proposer can explore.
"""

from __future__ import annotations

from collections.abc import Callable

from stateset_agents.training.hpo.base import (
    SearchDimension,
    SearchSpace,
    SearchSpaceType,
)


def create_auto_research_search_space() -> SearchSpace:
    """Comprehensive search space for autonomous research.

    Covers the same parameters that autoresearch's train.py exposes:
    learning rate, LoRA config, GSPO params, reward domain, generation
    settings, and training budget.
    """
    return SearchSpace([
        # Learning rate (most impactful)
        SearchDimension(
            "learning_rate", SearchSpaceType.LOGUNIFORM,
            low=1e-6, high=1e-3, default=5e-6,
        ),
        # LoRA configuration
        SearchDimension(
            "lora_r", SearchSpaceType.CHOICE,
            choices=[4, 8, 16, 32], default=8,
        ),
        SearchDimension(
            "lora_alpha", SearchSpaceType.CHOICE,
            choices=[8, 16, 32, 64], default=16,
        ),
        SearchDimension(
            "lora_dropout", SearchSpaceType.UNIFORM,
            low=0.0, high=0.2, default=0.05,
        ),
        # GSPO specific
        SearchDimension(
            "num_outer_iterations", SearchSpaceType.INT,
            low=1, high=5, default=1,
        ),
        SearchDimension(
            "generations_per_iteration", SearchSpaceType.INT,
            low=1, high=10, default=3,
        ),
        SearchDimension(
            "num_generations", SearchSpaceType.INT,
            low=2, high=16, default=4,
        ),
        SearchDimension(
            "clip_range_left", SearchSpaceType.LOGUNIFORM,
            low=1e-5, high=1e-2, default=3e-4,
        ),
        SearchDimension(
            "clip_range_right", SearchSpaceType.LOGUNIFORM,
            low=1e-5, high=1e-2, default=4e-4,
        ),
        # General RL
        SearchDimension(
            "entropy_coef", SearchSpaceType.LOGUNIFORM,
            low=1e-4, high=0.1, default=0.01,
        ),
        SearchDimension(
            "beta", SearchSpaceType.UNIFORM,
            low=0.0, high=0.5, default=0.0,
        ),
        SearchDimension(
            "warmup_ratio", SearchSpaceType.UNIFORM,
            low=0.0, high=0.3, default=0.1,
        ),
        # Generation
        SearchDimension(
            "temperature", SearchSpaceType.UNIFORM,
            low=0.1, high=1.5, default=0.7,
        ),
        SearchDimension(
            "top_p", SearchSpaceType.UNIFORM,
            low=0.5, high=1.0, default=0.9,
        ),
    ])


def create_multi_algorithm_search_space() -> SearchSpace:
    """Search space that includes the training algorithm as a dimension.

    Use with ``trainer_algorithm="auto"`` to let the proposer explore
    which RL algorithm works best for your task.
    """
    return SearchSpace([
        SearchDimension(
            "algorithm", SearchSpaceType.CATEGORICAL,
            choices=["gspo", "grpo", "dapo", "vapo"],
        ),
        SearchDimension(
            "learning_rate", SearchSpaceType.LOGUNIFORM,
            low=1e-6, high=1e-3, default=5e-6,
        ),
        SearchDimension(
            "num_generations", SearchSpaceType.INT,
            low=2, high=16, default=4,
        ),
        SearchDimension(
            "lora_r", SearchSpaceType.CHOICE,
            choices=[4, 8, 16, 32], default=8,
        ),
        SearchDimension(
            "temperature", SearchSpaceType.UNIFORM,
            low=0.1, high=1.5, default=0.7,
        ),
        SearchDimension(
            "entropy_coef", SearchSpaceType.LOGUNIFORM,
            low=1e-4, high=0.1, default=0.01,
        ),
        SearchDimension(
            "warmup_ratio", SearchSpaceType.UNIFORM,
            low=0.0, high=0.3, default=0.1,
        ),
    ])


def create_quick_search_space() -> SearchSpace:
    """Small search space for quick experimentation.

    Just the 4 most impactful hyperparameters.
    """
    return SearchSpace([
        SearchDimension(
            "learning_rate", SearchSpaceType.LOGUNIFORM,
            low=1e-6, high=1e-3, default=5e-6,
        ),
        SearchDimension(
            "num_generations", SearchSpaceType.INT,
            low=2, high=16, default=4,
        ),
        SearchDimension(
            "temperature", SearchSpaceType.UNIFORM,
            low=0.1, high=1.5, default=0.7,
        ),
        SearchDimension(
            "lora_r", SearchSpaceType.CHOICE,
            choices=[4, 8, 16, 32], default=8,
        ),
    ])


def create_reward_search_space() -> SearchSpace:
    """Search space focused on reward engineering.

    Explores reward weights and domain combinations.
    """
    return SearchSpace([
        SearchDimension(
            "reward_domain", SearchSpaceType.CATEGORICAL,
            choices=["customer_service", "technical_support", "sales"],
        ),
        SearchDimension(
            "empathy_weight", SearchSpaceType.UNIFORM,
            low=0.0, high=2.0, default=1.0,
        ),
        SearchDimension(
            "professionalism_weight", SearchSpaceType.UNIFORM,
            low=0.0, high=2.0, default=1.0,
        ),
        SearchDimension(
            "action_oriented_weight", SearchSpaceType.UNIFORM,
            low=0.0, high=2.0, default=1.0,
        ),
        SearchDimension(
            "reasoning_weight", SearchSpaceType.UNIFORM,
            low=0.0, high=2.0, default=1.0,
        ),
        SearchDimension(
            "length_weight", SearchSpaceType.UNIFORM,
            low=0.0, high=2.0, default=1.0,
        ),
    ])


def create_model_search_space() -> SearchSpace:
    """Search space for model architecture exploration.

    Explores LoRA configuration, quantization, and generation params.
    """
    return SearchSpace([
        SearchDimension(
            "lora_r", SearchSpaceType.CHOICE,
            choices=[4, 8, 16, 32, 64], default=8,
        ),
        SearchDimension(
            "lora_alpha", SearchSpaceType.CHOICE,
            choices=[8, 16, 32, 64, 128], default=16,
        ),
        SearchDimension(
            "lora_dropout", SearchSpaceType.UNIFORM,
            low=0.0, high=0.3, default=0.05,
        ),
        SearchDimension(
            "max_completion_length", SearchSpaceType.CHOICE,
            choices=[32, 64, 128, 256], default=64,
        ),
        SearchDimension(
            "temperature", SearchSpaceType.UNIFORM,
            low=0.1, high=1.5, default=0.7,
        ),
        SearchDimension(
            "top_p", SearchSpaceType.UNIFORM,
            low=0.5, high=1.0, default=0.9,
        ),
    ])


# Registry for auto-research search spaces
AUTO_RESEARCH_SPACES: dict[str, Callable[[], SearchSpace]] = {
    "auto_research": create_auto_research_search_space,
    "multi_algorithm": create_multi_algorithm_search_space,
    "quick": create_quick_search_space,
    "reward": create_reward_search_space,
    "model": create_model_search_space,
}


def get_auto_research_search_space(name: str) -> SearchSpace:
    """Get a predefined auto-research search space by name."""
    factory = AUTO_RESEARCH_SPACES.get(name)
    if factory is None:
        available = ", ".join(sorted(AUTO_RESEARCH_SPACES))
        raise ValueError(
            f"Unknown auto-research search space: {name!r}. "
            f"Available: {available}"
        )
    return factory()


def list_auto_research_search_spaces() -> list[str]:
    """List available auto-research search spaces."""
    return sorted(AUTO_RESEARCH_SPACES)


def validate_params_against_space(
    params: dict,
    search_space: SearchSpace,
) -> list[str]:
    """Check that baseline params fit within search space bounds.

    Returns a list of warning strings for any out-of-bounds or missing params.
    """
    warnings: list[str] = []

    for dim in search_space.dimensions:
        if dim.name not in params:
            continue

        val = params[dim.name]
        if dim.type in (SearchSpaceType.CATEGORICAL, SearchSpaceType.CHOICE):
            if val not in dim.choices:
                warnings.append(
                    f"{dim.name}={val!r} not in choices {dim.choices}"
                )
        elif dim.low is not None and dim.high is not None:
            if isinstance(val, (int, float)):
                if val < dim.low or val > dim.high:
                    warnings.append(
                        f"{dim.name}={val} outside bounds [{dim.low}, {dim.high}]"
                    )

    return warnings
