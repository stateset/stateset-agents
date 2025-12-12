"""
Pre-defined search spaces for common GRPO training scenarios.

This module provides battle-tested search spaces for:
- GRPO hyperparameters
- Model architecture parameters
- Training optimization parameters
- Domain-specific configurations
"""

from typing import Dict, List, Optional
from .base import SearchSpace, SearchDimension, SearchSpaceType


# ============================================================================
# GRPO Algorithm Search Spaces
# ============================================================================

def create_grpo_search_space(
    include_value_function: bool = True,
    include_kl_penalty: bool = True,
    include_ppo_clipping: bool = False
) -> SearchSpace:
    """Create search space for core GRPO hyperparameters.

    Args:
        include_value_function: Include value function hyperparameters
        include_kl_penalty: Include KL penalty hyperparameters
        include_ppo_clipping: Include PPO-style clipping parameters

    Returns:
        SearchSpace for GRPO algorithm
    """
    dimensions = [
        # Learning rates
        SearchDimension(
            "learning_rate",
            SearchSpaceType.LOGUNIFORM,
            low=1e-6,
            high=1e-3,
            default=1e-5
        ),
        SearchDimension(
            "value_lr_multiplier",
            SearchSpaceType.UNIFORM,
            low=0.5,
            high=2.0,
            default=1.0
        ) if include_value_function else None,

        # Batch parameters
        SearchDimension(
            "per_device_train_batch_size",
            SearchSpaceType.CHOICE,
            choices=[1, 2, 4, 8, 16],
            default=4
        ),
        SearchDimension(
            "gradient_accumulation_steps",
            SearchSpaceType.CHOICE,
            choices=[1, 2, 4, 8, 16],
            default=4
        ),
        SearchDimension(
            "group_size",
            SearchSpaceType.CHOICE,
            choices=[4, 8, 16, 32],
            default=8
        ),

        # Discount and GAE
        SearchDimension(
            "gamma",
            SearchSpaceType.UNIFORM,
            low=0.9,
            high=0.999,
            default=0.99
        ),
        SearchDimension(
            "gae_lambda",
            SearchSpaceType.UNIFORM,
            low=0.9,
            high=0.99,
            default=0.95
        ) if include_value_function else None,

        # KL penalty
        SearchDimension(
            "kl_penalty_coef",
            SearchSpaceType.LOGUNIFORM,
            low=0.001,
            high=0.1,
            default=0.01
        ) if include_kl_penalty else None,

        # PPO clipping
        SearchDimension(
            "clip_range",
            SearchSpaceType.UNIFORM,
            low=0.1,
            high=0.3,
            default=0.2
        ) if include_ppo_clipping else None,

        # Gradient clipping
        SearchDimension(
            "max_grad_norm",
            SearchSpaceType.UNIFORM,
            low=0.5,
            high=2.0,
            default=1.0
        ),

        # Training schedule
        SearchDimension(
            "num_episodes",
            SearchSpaceType.CHOICE,
            choices=[100, 200, 500, 1000],
            default=500
        ),
        SearchDimension(
            "warmup_ratio",
            SearchSpaceType.UNIFORM,
            low=0.0,
            high=0.2,
            default=0.1
        ),
    ]

    # Filter out None dimensions
    dimensions = [d for d in dimensions if d is not None]

    return SearchSpace(dimensions)


def create_optimizer_search_space() -> SearchSpace:
    """Create search space for optimizer hyperparameters."""
    return SearchSpace([
        SearchDimension(
            "optimizer_type",
            SearchSpaceType.CATEGORICAL,
            choices=["adamw", "adam", "sgd", "adafactor"]
        ),
        SearchDimension(
            "adam_beta1",
            SearchSpaceType.UNIFORM,
            low=0.85,
            high=0.95,
            default=0.9
        ),
        SearchDimension(
            "adam_beta2",
            SearchSpaceType.UNIFORM,
            low=0.95,
            high=0.999,
            default=0.999
        ),
        SearchDimension(
            "adam_epsilon",
            SearchSpaceType.LOGUNIFORM,
            low=1e-9,
            high=1e-7,
            default=1e-8
        ),
        SearchDimension(
            "weight_decay",
            SearchSpaceType.LOGUNIFORM,
            low=1e-4,
            high=1e-1,
            default=1e-2
        ),
    ])


def create_model_architecture_search_space() -> SearchSpace:
    """Create search space for model architecture parameters."""
    return SearchSpace([
        SearchDimension(
            "lora_r",
            SearchSpaceType.CHOICE,
            choices=[8, 16, 32, 64],
            default=16
        ),
        SearchDimension(
            "lora_alpha",
            SearchSpaceType.CHOICE,
            choices=[8, 16, 32, 64],
            default=32
        ),
        SearchDimension(
            "lora_dropout",
            SearchSpaceType.UNIFORM,
            low=0.0,
            high=0.2,
            default=0.05
        ),
        SearchDimension(
            "value_head_hidden_size",
            SearchSpaceType.CHOICE,
            choices=[128, 256, 512, 1024],
            default=256
        ),
        SearchDimension(
            "value_head_dropout",
            SearchSpaceType.UNIFORM,
            low=0.0,
            high=0.3,
            default=0.1
        ),
    ])


def create_generation_search_space() -> SearchSpace:
    """Create search space for generation parameters."""
    return SearchSpace([
        SearchDimension(
            "temperature",
            SearchSpaceType.UNIFORM,
            low=0.7,
            high=1.5,
            default=1.0
        ),
        SearchDimension(
            "top_p",
            SearchSpaceType.UNIFORM,
            low=0.8,
            high=0.99,
            default=0.95
        ),
        SearchDimension(
            "top_k",
            SearchSpaceType.CHOICE,
            choices=[0, 10, 20, 50, 100],
            default=50
        ),
        SearchDimension(
            "max_new_tokens",
            SearchSpaceType.CHOICE,
            choices=[128, 256, 512, 1024],
            default=512
        ),
        SearchDimension(
            "repetition_penalty",
            SearchSpaceType.UNIFORM,
            low=1.0,
            high=1.3,
            default=1.1
        ),
    ])


# ============================================================================
# Composite Search Spaces
# ============================================================================

def create_full_search_space(
    include_optimizer: bool = True,
    include_architecture: bool = True,
    include_generation: bool = True
) -> SearchSpace:
    """Create comprehensive search space combining all components.

    Args:
        include_optimizer: Include optimizer hyperparameters
        include_architecture: Include model architecture parameters
        include_generation: Include generation parameters

    Returns:
        Combined SearchSpace
    """
    dimensions = []

    # Core GRPO parameters (always included)
    grpo_space = create_grpo_search_space(
        include_value_function=True,
        include_kl_penalty=True,
        include_ppo_clipping=False
    )
    dimensions.extend(grpo_space.dimensions)

    # Optional components
    if include_optimizer:
        opt_space = create_optimizer_search_space()
        dimensions.extend(opt_space.dimensions)

    if include_architecture:
        arch_space = create_model_architecture_search_space()
        dimensions.extend(arch_space.dimensions)

    if include_generation:
        gen_space = create_generation_search_space()
        dimensions.extend(gen_space.dimensions)

    return SearchSpace(dimensions)


# ============================================================================
# Domain-Specific Search Spaces
# ============================================================================

def create_customer_service_search_space() -> SearchSpace:
    """Search space optimized for customer service agents."""
    base_space = create_grpo_search_space(
        include_value_function=True,
        include_kl_penalty=True
    )

    # Add customer service specific parameters
    cs_dimensions = [
        SearchDimension(
            "helpfulness_weight",
            SearchSpaceType.UNIFORM,
            low=0.2,
            high=0.5,
            default=0.35
        ),
        SearchDimension(
            "safety_weight",
            SearchSpaceType.UNIFORM,
            low=0.15,
            high=0.35,
            default=0.25
        ),
        SearchDimension(
            "engagement_weight",
            SearchSpaceType.UNIFORM,
            low=0.15,
            high=0.30,
            default=0.20
        ),
        SearchDimension(
            "conciseness_weight",
            SearchSpaceType.UNIFORM,
            low=0.15,
            high=0.30,
            default=0.20
        ),
    ]

    dimensions = base_space.dimensions + cs_dimensions
    return SearchSpace(dimensions)


def create_technical_support_search_space() -> SearchSpace:
    """Search space optimized for technical support agents."""
    base_space = create_grpo_search_space(
        include_value_function=True,
        include_kl_penalty=True
    )

    # Add technical support specific parameters
    tech_dimensions = [
        SearchDimension(
            "correctness_weight",
            SearchSpaceType.UNIFORM,
            low=0.3,
            high=0.5,
            default=0.4
        ),
        SearchDimension(
            "helpfulness_weight",
            SearchSpaceType.UNIFORM,
            low=0.2,
            high=0.4,
            default=0.3
        ),
        SearchDimension(
            "detail_level",
            SearchSpaceType.CHOICE,
            choices=["brief", "moderate", "detailed"],
            default="moderate"
        ),
    ]

    dimensions = base_space.dimensions + tech_dimensions
    return SearchSpace(dimensions)


def create_sales_assistant_search_space() -> SearchSpace:
    """Search space optimized for sales assistant agents."""
    base_space = create_grpo_search_space(
        include_value_function=True,
        include_kl_penalty=True
    )

    # Add sales specific parameters
    sales_dimensions = [
        SearchDimension(
            "engagement_weight",
            SearchSpaceType.UNIFORM,
            low=0.3,
            high=0.5,
            default=0.4
        ),
        SearchDimension(
            "persuasiveness_weight",
            SearchSpaceType.UNIFORM,
            low=0.2,
            high=0.4,
            default=0.3
        ),
        SearchDimension(
            "task_completion_weight",
            SearchSpaceType.UNIFORM,
            low=0.2,
            high=0.4,
            default=0.3
        ),
    ]

    dimensions = base_space.dimensions + sales_dimensions
    return SearchSpace(dimensions)


# ============================================================================
# Training Profile Search Spaces
# ============================================================================

def create_conservative_search_space() -> SearchSpace:
    """Conservative search space with narrow ranges."""
    return SearchSpace([
        SearchDimension(
            "learning_rate",
            SearchSpaceType.LOGUNIFORM,
            low=1e-6,
            high=1e-5,
            default=5e-6
        ),
        SearchDimension(
            "kl_penalty_coef",
            SearchSpaceType.UNIFORM,
            low=0.05,
            high=0.15,
            default=0.1
        ),
        SearchDimension(
            "gamma",
            SearchSpaceType.UNIFORM,
            low=0.95,
            high=0.99,
            default=0.97
        ),
        SearchDimension(
            "max_grad_norm",
            SearchSpaceType.UNIFORM,
            low=0.8,
            high=1.5,
            default=1.0
        ),
    ])


def create_aggressive_search_space() -> SearchSpace:
    """Aggressive search space with wide ranges."""
    return SearchSpace([
        SearchDimension(
            "learning_rate",
            SearchSpaceType.LOGUNIFORM,
            low=5e-6,
            high=1e-3,
            default=1e-4
        ),
        SearchDimension(
            "kl_penalty_coef",
            SearchSpaceType.LOGUNIFORM,
            low=0.001,
            high=0.05,
            default=0.01
        ),
        SearchDimension(
            "gamma",
            SearchSpaceType.UNIFORM,
            low=0.9,
            high=0.999,
            default=0.99
        ),
        SearchDimension(
            "temperature",
            SearchSpaceType.UNIFORM,
            low=0.5,
            high=2.0,
            default=1.0
        ),
    ])


# ============================================================================
# Utility Functions
# ============================================================================

PREDEFINED_SPACES: Dict[str, callable] = {
    "grpo": create_grpo_search_space,
    "full": create_full_search_space,
    "optimizer": create_optimizer_search_space,
    "architecture": create_model_architecture_search_space,
    "generation": create_generation_search_space,
    "customer_service": create_customer_service_search_space,
    "technical_support": create_technical_support_search_space,
    "sales_assistant": create_sales_assistant_search_space,
    "conservative": create_conservative_search_space,
    "aggressive": create_aggressive_search_space,
}


def get_search_space(name: str, **kwargs) -> SearchSpace:
    """Get a pre-defined search space by name.

    Args:
        name: Name of the search space (e.g., "grpo", "full", "customer_service")
        **kwargs: Additional arguments to pass to the search space constructor

    Returns:
        SearchSpace instance

    Raises:
        ValueError: If search space name is not recognized
    """
    if name not in PREDEFINED_SPACES:
        available = ", ".join(PREDEFINED_SPACES.keys())
        raise ValueError(f"Unknown search space '{name}'. Available: {available}")

    creator = PREDEFINED_SPACES[name]
    return creator(**kwargs)


def list_available_search_spaces() -> List[str]:
    """List all available pre-defined search spaces."""
    return list(PREDEFINED_SPACES.keys())
