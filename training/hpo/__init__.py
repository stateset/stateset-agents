"""
Hyperparameter Optimization (HPO) for StateSet Agents.

This module provides comprehensive HPO capabilities for GRPO training:

- **Multiple Backends**: Optuna, Ray Tune, W&B Sweeps
- **Pre-defined Search Spaces**: Domain-specific and algorithm-specific
- **Easy Integration**: Works seamlessly with existing GRPO trainers
- **Rich Analysis**: Visualization, importance analysis, optimization history

Quick Start:
    >>> from training.hpo import GRPOHPOTrainer, HPOConfig, quick_hpo
    >>> from training.hpo.search_spaces import create_grpo_search_space
    >>>
    >>> # Quick HPO with defaults
    >>> summary = await quick_hpo(
    ...     agent=agent,
    ...     environment=env,
    ...     reward_function=reward_fn,
    ...     base_config=config,
    ...     n_trials=30
    ... )
    >>>
    >>> # Or configure manually
    >>> hpo_config = HPOConfig(
    ...     backend="optuna",
    ...     search_space=create_grpo_search_space(),
    ...     n_trials=100
    ... )
    >>> trainer = GRPOHPOTrainer(..., hpo_config=hpo_config)
    >>> summary = await trainer.optimize()
    >>> best_agent = await trainer.train_with_best_params()

Available Search Spaces:
    - grpo: Core GRPO hyperparameters
    - full: Comprehensive search (GRPO + optimizer + architecture + generation)
    - customer_service: Optimized for customer service agents
    - technical_support: Optimized for technical support
    - sales_assistant: Optimized for sales agents
    - conservative: Narrow search ranges
    - aggressive: Wide search ranges

Available Backends:
    - optuna: Tree-structured Parzen Estimator (TPE), best for most cases
    - ray_tune: Distributed optimization, good for large-scale searches
    - wandb: W&B Sweeps integration, good for team collaboration
"""

from .base import (
    HPOBackend,
    HPOCallback,
    HPOResult,
    HPOSummary,
    SearchSpace,
    SearchDimension,
    SearchSpaceType,
)

from .config import (
    HPOConfig,
    get_hpo_config,
    CONSERVATIVE_HPO_CONFIG,
    AGGRESSIVE_HPO_CONFIG,
    QUICK_HPO_CONFIG,
    DISTRIBUTED_HPO_CONFIG,
)

from .search_spaces import (
    create_grpo_search_space,
    create_full_search_space,
    create_optimizer_search_space,
    create_model_architecture_search_space,
    create_generation_search_space,
    create_customer_service_search_space,
    create_technical_support_search_space,
    create_sales_assistant_search_space,
    create_conservative_search_space,
    create_aggressive_search_space,
    get_search_space,
    list_available_search_spaces,
)

from .grpo_hpo_trainer import (
    GRPOHPOTrainer,
    quick_hpo,
)

# Conditionally import backends based on availability
try:
    from .optuna_backend import OptunaBackend
    __optuna_available__ = True
except ImportError:
    __optuna_available__ = False

try:
    from .ray_tune_backend import RayTuneBackend, RAY_AVAILABLE as __ray_tune_available__
except ImportError:  # pragma: no cover
    __ray_tune_available__ = False
    RayTuneBackend = None  # type: ignore

try:
    from .wandb_backend import WandBSweepsBackend, WANDB_AVAILABLE as __wandb_available__
except ImportError:  # pragma: no cover
    __wandb_available__ = False
    WandBSweepsBackend = None  # type: ignore


__all__ = [
    # Base classes
    "HPOBackend",
    "HPOCallback",
    "HPOResult",
    "HPOSummary",
    "SearchSpace",
    "SearchDimension",
    "SearchSpaceType",

    # Configuration
    "HPOConfig",
    "get_hpo_config",
    "CONSERVATIVE_HPO_CONFIG",
    "AGGRESSIVE_HPO_CONFIG",
    "QUICK_HPO_CONFIG",
    "DISTRIBUTED_HPO_CONFIG",

    # Search spaces
    "create_grpo_search_space",
    "create_full_search_space",
    "create_optimizer_search_space",
    "create_model_architecture_search_space",
    "create_generation_search_space",
    "create_customer_service_search_space",
    "create_technical_support_search_space",
    "create_sales_assistant_search_space",
    "create_conservative_search_space",
    "create_aggressive_search_space",
    "get_search_space",
    "list_available_search_spaces",

    # Trainer
    "GRPOHPOTrainer",
    "quick_hpo",

    # Backends (conditionally)
    "OptunaBackend" if __optuna_available__ else None,
    "RayTuneBackend" if __ray_tune_available__ else None,
    "WandBSweepsBackend" if __wandb_available__ else None,

    # Availability flags
    "__optuna_available__",
    "__ray_tune_available__",
    "__wandb_available__",
]

# Remove None values from __all__
__all__ = [x for x in __all__ if x is not None]
