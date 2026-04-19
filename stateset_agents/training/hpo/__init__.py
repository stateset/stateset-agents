"""
Hyperparameter Optimization (HPO) for StateSet Agents.

This module provides comprehensive HPO capabilities for GRPO training:

- **Multiple Backends**: Optuna, Ray Tune, W&B Sweeps
- **Pre-defined Search Spaces**: Domain-specific and algorithm-specific
- **Easy Integration**: Works seamlessly with existing GRPO trainers
- **Rich Analysis**: Visualization, importance analysis, optimization history

Quick Start:
    >>> from stateset_agents.training.hpo import GRPOHPOTrainer, HPOConfig, quick_hpo
    >>> from stateset_agents.training.hpo.search_spaces import create_grpo_search_space
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

from typing import Any

from .base import (
    HPOBackend,
    HPOCallback,
    HPOResult,
    HPOSummary,
    SearchDimension,
    SearchSpace,
    SearchSpaceType,
)
from .config import (
    AGGRESSIVE_HPO_CONFIG,
    CONSERVATIVE_HPO_CONFIG,
    DISTRIBUTED_HPO_CONFIG,
    QUICK_HPO_CONFIG,
    HPOConfig,
    get_hpo_config,
)
from .grpo_hpo_trainer import GRPOHPOTrainer, quick_hpo
from .search_spaces import (
    create_aggressive_search_space,
    create_conservative_search_space,
    create_customer_service_search_space,
    create_full_search_space,
    create_generation_search_space,
    create_grpo_search_space,
    create_model_architecture_search_space,
    create_optimizer_search_space,
    create_sales_assistant_search_space,
    create_technical_support_search_space,
    get_search_space,
    list_available_search_spaces,
)

# Conditionally import backends based on availability
OptunaBackend: Any | None = None
RayTuneBackend: Any | None = None
WandBSweepsBackend: Any | None = None

try:
    from .optuna_backend import OPTUNA_AVAILABLE as __optuna_available__
    from .optuna_backend import OptunaBackend as _OptunaBackend
    OptunaBackend = _OptunaBackend
except ImportError:
    __optuna_available__ = False

try:
    from .ray_tune_backend import RAY_AVAILABLE as __ray_tune_available__
    from .ray_tune_backend import RayTuneBackend as _RayTuneBackend
    RayTuneBackend = _RayTuneBackend
except ImportError:  # pragma: no cover
    __ray_tune_available__ = False
    RayTuneBackend = None

try:
    from .wandb_backend import WANDB_AVAILABLE as __wandb_available__
    from .wandb_backend import WandBSweepsBackend as _WandBSweepsBackend
    WandBSweepsBackend = _WandBSweepsBackend
except ImportError:  # pragma: no cover
    __wandb_available__ = False
    WandBSweepsBackend = None


_exports = [
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
__all__: list[str] = [name for name in _exports if name is not None]

# Remove None values from __all__
__all__ = [x for x in __all__ if x is not None]
