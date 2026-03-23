"""
Autonomous Research Loop for StateSet Agents.

Generalizes the autoresearch pattern into a first-class platform feature:
an outer loop that proposes experiments, trains with a time budget, evaluates
on held-out scenarios, and keeps only improvements.

Quick Start:
    >>> from stateset_agents.training.auto_research import (
    ...     AutoResearchConfig,
    ...     AutoResearchLoop,
    ...     run_auto_research,
    ... )
    >>>
    >>> config = AutoResearchConfig(time_budget=300, max_experiments=50)
    >>> summary = await run_auto_research(
    ...     agent=agent,
    ...     environment=env,
    ...     eval_scenarios=eval_scenarios,
    ...     reward_fn=reward_fn,
    ...     config=config,
    ... )

Proposer Strategies:
    - random: Random sampling from a search space
    - grid: Systematic grid search
    - bayesian: Optuna-backed Bayesian optimization (requires optuna)
    - perturbation: Small random perturbations of the current best config
    - llm: LLM-driven proposals using Claude or OpenAI (requires anthropic or openai)

Search Spaces:
    - auto_research: Comprehensive (LR, LoRA, GSPO, RL, generation params)
    - quick: Just the 4 most impactful hyperparameters
    - reward: Reward weight exploration
    - model: Model architecture (LoRA, quantization, generation)
    - grpo: Core GRPO hyperparameters (from HPO module)

Features:
    - Resume from a previous run (reads experiments.jsonl)
    - W&B experiment logging (optional)
    - Filesystem-based model checkpointing
    - Graceful shutdown via SIGINT/SIGTERM
    - Early abort on NaN loss / loss explosion / plateau
    - ASCII convergence chart in analysis report
    - Parameter importance analysis (numeric + categorical)

Post-Run Analysis:
    >>> tracker = ExperimentTracker.load("./auto_research_results")
    >>> tracker.print_summary()
    >>> analysis = tracker.get_analysis()

    Compare multiple runs:
    >>> from stateset_agents.training.auto_research import compare_runs
    >>> print(compare_runs("./run_perturbation", "./run_smart"))

Configuration from file:
    >>> config = AutoResearchConfig.from_file("config.yaml")
"""

from .checkpoint_manager import CheckpointManager
from .config import AutoResearchConfig
from .experiment_loop import AutoResearchLoop, run_auto_research
from .experiment_tracker import ExperimentRecord, ExperimentTracker
from .analysis import (
    compare_runs,
    compute_convergence_curve,
    compute_diminishing_returns,
    compute_parameter_importance,
    generate_report,
)
from .early_abort import EarlyAbortCallback
from .proposer import (
    AdaptivePerturbationProposer,
    BayesianProposer,
    ExperimentProposer,
    GridProposer,
    PerturbationProposer,
    RandomProposer,
    SmartPerturbationProposer,
    create_proposer,
)
from .search_spaces import (
    AUTO_RESEARCH_SPACES,
    create_auto_research_search_space,
    create_model_search_space,
    create_multi_algorithm_search_space,
    create_quick_search_space,
    create_reward_search_space,
    get_auto_research_search_space,
    list_auto_research_search_spaces,
    validate_params_against_space,
)

# LLM proposer is optional (requires anthropic or openai)
try:
    from .llm_proposer import LLMProposer

    LLM_PROPOSER_AVAILABLE = True
except ImportError:
    LLM_PROPOSER_AVAILABLE = False

__all__ = [
    "AutoResearchConfig",
    "AutoResearchLoop",
    "CheckpointManager",
    "ExperimentProposer",
    "ExperimentRecord",
    "ExperimentTracker",
    "RandomProposer",
    "GridProposer",
    "BayesianProposer",
    "PerturbationProposer",
    "AdaptivePerturbationProposer",
    "SmartPerturbationProposer",
    "EarlyAbortCallback",
    # Analysis
    "compute_parameter_importance",
    "compute_convergence_curve",
    "compute_diminishing_returns",
    "generate_report",
    "compare_runs",
    "LLM_PROPOSER_AVAILABLE",
    "create_proposer",
    "run_auto_research",
    # Search spaces
    "AUTO_RESEARCH_SPACES",
    "create_auto_research_search_space",
    "create_multi_algorithm_search_space",
    "create_quick_search_space",
    "create_reward_search_space",
    "create_model_search_space",
    "get_auto_research_search_space",
    "list_auto_research_search_spaces",
    "validate_params_against_space",
]

if LLM_PROPOSER_AVAILABLE:
    __all__.append("LLMProposer")
