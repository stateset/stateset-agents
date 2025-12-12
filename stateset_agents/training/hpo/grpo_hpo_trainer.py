"""
GRPO Trainer with integrated hyperparameter optimization.

This module provides a wrapper around the standard GRPO trainer
that adds automatic hyperparameter optimization capabilities.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json

from training.config import TrainingConfig
from training.trainer import MultiTurnGRPOTrainer
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import Environment
from stateset_agents.core.reward import RewardFunction

from .base import HPOBackend, HPOCallback, HPOResult, HPOSummary, SearchSpace
from .config import HPOConfig
from .search_spaces import get_search_space
from .optuna_backend import OptunaBackend
from .ray_tune_backend import RayTuneBackend
from .wandb_backend import WandBSweepsBackend

logger = logging.getLogger(__name__)


class GRPOHPOTrainer:
    """GRPO Trainer with automatic hyperparameter optimization.

    This class wraps the standard MultiTurnGRPOTrainer and adds HPO capabilities.
    It can:
    - Automatically search for optimal hyperparameters
    - Train with best parameters after HPO
    - Save and load HPO results
    - Provide rich analysis and visualization

    Example:
        >>> from training.hpo import GRPOHPOTrainer, HPOConfig
        >>> from training.hpo.search_spaces import create_grpo_search_space
        >>>
        >>> # Setup
        >>> agent = MultiTurnAgent(agent_config)
        >>> environment = ConversationEnvironment(scenarios)
        >>> reward_fn = CompositeReward([...])
        >>>
        >>> # Configure HPO
        >>> hpo_config = HPOConfig(
        ...     backend="optuna",
        ...     search_space=create_grpo_search_space(),
        ...     n_trials=50,
        ...     objective_metric="reward"
        ... )
        >>>
        >>> # Run HPO
        >>> hpo_trainer = GRPOHPOTrainer(
        ...     agent=agent,
        ...     environment=environment,
        ...     reward_function=reward_fn,
        ...     base_config=base_training_config,
        ...     hpo_config=hpo_config
        ... )
        >>>
        >>> summary = await hpo_trainer.optimize()
        >>> print(f"Best params: {summary.best_params}")
        >>>
        >>> # Train with best params
        >>> final_agent = await hpo_trainer.train_with_best_params()
    """

    def __init__(
        self,
        agent: MultiTurnAgent,
        environment: Environment,
        reward_function: RewardFunction,
        base_config: TrainingConfig,
        hpo_config: HPOConfig,
        callbacks: Optional[List[HPOCallback]] = None,
    ):
        """Initialize GRPO HPO Trainer.

        Args:
            agent: Agent to train
            environment: Training environment
            reward_function: Reward function
            base_config: Base training configuration (will be overridden by HPO)
            hpo_config: HPO configuration
            callbacks: Optional HPO callbacks
        """
        self.agent = agent
        self.environment = environment
        self.reward_function = reward_function
        self.base_config = base_config
        self.hpo_config = hpo_config
        self.callbacks = callbacks or []

        # Get or create search space
        if hpo_config.search_space is None:
            if hpo_config.search_space_name is None:
                raise ValueError("Must provide search_space or search_space_name")
            self.search_space = get_search_space(hpo_config.search_space_name)
        else:
            self.search_space = hpo_config.search_space

        # Create HPO backend
        self.backend = self._create_backend()

        # Track results
        self.hpo_summary: Optional[HPOSummary] = None
        self.best_agent: Optional[MultiTurnAgent] = None

    def _create_backend(self) -> HPOBackend:
        """Create the HPO backend based on configuration."""
        backend_config = self.hpo_config.get_backend_config()

        if self.hpo_config.backend == "optuna":
            return OptunaBackend(
                search_space=self.search_space,
                objective_metric=self.hpo_config.objective_metric,
                direction=self.hpo_config.direction,
                callbacks=self.callbacks,
                study_name=self.hpo_config.study_name,
                **backend_config
            )
        elif self.hpo_config.backend == "ray_tune":
            return RayTuneBackend(
                search_space=self.search_space,
                objective_metric=self.hpo_config.objective_metric,
                direction=self.hpo_config.direction,
                callbacks=self.callbacks,
                output_dir=self.hpo_config.output_dir,
                study_name=self.hpo_config.study_name,
                scheduler=backend_config.get("scheduler", "asha"),
                search_alg=backend_config.get("search_alg", "bayesopt"),
                max_concurrent=backend_config.get("max_concurrent", self.hpo_config.n_parallel_trials),
                cpu_per_trial=self.hpo_config.cpu_per_trial,
                gpu_per_trial=self.hpo_config.gpu_per_trial,
                ray_init_kwargs=backend_config.get("ray_init_kwargs"),
            )
        elif self.hpo_config.backend == "wandb":
            return WandBSweepsBackend(
                search_space=self.search_space,
                objective_metric=self.hpo_config.objective_metric,
                direction=self.hpo_config.direction,
                callbacks=self.callbacks,
                output_dir=self.hpo_config.output_dir,
                method=backend_config.get("method", "bayes"),
                project=backend_config.get("project", "stateset-hpo"),
                entity=backend_config.get("entity"),
                sweep_name=self.hpo_config.study_name,
            )
        else:
            raise ValueError(f"Unknown backend: {self.hpo_config.backend}")

    def _create_training_config(self, params: Dict[str, Any]) -> TrainingConfig:
        """Create a training config by applying HPO params to base config.

        Args:
            params: Hyperparameters suggested by HPO

        Returns:
            TrainingConfig with HPO params applied
        """
        # Start with base config
        config_dict = {
            k: v for k, v in self.base_config.__dict__.items()
            if not k.startswith('_')
        }

        # Override with HPO params
        for param_name, param_value in params.items():
            if hasattr(self.base_config, param_name):
                config_dict[param_name] = param_value
            else:
                logger.warning(f"Parameter {param_name} not found in TrainingConfig")

        # Create new config
        return TrainingConfig(**config_dict)

    async def _objective_function(self, params: Dict[str, Any]) -> float:
        """Objective function for HPO.

        This trains the agent with given parameters and returns the metric.

        Args:
            params: Hyperparameters to evaluate

        Returns:
            Metric value (reward, loss, etc.)
        """
        # Create training config with HPO params
        training_config = self._create_training_config(params)

        # Create trainer
        trainer = MultiTurnGRPOTrainer(
            agent=self.agent,
            environment=self.environment,
            reward_function=self.reward_function,
            config=training_config
        )

        # Train (use a subset of episodes for faster HPO)
        # For HPO, we typically use fewer episodes to speed up search
        hpo_episodes = min(training_config.num_episodes, 50)

        logger.info(f"Training with params: {params}")
        logger.info(f"Running {hpo_episodes} episodes for HPO trial")

        try:
            metrics = await trainer.train(num_episodes=hpo_episodes)

            # Extract objective metric
            metric_value = metrics.get(self.hpo_config.objective_metric, 0.0)

            logger.info(f"Trial completed: {self.hpo_config.objective_metric} = {metric_value:.4f}")

            return metric_value

        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Return worst possible value
            if self.hpo_config.direction == "maximize":
                return float('-inf')
            else:
                return float('inf')

    async def optimize(self) -> HPOSummary:
        """Run hyperparameter optimization.

        Returns:
            HPOSummary with best parameters and full results
        """
        logger.info("Starting hyperparameter optimization")
        logger.info(f"Backend: {self.hpo_config.backend}")
        logger.info(f"Search space: {len(self.search_space.dimensions)} dimensions")
        logger.info(f"Trials: {self.hpo_config.n_trials}")

        # Run optimization
        self.hpo_summary = await self.backend.optimize(
            objective_fn=self._objective_function,
            n_trials=self.hpo_config.n_trials,
            timeout=self.hpo_config.timeout
        )

        # Save results
        self._save_results()

        # Print summary
        self.hpo_summary.print_summary()

        return self.hpo_summary

    async def train_with_best_params(
        self,
        full_episodes: Optional[int] = None
    ) -> MultiTurnAgent:
        """Train agent using the best parameters found by HPO.

        Args:
            full_episodes: Number of episodes for full training
                          (defaults to base_config.num_episodes)

        Returns:
            Trained agent

        Raises:
            ValueError: If optimize() hasn't been called yet
        """
        if self.hpo_summary is None:
            raise ValueError("Must call optimize() before train_with_best_params()")

        logger.info("Training with best hyperparameters")
        logger.info(f"Best params: {self.hpo_summary.best_params}")

        # Create config with best params
        training_config = self._create_training_config(self.hpo_summary.best_params)

        # Override num_episodes if specified
        if full_episodes is not None:
            training_config.num_episodes = full_episodes

        # Create trainer
        trainer = MultiTurnGRPOTrainer(
            agent=self.agent,
            environment=self.environment,
            reward_function=self.reward_function,
            config=training_config
        )

        # Train
        logger.info(f"Running full training: {training_config.num_episodes} episodes")
        await trainer.train()

        self.best_agent = trainer.agent

        # Save best agent
        best_agent_path = self.hpo_config.output_dir / "best_agent"
        self.best_agent.save(best_agent_path)
        logger.info(f"Best agent saved to {best_agent_path}")

        return self.best_agent

    def _save_results(self):
        """Save HPO results to disk."""
        if self.hpo_summary is None:
            return

        # Save summary
        summary_path = self.hpo_config.output_dir / "hpo_summary.json"
        self.hpo_summary.save(summary_path)
        logger.info(f"HPO summary saved to {summary_path}")

        # Save search space
        search_space_path = self.hpo_config.output_dir / "search_space.json"
        with open(search_space_path, 'w') as f:
            json.dump(self.search_space.to_dict(), f, indent=2)

        # Save config
        config_path = self.hpo_config.output_dir / "hpo_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.hpo_config.to_dict(), f, indent=2)

        # Save individual trial results
        trials_dir = self.hpo_config.output_dir / "trials"
        trials_dir.mkdir(exist_ok=True)
        for result in self.hpo_summary.all_results:
            result_path = trials_dir / f"trial_{result.trial_id}.json"
            result.save(result_path)

        logger.info(f"All results saved to {self.hpo_config.output_dir}")

    def load_results(self, output_dir: Path) -> HPOSummary:
        """Load HPO results from disk.

        Args:
            output_dir: Directory containing saved results

        Returns:
            Loaded HPOSummary
        """
        summary_path = output_dir / "hpo_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"No results found at {summary_path}")

        with open(summary_path) as f:
            data = json.load(f)

        # Reconstruct HPOSummary (simplified)
        self.hpo_summary = HPOSummary(
            best_params=data["best_params"],
            best_metric=data["best_metric"],
            best_trial_id=data["best_trial_id"],
            total_trials=data["total_trials"],
            successful_trials=data["successful_trials"],
            failed_trials=data.get("failed_trials", 0),
            pruned_trials=data.get("pruned_trials", 0),
            total_time=data.get("total_time", 0.0),
        )

        return self.hpo_summary

    def get_best_params(self) -> Dict[str, Any]:
        """Get the best hyperparameters found.

        Returns:
            Best parameters dictionary

        Raises:
            ValueError: If optimize() hasn't been called yet
        """
        if self.hpo_summary is None:
            raise ValueError("Must call optimize() before getting best params")

        return self.hpo_summary.best_params

    def plot_results(self, save_dir: Optional[Path] = None):
        """Generate and save HPO visualization plots.

        Args:
            save_dir: Directory to save plots (defaults to hpo_config.output_dir)
        """
        if self.hpo_summary is None:
            raise ValueError("Must call optimize() before plotting")

        save_dir = save_dir or self.hpo_config.output_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(self.backend, OptunaBackend):
            # Generate Optuna plots
            try:
                self.backend.plot_optimization_history(
                    save_path=save_dir / "optimization_history.png"
                )
                self.backend.plot_param_importances(
                    save_path=save_dir / "param_importances.png"
                )
                self.backend.plot_parallel_coordinate(
                    save_path=save_dir / "parallel_coordinate.png"
                )
                logger.info(f"Plots saved to {save_dir}")
            except Exception as e:
                logger.warning(f"Could not generate plots: {e}")
        else:
            logger.warning(f"Plotting not yet implemented for {self.hpo_config.backend}")


# Convenience function for quick HPO

async def quick_hpo(
    agent: MultiTurnAgent,
    environment: Environment,
    reward_function: RewardFunction,
    base_config: TrainingConfig,
    n_trials: int = 20,
    search_space_name: str = "grpo",
    output_dir: Path = Path("./quick_hpo")
) -> HPOSummary:
    """Quick HPO with sensible defaults.

    Args:
        agent: Agent to optimize
        environment: Training environment
        reward_function: Reward function
        base_config: Base training configuration
        n_trials: Number of HPO trials
        search_space_name: Name of predefined search space
        output_dir: Output directory

    Returns:
        HPOSummary with results

    Example:
        >>> summary = await quick_hpo(
        ...     agent=agent,
        ...     environment=env,
        ...     reward_function=reward_fn,
        ...     base_config=config,
        ...     n_trials=30
        ... )
        >>> print(summary.best_params)
    """
    hpo_config = HPOConfig(
        backend="optuna",
        search_space_name=search_space_name,
        n_trials=n_trials,
        output_dir=output_dir,
        optuna_config={
            "sampler": "tpe",
            "pruner": "median",
        }
    )

    trainer = GRPOHPOTrainer(
        agent=agent,
        environment=environment,
        reward_function=reward_fn,
        base_config=base_config,
        hpo_config=hpo_config
    )

    return await trainer.optimize()
