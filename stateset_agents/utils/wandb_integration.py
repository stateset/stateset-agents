"""
Enhanced Weights & Biases Integration for GRPO Agent Framework

This module provides comprehensive experiment tracking, visualization,
and analysis capabilities for GRPO training with rich metrics and insights.
"""

import asyncio
import base64
import io
import importlib.util
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

plt = None  # type: ignore[assignment]
sns = None  # type: ignore[assignment]
wandb = None

WANDB_INSTALLED = importlib.util.find_spec("wandb") is not None
MATPLOTLIB_INSTALLED = importlib.util.find_spec("matplotlib") is not None
SEABORN_INSTALLED = importlib.util.find_spec("seaborn") is not None

# Backwards-compat flag; indicates whether wandb can be imported.
WANDB_AVAILABLE = WANDB_INSTALLED


def _load_wandb() -> bool:
    """Import wandb on-demand to avoid slow import at module load time."""
    global WANDB_AVAILABLE, wandb
    if wandb is not None:
        return True
    try:
        import wandb as _wandb  # type: ignore

        wandb = _wandb
        WANDB_AVAILABLE = True
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        wandb = None
        WANDB_AVAILABLE = False
        logging.getLogger(__name__).warning("Failed to import wandb: %s", exc)
        return False


def _load_plotting() -> bool:
    """Import matplotlib/seaborn on-demand."""
    global plt, sns
    if plt is not None and sns is not None:
        return True
    try:
        if plt is None:
            import matplotlib.pyplot as _plt  # type: ignore

            plt = _plt
        if sns is None:
            import seaborn as _sns  # type: ignore

            sns = _sns
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        plt = None  # type: ignore[assignment]
        sns = None  # type: ignore[assignment]
        logging.getLogger(__name__).warning(
            "Failed to import matplotlib/seaborn: %s", exc
        )
        return False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from stateset_agents.core import trajectory as core_trajectory

from .monitoring import MonitoringService

MultiTurnTrajectory = core_trajectory.MultiTurnTrajectory
TrajectoryGroup = core_trajectory.TrajectoryGroup
Trajectory = core_trajectory.Trajectory

from stateset_agents.core import agent as core_agent

AgentConfig = core_agent.AgentConfig

from stateset_agents.core import reward as core_reward

RewardResult = core_reward.RewardResult

TrainingConfig = None  # Lazy import
TrainingConfig = None

logger = logging.getLogger(__name__)


@dataclass
class WandBConfig:
    """Configuration for Weights & Biases integration"""

    # Project settings
    project: str = "grpo-agent-framework"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    # Logging settings
    log_frequency: int = 10  # Log every N steps
    log_trajectories: bool = True
    log_rewards: bool = True
    log_model_gradients: bool = True
    log_model_parameters: bool = False
    log_system_metrics: bool = True

    # Visualization settings
    create_reward_plots: bool = True
    create_trajectory_plots: bool = True
    create_loss_plots: bool = True
    plot_frequency: int = 100  # Create plots every N steps

    # Advanced features
    log_code: bool = True
    log_model_architecture: bool = True
    save_model_artifacts: bool = True
    watch_model: bool = True

    # Storage settings
    offline_mode: bool = False
    save_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project": self.project,
            "entity": self.entity,
            "name": self.name,
            "tags": self.tags,
            "notes": self.notes,
            "log_frequency": self.log_frequency,
            "log_trajectories": self.log_trajectories,
            "log_rewards": self.log_rewards,
            "log_model_gradients": self.log_model_gradients,
            "log_model_parameters": self.log_model_parameters,
            "log_system_metrics": self.log_system_metrics,
            "create_reward_plots": self.create_reward_plots,
            "create_trajectory_plots": self.create_trajectory_plots,
            "create_loss_plots": self.create_loss_plots,
            "plot_frequency": self.plot_frequency,
            "log_code": self.log_code,
            "log_model_architecture": self.log_model_architecture,
            "save_model_artifacts": self.save_model_artifacts,
            "watch_model": self.watch_model,
            "offline_mode": self.offline_mode,
            "save_dir": self.save_dir,
        }


class WandBIntegration:
    """
    Enhanced Weights & Biases integration for GRPO training
    """

    def __init__(
        self,
        config: WandBConfig,
        training_config: Optional[TrainingConfig] = None,
        monitoring_service: Optional[MonitoringService] = None,
    ):
        if not (WANDB_INSTALLED and MATPLOTLIB_INSTALLED and SEABORN_INSTALLED):
            raise ImportError(
                "wandb and matplotlib are required. Install with: pip install wandb matplotlib seaborn"
            )
        if not (_load_wandb() and _load_plotting()):
            raise ImportError(
                "wandb and matplotlib are required. Install with: pip install wandb matplotlib seaborn"
            )

        self.config = config
        self.training_config = training_config
        self.monitoring = monitoring_service

        # State tracking
        self.is_initialized = False
        self.run = None
        self.step_count = 0
        self.epoch_count = 0

        # Data collection
        self.trajectory_history = []
        self.reward_history = []
        self.loss_history = []
        self.gradient_history = []

        # Metrics aggregation
        self.metrics_buffer = {}
        self.last_log_step = 0

        # Visualization
        self.plot_cache = {}

    def initialize(
        self, model: Optional[Any] = None, optimizer: Optional[Any] = None, **kwargs
    ):
        """Initialize Weights & Biases run"""
        if self.is_initialized:
            logger.warning("WandB already initialized")
            return

        # Setup run configuration
        run_config = {
            "framework": "grpo-agent-framework",
            "timestamp": datetime.now().isoformat(),
            **self.config.to_dict(),
        }

        if self.training_config:
            run_config.update(self.training_config.to_dict())

        run_config.update(kwargs)

        # Initialize W&B run
        self.run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=self.config.name,
            tags=self.config.tags,
            notes=self.config.notes,
            config=run_config,
            mode="offline" if self.config.offline_mode else "online",
            dir=self.config.save_dir,
        )

        # Watch model if provided
        if model and self.config.watch_model:
            wandb.watch(
                model,
                log="all" if self.config.log_model_parameters else "gradients",
                log_freq=self.config.log_frequency,
            )

        # Log model architecture
        if model and self.config.log_model_architecture:
            self._log_model_architecture(model)

        # Log code if enabled
        if self.config.log_code:
            self._log_code()

        self.is_initialized = True
        logger.info(f"Initialized WandB run: {self.run.id}")

    def log_step(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        commit: bool = True,
    ):
        """Log metrics for a training step"""
        if not self.is_initialized:
            logger.warning("WandB not initialized")
            return

        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1

        if epoch is not None:
            self.epoch_count = epoch

        # Add step and epoch to metrics
        log_metrics = {"step": self.step_count, "epoch": self.epoch_count, **metrics}

        # Buffer metrics for aggregation
        for key, value in metrics.items():
            if key not in self.metrics_buffer:
                self.metrics_buffer[key] = []
            self.metrics_buffer[key].append(value)

        # Log to W&B
        if commit or self.step_count % self.config.log_frequency == 0:
            wandb.log(log_metrics, step=self.step_count, commit=commit)
            self.last_log_step = self.step_count

    def log_trajectories(
        self,
        trajectories: List[Trajectory],
        rewards: List[RewardResult],
        step: Optional[int] = None,
    ):
        """Log trajectory data with rich analysis"""
        if not self.is_initialized or not self.config.log_trajectories:
            return

        # Store trajectory history
        trajectory_data = []
        for i, (traj, reward) in enumerate(zip(trajectories, rewards)):
            data = {
                "trajectory_id": i,
                "step": step or self.step_count,
                "prompt": traj.get_prompt(),
                "response": traj.get_last_response(),
                "reward_score": reward.score,
                "reward_breakdown": reward.breakdown,
                "length": len(traj.turns),
                "timestamp": datetime.now().isoformat(),
            }
            trajectory_data.append(data)

        self.trajectory_history.extend(trajectory_data)

        # Log trajectory metrics
        trajectory_metrics = {
            "trajectories/count": len(trajectories),
            "trajectories/avg_length": np.mean([len(t.turns) for t in trajectories]),
            "trajectories/avg_reward": np.mean([r.score for r in rewards]),
            "trajectories/reward_std": np.std([r.score for r in rewards]),
            "trajectories/min_reward": np.min([r.score for r in rewards]),
            "trajectories/max_reward": np.max([r.score for r in rewards]),
        }

        self.log_step(trajectory_metrics, step=step, commit=False)

        # Create trajectory table
        if len(trajectory_data) > 0:
            table = wandb.Table(
                columns=["trajectory_id", "prompt", "response", "reward", "length"],
                data=[
                    [
                        d["trajectory_id"],
                        d["prompt"][:100],
                        d["response"][:100],
                        d["reward_score"],
                        d["length"],
                    ]
                    for d in trajectory_data
                ],
            )
            wandb.log({"trajectories/table": table}, step=step or self.step_count)

        # Create visualizations
        if (
            self.config.create_trajectory_plots
            and self.step_count % self.config.plot_frequency == 0
        ):
            self._create_trajectory_plots()

    def log_reward_analysis(
        self, rewards: List[RewardResult], step: Optional[int] = None
    ):
        """Log detailed reward analysis"""
        if not self.is_initialized or not self.config.log_rewards:
            return

        # Store reward history
        reward_data = []
        for i, reward in enumerate(rewards):
            data = {
                "reward_id": i,
                "step": step or self.step_count,
                "score": reward.score,
                "breakdown": reward.breakdown,
                "timestamp": datetime.now().isoformat(),
            }
            reward_data.append(data)

        self.reward_history.extend(reward_data)

        # Analyze reward components
        component_analysis = {}
        for reward in rewards:
            if hasattr(reward, "breakdown") and reward.breakdown:
                for component, value in reward.breakdown.items():
                    if isinstance(value, (int, float)):
                        if component not in component_analysis:
                            component_analysis[component] = []
                        component_analysis[component].append(value)

        # Log component metrics
        for component, values in component_analysis.items():
            if values:
                self.log_step(
                    {
                        f"rewards/{component}/mean": np.mean(values),
                        f"rewards/{component}/std": np.std(values),
                        f"rewards/{component}/min": np.min(values),
                        f"rewards/{component}/max": np.max(values),
                    },
                    step=step,
                    commit=False,
                )

        # Create reward distribution plots
        if (
            self.config.create_reward_plots
            and self.step_count % self.config.plot_frequency == 0
        ):
            self._create_reward_plots()

    def log_training_metrics(
        self,
        loss: float,
        learning_rate: float,
        gradient_norm: Optional[float] = None,
        step: Optional[int] = None,
    ):
        """Log training-specific metrics"""
        if not self.is_initialized:
            return

        # Store loss history
        loss_data = {
            "step": step or self.step_count,
            "loss": loss,
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
            "timestamp": datetime.now().isoformat(),
        }
        self.loss_history.append(loss_data)

        # Log metrics
        metrics = {"training/loss": loss, "training/learning_rate": learning_rate}

        if gradient_norm is not None:
            metrics["training/gradient_norm"] = gradient_norm

        self.log_step(metrics, step=step, commit=False)

        # Create loss plots
        if (
            self.config.create_loss_plots
            and self.step_count % self.config.plot_frequency == 0
        ):
            self._create_loss_plots()

    def log_model_metrics(self, model: Any, step: Optional[int] = None):
        """Log model-specific metrics"""
        if not self.is_initialized or not TORCH_AVAILABLE:
            return

        if not hasattr(model, "parameters"):
            return

        # Parameter statistics
        param_stats = {}
        total_params = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                param_data = param.data.detach()
                total_params += param.numel()

                param_stats[f"params/{name}/mean"] = param_data.mean().item()
                param_stats[f"params/{name}/std"] = param_data.std().item()
                param_stats[f"params/{name}/norm"] = param_data.norm().item()

                if param.grad is not None:
                    grad_data = param.grad.detach()
                    param_stats[f"gradients/{name}/mean"] = grad_data.mean().item()
                    param_stats[f"gradients/{name}/std"] = grad_data.std().item()
                    param_stats[f"gradients/{name}/norm"] = grad_data.norm().item()

        param_stats["model/total_parameters"] = total_params

        self.log_step(param_stats, step=step, commit=False)

    def log_system_metrics(
        self,
        gpu_memory: Optional[float] = None,
        cpu_usage: Optional[float] = None,
        step: Optional[int] = None,
    ):
        """Log system performance metrics"""
        if not self.is_initialized or not self.config.log_system_metrics:
            return

        system_metrics = {}

        # GPU metrics
        if TORCH_AVAILABLE and torch.cuda.is_available():
            system_metrics["system/gpu_memory_allocated"] = (
                torch.cuda.memory_allocated() / 1024**3
            )
            system_metrics["system/gpu_memory_cached"] = (
                torch.cuda.memory_reserved() / 1024**3
            )
            system_metrics["system/gpu_utilization"] = gpu_memory or 0.0

        # CPU metrics
        if cpu_usage is not None:
            system_metrics["system/cpu_usage"] = cpu_usage

        # Memory metrics
        try:
            import psutil

            memory = psutil.virtual_memory()
            system_metrics["system/memory_usage"] = memory.percent
            system_metrics["system/memory_available"] = memory.available / 1024**3
        except ImportError:
            pass

        if system_metrics:
            self.log_step(system_metrics, step=step, commit=False)

    def _log_model_architecture(self, model: Any):
        """Log model architecture information"""
        if not TORCH_AVAILABLE or not hasattr(model, "parameters"):
            return

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Log architecture info
        arch_info = {
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/model_size_mb": total_params * 4 / 1024**2,  # Assuming float32
        }

        # Try to get model summary
        try:
            model_summary = str(model)
            arch_info["model/architecture"] = model_summary
        except:
            pass

        wandb.log(arch_info)

    def _log_code(self):
        """Log code artifacts"""
        try:
            # Log current directory as code
            wandb.run.log_code(
                ".",
                include_fn=lambda path: path.endswith(
                    (".py", ".yaml", ".yml", ".json")
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to log code: {e}")

    def _create_trajectory_plots(self):
        """Create trajectory visualization plots"""
        if not self.trajectory_history:
            return

        try:
            # Reward distribution over time
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Reward over time
            steps = [t["step"] for t in self.trajectory_history]
            rewards = [t["reward_score"] for t in self.trajectory_history]

            axes[0, 0].plot(steps, rewards, alpha=0.7)
            axes[0, 0].set_title("Reward Over Time")
            axes[0, 0].set_xlabel("Step")
            axes[0, 0].set_ylabel("Reward")

            # Reward distribution
            axes[0, 1].hist(rewards, bins=50, alpha=0.7)
            axes[0, 1].set_title("Reward Distribution")
            axes[0, 1].set_xlabel("Reward")
            axes[0, 1].set_ylabel("Frequency")

            # Trajectory length over time
            lengths = [t["length"] for t in self.trajectory_history]
            axes[1, 0].plot(steps, lengths, alpha=0.7)
            axes[1, 0].set_title("Trajectory Length Over Time")
            axes[1, 0].set_xlabel("Step")
            axes[1, 0].set_ylabel("Length")

            # Length distribution
            axes[1, 1].hist(lengths, bins=20, alpha=0.7)
            axes[1, 1].set_title("Length Distribution")
            axes[1, 1].set_xlabel("Length")
            axes[1, 1].set_ylabel("Frequency")

            plt.tight_layout()
            wandb.log({"trajectories/analysis": wandb.Image(fig)})
            plt.close(fig)

        except Exception as e:
            logger.error(f"Failed to create trajectory plots: {e}")

    def _create_reward_plots(self):
        """Create reward analysis plots"""
        if not self.reward_history:
            return

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Reward trend
            steps = [r["step"] for r in self.reward_history]
            scores = [r["score"] for r in self.reward_history]

            axes[0, 0].plot(steps, scores, alpha=0.7)
            axes[0, 0].set_title("Reward Trend")
            axes[0, 0].set_xlabel("Step")
            axes[0, 0].set_ylabel("Reward Score")

            # Reward distribution
            axes[0, 1].hist(scores, bins=50, alpha=0.7)
            axes[0, 1].set_title("Reward Score Distribution")
            axes[0, 1].set_xlabel("Score")
            axes[0, 1].set_ylabel("Frequency")

            # Moving average
            window_size = min(50, len(scores) // 10)
            if window_size > 1:
                moving_avg = np.convolve(
                    scores, np.ones(window_size) / window_size, mode="valid"
                )
                axes[1, 0].plot(steps[: len(moving_avg)], moving_avg, alpha=0.7)
                axes[1, 0].set_title(f"Reward Moving Average (window={window_size})")
                axes[1, 0].set_xlabel("Step")
                axes[1, 0].set_ylabel("Average Reward")

            # Reward variance over time
            if len(scores) > 10:
                variance_window = 20
                variances = []
                var_steps = []
                for i in range(variance_window, len(scores)):
                    window = scores[i - variance_window : i]
                    variances.append(np.var(window))
                    var_steps.append(steps[i])

                axes[1, 1].plot(var_steps, variances, alpha=0.7)
                axes[1, 1].set_title("Reward Variance Over Time")
                axes[1, 1].set_xlabel("Step")
                axes[1, 1].set_ylabel("Variance")

            plt.tight_layout()
            wandb.log({"rewards/analysis": wandb.Image(fig)})
            plt.close(fig)

        except Exception as e:
            logger.error(f"Failed to create reward plots: {e}")

    def _create_loss_plots(self):
        """Create loss analysis plots"""
        if not self.loss_history:
            return

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Loss trend
            steps = [l["step"] for l in self.loss_history]
            losses = [l["loss"] for l in self.loss_history]

            axes[0, 0].plot(steps, losses, alpha=0.7)
            axes[0, 0].set_title("Training Loss")
            axes[0, 0].set_xlabel("Step")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].set_yscale("log")

            # Learning rate
            learning_rates = [l["learning_rate"] for l in self.loss_history]
            axes[0, 1].plot(steps, learning_rates, alpha=0.7)
            axes[0, 1].set_title("Learning Rate")
            axes[0, 1].set_xlabel("Step")
            axes[0, 1].set_ylabel("Learning Rate")

            # Gradient norms
            grad_norms = [
                l["gradient_norm"]
                for l in self.loss_history
                if l["gradient_norm"] is not None
            ]
            if grad_norms:
                grad_steps = [
                    l["step"]
                    for l in self.loss_history
                    if l["gradient_norm"] is not None
                ]
                axes[1, 0].plot(grad_steps, grad_norms, alpha=0.7)
                axes[1, 0].set_title("Gradient Norm")
                axes[1, 0].set_xlabel("Step")
                axes[1, 0].set_ylabel("Gradient Norm")

            # Loss histogram
            axes[1, 1].hist(losses, bins=50, alpha=0.7)
            axes[1, 1].set_title("Loss Distribution")
            axes[1, 1].set_xlabel("Loss")
            axes[1, 1].set_ylabel("Frequency")

            plt.tight_layout()
            wandb.log({"training/analysis": wandb.Image(fig)})
            plt.close(fig)

        except Exception as e:
            logger.error(f"Failed to create loss plots: {e}")

    def save_model_artifact(
        self, model: Any, name: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Save model as W&B artifact"""
        if not self.is_initialized or not self.config.save_model_artifacts:
            return

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
                if TORCH_AVAILABLE and hasattr(model, "state_dict"):
                    torch.save(model.state_dict(), tmp_file.name)
                else:
                    # Fallback for non-PyTorch models
                    import pickle

                    pickle.dump(model, tmp_file)

                # Create artifact
                artifact = wandb.Artifact(
                    name=name, type="model", metadata=metadata or {}
                )

                artifact.add_file(tmp_file.name)
                wandb.log_artifact(artifact)

                logger.info(f"Saved model artifact: {name}")

        except Exception as e:
            logger.error(f"Failed to save model artifact: {e}")

    def create_summary_report(self) -> Dict[str, Any]:
        """Create a comprehensive summary report"""
        if not self.is_initialized:
            return {}

        summary = {
            "run_id": self.run.id,
            "run_name": self.run.name,
            "total_steps": self.step_count,
            "total_epochs": self.epoch_count,
            "trajectories_logged": len(self.trajectory_history),
            "rewards_logged": len(self.reward_history),
            "losses_logged": len(self.loss_history),
        }

        # Add performance metrics
        if self.reward_history:
            rewards = [r["score"] for r in self.reward_history]
            summary["final_reward"] = rewards[-1]
            summary["best_reward"] = max(rewards)
            summary["average_reward"] = np.mean(rewards)
            summary["reward_improvement"] = (
                rewards[-1] - rewards[0] if len(rewards) > 1 else 0
            )

        if self.loss_history:
            losses = [l["loss"] for l in self.loss_history]
            summary["final_loss"] = losses[-1]
            summary["best_loss"] = min(losses)
            summary["average_loss"] = np.mean(losses)
            summary["loss_improvement"] = (
                losses[0] - losses[-1] if len(losses) > 1 else 0
            )

        # Log summary
        wandb.log({"summary": summary})

        return summary

    def finish(self):
        """Finish the W&B run"""
        if self.is_initialized:
            # Create final summary
            self.create_summary_report()

            # Finish run
            wandb.finish()

            self.is_initialized = False
            logger.info("Finished W&B run")


# Utility functions
def create_wandb_config(
    project: str, name: Optional[str] = None, tags: Optional[List[str]] = None, **kwargs
) -> WandBConfig:
    """Create a W&B configuration with sensible defaults"""
    return WandBConfig(
        project=project,
        name=name or f"grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        tags=tags or [],
        **kwargs,
    )


def setup_wandb_from_env() -> Optional[WandBConfig]:
    """Setup W&B configuration from environment variables"""
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        logger.warning("WANDB_API_KEY not found in environment")
        return None

    return WandBConfig(
        project=os.getenv("WANDB_PROJECT", "grpo-agent-framework"),
        entity=os.getenv("WANDB_ENTITY"),
        name=os.getenv("WANDB_RUN_NAME"),
        tags=os.getenv("WANDB_TAGS", "").split(",") if os.getenv("WANDB_TAGS") else [],
        notes=os.getenv("WANDB_NOTES"),
        offline_mode=os.getenv("WANDB_MODE") == "offline",
    )


class WandBMonitoringService(MonitoringService):
    """Monitoring service that logs to W&B"""

    def __init__(self, wandb_integration: WandBIntegration):
        self.wandb = wandb_integration

    async def log_metric(
        self,
        name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Log a metric to W&B"""
        if self.wandb.is_initialized:
            self.wandb.log_step({name: value}, commit=False)

    async def log_event(
        self,
        name: str,
        data: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Log an event to W&B"""
        if self.wandb.is_initialized:
            self.wandb.log_step({f"events/{name}": data}, commit=False)

    async def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Log an error to W&B"""
        if self.wandb.is_initialized:
            error_data = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {},
            }
            self.wandb.log_step({"errors/error": error_data}, commit=False)


class WandBLogger:
    """
    Comprehensive W&B logger for GRPO agent training

    Provides seamless integration with Weights & Biases for tracking:
    - Training metrics and losses
    - Model configurations
    - Conversation examples
    - Reward distributions
    - System metrics
    - Model checkpoints
    """

    def __init__(
        self,
        project: str = "grpo-agent-framework",
        entity: Optional[str] = None,
        api_key: Optional[str] = None,
        enabled: bool = True,
        **kwargs,
    ):
        self.project = project
        self.entity = entity
        self.enabled = bool(enabled)
        self.run = None
        self.start_time = None

        if self.enabled:
            if not WANDB_INSTALLED:
                logger.warning(
                    "wandb package not available. Install with: pip install wandb"
                )
                self.enabled = False
            elif not _load_wandb():
                logger.warning("wandb import failed. W&B logging will be disabled.")
                self.enabled = False

        # Handle API key
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key

        # Check if W&B is properly configured
        if self.enabled and not os.getenv("WANDB_API_KEY"):
            logger.warning("WANDB_API_KEY not found. W&B logging will be disabled.")
            self.enabled = False

        logger.info(f"W&B Logger initialized (enabled: {self.enabled})")

    def init_run(
        self,
        config: Dict[str, Any],
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        group: Optional[str] = None,
        job_type: str = "train",
    ):
        """Initialize a new W&B run"""
        if not self.enabled:
            return

        try:
            # Clean config for W&B (remove non-serializable objects)
            clean_config = self._clean_config(config)

            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=name,
                tags=tags or [],
                notes=notes,
                group=group,
                job_type=job_type,
                config=clean_config,
                reinit=True,
            )

            self.start_time = datetime.now()
            logger.info(f"W&B run initialized: {self.run.name}")

        except Exception as e:
            logger.error(f"Failed to initialize W&B run: {e}")
            self.enabled = False

    def _clean_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Clean configuration dictionary for W&B serialization"""
        clean_config = {}

        for key, value in config.items():
            try:
                # Skip non-serializable objects
                if callable(value) or hasattr(value, "__dict__"):
                    if hasattr(value, "__dict__"):
                        # Try to serialize object attributes
                        clean_config[key] = {
                            k: v
                            for k, v in value.__dict__.items()
                            if not callable(v) and not k.startswith("_")
                        }
                    else:
                        clean_config[key] = str(value)
                else:
                    clean_config[key] = value
            except Exception:
                # Skip problematic values
                continue

        return clean_config

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ):
        """Log metrics to W&B"""
        if not self.enabled or not self.run:
            return

        try:
            # Add prefix if specified
            if prefix:
                metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

            # Filter out non-numeric values for main logging
            numeric_metrics = {}
            for k, v in metrics.items():
                try:
                    if isinstance(v, (int, float, np.number)):
                        numeric_metrics[k] = float(v)
                    elif hasattr(v, "item"):  # PyTorch tensors
                        numeric_metrics[k] = float(v.item())
                    else:
                        # Log as string or handle special W&B objects
                        numeric_metrics[k] = v
                except Exception:
                    continue

            self.run.log(numeric_metrics, step=step)

        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_agent_config(self, agent_config: AgentConfig):
        """Log agent configuration details"""
        if not self.enabled:
            return

        try:
            config_dict = {
                "agent/model_name": agent_config.model_name,
                "agent/system_prompt_length": len(agent_config.system_prompt)
                if agent_config.system_prompt
                else 0,
                "agent/temperature": agent_config.temperature,
                "agent/max_new_tokens": agent_config.max_new_tokens,
                "agent/torch_dtype": agent_config.torch_dtype,
                "agent/device_map": str(agent_config.device_map),
                "agent/use_peft": agent_config.use_peft,
            }

            if agent_config.peft_config:
                config_dict.update(
                    {
                        f"agent/peft_{k}": v
                        for k, v in agent_config.peft_config.items()
                        if isinstance(v, (str, int, float, bool))
                    }
                )

            self.log_metrics(config_dict)

        except Exception as e:
            logger.error(f"Failed to log agent config: {e}")

    def log_training_step(
        self,
        losses: Dict[str, float],
        learning_rate: float,
        step: int,
        trajectory_groups: Optional[List[TrajectoryGroup]] = None,
    ):
        """Log training step metrics"""
        if not self.enabled:
            return

        try:
            # Base training metrics
            metrics = {
                "train/learning_rate": learning_rate,
                "train/step": step,
                **{f"train/{k}": v for k, v in losses.items()},
            }

            # Add trajectory group metrics if provided
            if trajectory_groups:
                traj_metrics = self._compute_trajectory_metrics(trajectory_groups)
                metrics.update({f"train/{k}": v for k, v in traj_metrics.items()})

            self.log_metrics(metrics, step=step)

        except Exception as e:
            logger.error(f"Failed to log training step: {e}")

    def log_evaluation(
        self,
        eval_metrics: Dict[str, float],
        step: int,
        trajectories: Optional[List[MultiTurnTrajectory]] = None,
    ):
        """Log evaluation metrics"""
        if not self.enabled:
            return

        try:
            # Prefix evaluation metrics
            metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}

            # Add trajectory analysis if provided
            if trajectories:
                traj_analysis = self._analyze_trajectories(trajectories)
                metrics.update(
                    {f"eval_analysis/{k}": v for k, v in traj_analysis.items()}
                )

            self.log_metrics(metrics, step=step)

        except Exception as e:
            logger.error(f"Failed to log evaluation: {e}")

    def _compute_trajectory_metrics(
        self, trajectory_groups: List[TrajectoryGroup]
    ) -> Dict[str, float]:
        """Compute metrics from trajectory groups"""
        if not trajectory_groups:
            return {}

        all_rewards = []
        group_sizes = []
        reward_diversities = []

        for group in trajectory_groups:
            if hasattr(group, "trajectories"):
                rewards = []
                for traj in group.trajectories:
                    if hasattr(traj, "total_reward"):
                        rewards.append(traj.total_reward)
                    elif hasattr(traj, "reward"):
                        rewards.append(traj.reward)

                if rewards:
                    all_rewards.extend(rewards)
                    group_sizes.append(len(rewards))
                    if len(rewards) > 1:
                        reward_diversities.append(np.std(rewards))

        if not all_rewards:
            return {}

        metrics = {
            "num_groups": len(trajectory_groups),
            "total_trajectories": len(all_rewards),
            "avg_group_size": np.mean(group_sizes) if group_sizes else 0,
            "reward_mean": np.mean(all_rewards),
            "reward_std": np.std(all_rewards),
            "reward_min": np.min(all_rewards),
            "reward_max": np.max(all_rewards),
            "reward_median": np.median(all_rewards),
        }

        if reward_diversities:
            metrics["reward_diversity_mean"] = np.mean(reward_diversities)
            metrics["reward_diversity_std"] = np.std(reward_diversities)

        return metrics

    def _analyze_trajectories(
        self, trajectories: List[MultiTurnTrajectory]
    ) -> Dict[str, float]:
        """Analyze trajectory characteristics"""
        if not trajectories:
            return {}

        episode_lengths = []
        total_rewards = []
        turn_counts = []

        for traj in trajectories:
            episode_lengths.append(traj.episode_length)
            total_rewards.append(traj.total_reward)

            # Count turns by role
            user_turns = len([t for t in traj.turns if t.role == "user"])
            assistant_turns = len([t for t in traj.turns if t.role == "assistant"])
            turn_counts.append({"user": user_turns, "assistant": assistant_turns})

        analysis = {
            "num_episodes": len(trajectories),
            "avg_episode_length": np.mean(episode_lengths),
            "episode_length_std": np.std(episode_lengths),
            "avg_total_reward": np.mean(total_rewards),
            "total_reward_std": np.std(total_rewards),
            "avg_user_turns": np.mean([tc["user"] for tc in turn_counts]),
            "avg_assistant_turns": np.mean([tc["assistant"] for tc in turn_counts]),
        }

        return analysis

    def log_model_checkpoint(
        self,
        checkpoint_path: str,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ):
        """Log model checkpoint information"""
        if not self.enabled:
            return

        try:
            checkpoint_metrics = {
                "checkpoint/step": step,
                "checkpoint/is_best": is_best,
                "checkpoint/path": checkpoint_path,
                "checkpoint/timestamp": datetime.now().isoformat(),
            }

            if metrics:
                checkpoint_metrics.update(
                    {f"checkpoint/{k}": v for k, v in metrics.items()}
                )

            self.log_metrics(checkpoint_metrics, step=step)

        except Exception as e:
            logger.error(f"Failed to log checkpoint: {e}")

    def log_conversation_examples(
        self, trajectories: List[MultiTurnTrajectory], step: int, num_examples: int = 3
    ):
        """Log conversation examples as W&B tables"""
        if not self.enabled or not trajectories:
            return

        try:
            # Select examples
            examples = trajectories[:num_examples]

            # Create table data
            table_data = []
            for i, traj in enumerate(examples):
                conversation = ""
                for turn in traj.turns:
                    conversation += f"{turn.role.capitalize()}: {turn.content}\n\n"

                table_data.append(
                    [i, traj.total_reward, traj.episode_length, conversation.strip()]
                )

            # Create W&B table
            table = wandb.Table(
                columns=["Example", "Total Reward", "Episode Length", "Conversation"],
                data=table_data,
            )

            self.log_metrics({f"examples/conversations_step_{step}": table}, step=step)

        except Exception as e:
            logger.error(f"Failed to log conversation examples: {e}")

    def log_reward_distribution(
        self, rewards: List[float], step: int, name: str = "rewards"
    ):
        """Log reward distribution histogram"""
        if not self.enabled or not rewards:
            return

        try:
            self.log_metrics(
                {
                    f"{name}/distribution": wandb.Histogram(rewards),
                    f"{name}/mean": np.mean(rewards),
                    f"{name}/std": np.std(rewards),
                },
                step=step,
            )
        except Exception as e:
            logger.error(f"Failed to log reward distribution: {e}")

    def finish_run(self, summary: Optional[Dict[str, Any]] = None):
        """Finish the W&B run and log summary"""
        if not self.enabled or not self.run:
            return

        try:
            # Add training duration to summary
            if self.start_time:
                duration = (datetime.now() - self.start_time).total_seconds()
                if summary is None:
                    summary = {}
                summary["training_duration_seconds"] = duration
                summary["training_duration_hours"] = duration / 3600

            # Log summary
            if summary:
                for key, value in summary.items():
                    self.run.summary[key] = value

            self.run.finish()
            logger.info("W&B run finished")

        except Exception as e:
            logger.error(f"Failed to finish W&B run: {e}")

    def log_system_metrics(self):
        """Log system metrics (GPU, memory, etc.)"""
        if not self.enabled:
            return

        try:
            import psutil
            import torch

            metrics = {
                "system/cpu_percent": psutil.cpu_percent(),
                "system/memory_percent": psutil.virtual_memory().percent,
            }

            # Add GPU metrics if available
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                    metrics[f"gpu_{i}/memory_allocated_gb"] = memory_allocated
                    metrics[f"gpu_{i}/memory_reserved_gb"] = memory_reserved

            self.log_metrics(metrics)

        except Exception as e:
            logger.error(f"Failed to log system metrics: {e}")


# Convenience functions
def init_wandb(
    project: str = "grpo-agent-framework",
    entity: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> WandBLogger:
    """Initialize W&B logger with convenience wrapper"""
    return WandBLogger(project=project, entity=entity, api_key=api_key, **kwargs)
