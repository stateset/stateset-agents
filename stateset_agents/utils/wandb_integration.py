"""
Enhanced Weights & Biases Integration for GRPO Agent Framework

This module provides comprehensive experiment tracking, visualization,
and analysis capabilities for GRPO training with rich metrics and insights.
"""

from __future__ import annotations

import importlib.util
import logging
import tempfile
from datetime import datetime
from typing import Any

import numpy as np

from stateset_agents.core.reward import RewardResult
from stateset_agents.core.trajectory import Trajectory

from .monitoring import MonitoringService
from .wandb_config import WandBConfig, create_wandb_config, setup_wandb_from_env

plt = None  # type: ignore[assignment]
sns = None  # type: ignore[assignment]
wandb = None

WANDB_INSTALLED = importlib.util.find_spec("wandb") is not None
MATPLOTLIB_INSTALLED = importlib.util.find_spec("matplotlib") is not None
SEABORN_INSTALLED = importlib.util.find_spec("seaborn") is not None

# Backwards-compat flag; indicates whether wandb can be imported.
WANDB_AVAILABLE = WANDB_INSTALLED

WANDB_EXCEPTIONS: tuple[type[BaseException], ...] = (
    RuntimeError,
    ValueError,
    TypeError,
    OSError,
    KeyError,
    AttributeError,
)


def _register_wandb_exceptions() -> None:
    """Extend exception tuple with wandb-specific errors if available."""
    global WANDB_EXCEPTIONS
    if wandb is None:
        return
    errors_mod = getattr(wandb, "errors", None)
    if errors_mod is None:
        return
    excs = list(WANDB_EXCEPTIONS)
    for name in ("Error", "CommError", "UsageError"):
        exc = getattr(errors_mod, name, None)
        if isinstance(exc, type) and issubclass(exc, Exception):
            excs.append(exc)
    WANDB_EXCEPTIONS = tuple(dict.fromkeys(excs))


def _load_wandb() -> bool:
    """Import wandb on-demand to avoid slow import at module load time."""
    global WANDB_AVAILABLE, wandb
    if wandb is not None:
        return True
    try:
        import wandb as _wandb  # type: ignore

        wandb = _wandb
        WANDB_AVAILABLE = True
        _register_wandb_exceptions()
        return True
    except (ImportError, RuntimeError, OSError) as exc:  # pragma: no cover - env
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
    except (ImportError, RuntimeError, OSError) as exc:  # pragma: no cover - env
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

TrainingConfig = None  # Lazy import

logger = logging.getLogger(__name__)
class WandBIntegration:
    """
    Enhanced Weights & Biases integration for GRPO training
    """

    def __init__(
        self,
        config: WandBConfig,
        training_config: TrainingConfig | None = None,
        monitoring_service: MonitoringService | None = None,
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
        self, model: Any | None = None, optimizer: Any | None = None, **kwargs
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
        metrics: dict[str, Any],
        step: int | None = None,
        epoch: int | None = None,
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
        trajectories: list[Trajectory],
        rewards: list[RewardResult],
        step: int | None = None,
    ):
        """Log trajectory data with rich analysis"""
        if not self.is_initialized or not self.config.log_trajectories:
            return

        # Store trajectory history
        trajectory_data = []
        for i, (traj, reward) in enumerate(zip(trajectories, rewards, strict=False)):
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
        self, rewards: list[RewardResult], step: int | None = None
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
        gradient_norm: float | None = None,
        step: int | None = None,
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

    def log_model_metrics(self, model: Any, step: int | None = None):
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
        gpu_memory: float | None = None,
        cpu_usage: float | None = None,
        step: int | None = None,
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
        except Exception:
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
        except WANDB_EXCEPTIONS as e:
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

        except WANDB_EXCEPTIONS as e:
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

        except WANDB_EXCEPTIONS as e:
            logger.error(f"Failed to create reward plots: {e}")

    def _create_loss_plots(self):
        """Create loss analysis plots"""
        if not self.loss_history:
            return

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Loss trend
            steps = [entry["step"] for entry in self.loss_history]
            losses = [entry["loss"] for entry in self.loss_history]

            axes[0, 0].plot(steps, losses, alpha=0.7)
            axes[0, 0].set_title("Training Loss")
            axes[0, 0].set_xlabel("Step")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].set_yscale("log")

            # Learning rate
            learning_rates = [entry["learning_rate"] for entry in self.loss_history]
            axes[0, 1].plot(steps, learning_rates, alpha=0.7)
            axes[0, 1].set_title("Learning Rate")
            axes[0, 1].set_xlabel("Step")
            axes[0, 1].set_ylabel("Learning Rate")

            # Gradient norms
            grad_norms = [
                entry["gradient_norm"]
                for entry in self.loss_history
                if entry["gradient_norm"] is not None
            ]
            if grad_norms:
                grad_steps = [
                    entry["step"]
                    for entry in self.loss_history
                    if entry["gradient_norm"] is not None
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

        except WANDB_EXCEPTIONS as e:
            logger.error(f"Failed to create loss plots: {e}")

    def save_model_artifact(
        self, model: Any, name: str, metadata: dict[str, Any] | None = None
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

        except WANDB_EXCEPTIONS as e:
            logger.error(f"Failed to save model artifact: {e}")

    def create_summary_report(self) -> dict[str, Any]:
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
            losses = [entry["loss"] for entry in self.loss_history]
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

class WandBMonitoringService(MonitoringService):
    """Monitoring service that logs to W&B"""

    def __init__(self, wandb_integration: WandBIntegration):
        self.wandb = wandb_integration

    async def log_metric(
        self,
        name: str,
        value: int | float,
        tags: dict[str, str] | None = None,
        timestamp: datetime | None = None,
    ):
        """Log a metric to W&B"""
        if self.wandb.is_initialized:
            self.wandb.log_step({name: value}, commit=False)

    async def log_event(
        self,
        name: str,
        data: dict[str, Any],
        tags: dict[str, str] | None = None,
        timestamp: datetime | None = None,
    ):
        """Log an event to W&B"""
        if self.wandb.is_initialized:
            self.wandb.log_step({f"events/{name}": data}, commit=False)

    async def log_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        timestamp: datetime | None = None,
    ):
        """Log an error to W&B"""
        if self.wandb.is_initialized:
            error_data = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {},
            }
            self.wandb.log_step({"errors/error": error_data}, commit=False)

from .wandb_logger import WandBLogger, init_wandb

__all__ = [
    "MATPLOTLIB_INSTALLED",
    "SEABORN_INSTALLED",
    "TORCH_AVAILABLE",
    "WANDB_AVAILABLE",
    "WANDB_EXCEPTIONS",
    "WANDB_INSTALLED",
    "WandBConfig",
    "WandBIntegration",
    "WandBLogger",
    "WandBMonitoringService",
    "_load_plotting",
    "_load_wandb",
    "create_wandb_config",
    "init_wandb",
    "setup_wandb_from_env",
]
