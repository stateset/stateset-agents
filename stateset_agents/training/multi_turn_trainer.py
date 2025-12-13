"""
Multi-turn GRPO trainer implementation.

This module provides the MultiTurnGRPOTrainer class for training
multi-turn conversational agents using Group Relative Policy Optimization.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from stateset_agents.core import trajectory as core_trajectory

from .loss_computation import (
    _compute_group_policy_loss,
    compute_enhanced_grpo_loss,
    compute_grpo_loss,
)
from .trainer_utils import (
    get_amp,
    get_cosine_schedule_with_warmup,
    get_functional,
    get_linear_schedule_with_warmup,
    get_torch,
    require_torch,
    require_transformers,
)

MultiTurnTrajectory = core_trajectory.MultiTurnTrajectory
TrajectoryGroup = core_trajectory.TrajectoryGroup

logger = logging.getLogger(__name__)


class MultiTurnGRPOTrainer:
    """
    GRPO trainer for multi-turn agents with HuggingFace and W&B integration

    This trainer implements Group Relative Policy Optimization specifically
    designed for multi-turn conversational agents, with full integration
    to HuggingFace ecosystem and Weights & Biases tracking.
    """

    def __init__(
        self,
        agent: Any,
        environment: Any,
        reward_fn: Optional[Any] = None,
        config: Optional[Any] = None,
        wandb_logger: Optional[Any] = None,
        callbacks: Optional[List] = None,
    ):
        self.agent = agent
        self.environment = environment
        self.reward_fn = reward_fn
        self.config = config
        self.wandb_logger = wandb_logger
        self.callbacks = callbacks or []

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_metric = float("-inf")
        self.steps_without_improvement = 0

        # HuggingFace components
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None

        # Training data
        self.train_dataset = None
        self.eval_dataset = None

        # GRPO enhancements
        self._global_reward_mean: float = 0.0
        self._global_reward_count: int = 0
        self.reference_model = None

        logger.info(f"GRPO Trainer initialized with {type(agent).__name__}")

    async def initialize(self):
        """Initialize trainer components"""
        logger.info("Initializing GRPO trainer...")

        torch = require_torch()
        require_transformers()
        amp = get_amp()

        # Initialize agent if not already done
        if not hasattr(self.agent, "model") or self.agent.model is None:
            await self.agent.initialize()

        # Set seeds for reproducibility if provided
        try:
            import random

            seed_value = getattr(self.config, "seed", None)
            if seed_value is not None:
                random.seed(seed_value)
                np.random.seed(seed_value)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_value)
                torch.manual_seed(seed_value)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception as seed_err:
            logger.warning(f"Failed to fully set deterministic seeds: {seed_err}")

        # Initialize optimizer with HuggingFace best practices
        self._setup_optimizer()

        # Initialize mixed precision scaler (CUDA only)
        if torch.cuda.is_available() and (
            getattr(self.config, "bf16", False) or getattr(self.config, "fp16", False)
        ):
            if amp is not None:
                self.scaler = amp.GradScaler("cuda")
            else:  # pragma: no cover - AMP not available
                logger.warning(
                    "Mixed precision requested but torch.amp is unavailable; "
                    "proceeding without GradScaler."
                )

        # Optional reference model for KL penalty
        try:
            if (
                getattr(self.config, "use_reference_model", False)
                and self.reference_model is None
            ):
                from copy import deepcopy

                self.reference_model = deepcopy(self.agent.model).eval()
                for param in self.reference_model.parameters():
                    param.requires_grad = False
                logger.info("Reference model initialized for KL regularization")
        except Exception as ref_err:
            logger.warning(f"Could not initialize reference model: {ref_err}")
            self.reference_model = None

        # Initialize W&B if configured
        if (
            self.wandb_logger
            and hasattr(self.config, "report_to")
            and self.config.report_to == "wandb"
        ):
            await self._init_wandb()

        logger.info("GRPO trainer initialization complete")

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        torch = require_torch()

        # Get model parameters
        model_params = list(self.agent.model.named_parameters())

        # Apply weight decay to all parameters except biases and layer norms
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": getattr(self.config, "weight_decay", 0.01),
            },
            {
                "params": [
                    p for n, p in model_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=getattr(self.config, "learning_rate", 5e-6),
            betas=(
                getattr(self.config, "adam_beta1", 0.9),
                getattr(self.config, "adam_beta2", 0.99),
            ),
            eps=1e-8,
        )

        logger.info(
            f"Optimizer initialized: AdamW with lr={getattr(self.config, 'learning_rate', 5e-6)}"
        )

    def _setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler"""
        require_transformers()

        num_warmup_steps = int(
            num_training_steps * getattr(self.config, "warmup_ratio", 0.1)
        )

        lr_scheduler_type = getattr(self.config, "lr_scheduler_type", "cosine")

        if lr_scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif lr_scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            # Constant learning rate
            self.lr_scheduler = None

        if self.lr_scheduler:
            logger.info(
                f"Scheduler initialized: {lr_scheduler_type} with {num_warmup_steps} warmup steps"
            )

    async def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        try:
            # Prepare configuration for W&B
            config_dict = {
                "framework": "grpo-agent-framework",
                "agent_type": type(self.agent).__name__,
                "environment_type": type(self.environment).__name__,
            }

            if hasattr(self.config, "__dict__"):
                config_dict.update(self.config.__dict__)
            if hasattr(self.agent.config, "__dict__"):
                config_dict.update(self.agent.config.__dict__)

            self.wandb_logger.init_run(
                config=config_dict,
                name=getattr(self.config, "run_name", None),
                tags=getattr(self.config, "wandb_tags", ["grpo", "multi-turn"]),
                notes=f"Training {type(self.agent).__name__} with GRPO",
            )

            logger.info("W&B tracking initialized")

        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")

    async def generate_trajectories(
        self, scenarios: List[Dict[str, Any]], num_generations: Optional[int] = None
    ) -> List[TrajectoryGroup]:
        """Generate trajectory groups for training"""
        num_generations = num_generations or getattr(self.config, "num_generations", 16)
        trajectory_groups = []

        for scenario in scenarios:
            # Generate multiple trajectories for the same scenario
            trajectories = []

            for _ in range(num_generations):
                try:
                    # Create agent function wrapper
                    async def agent_fn(history, context):
                        return await self.agent.generate_response(history, context)

                    # Generate trajectory
                    trajectory = await self.environment.run_episode(agent_fn, scenario)

                    # Apply reward function if provided
                    if self.reward_fn:
                        reward_result = await self.reward_fn.compute_reward(
                            trajectory.turns, scenario
                        )

                        # Handle both dict and RewardResult object formats
                        if hasattr(reward_result, "score"):
                            trajectory.total_reward = reward_result.score
                            trajectory.metadata["reward_breakdown"] = (
                                reward_result.breakdown
                            )
                        else:
                            trajectory.total_reward = reward_result["score"]
                            trajectory.metadata["reward_breakdown"] = reward_result[
                                "breakdown"
                            ]

                    trajectories.append(trajectory)

                except Exception as e:
                    logger.warning(f"Failed to generate trajectory: {e}")
                    continue

            if trajectories:
                group = TrajectoryGroup(
                    scenario_id=scenario.get(
                        "id", f"scenario_{len(trajectory_groups)}"
                    ),
                    trajectories=trajectories,
                    scenario_metadata=scenario,
                )
                trajectory_groups.append(group)

        return trajectory_groups

    def compute_grpo_loss(
        self, trajectory_groups: List[TrajectoryGroup]
    ) -> Dict[str, Any]:
        """Compute GRPO loss from trajectory groups"""
        return compute_grpo_loss(
            trajectory_groups=trajectory_groups,
            config=self.config,
            agent=self.agent,
            global_reward_mean=self._global_reward_mean,
            global_reward_count=self._global_reward_count,
            update_global_stats=self._update_global_stats,
        )

    def _compute_group_policy_loss(self, group: TrajectoryGroup, advantages: Any) -> Any:
        """Compute per-group policy loss (compatibility shim for tests)."""
        return _compute_group_policy_loss(
            group=group,
            advantages=advantages,
            config=self.config,
            agent=self.agent,
        )

    def compute_enhanced_grpo_loss(
        self, trajectory_groups: List[TrajectoryGroup], beta: float = 0.0
    ) -> Dict[str, Any]:
        """Enhanced GRPO loss computation with KL penalty"""
        return compute_enhanced_grpo_loss(
            trajectory_groups=trajectory_groups,
            beta=beta,
            config=self.config,
            agent=self.agent,
            reference_model=self.reference_model,
        )

    def _update_global_stats(self, batch_mean: float, batch_size: int):
        """Update global reward statistics"""
        self._global_reward_mean = (
            self._global_reward_mean * self._global_reward_count
            + batch_mean * batch_size
        ) / max(1, self._global_reward_count + batch_size)
        self._global_reward_count += batch_size

    def _format_trajectory_for_model(self, trajectory: MultiTurnTrajectory) -> str:
        """Format trajectory into text for model input"""
        def _as_message(turn: Any) -> Dict[str, str]:
            if isinstance(turn, dict):
                role = str(turn.get("role", "user"))
                content = str(turn.get("content", ""))
                return {"role": role, "content": content}
            role = getattr(turn, "role", None) or "user"
            content = getattr(turn, "content", None) or ""
            return {"role": str(role), "content": str(content)}

        if hasattr(self.agent.tokenizer, "apply_chat_template"):
            # Use tokenizer's chat template
            messages = [_as_message(turn) for turn in trajectory.turns]

            return self.agent.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            # Simple formatting
            parts = []
            for turn in trajectory.turns:
                msg = _as_message(turn)
                if msg["role"] == "user":
                    parts.append(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    parts.append(f"Assistant: {msg['content']}")

            return "\n".join(parts)

    async def training_step(
        self, trajectory_groups: List[TrajectoryGroup]
    ) -> Dict[str, Any]:
        """Execute a single training step"""
        torch = require_torch()
        amp = get_amp()

        self.agent.model.train()

        # Compute GRPO loss
        use_amp = bool(
            getattr(self.config, "bf16", False) or getattr(self.config, "fp16", False)
        )
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        use_enhanced = bool(
            getattr(self.config, "beta", 0.0) > 0.0
            or getattr(self.config, "use_reference_model", False)
        )

        autocast_ctx = (
            amp.autocast(device_type=device_type, enabled=use_amp)
            if amp is not None
            else contextlib.nullcontext()
        )

        with autocast_ctx:
            if use_enhanced:
                loss_dict = self.compute_enhanced_grpo_loss(
                    trajectory_groups, beta=float(getattr(self.config, "beta", 0.0))
                )
            else:
                loss_dict = self.compute_grpo_loss(trajectory_groups)
            loss = loss_dict["total_loss"]

        # Backward pass with gradient scaling
        if self.scaler:
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.agent.model.parameters(),
                getattr(self.config, "max_grad_norm", 1.0),
            )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.agent.model.parameters(),
                getattr(self.config, "max_grad_norm", 1.0),
            )
            self.optimizer.step()

        # Learning rate schedule step
        if self.lr_scheduler:
            self.lr_scheduler.step()

        # Clear gradients
        self.optimizer.zero_grad()

        # Update global step
        self.global_step += 1

        # Prepare metrics
        metrics = {
            **{
                k: v.item() if torch.is_tensor(v) else v
                for k, v in loss_dict.items()
            },
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "global_step": self.global_step,
        }

        return metrics

    async def evaluate(
        self, eval_scenarios: List[Dict[str, Any]], num_eval_episodes: int = 10
    ) -> Dict[str, float]:
        """Evaluate the agent"""
        self.agent.model.eval()

        # Generate evaluation trajectories
        eval_trajectory_groups = await self.generate_trajectories(
            eval_scenarios[:num_eval_episodes],
            num_generations=4,  # Fewer generations for evaluation
        )

        # Compute evaluation metrics
        all_rewards = []
        episode_lengths = []

        for group in eval_trajectory_groups:
            for trajectory in group.trajectories:
                all_rewards.append(trajectory.total_reward)
                episode_lengths.append(trajectory.episode_length)

        if not all_rewards:
            return {"eval_reward": 0.0, "eval_episode_length": 0.0}

        eval_metrics = {
            "eval_reward": np.mean(all_rewards),
            "eval_reward_std": np.std(all_rewards),
            "eval_episode_length": np.mean(episode_lengths),
            "eval_success_rate": np.mean([r > 0.5 for r in all_rewards]),
        }

        # Check for best model
        current_metric = eval_metrics["eval_reward"]
        if current_metric > self.best_eval_metric:
            self.best_eval_metric = current_metric
            self.steps_without_improvement = 0

            # Save best checkpoint
            await self.save_checkpoint(is_best=True)
        else:
            self.steps_without_improvement += 1

        return eval_metrics

    async def train(self) -> Any:
        """Main training loop"""
        logger.info("Starting GRPO training")

        # Calculate total training steps
        num_episodes = getattr(self.config, "num_episodes", 100)
        gradient_accumulation_steps = getattr(
            self.config, "gradient_accumulation_steps", 4
        )
        total_steps = num_episodes // gradient_accumulation_steps

        # Setup learning rate scheduler
        self._setup_scheduler(total_steps)

        # Training scenarios (this would come from your dataset)
        training_scenarios = self._get_training_scenarios()
        eval_scenarios = self._get_eval_scenarios()

        try:
            for episode in range(num_episodes):
                # Generate trajectory groups
                trajectory_groups = await self.generate_trajectories(
                    training_scenarios[episode : episode + 1]  # One scenario per episode
                )

                if not trajectory_groups:
                    logger.warning(
                        f"No trajectory groups generated for episode {episode}"
                    )
                    continue

                # Training step
                metrics = await self.training_step(trajectory_groups)

                # Log training metrics
                logging_steps = getattr(self.config, "logging_steps", 10)
                if self.global_step % logging_steps == 0:
                    await self._log_training_metrics(metrics, trajectory_groups)

                # Evaluation
                eval_steps = getattr(self.config, "eval_steps", 50)
                if self.global_step % eval_steps == 0:
                    eval_metrics = await self.evaluate(eval_scenarios)
                    await self._log_eval_metrics(eval_metrics)

                    logger.info(
                        f"Step {self.global_step}: "
                        f"Train Loss = {metrics['total_loss']:.4f}, "
                        f"Eval Reward = {eval_metrics['eval_reward']:.4f}"
                    )

                # Save checkpoint
                save_steps = getattr(self.config, "save_steps", 100)
                if self.global_step % save_steps == 0:
                    await self.save_checkpoint()

                # Early stopping check
                early_stopping = getattr(self.config, "early_stopping", False)
                patience = getattr(self.config, "patience", 50)
                if early_stopping and self.steps_without_improvement >= patience:
                    logger.info(f"Early stopping at step {self.global_step}")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        finally:
            # Final checkpoint
            await self.save_checkpoint()

            # Finish W&B run
            if self.wandb_logger:
                self.wandb_logger.finish_run(
                    {
                        "final_step": self.global_step,
                        "best_eval_metric": self.best_eval_metric,
                    }
                )

        logger.info("Training completed")
        return self.agent

    async def run_post_training_evaluation(
        self,
        eval_scenarios: List[Dict[str, Any]],
        num_samples: int = 5,
        detailed: bool = True,
    ) -> Dict[str, Any]:
        """Run comprehensive post-training evaluation"""
        logger.info(f"Running post-training evaluation on {num_samples} samples...")

        self.agent.model.eval()
        eval_results = []

        # Sample scenarios for evaluation
        sample_scenarios = (
            eval_scenarios[:num_samples]
            if len(eval_scenarios) > num_samples
            else eval_scenarios
        )

        for scenario_idx, scenario in enumerate(sample_scenarios):
            scenario_results = {
                "scenario_id": scenario.get("id", f"eval_{scenario_idx}"),
                "trajectories": [],
            }

            # Generate multiple trajectories for this scenario
            num_eval_generations = min(4, getattr(self.config, "num_generations", 4))

            for gen_idx in range(num_eval_generations):
                try:
                    # Create agent function
                    async def agent_fn(history, context):
                        return await self.agent.generate_response(history, context)

                    # Generate trajectory
                    trajectory = await self.environment.run_episode(agent_fn, scenario)

                    # Compute reward
                    if self.reward_fn:
                        reward_result = await self.reward_fn.compute_reward(
                            trajectory.turns, scenario
                        )

                        if hasattr(reward_result, "score"):
                            trajectory.total_reward = reward_result.score
                            reward_breakdown = reward_result.breakdown
                        else:
                            trajectory.total_reward = reward_result["score"]
                            reward_breakdown = reward_result.get("breakdown", {})
                    else:
                        trajectory.total_reward = 0.0
                        reward_breakdown = {}

                    # Store trajectory info
                    traj_info = {
                        "generation_idx": gen_idx,
                        "reward": trajectory.total_reward,
                        "reward_breakdown": reward_breakdown,
                        "num_turns": len(trajectory.turns),
                        "episode_length": trajectory.episode_length,
                    }

                    if detailed:
                        # Include actual conversation
                        traj_info["conversation"] = [
                            {
                                "role": turn.role,
                                "content": turn.content[:200] + "..."
                                if len(turn.content) > 200
                                else turn.content,
                            }
                            for turn in trajectory.turns
                        ]

                    scenario_results["trajectories"].append(traj_info)

                except Exception as e:
                    logger.warning(f"Failed to evaluate trajectory: {e}")
                    continue

            # Compute scenario statistics
            if scenario_results["trajectories"]:
                rewards = [t["reward"] for t in scenario_results["trajectories"]]
                scenario_results["stats"] = {
                    "mean_reward": np.mean(rewards),
                    "std_reward": np.std(rewards),
                    "max_reward": np.max(rewards),
                    "min_reward": np.min(rewards),
                }

            eval_results.append(scenario_results)

        # Compute overall statistics
        all_rewards = []
        all_lengths = []
        for result in eval_results:
            for traj in result.get("trajectories", []):
                all_rewards.append(traj["reward"])
                all_lengths.append(traj["episode_length"])

        overall_stats = {
            "num_scenarios_evaluated": len(eval_results),
            "total_trajectories": len(all_rewards),
            "overall_mean_reward": np.mean(all_rewards) if all_rewards else 0.0,
            "overall_std_reward": np.std(all_rewards) if all_rewards else 0.0,
            "overall_mean_length": np.mean(all_lengths) if all_lengths else 0.0,
            "reward_distribution": {
                "p25": np.percentile(all_rewards, 25) if all_rewards else 0.0,
                "p50": np.percentile(all_rewards, 50) if all_rewards else 0.0,
                "p75": np.percentile(all_rewards, 75) if all_rewards else 0.0,
                "p90": np.percentile(all_rewards, 90) if all_rewards else 0.0,
            },
        }

        # Log evaluation summary
        logger.info("Post-training evaluation complete:")
        logger.info(f"  Mean reward: {overall_stats['overall_mean_reward']:.4f}")
        logger.info(f"  Std reward: {overall_stats['overall_std_reward']:.4f}")
        logger.info(
            f"  Mean episode length: {overall_stats['overall_mean_length']:.2f}"
        )

        return {
            "overall_stats": overall_stats,
            "scenario_results": eval_results if detailed else None,
        }

    def _get_training_scenarios(self) -> List[Dict[str, Any]]:
        """Get training scenarios (placeholder)"""
        num_episodes = getattr(self.config, "num_episodes", 100)
        return [
            {"id": f"train_{i}", "context": f"Training scenario {i}"}
            for i in range(num_episodes)
        ]

    def _get_eval_scenarios(self) -> List[Dict[str, Any]]:
        """Get evaluation scenarios (placeholder)"""
        return [
            {"id": f"eval_{i}", "context": f"Evaluation scenario {i}"}
            for i in range(20)
        ]

    async def _log_training_metrics(
        self, metrics: Dict[str, Any], trajectory_groups: List[TrajectoryGroup]
    ):
        """Log training metrics"""
        if self.wandb_logger:
            self.wandb_logger.log_training_step(
                losses={k: v for k, v in metrics.items() if "loss" in k},
                learning_rate=metrics["learning_rate"],
                step=self.global_step,
                trajectory_groups=trajectory_groups,
            )

    async def _log_eval_metrics(self, eval_metrics: Dict[str, float]):
        """Log evaluation metrics"""
        if self.wandb_logger:
            self.wandb_logger.log_evaluation(
                eval_metrics=eval_metrics, step=self.global_step
            )

    async def save_checkpoint(
        self, is_best: bool = False, checkpoint_name: Optional[str] = None
    ):
        """Save model checkpoint with HuggingFace format"""
        torch = require_torch()

        if checkpoint_name is None:
            checkpoint_name = f"checkpoint-{self.global_step}"
            if is_best:
                checkpoint_name = "best-checkpoint"

        output_dir = getattr(self.config, "output_dir", "./outputs")
        checkpoint_path = Path(output_dir) / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save using HuggingFace save_pretrained
        self.agent.model.save_pretrained(checkpoint_path)
        self.agent.tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        training_state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_eval_metric": self.best_eval_metric,
            "steps_without_improvement": self.steps_without_improvement,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if hasattr(self.config, "__dict__"):
            training_state["config"] = self.config.__dict__

        if self.lr_scheduler:
            training_state["scheduler_state_dict"] = self.lr_scheduler.state_dict()

        torch.save(training_state, checkpoint_path / "training_state.pt")

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Log to W&B
        if self.wandb_logger:
            self.wandb_logger.log_model_checkpoint(
                str(checkpoint_path), self.global_step, is_best=is_best
            )

    def add_callback(self, callback):
        """Add training callback"""
        self.callbacks.append(callback)


# Alias for backwards compatibility
GRPOTrainer = MultiTurnGRPOTrainer
