"""
GRPO Training Implementation with HuggingFace and W&B Integration

This module provides the core training infrastructure for multi-turn agents
using Group Relative Policy Optimization (GRPO) with seamless integration
to HuggingFace transformers and Weights & Biases.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

import numpy as np

try:  # Lazy torch import; resolved on demand for optional dependency
    import torch  # type: ignore[import-not-found]
    import torch.nn.functional as F  # type: ignore[import-not-found]
    import torch.amp as amp  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - handled via helper functions
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    amp = None  # type: ignore[assignment]

try:  # Lazy transformers import; resolved on demand
    from transformers import (  # type: ignore[import-not-found]
        DataCollatorForLanguageModeling,
        TrainingArguments,
        get_cosine_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )
except ImportError:  # pragma: no cover
    DataCollatorForLanguageModeling = None  # type: ignore[assignment]
    TrainingArguments = None  # type: ignore[assignment]
    get_cosine_schedule_with_warmup = None  # type: ignore[assignment]
    get_linear_schedule_with_warmup = None  # type: ignore[assignment]


def _require_torch() -> Any:
    """Ensure torch is available, importing lazily if needed."""
    global torch, F, amp
    if torch is None:
        try:
            import torch as _torch  # type: ignore[import-not-found]
            import torch.nn.functional as _F  # type: ignore[import-not-found]
            import torch.amp as _amp  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - import guarding
            raise ImportError(
                "PyTorch is required for training features. "
                "Install the 'training' extra: pip install stateset-agents[training]"
            ) from exc
        torch = _torch  # type: ignore[assignment]
        F = _F  # type: ignore[assignment]
        amp = _amp  # type: ignore[assignment]
    return torch  # type: ignore[return-value]


def _require_transformers() -> None:
    """Ensure transformers scheduling utilities are available."""
    global (
        DataCollatorForLanguageModeling,
        TrainingArguments,
        get_cosine_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )
    if (
        DataCollatorForLanguageModeling is None
        or TrainingArguments is None
        or get_cosine_schedule_with_warmup is None
        or get_linear_schedule_with_warmup is None
    ):
        try:
            from transformers import (  # type: ignore[import-not-found]
                DataCollatorForLanguageModeling as _DataCollatorForLanguageModeling,
                TrainingArguments as _TrainingArguments,
                get_cosine_schedule_with_warmup as _get_cosine_schedule_with_warmup,
                get_linear_schedule_with_warmup as _get_linear_schedule_with_warmup,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required for the training utilities. "
                "Install the 'training' extra: pip install stateset-agents[training]"
            ) from exc

        DataCollatorForLanguageModeling = _DataCollatorForLanguageModeling  # type: ignore[assignment]
        TrainingArguments = _TrainingArguments  # type: ignore[assignment]
        get_cosine_schedule_with_warmup = _get_cosine_schedule_with_warmup  # type: ignore[assignment]
        get_linear_schedule_with_warmup = _get_linear_schedule_with_warmup  # type: ignore[assignment]

from stateset_agents.core import agent as core_agent
from stateset_agents.core import environment as core_environment
from stateset_agents.core import reward as core_reward
from stateset_agents.core import trajectory as core_trajectory

Agent = core_agent.Agent
MultiTurnAgent = core_agent.MultiTurnAgent
AgentConfig = core_agent.AgentConfig

Environment = core_environment.Environment

MultiTurnTrajectory = core_trajectory.MultiTurnTrajectory
TrajectoryGroup = core_trajectory.TrajectoryGroup
ConversationTurn = core_trajectory.ConversationTurn

RewardFunction = core_reward.RewardFunction
CompositeReward = core_reward.CompositeReward

try:
    from utils.wandb_integration import WandBLogger as _WandBLogger
except ImportError:  # pragma: no cover - optional dependency
    _WandBLogger = None  # type: ignore[assignment]

WandBLogger = _WandBLogger  # type: ignore[assignment]
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
        agent: MultiTurnAgent,
        environment: Environment,
        reward_fn: Optional[RewardFunction] = None,
        config: Optional = None,
        wandb_logger: Optional[WandBLogger] = None,
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

        _require_torch()
        _require_transformers()

        # Torch imports are available after the helper call
        assert torch is not None  # for type checkers
        # Initialize agent if not already done
        if not hasattr(self.agent, "model") or self.agent.model is None:
            await self.agent.initialize()

        # Set seeds for reproducibility if provided
        try:
            import random

            import numpy as np

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
        if (
            torch.cuda.is_available()
            and (getattr(self.config, "bf16", False) or getattr(self.config, "fp16", False))
        ):
            if amp is not None:
                self.scaler = amp.GradScaler("cuda")
            else:  # pragma: no cover - AMP not available
                logger.warning(
                    "Mixed precision requested but torch.amp is unavailable; proceeding without GradScaler."
                )

        # Optional reference model for KL penalty
        try:
            if (
                getattr(self.config, "use_reference_model", False)
                and self.reference_model is None
            ):
                # Lazy import to avoid circulars
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
        _require_torch()
        assert torch is not None

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
        _require_transformers()

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
                            trajectory.metadata[
                                "reward_breakdown"
                            ] = reward_result.breakdown
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
    ) -> Dict[str, torch.Tensor]:
        """Compute GRPO loss from trajectory groups with configurable baseline and normalization"""
        _require_torch()
        assert torch is not None

        total_loss = 0.0
        policy_losses = []
        all_advantages_for_logging: List[float] = []

        # Configuration controls
        baseline_type = getattr(self.config, "baseline_type", "group_mean")
        normalize_adv = getattr(self.config, "advantage_normalization", True)
        reward_clip = getattr(self.config, "reward_clip", None)

        for group in trajectory_groups:
            if not group.trajectories:
                continue

            # Extract rewards and optionally clip
            rewards = torch.tensor(
                [t.total_reward for t in group.trajectories], dtype=torch.float32
            )
            if reward_clip is not None:
                rewards = torch.clamp(
                    rewards, min=-float(reward_clip), max=float(reward_clip)
                )

            # Select baseline
            if baseline_type == "group_median":
                baseline = rewards.median()
            elif baseline_type == "global_mean":
                # Update running global mean baseline
                with torch.no_grad():
                    batch_mean = rewards.mean().item()
                    self._global_reward_mean = (
                        self._global_reward_mean * self._global_reward_count
                        + batch_mean * len(rewards)
                    ) / max(1, self._global_reward_count + len(rewards))
                    self._global_reward_count += len(rewards)
                baseline = torch.tensor(self._global_reward_mean, dtype=torch.float32)
            else:  # group_mean (default)
                baseline = rewards.mean()

            advantages = rewards - baseline

            # Normalize advantages if configured and variance > 0
            if normalize_adv and len(advantages) > 1:
                std = advantages.std()
                if torch.isfinite(std) and std > 0:
                    advantages = (advantages - advantages.mean()) / (std + 1e-8)

            all_advantages_for_logging.extend(advantages.detach().cpu().tolist())

            # Compute policy loss for this group (placeholder; model-based loss computed elsewhere)
            group_loss = self._compute_group_policy_loss(group, advantages)
            policy_losses.append(group_loss)

        # Aggregate losses
        if policy_losses:
            total_loss_tensor = torch.stack(policy_losses).mean()
        else:
            total_loss_tensor = torch.tensor(0.0, requires_grad=True)

        return {
            "policy_loss": total_loss_tensor,
            "total_loss": total_loss_tensor,
            "mean_advantage": float(np.mean(all_advantages_for_logging))
            if all_advantages_for_logging
            else 0.0,
            "advantage_std": float(np.std(all_advantages_for_logging))
            if all_advantages_for_logging
            else 0.0,
        }

    def _compute_group_policy_loss(
        self, group: TrajectoryGroup, advantages: torch.Tensor
    ) -> torch.Tensor:
        """Compute policy loss for a single trajectory group"""
        _require_torch()
        assert torch is not None

        # This is a simplified placeholder implementation
        # In a full GRPO implementation, you would:
        # 1. Tokenize the conversations
        # 2. Run forward pass to get log probabilities
        # 3. Compute importance sampling ratios
        # 4. Apply PPO-style clipping with advantages

        # For demonstration, return a simple loss
        loss = torch.tensor(0.1, requires_grad=True) * advantages.mean()
        return loss

    def compute_enhanced_grpo_loss(
        self, trajectory_groups: List[TrajectoryGroup], beta: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """Enhanced GRPO loss computation with KL penalty and proper advantages"""

        _require_torch()
        assert torch is not None
        if F is None:
            raise ImportError(
                "torch.nn.functional is required for enhanced GRPO loss computation."
            )

        all_losses = []
        all_advantages = []
        all_kl_divs = []

        for group in trajectory_groups:
            if not group.trajectories:
                continue

            # Extract rewards for this group
            rewards = torch.tensor(
                [t.total_reward for t in group.trajectories], dtype=torch.float32
            )

            # GRPO: Use group mean as baseline
            baseline = rewards.mean()
            advantages = rewards - baseline

            # Normalize advantages within group
            if len(advantages) > 1 and advantages.std() > 0:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            all_advantages.extend(advantages.tolist())

            # Process each trajectory in the group
            for traj_idx, (trajectory, advantage) in enumerate(
                zip(group.trajectories, advantages)
            ):
                # Prepare inputs for model
                conversation_text = self._format_trajectory_for_model(trajectory)

                # Tokenize
                inputs = self.agent.tokenizer(
                    conversation_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=getattr(self.config, "max_prompt_length", 512)
                    + getattr(self.config, "max_completion_length", 512),
                    padding=True,
                )

                # Move to device
                inputs = {k: v.to(self.agent.model.device) for k, v in inputs.items()}

                # Forward pass
                use_amp = getattr(self.config, "bf16", False) or getattr(
                    self.config, "fp16", False
                )
                if amp is not None:
                    autocast_ctx = amp.autocast(
                        device_type="cuda" if torch.cuda.is_available() else "cpu",
                        enabled=bool(use_amp),
                    )
                else:  # pragma: no cover - fallback when amp missing
                    autocast_ctx = contextlib.nullcontext()

                with autocast_ctx:
                    outputs = self.agent.model(**inputs, labels=inputs["input_ids"])

                    # Get log probabilities
                    log_probs = outputs.logits.log_softmax(dim=-1)

                    # Compute policy loss with advantage weighting
                    policy_loss = -advantage * outputs.loss
                    all_losses.append(policy_loss)

                    # KL divergence penalty (if beta > 0)
                    if beta > 0 and hasattr(self, "reference_model"):
                        with torch.no_grad():
                            ref_outputs = self.reference_model(**inputs)
                            ref_log_probs = ref_outputs.logits.log_softmax(dim=-1)

                        kl_div = F.kl_div(  # type: ignore[operator]
                            log_probs, ref_log_probs, reduction="batchmean"
                        )
                        all_kl_divs.append(kl_div)

        # Aggregate losses
        if all_losses:
            policy_loss = torch.stack(all_losses).mean()

            # Add KL penalty if applicable
            if all_kl_divs and beta > 0:
                kl_penalty = torch.stack(all_kl_divs).mean()
                total_loss = policy_loss + beta * kl_penalty
            else:
                total_loss = policy_loss
                kl_penalty = torch.tensor(0.0)
        else:
            total_loss = torch.tensor(
                0.0, requires_grad=True, device=self.agent.model.device
            )
            policy_loss = total_loss
            kl_penalty = torch.tensor(0.0)

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "kl_penalty": kl_penalty,
            "mean_advantage": np.mean(all_advantages) if all_advantages else 0.0,
            "advantage_std": np.std(all_advantages) if all_advantages else 0.0,
            "num_trajectories": sum(len(g.trajectories) for g in trajectory_groups),
        }

    def _format_trajectory_for_model(self, trajectory: MultiTurnTrajectory) -> str:
        """Format trajectory into text for model input"""

        if hasattr(self.agent.tokenizer, "apply_chat_template"):
            # Use tokenizer's chat template
            messages = []
            for turn in trajectory.turns:
                messages.append({"role": turn.role, "content": turn.content})

            return self.agent.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            # Simple formatting
            parts = []
            for turn in trajectory.turns:
                if turn.role == "user":
                    parts.append(f"User: {turn.content}")
                elif turn.role == "assistant":
                    parts.append(f"Assistant: {turn.content}")

            return "\n".join(parts)

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
        logger.info(f"Post-training evaluation complete:")
        logger.info(f"  Mean reward: {overall_stats['overall_mean_reward']:.4f}")
        logger.info(f"  Std reward: {overall_stats['overall_std_reward']:.4f}")
        logger.info(
            f"  Mean episode length: {overall_stats['overall_mean_length']:.2f}"
        )

        return {
            "overall_stats": overall_stats,
            "scenario_results": eval_results if detailed else None,
        }

    async def training_step(
        self, trajectory_groups: List[TrajectoryGroup]
    ) -> Dict[str, Any]:
        """Execute a single training step"""
        _require_torch()
        assert torch is not None

        self.agent.model.train()

        # Compute GRPO loss
        use_amp = bool(getattr(self.config, "bf16", False) or getattr(self.config, "fp16", False))
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        use_enhanced = bool(
            getattr(self.config, "beta", 0.0) > 0.0
            or getattr(self.config, "use_reference_model", False)
        )
        # Use new torch.amp.autocast API with proper device selection
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
            **{k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()},
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

    async def train(self) -> MultiTurnAgent:
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
                    training_scenarios[
                        episode : episode + 1
                    ]  # One scenario per episode
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

    def _get_training_scenarios(self) -> List[Dict[str, Any]]:
        """Get training scenarios (placeholder)"""
        # This would be replaced with actual scenario loading
        num_episodes = getattr(self.config, "num_episodes", 100)
        return [
            {"id": f"train_{i}", "context": f"Training scenario {i}"}
            for i in range(num_episodes)
        ]

    def _get_eval_scenarios(self) -> List[Dict[str, Any]]:
        """Get evaluation scenarios (placeholder)"""
        # This would be replaced with actual scenario loading
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
        _require_torch()
        assert torch is not None

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
