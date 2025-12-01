"""
Group Sequence Policy Optimization (GSPO) Training for StateSet Agents

This module implements GSPO, a stable and efficient RL algorithm for training
large language models. GSPO uses sequence-level importance ratios and performs
sequence-level clipping, rewarding, and optimization.

Reference: https://arxiv.org/abs/2507.18071v2
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import framework components
from ..core.agent import Agent, AgentConfig, MultiTurnAgent
from ..core.environment import ConversationEnvironment
from ..core.trajectory import ConversationTurn, MultiTurnTrajectory
from ..rewards.multi_objective_reward import MultiObjectiveReward
from .config import TrainingConfig, get_config_for_task

# Import additional dependencies
try:
    import numpy as np
    import wandb
    from datasets import Dataset
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        get_cosine_schedule_with_warmup,
    )
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    logger.error("Please install: pip install transformers peft datasets")
    raise

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not installed. Install with `pip install vllm` for faster generation.")


@dataclass
class GSPOConfig(TrainingConfig):
    """Configuration for GSPO training"""

    # GSPO specific parameters
    num_generations: int = 4  # Group size (G) - number of responses per query
    beta: float = 0.0  # KL penalty coefficient

    # Clipping ranges (sequence-level)
    # Note: GSPO uses smaller clipping ranges than GRPO (different order of magnitude)
    clip_range_left: float = 3e-4  # Left clipping range (1 - ε)
    clip_range_right: float = 4e-4  # Right clipping range (1 + ε)

    # Training parameters
    num_iterations: int = 1  # Number of training iterations per batch
    mini_batch_size: int = 1  # Mini batch size

    # Online/Iterative training
    num_outer_iterations: int = 1  # Number of Generate -> Train cycles
    generations_per_iteration: int = 100  # Prompts per cycle

    # Generation parameters
    max_prompt_length: int = 256
    max_completion_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    # Model optimization
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # Memory optimization
    gradient_checkpointing: bool = True
    use_8bit: bool = False
    use_4bit: bool = False

    # Backend
    use_vllm: bool = False  # Enable vLLM for generation

    # GSPO variant
    use_gspo_token: bool = False  # Use GSPO-token variant for token-level advantages

    @classmethod
    def from_training_config(cls, config: TrainingConfig, **kwargs) -> "GSPOConfig":
        """Create GSPO config from standard training config"""
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        return cls(**config_dict)


class GSPOModelManager:
    """Manages model loading and LoRA configuration for GSPO training"""

    def __init__(self, config: GSPOConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.ref_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load model and tokenizer with LoRA if specified"""
        logger.info(f"Loading model: {self.config.model_name}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="left",
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Model loading kwargs
            model_kwargs = {
                "torch_dtype": torch.float16
                if self.config.fp16
                else (torch.bfloat16 if self.config.bf16 else torch.float32),
                "device_map": "auto" if torch.cuda.is_available() else None,
                "trust_remote_code": True,
            }

            # Add quantization if specified
            if self.config.use_8bit:
                model_kwargs["load_in_8bit"] = True
            elif self.config.use_4bit:
                model_kwargs["load_in_4bit"] = True

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, **model_kwargs
            )

            # Enable gradient checkpointing if specified
            if self.config.gradient_checkpointing:
                base_model.gradient_checkpointing_enable()

            # Prepare model for training if using quantization
            if self.config.use_8bit or self.config.use_4bit:
                base_model = prepare_model_for_kbit_training(base_model)

            # Add LoRA adapters
            if self.config.use_lora:
                # Determine target modules based on model architecture
                if self.config.lora_target_modules:
                    target_modules = self.config.lora_target_modules
                elif "gpt2" in self.config.model_name.lower():
                    target_modules = ["c_attn", "c_proj"]
                elif "llama" in self.config.model_name.lower():
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                else:
                    target_modules = ["q_proj", "v_proj"]

                lora_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )

                self.model = get_peft_model(base_model, lora_config)
                self.model.print_trainable_parameters()

                logger.info("LoRA adapters added to model")
            else:
                self.model = base_model

            # Load reference model if using KL penalty
            if self.config.beta > 0 and self.config.use_reference_model:
                logger.info("Loading reference model for KL penalty...")
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=model_kwargs["torch_dtype"],
                    device_map="auto" if torch.cuda.device_count() > 1 else None,
                )
                self.ref_model.eval()

            logger.info(f"Model loaded successfully on {self.device}")
            return self.model, self.tokenizer

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


class GSPOTrajectoryGenerator:
    """Handles efficient trajectory generation for GSPO training"""

    def __init__(self, config: GSPOConfig, agent: Agent, environment: ConversationEnvironment):
        self.config = config
        self.agent = agent
        self.environment = environment
        self.vllm_engine = None

        if self.config.use_vllm and VLLM_AVAILABLE:
            self._init_vllm()

    def _init_vllm(self):
        """Initialize vLLM engine"""
        logger.info("Initializing vLLM engine for fast generation...")
        try:
            self.vllm_engine = LLM(
                model=self.config.model_name,
                trust_remote_code=True,
                dtype="float16" if self.config.fp16 else "bfloat16",
                gpu_memory_utilization=0.6,
            )
            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_completion_length,
            )
            logger.info("vLLM engine initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize vLLM: {e}. Falling back to standard generation.")
            self.vllm_engine = None

    async def generate_group_responses(
        self, prompt: str, num_responses: int
    ) -> List[Tuple[str, float]]:
        """
        Generate a group of responses for a single prompt.

        Returns:
            List of (response_text, log_prob) tuples
        """
        responses = []

        for _ in range(num_responses):
            # Generate response using agent
            messages = [{"role": "user", "content": prompt}]
            response = await self.agent.generate_response(messages)

            # Compute log probability of the response
            log_prob = await self._compute_sequence_log_prob(prompt, response)

            responses.append((response, log_prob))

        return responses

    async def _compute_sequence_log_prob(self, prompt: str, response: str) -> float:
        """Compute the log probability of a sequence"""
        # Tokenize prompt and response
        full_text = prompt + " " + response
        inputs = self.agent.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length + self.config.max_completion_length,
        ).to(self.agent.model.device)

        # Get logits from model
        with torch.no_grad():
            outputs = self.agent.model(**inputs)
            logits = outputs.logits

        # Compute log probabilities
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Sum log probs to get sequence log prob
        sequence_log_prob = token_log_probs.sum().item()

        return sequence_log_prob


class GSPOTrainer:
    """
    Group Sequence Policy Optimization trainer

    Implements the GSPO algorithm from: https://arxiv.org/abs/2507.18071v2
    """

    def __init__(
        self,
        config: GSPOConfig,
        model: Any,
        tokenizer: Any,
        agent: Agent,
        environment: ConversationEnvironment,
        reward_model: MultiObjectiveReward,
        ref_model: Optional[Any] = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.agent = agent
        self.environment = environment
        self.reward_model = reward_model
        self.ref_model = ref_model

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )

        # Scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.num_episodes * config.num_iterations,
        )

        # Trajectory generator
        self.generator = GSPOTrajectoryGenerator(config, agent, environment)

        # Metrics
        self.training_metrics = {
            "policy_loss": [],
            "clipping_fraction": [],
            "average_reward": [],
            "sequence_importance_ratio": [],
        }

    def compute_sequence_importance_ratio(
        self,
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sequence-level importance ratio with length normalization.

        s_i(θ) = (π_θ(y_i|x) / π_θ_old(y_i|x))^(1/|y_i|)
               = exp(1/|y_i| * Σ log(π_θ(y_i,t|x,y_<t) / π_θ_old(y_i,t|x,y_<t)))

        Args:
            current_log_probs: Log probabilities under current policy (sum over sequence)
            old_log_probs: Log probabilities under old policy (sum over sequence)
            sequence_lengths: Length of each sequence

        Returns:
            Sequence importance ratios
        """
        # Compute log ratio sum
        log_ratio_sum = current_log_probs - old_log_probs

        # Apply length normalization
        normalized_log_ratio = log_ratio_sum / sequence_lengths

        # Convert to ratio via exp
        importance_ratio = torch.exp(normalized_log_ratio)

        return importance_ratio

    def compute_group_advantages(
        self, rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute group-relative advantages (normalized rewards).

        Â_i = (r(x, y_i) - mean(rewards)) / std(rewards)

        Args:
            rewards: Tensor of rewards for a group of responses [group_size]

        Returns:
            advantages: Normalized advantages
            stats: Statistics about rewards
        """
        mean_reward = rewards.mean()
        std_reward = rewards.std()

        # Avoid division by zero
        if std_reward < 1e-8:
            std_reward = 1.0

        advantages = (rewards - mean_reward) / std_reward

        stats = {
            "mean_reward": mean_reward.item(),
            "std_reward": std_reward.item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
        }

        return advantages, stats

    async def train_step(
        self, queries: List[str], num_groups: int = 1
    ) -> Dict[str, float]:
        """
        Execute one GSPO training step.

        Args:
            queries: List of prompts/queries
            num_groups: Number of query groups to process

        Returns:
            Training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_clipped = 0
        total_samples = 0
        all_rewards = []
        all_importance_ratios = []

        for query in queries[:num_groups]:
            # Generate group of responses for this query
            group_responses = await self.generator.generate_group_responses(
                query, self.config.num_generations
            )

            # Extract responses and old log probs
            responses = [resp for resp, _ in group_responses]
            old_log_probs = torch.tensor(
                [log_prob for _, log_prob in group_responses],
                dtype=torch.float32,
            ).to(self.model.device)

            # Compute rewards for each response
            rewards = []
            for response in responses:
                turn = ConversationTurn(
                    role="assistant", content=response, metadata={"generated": True}
                )
                reward_info = await self.reward_model.compute_reward(
                    trajectory=None,
                    turn=turn,
                    context={"user_query": query},
                )
                rewards.append(reward_info.total_reward)

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(
                self.model.device
            )
            all_rewards.extend(rewards)

            # Compute group advantages
            advantages, reward_stats = self.compute_group_advantages(rewards_tensor)

            # Compute current log probs for each response
            current_log_probs = []
            sequence_lengths = []

            for response in responses:
                log_prob = await self.generator._compute_sequence_log_prob(query, response)
                current_log_probs.append(log_prob)

                # Compute sequence length
                tokens = self.tokenizer(response, return_tensors="pt")
                seq_len = tokens["input_ids"].shape[1]
                sequence_lengths.append(seq_len)

            current_log_probs = torch.tensor(
                current_log_probs, dtype=torch.float32
            ).to(self.model.device)
            sequence_lengths = torch.tensor(
                sequence_lengths, dtype=torch.float32
            ).to(self.model.device)

            # Compute sequence importance ratios
            importance_ratios = self.compute_sequence_importance_ratio(
                current_log_probs, old_log_probs, sequence_lengths
            )
            all_importance_ratios.extend(importance_ratios.tolist())

            # Apply clipping to importance ratios
            clipped_ratios = torch.clamp(
                importance_ratios,
                1 - self.config.clip_range_left,
                1 + self.config.clip_range_right,
            )

            # Count clipped sequences
            num_clipped = (importance_ratios != clipped_ratios).sum().item()
            total_clipped += num_clipped
            total_samples += len(responses)

            # Compute policy loss using GSPO objective
            # J_GSPO = E[1/G * Σ min(s_i(θ) * Â_i, clip(s_i(θ)) * Â_i)]

            # Unclipped objective
            unclipped_obj = importance_ratios * advantages

            # Clipped objective
            clipped_obj = clipped_ratios * advantages

            # Take minimum (conservative policy update)
            policy_loss = -torch.min(unclipped_obj, clipped_obj).mean()

            # Add KL penalty if specified
            if self.config.beta > 0 and self.ref_model is not None:
                # Compute KL divergence with reference model
                ref_log_probs = []
                for response in responses:
                    ref_log_prob = await self._compute_ref_log_prob(query, response)
                    ref_log_probs.append(ref_log_prob)

                ref_log_probs = torch.tensor(ref_log_probs, dtype=torch.float32).to(
                    self.model.device
                )

                # KL = log(π_θ/π_ref) = log π_θ - log π_ref
                kl_div = (current_log_probs - ref_log_probs) / sequence_lengths
                kl_penalty = self.config.beta * kl_div.mean()

                total_loss_item = policy_loss + kl_penalty
            else:
                total_loss_item = policy_loss

            # Accumulate loss
            total_loss += total_loss_item

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )

        # Update parameters
        self.optimizer.step()
        self.scheduler.step()

        # Compute metrics
        clipping_fraction = total_clipped / max(total_samples, 1)
        avg_reward = np.mean(all_rewards) if all_rewards else 0.0
        avg_importance_ratio = np.mean(all_importance_ratios) if all_importance_ratios else 1.0

        metrics = {
            "policy_loss": total_loss.item(),
            "clipping_fraction": clipping_fraction,
            "average_reward": avg_reward,
            "sequence_importance_ratio": avg_importance_ratio,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

        # Store metrics
        for key, value in metrics.items():
            if key in self.training_metrics:
                self.training_metrics[key].append(value)

        return metrics

    async def _compute_ref_log_prob(self, prompt: str, response: str) -> float:
        """Compute log probability under reference model"""
        full_text = prompt + " " + response
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length + self.config.max_completion_length,
        ).to(self.ref_model.device)

        with torch.no_grad():
            outputs = self.ref_model(**inputs)
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        sequence_log_prob = token_log_probs.sum().item()

        return sequence_log_prob

    def save_model(self, output_dir: str):
        """Save the trained model"""
        os.makedirs(output_dir, exist_ok=True)

        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save training metrics
        metrics_path = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.training_metrics, f, indent=2)

        logger.info(f"Model saved to {output_dir}")


async def train_with_gspo(
    config: GSPOConfig,
    agent: Agent,
    environment: ConversationEnvironment,
    reward_model: MultiObjectiveReward,
    train_queries: Optional[List[str]] = None,
) -> Agent:
    """
    Main training function using GSPO

    Args:
        config: GSPO configuration
        agent: Agent to train
        environment: Training environment
        reward_model: Reward function
        train_queries: Optional list of training queries

    Returns:
        Trained agent
    """
    logger.info("Initializing GSPO training")
    logger.info(f"Configuration: {json.dumps(config.to_dict(), indent=2)}")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize wandb if configured
    if config.report_to == "wandb" and config.wandb_project:
        wandb.init(
            project=config.wandb_project,
            name=config.run_name
            or f"gspo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config.to_dict(),
            tags=["gspo"] + (config.wandb_tags or []),
        )

    # Initialize model manager
    model_manager = GSPOModelManager(config)
    model, tokenizer = model_manager.load_model_and_tokenizer()

    # Update agent with loaded model
    agent.model = model
    agent.tokenizer = tokenizer

    # Generate training queries if not provided
    if not train_queries:
        logger.info("Generating training queries from environment scenarios...")
        train_queries = []
        for scenario in environment.scenarios[:config.generations_per_iteration]:
            query = scenario.get("context", "Hello")
            train_queries.append(query)

    logger.info(f"Training with {len(train_queries)} queries")

    # Create GSPO trainer
    trainer = GSPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        agent=agent,
        environment=environment,
        reward_model=reward_model,
        ref_model=model_manager.ref_model,
    )

    # Training loop
    for iteration in range(config.num_outer_iterations):
        logger.info(f"=== Iteration {iteration + 1}/{config.num_outer_iterations} ===")

        # Train step
        metrics = await trainer.train_step(
            queries=train_queries, num_groups=min(len(train_queries), 10)
        )

        # Log metrics
        logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

        if config.report_to == "wandb":
            wandb.log(metrics, step=iteration)

        # Save checkpoint
        if (iteration + 1) % config.save_steps == 0:
            checkpoint_dir = os.path.join(
                config.output_dir, f"checkpoint-{iteration + 1}"
            )
            trainer.save_model(checkpoint_dir)

    # Save final model
    final_model_path = os.path.join(config.output_dir, "final_model")
    trainer.save_model(final_model_path)

    # Finish wandb run
    if config.report_to == "wandb":
        wandb.finish()

    logger.info("✨ GSPO training completed successfully!")
    return agent


# Convenience functions

async def train_customer_service_with_gspo(
    model_name: str = "gpt2",
    num_episodes: int = 100,
    output_dir: str = "./outputs/gspo",
    **kwargs,
) -> Agent:
    """Train a customer service agent using GSPO"""

    # Get base configuration
    base_config = get_config_for_task("customer_service", model_name=model_name)

    # Create GSPO config
    config = GSPOConfig.from_training_config(
        base_config, num_episodes=num_episodes, output_dir=output_dir, **kwargs
    )

    # Create agent
    agent_config = AgentConfig(
        model_name=model_name,
        system_prompt="You are a helpful and empathetic customer service representative.",
    )
    agent = MultiTurnAgent(agent_config)
    await agent.initialize()

    # Create environment
    from ..core.environment import CONVERSATION_CONFIGS

    env_config = CONVERSATION_CONFIGS["customer_service"].copy()
    environment = ConversationEnvironment(**env_config)

    # Create reward model
    from ..rewards.multi_objective_reward import create_customer_service_reward

    reward_model = create_customer_service_reward()

    # Run training
    return await train_with_gspo(
        config=config,
        agent=agent,
        environment=environment,
        reward_model=reward_model,
    )


# Export main components
__all__ = [
    "GSPOConfig",
    "GSPOModelManager",
    "GSPOTrajectoryGenerator",
    "GSPOTrainer",
    "train_with_gspo",
    "train_customer_service_with_gspo",
]
