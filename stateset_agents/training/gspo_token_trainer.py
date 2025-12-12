"""
GSPO-token: Token-level variant of GSPO for fine-grained advantage adjustment

This module implements the GSPO-token variant which allows token-wise advantage
customization while maintaining sequence-level importance ratios.

Reference: https://arxiv.org/abs/2507.18071v2 (Section 4.3)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from stateset_agents.core.agent import Agent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards.multi_objective_reward import (
    MultiObjectiveRewardFunction as MultiObjectiveReward,
)
from .gspo_trainer import GSPOConfig, GSPOTrainer, GSPOTrajectoryGenerator

logger = logging.getLogger(__name__)


class GSPOTokenTrainer(GSPOTrainer):
    """
    GSPO-token trainer for token-level advantage customization.

    The key difference from standard GSPO is:
    - Allows different advantages for each token in a response
    - Uses a special importance ratio: s_{i,t}(θ) = sg[s_i(θ)] * π_θ(y_{i,t}|...) / sg[π_θ(y_{i,t}|...)]
    - This ensures clipping is still sequence-level while advantages can be token-level
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
        super().__init__(
            config, model, tokenizer, agent, environment, reward_model, ref_model
        )

        # Override config flag
        self.config.use_gspo_token = True

    def compute_token_importance_ratio(
        self,
        sequence_importance_ratio: torch.Tensor,
        token_log_probs_current: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute token-level importance ratio for GSPO-token.

        s_{i,t}(θ) = sg[s_i(θ)] * π_θ(y_{i,t}|x, y_{<t}) / sg[π_θ(y_{i,t}|x, y_{<t})]

        The term π_θ(y_{i,t}|...) / sg[π_θ(y_{i,t}|...)] has numerical value of 1,
        so s_{i,t}(θ) is numerically equal to s_i(θ), but allows gradients to flow
        through individual tokens.

        Args:
            sequence_importance_ratio: Sequence-level importance ratio s_i(θ)
            token_log_probs_current: Current token log probs

        Returns:
            Token importance ratios (numerically equal to sequence ratio)
        """
        # Detach sequence ratio (stop gradient)
        detached_seq_ratio = sequence_importance_ratio.detach()

        # The multiplication by π_θ / sg[π_θ] is implicit in how we compute gradients
        # We return the detached sequence ratio which will be used for clipping
        # but gradients will flow through token log probs in the loss computation

        return detached_seq_ratio

    async def train_step_token_level(
        self, queries: List[str], num_groups: int = 1
    ) -> Dict[str, float]:
        """
        Execute one GSPO-token training step with token-level advantages.

        This is similar to the standard GSPO train_step but allows for
        token-specific advantages (e.g., for multi-turn RL where different
        parts of the response may have different quality).

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

            # Compute rewards for each response (sequence-level)
            # In a real multi-turn scenario, you could compute rewards per token
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

            # Compute group advantages (same as standard GSPO for this demo)
            # In practice, you could assign different advantages to different tokens
            advantages, reward_stats = self.compute_group_advantages(rewards_tensor)

            # Compute current log probs for each response and get token-level details
            current_log_probs = []
            sequence_lengths = []
            token_log_probs_list = []

            for response in responses:
                full_text = query + " " + response
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_prompt_length
                    + self.config.max_completion_length,
                ).to(self.model.device)

                # Get logits and compute token log probs
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()

                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    dim=-1, index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)

                # Store token log probs
                token_log_probs_list.append(token_log_probs)

                # Sum for sequence log prob
                sequence_log_prob = token_log_probs.sum().item()
                current_log_probs.append(sequence_log_prob)

                # Sequence length
                seq_len = shift_labels.shape[1]
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

            # Apply clipping to sequence-level importance ratios
            clipped_ratios = torch.clamp(
                importance_ratios,
                1 - self.config.clip_range_left,
                1 + self.config.clip_range_right,
            )

            # Count clipped sequences
            num_clipped = (importance_ratios != clipped_ratios).sum().item()
            total_clipped += num_clipped
            total_samples += len(responses)

            # Compute policy loss using GSPO-token objective
            # For each response, we compute token-level weighted loss

            loss = 0.0
            for i, (token_log_probs, response) in enumerate(
                zip(token_log_probs_list, responses)
            ):
                seq_len = token_log_probs.shape[1]

                # Create token-level advantages (in this demo, we use sequence advantage)
                # In practice, you could assign different advantages per token
                # For multi-turn: assign advantages based on which turn each token belongs to
                token_advantages = (
                    advantages[i].unsqueeze(0).expand(seq_len).detach()
                )

                # Detached sequence ratio for clipping (no gradients)
                detached_seq_ratio = clipped_ratios[i].detach()

                # GSPO-token objective:
                # The gradient will flow through token log probs, weighted by:
                # - Detached sequence ratio (for clipping)
                # - Token-level advantages

                # Token-level loss
                token_loss = -(
                    detached_seq_ratio * token_advantages * token_log_probs
                ).mean()

                loss += token_loss / len(responses)

            # Add KL penalty if specified
            if self.config.beta > 0 and self.ref_model is not None:
                ref_log_probs = []
                for response in responses:
                    ref_log_prob = await self._compute_ref_log_prob(query, response)
                    ref_log_probs.append(ref_log_prob)

                ref_log_probs = torch.tensor(ref_log_probs, dtype=torch.float32).to(
                    self.model.device
                )

                kl_div = (current_log_probs - ref_log_probs) / sequence_lengths
                kl_penalty = self.config.beta * kl_div.mean()

                loss += kl_penalty

            total_loss += loss

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
        avg_importance_ratio = (
            np.mean(all_importance_ratios) if all_importance_ratios else 1.0
        )

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


async def train_with_gspo_token(
    config: GSPOConfig,
    agent: Agent,
    environment: ConversationEnvironment,
    reward_model: MultiObjectiveReward,
    train_queries: Optional[List[str]] = None,
) -> Agent:
    """
    Train using GSPO-token variant with token-level advantages.

    This is useful for scenarios where different parts of the response
    should be weighted differently (e.g., multi-turn conversations where
    some turns are more important than others).

    Args:
        config: GSPO configuration
        agent: Agent to train
        environment: Training environment
        reward_model: Reward function
        train_queries: Optional list of training queries

    Returns:
        Trained agent
    """
    from .gspo_trainer import GSPOModelManager, train_with_gspo
    import json
    import os
    from datetime import datetime

    logger.info("Initializing GSPO-token training")
    logger.info(f"Configuration: {json.dumps(config.to_dict(), indent=2)}")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize wandb if configured
    if config.report_to == "wandb" and config.wandb_project:
        import wandb

        wandb.init(
            project=config.wandb_project,
            name=config.run_name
            or f"gspo-token-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config.to_dict(),
            tags=["gspo-token"] + (config.wandb_tags or []),
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
        for scenario in environment.scenarios[: config.generations_per_iteration]:
            query = scenario.get("context", "Hello")
            train_queries.append(query)

    logger.info(f"Training with {len(train_queries)} queries")

    # Create GSPO-token trainer
    trainer = GSPOTokenTrainer(
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

        # Train step with token-level advantages
        metrics = await trainer.train_step_token_level(
            queries=train_queries, num_groups=min(len(train_queries), 10)
        )

        # Log metrics
        logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

        if config.report_to == "wandb":
            import wandb

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
        import wandb

        wandb.finish()

    logger.info("✨ GSPO-token training completed successfully!")
    return agent


__all__ = [
    "GSPOTokenTrainer",
    "train_with_gspo_token",
]
