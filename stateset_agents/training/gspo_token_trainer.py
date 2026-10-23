"""
GSPO-token: Token-level variant of GSPO for fine-grained advantage adjustment

This module implements the GSPO-token variant which allows token-wise advantage
customization while maintaining sequence-level importance ratios.

Reference: https://arxiv.org/abs/2507.18071v2 (Section 4.3)
"""

import logging
from typing import Any

import torch

from stateset_agents.core.agent import Agent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.rewards.multi_objective_reward import (
    MultiObjectiveRewardFunction as MultiObjectiveReward,
)

from . import objectives, rl_losses
from .gspo_generation import (
    _get_model_device,
    build_scoring_text,
    render_prompt_for_scoring,
    scoring_prompt_length,
)
from .gspo_trainer import (
    GSPOConfig,
    GSPOTrainer,
    _require_finite_vector,
)
from .runtime_state import _reject_unsupported_resume

logger = logging.getLogger(__name__)


class GSPOTokenTrainer(GSPOTrainer):
    """
    GSPO-token trainer for token-level advantage customization.

    The key difference from standard GSPO is:
    - Allows different advantages for each token in a response
    - Uses a special importance ratio: s_{i,t}(θ) = sg[s_i(θ)] * π_θ(y_{i,t}|...) / sg[π_θ(y_{i,t}|...)]
    - This ensures clipping is still sequence-level while advantages can be token-level
    """

    _native_objective = "gspo_token"

    def __init__(
        self,
        config: GSPOConfig,
        model: Any,
        tokenizer: Any,
        agent: Agent,
        environment: ConversationEnvironment,
        reward_model: MultiObjectiveReward,
        ref_model: Any | None = None,
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

    def compute_gspo_token_loss(
        self,
        token_log_probs_list: list[torch.Tensor],
        sequence_lengths: torch.Tensor,
        importance_ratios: torch.Tensor,
        advantages: torch.Tensor,
        current_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """GSPO-token objective for one prompt group via ``objectives.policy_loss``.

        Gradients flow through the per-token log-probs; the sequence ratio is
        stop-gradient. Rows are padded to a common width with a response mask
        built from ``sequence_lengths`` counted from the end of each row (the
        gathered rows carry zeros on prompt and pad positions).
        """
        device = importance_ratios.device
        width = max(int(t.shape[-1]) for t in token_log_probs_list)
        rows = []
        mask = torch.zeros(len(token_log_probs_list), width, device=device)
        for i, row in enumerate(token_log_probs_list):
            row = row.reshape(-1)  # gathered rows may carry a batch dim of 1
            n = int(row.shape[-1])
            rows.append(torch.nn.functional.pad(row, (0, width - n)))
            start = max(n - int(sequence_lengths[i].item()), 0)
            mask[i, start:n] = 1.0
        logp_cur = torch.stack(rows)
        lengths = sequence_lengths.to(logp_cur.dtype).clamp(min=1.0)
        # Recover sequence-sum old log-probs from the (detached) ratios.
        old_sums = (
            current_log_probs.detach() - torch.log(importance_ratios.detach()) * lengths
        )
        ref = None
        if self.config.beta > 0 and ref_log_probs is not None:
            ref = ref_log_probs
        result = objectives.policy_loss(
            logp_cur=logp_cur,
            mask=mask,
            advantages=advantages,
            objective=self._objective,
            logp_old=old_sums,
            logp_ref=ref,
        )
        return result.loss

    def _score_token_rows(
        self, query: str, responses: list[str], system_prompt: str | None
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Per-token response log-probs (with gradients) for one prompt group.

        Returns the masked token log-prob rows, their per-sequence sums and the
        response lengths used for length normalisation.
        """
        model_device = _get_model_device(self.model)
        rendered_prompt = render_prompt_for_scoring(
            self.tokenizer, query, system_prompt
        )
        prompt_length = scoring_prompt_length(self.tokenizer, rendered_prompt)

        sequence_lengths_list = []
        token_log_probs_list = []

        for response in responses:
            full_text = build_scoring_text(rendered_prompt, response)
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=prompt_length + self.config.max_completion_length,
                add_special_tokens=False,
            )
            if model_device is not None and hasattr(inputs, "to"):
                inputs = inputs.to(model_device)

            # Gradients must flow through this forward pass: the
            # GSPO-token loss below differentiates through
            # `token_log_probs`.
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Shared gather (fp32 log-softmax). The response mask is
            # built on the unshifted axis: position p is a response
            # token when p >= prompt_length, which after the helper's
            # shift-by-one keeps the first response token at index
            # max(prompt_length - 1, 0) — the same convention as
            # gspo_generation._compute_sequence_log_prob.
            input_ids = inputs["input_ids"]
            response_mask = torch.zeros_like(input_ids, dtype=torch.float32)
            if prompt_length < response_mask.shape[-1]:
                response_mask[..., prompt_length:] = 1.0

            masked_token_log_probs, _ = rl_losses.gather_token_logprobs(
                logits, input_ids, response_mask
            )
            token_log_probs_list.append(masked_token_log_probs)

            response_start = max(prompt_length - 1, 0)
            response_len = max(masked_token_log_probs.shape[-1] - response_start, 1)
            sequence_lengths_list.append(float(response_len))

        # Keep tensors (not .item()) so gradients survive into the loss.
        current_log_probs = torch.stack([t.sum() for t in token_log_probs_list])
        sequence_lengths = torch.tensor(
            sequence_lengths_list, dtype=torch.float32, device=model_device
        )
        return token_log_probs_list, current_log_probs, sequence_lengths

    def _score_policy_group(
        self, prompt: str, responses: list[str]
    ) -> tuple[list[torch.Tensor] | None, torch.Tensor, torch.Tensor]:
        """Per-token rows (gradients flow through them), sums and lengths."""
        agent_config = getattr(getattr(self, "agent", None), "config", None)
        return self._score_token_rows(
            prompt, responses, getattr(agent_config, "system_prompt", None)
        )

    def _policy_group_loss(
        self,
        importance_ratios: torch.Tensor,
        advantages: torch.Tensor,
        current_log_probs: torch.Tensor,
        sequence_lengths: torch.Tensor,
        ref_log_probs: torch.Tensor | None,
        token_rows: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        """GSPO-token objective: the sequence ratio is stop-gradient (it only
        sets the clip); gradients flow through the per-token log-probs."""
        assert token_rows is not None
        return self.compute_gspo_token_loss(
            token_rows,
            sequence_lengths,
            importance_ratios.detach(),
            advantages,
            current_log_probs,
            ref_log_probs,
        )

    async def train_step_token_level(
        self, queries: list[str | dict[str, Any]], num_groups: int = 1
    ) -> dict[str, float]:
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
        self.optimizer.zero_grad(set_to_none=True)
        model_device = _get_model_device(self.model)

        all_rewards = []
        processed_groups = 0
        nonzero_advantages = 0
        advantage_count = 0
        zero_advantage_groups = 0

        scored_groups: list[
            tuple[str, list[str], torch.Tensor, torch.Tensor, torch.Tensor | None]
        ] = []
        step_queries = list(queries[:num_groups])
        # One batched generate samples every group before any scoring.
        step_groups = await self.generator.generate_groups_batch(
            [
                str(q.get("prompt", "")) if isinstance(q, dict) else str(q)
                for q in step_queries
            ],
            self.config.num_generations,
        )
        for raw_query, group_responses in zip(step_queries, step_groups, strict=True):
            processed_groups += 1
            # A query is a prompt string or {"prompt": ..., "context": {...}};
            # the context (gold answers, expected tools, ...) reaches the reward.
            query_context: dict[str, Any] = {}
            if isinstance(raw_query, dict):
                query = str(raw_query.get("prompt", ""))
                raw_context = raw_query.get("context")
                if isinstance(raw_context, dict):
                    query_context = dict(raw_context)
            else:
                query = str(raw_query)
            # Extract responses and old log probs
            responses = [resp for resp, _ in group_responses]
            old_log_probs = torch.tensor(
                [log_prob for _, log_prob in group_responses],
                dtype=torch.float32,
                device=model_device,
            )
            _require_finite_vector(old_log_probs, "GSPO rollout log probabilities")

            # Sequence-level rewards on the shared single-response path.
            rewards = [
                await self._reward_for_response(query, query_context, response)
                for response in responses
            ]

            rewards_tensor = torch.tensor(
                rewards, dtype=torch.float32, device=model_device
            )
            all_rewards.extend(rewards)

            # Compute group advantages (same as standard GSPO for this demo)
            # In practice, you could assign different advantages to different tokens
            advantages, reward_stats = self.compute_group_advantages(rewards_tensor)
            active = int(torch.count_nonzero(advantages).item())
            nonzero_advantages += active
            advantage_count += advantages.numel()
            zero_advantage_groups += int(active == 0)

            ref_log_probs = None
            if self.config.beta > 0 and self.ref_model is not None:
                ref_log_prob_values: list[float] = []
                for response in responses:
                    ref_log_prob_values.append(
                        await self._compute_ref_log_prob(query, response)
                    )
                ref_log_probs = torch.tensor(
                    ref_log_prob_values, dtype=torch.float32, device=model_device
                )
            scored_groups.append(
                (query, responses, old_log_probs, advantages, ref_log_probs)
            )

        # Match GSPO: each processed prompt group has equal objective weight;
        # the full loss, including KL, is normalised before backward.
        passes, stats = self._update_policy(
            scored_groups, processed_groups=processed_groups, device=model_device
        )
        return self._record_step_metrics(
            passes,
            stats,
            rewards=all_rewards,
            groups=processed_groups,
            nonzero_advantages=nonzero_advantages,
            advantage_count=advantage_count,
            zero_advantage_groups=zero_advantage_groups,
        )


async def train_with_gspo_token(
    config: GSPOConfig,
    agent: Agent,
    environment: ConversationEnvironment,
    reward_model: MultiObjectiveReward,
    train_queries: list[str] | None = None,
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
    _reject_unsupported_resume(config, "train_with_gspo_token")
    import json
    import os
    from datetime import datetime

    from .gspo_trainer import GSPOModelManager

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
    from .gspo_entrypoints import queries_from_scenarios, query_window

    queries: list[Any] = list(train_queries) if train_queries else []
    if not queries:
        logger.info("Generating training queries from environment scenarios...")
        queries = list(queries_from_scenarios(environment.scenarios))
    queries_per_iteration = max(1, int(config.generations_per_iteration))
    logger.info(
        "Training with %s queries (%s per iteration, rotating)",
        len(queries),
        min(queries_per_iteration, len(queries)),
    )

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
        window = query_window(queries, iteration, queries_per_iteration)
        metrics = await trainer.train_step_token_level(
            queries=window, num_groups=len(window)
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
