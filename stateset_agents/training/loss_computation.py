"""
GRPO loss computation functions.

This module provides the core loss computation functions for GRPO training,
including standard GRPO loss and enhanced GRPO loss with KL penalty.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .trainer_utils import get_amp, get_functional, get_torch, require_torch

logger = logging.getLogger(__name__)


def compute_grpo_loss(
    trajectory_groups: List[Any],
    config: Any,
    agent: Any,
    global_reward_mean: float,
    global_reward_count: int,
    update_global_stats: Callable[[float, int], None],
) -> Dict[str, Any]:
    """
    Compute GRPO loss from trajectory groups with configurable baseline and normalization.

    Args:
        trajectory_groups: List of TrajectoryGroup objects
        config: Training configuration
        agent: The agent being trained
        global_reward_mean: Running mean of rewards
        global_reward_count: Count of rewards seen
        update_global_stats: Callback to update global reward statistics

    Returns:
        Dictionary containing loss tensors and metrics
    """
    torch = require_torch()

    total_loss = 0.0
    policy_losses = []
    all_advantages_for_logging: List[float] = []

    # Configuration controls
    baseline_type = getattr(config, "baseline_type", "group_mean")
    normalize_adv = getattr(config, "advantage_normalization", True)
    reward_clip = getattr(config, "reward_clip", None)

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
                update_global_stats(batch_mean, len(rewards))
            baseline = torch.tensor(global_reward_mean, dtype=torch.float32)
        else:  # group_mean (default)
            baseline = rewards.mean()

        advantages = rewards - baseline

        # Normalize advantages if configured and variance > 0
        if normalize_adv and len(advantages) > 1:
            std = advantages.std()
            if torch.isfinite(std) and std > 0:
                advantages = (advantages - advantages.mean()) / (std + 1e-8)

        all_advantages_for_logging.extend(advantages.detach().cpu().tolist())

        # Compute policy loss for this group
        group_loss = _compute_group_policy_loss(group, advantages, config, agent)
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
    group: Any,
    advantages: Any,
    config: Any,
    agent: Any,
) -> Any:
    """
    Compute policy loss for a single trajectory group with proper GRPO implementation.

    Args:
        group: TrajectoryGroup object
        advantages: Tensor of advantages for each trajectory
        config: Training configuration
        agent: The agent being trained

    Returns:
        Policy loss tensor
    """
    torch = require_torch()
    F = get_functional()

    device = agent.model.device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    num_trajectories = 0

    # PPO clipping ratio
    clip_epsilon = getattr(
        config, "clip_ratio", getattr(config, "clip_epsilon", 0.2)
    )

    for traj_idx, (trajectory, advantage) in enumerate(
        zip(group.trajectories, advantages)
    ):
        try:
            inputs, labels = _prepare_inputs_and_labels(trajectory, agent, config)

            # Forward pass to get log probabilities
            with torch.set_grad_enabled(True):
                outputs = agent.model(**inputs, labels=labels)

                # Get the negative log likelihood (this is the loss from the model)
                nll = outputs.loss

                # GRPO policy gradient: -advantage * log_prob
                # Since outputs.loss is already negative log likelihood,
                # For GRPO, we want to maximize reward-weighted log prob
                # policy loss = -advantage * log_prob = advantage * nll (since nll = -log_prob)
                policy_loss = advantage * nll

                # Optional: PPO-style clipping for stability
                # Note: In PPO, we use min() not max() to be conservative
                # and clip the advantage (surrogate objective) for stability
                if clip_epsilon > 0:
                    # Clamp the advantage to prevent extremely large updates
                    clamped_advantage = advantage.clamp(-clip_epsilon, clip_epsilon)
                    clipped_loss = clamped_advantage * nll
                    # Use min for conservative updates (take the more pessimistic estimate)
                    # When advantage > 0: min ensures we don't over-incentivize
                    # When advantage < 0: min ensures we don't over-penalize
                    policy_loss = torch.min(policy_loss, clipped_loss)

                total_loss = total_loss + policy_loss
                num_trajectories += 1

        except Exception as e:
            logger.warning(
                f"Failed to compute policy loss for trajectory {traj_idx}: {e}"
            )
            continue

    # Average over trajectories in the group
    if num_trajectories > 0:
        return total_loss / num_trajectories
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def compute_enhanced_grpo_loss(
    trajectory_groups: List[Any],
    beta: float,
    config: Any,
    agent: Any,
    reference_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Enhanced GRPO loss computation with KL penalty and proper advantages.

    Args:
        trajectory_groups: List of TrajectoryGroup objects
        beta: KL penalty coefficient
        config: Training configuration
        agent: The agent being trained
        reference_model: Optional reference model for KL computation

    Returns:
        Dictionary containing loss tensors and metrics
    """
    torch = require_torch()
    F = get_functional()
    amp = get_amp()

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
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        all_advantages.extend(advantages.tolist())

        # Process each trajectory in the group
        for traj_idx, (trajectory, advantage) in enumerate(
            zip(group.trajectories, advantages)
        ):
            inputs, labels = _prepare_inputs_and_labels(trajectory, agent, config)

            # Forward pass
            use_amp = getattr(config, "bf16", False) or getattr(config, "fp16", False)
            if amp is not None:
                autocast_ctx = amp.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                    enabled=bool(use_amp),
                )
            else:  # pragma: no cover
                autocast_ctx = contextlib.nullcontext()

            with autocast_ctx:
                outputs = agent.model(**inputs, labels=labels)

                # Get log probabilities
                log_probs = outputs.logits.log_softmax(dim=-1)

                # Compute policy loss with advantage weighting
                policy_loss = advantage * outputs.loss
                all_losses.append(policy_loss)

                # KL divergence penalty (if beta > 0)
                if beta > 0 and reference_model is not None:
                    with torch.no_grad():
                        ref_outputs = reference_model(**inputs)
                        ref_log_probs = ref_outputs.logits.log_softmax(dim=-1)

                    # Token-wise KL(p || p_ref) = sum_v p(v) * (log p(v) - log p_ref(v))
                    kl_per_token = (log_probs.exp() * (log_probs - ref_log_probs)).sum(
                        dim=-1
                    )

                    # Prefer masking to the same tokens contributing to the LM loss.
                    if torch.is_tensor(labels):
                        loss_mask = labels.ne(-100)
                        if loss_mask.any():
                            kl_div = (kl_per_token * loss_mask).sum() / loss_mask.sum()
                        else:
                            kl_div = kl_per_token.mean()
                    else:  # pragma: no cover - non-tensor labels (stub tokenizer)
                        kl_div = kl_per_token.mean()
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
            kl_penalty = torch.tensor(0.0, device=agent.model.device)
    else:
        total_loss = torch.tensor(
            0.0, requires_grad=True, device=agent.model.device
        )
        policy_loss = total_loss
        kl_penalty = torch.tensor(0.0, device=agent.model.device)

    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "kl_penalty": kl_penalty,
        "mean_advantage": np.mean(all_advantages) if all_advantages else 0.0,
        "advantage_std": np.std(all_advantages) if all_advantages else 0.0,
        "num_trajectories": sum(len(g.trajectories) for g in trajectory_groups),
    }


def _format_trajectory_for_model(trajectory: Any, agent: Any) -> str:
    """Format trajectory into text for model input."""
    messages = _trajectory_to_messages(trajectory)

    if hasattr(agent.tokenizer, "apply_chat_template"):
        # Use tokenizer's chat template
        return agent.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    else:
        # Simple formatting
        parts = []
        for msg in messages:
            if msg["role"] == "user":
                parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                parts.append(f"Assistant: {msg['content']}")

        return "\n".join(parts)


def _trajectory_to_messages(trajectory: Any) -> List[Dict[str, str]]:
    """Convert trajectory turns into normalized chat messages."""
    messages: List[Dict[str, str]] = []
    for turn in getattr(trajectory, "turns", []):
        if isinstance(turn, dict):
            role = str(turn.get("role", "user"))
            content = str(turn.get("content", ""))
        else:
            role = str(getattr(turn, "role", None) or "user")
            content = str(getattr(turn, "content", None) or "")
        messages.append({"role": role, "content": content})
    return messages


def _prepare_inputs_and_labels(
    trajectory: Any, agent: Any, config: Any
) -> Tuple[Dict[str, Any], Any]:
    """Prepare model inputs and loss labels, masking to assistant tokens when possible."""
    torch = get_torch() or require_torch()
    max_length = int(getattr(config, "max_prompt_length", 512)) + int(
        getattr(config, "max_completion_length", 512)
    )

    tokenizer = agent.tokenizer
    device = agent.model.device

    # Best-effort: use chat-template assistant token masking when supported.
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            params = inspect.signature(tokenizer.apply_chat_template).parameters
            supports_mask = (
                "return_dict" in params and "return_assistant_tokens_mask" in params
            )
        except (TypeError, ValueError):
            supports_mask = False

        if supports_mask:
            messages = _trajectory_to_messages(trajectory)
            try:
                out = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                    return_dict=True,
                    return_assistant_tokens_mask=True,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                )
                if isinstance(out, dict) and "input_ids" in out:
                    input_ids = out["input_ids"]
                    attention_mask = out.get("attention_mask")
                    assistant_mask = out.get("assistant_tokens_mask")

                    labels = (
                        input_ids.clone()
                        if hasattr(input_ids, "clone")
                        else input_ids
                    )
                    if assistant_mask is not None and hasattr(labels, "masked_fill"):
                        mask = assistant_mask
                        if hasattr(mask, "dim") and mask.dim() == 1:
                            mask = mask.unsqueeze(0)
                        labels = labels.masked_fill(~mask.bool(), -100)
                    if attention_mask is not None and hasattr(labels, "masked_fill"):
                        labels = labels.masked_fill(attention_mask.eq(0), -100)

                    inputs: Dict[str, Any] = {"input_ids": input_ids}
                    if attention_mask is not None:
                        inputs["attention_mask"] = attention_mask

                    inputs = {
                        k: v.to(device) if hasattr(v, "to") else v
                        for k, v in inputs.items()
                    }
                    if hasattr(labels, "to"):
                        labels = labels.to(device)
                    return inputs, labels
            except TypeError:
                # Tokenizer does not accept one of the requested kwargs.
                pass
            except Exception:
                # Any other failure: fall back to plain tokenization.
                pass

    # Fallback: compute loss over the full token stream.
    conversation_text = _format_trajectory_for_model(trajectory, agent)
    inputs = tokenizer(
        conversation_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    input_ids = inputs.get("input_ids")
    labels = input_ids.clone() if hasattr(input_ids, "clone") else input_ids
    if "attention_mask" in inputs and torch.is_tensor(labels):
        labels = labels.masked_fill(inputs["attention_mask"].eq(0), -100)

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    if hasattr(labels, "to"):
        labels = labels.to(device)
    return inputs, labels
