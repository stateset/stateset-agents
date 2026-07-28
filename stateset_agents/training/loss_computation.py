"""
GRPO loss computation functions.

This module provides the core loss computation functions for GRPO training,
including standard GRPO loss and enhanced GRPO loss with KL penalty.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from typing import Any
from collections.abc import Callable

import numpy as np

from ..exceptions import LOSS_EXCEPTIONS
from .trainer_utils import get_amp, get_functional, get_torch, require_torch

logger = logging.getLogger(__name__)

__all__ = [
    "LOSS_EXCEPTIONS",
    "compute_grpo_loss",
    "compute_enhanced_grpo_loss",
    "compute_ppo_ratio",
    "compute_entropy_bonus",
]

_warned_missing_log_probs = False
_warned_missing_assistant_mask = False


def _resolve_model_device(agent: Any, torch_mod: Any) -> Any:
    """Resolve the device for model tensors with safe CPU fallback."""
    device = getattr(agent.model, "device", None)
    if device is None and hasattr(agent.model, "parameters"):
        try:
            first_param = next(agent.model.parameters())
            device = getattr(first_param, "device", None)
        except StopIteration:
            device = None
    if device is None:
        device = torch_mod.device("cpu")
    return device


def compute_grpo_loss(
    trajectory_groups: list[Any],
    config: Any,
    agent: Any,
    global_reward_mean: float,
    global_reward_count: int,
    update_global_stats: Callable[[float, int], None],
) -> dict[str, Any]:
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
    device = _resolve_model_device(agent, torch)

    policy_losses = []
    all_advantages_for_logging: list[float] = []
    entropy_values_for_logging: list[float] = []
    entropy_coef = float(getattr(config, "entropy_coef", 0.0))

    # Configuration controls
    baseline_type = getattr(config, "baseline_type", "group_mean")
    normalize_adv = getattr(config, "advantage_normalization", True)
    reward_clip = getattr(config, "reward_clip", None)

    for group in trajectory_groups:
        if not group.trajectories:
            continue

        # Extract rewards and optionally clip
        rewards = torch.tensor(
            [t.total_reward for t in group.trajectories],
            dtype=torch.float32,
            device=device,
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
            baseline = torch.tensor(
                global_reward_mean, dtype=torch.float32, device=device
            )
        elif baseline_type == "leave_one_out":
            # Leave-one-out baseline: for trajectory i, baseline = mean
            # of all OTHER trajectories.  Lower variance than group mean.
            n = rewards.numel()
            if n > 1:
                total = rewards.sum()
                baseline = (total - rewards) / (n - 1)
            else:
                baseline = rewards.mean()
        else:  # group_mean (default)
            baseline = rewards.mean()

        advantages = rewards - baseline

        # Normalize advantages if configured and variance > 0
        if normalize_adv and advantages.numel() > 1:
            # Avoid torch warnings for small tensors by using correction=0.
            try:
                std = advantages.std(correction=0)
            except TypeError:  # Older PyTorch.
                std = advantages.std(unbiased=False)
            if torch.isfinite(std) and std > 0:
                advantages = (advantages - advantages.mean()) / (std + 1e-8)

        all_advantages_for_logging.extend(advantages.detach().cpu().tolist())

        # Compute policy loss for this group
        group_loss, group_entropy = _compute_group_policy_loss(
            group, advantages, config, agent, entropy_coef=entropy_coef
        )
        policy_losses.append(group_loss)
        if group_entropy is not None:
            entropy_values_for_logging.append(group_entropy)

    # Aggregate losses
    if policy_losses:
        total_loss_tensor = torch.stack(policy_losses).mean()
    else:
        total_loss_tensor = torch.tensor(0.0, requires_grad=True)

    # Entropy bonus — encourages exploration by penalizing overly confident
    # distributions.  Computed inside each trajectory's grad-enabled forward
    # pass (see `_compute_group_policy_loss`) so it contributes a real
    # gradient rather than being estimated under `torch.no_grad()`.
    entropy_value = (
        float(np.mean(entropy_values_for_logging))
        if entropy_values_for_logging
        else 0.0
    )

    return {
        "policy_loss": total_loss_tensor,
        "total_loss": total_loss_tensor,
        "mean_advantage": float(np.mean(all_advantages_for_logging))
        if all_advantages_for_logging
        else 0.0,
        "advantage_std": float(np.std(all_advantages_for_logging))
        if all_advantages_for_logging
        else 0.0,
        "entropy": entropy_value,
    }


def compute_ppo_ratio(
    new_log_prob_sum: Any, old_log_prob_sum: Any, token_count: int
) -> Any:
    """Length-normalized PPO importance ratio.

    Ratios computed from raw summed log-probs blow up (overflow) or vanish
    (underflow) for long responses, since the sum scales with sequence
    length. Dividing by `token_count` first yields a bounded, O(1)
    sequence-mean ratio, matching standard PPO/GRPO implementations.
    """
    torch = get_torch() or require_torch()
    return torch.exp((new_log_prob_sum - old_log_prob_sum) / max(token_count, 1))


def compute_entropy_bonus(logits: Any, response_mask: Any) -> Any:
    """Differentiable masked-mean policy entropy from logits.

    Unlike a `torch.no_grad()` estimate, this stays in the autograd graph so
    the entropy bonus actually contributes gradient to the loss it's added
    to (encouraging exploration), rather than being a dead metric.
    """
    F = get_functional()
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    # H = -sum(p * log p)
    token_entropy = -(probs * log_probs).sum(dim=-1)

    mask = response_mask
    if mask is None:
        return token_entropy.mean()
    mask = mask.to(dtype=token_entropy.dtype)
    denom = mask.sum()
    if denom.item() == 0:
        return token_entropy.mean()
    return (token_entropy * mask).sum() / denom


# NOT a drop-in replacement: `compute_entropy_bonus(logits, response_mask)`
# has a different signature than the old no-grad `_estimate_policy_entropy`
# estimator it replaces here. This alias exists only because the sole known
# external caller checks `callable(_estimate_policy_entropy)` rather than
# invoking it with the old signature — do not call this alias directly
# expecting the old API.
_estimate_policy_entropy = compute_entropy_bonus


def _compute_group_policy_loss(
    group: Any,
    advantages: Any,
    config: Any,
    agent: Any,
    entropy_coef: float = 0.0,
) -> tuple[Any, float | None]:
    """
    Compute policy loss for a single trajectory group with proper GRPO implementation.

    Args:
        group: TrajectoryGroup object
        advantages: Tensor of advantages for each trajectory
        config: Training configuration
        agent: The agent being trained
        entropy_coef: Coefficient for the (differentiable) entropy bonus

    Returns:
        Tuple of (policy loss tensor, mean entropy for logging or None)
    """
    torch = require_torch()

    device = _resolve_model_device(agent, torch)
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    num_trajectories = 0
    entropy_sum = 0.0
    entropy_count = 0

    # PPO-style clipping ratio (falls back to advantage clipping when old log probs are absent).
    clip_ratio = getattr(config, "clip_ratio", getattr(config, "clip_epsilon", 0.2))
    # Token-level loss normalization — divide loss by response token count to
    # prevent bias toward longer sequences (aligned with DAPO's approach).
    use_token_level = getattr(config, "token_level_loss", False)

    for traj_idx, (trajectory, advantage) in enumerate(
        zip(group.trajectories, advantages, strict=False)
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
                # policy loss = advantage * nll (nll = -log_prob).
                policy_loss = advantage * nll

                old_log_prob = None
                token_count = 0
                if torch.is_tensor(labels):
                    loss_mask = labels.ne(-100)
                    token_count = int(loss_mask.sum().item())

                log_probs = getattr(trajectory, "log_probs", None)
                if log_probs is None and hasattr(trajectory, "metadata"):
                    log_probs = trajectory.metadata.get("log_probs")
                if log_probs is not None:
                    if torch.is_tensor(log_probs):
                        old_log_prob = log_probs.sum().detach()
                    elif isinstance(log_probs, (list, tuple)):
                        old_log_prob = torch.tensor(
                            float(sum(log_probs)), device=device
                        )
                    elif isinstance(log_probs, (int, float)):
                        old_log_prob = torch.tensor(float(log_probs), device=device)

                new_log_prob = None
                if token_count > 0:
                    # outputs.loss is the mean per-token NLL; recover the
                    # summed log prob so the ratio helper can length-normalize
                    # consistently against `old_log_prob` (also a raw sum).
                    new_log_prob = -(outputs.loss * token_count)

                used_ratio_clipping = False
                # Optional: PPO-style clipping when old log probs are available.
                if (
                    clip_ratio > 0
                    and old_log_prob is not None
                    and new_log_prob is not None
                    and token_count > 0
                ):
                    ratio = compute_ppo_ratio(new_log_prob, old_log_prob, token_count)
                    surrogate = ratio * advantage
                    clipped = (
                        torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                        * advantage
                    )
                    policy_loss = -torch.min(surrogate, clipped)
                    used_ratio_clipping = True
                elif clip_ratio > 0:
                    global _warned_missing_log_probs
                    if not _warned_missing_log_probs:
                        logger.warning(
                            "clip_ratio set but trajectories lack log_probs; "
                            "using unclipped REINFORCE loss."
                        )
                        _warned_missing_log_probs = True
                    # Without old log_probs, PPO-style clipping is impossible.
                    # clip_ratio operates in ratio space, not advantage magnitude.
                    policy_loss = advantage * nll

                # Token-level normalization: divide by response token count
                # to avoid biasing toward longer sequences. When ratio
                # clipping was used, the length normalization already lives
                # inside the ratio (mean log-prob), so skip it here to avoid
                # double-normalizing.
                if (
                    use_token_level
                    and not used_ratio_clipping
                    and torch.is_tensor(labels)
                ):
                    _mask = labels.ne(-100)
                    _tc = int(_mask.sum().item())
                    if _tc > 1:
                        policy_loss = policy_loss / _tc

                # Differentiable entropy bonus: computed from the same
                # grad-enabled forward pass's logits, so it actually
                # contributes gradient (encourages exploration) rather than
                # being a dead no-grad metric.
                if entropy_coef > 0 and torch.is_tensor(labels):
                    response_mask = labels.ne(-100).to(dtype=outputs.logits.dtype)
                    entropy_bonus = compute_entropy_bonus(
                        outputs.logits, response_mask
                    )
                    policy_loss = policy_loss - entropy_coef * entropy_bonus
                    entropy_sum += float(entropy_bonus.detach().item())
                    entropy_count += 1

                total_loss = total_loss + policy_loss
                num_trajectories += 1

        except LOSS_EXCEPTIONS as e:
            logger.warning(
                f"Failed to compute policy loss for trajectory {traj_idx}: {e}"
            )
            continue

    mean_entropy = (entropy_sum / entropy_count) if entropy_count > 0 else None

    # Average over trajectories in the group
    if num_trajectories > 0:
        return total_loss / num_trajectories, mean_entropy
    else:
        return torch.tensor(0.0, device=device, requires_grad=True), mean_entropy


def compute_enhanced_grpo_loss(
    trajectory_groups: list[Any],
    beta: float,
    config: Any,
    agent: Any,
    reference_model: Any | None = None,
) -> dict[str, Any]:
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
    device = _resolve_model_device(agent, torch)
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
            [t.total_reward for t in group.trajectories],
            dtype=torch.float32,
            device=device,
        )

        # GRPO: Use group mean as baseline
        baseline = rewards.mean()
        advantages = rewards - baseline

        # Normalize advantages within group
        if len(advantages) > 1 and advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        all_advantages.extend(advantages.tolist())

        # Process each trajectory in the group
        for _traj_idx, (trajectory, advantage) in enumerate(
            zip(group.trajectories, advantages, strict=False)
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

                # Compute policy loss with advantage weighting.
                nll = outputs.loss
                policy_loss = advantage * nll

                # Optional PPO-style clipping, mirroring `_compute_group_policy_loss`.
                clip_ratio = getattr(
                    config, "clip_ratio", getattr(config, "clip_epsilon", 0.2)
                )
                old_log_prob = None
                log_probs_attr = getattr(trajectory, "log_probs", None)
                if log_probs_attr is None and hasattr(trajectory, "metadata"):
                    log_probs_attr = trajectory.metadata.get("log_probs")
                if log_probs_attr is not None:
                    if torch.is_tensor(log_probs_attr):
                        old_log_prob = log_probs_attr.sum().detach()
                    elif isinstance(log_probs_attr, (list, tuple)):
                        old_log_prob = torch.tensor(
                            float(sum(log_probs_attr)), device=device
                        )
                    elif isinstance(log_probs_attr, (int, float)):
                        old_log_prob = torch.tensor(
                            float(log_probs_attr), device=device
                        )

                token_count = 0
                if torch.is_tensor(labels):
                    token_count = int(labels.ne(-100).sum().item())

                if (
                    clip_ratio > 0
                    and old_log_prob is not None
                    and token_count > 0
                ):
                    new_log_prob = -(nll * token_count)
                    ratio = compute_ppo_ratio(new_log_prob, old_log_prob, token_count)
                    surrogate = ratio * advantage
                    clipped = (
                        torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                        * advantage
                    )
                    policy_loss = -torch.min(surrogate, clipped)

                all_losses.append(policy_loss)

                # Skip the full-vocab log_softmax entirely when there's no KL
                # penalty to compute — it's an expensive (batch, seq, vocab)
                # materialization that's otherwise dead weight.
                if beta > 0 and reference_model is not None:
                    log_probs = outputs.logits.log_softmax(dim=-1)
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
            kl_penalty = torch.tensor(0.0, device=device)
    else:
        total_loss = torch.tensor(0.0, requires_grad=True, device=device)
        policy_loss = total_loss
        kl_penalty = torch.tensor(0.0, device=device)

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
        rendered: object = agent.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return rendered if isinstance(rendered, str) else str(rendered)
    else:
        # Simple formatting
        parts = []
        for msg in messages:
            if msg["role"] == "user":
                parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                parts.append(f"Assistant: {msg['content']}")

        return "\n".join(parts)


def _trajectory_to_messages(trajectory: Any) -> list[dict[str, str]]:
    """Convert trajectory turns into normalized chat messages."""
    messages: list[dict[str, str]] = []
    for turn in getattr(trajectory, "turns", []):
        if isinstance(turn, dict):
            role = str(turn.get("role", "user"))
            content = str(turn.get("content", ""))
        else:
            role = str(getattr(turn, "role", None) or "user")
            content = str(getattr(turn, "content", None) or "")
        messages.append({"role": role, "content": content})
    return messages


def _format_trajectory_with_spans(
    trajectory: Any,
) -> tuple[str, list[tuple[int, int]]]:
    """Format trajectory text and track assistant spans for masking."""
    messages = _trajectory_to_messages(trajectory)
    parts: list[str] = []
    assistant_spans: list[tuple[int, int]] = []
    cursor = 0

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            prefix = "Assistant: "
        elif role == "system":
            prefix = "System: "
        elif role == "tool":
            prefix = "Tool: "
        else:
            prefix = "User: "

        segment = f"{prefix}{content}\n"
        if role == "assistant":
            assistant_spans.append((cursor, cursor + len(segment)))
        parts.append(segment)
        cursor += len(segment)

    return "".join(parts), assistant_spans


def _prepare_inputs_and_labels(
    trajectory: Any, agent: Any, config: Any
) -> tuple[dict[str, Any], Any]:
    """Prepare model inputs and loss labels, masking to assistant tokens when possible."""
    torch = get_torch() or require_torch()
    max_length = int(getattr(config, "max_prompt_length", 512)) + int(
        getattr(config, "max_completion_length", 512)
    )

    tokenizer = agent.tokenizer
    device = agent.model.device

    # Best-effort: use chat-template assistant token masking when supported.
    use_chat_template = hasattr(tokenizer, "apply_chat_template")
    if use_chat_template:
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
                        input_ids.clone() if hasattr(input_ids, "clone") else input_ids
                    )
                    if assistant_mask is not None and hasattr(labels, "masked_fill"):
                        mask = assistant_mask
                        if hasattr(mask, "dim") and mask.dim() == 1:
                            mask = mask.unsqueeze(0)
                        labels = labels.masked_fill(~mask.bool(), -100)
                    if attention_mask is not None and hasattr(labels, "masked_fill"):
                        labels = labels.masked_fill(attention_mask.eq(0), -100)

                    inputs: dict[str, Any] = {"input_ids": input_ids}
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
            except LOSS_EXCEPTIONS:
                # Any other failure: fall back to plain tokenization.
                pass
        else:
            global _warned_missing_assistant_mask
            if not _warned_missing_assistant_mask:
                logger.warning(
                    "Tokenizer chat template does not expose assistant token masks; "
                    "loss will include non-assistant tokens."
                )
                _warned_missing_assistant_mask = True

    if not use_chat_template:
        # Try assistant-only masking with offsets for non-chat-template tokenizers.
        conversation_text, assistant_spans = _format_trajectory_with_spans(trajectory)
        if assistant_spans:
            try:
                inputs = tokenizer(
                    conversation_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_offsets_mapping=True,
                )
                offsets = inputs.pop("offset_mapping", None)
                input_ids = inputs.get("input_ids")
                if offsets is not None and input_ids is not None:
                    if torch.is_tensor(offsets):
                        offsets_list = offsets[0].tolist()
                    elif isinstance(offsets, (list, tuple)):
                        offsets_list = offsets[0] if offsets else []
                    else:
                        offsets_list = []

                    if offsets_list and len(offsets_list) == input_ids.shape[1]:
                        assistant_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                        for idx, (start, end) in enumerate(offsets_list):
                            if start == end:
                                continue
                            if any(
                                start < span_end and end > span_start
                                for span_start, span_end in assistant_spans
                            ):
                                assistant_mask[0, idx] = True

                        labels = (
                            input_ids.clone()
                            if hasattr(input_ids, "clone")
                            else input_ids
                        )
                        labels = labels.masked_fill(~assistant_mask, -100)
                        if "attention_mask" in inputs and torch.is_tensor(labels):
                            labels = labels.masked_fill(
                                inputs["attention_mask"].eq(0), -100
                            )

                        inputs = {
                            k: v.to(device) if hasattr(v, "to") else v
                            for k, v in inputs.items()
                        }
                        if hasattr(labels, "to"):
                            labels = labels.to(device)
                        return inputs, labels
            except TypeError:
                pass
            except LOSS_EXCEPTIONS:
                pass

    # Fallback: compute loss over the full token stream.
    if use_chat_template:
        conversation_text = _format_trajectory_for_model(trajectory, agent)
    else:
        conversation_text, _assistant_spans = _format_trajectory_with_spans(trajectory)
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
