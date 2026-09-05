"""
Value-Augmented Policy Optimization (VAPO) Training for StateSet Agents

VAPO is an advanced PPO variant that addresses key weaknesses in value-based RL
for long chain-of-thought reasoning. It achieves state-of-the-art results
(60.4 on AIME 2024) through seven key modifications:

1. Value network warmup (50 steps)
2. Decoupled GAE computation (separate lambda for critic/policy)
3. Length-adaptive lambda for policy
4. Asymmetric clipping (Clip-Higher)
5. Token-level loss normalization
6. Positive example LM loss addition
7. Group-sampling strategy

Reference: https://arxiv.org/abs/2504.05118
"""

import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.rust_accelerator import compute_gae as _rust_compute_gae
from ..core.rust_accelerator import is_rust_available as _rust_gae_available
from . import objectives, rl_losses
from .checkpoint_io import load_checkpoint_file
from .trainer_runtime import (
    SharedModelManager,
    build_group_batch,
    hf_generate_group,
    save_checkpoint_artifacts,
)
from .vapo_config import VAPOConfig

logger = logging.getLogger(__name__)

try:
    import wandb as _wandb

    wandb: Any = _wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

try:
    from peft import LoraConfig as _LoraConfig
    from peft import TaskType as _TaskType
    from peft import get_peft_model as _get_peft_model

    LoraConfig: Any = _LoraConfig
    TaskType: Any = _TaskType
    get_peft_model: Any = _get_peft_model
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None
    TaskType = None
    get_peft_model = None

# Lazy import transformers to avoid torch/torchvision compatibility issues
_transformers_vapo_loaded = False
AutoModelForCausalLM: Any | None = None
AutoTokenizer: Any | None = None
get_cosine_schedule_with_warmup: Any | None = None


def _load_transformers_vapo() -> bool:
    """Lazily load transformers to avoid import-time errors."""
    global _transformers_vapo_loaded, AutoModelForCausalLM, AutoTokenizer
    global get_cosine_schedule_with_warmup
    if _transformers_vapo_loaded:
        return True
    # Allow pre-injected mocks without importing transformers.
    if AutoModelForCausalLM is not None and AutoTokenizer is not None:
        _transformers_vapo_loaded = True
        return True
    try:
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer
        from transformers import get_cosine_schedule_with_warmup as _get_cosine

        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
        get_cosine_schedule_with_warmup = _get_cosine
        _transformers_vapo_loaded = True
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Failed to load transformers: {e}")
        return False


def _require_transformers_vapo() -> None:
    """Ensure transformers components are available before model loading."""
    if not _load_transformers_vapo():
        raise ImportError(
            "transformers is required for VAPO training. "
            "Install with `pip install stateset-agents[training]` or `pip install transformers`."
        )


def _require_peft() -> None:
    """Ensure PEFT is available before using LoRA features."""
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError(
            "PEFT is required for VAPO LoRA training. "
            "Install with `pip install stateset-agents[training]` or `pip install peft`."
        )


def _require_wandb() -> None:
    """Ensure Weights & Biases is available before logging."""
    if wandb is None:
        raise ImportError(
            "wandb is required for VAPO logging. "
            "Install with `pip install stateset-agents[training]` or `pip install wandb`."
        )


class VAPOModelManager(SharedModelManager):
    """Manages model loading for VAPO training"""

    def __init__(self, config: VAPOConfig):
        super().__init__(config)

    def _get_transformers(self) -> tuple[Any, Any]:
        _require_transformers_vapo()
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("transformers is required for VAPO training")
        return AutoTokenizer, AutoModelForCausalLM

    def _peft_components(self) -> tuple[Any, Any, Any]:
        _require_peft()
        if get_peft_model is None:
            raise ImportError("PEFT is required for VAPO LoRA training")
        return LoraConfig, TaskType, get_peft_model


class ValueHead(nn.Module):
    """
    Value head network for VAPO.

    Predicts state values for advantage estimation.
    Uses a simple MLP on top of the language model hidden states.
    """

    def __init__(
        self,
        hidden_size: int,
        value_hidden_size: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_size = hidden_size

        for _i in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_size, value_hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_size = value_hidden_size

        # Final layer outputs scalar value
        layers.append(nn.Linear(in_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute values from hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            values: [batch, seq_len, 1]
        """
        return self.network(hidden_states)


class LengthAdaptiveGAE:
    """
    Implements VAPO's Length-Adaptive GAE.

    Uses different lambda values for critic (lambda=1.0 for unbiased)
    and policy (adaptive based on sequence length).

    lambda_policy = 1 - 1/(alpha * length)

    This ensures the sum of GAE coefficients is proportional to output length,
    balancing bias-variance across variable-length sequences.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        lambda_critic: float = 1.0,
        lambda_policy_alpha: float = 0.05,
    ):
        self.gamma = gamma
        self.lambda_critic = lambda_critic
        self.lambda_policy_alpha = lambda_policy_alpha

    def compute_lambda_policy(self, sequence_length: int) -> float:
        """
        Compute length-adaptive lambda for policy.

        lambda = 1 - 1/(alpha * length + 1)

        This ensures lambda is in (0, 1) and increases with sequence length.
        """
        return 1.0 - 1.0 / (self.lambda_policy_alpha * sequence_length + 1.0)

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        lambda_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: [batch, seq_len] rewards at each step
            values: [batch, seq_len] value predictions
            dones: [batch, seq_len] episode termination flags
            lambda_value: GAE lambda parameter

        Returns:
            advantages: [batch, seq_len]
            returns: [batch, seq_len] (advantages + values)
        """
        batch_size, seq_len = rewards.shape
        device = rewards.device

        rust_advantages = self._try_rust_gae(rewards, values, dones, lambda_value)
        if rust_advantages is not None:
            returns = rust_advantages + values
            return rust_advantages, returns

        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(batch_size, device=device)

        # Compute GAE backwards
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = torch.zeros(batch_size, device=device)
            else:
                next_value = values[:, t + 1]

            delta = (
                rewards[:, t]
                + self.gamma * next_value * (1 - dones[:, t])
                - values[:, t]
            )
            last_gae = delta + self.gamma * lambda_value * (1 - dones[:, t]) * last_gae
            advantages[:, t] = last_gae

        returns = advantages + values

        return advantages, returns

    def _try_rust_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        lambda_value: float,
    ) -> torch.Tensor | None:
        """Optional fast path using the Rust ``compute_gae`` kernel.

        Only used when every row has at most one termination step (the
        common single-episode-per-response case used by VAPO). Rows are
        truncated at their terminal step before delegating to the Rust
        kernel (a single termination at the end of a sequence is
        equivalent to simply not bootstrapping past it), and any padding
        after termination is left as zero advantage. Returns ``None`` to
        signal the caller should fall back to the pure-torch loop, either
        because the Rust extension isn't installed or a row has more than
        one termination flag (a case the vectorized kernel can't express).
        """
        if not _rust_gae_available():
            return None

        batch_size, seq_len = rewards.shape
        rewards_np = rewards.detach().cpu().numpy()
        values_np = values.detach().cpu().numpy()
        dones_np = dones.detach().cpu().numpy()

        advantages_np = np.zeros((batch_size, seq_len), dtype=np.float64)
        for i in range(batch_size):
            done_indices = np.nonzero(dones_np[i])[0]
            if len(done_indices) > 1:
                # Multiple terminations within one row aren't representable
                # by the plain (no-dones) GAE kernel; bail out entirely so
                # the whole batch uses the torch fallback for consistency.
                return None
            end = int(done_indices[0]) + 1 if len(done_indices) else seq_len
            row_advantages = _rust_compute_gae(
                rewards_np[i, :end], values_np[i, :end], self.gamma, lambda_value
            )
            advantages_np[i, :end] = row_advantages

        return torch.as_tensor(
            advantages_np, dtype=rewards.dtype, device=rewards.device
        )

    def compute_decoupled_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute decoupled GAE for VAPO.

        Returns separate advantages for critic (lambda=1) and policy (length-adaptive).

        Args:
            rewards: [batch, seq_len]
            values: [batch, seq_len]
            dones: [batch, seq_len]
            sequence_lengths: [batch] length of each sequence

        Returns:
            critic_advantages: For value function training (lambda=1)
            policy_advantages: For policy training (length-adaptive lambda)
            returns: Value targets
        """
        # Critic GAE (unbiased, lambda=1)
        critic_advantages, returns = self.compute_gae(
            rewards, values, dones, self.lambda_critic
        )

        # Policy GAE (length-adaptive)
        batch_size = rewards.shape[0]
        policy_advantages = torch.zeros_like(rewards)

        for i in range(batch_size):
            seq_len = int(sequence_lengths[i].item())
            lambda_policy = self.compute_lambda_policy(seq_len)

            # Compute per-sample GAE with adaptive lambda
            single_rewards = rewards[i : i + 1]
            single_values = values[i : i + 1]
            single_dones = dones[i : i + 1]

            adv, _ = self.compute_gae(
                single_rewards, single_values, single_dones, lambda_policy
            )
            policy_advantages[i] = adv[0]

        return critic_advantages, policy_advantages, returns


class VAPOTrainer:
    """
    Value-Augmented Policy Optimization (VAPO) Trainer

    VAPO achieves 60.4 on AIME 2024 (SOTA) through seven key modifications to PPO:

    1. Value warmup: Pretrain value network for 50 steps
    2. Decoupled GAE: Separate lambda for critic (1.0) and policy (adaptive)
    3. Length-adaptive lambda: lambda = 1 - 1/(alpha * length)
    4. Clip-Higher: Asymmetric clipping [1-0.2, 1+0.28]
    5. Token-level loss: Normalize by total tokens
    6. Positive LM loss: Add NLL on correct samples
    7. Group sampling: More samples per prompt, fewer prompts

    Reference: https://arxiv.org/abs/2504.05118
    """

    def __init__(
        self,
        config: VAPOConfig,
        model: Any,
        tokenizer: Any,
        reward_fn: Callable[[str, str], float | Awaitable[float]],
        verifier_fn: Callable[[str, str], bool] | None = None,
    ):
        # Ensure transformers is loaded for scheduler
        _load_transformers_vapo()

        self.config = config
        # Parsed once here rather than per forward pass.
        self._logprob_dtype = rl_losses.resolve_logprob_dtype(
            getattr(config, "logprob_dtype", None)
        )
        # Policy half of VAPO: Clip-Higher surrogate on external (GAE)
        # per-token advantages; value and positive-LM losses stay local.
        self._objective = objectives.resolve_objective(
            config,
            "ppo",
            max_completion_length=int(config.max_completion_length),
            supported_ratios=("token", "sequence", "sequence_token"),
            name="vapo",
            clip_low=float(config.clip_eps_low),
            clip_high=float(config.clip_eps_high),
            aggregate="token_mean" if config.use_token_level_loss else "seq_mean",
            kl="none",
        )
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.verifier_fn = verifier_fn
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        # Get hidden size from model config
        if hasattr(model, "config"):
            hidden_size = getattr(model.config, "hidden_size", 768)
        else:
            hidden_size = 768

        # Initialize value head
        self.value_head = ValueHead(
            hidden_size=hidden_size,
            value_hidden_size=config.value_hidden_size,
            num_layers=config.value_num_layers,
        ).to(self.device)

        # Initialize GAE computer
        self.gae_computer = LengthAdaptiveGAE(
            gamma=0.99,
            lambda_critic=config.lambda_critic,
            lambda_policy_alpha=config.lambda_policy_alpha,
        )

        # Separate optimizers for actor and critic
        params = list(self.model.parameters())
        if not params:
            self._stub_param = torch.nn.Parameter(torch.zeros(1))
            params = [self._stub_param]
        self.actor_optimizer = torch.optim.AdamW(
            params,
            lr=config.actor_learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )

        self.critic_optimizer = torch.optim.AdamW(
            self.value_head.parameters(),
            lr=config.critic_learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )

        # Schedulers
        total_steps = config.num_episodes * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)

        if get_cosine_schedule_with_warmup is not None:
            self.actor_scheduler = get_cosine_schedule_with_warmup(
                self.actor_optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

            self.critic_scheduler = get_cosine_schedule_with_warmup(
                self.critic_optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            # Fallback to constant learning rate if scheduler unavailable
            self.actor_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.actor_optimizer, factor=1.0, total_iters=total_steps
            )
            self.critic_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.critic_optimizer, factor=1.0, total_iters=total_steps
            )

        # Metrics
        self.metrics_history: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "positive_lm_loss": [],
            "average_reward": [],
            "accuracy": [],
            "explained_variance": [],
        }

        self.global_step = 0
        self.value_warmup_complete = False
        self.rollout_samples_total = 0

    async def _compute_reward(self, prompt: str, response: str) -> float:
        """Resolve sync or async reward callbacks to a float."""
        reward = self.reward_fn(prompt, response)
        if isinstance(reward, Awaitable):
            reward = await reward
        return float(reward)

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract hidden states from model"""
        with torch.set_grad_enabled(self.model.training):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Use last hidden state
            hidden_states = outputs.hidden_states[-1]
        return hidden_states

    def compute_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute values for a sequence"""
        hidden_states = self.get_hidden_states(input_ids, attention_mask)
        # Keep the critic in fp32 for stable regression while allowing the
        # policy backbone to run in bf16/fp16.  Linear layers require matching
        # dtypes; an explicit differentiable cast avoids the live CUDA
        # ``BFloat16 and Float`` matmul failure without severing actor grads.
        value_dtype = next(self.value_head.parameters()).dtype
        values = self.value_head(hidden_states.to(dtype=value_dtype))
        return values.squeeze(-1)

    def compute_token_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-token log probabilities.

        ``response_mask`` (optional, for backward compatibility) zeroes
        prompt positions up front. Every consumer masks to response tokens
        anyway, so the losses are identical — passing the real mask just
        avoids carrying scores that are guaranteed to be discarded.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if response_mask is None:
            response_mask = torch.ones_like(input_ids)
        token_log_probs, _ = rl_losses.gather_token_logprobs(
            outputs.logits, input_ids, response_mask, dtype=self._logprob_dtype
        )
        return token_log_probs

    async def generate_group_responses(
        self,
        prompt: str,
    ) -> list[dict[str, Any]]:
        """Generate a group of responses for VAPO"""
        responses: list[dict[str, Any]] = await hf_generate_group(
            self.model,
            self.tokenizer,
            self.config,
            self.device,
            prompt,
            self.config.group_size,
        )
        self.rollout_samples_total += len(responses)
        return responses

    async def warmup_value_network(
        self,
        prompts: list[str],
    ) -> dict[str, float]:
        """
        Pretrain value network with Monte-Carlo returns.

        This mitigates value initialization bias by training the value
        network to predict actual returns before joint training.
        """
        logger.info(
            f"Starting value network warmup ({self.config.value_warmup_steps} steps)"
        )

        total_value_loss = 0.0
        explained_variances = []

        for step in range(self.config.value_warmup_steps):
            # Sample prompt
            prompt = np.random.choice(prompts)

            # Generate responses
            responses = await self.generate_group_responses(prompt)

            if len(responses) == 0:
                continue

            # Compute rewards (Monte-Carlo returns)
            rewards = []
            for resp in responses:
                reward = await self._compute_reward(prompt, resp["response"])
                rewards.append(reward)

            # Prepare batch
            batch_size = len(responses)
            (
                batch_input_ids,
                batch_attention_mask,
                batch_response_mask,
            ) = build_group_batch(responses, self.device)
            assert batch_response_mask is not None
            max_len = batch_input_ids.shape[1]

            # Monte-Carlo return targets (same reward for all tokens in response)
            rewards_tensor = torch.tensor(rewards, device=self.device)
            mc_returns = rewards_tensor.unsqueeze(1).expand(batch_size, max_len)
            mc_returns = mc_returns * batch_response_mask

            # Compute values
            with torch.no_grad():
                hidden_states = self.get_hidden_states(
                    batch_input_ids, batch_attention_mask
                )

            value_dtype = next(self.value_head.parameters()).dtype
            values = self.value_head(hidden_states.to(dtype=value_dtype)).squeeze(-1)

            # Value loss (MSE)
            value_loss = F.mse_loss(
                values * batch_response_mask, mc_returns, reduction="sum"
            ) / batch_response_mask.sum().clamp(min=1)

            # Update value network
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.value_head.parameters(), self.config.max_grad_norm
            )
            self.critic_optimizer.step()

            total_value_loss += value_loss.item()

            # Compute explained variance
            with torch.no_grad():
                var_returns = mc_returns[batch_response_mask > 0].var()
                var_residual = (mc_returns - values)[batch_response_mask > 0].var()
                if var_returns > 1e-8:
                    explained_var = 1 - var_residual / var_returns
                    explained_variances.append(explained_var.item())

            if (step + 1) % 10 == 0:
                avg_loss = total_value_loss / (step + 1)
                avg_ev = (
                    np.mean(explained_variances[-10:]) if explained_variances else 0
                )
                logger.info(
                    f"Value warmup step {step + 1}/{self.config.value_warmup_steps} | "
                    f"Loss: {avg_loss:.4f} | EV: {avg_ev:.4f}"
                )

        self.value_warmup_complete = True
        logger.info("Value network warmup complete")

        return {
            "warmup_value_loss": total_value_loss / self.config.value_warmup_steps,
            "warmup_explained_variance": (
                float(np.mean(explained_variances)) if explained_variances else 0
            ),
        }

    def compute_value_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Value loss (MSE with optional PPO-style clipping).

        Clipping compares fresh ``values`` against the *rollout-time* value
        predictions (``old_values``), not against a detached copy of
        ``values`` itself (which would make the clipped and unclipped
        branches identical).
        """
        value_pred = values * response_mask
        value_target = returns * response_mask

        if self.config.value_clip > 0:
            clipped_values = old_values + torch.clamp(
                values - old_values,
                -self.config.value_clip,
                self.config.value_clip,
            )
            value_loss_unclipped = (value_pred - value_target) ** 2
            value_loss_clipped = (clipped_values * response_mask - value_target) ** 2
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
        else:
            value_loss = (value_pred - value_target) ** 2

        return value_loss.sum() / response_mask.sum().clamp(min=1)

    def build_token_rewards(
        self,
        scalar_reward: float | torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Place a scalar (episode-level) reward on the terminal response token
        of each row only, instead of broadcasting it across every response
        token (which inflates GAE returns).

        ``scalar_reward`` may be a python float (applied to every row) or a
        1-D tensor of per-row rewards.
        """
        if not isinstance(scalar_reward, torch.Tensor):
            scalar_reward = torch.tensor(
                scalar_reward, dtype=response_mask.dtype, device=response_mask.device
            )
        scalar_reward = scalar_reward.to(
            dtype=response_mask.dtype, device=response_mask.device
        )
        if scalar_reward.dim() == 0:
            scalar_reward = scalar_reward.expand(response_mask.shape[0])

        batch_size, seq_len = response_mask.shape
        rewards = torch.zeros_like(response_mask)
        mask_bool = response_mask > 0

        idx = (
            torch.arange(seq_len, device=response_mask.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        last_idx = (
            torch.where(mask_bool, idx, torch.full_like(idx, -1)).max(dim=1).values
        )
        valid = last_idx >= 0
        rows = torch.arange(batch_size, device=response_mask.device)[valid]
        rewards[rows, last_idx[valid]] = scalar_reward[valid]

        return rewards

    def compute_vapo_losses(
        self,
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        policy_advantages: torch.Tensor,
        critic_advantages: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        response_mask: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAPO losses:
        1. Policy loss with Clip-Higher and token-level normalization
        2. Value loss (decoupled GAE: value target from critic-lambda GAE)
        3. Positive example LM loss
        """
        # Clip-Higher surrogate and token-/sequence-level aggregation via the
        # shared objective (ratios are clamped before exp inside it).
        policy_result = objectives.policy_loss(
            logp_cur=current_log_probs,
            mask=response_mask,
            advantages=policy_advantages,
            objective=self._objective,
            logp_old=old_log_probs,
        )
        policy_loss = policy_result.loss

        # Decoupled GAE: value target is built from the critic-lambda
        # advantages, not the policy-lambda advantages used above.
        returns = critic_advantages + old_values
        value_loss = self.compute_value_loss(values, old_values, returns, response_mask)

        # Positive example LM loss
        if self.config.use_positive_lm_loss and positive_mask.sum() > 0:
            # NLL loss on correct samples
            positive_log_probs = current_log_probs * positive_mask
            positive_lm_loss = -positive_log_probs.sum() / positive_mask.sum().clamp(
                min=1
            )
        else:
            positive_lm_loss = torch.tensor(0.0, device=self.device)

        return policy_loss, value_loss, positive_lm_loss

    async def train_step(
        self,
        prompts: list[str],
    ) -> dict[str, float]:
        """
        Execute one VAPO training step.

        1. Generate responses with group sampling
        2. Compute rewards and verify correctness
        3. Compute values and decoupled GAE
        4. Update policy and value networks
        """
        self.model.train()
        self.value_head.train()

        # Warmup value network if not done
        if not self.value_warmup_complete:
            warmup_metrics = await self.warmup_value_network(prompts)
            return warmup_metrics

        # Per-update metrics are (re)initialised inside the update loop below,
        # which always runs at least once; rollout-level metrics accumulate
        # across the whole step.
        all_rewards: list[float] = []
        all_accuracies: list[float] = []

        # --- Rollout phase --------------------------------------------------
        # Generate every group and freeze the old-policy log probs and values
        # here, before any inner update moves the weights. Recomputing them
        # from the current model made the importance ratio identically 1 and
        # the Clip-Higher trust region a no-op.
        rollouts: list[dict[str, Any]] = []

        for prompt in prompts[: self.config.per_device_train_batch_size]:
            # Generate group responses
            responses = await self.generate_group_responses(prompt)

            if len(responses) < 2:
                continue

            # Compute rewards and identify correct samples
            rewards = []
            is_correct = []

            for resp in responses:
                reward = await self._compute_reward(prompt, resp["response"])
                rewards.append(reward)

                if self.verifier_fn:
                    correct = self.verifier_fn(prompt, resp["response"])
                else:
                    correct = reward > 0.5
                is_correct.append(correct)

            accuracy = sum(is_correct) / len(is_correct)
            all_accuracies.append(accuracy)
            all_rewards.extend(rewards)

            # Prepare batch
            batch_size = len(responses)
            (
                batch_input_ids,
                batch_attention_mask,
                batch_response_mask,
            ) = build_group_batch(responses, self.device)
            assert batch_response_mask is not None
            max_len = batch_input_ids.shape[1]

            sequence_lengths = torch.zeros(batch_size, device=self.device)
            for i, resp in enumerate(responses):
                sequence_lengths[i] = resp["sequence_length"]

            # Positive mask for LM loss (correct samples only)
            positive_mask = torch.zeros(batch_size, max_len - 1, device=self.device)
            for i, correct in enumerate(is_correct):
                if correct:
                    # Mask response tokens for correct samples
                    resp_mask = batch_response_mask[i, 1:]  # Shift for next-token
                    positive_mask[i] = resp_mask

            # Get old log probs and values
            with torch.no_grad():
                old_log_probs = self.compute_token_log_probs(
                    batch_input_ids, batch_attention_mask, batch_response_mask
                )
                old_values = self.compute_values(batch_input_ids, batch_attention_mask)

            # Create reward tensor: place each episode reward on the terminal
            # response token only (broadcasting it across every response
            # token would inflate GAE returns).
            rewards_tensor = torch.tensor(rewards, device=self.device)
            reward_sequence = self.build_token_rewards(
                rewards_tensor, batch_response_mask
            )

            # Terminal mask (1 at last token of each response)
            dones = torch.zeros(batch_size, max_len, device=self.device)
            for i, resp in enumerate(responses):
                last_idx = resp["prompt_length"] + resp["sequence_length"] - 1
                if last_idx < max_len:
                    dones[i, last_idx] = 1.0

            # Compute decoupled GAE
            (
                critic_advantages,
                policy_advantages,
                returns,
            ) = self.gae_computer.compute_decoupled_gae(
                reward_sequence, old_values, dones, sequence_lengths
            )

            # Normalize advantages
            policy_adv_masked = policy_advantages[batch_response_mask > 0]
            if len(policy_adv_masked) > 1:
                policy_advantages = (policy_advantages - policy_adv_masked.mean()) / (
                    policy_adv_masked.std() + 1e-8
                )

            rollouts.append(
                {
                    "input_ids": batch_input_ids,
                    "attention_mask": batch_attention_mask,
                    "response_mask": batch_response_mask,
                    "positive_mask": positive_mask,
                    "old_log_probs": old_log_probs.detach(),
                    "old_values": old_values.detach(),
                    "critic_advantages": critic_advantages,
                    "policy_advantages": policy_advantages,
                    "returns": returns,
                }
            )

        # --- Update phase ---------------------------------------------------
        # ``num_gradient_updates`` (mu) inner updates reuse the same rollouts
        # against the frozen old log probs/values, exactly like DAPO. With the
        # default of 1 the formula is identical to the previous single-update
        # path (only dropout RNG ordering differs).
        num_updates = max(1, getattr(self.config, "num_gradient_updates", 1))

        for _ in range(num_updates):
            prompt_count = len(rollouts)
            all_policy_losses: list[float] = []
            all_value_losses: list[float] = []
            all_positive_lm_losses: list[float] = []
            all_explained_variances: list[float] = []

            if prompt_count > 0:
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

            for rollout in rollouts:
                batch_input_ids = rollout["input_ids"]
                batch_attention_mask = rollout["attention_mask"]
                batch_response_mask = rollout["response_mask"]

                # Compute current log probs and values
                current_log_probs = self.compute_token_log_probs(
                    batch_input_ids, batch_attention_mask, batch_response_mask
                )
                current_values = self.compute_values(
                    batch_input_ids, batch_attention_mask
                )

                # Shift masks for loss computation
                shifted_response_mask = batch_response_mask[:, 1:]
                shifted_policy_adv = rollout["policy_advantages"][:, :-1]
                shifted_critic_adv = rollout["critic_advantages"][:, :-1]
                shifted_old_values = rollout["old_values"][:, :-1]
                shifted_values = current_values[:, :-1]

                # Compute VAPO losses
                policy_loss, value_loss, positive_lm_loss = self.compute_vapo_losses(
                    current_log_probs,
                    rollout["old_log_probs"],
                    shifted_policy_adv,
                    shifted_critic_adv,
                    shifted_values,
                    shifted_old_values,
                    shifted_response_mask,
                    rollout["positive_mask"],
                )

                # Backpropagate each prompt's normalized contribution
                # immediately, then step once after the loop. This is gradient
                # accumulation with the same mean objective, but it releases
                # each pair of large policy/value forward graphs before the
                # next prompt instead of retaining the whole batch in VRAM.
                total_loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    + self.config.positive_lm_weight * positive_lm_loss
                )

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_positive_lm_losses.append(positive_lm_loss.item())

                # Compute explained variance
                with torch.no_grad():
                    returns_masked = rollout["returns"][batch_response_mask > 0]
                    values_masked = current_values[batch_response_mask > 0]
                    if len(returns_masked) > 1:
                        var_returns = returns_masked.var()
                        var_residual = (returns_masked - values_masked).var()
                        if var_returns > 1e-8:
                            explained_var = 1 - var_residual / var_returns
                            all_explained_variances.append(explained_var.item())

                (total_loss / prompt_count).backward()

            # A single optimizer update consumes the mean gradient accumulated
            # across prompts without requiring their graphs to coexist.
            if prompt_count > 0:
                _params = list(self.model.parameters())
                if _params:
                    torch.nn.utils.clip_grad_norm_(_params, self.config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(
                    self.value_head.parameters(), self.config.max_grad_norm
                )

                self.actor_optimizer.step()
                self.critic_optimizer.step()

            # Cadence convention shared with DAPO/GEPO: the LR schedulers
            # advance once per inner update, global_step counts train_steps.
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        self.global_step += 1

        # Compute metrics
        metrics = {
            "policy_loss": np.mean(all_policy_losses) if all_policy_losses else 0.0,
            "value_loss": np.mean(all_value_losses) if all_value_losses else 0.0,
            "positive_lm_loss": (
                np.mean(all_positive_lm_losses) if all_positive_lm_losses else 0.0
            ),
            "average_reward": np.mean(all_rewards) if all_rewards else 0.0,
            "accuracy": np.mean(all_accuracies) if all_accuracies else 0.0,
            "explained_variance": (
                np.mean(all_explained_variances) if all_explained_variances else 0.0
            ),
            "actor_lr": self.actor_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
            "global_step": self.global_step,
        }

        # Store metrics
        for key in [
            "policy_loss",
            "value_loss",
            "average_reward",
            "accuracy",
            "explained_variance",
        ]:
            if key in self.metrics_history:
                self.metrics_history[key].append(metrics[key])

        return metrics

    def save_checkpoint(self, output_dir: str) -> None:
        """Save model checkpoint"""
        os.makedirs(output_dir, exist_ok=True)

        # Save value head
        torch.save(
            self.value_head.state_dict(), os.path.join(output_dir, "value_head.pt")
        )

        save_checkpoint_artifacts(
            self.model,
            self.tokenizer,
            output_dir,
            training_state={
                "global_step": self.global_step,
                "rollout_samples_total": self.rollout_samples_total,
                "value_warmup_complete": self.value_warmup_complete,
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "actor_scheduler_state_dict": self.actor_scheduler.state_dict(),
                "critic_scheduler_state_dict": self.critic_scheduler.state_dict(),
                "metrics_history": self.metrics_history,
            },
            config_dict=self.config.to_dict(),
            config_filename="vapo_config.json",
        )

    def load_checkpoint(self, checkpoint_dir: str, *, trusted: bool = False) -> None:
        """Load checkpoint.

        Args:
            checkpoint_dir: Directory written by :meth:`save_checkpoint`.
            trusted: Pass ``True`` only for checkpoints from a source you
                control; the default unpickles with ``weights_only=True`` so a
                malicious checkpoint cannot execute code.
        """
        # Load value head
        value_head_path = os.path.join(checkpoint_dir, "value_head.pt")
        if os.path.exists(value_head_path):
            self.value_head.load_state_dict(
                load_checkpoint_file(
                    value_head_path, map_location=self.device, trusted=trusted
                )
            )

        # Load training state
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(state_path):
            state = load_checkpoint_file(
                state_path, map_location=self.device, trusted=trusted
            )
            self.global_step = state["global_step"]
            self.rollout_samples_total = int(
                state.get("rollout_samples_total", self.rollout_samples_total)
            )
            self.value_warmup_complete = state.get("value_warmup_complete", True)
            self.actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
            self.actor_scheduler.load_state_dict(state["actor_scheduler_state_dict"])
            self.critic_scheduler.load_state_dict(state["critic_scheduler_state_dict"])
            self.metrics_history = state.get("metrics_history", self.metrics_history)

        logger.info(f"Checkpoint loaded from {checkpoint_dir}")


from .vapo_entrypoints import train_with_vapo

# Export
__all__ = [
    "VAPOConfig",
    "VAPOTrainer",
    "ValueHead",
    "LengthAdaptiveGAE",
    "train_with_vapo",
]
