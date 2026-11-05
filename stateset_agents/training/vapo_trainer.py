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
import warnings
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.rust_accelerator import compute_gae as _rust_compute_gae
from ..core.rust_accelerator import is_rust_available as _rust_gae_available
from . import objectives, rl_losses
from .config import config_value
from .runtime_state import TrainingRuntime, validate_training_config
from .spine.rollouts import PromptGroupRollouts
from .spine.schedule import build_lr_scheduler, optimizer_step_budget
from .spine.scoring import CallableScorer
from .spine.update import (
    OptimizerStep,
    UpdateLoop,
    batch_shares,
    nonfinite_grad_settings,
    share_size,
)
from .trainer_runtime import (
    SharedModelManager,
    build_group_batch,
    hf_generate_group,
)
from .vapo_config import VAPOConfig

logger = logging.getLogger(__name__)


class VAPOModelManager(SharedModelManager):
    """Manages model loading for VAPO training (transformers and PEFT through
    :mod:`~stateset_agents.training.dedicated_deps`)."""

    algorithm = "VAPO"

    def __init__(self, config: VAPOConfig):
        super().__init__(config)


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
        values: torch.Tensor = self.network(hidden_states)
        return values


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
        self._warned_zero_lambda = False

    def compute_lambda_policy(self, sequence_length: int) -> float:
        """
        Compute length-adaptive lambda for policy (VAPO eq. 7).

        lambda = 1 - 1/(alpha * length), so the GAE coefficients sum to about
        ``alpha * length``. Lengths with ``alpha * length <= 1`` would give a
        non-positive lambda; they use 0 (one-step TD advantages).
        """
        scaled = self.lambda_policy_alpha * sequence_length
        if scaled <= 1.0:
            if not self._warned_zero_lambda:
                self._warned_zero_lambda = True
                logger.warning(
                    "VAPO length-adaptive lambda is 0 for a %d-token completion "
                    "(lambda_policy_alpha * length = %.3g <= 1): policy advantages "
                    "fall back to one-step TD and rely entirely on the critic. "
                    "For short completions raise lambda_policy_alpha (e.g. above "
                    "1 / typical_length).",
                    sequence_length,
                    scaled,
                )
            return 0.0
        return 1.0 - 1.0 / scaled

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


class VAPOTrainer(TrainingRuntime):
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
        validate_training_config(config, VAPOConfig)
        self.config = config
        # Value warmup draws its prompts from ``config.seed``, not the global
        # NumPy generator.
        self._warmup_rng = np.random.default_rng(config_value(config, "seed"))
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
            gamma=float(config.gamma),
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

        # Schedulers: ``lr_scheduler_type`` over every update pass of the run
        # (num_episodes x num_gradient_updates, the spine budget), clamped at
        # the end. Sizing them over num_episodes alone made the cosine reach
        # 0 halfway through a two-pass run and then climb back to the full LR.
        # The value-warmup iteration holds its K positions (train_step).
        total_steps = optimizer_step_budget(
            int(config.num_episodes), self._update_passes()
        )
        self.actor_scheduler = build_lr_scheduler(
            torch,
            self.actor_optimizer,
            kind=str(config.lr_scheduler_type),
            total_steps=total_steps,
            warmup_ratio=float(config.warmup_ratio),
        )
        self.critic_scheduler = build_lr_scheduler(
            torch,
            self.critic_optimizer,
            kind=str(config.lr_scheduler_type),
            total_steps=total_steps,
            warmup_ratio=float(config.warmup_ratio),
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

    def _update_passes(self) -> int:
        """Inner update passes per rollout batch (``num_gradient_updates``)."""
        return max(1, int(config_value(self.config, "num_gradient_updates")))

    def _advance_schedulers(self, steps: int) -> None:
        """Move both LR schedules ``steps`` positions without an update (the
        value-warmup iteration and an empty batch keep their place).

        Advancing before any ``optimizer.step()`` is deliberate here, so
        PyTorch's call-order warning is silenced for these calls only.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"Detected call of `lr_scheduler\.step\(\)`"
            )
            for _ in range(steps):
                self.actor_scheduler.step()
                self.critic_scheduler.step()

    def _build_optimizer_step(self, *, critic_only: bool = False) -> OptimizerStep:
        """The spine commit for one actor+critic update (or a critic warmup step).

        Gradients of every stepped optimizer must be finite
        (``config.nonfinite_grad_policy`` raises or skips the whole pass
        otherwise, actor and critic together). The actor and the value head
        are clipped to ``max_grad_norm`` independently, then each optimizer
        and LR scheduler steps once. ``global_step`` counts ``train_step``
        calls; value warmup steps the critic only and counts nothing.
        """
        nonfinite = nonfinite_grad_settings(self.config)
        critic_params = list(self.value_head.parameters())
        if critic_only:
            return OptimizerStep(
                self,
                torch,
                optimizers=[self.critic_optimizer],
                clip_groups=[critic_params],
                max_grad_norm=self.config.max_grad_norm,
                counter="none",
                **nonfinite,
                named_modules=[self.value_head],
                label="VAPO value warmup",
            )
        actor_params = list(self.model.parameters())
        return OptimizerStep(
            self,
            torch,
            optimizers=[self.actor_optimizer, self.critic_optimizer],
            schedulers=[self.actor_scheduler, self.critic_scheduler],
            clip_groups=[actor_params, critic_params],
            max_grad_norm=self.config.max_grad_norm,
            counter="train_steps",
            **nonfinite,
            named_modules=[self.model, self.value_head],
            label="VAPO",
        )

    async def _score_response(self, prompt: str, response: dict[str, Any]) -> Any:
        """One sampled reply's reward and reward-fallback flag
        (``spine.scoring.ScoredReply``)."""
        return await CallableScorer(self.reward_fn).score_checked(
            prompt, response["response"]
        )

    async def _compute_reward(self, prompt: str, response: str) -> float:
        """Score one reply with ``reward_fn``, sync or async (spine scorer)."""
        return await CallableScorer(self.reward_fn).score(prompt, response)

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract hidden states from model.

        Grad follows the caller: an outer ``torch.no_grad()`` is honoured even
        while the model is in train mode.
        """
        with torch.set_grad_enabled(self.model.training and torch.is_grad_enabled()):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Use last hidden state
            hidden_states: torch.Tensor = outputs.hidden_states[-1]
        return hidden_states

    def compute_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute values for a sequence.

        VAPO trains a separate critic, so the value loss must only update the
        value head: the backbone features are taken without gradient, and the
        policy is moved by the policy and positive-LM losses alone.
        """
        with torch.no_grad():
            hidden_states = self.get_hidden_states(input_ids, attention_mask)
        # Keep the critic in fp32 for stable regression while allowing the
        # policy backbone to run in bf16/fp16.  Linear layers require matching
        # dtypes (the live CUDA ``BFloat16 and Float`` matmul failure).
        value_dtype = next(self.value_head.parameters()).dtype
        values: torch.Tensor = self.value_head(
            hidden_states.detach().to(dtype=value_dtype)
        )
        return values.squeeze(-1)

    def compute_rollout_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Rollout-time values (GAE baseline and value-clip anchor), taken with
        value-head dropout off so they are not a random draw of the critic."""
        was_training = self.value_head.training
        self.value_head.eval()
        try:
            with torch.no_grad():
                return self.compute_values(input_ids, attention_mask)
        finally:
            self.value_head.train(was_training)

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
        token_log_probs: torch.Tensor
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

        explained_variances: list[float] = []
        critic_step = self._build_optimizer_step(critic_only=True)

        # Value pretraining runs at the critic's base learning rate: the joint
        # schedule's warmup ramp starts at 0, which left the value head
        # untrained whenever warmup_ratio > 0.
        scheduled = [group["lr"] for group in self.critic_optimizer.param_groups]
        for group, base in zip(
            self.critic_optimizer.param_groups,
            self.critic_scheduler.base_lrs,
            strict=True,
        ):
            group["lr"] = base
        try:
            total_value_loss = await self._value_warmup_steps(
                prompts, critic_step, explained_variances
            )
        finally:
            for group, rate in zip(
                self.critic_optimizer.param_groups, scheduled, strict=True
            ):
                group["lr"] = rate

        self.value_warmup_complete = True
        logger.info("Value network warmup complete")

        return {
            "warmup_value_loss": total_value_loss / self.config.value_warmup_steps,
            "warmup_explained_variance": (
                float(np.mean(explained_variances)) if explained_variances else 0
            ),
        }

    async def _value_warmup_steps(
        self,
        prompts: list[str],
        critic_step: OptimizerStep,
        explained_variances: list[float],
    ) -> float:
        """Run the value-warmup updates; return their summed value loss."""
        total_value_loss = 0.0
        for step in range(self.config.value_warmup_steps):
            # Sample prompt
            prompt = prompts[int(self._warmup_rng.integers(len(prompts)))]

            # Generate responses
            responses = await self.generate_group_responses(prompt)

            if len(responses) == 0:
                continue

            # Compute rewards (Monte-Carlo returns); a reply whose reward is a
            # fallback never trains the critic either.
            replies = [await self._score_response(prompt, resp) for resp in responses]
            responses = [
                resp
                for resp, reply in zip(responses, replies, strict=True)
                if reply.fallback is None
            ]
            rewards = [reply.score for reply in replies if reply.fallback is None]
            if not responses:
                continue

            # Prepare batch
            (
                batch_input_ids,
                batch_attention_mask,
                batch_response_mask,
            ) = build_group_batch(responses, self.device)
            assert batch_response_mask is not None

            # Monte-Carlo return targets in the action frame (see train_step)
            rewards_tensor = torch.tensor(rewards, device=self.device)
            mc_returns, action_mask = self.monte_carlo_value_targets(
                rewards_tensor, batch_response_mask
            )

            # Compute values
            with torch.no_grad():
                hidden_states = self.get_hidden_states(
                    batch_input_ids, batch_attention_mask
                )

            value_dtype = next(self.value_head.parameters()).dtype
            values = self.value_head(hidden_states.to(dtype=value_dtype)).squeeze(-1)
            values = values[:, :-1]

            # Value loss (MSE)
            value_loss = F.mse_loss(
                values * action_mask, mc_returns, reduction="sum"
            ) / action_mask.sum().clamp(min=1)

            # Update value network (finite gradients only, clipped)
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            critic_step()

            total_value_loss += value_loss.item()

            # Compute explained variance
            with torch.no_grad():
                var_returns = mc_returns[action_mask > 0].var()
                var_residual = (mc_returns - values)[action_mask > 0].var()
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
        return total_value_loss

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

    def monte_carlo_value_targets(
        self,
        rewards: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Discounted Monte-Carlo returns for value warmup, in the action frame.

        Column ``t`` is the state that emits token ``t+1``; each response's
        reward lands on its last action, and earlier actions see it discounted
        by the same ``gamma`` the GAE uses. Returns ``(targets, action_mask)``,
        both ``[batch, seq_len - 1]`` with targets zeroed outside the mask.
        """
        action_mask = response_mask[:, 1:]
        token_rewards = self.build_token_rewards(rewards, action_mask)
        dones = self.build_token_rewards(1.0, action_mask)
        returns, _ = self.gae_computer.compute_gae(
            token_rewards, torch.zeros_like(token_rewards), dones, 1.0
        )
        return returns * action_mask, action_mask

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

        # Warmup value network if not done. The warmup iteration is one of
        # the run's outer iterations, so the LR schedules advance by its K
        # positions (as for an empty batch) and a run of num_episodes
        # iterations still ends at the schedule's final value.
        if not self.value_warmup_complete:
            warmup_metrics = await self.warmup_value_network(prompts)
            self._advance_schedulers(self._update_passes())
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
        async def build(
            prompt: str, responses: list[dict[str, Any]]
        ) -> dict[str, Any] | None:
            # Compute rewards and identify correct samples. A reply whose
            # reward is a fallback is left out (never trained on); the group
            # still needs two replies for group statistics.
            responses, rewards = await sampler.score_group(
                responses, lambda resp: self._score_response(prompt, resp)
            )
            if len(responses) < 2:
                return None
            is_correct = []

            for resp, reward in zip(responses, rewards, strict=True):
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
                old_values = self.compute_rollout_values(
                    batch_input_ids, batch_attention_mask
                )

            # GAE runs in the shifted *action* frame shared by the log-probs,
            # values and loss masks: column t is the action at state t that
            # emits token t+1. The episode reward and terminal flag sit on the
            # last action column, so the value of the post-terminal state
            # (never trained) cannot leak into any advantage.
            action_mask = batch_response_mask[:, 1:]
            rewards_tensor = torch.tensor(rewards, device=self.device)
            reward_sequence = self.build_token_rewards(rewards_tensor, action_mask)
            dones = self.build_token_rewards(1.0, action_mask)

            # Compute decoupled GAE
            (
                critic_advantages,
                policy_advantages,
                returns,
            ) = self.gae_computer.compute_decoupled_gae(
                reward_sequence, old_values[:, :-1], dones, sequence_lengths
            )

            # Normalize advantages over the actions that are actually trained
            policy_advantages = rl_losses.whiten(policy_advantages, action_mask)

            return {
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

        # One group per prompt (at least two responses for group statistics).
        sampler = PromptGroupRollouts(
            self.generate_group_responses, min_group_size=2, label="VAPO"
        )
        rollouts = await sampler.collect(
            prompts[: self.config.per_device_train_batch_size], build
        )

        # --- Update phase ---------------------------------------------------
        # ``num_gradient_updates`` (mu) inner updates reuse the same rollouts
        # against the frozen old log probs/values, exactly like DAPO. With the
        # default of 1 the formula is identical to the previous single-update
        # path (only dropout RNG ordering differs).
        num_updates = self._update_passes()

        # Every term aggregates over the whole batch, not per prompt: each
        # prompt's loss is a mean over its own rows/tokens, so weighting it by
        # its share of the batch's tokens (rows for seq_mean) makes the
        # accumulated gradient the batch-level loss. Averaging prompts equally
        # instead would weight a token by 1/(its group's length), the length
        # bias token-level loss exists to remove. Matches DAPO.
        aggregate = self._objective.aggregate
        policy_weights = batch_shares(
            [share_size(r["response_mask"], aggregate) for r in rollouts]
        )
        value_weights = batch_shares(
            [share_size(r["response_mask"], "token_mean") for r in rollouts]
        )
        positive_weights = batch_shares(
            [float(r["positive_mask"].sum()) for r in rollouts]
        )
        # Metrics report the last pass.
        all_policy_losses: list[float] = []
        all_value_losses: list[float] = []
        all_positive_lm_losses: list[float] = []
        all_explained_variances: list[float] = []

        def start_pass(_pass: int) -> None:
            for values in (
                all_policy_losses,
                all_value_losses,
                all_positive_lm_losses,
                all_explained_variances,
            ):
                values.clear()

        def prompt_loss(_pass: int, index: int, rollout: dict[str, Any]) -> Any:
            batch_input_ids = rollout["input_ids"]
            batch_attention_mask = rollout["attention_mask"]
            batch_response_mask = rollout["response_mask"]

            # Compute current log probs and values
            current_log_probs = self.compute_token_log_probs(
                batch_input_ids, batch_attention_mask, batch_response_mask
            )
            current_values = self.compute_values(batch_input_ids, batch_attention_mask)

            # Shift masks for loss computation
            shifted_response_mask = batch_response_mask[:, 1:]
            shifted_values = current_values[:, :-1]

            # Compute VAPO losses
            policy_loss, value_loss, positive_lm_loss = self.compute_vapo_losses(
                current_log_probs,
                rollout["old_log_probs"],
                rollout["policy_advantages"],
                rollout["critic_advantages"],
                shifted_values,
                rollout["old_values"][:, :-1],
                shifted_response_mask,
                rollout["positive_mask"],
            )

            # Each prompt's weighted contribution is backpropagated
            # immediately (chunked UpdateLoop), then one step follows the
            # last prompt: gradient accumulation of the batch-level
            # objective that releases each pair of large policy/value
            # forward graphs before the next prompt instead of retaining the
            # whole batch in VRAM.
            weighted_loss = (
                policy_weights[index] * policy_loss
                + self.config.value_loss_coef * value_weights[index] * value_loss
                + self.config.positive_lm_weight
                * positive_weights[index]
                * positive_lm_loss
            )

            all_policy_losses.append(policy_loss.item())
            all_value_losses.append(value_loss.item())
            all_positive_lm_losses.append(positive_lm_loss.item())

            # Compute explained variance
            with torch.no_grad():
                returns_masked = rollout["returns"][shifted_response_mask > 0]
                values_masked = shifted_values[shifted_response_mask > 0]
                if len(returns_masked) > 1:
                    var_returns = returns_masked.var()
                    var_residual = (returns_masked - values_masked).var()
                    if var_returns > 1e-8:
                        explained_var = 1 - var_residual / var_returns
                        all_explained_variances.append(explained_var.item())

            return weighted_loss if weighted_loss.requires_grad else None

        # Cadence convention shared with DAPO/GEPO: the LR schedulers advance
        # once per committed inner update (not for a pass skipped under
        # nonfinite_grad_policy="skip"), global_step counts train_steps.
        passes = UpdateLoop(self._build_optimizer_step(), passes=num_updates).run(
            rollouts, prompt_loss, on_pass_start=start_pass
        )
        if not rollouts:
            # Nothing to update: the schedulers still advance once per inner
            # update, as VAPO always has for an empty batch.
            self._advance_schedulers(num_updates)

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
            "optimizer_updates": passes.committed,
            **sampler.fallbacks.metrics(),
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
        """Publish actor, critic, optimizer and RNG state together."""
        self.save_runtime_checkpoint(output_dir)

    def load_checkpoint(self, checkpoint_dir: str, *, trusted: bool = False) -> bool:
        """Restore policy, value head and all training components."""
        return self.load_runtime_checkpoint(checkpoint_dir, trusted=trusted)


from .vapo_entrypoints import train_with_vapo

# Export
__all__ = [
    "VAPOConfig",
    "VAPOTrainer",
    "ValueHead",
    "LengthAdaptiveGAE",
    "train_with_vapo",
]
