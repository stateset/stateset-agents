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
import math
from collections.abc import Awaitable, Callable
from copy import deepcopy
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..core.rust_accelerator import compute_gae_targets as _rust_gae_targets
from ..core.rust_accelerator import is_rust_available as _rust_gae_available
from . import objectives, rl_losses
from .adamw_weights import model_adamw
from .advantage_estimation import (
    gae_dtype,
    gae_targets,
    normalize_advantages,
    terminal_rewards,
)
from .runtime_state import (
    TrainingRuntime,
    _require_runtime_ready,
    validate_training_config,
)
from .trainer_runtime import (
    SharedModelManager,
    build_group_batch,
    group_old_logprobs,
    hf_generate_group,
)
from .trainer_utils import (
    _clip_gradient_norm,
    deterministic_scoring,
    inner_update_scheduler,
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

    The critic uses a fixed lambda. The policy uses the number of response
    actions to set its lambda.

    lambda_policy = 1 - 1/(alpha * length + 1)

    With n actions remaining, the residual weights sum to the first n terms
    of the geometric series with ratio gamma * lambda_policy.
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

        For nonnegative inputs the real coefficient increases toward one.
        """
        scaled_length = self.lambda_policy_alpha * sequence_length
        return (
            1.0 if scaled_length == math.inf else scaled_length / (1.0 + scaled_length)
        )

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
        result = self._try_rust_targets(rewards, values, dones, lambda_value)
        if result is not None:
            return result
        return gae_targets(rewards, values, self.gamma, lambda_value, dones)

    def _try_rust_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        lambda_value: float,
    ) -> torch.Tensor | None:
        """Return just the advantage component for existing callers."""
        result = self._try_rust_targets(rewards, values, dones, lambda_value)
        return None if result is None else result[0]

    def _try_rust_targets(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        lambda_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Optional fast path using the Rust paired GAE kernel.

        Only used when every row has at most one termination step (the
        common single-episode-per-response case used by VAPO). Rows are
        truncated at their terminal step before delegating to the Rust
        kernel (a single termination at the end of a sequence is
        equivalent to simply not bootstrapping past it). The suffix starts a
        new episode and is computed independently as in the torch recursion. Returns ``None`` to
        signal the caller should fall back to the pure-torch loop, either
        because the Rust extension isn't installed or a row has more than
        one termination flag (a case the vectorized kernel can't express).
        """
        if (
            not _rust_gae_available()
            or rewards.ndim != 2
            or values.shape != rewards.shape
            or dones.shape != rewards.shape
        ):
            return None

        batch_size, seq_len = rewards.shape
        dtype = gae_dtype(rewards, values)
        rewards_np = rewards.detach().to(device="cpu", dtype=torch.float64).numpy()
        values_np = values.detach().to(device="cpu", dtype=torch.float64).numpy()
        dones_np = dones.detach().cpu().numpy()

        advantages_np = np.empty((batch_size, seq_len), dtype=np.float64)
        targets_np = np.empty_like(advantages_np)
        for i in range(batch_size):
            done_indices = np.nonzero(dones_np[i])[0]
            if len(done_indices) > 1:
                return None
            end = int(done_indices[0]) + 1 if len(done_indices) else seq_len
            for start, stop in ((0, end), (end, seq_len)):
                if start < stop:
                    advantages_np[i, start:stop], targets_np[i, start:stop] = (
                        _rust_gae_targets(
                            rewards_np[i, start:stop],
                            values_np[i, start:stop],
                            self.gamma,
                            lambda_value,
                        )
                    )
        return (
            torch.as_tensor(advantages_np, dtype=dtype, device=rewards.device),
            torch.as_tensor(targets_np, dtype=dtype, device=rewards.device),
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
        policy_advantages = torch.empty_like(critic_advantages)

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
    3. Length-adaptive lambda: lambda = 1 - 1/(1 + alpha * length)
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
        ref_model: Any | None = None,
    ):
        # Ensure transformers is loaded for scheduler
        _load_transformers_vapo()

        validate_training_config(config)
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
        if "entropy_coef" not in (config.objective_overrides or {}):
            self._objective = self._objective.with_(
                entropy_coef=float(config.entropy_coef)
            )
        self.model = model
        self.ref_model = None
        if self._objective.kl_coef > 0:
            reference = model if ref_model is None else ref_model
            self.ref_model = deepcopy(reference).requires_grad_(False).eval()
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
            gamma=config.gamma,
            lambda_critic=config.lambda_critic,
            lambda_policy_alpha=config.lambda_policy_alpha,
        )

        # Separate optimizers for actor and critic
        params = list(self.model.parameters())
        if not params:
            self._stub_param = torch.nn.Parameter(torch.zeros(1))
            params = [self._stub_param]
        self.actor_optimizer = model_adamw(
            params,
            lr=config.actor_learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )

        self.critic_optimizer = model_adamw(
            self.value_head.parameters(),
            lr=config.critic_learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )

        # Schedulers
        cosine = get_cosine_schedule_with_warmup is not None
        self.actor_scheduler = inner_update_scheduler(
            self.actor_optimizer, config, cosine=cosine
        )
        self.critic_scheduler = inner_update_scheduler(
            self.critic_optimizer, config, cosine=cosine
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
        self.value_warmup_updates = 0
        self.value_warmup_complete = self.config.value_warmup_steps == 0
        self.rollout_samples_total = 0

    async def _compute_reward(self, prompt: str, response: str) -> float:
        """Resolve sync or async reward callbacks to a float."""
        reward = self.reward_fn(prompt, response)
        if isinstance(reward, Awaitable):
            reward = await reward
        result = float(reward)
        if not math.isfinite(result):
            raise ValueError("VAPO requires finite rollout rewards")
        return result

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract hidden states from model"""
        with deterministic_scoring(self.model):
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
        response_mask: torch.Tensor | None = None,
        *,
        detach_backbone: bool = False,
    ) -> torch.Tensor:
        """Compute values for a sequence"""
        if detach_backbone:
            with torch.no_grad():
                hidden_states = self.get_hidden_states(input_ids, attention_mask)
        else:
            hidden_states = self.get_hidden_states(input_ids, attention_mask)
        if response_mask is not None:
            # Value at column t predicts the return of the action at t+1.
            active = torch.nn.functional.pad(response_mask[:, 1:].bool(), (0, 1))
            hidden_states = hidden_states.masked_fill(~active.unsqueeze(-1), 0)

        # Keep the critic in fp32 for stable regression while allowing the
        # policy backbone to run in bf16/fp16.  Linear layers require matching
        # dtypes; an explicit differentiable cast avoids the live CUDA
        # ``BFloat16 and Float`` matmul failure without severing actor grads.
        value_dtype = next(self.value_head.parameters()).dtype
        with deterministic_scoring(self.value_head):
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
        with deterministic_scoring(self.model):
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
        _require_runtime_ready(self)
        logger.info(
            f"Starting value network warmup ({self.config.value_warmup_steps} steps)"
        )

        total_value_loss = 0.0
        explained_variances = []

        updates = 0
        remaining = self.config.value_warmup_steps - self.value_warmup_updates
        for _ in range(remaining if prompts else 0):
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
            (
                batch_input_ids,
                batch_attention_mask,
                batch_response_mask,
            ) = build_group_batch(responses, self.device)
            assert batch_response_mask is not None

            if not batch_response_mask[:, 1:].any():
                continue

            # Monte-Carlo return targets in the action frame (see train_step)
            reward_device = "cpu" if self.device.type == "mps" else self.device
            rewards_tensor = torch.tensor(
                rewards, dtype=torch.float64, device=reward_device
            )
            mc_returns, action_mask = self.monte_carlo_value_targets(
                rewards_tensor, batch_response_mask
            )

            values = self.compute_values(
                batch_input_ids,
                batch_attention_mask,
                batch_response_mask,
                detach_backbone=True,
            )[:, :-1]

            # Compare the critic to the original binary64 rollout rewards.
            values = values.to(mc_returns.device)
            value_loss = rl_losses.critic_value_loss(
                values, values.detach(), mc_returns, action_mask, None
            )

            # Update value network
            self.critic_optimizer.zero_grad()
            try:
                if not torch.isfinite(value_loss):
                    raise ValueError("VAPO warmup value loss must be finite")
                value_loss.backward()
                gradients = [
                    p.grad for p in self.value_head.parameters() if p.grad is not None
                ]
                _clip_gradient_norm(gradients, self.config.max_grad_norm, torch)
            except BaseException:
                self.critic_optimizer.zero_grad(set_to_none=True)
                raise
            try:
                self.critic_optimizer.step()
            except BaseException:
                self._runtime_checkpoint_poisoned = True
                raise

            self.value_warmup_updates += 1
            updates += 1
            total_value_loss += value_loss.item()

            # Compute explained variance
            with torch.no_grad():
                targets = mc_returns[action_mask > 0]
                if targets.numel() > 1:
                    var_returns = targets.var()
                    if var_returns > 1e-8:
                        residuals = targets - values[action_mask > 0]
                        explained_variances.append(
                            (1 - residuals.var() / var_returns).item()
                        )

            if self.value_warmup_updates % 10 == 0:
                avg_loss = total_value_loss / updates
                avg_ev = (
                    np.mean(explained_variances[-10:]) if explained_variances else 0
                )
                logger.info(
                    f"Value warmup step {self.value_warmup_updates}/{self.config.value_warmup_steps} | "
                    f"Loss: {avg_loss:.4f} | EV: {avg_ev:.4f}"
                )

        self.value_warmup_complete = (
            self.value_warmup_updates >= self.config.value_warmup_steps
        )

        return {
            "warmup_value_loss": total_value_loss / max(updates, 1),
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
        return rl_losses.critic_value_loss(
            values,
            old_values,
            returns,
            response_mask,
            self.config.value_clip if self.config.value_clip > 0 else None,
            square_mask=True,
        )

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
        if not ((response_mask == 0) | (response_mask == 1)).all():
            raise ValueError("VAPO requires binary response masks")
        action_mask = response_mask[:, 1:].to(rewards.device)
        token_rewards = self.build_token_rewards(rewards, action_mask)
        dones = self.build_token_rewards(1.0, action_mask)
        _, returns = self.gae_computer.compute_gae(
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
        return terminal_rewards(scalar_reward, response_mask)

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
        critic_returns: torch.Tensor | None = None,
        *,
        entropy: torch.Tensor | None = None,
        kl: torch.Tensor | None = None,
        logp_ref: torch.Tensor | None = None,
        batch_population: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAPO losses:
        1. Policy loss with Clip-Higher and token-level normalization
        2. Value loss (decoupled GAE: value target from critic-lambda GAE)
        3. Positive example LM loss
        """
        # Clip-Higher surrogate and token-/sequence-level aggregation via the
        # shared objective (ratios are clamped before exp inside it).
        if self._objective.entropy_coef > 0 and entropy is None:
            raise ValueError(
                "The VAPO entropy bonus requires per-token categorical entropy"
            )
        scores = current_log_probs
        if policy_advantages.dtype == torch.float64:
            scores = scores.to(policy_advantages.device, torch.float64)
        policy_result = objectives.policy_loss(
            logp_cur=scores,
            mask=response_mask,
            advantages=policy_advantages,
            objective=self._objective,
            logp_old=old_log_probs,
            entropy=entropy,
            kl=kl,
            logp_ref=logp_ref,
            batch_population=batch_population,
        )
        policy_loss = policy_result.loss

        # Legacy callers supply only critic advantages. The training loop has
        # the unrounded critic-lambda target from the same rollout scan.
        returns = (
            critic_returns
            if critic_returns is not None
            else critic_advantages + old_values
        )
        value_loss = self.compute_value_loss(
            values.to(returns.device),
            old_values.to(returns.device),
            returns,
            response_mask.to(returns.device),
        )

        # Positive example LM loss
        if self.config.use_positive_lm_loss:
            selected = (positive_mask.bool() & response_mask.bool()).to(
                current_log_probs.device
            )
            observed = current_log_probs[selected]
            positive_lm_loss = -(observed.to(scores) / max(1, observed.numel())).sum()
        else:
            positive_lm_loss = current_log_probs.new_zeros(())

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

        all_rewards: list[float] = []
        all_accuracies: list[float] = []

        # --- Rollout phase --------------------------------------------------
        # Generate every group and freeze the old-policy log probs and values
        # here, before any inner update moves the weights. Recomputing them
        # from the current model made the importance ratio identically 1 and
        # the Clip-Higher trust region a no-op.
        rollouts: list[dict[str, Any]] = []
        response_rows = 0

        for prompt in prompts[: self.config.per_device_train_batch_size]:
            # Generate group responses
            responses = await self.generate_group_responses(prompt)

            if not responses:
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
            if not ((batch_response_mask == 0) | (batch_response_mask == 1)).all():
                raise ValueError("VAPO requires binary response masks")
            response_rows += batch_size
            if not batch_response_mask[:, 1:].any():
                continue
            max_len = batch_input_ids.shape[1]

            sequence_lengths = batch_response_mask[:, 1:].count_nonzero(dim=1)

            # Positive mask for LM loss (correct samples only)
            positive_mask = torch.zeros(batch_size, max_len - 1, device=self.device)
            for i, correct in enumerate(is_correct):
                if correct:
                    # Mask response tokens for correct samples
                    resp_mask = batch_response_mask[i, 1:]  # Shift for next-token
                    positive_mask[i] = resp_mask

            # Get old log probs and values
            with torch.no_grad():
                old_log_probs = group_old_logprobs(
                    responses,
                    batch_response_mask,
                    self.config,
                    partial(
                        self.compute_token_log_probs,
                        batch_input_ids,
                        batch_attention_mask,
                        batch_response_mask,
                    ),
                )
                old_values = self.compute_values(
                    batch_input_ids, batch_attention_mask, batch_response_mask
                )

            # GAE runs in the shifted *action* frame shared by the log-probs,
            # values and loss masks: column t is the action at state t that
            # emits token t+1. The episode reward and terminal flag sit on the
            # last action column, so the value of the post-terminal state
            # (never trained) cannot leak into any advantage.
            reward_device = "cpu" if self.device.type == "mps" else self.device
            rewards_tensor = torch.tensor(
                rewards, dtype=torch.float64, device=reward_device
            )
            action_mask = batch_response_mask[:, 1:].to(reward_device)
            reward_sequence = self.build_token_rewards(rewards_tensor, action_mask)
            dones = self.build_token_rewards(1.0, action_mask)

            # Compute decoupled GAE
            (
                critic_advantages,
                policy_advantages,
                returns,
            ) = self.gae_computer.compute_decoupled_gae(
                reward_sequence,
                old_values[:, :-1].to(reward_device).masked_fill(action_mask == 0, 0),
                dones,
                sequence_lengths,
            )

            # Normalize advantages over the actions that are actually trained
            policy_advantages = normalize_advantages(policy_advantages, action_mask > 0)

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

        token_counts = [
            int(r["response_mask"][:, 1:].count_nonzero().item()) for r in rollouts
        ]
        positive_counts = [
            int(r["positive_mask"].count_nonzero().item()) for r in rollouts
        ]
        policy_counts = (
            token_counts
            if self._objective.aggregate == "token_mean"
            else [r["input_ids"].shape[0] for r in rollouts]
        )
        policy_count = max(
            1,
            (
                sum(token_counts)
                if self._objective.aggregate == "token_mean"
                else response_rows
            ),
        )
        token_count = max(1, sum(token_counts))
        positive_count = max(1, sum(positive_counts))

        regularized = self._objective.entropy_coef > 0 or self._objective.kl_coef > 0
        for _ in range(num_updates):
            all_policy_losses: list[float] = []
            all_value_losses: list[float] = []
            all_positive_lm_losses: list[float] = []
            all_explained_variances: list[float] = []

            if not rollouts:
                break
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            try:
                for rollout, policy_n, token_n, positive_n in zip(
                    rollouts, policy_counts, token_counts, positive_counts, strict=True
                ):
                    batch_input_ids = rollout["input_ids"]
                    batch_attention_mask = rollout["attention_mask"]
                    batch_response_mask = rollout["response_mask"]

                    # Compute current log probs and values
                    terms: dict[str, Any] = {}
                    if regularized:
                        from .causal_likelihood import regularized_policy_scores

                        current_log_probs, terms = regularized_policy_scores(
                            self.model,
                            self.ref_model,
                            batch_input_ids,
                            batch_attention_mask,
                            batch_response_mask,
                            self._objective,
                            self._logprob_dtype,
                        )
                        terms["batch_population"] = (response_rows, token_count)
                    else:
                        current_log_probs = self.compute_token_log_probs(
                            batch_input_ids, batch_attention_mask, batch_response_mask
                        )
                    current_values = self.compute_values(
                        batch_input_ids, batch_attention_mask, batch_response_mask
                    )

                    # Shift masks for loss computation
                    shifted_response_mask = batch_response_mask[:, 1:]
                    shifted_policy_adv = rollout["policy_advantages"]
                    shifted_critic_adv = rollout["critic_advantages"]
                    shifted_old_values = rollout["old_values"][:, :-1]
                    shifted_values = current_values[:, :-1]

                    # Compute VAPO losses
                    policy_loss, value_loss, positive_lm_loss = (
                        self.compute_vapo_losses(
                            current_log_probs,
                            rollout["old_log_probs"],
                            shifted_policy_adv,
                            shifted_critic_adv,
                            shifted_values,
                            shifted_old_values,
                            shifted_response_mask,
                            rollout["positive_mask"],
                            rollout["returns"],
                            **terms,
                        )
                    )

                    # Each local mean contributes its share of the batch population.
                    if not regularized:
                        policy_loss = policy_loss * (policy_n / policy_count)
                    value_loss = value_loss * (token_n / token_count)
                    positive_lm_loss = positive_lm_loss * (positive_n / positive_count)
                    total_loss = (
                        policy_loss
                        + self.config.value_loss_coef
                        * value_loss.to(policy_loss.device)
                        + self.config.positive_lm_weight
                        * positive_lm_loss.to(policy_loss.device)
                    )
                    all_policy_losses.append(policy_loss.item())
                    all_value_losses.append(value_loss.item())
                    all_positive_lm_losses.append(positive_lm_loss.item())

                    # Compute explained variance
                    with torch.no_grad():
                        returns_masked = rollout["returns"][
                            shifted_response_mask.to(rollout["returns"].device) > 0
                        ]
                        values_masked = shifted_values[shifted_response_mask > 0].to(
                            returns_masked.device
                        )
                        if len(returns_masked) > 1:
                            var_returns = returns_masked.var()
                            var_residual = (returns_masked - values_masked).var()
                            if var_returns > 1e-8:
                                explained_var = 1 - var_residual / var_returns
                                all_explained_variances.append(explained_var.item())

                    if not torch.isfinite(total_loss):
                        raise ValueError("VAPO batch loss must be finite")
                    total_loss.backward()

                for network in (self.model, self.value_head):
                    gradients = [
                        p.grad for p in network.parameters() if p.grad is not None
                    ]
                    _clip_gradient_norm(gradients, self.config.max_grad_norm, torch)
            except BaseException:
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                raise

            # A failed optimizer or scheduler may already have changed its state.
            # Recover the four coupled states together from a complete checkpoint.
            try:
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                self.actor_scheduler.step()
                self.critic_scheduler.step()
            except BaseException:
                self._runtime_checkpoint_poisoned = True
                raise

        self.global_step += 1

        # Compute metrics
        metrics = {
            "policy_loss": sum(all_policy_losses, 0.0),
            "value_loss": sum(all_value_losses, 0.0),
            "positive_lm_loss": (sum(all_positive_lm_losses, 0.0)),
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
