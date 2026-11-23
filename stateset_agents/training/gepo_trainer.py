"""
Group Expectation Policy Optimization (GEPO) Training for StateSet Agents

GEPO is an advanced RL algorithm that uses group-level importance weights to
exponentially reduce variance under high KL divergence. This makes it particularly
robust for heterogeneous and distributed training environments.

Key innovations:
- Group Expectation Importance Weights (GEIW) instead of per-token or per-sequence
- Exponentially reduces variance when KL divergence is high
- Superior stability under network latency and heterogeneous compute

Reference: https://arxiv.org/abs/2508.17850
"""

import logging
import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import objectives
from .adamw_weights import model_adamw
from .config import TrainingConfig
from .reward_advantages import reward_tensor
from .runtime_state import (
    TrainingRuntime,
    _reject_unsupported_resume,
    validate_training_config,
)
from .trainer_runtime import (
    SharedModelManager,
    build_group_batch,
)
from .trainer_utils import (
    clip_parameter_gradients,
    deterministic_scoring,
    inner_update_scheduler,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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
_transformers_gepo_loaded = False
AutoModelForCausalLM: Any | None = None
AutoTokenizer: Any | None = None
get_cosine_schedule_with_warmup: Any | None = None


def _load_transformers_gepo() -> bool:
    """Lazily load transformers to avoid import-time errors."""
    global _transformers_gepo_loaded, AutoModelForCausalLM, AutoTokenizer
    global get_cosine_schedule_with_warmup
    if _transformers_gepo_loaded:
        return True
    try:
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer
        from transformers import get_cosine_schedule_with_warmup as _get_cosine

        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
        get_cosine_schedule_with_warmup = _get_cosine
        _transformers_gepo_loaded = True
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Failed to load transformers: {e}")
        return False


def _require_peft() -> None:
    """Require PEFT only when LoRA model loading actually needs it."""
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError(
            "PEFT is required for GEPO LoRA training. "
            "Install with `pip install stateset-agents[training]` or `pip install peft`."
        )


def _require_wandb() -> None:
    """Require W&B only when experiment logging is enabled."""
    if wandb is None:
        raise ImportError(
            "wandb is required for GEPO logging. "
            "Install with `pip install stateset-agents[training]` or `pip install wandb`."
        )


@dataclass
class GEPOConfig(TrainingConfig):
    """
    Configuration for GEPO training.

    GEPO uses group-level importance weights which provide superior stability
    compared to token-level (GRPO) or sequence-level (GSPO) weights.
    """

    # Model
    model_name: str = "gpt2"

    # GEPO specific parameters
    group_size: int = 8  # Number of responses per prompt (G)

    # Clipping (standard PPO-style, applied after GEPO coefficient computation)
    clip_eps: float = 0.2  # Clipping epsilon for policy ratio

    # KL penalty (typically set to 0 for GEPO as group weights handle divergence)
    beta: float = 0.0
    use_reference_model: bool = False

    # Training parameters from the paper
    learning_rate: float = 1e-6
    warmup_ratio: float = 0.03  # 3% linear warmup
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 8

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

    # Advantage computation
    use_group_baseline: bool = True  # Use within-group baseline normalization

    # One inner update uses fresh rollouts. Larger values reuse their frozen
    # sampler scores. Filtering and temperature can make the generator differ
    # from the raw learner policy even on the first update.
    # Cadence (shared with DAPO and VAPO): the LR scheduler advances once per
    # inner update; global_step counts train_steps, not inner updates.
    num_gradient_updates: int = 1

    # Opt-in memory-efficient chunked LM-head log-prob projection (avoids
    # materializing [B, T, V] logits on long-context RL).
    chunked_loss_chunk_size: int | None = None
    use_loss_checkpointing: bool = False

    @classmethod
    def from_training_config(
        cls, config: TrainingConfig, **kwargs: Any
    ) -> "GEPOConfig":
        """Create GEPO config from standard training config"""
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        # Override num_generations with group_size
        if "group_size" in kwargs:
            config_dict["num_generations"] = kwargs["group_size"]
        return cls(**config_dict)


class GEPOModelManager(SharedModelManager):
    """Manages model loading for GEPO training"""

    def __init__(self, config: GEPOConfig):
        super().__init__(config)

    def _get_transformers(self) -> tuple[Any, Any]:
        if not _load_transformers_gepo():
            raise ImportError(
                "transformers is required for GEPO training. "
                "Install with `pip install stateset-agents[training]`."
            )
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers GEPO loader did not initialize correctly")
        return AutoTokenizer, AutoModelForCausalLM

    def _peft_components(self) -> tuple[Any, Any, Any]:
        _require_peft()
        return LoraConfig, TaskType, get_peft_model


class GEPOTrainer(TrainingRuntime):
    """
    Group Expectation Policy Optimization (GEPO) Trainer

    GEPO improves upon GRPO and GSPO by using group-level importance weights:

    w_GEIW(y|x) = p(y|x) / E_q[q(y|x)]

    where the group expectation is computed as:
    E_q[q(y|x)] ≈ Σ q(y^i|x)² / Σ q(y^i|x)

    StateSet uses the summed log probability of the retained response tokens
    when forming each sequence probability.

    Reference: https://arxiv.org/abs/2508.17850
    """

    def __init__(
        self,
        config: GEPOConfig,
        model: Any,
        tokenizer: Any,
        reward_fn: Callable[[str, str], float],
        ref_model: Any | None = None,
    ):
        # Ensure transformers is loaded for scheduler
        _load_transformers_gepo()

        validate_training_config(config)
        self.config = config
        # GEPO objective: group-expectation importance coefficient, PPO clip.
        self._objective = objectives.resolve_objective(
            config,
            "gepo",
            kl_coef=float(config.beta),
            kl="k3_token" if config.beta > 0 else "none",
            max_completion_length=int(config.max_completion_length),
            supported_ratios=("group_expectation", "sequence", "sequence_token"),
            clip_low=float(config.clip_eps),
            clip_high=float(config.clip_eps),
        )
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.ref_model = ref_model
        if self._objective.kl_coef > 0 and (
            self.ref_model is None or self.ref_model is model
        ):
            self.ref_model = deepcopy(model).requires_grad_(False).eval()
        self.device = next(model.parameters()).device

        # Optimizer
        self.optimizer = model_adamw(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = inner_update_scheduler(
            self.optimizer, config, cosine=get_cosine_schedule_with_warmup is not None
        )

        # Metrics tracking
        self.metrics_history: dict[str, list[float]] = {
            "policy_loss": [],
            "average_reward": [],
            "kl_divergence": [],
            "gepo_coefficient": [],
            "advantage_std": [],
        }

        self.global_step = 0
        self.rollout_samples_total = 0

    @staticmethod
    def build_response_mask(
        attention_mask: torch.Tensor,
        response_start_idx: int | list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Build a mask (on the shifted next-token-prediction axis) that selects
        response tokens only, excluding prompt tokens and padding.

        `response_start_idx` is the (unshifted) index of the first response
        token; the shifted axis has length `seq_len - 1`, so the equivalent
        shifted start index is `max(response_start_idx - 1, 0)`, matching the
        convention used in gspo_trainer.py.
        """
        shift_mask = attention_mask[:, 1:]
        starts = torch.as_tensor(response_start_idx, device=attention_mask.device)
        if starts.ndim == 0:
            starts = starts.expand(len(shift_mask))
        if starts.shape != (len(shift_mask),):
            raise ValueError("GEPO requires one response start per sequence")
        columns = torch.arange(1, attention_mask.shape[1], device=attention_mask.device)
        return shift_mask * (columns >= starts[:, None])

    def compute_sequence_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_start_idx: int | list[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities for response tokens.

        Returns:
            token_log_probs: Log probs for each token [batch, seq_len]
            sequence_log_probs: Sum of log probs per sequence [batch]
        """
        response_mask = self.build_response_mask(attention_mask, response_start_idx)
        chunk_size = getattr(self.config, "chunked_loss_chunk_size", None)
        if chunk_size is not None and int(chunk_size) > 0:
            from .chunked_rl_loss import chunked_model_token_logprobs

            full_resp_mask = torch.zeros_like(attention_mask)
            full_resp_mask[:, 1:] = response_mask.to(dtype=attention_mask.dtype)
            with deterministic_scoring(self.model):
                masked_log_probs, _ = chunked_model_token_logprobs(
                    self.model,
                    input_ids,
                    full_resp_mask,
                    attention_mask=attention_mask,
                    chunk_size=int(chunk_size),
                    use_checkpointing=bool(
                        getattr(self.config, "use_loss_checkpointing", False)
                    ),
                )
            score_tokens = (
                masked_log_probs.cpu()
                if masked_log_probs.device.type == "mps"
                else masked_log_probs
            )
            sequence_log_probs = score_tokens.sum(dim=-1, dtype=torch.float64)
            return masked_log_probs, sequence_log_probs

        log_probs = self._log_distribution(
            self.model, input_ids, attention_mask, response_mask
        )
        actions = input_ids[:, 1:].masked_fill(response_mask == 0, 0)
        token_log_probs = log_probs.gather(-1, actions[..., None]).squeeze(-1)

        masked_log_probs = token_log_probs * response_mask

        # Sum over sequence
        score_tokens = (
            masked_log_probs.cpu()
            if masked_log_probs.device.type == "mps"
            else masked_log_probs
        )
        sequence_log_probs = score_tokens.sum(dim=-1, dtype=torch.float64)

        return token_log_probs, sequence_log_probs

    @staticmethod
    def _log_distribution(
        model: Any,
        ids: torch.Tensor,
        mask: torch.Tensor,
        responses: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with deterministic_scoring(model):
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
        if responses is not None:
            logits = logits.masked_fill((responses == 0)[..., None], 0)
        return F.log_softmax(
            logits, dim=-1, dtype=torch.promote_types(logits.dtype, torch.float32)
        )

    @staticmethod
    def compute_gepo_coefficient_static(
        learner_seq_log_probs: torch.Tensor,
        sampler_seq_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GEPO coefficient using Group Expectation Importance Weights,
        entirely in log space to avoid underflow for realistic (very negative)
        sequence log-probability sums.

        coef_i = p_i / E_qhat[q], where E_qhat[q] = sum(q^2) / sum(q)

        Args:
            learner_seq_log_probs: Sequence log-probs from current policy [group_size]
            sampler_seq_log_probs: Sequence log-probs from sampling policy [group_size]

        Returns:
            GEPO coefficients (linear space) for each sequence [group_size]
        """
        from .gepo_weights import group_coefficients

        if learner_seq_log_probs.ndim != 1:
            raise ValueError("GEPO expects a vector of learner sequence sums")
        current = learner_seq_log_probs[:, None]
        return group_coefficients(
            current, torch.ones_like(current), sampler_seq_log_probs, 30.0
        )

    def compute_gepo_coefficient(
        self,
        learner_seq_log_probs: torch.Tensor,
        sampler_seq_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GEPO coefficient using Group Expectation Importance Weights.

        The GEPO coefficient is:
        coef = p_learner / E_q[q_sampler]

        where E_q[q] ≈ Σ q² / Σ q (normalized within-group probabilities)

        The frozen sampler scores give every response in the group a common
        denominator.

        Args:
            learner_seq_log_probs: Sequence log-probs from current policy [group_size]
            sampler_seq_log_probs: Sequence log-probs from sampling policy [group_size]

        Returns:
            GEPO coefficients for each sequence [group_size]
        """
        return self.compute_gepo_coefficient_static(
            learner_seq_log_probs, sampler_seq_log_probs
        )

    def compute_gepo_loss(
        self,
        learner_seq_log_probs: torch.Tensor,
        sampler_seq_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """GEPO objective for one prompt group via ``objectives.policy_loss``.

        Each sequence is one "token" carrying its summed log-prob; the whole
        call is one group, so the group-expectation denominator spans it.
        """
        logp_cur = learner_seq_log_probs.unsqueeze(-1)
        group_ids = torch.zeros(
            logp_cur.shape[0], dtype=torch.long, device=logp_cur.device
        )
        result = objectives.policy_loss(
            logp_cur=logp_cur,
            mask=torch.ones_like(logp_cur),
            advantages=advantages,
            objective=self._objective,
            logp_old=sampler_seq_log_probs.detach(),
            group_ids=group_ids,
        )
        return result.loss

    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute group-relative advantages using within-group baseline normalization.

        A_i = (r_i - mean(rewards)) / std(rewards)

        Args:
            rewards: Tensor of rewards for group [group_size]

        Returns:
            advantages: Normalized advantages [group_size]
            stats: Reward statistics
        """
        group_ids = torch.zeros(
            rewards.numel(), dtype=torch.long, device=rewards.device
        )
        objective = getattr(self, "_objective", None) or objectives.OBJECTIVES["gepo"]
        advantages = objectives.compute_advantages(rewards, group_ids, objective)

        std_reward = rewards.float().std(correction=0) if rewards.numel() > 1 else 0.0
        stats = {
            "mean_reward": rewards.mean().item(),
            "std_reward": float(std_reward),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
        }

        return advantages, stats

    async def generate_group_responses(
        self,
        prompt: str,
        group_size: int,
    ) -> list[dict[str, Any]]:
        """
        Generate a group of responses for a single prompt.

        Returns list of dicts with:
            - response: Generated text
            - input_ids: Tokenized full sequence
            - attention_mask: Attention mask
            - response_start_idx: Index where response starts
        """
        from stateset_agents.core.generation_evidence import selected_logprobs

        responses = []

        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
        )
        prompt_length = prompt_tokens["input_ids"].shape[1]

        with deterministic_scoring(self.model), torch.no_grad():
            for _ in range(group_size):
                # Generate response
                input_ids = prompt_tokens["input_ids"].to(self.device)
                attention_mask = prompt_tokens["attention_mask"].to(self.device)

                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_completion_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    num_beams=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                sequence = outputs.sequences[0]
                actions = sequence[prompt_length:]
                if not actions.numel():
                    raise ValueError("GEPO sampling returned no response tokens")
                # Generation scores include the sampler's temperature and filters.
                chosen = selected_logprobs(outputs.scores, actions[None], sampled=True)
                if chosen is None or not torch.isfinite(chosen).all():
                    raise ValueError(
                        "GEPO sampler returned a nonfinite action likelihood"
                    )
                response_text = self.tokenizer.decode(actions, skip_special_tokens=True)

                responses.append(
                    {
                        "response": response_text,
                        "input_ids": sequence,
                        "attention_mask": torch.ones_like(sequence),
                        "sampler_log_prob": chosen.sum().detach(),
                        "response_start_idx": prompt_length,
                    }
                )

        self.rollout_samples_total += len(responses)
        return responses

    async def train_step(
        self,
        prompts: list[str],
    ) -> dict[str, float]:
        """
        Execute one GEPO training step.

        For each prompt:
        1. Generate G responses (group)
        2. Compute rewards for each response
        3. Compute group advantages
        4. Compute GEPO coefficients
        5. Apply clipped policy gradient

        Args:
            prompts: List of prompts to train on

        Returns:
            Training metrics
        """
        self.model.train()

        all_rewards: list[float] = []
        all_gepo_coefs: list[float] = []

        # --- Rollout phase ------------------------------------------------
        # Generate every group and freeze the sampler (old-policy) sequence
        # log probs here, before any inner update can move the weights. When
        # they were recomputed from the just-updated model the GEPO
        # coefficient collapsed to ~1 and clipping never fired.
        rollouts: list[dict[str, Any]] = []
        for prompt in prompts:
            group_responses = await self.generate_group_responses(
                prompt, self.config.group_size
            )

            if len(group_responses) < 2:
                logger.warning("Insufficient responses for prompt, skipping")
                continue

            rewards = [
                self.reward_fn(prompt, resp["response"]) for resp in group_responses
            ]
            rewards_tensor = reward_tensor(rewards, self.device)
            all_rewards.extend(rewards)

            batch_input_ids, batch_attention_mask, _ = build_group_batch(
                group_responses, self.device, include_response_mask=False
            )
            response_start_idx = [row["response_start_idx"] for row in group_responses]
            score_device = (
                "cpu" if torch.device(self.device).type == "mps" else self.device
            )

            if all("sampler_log_prob" in row for row in group_responses):
                sampler_seq_log_probs = torch.stack(
                    [
                        torch.as_tensor(
                            row["sampler_log_prob"],
                            device=score_device,
                            dtype=torch.float64,
                        ).reshape(())
                        for row in group_responses
                    ]
                )
            else:
                # Legacy custom generators may only provide token IDs.
                with torch.no_grad():
                    _, sampler_seq_log_probs = self.compute_sequence_log_probs(
                        batch_input_ids, batch_attention_mask, response_start_idx
                    )
                for i, row in enumerate(group_responses):
                    if "sampler_log_prob" in row:
                        sampler_seq_log_probs[i] = row["sampler_log_prob"]
            if not torch.isfinite(sampler_seq_log_probs).all():
                raise ValueError("GEPO sampler likelihoods must be finite")

            rollouts.append(
                {
                    "input_ids": batch_input_ids,
                    "attention_mask": batch_attention_mask,
                    "rewards": rewards_tensor,
                    "mask": self.build_response_mask(
                        batch_attention_mask, response_start_idx
                    ),
                    "sampler_seq_log_probs": sampler_seq_log_probs.detach(),
                }
            )

        num_groups = len(rollouts)
        if not num_groups or not any(r["mask"].any() for r in rollouts):
            logger.warning("No sampled actions in GEPO batch")
            return {
                "policy_loss": 0.0,
                "average_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            }

        group_ids = torch.cat(
            [
                torch.full_like(r["rewards"], i, dtype=torch.long)
                for i, r in enumerate(rollouts)
            ]
        )
        advantages = objectives.compute_advantages(
            torch.cat([r["rewards"] for r in rollouts]), group_ids, self._objective
        ).to(self.device)
        all_advantages = advantages.tolist()
        sampler_scores = torch.cat([r["sampler_seq_log_probs"] for r in rollouts])
        mask = torch.nn.utils.rnn.pad_sequence(
            [row for r in rollouts for row in r["mask"]], batch_first=True
        )

        # --- Update phase --------------------------------------------------
        # ``num_gradient_updates`` (mu) inner updates reuse the same rollouts
        # against the frozen sampler log probs, exactly like DAPO. With the
        # default of 1 the formula is identical to the previous single-update
        # path (only dropout RNG ordering differs, since the sampler log probs
        # are now drawn before the learner ones).
        num_updates = max(1, getattr(self.config, "num_gradient_updates", 1))
        final_loss = torch.tensor(0.0, device=self.device)

        for _ in range(num_updates):
            all_gepo_coefs = []
            token_rows = []
            ref_rows = []
            kl_rows = []
            entropy_rows = []

            for rollout in rollouts:
                ids, attn = rollout["input_ids"], rollout["attention_mask"]
                logp = self._log_distribution(self.model, ids, attn, rollout["mask"])
                actions = ids[:, 1:].masked_fill(rollout["mask"] == 0, 0)[..., None]
                token_log_probs = logp.gather(-1, actions).squeeze(-1)
                if self._objective.entropy_coef > 0:
                    finite_logp = logp.masked_fill(logp.isneginf(), 0)
                    entropy_rows.extend((-(logp.exp() * finite_logp).sum(-1)).unbind())
                if self._objective.kl_coef > 0:
                    with torch.no_grad():
                        ref = self._log_distribution(
                            self.ref_model, ids, attn, rollout["mask"]
                        )
                    if self._objective.kl == "external":
                        zero = logp.isneginf()
                        delta = logp.masked_fill(zero, 0) - ref.masked_fill(zero, 0)
                        kl_rows.extend((logp.exp() * delta).sum(-1).unbind())
                    else:
                        ref_rows.extend(ref.gather(-1, actions).squeeze(-1).unbind())

                if self._objective.ratio != "group_expectation":
                    with torch.no_grad():
                        learner = (
                            (token_log_probs * rollout["mask"])
                            .to(sampler_scores)
                            .sum(-1)
                        )
                        all_gepo_coefs.extend(
                            self.compute_gepo_coefficient(
                                learner, rollout["sampler_seq_log_probs"]
                            ).tolist()
                        )
                token_rows.extend(token_log_probs.unbind())

            pad = torch.nn.utils.rnn.pad_sequence
            result = objectives.policy_loss(
                logp_cur=pad(token_rows, batch_first=True),
                logp_old=sampler_scores,
                mask=mask,
                advantages=advantages,
                objective=self._objective,
                group_ids=group_ids.to(self.device),
                logp_ref=pad(ref_rows, batch_first=True) if ref_rows else None,
                kl=pad(kl_rows, batch_first=True) if kl_rows else None,
                entropy=pad(entropy_rows, batch_first=True) if entropy_rows else None,
            )
            final_loss = result.loss
            if self._objective.ratio == "group_expectation":
                all_gepo_coefs = result.ratio.flatten().tolist()

            self.optimizer.zero_grad()
            final_loss.backward()

            clip_parameter_gradients(self.model.parameters(), self.config.max_grad_norm)

            self.optimizer.step()
            # Cadence convention shared with DAPO/VAPO: the LR scheduler
            # advances once per inner update, global_step counts train_steps.
            self.scheduler.step()

        self.global_step += 1

        # Compute metrics
        metrics = {
            "policy_loss": final_loss.item(),
            "average_reward": np.mean(all_rewards) if all_rewards else 0.0,
            "advantage_std": np.std(all_advantages) if all_advantages else 0.0,
            "gepo_coefficient_mean": np.mean(all_gepo_coefs) if all_gepo_coefs else 1.0,
            "gepo_coefficient_std": np.std(all_gepo_coefs) if all_gepo_coefs else 0.0,
            "learning_rate": self.scheduler.get_last_lr()[0],
            "global_step": self.global_step,
        }

        # Store metrics
        for key in ["policy_loss", "average_reward"]:
            if key in self.metrics_history:
                self.metrics_history[key].append(metrics[key])

        return metrics

    def save_checkpoint(self, output_dir: str) -> None:
        """Publish a complete resumable checkpoint."""
        self.save_runtime_checkpoint(output_dir)

    def load_checkpoint(self, checkpoint_dir: str, *, trusted: bool = False) -> bool:
        """Restore policy weights and all training components."""
        return self.load_runtime_checkpoint(checkpoint_dir, trusted=trusted)


async def train_with_gepo(
    model_name: str,
    reward_fn: Callable[[str, str], float],
    train_prompts: list[str],
    config: GEPOConfig | None = None,
    output_dir: str = "./outputs/gepo",
    use_wandb: bool = False,
    wandb_project: str | None = None,
) -> tuple[Any, Any, dict[str, list[float]]]:
    """
    Train a model using GEPO algorithm.

    Args:
        model_name: HuggingFace model name or path
        reward_fn: Function that takes (prompt, response) and returns reward
        train_prompts: List of training prompts
        config: GEPO configuration (uses defaults if None)
        output_dir: Directory to save checkpoints
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name

    Returns:
        Tuple of (model, tokenizer, metrics_history)
    """
    if config is None:
        config = GEPOConfig(model_name=model_name, output_dir=output_dir)
    _reject_unsupported_resume(config, "train_with_gepo")

    logger.info("=" * 60)
    logger.info("GEPO Training - Group Expectation Policy Optimization")
    logger.info("=" * 60)

    # Initialize W&B
    if use_wandb and wandb_project:
        _require_wandb()
        wandb.init(
            project=wandb_project,
            name=f"gepo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config.to_dict(),
            tags=["gepo", "rl-training"],
        )

    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    model_manager = GEPOModelManager(config)
    model, tokenizer = model_manager.load_model_and_tokenizer()

    # Create trainer
    trainer = GEPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        ref_model=model_manager.ref_model,
    )

    # Training loop
    logger.info(f"Starting training with {len(train_prompts)} prompts")
    logger.info(f"Group size: {config.group_size}")
    logger.info(f"Total iterations: {config.num_episodes}")

    os.makedirs(output_dir, exist_ok=True)

    for iteration in range(config.num_episodes):
        # Sample batch of prompts
        batch_size = min(config.per_device_train_batch_size, len(train_prompts))
        batch_indices = np.random.choice(len(train_prompts), batch_size, replace=False)
        batch_prompts = [train_prompts[i] for i in batch_indices]

        # Train step
        metrics = await trainer.train_step(batch_prompts)

        # Log metrics
        if iteration % config.logging_steps == 0:
            logger.info(
                f"Iteration {iteration}/{config.num_episodes} | "
                f"Loss: {metrics['policy_loss']:.4f} | "
                f"Reward: {metrics['average_reward']:.4f} | "
                f"GEPO Coef: {metrics['gepo_coefficient_mean']:.4f}"
            )

            if use_wandb:
                wandb.log(metrics, step=iteration)

        # Save checkpoint
        if (iteration + 1) % config.save_steps == 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{iteration + 1}")
            trainer.save_checkpoint(checkpoint_dir)

    # Save final model
    final_dir = os.path.join(output_dir, "final")
    trainer.save_checkpoint(final_dir)
    trainer.metrics_history["rollout_samples_total"] = [
        float(trainer.rollout_samples_total)
    ]

    if use_wandb:
        wandb.finish()

    logger.info("=" * 60)
    logger.info("GEPO Training Complete!")
    logger.info("=" * 60)

    return model, tokenizer, trainer.metrics_history


# Export
__all__ = [
    "GEPOConfig",
    "GEPOTrainer",
    "train_with_gepo",
]
