"""
Advanced RL Algorithms for StateSet Agents

The experimental orchestrator provides Proximal Policy Optimization (PPO),
Direct Preference Optimization (DPO) and Advantage Actor-Critic (A2C).
The full GSPO implementation lives in stateset_agents.training.gspo_trainer.

Not the supported training path. ``PPOTrainer`` here is a second, older PPO
implementation kept for the ``AdvancedRLOrchestrator`` experiments; the
maintained one is ``stateset_agents.training.ppo_trainer``, which is what the
CLI, the trainer registry and the shared loss primitives in
``training/rl_losses.py`` use. Prefer the ``training/`` trainers for anything
you intend to run; treat the classes below as research scaffolding whose
numerics are not kept in step with the ``training/`` spine.
"""

from __future__ import annotations

import asyncio
import logging
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.distributions import Categorical

from ..environment import Environment
from ..trajectory import ConversationTurn, MultiTurnTrajectory
from .enhanced_agent import EnhancedMultiTurnAgent

if TYPE_CHECKING:
    from ...training.gspo_config import GSPOConfig

logger = logging.getLogger(__name__)

FloatArray: TypeAlias = NDArray[np.float32]


def _require_tokenizer(agent: EnhancedMultiTurnAgent) -> Any:
    """Return the agent tokenizer or raise if unavailable."""
    tokenizer = getattr(agent, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Agent tokenizer is required for advanced RL trainers")
    return tokenizer


def _require_model(agent: EnhancedMultiTurnAgent) -> Any:
    """Return the agent model or raise if unavailable."""
    model = getattr(agent, "model", None)
    if model is None:
        raise RuntimeError("Agent model is required for advanced RL trainers")
    return model


def _extract_step_result(
    step_result: Any,
) -> tuple[Any, ConversationTurn | None, float, bool]:
    """Normalize environment.step outputs across legacy and current contracts."""
    if not isinstance(step_result, tuple) or len(step_result) != 4:
        raise RuntimeError("Unsupported environment step result")

    next_state = step_result[0]
    second = step_result[1]
    third = step_result[2]
    fourth = step_result[3]

    if isinstance(second, ConversationTurn) or (
        second is None and isinstance(fourth, bool)
    ):
        return next_state, second, float(third), bool(fourth)

    info = fourth if isinstance(fourth, dict) else {}
    observation_turn = info.get("observation_turn")
    normalized_turn = (
        observation_turn if isinstance(observation_turn, ConversationTurn) else None
    )
    return next_state, normalized_turn, float(second), bool(third)


def _rollout_advantages(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    trace_decay: float,
    bootstraps: dict[int, float],
) -> list[float]:
    return _rollout_targets(rewards, values, dones, gamma, trace_decay, bootstraps)[0]


def _rollout_targets(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    trace_decay: float,
    bootstraps: dict[int, float],
) -> tuple[list[float], list[float]]:
    """Close each collected segment at its terminal flag or frozen time-limit value."""
    from ..rust_accelerator import compute_gae_targets

    if not len(rewards) == len(values) == len(dones):
        raise ValueError("rollout rewards, values, and dones must align")
    advantages: list[float] = []
    targets: list[float] = []
    start = 0
    for i, done in enumerate(dones):
        if not done and i not in bootstraps and i + 1 < len(rewards):
            continue
        end = i + 1
        final_value = 0.0 if done else bootstraps.get(i, 0.0)
        adv, ret = compute_gae_targets(
            np.asarray(rewards[start:end]),
            np.asarray([*values[start:end], final_value]),
            gamma,
            trace_decay,
        )
        advantages.extend(adv.tolist())
        targets.extend(ret.tolist())
        start = end
    return advantages, targets


def _actor_distribution(logits: torch.Tensor) -> Categorical:
    """Center before Categorical subtracts its log partition in the input dtype."""
    scores = (
        logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    )
    return Categorical(logits=scores - scores.amax(-1, keepdim=True).detach())


def _actor_critic_targets(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    trace_decay: float,
    bootstraps: dict[int, float],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Freeze lambda returns and standardize the actor signal before narrowing."""
    from ...training.advantage_estimation import normalize_advantages

    raw, targets = _rollout_targets(
        rewards, values, dones, gamma, trace_decay, bootstraps
    )
    work = "cpu" if device.type == "mps" else device
    weights = normalize_advantages(torch.tensor(raw, dtype=torch.float64, device=work))
    return weights, torch.tensor(targets, dtype=torch.float64, device=work)


@dataclass
class PPOConfig:
    """Configuration for PPO training"""

    learning_rate: float = 3e-4
    n_epochs: int = 10
    batch_size: int = 64
    mini_batch_size: int = 16
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    target_kl: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95


@dataclass
class DPOConfig:
    """Configuration for DPO training"""

    learning_rate: float = 5e-7
    batch_size: int = 32
    beta: float = 0.1  # Temperature parameter
    max_length: int = 512
    max_grad_norm: float = 1.0
    warmup_steps: int = 100


@dataclass
class A2CConfig:
    """Configuration for A2C training"""

    learning_rate: float = 7e-4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 5


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO and A2C"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Actor head
        self.actor = nn.Linear(hidden_size, output_size)

        # Critic head
        self.critic = nn.Linear(hidden_size, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        shared_output = self.shared_layers(x)
        action_logits = self.actor(shared_output)
        value = self.critic(shared_output)
        return action_logits, value

    def get_action(self, x, deterministic=False):
        with torch.no_grad():
            logits, value = self(x)

            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                dist = _actor_distribution(logits)
                action = dist.sample()

            return action.item(), value.item()


class PPOTrainer:
    """
    Proximal Policy Optimization trainer
    """

    def __init__(
        self, agent: EnhancedMultiTurnAgent, config: PPOConfig, device: str = "auto"
    ):
        self.agent = agent
        self.config = config
        tokenizer = _require_tokenizer(agent)

        self.device = torch.device(
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize networks
        vocab_size = len(tokenizer)
        width = getattr(_require_model(agent).config, "hidden_size", 768)
        self.actor_critic = ActorCriticNetwork(
            input_size=width,
            hidden_size=256,
            output_size=vocab_size,
        ).to(self.device)

        self.old_actor_critic = ActorCriticNetwork(
            input_size=width, hidden_size=256, output_size=vocab_size
        ).to(self.device)

        # Copy parameters
        self.old_actor_critic.load_state_dict(self.actor_critic.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=config.learning_rate
        )

        # Storage for trajectories
        self.states: list[FloatArray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.bootstraps: dict[int, float] = {}

    async def collect_trajectory(
        self, environment: Environment, max_steps: int = 100
    ) -> MultiTurnTrajectory:
        """Collect a trajectory using current policy"""

        state = await environment.reset()
        trajectory_turns: list[ConversationTurn] = []
        trajectory_rewards: list[float] = []

        messages = [
            {"role": "user", "content": str(environment.observation_context(state))}
        ]
        for _step in range(max_steps):
            # Generate embedding for state
            state_embedding = await self._get_state_embedding(messages)

            # Get action from policy
            state_tensor = (
                torch.tensor(state_embedding, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                logits, value_tensor = self.actor_critic(state_tensor)
                dist = _actor_distribution(logits)
                action_tensor = dist.sample()
                action_token = int(action_tensor.item())
                value = float(value_tensor.item())
                log_prob = float(dist.log_prob(action_tensor).item())

            # Convert token to text
            tokenizer = _require_tokenizer(self.agent)
            action_text = str(tokenizer.decode([action_token]))

            # Create conversation turn
            agent_turn = ConversationTurn(role="assistant", content=action_text)

            # Step environment
            step_result = await environment.step(state, agent_turn)
            new_state, env_response, reward, done = _extract_step_result(step_result)

            # Store transition
            self.states.append(state_embedding)
            self.actions.append(action_token)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.rewards.append(reward)
            self.dones.append(done)
            trajectory_rewards.append(reward)

            trajectory_turns.append(agent_turn)
            if env_response:
                trajectory_turns.append(env_response)

            messages.append({"role": "assistant", "content": action_text})
            observation = (
                env_response.content
                if env_response
                else environment.observation_context(new_state)
            )
            messages.append({"role": "user", "content": str(observation)})
            state = new_state

            if done:
                break

        if trajectory_rewards and not self.dones[-1]:
            state_embedding = await self._get_state_embedding(messages)
            with torch.no_grad():
                _, bootstrap = self.actor_critic(
                    torch.as_tensor(state_embedding, device=self.device)
                )
            self.bootstraps[len(self.rewards) - 1] = bootstrap.item()

        trajectory = MultiTurnTrajectory(
            turns=trajectory_turns,
            total_reward=sum(trajectory_rewards),
            metadata={"algorithm": "PPO"},
        )

        return trajectory

    async def _get_state_embedding(self, messages: list[dict[str, str]]) -> FloatArray:
        """Get embedding representation of conversation state"""
        # Simple approach: concatenate last few messages
        conversation_text = " ".join([msg["content"] for msg in messages[-3:]])
        tokenizer = _require_tokenizer(self.agent)
        model = _require_model(self.agent)

        # Use tokenizer to get embeddings (simplified)
        inputs = tokenizer(
            conversation_text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states")
            # The last causal hidden state has seen the full tokenized observation.
            embedding = np.asarray(
                hidden_states[-1][:, -1, :].cpu().numpy().flatten(),
                dtype=np.float32,
            )

        return cast(FloatArray, embedding)

    def compute_advantages(
        self, rewards: list[float], values: list[float], dones: list[bool]
    ) -> list[float]:
        """Compute Generalized Advantage Estimation (GAE)"""
        return _rollout_advantages(
            rewards, values, dones, self.config.gamma, self.config.gae_lambda, {}
        )

    async def train_step(self):
        """Execute one PPO training step"""
        from ...training.rl_losses import safe_exp_ratio
        from ...training.trainer_utils import clip_parameter_gradients

        if not self.states:
            return {"loss": 0.0}

        # Convert lists to tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(
            self.device
        )
        actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(
            self.device
        )
        advantages, returns = _actor_critic_targets(
            self.rewards,
            self.values,
            self.dones,
            self.config.gamma,
            self.config.gae_lambda,
            self.bootstraps,
            self.device,
        )

        # PPO training loop
        for _ in range(self.config.ppo_epochs):
            # Get current policy outputs
            logits, current_values = self.actor_critic(states)
            dist = _actor_distribution(logits)
            current_log_probs = dist.log_prob(actions)

            # Compute ratios
            ratios = safe_exp_ratio(current_log_probs - old_log_probs)

            # Compute surrogate losses
            ratios = ratios.to(advantages)
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(
                    ratios, 1 - self.config.clip_param, 1 + self.config.clip_param
                )
                * advantages
            )

            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(current_values.squeeze(-1).to(returns), returns)

            # Entropy bonus
            entropy = dist.entropy().mean()

            # Total loss
            total_loss = (
                policy_loss
                + self.config.value_loss_coef * value_loss
                - self.config.entropy_coef * entropy.to(value_loss)
            )

            # Update network
            self.optimizer.zero_grad()
            total_loss.backward()
            clip_parameter_gradients(
                self.actor_critic.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

        # Update old policy
        self.old_actor_critic.load_state_dict(self.actor_critic.state_dict())

        # Clear trajectory buffers
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.bootstraps.clear()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
        }


class DPOTrainer:
    """
    Direct Preference Optimization trainer
    """

    def __init__(
        self, agent: EnhancedMultiTurnAgent, config: DPOConfig, device: str = "auto"
    ):
        self.agent = agent
        self.config = config
        model = _require_model(agent)
        tokenizer = _require_tokenizer(agent)

        self.device = (
            next(model.parameters()).device
            if device == "auto"
            else torch.device(device)
        )
        if next(model.parameters()).device != self.device:
            model.to(self.device)

        from ...training.adamw_weights import AdamWWeights

        self._weights = AdamWWeights(model.parameters(), lr=config.learning_rate)
        self.optimizer = self._weights.optimizer

        # Reference model (frozen copy)
        self.reference_model = self._create_reference_model()
        self.tokenizer = tokenizer

    def _create_reference_model(self):
        """Create a frozen reference model"""
        import copy

        reference_model = copy.deepcopy(_require_model(self.agent))

        # Freeze parameters
        for param in reference_model.parameters():
            param.requires_grad = False

        return reference_model.to(self.device).eval()

    def _completion_ids(self, prompt: str, completion: str) -> tuple[torch.Tensor, int]:
        """Encode the same prompt prefix for both candidates and keep its boundary."""
        if self.config.max_length <= 0:
            raise ValueError("DPO max_length must be positive")
        if not completion.strip():
            raise ValueError("DPO requires a nonempty completion")
        prefix = self.tokenizer.encode(prompt, add_special_tokens=False)
        ids = self.tokenizer.encode(prompt + " " + completion, add_special_tokens=False)
        if ids[: len(prefix)] != prefix:
            raise ValueError(
                "DPO prompt tokenization changes at the completion boundary"
            )
        if self.tokenizer.bos_token_id is not None:
            prefix = [self.tokenizer.bos_token_id, *prefix]
            ids = [self.tokenizer.bos_token_id, *ids]
        ids = ids[: self.config.max_length]
        if not prefix or len(ids) <= len(prefix):
            raise ValueError(
                "DPO requires a prompt context and at least one scored token"
            )
        return torch.tensor([ids], device=self.device), len(prefix)

    def _completion_evidence(
        self,
        model: nn.Module,
        ids: torch.Tensor,
        start: int,
        cancel: int,
        alternate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from ...training.preference_loss import completion_terms

        live = completion_terms(
            model(input_ids=ids).logits, ids, start, cancel, alternate
        )
        with torch.no_grad():
            ref = completion_terms(
                self.reference_model(input_ids=ids).logits,
                ids,
                start,
                cancel,
                alternate,
            )
        return live, ref

    def _pair_loss(self, model: nn.Module, pair: dict[str, Any]) -> torch.Tensor:
        from ...training.preference_loss import preference_pair_loss

        c_ids, c_start = self._completion_ids(pair["prompt"], pair["chosen"])
        r_ids, r_start = self._completion_ids(pair["prompt"], pair["rejected"])
        c_tokens, r_tokens = c_ids[0, c_start:], r_ids[0, r_start:]
        cancel = min(c_tokens.numel(), r_tokens.numel())
        differences = (c_tokens[:cancel] != r_tokens[:cancel]).nonzero()
        alternate = None
        if differences.numel():
            first = int(differences[0, 0])
            alternate = r_tokens[first]
            cancel = first + 1
        chosen, ref_chosen = self._completion_evidence(
            model, c_ids, c_start, cancel, alternate
        )
        rejected, ref_rejected = self._completion_evidence(
            model, r_ids, r_start, cancel
        )
        return preference_pair_loss(
            chosen, rejected, ref_chosen, ref_rejected, self.config.beta
        )

    async def train_step(
        self, preference_pairs: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Take one gradient step on the mean reference-adjusted preference loss."""
        if not preference_pairs:
            return {"dpo_loss": 0.0}

        from ...training.trainer_utils import _clip_gradient_norm, deterministic_scoring

        model = _require_model(self.agent)
        model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad()
        losses = []
        try:
            with deterministic_scoring(model, self.reference_model):
                for pair in preference_pairs:
                    loss = self._pair_loss(model, pair)
                    if not torch.isfinite(loss):
                        raise ValueError(
                            "DPO preference loss exceeds the finite scoring range"
                        )
                    mean_term = loss / len(preference_pairs)
                    losses.append(mean_term.item())
                    mean_term.backward()
                    if self.optimizer is self._weights.optimizer:
                        self._weights.collect_gradients()
            gradients = [
                p.grad
                for group in self.optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            _clip_gradient_norm(gradients, self.config.max_grad_norm, torch)
            mean_loss = math.fsum(losses)
        except BaseException:
            model.zero_grad(set_to_none=True)
            self.optimizer.zero_grad(set_to_none=True)
            raise

        if self.optimizer is self._weights.optimizer:
            self._weights.step(publish_gradients=True)
        else:
            self.optimizer.step()
        return {"dpo_loss": mean_loss}


class A2CTrainer:
    """
    Advantage Actor-Critic trainer
    """

    def __init__(
        self, agent: EnhancedMultiTurnAgent, config: A2CConfig, device: str = "auto"
    ):
        self.agent = agent
        self.config = config
        tokenizer = _require_tokenizer(agent)

        self.device = torch.device(
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize actor-critic network
        vocab_size = len(tokenizer)
        width = getattr(_require_model(agent).config, "hidden_size", 768)
        self.actor_critic = ActorCriticNetwork(
            input_size=width, hidden_size=256, output_size=vocab_size
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.RMSprop(
            self.actor_critic.parameters(),
            lr=config.learning_rate,
            alpha=0.99,
            eps=1e-5,
        )

        self._rollout_lock = asyncio.Lock()

        # Storage
        self.states: list[FloatArray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []
        self.dones: list[bool] = []
        self.bootstraps: dict[int, float] = {}

    async def collect_experience(
        self, environment: Environment, n_steps: int
    ) -> list[float]:
        """Collect and publish one complete segment under the current policy."""
        async with self._rollout_lock:
            return await self._collect_segment(environment, n_steps)

    async def _collect_segment(
        self, environment: Environment, n_steps: int
    ) -> list[float]:
        state = await environment.reset()
        messages = [
            {"role": "user", "content": str(environment.observation_context(state))}
        ]
        segment: list[tuple[FloatArray, int, float, float, float, bool]] = []
        tokenizer = _require_tokenizer(self.agent)
        for _ in range(n_steps):
            embedding = (await self._get_state_embedding(messages)).copy()
            with torch.no_grad():
                logits, value = self.actor_critic(
                    torch.as_tensor(
                        embedding, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                )
                policy = _actor_distribution(logits)
                action = policy.sample()
                token = int(action.item())
                log_prob = policy.log_prob(action).item()
            text = str(tokenizer.decode([token]))
            state, reply, reward, done = _extract_step_result(
                await environment.step(
                    state, ConversationTurn(role="assistant", content=text)
                )
            )
            segment.append((embedding, token, reward, value.item(), log_prob, done))
            if done:
                break
            observation = (
                reply.content if reply else environment.observation_context(state)
            )
            messages.extend(
                [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": str(observation)},
                ]
            )

        if segment and not segment[-1][-1]:
            embedding = await self._get_state_embedding(messages)
            with torch.no_grad():
                _, bootstrap = self.actor_critic(
                    torch.as_tensor(embedding, dtype=torch.float32, device=self.device)
                )
            self.bootstraps[len(self.rewards) + len(segment) - 1] = bootstrap.item()

        # No asynchronous callbacks remain. A failed step or bootstrap leaves
        # the previously published segments intact.
        for embedding, token, reward, value_estimate, log_prob, done in segment:
            self.states.append(embedding)
            self.actions.append(token)
            self.rewards.append(reward)
            self.values.append(value_estimate)
            self.log_probs.append(log_prob)
            self.dones.append(done)
        return [row[2] for row in segment]

    async def _get_state_embedding(self, messages: list[dict[str, str]]) -> FloatArray:
        """Get embedding for state (simplified)"""
        conversation_text = " ".join(msg["content"] for msg in messages[-3:])
        tokenizer = _require_tokenizer(self.agent)
        model = _require_model(self.agent)

        inputs = tokenizer(
            conversation_text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states")
            embedding = np.asarray(
                hidden_states[-1][:, -1, :].cpu().numpy().flatten(),
                dtype=np.float32,
            )

        return cast(FloatArray, embedding)

    async def train_step(self) -> dict[str, float]:
        """Consume complete segments and apply one synchronous A2C update."""
        async with self._rollout_lock:
            return self._update()

    def _update(self) -> dict[str, float]:
        from ...training.trainer_utils import _clip_gradient_norm

        if not self.states:
            return {"loss": 0.0}

        # Convert to tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(
            self.device
        )
        actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
        advantages, returns = _actor_critic_targets(
            self.rewards,
            self.values,
            self.dones,
            self.config.gamma,
            self.config.gae_lambda,
            self.bootstraps,
            self.device,
        )

        # Get current policy outputs
        logits, current_values = self.actor_critic(states)
        dist = _actor_distribution(logits)
        current_log_probs = dist.log_prob(actions)

        # Compute losses
        policy_loss = -(current_log_probs.to(advantages) * advantages).mean()
        value_loss = F.mse_loss(current_values.squeeze(-1).to(returns), returns)
        entropy = dist.entropy().mean()

        total_loss = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            - self.config.entropy_coef * entropy.to(value_loss)
        )

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        gradients = [
            p.grad for p in self.actor_critic.parameters() if p.grad is not None
        ]
        _clip_gradient_norm(gradients, self.config.max_grad_norm, torch)
        self.optimizer.step()

        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.bootstraps.clear()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
        }


class AdvancedRLOrchestrator:
    """
    Orchestrator for multiple RL algorithms with automatic algorithm selection
    """

    def __init__(self, agent: EnhancedMultiTurnAgent):
        self.agent = agent
        self.algorithms: dict[str, object] = {}
        self.current_algorithm: str | None = None
        self.performance_history: list[dict[str, Any]] = []

    def add_algorithm(
        self, name: str, trainer: PPOTrainer | DPOTrainer | A2CTrainer | GSPOTrainerStub
    ) -> None:
        """Add an RL algorithm"""
        self.algorithms[name] = trainer

    def select_algorithm(
        self, task_type: str, data_characteristics: dict[str, Any]
    ) -> str:
        """Automatically select the best algorithm for the task"""

        if task_type == "preference_learning" or data_characteristics.get(
            "has_preferences"
        ):
            return "dpo"
        elif data_characteristics.get("online_learning"):
            return "a2c"
        else:
            return "ppo"  # Default to PPO

    async def train(
        self,
        environment: Environment,
        training_data: dict[str, Any],
        task_type: str = "general",
        **kwargs,
    ):
        """Train using the best algorithm for the task"""

        # Analyze data characteristics
        data_characteristics = self._analyze_data(training_data)

        # Select algorithm
        algorithm_name = self.select_algorithm(task_type, data_characteristics)
        trainer = self.algorithms.get(algorithm_name)
        self.current_algorithm = algorithm_name

        if not trainer:
            logger.warning(f"Algorithm {algorithm_name} not available, using PPO")
            trainer = self.algorithms.get("ppo")
            algorithm_name = "ppo"

        logger.info(f"Selected algorithm: {algorithm_name}")

        # Train with selected algorithm
        if algorithm_name == "ppo" and isinstance(trainer, PPOTrainer):
            return await self._train_ppo(trainer, environment, **kwargs)
        elif algorithm_name == "dpo" and isinstance(trainer, DPOTrainer):
            return await self._train_dpo(trainer, training_data, **kwargs)
        elif algorithm_name == "a2c" and isinstance(trainer, A2CTrainer):
            return await self._train_a2c(trainer, environment, **kwargs)
        elif algorithm_name == "gspo" and isinstance(trainer, GSPOTrainerStub):
            return await trainer.train_step(environment, **kwargs)
        return None

    def _analyze_data(self, training_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze training data characteristics"""
        characteristics = {}

        if "preference_pairs" in training_data:
            characteristics["has_preferences"] = True

        if "trajectories" in training_data:
            characteristics["has_trajectories"] = True

        # Add more analysis logic here

        return characteristics

    async def _train_ppo(
        self, trainer: PPOTrainer, environment: Environment, num_episodes: int = 100
    ):
        """Train with PPO"""
        for episode in range(num_episodes):
            # Collect trajectory
            await trainer.collect_trajectory(environment)

            # Train step
            metrics = await trainer.train_step()

            logger.info(f"PPO Episode {episode}: Loss = {metrics['total_loss']:.4f}")

    async def _train_dpo(
        self, trainer: DPOTrainer, training_data: dict[str, Any], num_steps: int = 100
    ):
        """Train with DPO"""
        preference_pairs = training_data.get("preference_pairs", [])

        for step in range(num_steps):
            # Sample batch of preference pairs
            # np.random.choice returns an ndarray of objects; train_step wants
            # the list of preference dicts it was given.
            batch = list(
                np.random.choice(
                    preference_pairs,
                    size=min(len(preference_pairs), trainer.config.batch_size),
                )
            )

            # Train step
            metrics = await trainer.train_step(batch)

            logger.info(f"DPO Step {step}: Loss = {metrics['dpo_loss']:.4f}")

    async def _train_a2c(
        self, trainer: A2CTrainer, environment: Environment, num_episodes: int = 100
    ):
        """Train with A2C"""
        for episode in range(num_episodes):
            # Collect experience
            rewards = await trainer.collect_experience(
                environment, trainer.config.n_steps
            )

            # Train step
            await trainer.train_step()

            logger.info(f"A2C Episode {episode}: Total Reward = {sum(rewards):.2f}")


# Factory functions


def create_ppo_trainer(agent: EnhancedMultiTurnAgent, **kwargs) -> PPOTrainer:
    """Create PPO trainer"""
    config = PPOConfig(**kwargs)
    return PPOTrainer(agent, config)


def create_dpo_trainer(agent: EnhancedMultiTurnAgent, **kwargs) -> DPOTrainer:
    """Create DPO trainer"""
    config = DPOConfig(**kwargs)
    return DPOTrainer(agent, config)


def create_a2c_trainer(agent: EnhancedMultiTurnAgent, **kwargs) -> A2CTrainer:
    """Create A2C trainer"""
    config = A2CConfig(**kwargs)
    return A2CTrainer(agent, config)


class GSPOTrainerStub:
    """
    Stub for GSPO trainer integration.

    The real implementation is
    :class:`stateset_agents.training.gspo_trainer.GSPOTrainer`; this stub only
    provides a consistent interface for :class:`AdvancedRLOrchestrator` and is
    configured with the canonical
    :class:`stateset_agents.training.gspo_config.GSPOConfig`.
    """

    def __init__(
        self,
        agent: EnhancedMultiTurnAgent,
        config: GSPOConfig,
        device: str = "auto",
    ):
        self.agent = agent
        self.config = config
        self.device = torch.device(
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info("GSPO trainer stub initialized")
        logger.info(
            "For full GSPO training, use: from stateset_agents.training.gspo_trainer import train_with_gspo"
        )

    async def train_step(self, environment: Environment, **kwargs):
        """Train step stub - directs to full implementation"""
        logger.warning(
            "GSPO stub called. Use training.gspo_trainer for full implementation."
        )
        return {"info": "Use training.gspo_trainer.train_with_gspo for full training"}


def create_gspo_trainer(agent: EnhancedMultiTurnAgent, **kwargs) -> GSPOTrainerStub:
    """
    Create GSPO trainer stub.

    For production GSPO training, use:
        from stateset_agents.training.gspo_trainer import train_with_gspo
        await train_with_gspo(config, agent, environment, reward_model)
    """
    from ...training.gspo_config import GSPOConfig

    config = GSPOConfig(**kwargs)
    return GSPOTrainerStub(agent, config)


def create_advanced_rl_orchestrator(
    agent: EnhancedMultiTurnAgent,
) -> AdvancedRLOrchestrator:
    """Create advanced RL orchestrator with all algorithms"""
    orchestrator = AdvancedRLOrchestrator(agent)

    # Add all algorithms
    orchestrator.add_algorithm("ppo", create_ppo_trainer(agent))
    orchestrator.add_algorithm("dpo", create_dpo_trainer(agent))
    orchestrator.add_algorithm("a2c", create_a2c_trainer(agent))
    orchestrator.add_algorithm("gspo", create_gspo_trainer(agent))

    return orchestrator


def __getattr__(name: str) -> Any:
    """Serve the removed duplicate ``GSPOConfig`` from its canonical home."""
    if name == "GSPOConfig":
        warnings.warn(
            "import GSPOConfig from stateset_agents.training.gspo_config",
            DeprecationWarning,
            stacklevel=2,
        )
        from ...training.gspo_config import GSPOConfig

        return GSPOConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
