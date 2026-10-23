"""
Advanced RL Algorithms for StateSet Agents

Research scaffolding: an Advantage Actor-Critic (A2C) trainer, a DPO trainer
and an ``AdvancedRLOrchestrator`` that picks between them. Not the supported
training path; everything here is deprecated and removed in v0.71.0. Use the
``training/`` trainers (GRPO, GSPO, DAPO, GEPO, VAPO),
``stateset_agents.experimental.ppo`` for PPO and
``stateset_agents.training.preference`` for DPO and its relatives. The older
PPO duplicate and the GSPO stub were removed in v0.70.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.distributions import Categorical

from ..._deprecation import warn_deprecated
from ..environment import Environment
from ..numerics import whiten
from ..trajectory import ConversationTurn
from .enhanced_agent import EnhancedMultiTurnAgent

logger = logging.getLogger(__name__)

_REMOVAL_RELEASE = "0.71.0"


def _warn_deprecated(name: str, replacement: str) -> None:
    warn_deprecated(
        f"stateset_agents.core.enhanced.{name}",
        removal=_REMOVAL_RELEASE,
        replacement=replacement,
        stacklevel=3,
    )


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

    if isinstance(second, ConversationTurn):
        return next_state, second, float(third), bool(fourth)

    info = fourth if isinstance(fourth, dict) else {}
    observation_turn = info.get("observation_turn")
    normalized_turn = (
        observation_turn if isinstance(observation_turn, ConversationTurn) else None
    )
    return next_state, normalized_turn, float(second), bool(third)


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
    """Actor-Critic network for A2C"""

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
                dist = Categorical(logits=logits)
                action = dist.sample()

            return action.item(), value.item()


class DPOTrainer:
    """
    Direct Preference Optimization trainer.

    .. deprecated:: 0.70.0
        Research scaffolding. Use the maintained
        :class:`stateset_agents.training.preference.PreferenceTrainer`
        (DPO, IPO, SimPO, KTO, ORPO). Removed in v0.71.0.
    """

    def __init__(
        self, agent: EnhancedMultiTurnAgent, config: DPOConfig, device: str = "auto"
    ):
        warn_deprecated(
            "stateset_agents.core.enhanced.DPOTrainer",
            removal="0.71.0",
            replacement="stateset_agents.training.preference.PreferenceTrainer",
        )
        self.agent = agent
        self.config = config
        model = _require_model(agent)
        tokenizer = _require_tokenizer(agent)

        self.device = torch.device(
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

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

        return reference_model.to(self.device)

    def _encode_pair(self, prompt: str, response: str) -> tuple[Any, int]:
        """Token ids of ``prompt`` + ``response`` and where the response starts.

        Prompt and response are tokenized separately so the boundary is exact;
        the response is truncated to fit ``max_length``.
        """
        prompt_ids = list(self.tokenizer(prompt)["input_ids"])
        response_ids = list(
            self.tokenizer(" " + response, add_special_tokens=False)["input_ids"]
        )
        room = max(self.config.max_length - len(prompt_ids), 0)
        ids = prompt_ids + response_ids[:room]
        tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        return tensor, len(prompt_ids)

    @staticmethod
    def _response_log_prob(model: Any, input_ids: Any, start: int) -> Any:
        """Sum of log pi(token) over the response tokens (shifted labels)."""
        logits = model(input_ids=input_ids).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        token_log_probs = (
            F.log_softmax(logits.float(), dim=-1)
            .gather(-1, labels.unsqueeze(-1))
            .squeeze(-1)
        )
        # Label position j predicts token j + 1; response tokens start at
        # ``start``, so their labels start at ``start - 1``.
        return token_log_probs[:, max(start - 1, 0) :].sum()

    async def train_step(self, preference_pairs: list[dict[str, Any]]):
        """
        Train on preference pairs

        Each preference pair contains:
        - prompt: the input prompt
        - chosen: the preferred response
        - rejected: the non-preferred response
        """

        total_loss = 0.0
        num_pairs = 0
        model = _require_model(self.agent)
        model.train()
        self.reference_model.eval()
        self.optimizer.zero_grad()

        for pair in preference_pairs:
            prompt = pair["prompt"]
            chosen = pair["chosen"]
            rejected = pair["rejected"]

            # DPO: -log sigmoid(beta * [(log pi - log pi_ref)(y_w) -
            #                           (log pi - log pi_ref)(y_l)]),
            # with log pi(y|x) the summed log-prob of the response tokens.
            chosen_ids, chosen_start = self._encode_pair(prompt, chosen)
            rejected_ids, rejected_start = self._encode_pair(prompt, rejected)

            policy_chosen = self._response_log_prob(model, chosen_ids, chosen_start)
            policy_rejected = self._response_log_prob(
                model, rejected_ids, rejected_start
            )
            with torch.no_grad():
                ref_chosen = self._response_log_prob(
                    self.reference_model, chosen_ids, chosen_start
                )
                ref_rejected = self._response_log_prob(
                    self.reference_model, rejected_ids, rejected_start
                )

            chosen_ratio = policy_chosen - ref_chosen
            rejected_ratio = policy_rejected - ref_rejected
            loss = -F.logsigmoid(self.config.beta * (chosen_ratio - rejected_ratio))

            total_loss += loss.item()
            num_pairs += 1

            # Mean over pairs in the batch.
            (loss / max(len(preference_pairs), 1)).backward()

        # Update parameters
        model = _require_model(self.agent)
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {"dpo_loss": total_loss / max(num_pairs, 1)}


class A2CTrainer:
    """
    Advantage Actor-Critic trainer.

    .. deprecated:: 0.70.0
        Research scaffolding with no maintained counterpart. Use the
        ``training/`` trainers. Removed in v0.71.0.
    """

    def __init__(
        self, agent: EnhancedMultiTurnAgent, config: A2CConfig, device: str = "auto"
    ):
        _warn_deprecated("A2CTrainer", "the stateset_agents.training trainers")
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
        self.actor_critic = ActorCriticNetwork(
            input_size=768, hidden_size=256, output_size=vocab_size
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.RMSprop(
            self.actor_critic.parameters(),
            lr=config.learning_rate,
            alpha=0.99,
            eps=1e-5,
        )

        # Storage
        self.states: list[FloatArray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []

    async def collect_experience(self, environment: Environment, n_steps: int):
        """Collect experience for n steps"""

        state = await environment.reset()
        episode_rewards: list[float] = []

        for _step in range(n_steps):
            # Get state embedding
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            state_embedding = await self._get_state_embedding(messages)

            # Get action
            action_token, value = self.actor_critic.get_action(
                torch.tensor(state_embedding, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            # Convert to text
            tokenizer = _require_tokenizer(self.agent)
            action_text = str(tokenizer.decode([action_token]))

            # Step environment
            agent_turn = ConversationTurn(role="assistant", content=action_text)
            step_result = await environment.step(state, agent_turn)
            new_state, env_response, reward, done = _extract_step_result(step_result)

            # Store transition
            self.states.append(state_embedding)
            self.actions.append(action_token)
            self.rewards.append(reward)
            self.values.append(value)

            # Get log prob for the action taken
            with torch.no_grad():
                logits, _ = self.actor_critic(
                    torch.tensor(state_embedding, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                dist = Categorical(logits=logits)
                log_prob = dist.log_prob(torch.tensor(action_token).to(self.device))
                self.log_probs.append(log_prob.item())

            episode_rewards.append(reward)
            state = new_state

            if done:
                break

        return episode_rewards

    async def _get_state_embedding(self, messages: list[dict[str, str]]) -> FloatArray:
        """Get embedding for state (simplified)"""
        conversation_text = messages[-1]["content"] if messages else ""
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
            embedding: FloatArray = np.asarray(
                hidden_states[-1][:, 0, :].cpu().numpy().flatten(),
                dtype=np.float32,
            )

        return embedding

    async def train_step(self):
        """Execute A2C training step"""

        if not self.states:
            return {"loss": 0.0}

        # Convert to tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(
            self.device
        )
        actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
        values = torch.tensor(self.values, dtype=torch.float32).to(self.device)

        # Compute returns and advantages
        discounted_returns: list[float] = []

        R = 0.0
        for r in reversed(self.rewards):
            R = r + self.config.gamma * R
            discounted_returns.insert(0, R)

        returns = torch.tensor(discounted_returns, dtype=torch.float32).to(self.device)
        advantages = returns - values

        # Normalize advantages
        advantages = whiten(advantages)

        # Get current policy outputs
        logits, current_values = self.actor_critic(states)
        dist = Categorical(logits=logits)
        current_log_probs = dist.log_prob(actions)

        # Compute losses
        policy_loss = -(current_log_probs * advantages).mean()
        value_loss = F.mse_loss(current_values.squeeze(), returns)
        entropy = dist.entropy().mean()

        total_loss = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            - self.config.entropy_coef * entropy
        )

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()

        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
        }


class AdvancedRLOrchestrator:
    """
    Orchestrator for multiple RL algorithms with automatic algorithm selection.

    .. deprecated:: 0.70.0
        Selects between this module's research trainers only. Use the
        ``training/`` trainers directly. Removed in v0.71.0.
    """

    def __init__(self, agent: EnhancedMultiTurnAgent):
        _warn_deprecated(
            "AdvancedRLOrchestrator", "the stateset_agents.training trainers"
        )
        self.agent = agent
        self.algorithms: dict[str, object] = {}
        self.current_algorithm: str | None = None
        self.performance_history: list[dict[str, Any]] = []

    def add_algorithm(self, name: str, trainer: DPOTrainer | A2CTrainer) -> None:
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
            return "a2c"

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
            logger.warning(f"Algorithm {algorithm_name} not available, using A2C")
            trainer = self.algorithms.get("a2c")
            algorithm_name = "a2c"

        logger.info(f"Selected algorithm: {algorithm_name}")

        # Train with selected algorithm
        if algorithm_name == "dpo" and isinstance(trainer, DPOTrainer):
            return await self._train_dpo(trainer, training_data, **kwargs)
        elif algorithm_name == "a2c" and isinstance(trainer, A2CTrainer):
            return await self._train_a2c(trainer, environment, **kwargs)
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


def create_dpo_trainer(agent: EnhancedMultiTurnAgent, **kwargs) -> DPOTrainer:
    """Create DPO trainer"""
    config = DPOConfig(**kwargs)
    return DPOTrainer(agent, config)


def create_a2c_trainer(agent: EnhancedMultiTurnAgent, **kwargs) -> A2CTrainer:
    """Create A2C trainer"""
    config = A2CConfig(**kwargs)
    return A2CTrainer(agent, config)


def create_advanced_rl_orchestrator(
    agent: EnhancedMultiTurnAgent,
) -> AdvancedRLOrchestrator:
    """Create advanced RL orchestrator with all algorithms"""
    orchestrator = AdvancedRLOrchestrator(agent)

    # Add all algorithms
    orchestrator.add_algorithm("dpo", create_dpo_trainer(agent))
    orchestrator.add_algorithm("a2c", create_a2c_trainer(agent))

    return orchestrator


def __getattr__(name: str) -> Any:
    """Serve the removed duplicate ``GSPOConfig`` from its canonical home."""
    if name == "GSPOConfig":
        warn_deprecated(
            f"{__name__}.GSPOConfig",
            removal="0.71.0",
            replacement="stateset_agents.training.gspo_config.GSPOConfig",
        )
        from ...training.gspo_config import GSPOConfig

        return GSPOConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
