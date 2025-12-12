"""
Advanced RL Algorithms for StateSet Agents

This module implements multiple RL algorithms including:
- Proximal Policy Optimization (PPO)
- Direct Preference Optimization (DPO)
- Advantage Actor-Critic (A2C)
- Trust Region Policy Optimization (TRPO)
- Group Sequence Policy Optimization (GSPO)
- Group Sequence Policy Optimization - Token variant (GSPO-token)
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from ..agent import Agent, AgentConfig, MultiTurnAgent
from ..environment import ConversationEnvironment, Environment
from ..reward import RewardFunction, RewardResult
from ..trajectory import ConversationTurn, MultiTurnTrajectory, TrajectoryGroup
from .enhanced_agent import EnhancedMultiTurnAgent

logger = logging.getLogger(__name__)


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


@dataclass
class GSPOConfig:
    """Configuration for GSPO training"""

    learning_rate: float = 1e-5
    num_generations: int = 4  # Group size (G)
    clip_range_left: float = 3e-4  # Sequence-level clipping
    clip_range_right: float = 4e-4
    beta: float = 0.0  # KL penalty coefficient
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    use_gspo_token: bool = False  # Use token-level variant


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
                dist = Categorical(logits=logits)
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

        self.device = torch.device(
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize networks
        vocab_size = (
            len(self.agent.tokenizer) if hasattr(self.agent, "tokenizer") else 50000
        )
        self.actor_critic = ActorCriticNetwork(
            input_size=768,  # BERT-like embedding size
            hidden_size=256,
            output_size=vocab_size,
        ).to(self.device)

        self.old_actor_critic = ActorCriticNetwork(
            input_size=768, hidden_size=256, output_size=vocab_size
        ).to(self.device)

        # Copy parameters
        self.old_actor_critic.load_state_dict(self.actor_critic.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=config.learning_rate
        )

        # Storage for trajectories
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    async def collect_trajectory(
        self, environment: Environment, max_steps: int = 100
    ) -> MultiTurnTrajectory:
        """Collect a trajectory using current policy"""

        state = await environment.reset()
        trajectory_turns = []

        for step in range(max_steps):
            # Get current conversation state
            messages = [
                {"role": turn.role, "content": turn.content}
                for turn in trajectory_turns[-10:]  # Last 10 turns
            ]

            if not messages:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ]

            # Generate embedding for state
            state_embedding = await self._get_state_embedding(messages)

            # Get action from policy
            action_token, value = self.actor_critic.get_action(
                torch.tensor(state_embedding, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            # Convert token to text
            action_text = self.agent.tokenizer.decode([action_token])

            # Create conversation turn
            agent_turn = ConversationTurn(role="assistant", content=action_text)

            # Step environment
            new_state, env_response, reward, done = await environment.step(
                state, agent_turn
            )

            # Store transition
            self.states.append(state_embedding)
            self.actions.append(action_token)
            self.values.append(value)
            self.rewards.append(reward)

            trajectory_turns.append(agent_turn)
            if env_response:
                trajectory_turns.append(env_response)

            state = new_state

            if done:
                break

        trajectory = MultiTurnTrajectory(
            turns=trajectory_turns,
            total_reward=sum(self.rewards),
            metadata={"algorithm": "PPO"},
        )

        return trajectory

    async def _get_state_embedding(self, messages: List[Dict[str, str]]) -> np.ndarray:
        """Get embedding representation of conversation state"""
        # Simple approach: concatenate last few messages
        conversation_text = " ".join([msg["content"] for msg in messages[-3:]])

        # Use tokenizer to get embeddings (simplified)
        inputs = self.agent.tokenizer(
            conversation_text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.agent.model(**inputs, output_hidden_states=True)
            # Use last hidden state of [CLS] token or average pooling
            embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy().flatten()

        return embedding

    def compute_advantages(
        self, rewards: List[float], values: List[float], dones: List[bool]
    ):
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[i + 1]

            delta = (
                rewards[i] + self.config.gamma * next_value * (1 - dones[i]) - values[i]
            )
            gae = (
                delta
                + self.config.gamma * self.config.gae_lambda * (1 - dones[i]) * gae
            )
            advantages.insert(0, gae)

        return advantages

    async def train_step(self):
        """Execute one PPO training step"""

        # Convert lists to tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(
            self.device
        )
        actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(
            self.device
        )
        values = torch.tensor(self.values, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.dones, dtype=torch.bool).to(self.device)

        # Compute advantages and returns
        advantages = self.compute_advantages(
            rewards.tolist(), values.tolist(), dones.tolist()
        )
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO training loop
        for _ in range(self.config.ppo_epochs):
            # Get current policy outputs
            logits, current_values = self.actor_critic(states)
            dist = Categorical(logits=logits)
            current_log_probs = dist.log_prob(actions)

            # Compute ratios
            ratios = torch.exp(current_log_probs - old_log_probs)

            # Compute surrogate losses
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
            value_loss = F.mse_loss(current_values.squeeze(), returns)

            # Entropy bonus
            entropy = dist.entropy().mean()

            # Total loss
            total_loss = (
                policy_loss
                + self.config.value_loss_coef * value_loss
                - self.config.entropy_coef * entropy
            )

            # Update network
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
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

        self.device = torch.device(
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.agent.model.parameters(), lr=config.learning_rate
        )

        # Reference model (frozen copy)
        self.reference_model = self._create_reference_model()

    def _create_reference_model(self):
        """Create a frozen reference model"""
        import copy

        reference_model = copy.deepcopy(self.agent.model)

        # Freeze parameters
        for param in reference_model.parameters():
            param.requires_grad = False

        return reference_model.to(self.device)

    async def train_step(self, preference_pairs: List[Dict[str, Any]]):
        """
        Train on preference pairs

        Each preference pair contains:
        - prompt: the input prompt
        - chosen: the preferred response
        - rejected: the non-preferred response
        """

        total_loss = 0
        num_pairs = 0

        for pair in preference_pairs:
            prompt = pair["prompt"]
            chosen = pair["chosen"]
            rejected = pair["rejected"]

            # Tokenize inputs
            chosen_tokens = self.agent.tokenizer(
                prompt + " " + chosen,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.device)

            rejected_tokens = self.agent.tokenizer(
                prompt + " " + rejected,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.device)

            # Get log probabilities from current model
            with torch.no_grad():
                chosen_logits = self.agent.model(**chosen_tokens).logits
                rejected_logits = self.agent.model(**rejected_tokens).logits

                chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
                rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)

            # Get log probabilities from reference model
            with torch.no_grad():
                ref_chosen_logits = self.reference_model(**chosen_tokens).logits
                ref_rejected_logits = self.reference_model(**rejected_tokens).logits

                ref_chosen_log_probs = F.log_softmax(ref_chosen_logits, dim=-1)
                ref_rejected_log_probs = F.log_softmax(ref_rejected_logits, dim=-1)

            # Compute DPO loss
            # DPO objective: maximize E[log σ(β log(π/π_ref)(x,y_w) - β log(π/π_ref)(x,y_l))]

            # Simplified implementation (would need proper sequence masking in practice)
            chosen_ratio = chosen_log_probs.mean() - ref_chosen_log_probs.mean()
            rejected_ratio = rejected_log_probs.mean() - ref_rejected_log_probs.mean()

            loss = -F.logsigmoid(self.config.beta * (chosen_ratio - rejected_ratio))

            total_loss += loss.item()
            num_pairs += 1

            # Backward pass
            loss.backward()

        # Update parameters
        torch.nn.utils.clip_grad_norm_(
            self.agent.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {"dpo_loss": total_loss / max(num_pairs, 1)}


class A2CTrainer:
    """
    Advantage Actor-Critic trainer
    """

    def __init__(
        self, agent: EnhancedMultiTurnAgent, config: A2CConfig, device: str = "auto"
    ):
        self.agent = agent
        self.config = config

        self.device = torch.device(
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize actor-critic network
        vocab_size = (
            len(self.agent.tokenizer) if hasattr(self.agent, "tokenizer") else 50000
        )
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
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []

    async def collect_experience(self, environment: Environment, n_steps: int):
        """Collect experience for n steps"""

        state = await environment.reset()
        episode_rewards = []

        for step in range(n_steps):
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
            action_text = self.agent.tokenizer.decode([action_token])

            # Step environment
            agent_turn = ConversationTurn(role="assistant", content=action_text)
            new_state, env_response, reward, done = await environment.step(
                state, agent_turn
            )

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

    async def _get_state_embedding(self, messages: List[Dict[str, str]]) -> np.ndarray:
        """Get embedding for state (simplified)"""
        conversation_text = messages[-1]["content"] if messages else ""

        inputs = self.agent.tokenizer(
            conversation_text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.agent.model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy().flatten()

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
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        values = torch.tensor(self.values, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(
            self.device
        )

        # Compute returns and advantages
        returns = []
        advantages = []

        R = 0
        for r in reversed(self.rewards):
            R = r + self.config.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = returns - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
    Orchestrator for multiple RL algorithms with automatic algorithm selection
    """

    def __init__(self, agent: EnhancedMultiTurnAgent):
        self.agent = agent
        self.algorithms = {}
        self.current_algorithm = None
        self.performance_history = []

    def add_algorithm(
        self, name: str, trainer: Union[PPOTrainer, DPOTrainer, A2CTrainer]
    ):
        """Add an RL algorithm"""
        self.algorithms[name] = trainer

    def select_algorithm(
        self, task_type: str, data_characteristics: Dict[str, Any]
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
        training_data: Dict[str, Any],
        task_type: str = "general",
        **kwargs,
    ):
        """Train using the best algorithm for the task"""

        # Analyze data characteristics
        data_characteristics = self._analyze_data(training_data)

        # Select algorithm
        algorithm_name = self.select_algorithm(task_type, data_characteristics)
        trainer = self.algorithms.get(algorithm_name)

        if not trainer:
            logger.warning(f"Algorithm {algorithm_name} not available, using PPO")
            trainer = self.algorithms.get("ppo")

        logger.info(f"Selected algorithm: {algorithm_name}")

        # Train with selected algorithm
        if algorithm_name == "ppo":
            return await self._train_ppo(trainer, environment, **kwargs)
        elif algorithm_name == "dpo":
            return await self._train_dpo(trainer, training_data, **kwargs)
        elif algorithm_name == "a2c":
            return await self._train_a2c(trainer, environment, **kwargs)

    def _analyze_data(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
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
            trajectory = await trainer.collect_trajectory(environment)

            # Train step
            metrics = await trainer.train_step()

            logger.info(f"PPO Episode {episode}: Loss = {metrics['total_loss']:.4f}")

    async def _train_dpo(
        self, trainer: DPOTrainer, training_data: Dict[str, Any], num_steps: int = 100
    ):
        """Train with DPO"""
        preference_pairs = training_data.get("preference_pairs", [])

        for step in range(num_steps):
            # Sample batch of preference pairs
            batch = np.random.choice(
                preference_pairs,
                size=min(len(preference_pairs), trainer.config.batch_size),
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
            metrics = await trainer.train_step()

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

    For full GSPO training, use training.gspo_trainer.GSPOTrainer
    This stub provides a consistent interface for the RL orchestrator.
    """

    def __init__(
        self, agent: EnhancedMultiTurnAgent, config: GSPOConfig, device: str = "auto"
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
            "For full GSPO training, use: from training.gspo_trainer import train_with_gspo"
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
        from training.gspo_trainer import train_with_gspo
        await train_with_gspo(config, agent, environment, reward_model)
    """
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
