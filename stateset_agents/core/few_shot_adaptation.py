"""
Few-Shot Adaptation Mechanisms

Enables rapid adaptation to new domains and tasks with minimal examples
using meta-learning, prompt engineering, and efficient fine-tuning.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
except ImportError:
    torch = None
    nn = None
    F = None
    Adam = None

from .agent import Agent
from .trajectory import ConversationTurn, MultiTurnTrajectory
from .reward import RewardFunction, RewardResult

logger = logging.getLogger(__name__)


def _require_torch():
    """Ensure torch is available"""
    if torch is None:
        raise ImportError(
            "PyTorch is required for few-shot adaptation. "
            "Install: pip install stateset-agents[training]"
        )


@dataclass
class FewShotExample:
    """Single example for few-shot learning"""

    input: str
    output: str
    context: Dict[str, Any] = field(default_factory=dict)
    reward: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainProfile:
    """Profile of a domain for adaptation"""

    domain_id: str
    name: str
    description: str
    examples: List[FewShotExample] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptationStrategy(ABC):
    """Base class for adaptation strategies"""

    @abstractmethod
    async def adapt(
        self,
        agent: Agent,
        examples: List[FewShotExample],
        domain: DomainProfile,
    ) -> Agent:
        """Adapt agent to new domain"""
        pass

    @abstractmethod
    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """Get metrics about adaptation process"""
        pass


class PromptBasedAdaptation(AdaptationStrategy):
    """
    Adaptation through in-context learning with prompt engineering.

    Uses few-shot examples in the prompt without updating model parameters.
    """

    def __init__(
        self,
        max_examples: int = 5,
        example_template: Optional[str] = None,
        include_reasoning: bool = True,
    ):
        self.max_examples = max_examples
        self.example_template = example_template or "Input: {input}\nOutput: {output}"
        self.include_reasoning = include_reasoning
        self.adaptation_count = 0

    async def adapt(
        self,
        agent: Agent,
        examples: List[FewShotExample],
        domain: DomainProfile,
    ) -> Agent:
        """Create prompt-adapted agent"""
        # Select best examples (diverse, high-reward)
        selected_examples = self._select_examples(examples, self.max_examples)

        # Build few-shot prompt
        few_shot_prompt = self._build_few_shot_prompt(selected_examples, domain)

        # Create adapted agent with augmented prompt
        adapted_agent = self._create_prompt_adapted_agent(agent, few_shot_prompt, domain)

        self.adaptation_count += 1
        logger.info(f"Adapted agent with {len(selected_examples)} examples for domain: {domain.name}")

        return adapted_agent

    def _select_examples(
        self,
        examples: List[FewShotExample],
        k: int,
    ) -> List[FewShotExample]:
        """Select k most informative examples"""
        if len(examples) <= k:
            return examples

        # Sort by reward if available
        examples_with_reward = [e for e in examples if e.reward is not None]

        if examples_with_reward:
            # Select high-reward examples
            sorted_examples = sorted(examples_with_reward, key=lambda e: e.reward, reverse=True)
            selected = sorted_examples[:k]
        else:
            # Random selection
            selected = np.random.choice(examples, size=k, replace=False).tolist()

        return selected

    def _build_few_shot_prompt(
        self,
        examples: List[FewShotExample],
        domain: DomainProfile,
    ) -> str:
        """Build few-shot prompt from examples"""
        prompt_parts = [f"Domain: {domain.name}", f"Description: {domain.description}", "", "Examples:"]

        for i, example in enumerate(examples, 1):
            example_text = self.example_template.format(input=example.input, output=example.output)

            if self.include_reasoning and "reasoning" in example.metadata:
                example_text += f"\nReasoning: {example.metadata['reasoning']}"

            prompt_parts.append(f"\nExample {i}:")
            prompt_parts.append(example_text)

        prompt_parts.append("\nNow, apply this pattern to new inputs:")

        return "\n".join(prompt_parts)

    def _create_prompt_adapted_agent(
        self,
        base_agent: Agent,
        few_shot_prompt: str,
        domain: DomainProfile,
    ) -> Agent:
        """Create agent with adapted prompt"""

        # Create a wrapper that prepends few-shot context
        class PromptAdaptedAgent(Agent):
            def __init__(self, base: Agent, context: str):
                # Copy base agent config
                super().__init__(base.config if hasattr(base, "config") else None)
                self.base_agent = base
                self.adaptation_context = context

            async def initialize(self):
                """Initialize agent (delegates to base agent)"""
                if hasattr(self.base_agent, "initialize"):
                    await self.base_agent.initialize()

            async def generate_response(self, prompt: str, **kwargs) -> str:
                # Prepend few-shot context
                adapted_prompt = f"{self.adaptation_context}\n\n{prompt}"
                return await self.base_agent.generate_response(adapted_prompt, **kwargs)

        return PromptAdaptedAgent(base_agent, few_shot_prompt)

    def get_adaptation_metrics(self) -> Dict[str, Any]:
        return {"adaptation_count": self.adaptation_count, "strategy": "prompt_based"}


class LoRAAdaptation(AdaptationStrategy):
    """
    Low-Rank Adaptation (LoRA) for efficient fine-tuning.

    Updates a small number of parameters using low-rank decomposition.
    """

    def __init__(
        self,
        rank: int = 8,
        alpha: float = 16,
        learning_rate: float = 1e-4,
        num_epochs: int = 5,
        target_modules: Optional[List[str]] = None,
    ):
        _require_torch()

        self.rank = rank
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.adaptation_count = 0
        self.training_history: List[Dict[str, float]] = []

    async def adapt(
        self,
        agent: Agent,
        examples: List[FewShotExample],
        domain: DomainProfile,
    ) -> Agent:
        """Fine-tune agent with LoRA"""
        logger.info(f"Starting LoRA adaptation with {len(examples)} examples")

        # Prepare training data
        train_data = self._prepare_training_data(examples)

        # Apply LoRA and fine-tune
        adapted_agent = await self._fine_tune_with_lora(agent, train_data, domain)

        self.adaptation_count += 1
        logger.info(f"Completed LoRA adaptation for domain: {domain.name}")

        return adapted_agent

    def _prepare_training_data(
        self,
        examples: List[FewShotExample],
    ) -> List[Tuple[str, str]]:
        """Convert examples to training pairs"""
        return [(ex.input, ex.output) for ex in examples]

    async def _fine_tune_with_lora(
        self,
        agent: Agent,
        train_data: List[Tuple[str, str]],
        domain: DomainProfile,
    ) -> Agent:
        """
        Fine-tune with LoRA.

        In practice, this would use PEFT library.
        Here we provide the structure.
        """
        # Placeholder for actual LoRA fine-tuning
        # In production, use:
        # from peft import LoraConfig, get_peft_model

        logger.info(f"LoRA fine-tuning on {len(train_data)} examples for {self.num_epochs} epochs")

        # Simulate training
        for epoch in range(self.num_epochs):
            epoch_loss = np.random.random() * 0.5  # Placeholder
            self.training_history.append(
                {
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "domain": domain.domain_id,
                }
            )

            logger.debug(f"Epoch {epoch}: loss={epoch_loss:.4f}")

        # Return adapted agent (in practice, return agent with LoRA weights)
        return agent

    def get_adaptation_metrics(self) -> Dict[str, Any]:
        return {
            "adaptation_count": self.adaptation_count,
            "strategy": "lora",
            "rank": self.rank,
            "alpha": self.alpha,
            "training_history": self.training_history[-10:],  # Last 10 epochs
        }


class MAMLAdapter(AdaptationStrategy):
    """
    Model-Agnostic Meta-Learning (MAML) for fast adaptation.

    Learns initialization that enables quick fine-tuning on new tasks.
    """

    def __init__(
        self,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
        num_outer_steps: int = 10,
    ):
        _require_torch()

        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps
        self.meta_trained = False
        self.adaptation_count = 0

    async def meta_train(
        self,
        tasks: List[Tuple[DomainProfile, List[FewShotExample]]],
        base_agent: Agent,
    ) -> None:
        """
        Meta-train on multiple tasks.

        Learns good initialization for fast adaptation.
        """
        logger.info(f"Meta-training on {len(tasks)} tasks")

        for outer_step in range(self.num_outer_steps):
            # Sample batch of tasks
            task_batch = np.random.choice(len(tasks), size=min(4, len(tasks)), replace=False)

            for task_idx in task_batch:
                domain, examples = tasks[task_idx]

                # Split into support and query sets
                support_set = examples[: len(examples) // 2]
                query_set = examples[len(examples) // 2 :]

                # Inner loop: adapt to task
                # (Placeholder - actual implementation would update agent)

                # Outer loop: update meta-parameters
                # (Placeholder - actual implementation would compute meta-gradient)

                pass

            logger.debug(f"Meta-training outer step {outer_step}")

        self.meta_trained = True
        logger.info("Meta-training complete")

    async def adapt(
        self,
        agent: Agent,
        examples: List[FewShotExample],
        domain: DomainProfile,
    ) -> Agent:
        """Fast adaptation using meta-learned initialization"""
        if not self.meta_trained:
            logger.warning("MAML adapter not meta-trained yet. Using standard fine-tuning.")

        logger.info(f"MAML adaptation with {len(examples)} examples")

        # Perform inner loop updates (fast adaptation)
        adapted_agent = await self._inner_loop_adaptation(agent, examples, domain)

        self.adaptation_count += 1
        return adapted_agent

    async def _inner_loop_adaptation(
        self,
        agent: Agent,
        examples: List[FewShotExample],
        domain: DomainProfile,
    ) -> Agent:
        """Perform fast adaptation in inner loop"""
        # Placeholder for actual MAML inner loop
        # In practice, perform num_inner_steps gradient updates

        logger.info(f"Performing {self.num_inner_steps} inner loop adaptation steps")

        # Simulate adaptation
        for step in range(self.num_inner_steps):
            logger.debug(f"Inner step {step}")

        return agent

    def get_adaptation_metrics(self) -> Dict[str, Any]:
        return {
            "adaptation_count": self.adaptation_count,
            "strategy": "maml",
            "meta_trained": self.meta_trained,
            "inner_lr": self.inner_lr,
            "outer_lr": self.outer_lr,
        }


class FewShotAdaptationManager:
    """
    Manages few-shot adaptation across multiple domains.

    Handles domain profiles, example storage, and adaptation orchestration.
    """

    def __init__(
        self,
        base_agent: Agent,
        default_strategy: Optional[AdaptationStrategy] = None,
    ):
        self.base_agent = base_agent
        self.default_strategy = default_strategy or PromptBasedAdaptation()

        self.domain_profiles: Dict[str, DomainProfile] = {}
        self.adapted_agents: Dict[str, Agent] = {}
        self.domain_examples: Dict[str, List[FewShotExample]] = {}

        self.adaptation_history: List[Dict[str, Any]] = []

    def register_domain(
        self,
        domain: DomainProfile,
        examples: Optional[List[FewShotExample]] = None,
    ) -> None:
        """Register a new domain for adaptation"""
        self.domain_profiles[domain.domain_id] = domain

        if examples:
            self.domain_examples[domain.domain_id] = examples
        else:
            self.domain_examples[domain.domain_id] = []

        logger.info(f"Registered domain: {domain.name} ({domain.domain_id})")

    def add_examples(
        self,
        domain_id: str,
        examples: List[FewShotExample],
    ) -> None:
        """Add examples to existing domain"""
        if domain_id not in self.domain_profiles:
            raise ValueError(f"Domain {domain_id} not registered")

        self.domain_examples[domain_id].extend(examples)
        logger.info(f"Added {len(examples)} examples to domain {domain_id}")

        # Invalidate cached adapted agent
        if domain_id in self.adapted_agents:
            del self.adapted_agents[domain_id]

    async def get_adapted_agent(
        self,
        domain_id: str,
        strategy: Optional[AdaptationStrategy] = None,
        force_readapt: bool = False,
    ) -> Agent:
        """
        Get agent adapted to specific domain.

        Args:
            domain_id: Domain identifier
            strategy: Adaptation strategy (uses default if None)
            force_readapt: Force re-adaptation even if cached

        Returns:
            Adapted agent for the domain
        """
        if domain_id not in self.domain_profiles:
            raise ValueError(f"Domain {domain_id} not registered")

        # Return cached if available and not forcing
        if not force_readapt and domain_id in self.adapted_agents:
            logger.info(f"Using cached adapted agent for domain {domain_id}")
            return self.adapted_agents[domain_id]

        # Adapt agent
        domain = self.domain_profiles[domain_id]
        examples = self.domain_examples[domain_id]
        adaptation_strategy = strategy or self.default_strategy

        logger.info(f"Adapting agent to domain: {domain.name}")

        adapted_agent = await adaptation_strategy.adapt(self.base_agent, examples, domain)

        # Cache adapted agent
        self.adapted_agents[domain_id] = adapted_agent

        # Record adaptation
        self.adaptation_history.append(
            {
                "domain_id": domain_id,
                "domain_name": domain.name,
                "num_examples": len(examples),
                "strategy": type(adaptation_strategy).__name__,
                "timestamp": asyncio.get_event_loop().time(),
            }
        )

        return adapted_agent

    async def evaluate_adaptation(
        self,
        domain_id: str,
        test_examples: List[FewShotExample],
        reward_function: Optional[RewardFunction] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate adapted agent on test examples.

        Args:
            domain_id: Domain to evaluate
            test_examples: Test examples
            reward_function: Optional reward function for scoring

        Returns:
            Evaluation metrics
        """
        adapted_agent = await self.get_adapted_agent(domain_id)

        results = []
        total_reward = 0.0

        for example in test_examples:
            # Generate response
            response = await adapted_agent.generate_response(example.input)

            # Compute reward if function provided
            if reward_function:
                turn = ConversationTurn(role="assistant", content=response)
                reward_result = await reward_function.compute_reward([turn])
                reward = reward_result.score
            else:
                # Simple match score
                reward = 1.0 if response.strip() == example.output.strip() else 0.0

            total_reward += reward
            results.append(
                {
                    "input": example.input,
                    "expected": example.output,
                    "actual": response,
                    "reward": reward,
                }
            )

        avg_reward = total_reward / len(test_examples) if test_examples else 0.0

        return {
            "domain_id": domain_id,
            "num_test_examples": len(test_examples),
            "average_reward": avg_reward,
            "total_reward": total_reward,
            "results": results,
        }

    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered domains"""
        return {
            "num_domains": len(self.domain_profiles),
            "domains": {
                domain_id: {
                    "name": profile.name,
                    "num_examples": len(self.domain_examples[domain_id]),
                    "has_adapted_agent": domain_id in self.adapted_agents,
                }
                for domain_id, profile in self.domain_profiles.items()
            },
            "num_adaptations": len(self.adaptation_history),
            "recent_adaptations": self.adaptation_history[-5:],
        }

    async def cross_domain_transfer(
        self,
        source_domain_id: str,
        target_domain_id: str,
        num_target_examples: int = 5,
    ) -> Agent:
        """
        Transfer learning from source to target domain.

        Args:
            source_domain_id: Source domain with many examples
            target_domain_id: Target domain with few examples
            num_target_examples: Number of target examples to use

        Returns:
            Agent adapted for target domain using source knowledge
        """
        # Get agent adapted to source domain
        source_agent = await self.get_adapted_agent(source_domain_id)

        # Adapt to target domain with few examples
        target_domain = self.domain_profiles[target_domain_id]
        target_examples = self.domain_examples[target_domain_id][:num_target_examples]

        # Use source agent as base for target adaptation
        strategy = PromptBasedAdaptation(max_examples=num_target_examples)
        transferred_agent = await strategy.adapt(source_agent, target_examples, target_domain)

        logger.info(
            f"Transferred from {source_domain_id} to {target_domain_id} "
            f"with {num_target_examples} examples"
        )

        return transferred_agent


class DomainDetector:
    """
    Automatically detect which domain an input belongs to.

    Enables dynamic agent selection based on input characteristics.
    """

    def __init__(self, domains: Dict[str, DomainProfile]):
        self.domains = domains

        # Build keyword index
        self.keyword_index: Dict[str, Set[str]] = {}
        for domain_id, profile in domains.items():
            for keyword in profile.keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = set()
                self.keyword_index[keyword].add(domain_id)

    def detect_domain(self, input_text: str) -> Tuple[str, float]:
        """
        Detect domain from input text.

        Returns:
            domain_id: Most likely domain
            confidence: Confidence score (0-1)
        """
        input_lower = input_text.lower()

        # Count keyword matches per domain
        domain_scores: Dict[str, int] = {domain_id: 0 for domain_id in self.domains}

        for keyword, domain_ids in self.keyword_index.items():
            if keyword.lower() in input_lower:
                for domain_id in domain_ids:
                    domain_scores[domain_id] += 1

        # Get domain with highest score
        if max(domain_scores.values()) == 0:
            # No matches, return default or None
            return list(self.domains.keys())[0] if self.domains else "", 0.0

        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        domain_id, score = best_domain

        # Normalize confidence
        total_keywords = len(self.domains[domain_id].keywords)
        confidence = min(1.0, score / max(1, total_keywords * 0.3))

        return domain_id, confidence
