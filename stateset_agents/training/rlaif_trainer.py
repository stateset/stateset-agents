"""
RLAIF (Reinforcement Learning from AI Feedback) Trainer

This module provides a complete RLAIF training pipeline that combines:
- LLM-as-Judge for reward generation
- PPO/GRPO for policy optimization
- Constitutional AI principles
- Self-improvement loops

RLAIF enables training models without human annotation by using
AI evaluators to provide feedback signals.

Key features:
- Automatic reward generation from AI judges
- Support for multiple judge models
- Constitutional AI constraints
- Self-play training modes
- Iterative refinement

Reference: https://arxiv.org/abs/2212.08073 (Constitutional AI)
Reference: https://arxiv.org/abs/2309.00267 (RLAIF)
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Import framework components
from .config import TrainingConfig

try:
    from rewards.llm_judge import (
        JudgeConfig,
        JudgeProvider,
        LLMJudge,
        EvaluationCriteria,
        create_llm_judge_reward,
    )
    LLM_JUDGE_AVAILABLE = True
except ImportError:
    try:
        # Try relative import
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from rewards.llm_judge import (
            JudgeConfig,
            JudgeProvider,
            LLMJudge,
            EvaluationCriteria,
            create_llm_judge_reward,
        )
        LLM_JUDGE_AVAILABLE = True
    except ImportError:
        LLM_JUDGE_AVAILABLE = False
        logger.warning("LLM Judge not available for RLAIF")

# Lazy imports
_transformers_rlaif_loaded = False
AutoModelForCausalLM = None
AutoTokenizer = None


def _load_transformers_rlaif():
    """Lazily load transformers."""
    global _transformers_rlaif_loaded, AutoModelForCausalLM, AutoTokenizer

    if _transformers_rlaif_loaded:
        return True

    try:
        from transformers import (
            AutoModelForCausalLM as _AutoModelForCausalLM,
            AutoTokenizer as _AutoTokenizer,
        )
        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
        _transformers_rlaif_loaded = True
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Failed to load transformers: {e}")
        return False


@dataclass
class RLAIFConfig(TrainingConfig):
    """
    Configuration for RLAIF training.
    """

    # Model
    model_name: str = "gpt2"

    # Judge configuration
    judge_provider: str = "openai"  # "openai", "anthropic", "local"
    judge_model: str = "gpt-4o"
    judge_api_key: Optional[str] = None

    # Evaluation criteria
    criteria: List[str] = field(
        default_factory=lambda: ["helpfulness", "correctness", "harmlessness"]
    )

    # Constitutional AI constraints
    use_constitutional: bool = True
    constitutional_principles: List[str] = field(
        default_factory=lambda: [
            "Be helpful, harmless, and honest",
            "Avoid generating harmful, biased, or misleading content",
            "Acknowledge uncertainty when appropriate",
            "Respect user privacy and safety",
        ]
    )

    # Training algorithm
    rl_algorithm: str = "grpo"  # "ppo", "grpo", "gspo"

    # RL parameters
    beta: float = 0.1  # KL penalty
    clip_eps: float = 0.2
    num_ppo_epochs: int = 4
    num_generations: int = 4

    # Self-play settings
    use_self_play: bool = False
    self_play_interval: int = 100  # Steps between self-play rounds

    # Iterative refinement
    num_refinement_iterations: int = 1
    generate_critiques: bool = True

    # Generation
    max_prompt_length: int = 256
    max_completion_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    # Memory optimization
    use_lora: bool = True
    gradient_checkpointing: bool = True


# Constitutional AI principles for different domains
CONSTITUTIONAL_PRINCIPLES = {
    "general": [
        "Please choose the response that is the most helpful, accurate, and harmless.",
        "Please choose the response that is most respectful of everyone's privacy and autonomy.",
        "Please choose the response that is most objective and truthful.",
    ],
    "assistant": [
        "The response should be helpful and informative while being safe.",
        "The response should acknowledge limitations and uncertainties.",
        "The response should not make up false information.",
    ],
    "coding": [
        "The code should be correct, efficient, and well-documented.",
        "The response should explain the reasoning behind the solution.",
        "The code should follow best practices and be maintainable.",
    ],
    "reasoning": [
        "The reasoning should be step-by-step and logical.",
        "Intermediate steps should be clearly explained.",
        "The final answer should be clearly stated.",
    ],
}


class ConstitutionalAI:
    """
    Constitutional AI (CAI) module for RLAIF.

    Implements the critique-revision loop from Constitutional AI:
    1. Generate initial response
    2. Self-critique based on principles
    3. Revise response based on critique
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        principles: Optional[List[str]] = None,
        domain: str = "general",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.principles = principles or CONSTITUTIONAL_PRINCIPLES.get(
            domain, CONSTITUTIONAL_PRINCIPLES["general"]
        )
        self.device = next(model.parameters()).device

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from model."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

    def critique(self, query: str, response: str) -> str:
        """
        Generate self-critique of response.

        Args:
            query: Original user query
            response: Generated response to critique

        Returns:
            Critique text
        """
        principles_text = "\n".join([f"- {p}" for p in self.principles])

        critique_prompt = f"""Given the following principles:
{principles_text}

Critique the following response:

Query: {query}
Response: {response}

Please identify any ways the response could better align with the principles above.
Be specific and constructive.

Critique:"""

        return self._generate(critique_prompt, max_new_tokens=150, temperature=0.3)

    def revise(self, query: str, response: str, critique: str) -> str:
        """
        Revise response based on critique.

        Args:
            query: Original query
            response: Original response
            critique: Self-critique

        Returns:
            Revised response
        """
        revision_prompt = f"""Original query: {query}

Original response: {response}

Critique: {critique}

Please revise the response to address the critique while maintaining helpfulness.

Revised response:"""

        return self._generate(revision_prompt, max_new_tokens=256, temperature=0.5)

    def refine(
        self,
        query: str,
        response: str,
        num_iterations: int = 1,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Apply iterative critique-revision loop.

        Args:
            query: User query
            response: Initial response
            num_iterations: Number of refinement iterations

        Returns:
            Tuple of (final_response, history)
        """
        history = [{"iteration": 0, "response": response}]

        current_response = response
        for i in range(num_iterations):
            critique = self.critique(query, current_response)
            revised = self.revise(query, current_response, critique)

            history.append({
                "iteration": i + 1,
                "critique": critique,
                "response": revised,
            })

            current_response = revised

        return current_response, history


class RLAIFTrainer:
    """
    Reinforcement Learning from AI Feedback (RLAIF) Trainer.

    This trainer combines:
    - LLM-as-Judge for reward generation
    - Policy optimization (PPO/GRPO)
    - Optional Constitutional AI refinement

    Example:
        ```python
        config = RLAIFConfig(
            model_name="mistralai/Mistral-7B-v0.1",
            judge_model="gpt-4o",
            criteria=["helpfulness", "correctness"],
        )

        trainer = RLAIFTrainer(config)
        results = await trainer.train(prompts)
        ```
    """

    def __init__(
        self,
        config: RLAIFConfig,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not config.use_cpu else "cpu"
        )

        # Load model and tokenizer if not provided
        if model is None or tokenizer is None:
            _load_transformers_rlaif()
            if AutoModelForCausalLM is None:
                raise ImportError("transformers required for RLAIF")

            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
            )
        else:
            self.model = model
            self.tokenizer = tokenizer

        # Initialize LLM Judge
        self.judge = None
        if LLM_JUDGE_AVAILABLE:
            self.judge = LLMJudge(
                provider=JudgeProvider(config.judge_provider),
                model_name=config.judge_model,
                api_key=config.judge_api_key,
                criteria=[EvaluationCriteria(c) for c in config.criteria],
            )

        # Initialize Constitutional AI
        self.cai = None
        if config.use_constitutional:
            self.cai = ConstitutionalAI(
                model=self.model,
                tokenizer=self.tokenizer,
                principles=config.constitutional_principles,
            )

        # Initialize policy optimizer (will be created during training)
        self.policy_trainer = None

        # Metrics
        self.metrics_history = {
            "reward": [],
            "policy_loss": [],
            "value_loss": [],
            "kl_divergence": [],
            "refinement_improvement": [],
        }

        self.global_step = 0

    async def compute_reward(self, prompt: str, completion: str) -> float:
        """
        Compute reward using LLM judge.

        Args:
            prompt: User prompt
            completion: Model completion

        Returns:
            Reward score in [0, 1]
        """
        if self.judge is None:
            # Fallback to simple length-based reward
            return min(len(completion) / 200, 1.0)

        return await self.judge.evaluate(prompt, completion)

    def generate_completion(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion from model."""
        max_new_tokens = max_new_tokens or self.config.max_completion_length

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

    async def collect_trajectories(
        self,
        prompts: List[str],
        num_generations: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Collect trajectories with AI-generated rewards.

        Args:
            prompts: List of prompts
            num_generations: Generations per prompt

        Returns:
            List of trajectory dictionaries
        """
        trajectories = []

        for prompt in prompts:
            prompt_trajectories = []

            for _ in range(num_generations):
                # Generate completion
                completion = self.generate_completion(prompt)

                # Optional Constitutional AI refinement
                if self.cai and self.config.num_refinement_iterations > 0:
                    refined_completion, history = self.cai.refine(
                        prompt,
                        completion,
                        num_iterations=self.config.num_refinement_iterations,
                    )

                    # Use refined completion and track improvement
                    original_reward = await self.compute_reward(prompt, completion)
                    refined_reward = await self.compute_reward(prompt, refined_completion)

                    trajectory = {
                        "prompt": prompt,
                        "completion": refined_completion,
                        "original_completion": completion,
                        "reward": refined_reward,
                        "original_reward": original_reward,
                        "refinement_history": history,
                    }

                    self.metrics_history["refinement_improvement"].append(
                        refined_reward - original_reward
                    )
                else:
                    # Standard reward computation
                    reward = await self.compute_reward(prompt, completion)

                    trajectory = {
                        "prompt": prompt,
                        "completion": completion,
                        "reward": reward,
                    }

                prompt_trajectories.append(trajectory)

            trajectories.extend(prompt_trajectories)

        return trajectories

    async def train_step(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Execute one training step.

        Args:
            trajectories: Collected trajectories with rewards

        Returns:
            Training metrics
        """
        # Extract data
        prompts = [t["prompt"] for t in trajectories]
        completions = [t["completion"] for t in trajectories]
        rewards = [t["reward"] for t in trajectories]

        # Tokenize
        full_texts = [p + c for p, c in zip(prompts, completions)]
        encodings = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length + self.config.max_completion_length,
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        # Compute log probs
        self.model.train()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]

        # Shift labels
        labels = input_ids[:, 1:]

        # Log probs for chosen tokens
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)

        # Create completion mask
        prompt_encodings = self.tokenizer(prompts, padding=True, return_tensors="pt")
        prompt_lengths = prompt_encodings["attention_mask"].sum(dim=1)

        completion_mask = torch.zeros_like(labels, dtype=torch.float)
        for i, prompt_len in enumerate(prompt_lengths):
            completion_mask[i, prompt_len:] = 1.0

        completion_mask = completion_mask.to(self.device)

        # Compute advantages (simplified - reward broadcast to tokens)
        rewards_tensor = torch.tensor(rewards, device=self.device).unsqueeze(1)
        advantages = completion_mask * rewards_tensor

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy gradient loss
        policy_loss = -(token_log_probs * advantages * completion_mask).sum() / completion_mask.sum()

        # Backward pass
        self.model.zero_grad()
        policy_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )

        # Optimizer step (simple SGD for now)
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    param.add_(param.grad, alpha=-self.config.learning_rate)

        self.global_step += 1

        metrics = {
            "policy_loss": policy_loss.item(),
            "average_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
        }

        # Update history
        self.metrics_history["policy_loss"].append(metrics["policy_loss"])
        self.metrics_history["reward"].append(metrics["average_reward"])

        return metrics

    async def train(
        self,
        prompts: List[str],
        num_episodes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run full RLAIF training loop.

        Args:
            prompts: Training prompts
            num_episodes: Number of training episodes

        Returns:
            Training summary
        """
        num_episodes = num_episodes or self.config.num_episodes

        logger.info(f"Starting RLAIF training for {num_episodes} episodes")
        logger.info(f"Using judge: {self.config.judge_model}")
        logger.info(f"Criteria: {self.config.criteria}")

        for episode in range(num_episodes):
            # Sample prompts for this episode
            batch_size = min(self.config.per_device_train_batch_size, len(prompts))
            episode_prompts = prompts[:batch_size]

            # Collect trajectories with AI rewards
            trajectories = await self.collect_trajectories(
                episode_prompts,
                num_generations=self.config.num_generations,
            )

            # Training step
            metrics = await self.train_step(trajectories)

            # Logging
            if episode % self.config.logging_steps == 0:
                logger.info(
                    f"Episode {episode}/{num_episodes} | "
                    f"Loss: {metrics['policy_loss']:.4f} | "
                    f"Reward: {metrics['average_reward']:.4f}"
                )

            # Self-play (if enabled)
            if self.config.use_self_play and episode % self.config.self_play_interval == 0:
                await self._self_play_round()

        return {
            "final_metrics": metrics,
            "metrics_history": self.metrics_history,
            "total_episodes": num_episodes,
        }

    async def _self_play_round(self) -> None:
        """Execute a self-play round for additional training signal."""
        logger.info("Running self-play round...")

        # Generate prompts from model
        meta_prompt = "Generate an interesting question or task that would test an AI assistant's capabilities:"
        self_generated_prompts = []

        for _ in range(5):
            prompt = self.generate_completion(meta_prompt, max_new_tokens=100)
            self_generated_prompts.append(prompt)

        # Collect trajectories on self-generated prompts
        trajectories = await self.collect_trajectories(
            self_generated_prompts,
            num_generations=2,
        )

        # Training on self-generated data
        metrics = await self.train_step(trajectories)
        logger.info(f"Self-play metrics: {metrics}")

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save model checkpoint."""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(
                self.config.output_dir,
                f"rlaif-checkpoint-{self.global_step}-{timestamp}",
            )

        os.makedirs(path, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save training state
        torch.save({
            "global_step": self.global_step,
            "metrics_history": self.metrics_history,
            "config": self.config,
        }, os.path.join(path, "training_state.pt"))

        logger.info(f"Checkpoint saved to {path}")
        return path


# Convenience function for quick RLAIF training
async def train_rlaif(
    model_name: str,
    prompts: List[str],
    judge_model: str = "gpt-4o",
    num_episodes: int = 100,
    **kwargs,
) -> Dict[str, Any]:
    """
    Quick RLAIF training with minimal configuration.

    Args:
        model_name: HuggingFace model name
        prompts: Training prompts
        judge_model: Model to use for AI feedback
        num_episodes: Number of training episodes
        **kwargs: Additional config parameters

    Returns:
        Training results
    """
    config = RLAIFConfig(
        model_name=model_name,
        judge_model=judge_model,
        num_episodes=num_episodes,
        **kwargs,
    )

    trainer = RLAIFTrainer(config)
    return await trainer.train(prompts, num_episodes)
