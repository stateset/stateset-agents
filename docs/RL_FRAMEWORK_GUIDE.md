# StateSet Agents RL Framework Guide

A comprehensive guide to using the StateSet Agents reinforcement learning framework for training multi-turn conversational AI agents.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Creating Agents](#creating-agents)
5. [Setting Up Environments](#setting-up-environments)
6. [Reward Functions](#reward-functions)
7. [Training](#training)
8. [GRPO vs GSPO](#grpo-vs-gspo)
9. [Hyperparameter Optimization](#hyperparameter-optimization)
10. [Advanced Features](#advanced-features)
11. [CLI Reference](#cli-reference)
12. [Troubleshooting](#troubleshooting)

---

## Overview

StateSet Agents is a production-ready RL framework designed specifically for training multi-turn conversational AI agents. It implements state-of-the-art algorithms including **Group Relative Policy Optimization (GRPO)** and **Group Sequence Policy Optimization (GSPO)**.

### Key Features

- **Multi-turn conversation support** - Train agents for extended dialogues
- **Multiple RL algorithms** - GRPO, GSPO, PPO, DPO, A2C, TRPO
- **10+ pre-built reward functions** - Helpfulness, safety, correctness, and more
- **Async-first design** - Full async/await support for scalability
- **Production-ready** - Circuit breakers, health monitoring, type safety
- **Distributed training** - Multi-GPU support via Accelerate
- **Stub mode** - Fast prototyping without loading large models

### Architecture Overview

```
┌─────────────┐     ┌─────────────────┐     ┌────────────────┐
│    Agent    │────▶│   Environment   │────▶│ Reward Function│
│             │◀────│                 │◀────│                │
└─────────────┘     └─────────────────┘     └────────────────┘
       │                    │                       │
       │                    │                       │
       ▼                    ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                         Trainer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Value Head   │  │ Loss Compute │  │ Policy Gradient  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
pip install stateset-agents
```

### Minimal Example

```python
import asyncio
from stateset_agents import (
    MultiTurnAgent,
    ConversationEnvironment,
    HelpfulnessReward,
    SafetyReward,
    train
)
from stateset_agents.core.agent import AgentConfig
from stateset_agents.core.reward import CompositeReward

async def main():
    # 1. Create an agent
    config = AgentConfig(
        model_name="gpt2",
        system_prompt="You are a helpful AI assistant.",
        temperature=0.8,
        max_new_tokens=256,
    )
    agent = MultiTurnAgent(config)
    await agent.initialize()

    # 2. Set up environment with conversation scenarios
    scenarios = [
        {
            "id": "greeting",
            "topic": "general",
            "user_responses": [
                "Hello! Can you help me?",
                "What can you do?",
                "Thanks!",
            ],
        }
    ]
    environment = ConversationEnvironment(scenarios=scenarios, max_turns=6)

    # 3. Define reward function
    reward_fn = CompositeReward([
        HelpfulnessReward(weight=0.7),
        SafetyReward(weight=0.3)
    ])

    # 4. Train
    trained_agent = await train(
        agent=agent,
        environment=environment,
        reward_fn=reward_fn,
        num_episodes=100,
        profile="balanced"
    )

    # 5. Use the trained agent
    response = await trained_agent.generate_response([
        {"role": "user", "content": "Hello! How can you help me today?"}
    ])
    print(f"Agent: {response}")

asyncio.run(main())
```

---

## Core Concepts

### Trajectories

A **trajectory** represents a complete conversation episode:

```python
from stateset_agents.core.trajectory import ConversationTurn, MultiTurnTrajectory

# A single message in the conversation
turn = ConversationTurn(
    role="assistant",        # "user", "assistant", or "system"
    content="Hello! How can I help you today?",
    reward=0.8,              # Turn-level reward (optional)
    metadata={}
)

# A complete conversation
trajectory = MultiTurnTrajectory(
    trajectory_id="traj_001",
    turns=[turn1, turn2, turn3, ...],
    episode_reward=0.85,      # Total episode reward
    trajectory_group_id="group_001",  # For GRPO grouping
)
```

### Trajectory Groups

GRPO uses **trajectory groups** - multiple trajectories generated from the same prompt. This enables computing group-relative advantages:

```
Prompt: "Hello, can you help me?"
├── Trajectory 1: reward = 0.7
├── Trajectory 2: reward = 0.9  ← Above average, positive advantage
├── Trajectory 3: reward = 0.6
└── Trajectory 4: reward = 0.8
    Group mean = 0.75
```

### Value Function

The **value function** estimates expected future rewards and computes advantages using Generalized Advantage Estimation (GAE):

```python
from stateset_agents.core.value_function import ValueFunction

value_fn = ValueFunction(
    hidden_size=768,
    gamma=0.99,        # Discount factor
    gae_lambda=0.95,   # GAE lambda
)
```

---

## Creating Agents

### Basic Agent Configuration

```python
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

config = AgentConfig(
    # Model settings
    model_name="gpt2",                    # HuggingFace model ID
    system_prompt="You are helpful.",     # System prompt

    # Generation parameters
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    top_k=50,

    # Hardware optimization
    torch_dtype="bfloat16",               # or "float16", "float32"
    attn_implementation="flash_attention_2",
    device_map="auto",

    # Advanced features
    enable_reasoning=False,               # DeepSeek-R1 style reasoning
)

agent = MultiTurnAgent(config)
await agent.initialize()
```

### Stub Mode (Testing/CI)

For fast prototyping or CI pipelines without loading large models:

```python
config = AgentConfig(
    model_name="stub://demo",
    use_stub_model=True,
    stub_responses=[
        "Hello! How can I assist you?",
        "I'd be happy to help with that.",
        "Is there anything else you need?",
    ]
)

agent = MultiTurnAgent(config)
await agent.initialize()  # Instant, no model loading
```

### Custom Agent Subclass

```python
from stateset_agents.core.agent import MultiTurnAgent

class CustomerServiceAgent(MultiTurnAgent):
    async def generate_response(self, conversation_history):
        # Add custom preprocessing
        enhanced_history = self._add_context(conversation_history)

        # Call parent method
        response = await super().generate_response(enhanced_history)

        # Add custom postprocessing
        return self._ensure_professional_tone(response)

    def _add_context(self, history):
        # Add customer service context
        return history

    def _ensure_professional_tone(self, response):
        # Validate response tone
        return response
```

---

## Setting Up Environments

### ConversationEnvironment

The primary environment for multi-turn dialogue training:

```python
from stateset_agents.core.environment import ConversationEnvironment

# Define conversation scenarios
scenarios = [
    {
        "id": "product_inquiry",
        "topic": "customer_service",
        "user_responses": [
            "Hi, I have a question about my order.",
            "Order #12345, it hasn't arrived yet.",
            "It was supposed to arrive yesterday.",
            "Can you check the status?",
            "Thanks for your help!",
        ],
        "metadata": {
            "difficulty": "easy",
            "category": "order_status"
        }
    },
    {
        "id": "technical_issue",
        "topic": "technical_support",
        "user_responses": [
            "My app keeps crashing.",
            "It happens when I try to log in.",
            "I'm using version 2.3.1.",
            "I've already tried reinstalling.",
        ],
        "metadata": {
            "difficulty": "medium",
            "category": "bug_report"
        }
    }
]

environment = ConversationEnvironment(
    scenarios=scenarios,
    max_turns=10,              # Maximum conversation length
    randomize_scenarios=True,  # Shuffle scenario order
)
```

### Predefined Configurations

Use built-in configurations for common domains:

```python
from stateset_agents.core.environment import CONVERSATION_CONFIGS

# Available configs: "customer_service", "technical_support", "sales"
env = ConversationEnvironment(**CONVERSATION_CONFIGS["customer_service"])
```

### Custom Environment

```python
from stateset_agents.core.environment import Environment, EnvironmentState

class TaskCompletionEnvironment(Environment):
    def __init__(self, tasks):
        self.tasks = tasks

    async def reset(self, scenario=None):
        task = scenario or random.choice(self.tasks)
        return EnvironmentState(
            conversation_history=[],
            metadata={"task": task, "completed": False}
        )

    async def step(self, state, action):
        # Process agent action
        new_history = state.conversation_history + [action]

        # Check task completion
        completed = self._check_completion(action, state.metadata["task"])

        # Compute reward
        reward = 1.0 if completed else 0.0
        done = completed or len(new_history) >= 10

        # Generate user response
        user_response = self._generate_user_response(state, action)

        new_state = EnvironmentState(
            conversation_history=new_history + [user_response],
            metadata={**state.metadata, "completed": completed}
        )

        return new_state, user_response, reward, done
```

---

## Reward Functions

### Pre-built Rewards

StateSet Agents provides 10+ pre-built reward functions:

```python
from stateset_agents.core.reward import (
    HelpfulnessReward,      # Evaluates answer helpfulness
    SafetyReward,           # Flags unsafe content
    CorrectnessReward,      # Factual accuracy
    ConcisenessReward,      # Appropriate brevity
    EngagementReward,       # Conversation quality
    TaskCompletionReward,   # Goal achievement
)

from stateset_agents.rewards import (
    CustomerServiceReward,   # Domain-specific
    TechnicalSupportReward,  # Domain-specific
    SalesAssistantReward,    # Domain-specific
)
```

### Composite Rewards

Combine multiple rewards with weights:

```python
from stateset_agents.core.reward import CompositeReward

reward_fn = CompositeReward([
    HelpfulnessReward(weight=0.4),
    SafetyReward(weight=0.3),
    ConcisenessReward(weight=0.2),
    EngagementReward(weight=0.1),
])
```

### Custom Rewards with Decorator

```python
from stateset_agents.core.reward import reward_function

@reward_function(weight=0.5)
async def politeness_reward(turns, context=None):
    """Reward polite language."""
    polite_phrases = ["please", "thank you", "you're welcome", "happy to help"]

    assistant_turns = [t for t in turns if t.role == "assistant"]
    if not assistant_turns:
        return 0.0

    total_score = 0.0
    for turn in assistant_turns:
        content_lower = turn.content.lower()
        matches = sum(1 for phrase in polite_phrases if phrase in content_lower)
        total_score += min(matches * 0.25, 1.0)

    return total_score / len(assistant_turns)

# Use in composite
reward_fn = CompositeReward([
    HelpfulnessReward(weight=0.5),
    politeness_reward,  # Already has weight from decorator
])
```

### Custom Reward Class

```python
from stateset_agents.core.reward import RewardFunction, RewardResult

class ResponseQualityReward(RewardFunction):
    def __init__(self, min_length=50, max_length=500, weight=1.0):
        super().__init__(weight=weight)
        self.min_length = min_length
        self.max_length = max_length

    async def compute_reward(self, turns, context=None):
        assistant_turns = [t for t in turns if t.role == "assistant"]
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        scores = []
        for turn in assistant_turns:
            length = len(turn.content)

            # Penalize too short or too long
            if length < self.min_length:
                score = length / self.min_length
            elif length > self.max_length:
                score = max(0.5, 1.0 - (length - self.max_length) / 500)
            else:
                score = 1.0

            scores.append(score)

        avg_score = sum(scores) / len(scores)

        return RewardResult(
            score=avg_score,
            breakdown={
                "length_compliance": avg_score,
                "num_turns": len(assistant_turns),
            },
            metadata={"individual_scores": scores}
        )
```

### Multi-Objective Rewards

```python
from stateset_agents.rewards.multi_objective_reward import MultiObjectiveReward

reward_fn = MultiObjectiveReward(
    objectives=[
        {"name": "helpfulness", "reward": HelpfulnessReward(), "weight": 0.4},
        {"name": "safety", "reward": SafetyReward(), "weight": 0.3},
        {"name": "engagement", "reward": EngagementReward(), "weight": 0.3},
    ],
    aggregation="weighted_sum",  # or "min", "product"
    normalize=True,
)
```

---

## Training

### Basic Training

```python
from stateset_agents import train

trained_agent = await train(
    agent=agent,
    environment=environment,
    reward_fn=reward_fn,
    num_episodes=1000,
    profile="balanced",  # "conservative", "balanced", "aggressive"
)
```

### Training Configuration

```python
from stateset_agents.training.config import TrainingConfig

config = TrainingConfig(
    # Episodes and epochs
    num_episodes=1000,
    num_epochs=1,

    # Optimization
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    max_grad_norm=1.0,
    warmup_ratio=0.1,

    # Batching
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=16,          # Trajectories per scenario (group size)

    # Sequence lengths
    max_prompt_length=512,
    max_completion_length=512,

    # Hardware
    bf16=True,
    fp16=False,

    # Monitoring
    eval_steps=50,
    save_steps=100,
    logging_steps=10,

    # Weights & Biases
    report_to="wandb",
    wandb_project="my-agent-training",
)
```

### Task-Specific Configurations

```python
from stateset_agents.training.config import get_config_for_task

# Get optimized config for specific domains
config = get_config_for_task("customer_service", model_name="gpt2")
config = get_config_for_task("technical_support", model_name="llama-7b")
config = get_config_for_task("sales", model_name="mistral-7b")
```

### Training Profiles

| Profile | Description | Use Case |
|---------|-------------|----------|
| `conservative` | Small learning rate, tight clipping | Stable training, avoiding divergence |
| `balanced` | Default settings | General purpose training |
| `aggressive` | Higher learning rate, wider exploration | Fast convergence, risk of instability |
| `experimental` | Cutting-edge settings | Research and experimentation |

### MultiTurnGRPOTrainer

For fine-grained control:

```python
from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

trainer = MultiTurnGRPOTrainer(
    agent=agent,
    environment=environment,
    reward_function=reward_fn,
    config=config,
    value_function=value_fn,  # Optional custom value function
)

# Training loop with callbacks
trained_agent = await trainer.train(
    callbacks=[
        EarlyStoppingCallback(patience=10),
        CheckpointCallback(save_dir="./checkpoints"),
    ]
)
```

---

## GRPO vs GSPO

### GRPO (Group Relative Policy Optimization)

The default algorithm, optimized for stability:

- Uses **group-relative advantages** (compare to group mean, not global baseline)
- Per-token loss computation
- Works well for most use cases

```python
# GRPO is the default
trained_agent = await train(
    agent=agent,
    environment=environment,
    reward_fn=reward_fn,
)
```

### GSPO (Group Sequence Policy Optimization)

More stable variant with sequence-level optimization:

- **Sequence-level loss** instead of token-level
- **Tighter clipping** (3e-4 to 4e-4 vs typical 0.1-0.2)
- Better for long conversations and complex tasks

```python
from stateset_agents.training.gspo_trainer import GSPOConfig, train_with_gspo

gspo_config = GSPOConfig(
    # GSPO-specific parameters
    num_generations=4,         # Group size (smaller than GRPO)
    beta=0.0,                  # KL penalty coefficient
    clip_range_left=3e-4,      # Much tighter clipping
    clip_range_right=4e-4,
    num_outer_iterations=100,

    # Standard training params
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    # LoRA (recommended for GSPO)
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
)

trained_agent = await train_with_gspo(
    config=gspo_config,
    agent=agent,
    environment=environment,
    reward_model=reward_fn,
)
```

### When to Use Which

| Scenario | Recommended Algorithm |
|----------|----------------------|
| Short conversations (< 5 turns) | GRPO |
| Long conversations (> 10 turns) | GSPO |
| Stable, reliable training | GSPO |
| Fast experimentation | GRPO |
| Limited GPU memory | GSPO with LoRA |
| High reward variance | GSPO |

---

## Hyperparameter Optimization

### Quick HPO

```python
from stateset_agents.training.hpo import quick_hpo

summary = await quick_hpo(
    agent=agent,
    environment=environment,
    reward_function=reward_fn,
    base_config=config,
    n_trials=50,
    backend="optuna",  # or "ray_tune", "wandb"
)

print(f"Best parameters: {summary.best_params}")
print(f"Best reward: {summary.best_reward}")
```

### Search Spaces

```python
from stateset_agents.training.hpo import get_search_space

# Available search spaces
search_space = get_search_space("grpo")              # Core GRPO params
search_space = get_search_space("full")              # Comprehensive
search_space = get_search_space("customer_service")  # Domain-optimized
search_space = get_search_space("conservative")      # Narrow ranges
search_space = get_search_space("aggressive")        # Wide ranges
```

### Custom Search Space

```python
from stateset_agents.training.hpo import HPOConfig

hpo_config = HPOConfig(
    search_space={
        "learning_rate": {"type": "loguniform", "low": 1e-7, "high": 1e-4},
        "num_generations": {"type": "categorical", "choices": [4, 8, 16]},
        "temperature": {"type": "uniform", "low": 0.5, "high": 1.2},
        "clip_range": {"type": "uniform", "low": 0.1, "high": 0.3},
    },
    n_trials=100,
    optimization_direction="maximize",
    pruning=True,
)
```

---

## Advanced Features

### Distributed Training

```python
from stateset_agents.training.distributed import DistributedTrainer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

trainer = DistributedTrainer(
    agent=agent,
    environment=environment,
    reward_fn=reward_fn,
    config=config,
    num_gpus=4,
    strategy="ddp",  # Distributed Data Parallel
)

trained_agent = await trainer.train()
```

### LoRA Fine-tuning

Parameter-efficient training for large models:

```python
config = TrainingConfig(
    use_lora=True,
    lora_r=16,              # Rank
    lora_alpha=32,          # Scaling factor
    lora_dropout=0.05,
    lora_target_modules=["q_proj", "v_proj"],  # Layers to adapt
)
```

### Neural Reward Models

Train learned reward functions:

```python
from stateset_agents.training.neural_reward_trainer import NeuralRewardTrainer

# Train a neural reward model from human preferences
reward_trainer = NeuralRewardTrainer(
    model_name="gpt2",
    preference_data=preference_dataset,  # Pairs of (better, worse) responses
)

neural_reward = await reward_trainer.train()

# Use in RL training
trained_agent = await train(
    agent=agent,
    environment=environment,
    reward_fn=neural_reward,
)
```

### TRL Integration

Use HuggingFace TRL trainers:

```python
from stateset_agents.training.trl_grpo_wrapper import TRLGRPOTrainerWrapper

trainer = TRLGRPOTrainerWrapper(
    agent=agent,
    environment=environment,
    reward_fn=reward_fn,
    config=config,
)

trained_agent = await trainer.train()
```

### Context Compression

For long conversations that exceed context limits:

```python
config = AgentConfig(
    model_name="gpt2",
    enable_context_compression=True,
    compression_ratio=0.5,          # Keep 50% of context
    compression_strategy="summary", # or "truncate", "sliding_window"
)
```

### Conversation Memory

Maintain memory across episodes:

```python
from stateset_agents.core.memory import ConversationMemory

memory = ConversationMemory(
    max_turns=100,
    summarization_threshold=50,  # Summarize when exceeding this
)

agent = MultiTurnAgent(config, memory=memory)
```

---

## CLI Reference

### Environment Check

```bash
stateset-agents doctor
```

Checks:
- Python version
- Required dependencies
- GPU availability
- Memory requirements

### Initialize Configuration

```bash
stateset-agents init --path ./config.yaml
```

Creates a starter configuration file.

### Training

```bash
# Basic training
stateset-agents train --config ./config.yaml

# With checkpoint saving
stateset-agents train --config ./config.yaml --save ./checkpoints

# Stub mode (no GPU required)
stateset-agents train --config ./config.yaml --stub

# Resume from checkpoint
stateset-agents train --config ./config.yaml --resume ./checkpoints/step_1000
```

### Evaluation

```bash
# Interactive evaluation
stateset-agents evaluate --checkpoint ./checkpoints/final

# Single message
stateset-agents evaluate --checkpoint ./checkpoints/final --message "Hello!"

# Batch evaluation
stateset-agents evaluate --checkpoint ./checkpoints/final --input ./test_prompts.json
```

### Serving

```bash
# Start API server
stateset-agents serve --checkpoint ./checkpoints/final --port 8000

# With authentication
stateset-agents serve --checkpoint ./checkpoints/final --port 8000 --api-key $API_KEY
```

---

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)

```python
# Reduce batch size
config.per_device_train_batch_size = 1

# Increase gradient accumulation
config.gradient_accumulation_steps = 8

# Use LoRA
config.use_lora = True

# Reduce sequence length
config.max_completion_length = 256

# Use bfloat16
config.bf16 = True
```

#### Training Instability

```python
# Lower learning rate
config.learning_rate = 1e-6

# Use conservative profile
profile = "conservative"

# Switch to GSPO
from stateset_agents.training.gspo_trainer import train_with_gspo

# Reduce group size
config.num_generations = 4
```

#### Slow Training

```python
# Enable Flash Attention
agent_config.attn_implementation = "flash_attention_2"

# Use distributed training
trainer = DistributedTrainer(num_gpus=4)

# Increase batch size (if memory allows)
config.per_device_train_batch_size = 4
```

#### Reward Hacking

```python
# Add diversity reward
from stateset_agents.rewards import SimilarityAwareReward

reward_fn = CompositeReward([
    HelpfulnessReward(weight=0.5),
    SimilarityAwareReward(weight=0.2),  # Penalizes repetition
    SafetyReward(weight=0.3),
])

# Use KL penalty
config.kl_penalty = 0.1
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via environment variable
import os
os.environ["STATESET_DEBUG"] = "1"
```

### Getting Help

- Documentation: `docs/` directory
- Issues: https://github.com/stateset/stateset-agents/issues
- Check environment: `stateset-agents doctor`

---

## Example Projects

### 1. Customer Service Bot

```python
# See examples/customer_service_bot.py
```

### 2. Technical Support Agent

```python
# See examples/technical_support_agent.py
```

### 3. Sales Assistant

```python
# See examples/sales_assistant.py
```

### 4. Custom Domain Agent

```python
# See examples/custom_domain_agent.py
```

---

## Further Reading

- [Framework Overview](./FRAMEWORK_OVERVIEW.md)
- [GSPO Algorithm Guide](./GSPO_GUIDE.md)
- [TRL Integration Guide](./TRL_GRPO_TRAINING_GUIDE.md)
- [HPO Guide](./HPO_GUIDE.md)
- [Performance Tuning](./PERFORMANCE_TUNING_GUIDE.md)
- [Memory Requirements](./MEMORY_REQUIREMENTS_GUIDE.md)
