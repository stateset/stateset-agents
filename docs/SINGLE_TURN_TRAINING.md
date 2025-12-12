# Single-Turn Training Guide

## Overview

StateSet Agents v0.5.0+ now supports **single-turn training** for simpler use cases where you don't need multi-turn conversation handling. This is ideal for:

- Question-answering tasks
- Single-shot text generation
- Classification and labeling
- Simple prompt-response scenarios

## Quick Start

```python
import asyncio
from stateset_agents.core.agent import Agent, AgentConfig
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import HelpfulnessReward
from training.trainer import SingleTurnGRPOTrainer
from training.config import TrainingConfig

async def train_single_turn():
    # Create a basic agent
    config = AgentConfig(model_name="gpt2", max_new_tokens=50)
    agent = Agent(config)

    # Create environment
    scenarios = [
        {
            "id": "qa1",
            "topic": "question_answering",
            "context": "Answer questions accurately",
            "user_responses": ["What is Python?", "Explain variables"]
        }
    ]
    environment = ConversationEnvironment(scenarios=scenarios, max_turns=1)

    # Create reward function
    reward_fn = HelpfulnessReward(weight=1.0)

    # Create training config
    train_config = TrainingConfig(
        num_episodes=10,
        max_steps_per_episode=50,
        learning_rate=5e-5
    )

    # Create single-turn trainer
    trainer = SingleTurnGRPOTrainer(
        agent=agent,
        environment=environment,
        reward_fn=reward_fn,
        config=train_config
    )

    # Initialize and train
    await trainer.initialize()
    trained_agent = await trainer.train()

    # Save checkpoint
    await trainer.save_checkpoint("./checkpoints/single_turn_model")

    return trained_agent

# Run training
asyncio.run(train_single_turn())
```

## Architecture

### SingleTurnGRPOTrainer

The `SingleTurnGRPOTrainer` implements GRPO for single-turn interactions:

**Key Features:**
- Simplified conversation handling (one input → one output)
- Faster training (no conversation state management)
- Lower memory footprint
- HuggingFace transformer integration
- Mixed precision training support (FP16/BF16)
- Weights & Biases logging

**Methods:**
- `__init__(agent, environment, reward_fn, config, wandb_logger, callbacks)` - Initialize trainer
- `async initialize()` - Setup optimizer, seeds, and components
- `async train()` - Run training loop
- `async save_checkpoint(path)` - Save trained model
- `add_callback(callback)` - Add training callback

## Configuration

### TrainingConfig Options

```python
from training.config import TrainingConfig

config = TrainingConfig(
    num_episodes=50,              # Number of training episodes
    max_steps_per_episode=100,    # Max steps per episode
    learning_rate=1e-4,           # Optimizer learning rate
    weight_decay=0.01,            # Weight decay for regularization
    per_device_train_batch_size=8, # Batch size
    bf16=True,                     # Use bfloat16 mixed precision
    fp16=False,                    # Use float16 mixed precision
    max_grad_norm=1.0,            # Gradient clipping
    seed=42                        # Random seed for reproducibility
)
```

## Single-Turn vs Multi-Turn

| Feature | Single-Turn | Multi-Turn |
|---------|------------|------------|
| **Use Case** | Q&A, classification | Conversations, dialogues |
| **Memory** | Low | Higher (tracks history) |
| **Speed** | Faster | Slower |
| **Complexity** | Simple | Complex |
| **State Management** | None | Full conversation state |
| **Trainer Class** | `SingleTurnGRPOTrainer` | `MultiTurnGRPOTrainer` |

## CLI Usage

Train in single-turn mode using the CLI:

```bash
# Using automatic mode detection
stateset-agents train --config single_turn_config.yaml

# The trainer automatically detects single-turn vs multi-turn based on agent type
```

### Example Config (single_turn_config.yaml)

```yaml
agent:
  model_name: "gpt2"
  max_new_tokens: 50
  temperature: 0.7

environment:
  type: "conversation"
  scenarios:
    - id: "simple_qa"
      topic: "question_answering"
      context: "Answer questions"
      user_responses:
        - "What is machine learning?"
        - "Explain neural networks"

training:
  num_episodes: 20
  max_turns: 1  # Single turn
  learning_rate: 5e-5
```

## Advanced Features

### Custom Rewards

```python
from stateset_agents.core.reward import RewardFunction, RewardResult

class CustomSingleTurnReward(RewardFunction):
    async def compute_reward(self, turns, context=None):
        # Custom logic for single-turn scoring
        response = turns[-1].get("content", "") if turns else ""
        score = len(response) / 100.0  # Example: reward longer responses

        return RewardResult(
            score=min(1.0, score),
            breakdown={"length": len(response)},
            metadata={"type": "custom"}
        )

reward_fn = CustomSingleTurnReward()
```

### Callbacks

```python
class TrainingCallback:
    def on_episode_start(self, episode):
        print(f"Starting episode {episode}")

    def on_episode_end(self, episode, metrics):
        print(f"Episode {episode} complete: {metrics}")

trainer = SingleTurnGRPOTrainer(
    agent=agent,
    environment=environment,
    config=config,
    callbacks=[TrainingCallback()]
)
```

### W&B Integration

```python
from utils.wandb_integration import WandBLogger

wandb_logger = WandBLogger(project="single-turn-training", name="qa-model")

trainer = SingleTurnGRPOTrainer(
    agent=agent,
    environment=environment,
    wandb_logger=wandb_logger,
    config=config
)
```

## Performance Tips

1. **Use Mixed Precision**: Enable `bf16=True` for A100/H100 GPUs
2. **Batch Size**: Start with 8-16, increase based on GPU memory
3. **Learning Rate**: 1e-5 to 5e-5 works well for most cases
4. **Episodes**: 20-50 episodes sufficient for simple tasks
5. **Gradient Clipping**: Keep `max_grad_norm=1.0` for stability

## Examples

### Question Answering

```python
# See examples/single_turn_qa.py for full example
scenarios = [
    {
        "topic": "science_qa",
        "context": "Answer science questions accurately",
        "user_responses": [
            "What is photosynthesis?",
            "Explain gravity",
            "What are atoms?"
        ]
    }
]
```

### Text Classification

```python
scenarios = [
    {
        "topic": "sentiment_analysis",
        "context": "Classify text sentiment",
        "user_responses": [
            "This product is amazing!",
            "Terrible service, very disappointed",
            "It's okay, nothing special"
        ]
    }
]
```

## Troubleshooting

### Issue: Training is slow
**Solution**: Enable mixed precision (`bf16=True`), increase batch size, reduce sequence length

### Issue: Out of memory
**Solution**: Reduce `per_device_train_batch_size`, use `fp16` instead of `bf16`, reduce `max_new_tokens`

### Issue: Model not learning
**Solution**: Increase `num_episodes`, adjust `learning_rate`, check reward function is providing meaningful signal

## Migration from Multi-Turn

To convert multi-turn training to single-turn:

1. Change agent from `MultiTurnAgent` to `Agent`
2. Change trainer from `MultiTurnGRPOTrainer` to `SingleTurnGRPOTrainer`
3. Set `max_turns=1` in environment config
4. Simplify reward functions (no need to track conversation history)

## API Reference

See [API Documentation](api/training.md) for complete API reference.

## See Also

- [Multi-Turn Training Guide](MULTI_TURN_TRAINING.md)
- [Training Configuration](TRAINING_CONFIG.md)
- [Reward Functions](REWARD_FUNCTIONS.md)
- [CLI Reference](CLI.md)
