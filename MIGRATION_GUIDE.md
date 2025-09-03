# Migration Guide: v0.1.0 to v0.2.0

This guide helps you upgrade your code from GRPO Agent Framework v0.1.0 to v0.2.0, which includes significant enhancements based on real-world production learnings.

## Overview of Changes

### New Features in v0.2.0

1. **Data Processing Module**: Automatic data loading, validation, and splitting
2. **Domain-Specific Rewards**: Pre-built reward functions for common domains
3. **Enhanced GRPO Training**: Improved loss computation with KL penalties
4. **Post-Training Evaluation**: Comprehensive evaluation pipeline
5. **Model Optimization**: Built-in LoRA/PEFT support
6. **Mixed Precision Training**: BF16/FP16 support

### Breaking Changes

- `TrainingConfig` has new required parameters for enhanced training
- Reward functions now return `RewardResult` objects instead of raw scores
- Agent initialization now requires explicit `await agent.initialize()`

## Migration Steps

### 1. Update Data Loading

**Old (v0.1.0):**
```python
# Manual data loading
with open("data.json", "r") as f:
    data = json.load(f)

scenarios = []
for item in data:
    scenarios.append({
        "messages": item["conversation"]
    })
```

**New (v0.2.0):**
```python
from grpo_agent_framework import load_and_prepare_data

# Automatic loading with train/eval split
train_data, eval_data = load_and_prepare_data(
    "data.jsonl",
    max_examples=5000,
    validation_split=0.1
)
```

### 2. Update Reward Functions

**Old (v0.1.0):**
```python
class MyReward(RewardFunction):
    async def compute_reward(self, turns, context):
        score = calculate_score(turns)
        return score  # Just a float
```

**New (v0.2.0):**
```python
from grpo_agent_framework.core.reward import RewardResult

class MyReward(RewardFunction):
    async def compute_reward(self, turns, context):
        score = calculate_score(turns)
        return RewardResult(
            score=score,
            breakdown={"component": score},
            metadata={"computed": True}
        )
```

### 3. Use Domain-Specific Rewards

**Old (v0.1.0):**
```python
# Manual reward composition
reward_fn = CompositeReward([
    HelpfulnessReward(weight=0.5),
    SafetyReward(weight=0.3),
    CustomReward(weight=0.2)
])
```

**New (v0.2.0):**
```python
from grpo_agent_framework import create_domain_reward

# Automatic domain-specific reward
reward_fn = create_domain_reward(
    domain="customer_service",
    expected_responses=train_data.get_expected_responses()
)
```

### 4. Update Training Configuration

**Old (v0.1.0):**
```python
config = TrainingConfig(
    num_episodes=1000,
    learning_rate=5e-6,
    batch_size=4
)
```

**New (v0.2.0):**
```python
config = TrainingConfig(
    num_episodes=1000,
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    
    # New enhanced parameters
    num_generations=16,
    beta=0.01,  # KL penalty
    use_lora=True,
    bf16=True,
    run_post_eval=True,
    eval_split_size=0.1
)
```

### 5. Update Agent Initialization

**Old (v0.1.0):**
```python
agent = MultiTurnAgent(config)
# Agent was ready to use immediately
```

**New (v0.2.0):**
```python
agent = MultiTurnAgent(config)
await agent.initialize()  # Explicit initialization required
```

### 6. Add Post-Training Evaluation

**Old (v0.1.0):**
```python
# Manual evaluation after training
trained_agent = trainer.train()
# ... custom evaluation code ...
```

**New (v0.2.0):**
```python
# Automatic post-training evaluation
trained_agent = await trainer.train()

if config.run_post_eval:
    eval_results = await trainer.run_post_training_evaluation(
        eval_scenarios,
        num_samples=10
    )
    print(f"Mean reward: {eval_results['overall_stats']['overall_mean_reward']}")
```

## Complete Example: Before and After

### Before (v0.1.0)

```python
import json
from grpo_agent_framework import MultiTurnAgent, ConversationEnvironment, train

# Load data manually
with open("data.json", "r") as f:
    data = json.load(f)

# Create agent
agent = MultiTurnAgent.from_model("openai/gpt-oss-120b")

# Create environment
env = ConversationEnvironment(scenarios=data)

# Simple training
trained_agent = train(
    agent=agent,
    environment=env,
    num_episodes=1000
)
```

### After (v0.2.0)

```python
from grpo_agent_framework import (
    MultiTurnAgent, ConversationEnvironment,
    load_and_prepare_data, create_domain_reward
)
from grpo_agent_framework.training import MultiTurnGRPOTrainer, TrainingConfig

# Load and prepare data automatically
train_data, eval_data = load_and_prepare_data(
    "data.jsonl",
    max_examples=5000,
    validation_split=0.1
)

# Create enhanced agent
config = AgentConfig(
            model_name="openai/gpt-oss-120b",
    use_peft=True,
    bf16=True
)
agent = MultiTurnAgent(config)
await agent.initialize()

# Create environment with scenarios
env = ConversationEnvironment(
    scenarios=train_data.to_scenarios(),
    max_turns=10
)

# Domain-specific reward
reward_fn = create_domain_reward(
    "customer_service",
    expected_responses=train_data.get_expected_responses()
)

# Enhanced training configuration
training_config = TrainingConfig(
    num_episodes=1000,
    num_generations=16,
    use_lora=True,
    bf16=True,
    run_post_eval=True
)

# Train with enhanced features
trainer = MultiTurnGRPOTrainer(agent, env, reward_fn, training_config)
await trainer.initialize()
trained_agent = await trainer.train()

# Automatic evaluation
eval_results = await trainer.run_post_training_evaluation(eval_data)
```

## Deprecation Warnings

The following features are deprecated and will be removed in v0.3.0:

1. `Agent.from_model()` - Use `Agent(config)` with explicit initialization
2. Raw float returns from reward functions - Use `RewardResult` objects
3. `batch_size` parameter - Use `per_device_train_batch_size`

## Need Help?

- Check the [examples/enhanced_customer_service.py](examples/enhanced_customer_service.py) for a complete working example
- See the [API documentation](docs/api/) for detailed parameter descriptions
- Open an issue on GitHub if you encounter problems during migration 