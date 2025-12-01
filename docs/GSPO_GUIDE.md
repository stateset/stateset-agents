# Group Sequence Policy Optimization (GSPO) Guide

## Overview

**Group Sequence Policy Optimization (GSPO)** is a stable, efficient, and performant reinforcement learning algorithm for training large language models. GSPO was developed by the Qwen Team at Alibaba and has been proven to achieve superior training stability, efficiency, and performance compared to GRPO, especially for large Mixture-of-Experts (MoE) models.

## Key Features

### ✨ **Stability**
- **Sequence-level importance ratios** prevent token-level noise accumulation
- **No model collapse** even with long sequences and large models
- **Native MoE support** without requiring special stabilization strategies

### ⚡ **Efficiency**
- **Better sample utilization** despite higher clipping fractions
- **Superior training efficiency** compared to GRPO
- **Simplified infrastructure** requirements

### 🎯 **Performance**
- **Continuous improvement** with increased training compute
- **Better convergence** properties
- **Proven results** on Qwen3 models achieving state-of-the-art performance

## GSPO vs GRPO: Key Differences

| Aspect | GRPO | GSPO |
|--------|------|------|
| **Importance Ratio** | Token-level: π_θ(y_t\|x,y_<t) / π_θ_old(y_t\|x,y_<t) | Sequence-level: (π_θ(y\|x) / π_θ_old(y\|x))^(1/\|y\|) |
| **Clipping** | Per-token, range: [0.8, 1.2] | Per-sequence, range: [1-3e-4, 1+4e-4] |
| **Advantages** | Token-level (but all tokens share same value) | Sequence-level, group-normalized |
| **Stability** | Can suffer from noise accumulation | Highly stable, especially for long sequences |
| **MoE Training** | Requires Routing Replay strategy | No special strategies needed |
| **Clipping Fraction** | ~0.15 (15%) | ~15% (much higher but more efficient) |

## How GSPO Works

### 1. Sequence-Level Importance Ratio

GSPO uses a theoretically grounded sequence-level importance ratio with length normalization:

```
s_i(θ) = (π_θ(y_i|x) / π_θ_old(y_i|x))^(1/|y_i|)
       = exp(1/|y_i| * Σ log(π_θ(y_i,t|x,y_<t) / π_θ_old(y_i,t|x,y_<t)))
```

**Benefits:**
- Reflects how far the response deviates from the old policy
- Length normalization keeps ratios in a unified numerical range
- Aligns with sequence-level rewards

### 2. Group-Relative Advantages

Advantages are computed relative to a group of responses to the same query:

```
Â_i = (r(x, y_i) - mean({r(x, y_i)})) / std({r(x, y_i)})
```

### 3. Sequence-Level Clipping and Optimization

The GSPO objective applies clipping to entire responses:

```
J_GSPO(θ) = E[1/G * Σ min(s_i(θ) * Â_i, clip(s_i(θ), 1-ε, 1+ε) * Â_i)]
```

**Why this works:**
- Unit of optimization matches unit of reward (sequence-level)
- Prevents overly off-policy samples from affecting training
- More reliable gradient estimates than token-level approaches

## Installation and Setup

```bash
# Install StateSet Agents with all dependencies
pip install stateset-agents[dev]

# Or install from source
git clone https://github.com/stateset/stateset-agents
cd stateset-agents
pip install -e ".[dev]"
```

## Quick Start

### Basic GSPO Training

```python
import asyncio
from stateset_agents import MultiTurnAgent
from stateset_agents.core.agent import AgentConfig
from stateset_agents.core.environment import ConversationEnvironment, CONVERSATION_CONFIGS
from stateset_agents.rewards.multi_objective_reward import create_customer_service_reward
from training.gspo_trainer import GSPOConfig, train_with_gspo
from training.config import get_config_for_task

async def train_with_gspo_example():
    # Create agent
    agent = MultiTurnAgent(AgentConfig(
        model_name="gpt2",
        system_prompt="You are a helpful customer service representative.",
    ))
    await agent.initialize()

    # Create environment
    env_config = CONVERSATION_CONFIGS["customer_service"]
    environment = ConversationEnvironment(**env_config)

    # Create reward model
    reward_model = create_customer_service_reward()

    # Create GSPO configuration
    base_config = get_config_for_task("customer_service", model_name="gpt2")
    gspo_config = GSPOConfig.from_training_config(
        base_config,
        num_outer_iterations=100,
        num_generations=4,  # Group size
        clip_range_left=3e-4,
        clip_range_right=4e-4,
        output_dir="./outputs/gspo",
    )

    # Train
    trained_agent = await train_with_gspo(
        config=gspo_config,
        agent=agent,
        environment=environment,
        reward_model=reward_model,
    )

    return trained_agent

# Run training
asyncio.run(train_with_gspo_example())
```

### Using the Demo Script

```bash
# Train on customer service task
python examples/train_with_gspo.py --task customer_service --model gpt2

# Train on technical support
python examples/train_with_gspo.py --task technical_support --iterations 20

# Use GSPO-token variant
python examples/train_with_gspo.py --use-gspo-token

# View GSPO vs GRPO comparison
python examples/train_with_gspo.py --compare
```

## GSPO-token: Token-Level Variant

For scenarios requiring finer-grained advantage adjustment (e.g., multi-turn RL where different turns have different importance):

```python
from training.gspo_token_trainer import train_with_gspo_token

# Same setup as above, but call train_with_gspo_token instead
trained_agent = await train_with_gspo_token(
    config=gspo_config,
    agent=agent,
    environment=environment,
    reward_model=reward_model,
)
```

**Key Features of GSPO-token:**
- Allows token-wise advantage customization
- Maintains sequence-level importance ratios for clipping
- Numerically identical to GSPO when all tokens have same advantage
- Higher flexibility for multi-turn conversations

## Configuration Guide

### Essential Parameters

```python
gspo_config = GSPOConfig(
    # Group size (number of responses per query)
    num_generations=4,

    # Sequence-level clipping ranges
    clip_range_left=3e-4,   # 1 - ε
    clip_range_right=4e-4,  # 1 + ε

    # KL penalty coefficient (optional)
    beta=0.0,

    # Training iterations
    num_outer_iterations=100,
    generations_per_iteration=100,

    # Model optimization
    learning_rate=1e-5,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,

    # Memory optimization
    gradient_checkpointing=True,
    use_8bit=False,
    use_4bit=False,

    # Generation parameters
    max_prompt_length=256,
    max_completion_length=256,
    temperature=0.7,
    top_p=0.9,
)
```

### Recommended Settings by Use Case

#### Small Models (< 1B parameters)
```python
GSPOConfig(
    num_generations=4,
    clip_range_left=3e-4,
    clip_range_right=4e-4,
    learning_rate=1e-5,
    use_lora=True,
    lora_r=8,
)
```

#### Large Models (1B - 10B parameters)
```python
GSPOConfig(
    num_generations=4,
    clip_range_left=2e-4,
    clip_range_right=3e-4,
    learning_rate=5e-6,
    use_lora=True,
    lora_r=16,
    gradient_checkpointing=True,
)
```

#### MoE Models
```python
GSPOConfig(
    num_generations=6,  # Larger group for stability
    clip_range_left=2e-4,
    clip_range_right=3e-4,
    learning_rate=3e-6,
    use_lora=True,
    lora_r=32,
    gradient_checkpointing=True,
    # No Routing Replay needed!
)
```

## Advanced Usage

### Custom Reward Functions

```python
from stateset_agents.core.reward import RewardFunction, RewardResult

class CustomGSPOReward(RewardFunction):
    async def compute_reward(self, trajectory, turn, context):
        # Your custom reward logic
        score = compute_custom_score(turn.content)

        return RewardResult(
            total_reward=score,
            component_rewards={"custom": score},
            metadata={"details": "..."}
        )

# Use in training
reward_model = CustomGSPOReward()
trained_agent = await train_with_gspo(..., reward_model=reward_model)
```

### Monitoring with Weights & Biases

```python
gspo_config = GSPOConfig(
    ...
    report_to="wandb",
    wandb_project="my-gspo-project",
    wandb_tags=["gspo", "customer-service"],
    run_name="gspo-exp-001",
)
```

### Distributed Training

```python
# GSPO automatically uses all available GPUs
# Set CUDA_VISIBLE_DEVICES to control GPU usage
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Train normally - distribution is automatic
trained_agent = await train_with_gspo(...)
```

## Best Practices

### 1. **Start with Default Hyperparameters**
The default GSPO settings are well-tuned. Only adjust if you observe specific issues.

### 2. **Monitor Clipping Fractions**
GSPO typically has higher clipping fractions (~15%) than GRPO (~0.15%). This is normal and expected.

### 3. **Use LoRA for Large Models**
Always enable LoRA for models > 1B parameters to reduce memory usage and training time.

### 4. **Gradually Increase Sequence Length**
Start with shorter sequences and gradually increase `max_completion_length` as training progresses.

### 5. **Group Size Selection**
- Smaller groups (2-4): Faster training, less stable
- Larger groups (6-8): Slower but more stable advantages

### 6. **MoE Models**
GSPO handles MoE models naturally. No special configuration needed!

## Troubleshooting

### Issue: Training Loss Not Decreasing

**Solutions:**
- Reduce learning rate (try 1e-6 or 5e-7)
- Increase group size to 6 or 8
- Check reward function is returning reasonable values

### Issue: High Memory Usage

**Solutions:**
- Enable `gradient_checkpointing=True`
- Reduce `per_device_train_batch_size`
- Use `use_4bit=True` for quantization
- Reduce `max_completion_length`

### Issue: Training Instability

**Solutions:**
- Increase group size for more stable advantages
- Reduce clipping ranges slightly
- Check reward function isn't returning extreme values
- Enable reference model KL penalty with `beta=0.01`

## Performance Benchmarks

Based on the original GSPO paper (Qwen Team):

| Metric | GRPO | GSPO | Improvement |
|--------|------|------|-------------|
| Training Stability | Moderate | High | ✅ More stable |
| Clipping Fraction | 0.0013 | 0.15 | Different scale |
| Sample Efficiency | Baseline | +15-20% | ✅ Better |
| MoE Convergence | Requires tricks | Native | ✅ Simplified |
| Long Sequences | Unstable | Stable | ✅ Much better |

## Integration with Existing Code

### Adding GSPO to RL Orchestrator

```python
from core.enhanced.advanced_rl_algorithms import create_advanced_rl_orchestrator

# Create orchestrator with all algorithms including GSPO
orchestrator = create_advanced_rl_orchestrator(agent)

# GSPO is now available alongside PPO, DPO, A2C
# Note: Use training.gspo_trainer for full implementation
```

## References

- **Original Paper:** [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071v2)
- **Authors:** Qwen Team, Alibaba Inc.
- **Implementation:** StateSet Agents Framework

## Contributing

We welcome contributions to improve GSPO implementation:

1. Report issues on GitHub
2. Submit pull requests with improvements
3. Share your training results and insights

## License

This implementation is part of StateSet Agents and follows the same license (BUSL-1.1).

---

**Questions or need help?** Open an issue on [GitHub](https://github.com/stateset/stateset-agents/issues) or join our [Discord](https://discord.gg/stateset).
