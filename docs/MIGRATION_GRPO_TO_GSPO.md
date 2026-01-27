# Migration Guide: GRPO to GSPO

This guide helps you migrate your existing GRPO training code to GSPO (Group Sequence Policy Optimization) for improved stability and performance.

## Table of Contents

1. [Why Migrate to GSPO?](#why-migrate-to-gspo)
2. [Key Differences](#key-differences)
3. [Step-by-Step Migration](#step-by-step-migration)
4. [Configuration Mapping](#configuration-mapping)
5. [Code Examples](#code-examples)
6. [Troubleshooting](#troubleshooting)

---

## Why Migrate to GSPO?

### Benefits of GSPO over GRPO

| Aspect | GRPO | GSPO |
|--------|------|------|
| **Stability** | Can have noise accumulation | Highly stable |
| **Long Sequences** | May become unstable | Handles well |
| **MoE Models** | Needs Routing Replay | Works natively |
| **Sample Efficiency** | Baseline | 15-20% better |
| **Training Speed** | Baseline | Similar |

### When to Migrate

**Strongly Recommended:**
- Training on long sequences (> 1024 tokens)
- Using Mixture-of-Experts models
- Experiencing training instability
- Training large models (> 7B parameters)

**Optional:**
- Short sequence training that's already stable
- Small models with good convergence
- Existing training pipelines with acceptable results

---

## Key Differences

### 1. Importance Ratio Calculation

**GRPO (Token-level):**
```
r_t(θ) = π_θ(y_t|x,y_<t) / π_θ_old(y_t|x,y_<t)
```

**GSPO (Sequence-level):**
```
s_i(θ) = exp(1/|y_i| * Σ log(π_θ(y_i,t|x,y_<t) / π_θ_old(y_i,t|x,y_<t)))
```

### 2. Clipping Ranges

**GRPO:**
- Typical range: `[0.8, 1.2]` (20% deviation)
- Applied per token

**GSPO:**
- Typical range: `[1-3e-4, 1+4e-4]` (much tighter)
- Applied per sequence

### 3. Advantage Computation

Both use group-relative advantages, but GSPO applies them at the sequence level:

```python
# GRPO: Token-level advantage (all tokens share same value)
advantage_grpo = reward - baseline  # Applied to each token

# GSPO: Sequence-level advantage
advantage_gspo = (reward - mean(rewards)) / std(rewards)  # Per sequence
```

---

## Step-by-Step Migration

### Step 1: Update Imports

```python
# Before (GRPO)
from stateset_agents.training.trainer import GRPOTrainer
from stateset_agents.training.config import TrainingConfig

# After (GSPO)
from stateset_agents.training.gspo_trainer import GSPOTrainer, GSPOConfig, train_with_gspo
```

### Step 2: Update Configuration

```python
# Before (GRPO)
grpo_config = TrainingConfig(
    num_generations=4,
    cliprange=0.2,
    kl_coef=0.1,
    learning_rate=1e-5,
    max_completion_length=256,
)

# After (GSPO)
gspo_config = GSPOConfig(
    num_generations=4,
    clip_range_left=3e-4,    # NEW: Sequence-level clipping
    clip_range_right=4e-4,   # NEW: Asymmetric clipping
    beta=0.0,                # KL penalty (usually not needed)
    learning_rate=1e-5,
    max_completion_length=256,
)
```

### Step 3: Update Training Call

```python
# Before (GRPO)
trainer = GRPOTrainer(
    config=grpo_config,
    agent=agent,
    environment=environment,
    reward_model=reward_model,
)
trained_agent = await trainer.train()

# After (GSPO)
trained_agent = await train_with_gspo(
    config=gspo_config,
    agent=agent,
    environment=environment,
    reward_model=reward_model,
)
```

### Step 4: Adjust Hyperparameters (if needed)

GSPO typically needs minimal tuning from defaults, but you may need to adjust:

```python
# If training was stable with GRPO, start with these:
gspo_config = GSPOConfig(
    num_generations=4,        # Keep same group size
    clip_range_left=3e-4,     # Default
    clip_range_right=4e-4,    # Default
    learning_rate=1e-5,       # Keep same or slightly lower
)

# If training was unstable with GRPO, try:
gspo_config = GSPOConfig(
    num_generations=6,        # Larger group for stability
    clip_range_left=2e-4,     # Tighter clipping
    clip_range_right=3e-4,
    learning_rate=5e-6,       # Lower learning rate
)
```

---

## Configuration Mapping

### Direct Parameter Mapping

| GRPO Parameter | GSPO Parameter | Notes |
|----------------|----------------|-------|
| `num_generations` | `num_generations` | Same meaning, keep value |
| `cliprange` | N/A | Replaced by sequence-level clipping |
| N/A | `clip_range_left` | New: left clipping bound |
| N/A | `clip_range_right` | New: right clipping bound |
| `kl_coef` | `beta` | Similar, but often 0 in GSPO |
| `learning_rate` | `learning_rate` | Keep same or slightly lower |
| `max_completion_length` | `max_completion_length` | Same |
| `use_lora` | `use_lora` | Same |
| `gradient_checkpointing` | `gradient_checkpointing` | Same |

### Parameters to Remove

These GRPO-specific parameters are not used in GSPO:
- `cliprange` (replaced by sequence-level clipping)
- Token-level advantage parameters

### New Parameters to Add

```python
gspo_config = GSPOConfig(
    # Required GSPO parameters
    clip_range_left=3e-4,     # 1 - ε
    clip_range_right=4e-4,    # 1 + ε

    # Optional GSPO parameters
    beta=0.0,                 # KL penalty (default: 0)
    use_gspo_token=False,     # Token-level variant (default: False)
)
```

---

## Code Examples

### Example 1: Basic Migration

**Before (GRPO):**
```python
import asyncio
from stateset_agents import MultiTurnAgent
from stateset_agents.core.agent import AgentConfig
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.training.trainer import GRPOTrainer
from stateset_agents.training.config import TrainingConfig

async def train_grpo():
    agent = MultiTurnAgent(AgentConfig(model_name="gpt2"))
    await agent.initialize()

    environment = ConversationEnvironment(name="customer_service")
    reward_model = create_reward_model()

    config = TrainingConfig(
        num_generations=4,
        cliprange=0.2,
        learning_rate=1e-5,
        num_train_epochs=3,
    )

    trainer = GRPOTrainer(config=config, agent=agent,
                          environment=environment, reward_model=reward_model)
    return await trainer.train()

asyncio.run(train_grpo())
```

**After (GSPO):**
```python
import asyncio
from stateset_agents import MultiTurnAgent
from stateset_agents.core.agent import AgentConfig
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.training.gspo_trainer import GSPOConfig, train_with_gspo

async def train_gspo():
    agent = MultiTurnAgent(AgentConfig(model_name="gpt2"))
    await agent.initialize()

    environment = ConversationEnvironment(name="customer_service")
    reward_model = create_reward_model()

    config = GSPOConfig(
        num_generations=4,
        clip_range_left=3e-4,
        clip_range_right=4e-4,
        learning_rate=1e-5,
        num_outer_iterations=100,
    )

    return await train_with_gspo(
        config=config,
        agent=agent,
        environment=environment,
        reward_model=reward_model,
    )

asyncio.run(train_gspo())
```

### Example 2: Advanced Configuration

**Before (GRPO with all options):**
```python
grpo_config = TrainingConfig(
    # Model
    model_name="meta-llama/Llama-2-7b-hf",
    use_lora=True,
    lora_r=16,

    # Training
    num_generations=4,
    cliprange=0.2,
    kl_coef=0.1,
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    # Generation
    max_prompt_length=256,
    max_completion_length=256,
    temperature=0.7,

    # Optimization
    gradient_checkpointing=True,
    bf16=True,

    # Logging
    output_dir="./outputs/grpo",
    logging_steps=10,
    report_to="wandb",
)
```

**After (GSPO equivalent):**
```python
gspo_config = GSPOConfig(
    # Model
    model_name="meta-llama/Llama-2-7b-hf",
    use_lora=True,
    lora_r=16,

    # Training
    num_generations=4,
    clip_range_left=3e-4,   # Replaces cliprange
    clip_range_right=4e-4,
    beta=0.0,               # Usually don't need KL penalty
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    # Generation
    max_prompt_length=256,
    max_completion_length=256,
    temperature=0.7,

    # Optimization
    gradient_checkpointing=True,
    bf16=True,

    # Logging
    output_dir="./outputs/gspo",
    logging_steps=10,
    report_to="wandb",
)
```

### Example 3: Converting Existing Training Script

**create a conversion helper:**
```python
def convert_grpo_to_gspo_config(grpo_config):
    """Convert GRPO TrainingConfig to GSPOConfig."""
    from stateset_agents.training.gspo_trainer import GSPOConfig

    return GSPOConfig(
        # Model settings (keep same)
        model_name=grpo_config.model_name,
        use_lora=grpo_config.use_lora,
        lora_r=grpo_config.lora_r,
        lora_alpha=grpo_config.lora_alpha,

        # Training settings
        num_generations=grpo_config.num_generations,
        clip_range_left=3e-4,  # Default GSPO values
        clip_range_right=4e-4,
        beta=0.0 if grpo_config.kl_coef < 0.05 else grpo_config.kl_coef,
        learning_rate=grpo_config.learning_rate,
        per_device_train_batch_size=grpo_config.per_device_train_batch_size,
        gradient_accumulation_steps=grpo_config.gradient_accumulation_steps,

        # Generation settings (keep same)
        max_prompt_length=grpo_config.max_prompt_length,
        max_completion_length=grpo_config.max_completion_length,
        temperature=grpo_config.temperature,
        top_p=grpo_config.top_p,

        # Optimization settings (keep same)
        gradient_checkpointing=grpo_config.gradient_checkpointing,
        bf16=grpo_config.bf16,
        fp16=grpo_config.fp16,

        # Output settings (keep same)
        output_dir=grpo_config.output_dir.replace('grpo', 'gspo'),
        logging_steps=grpo_config.logging_steps,
        report_to=grpo_config.report_to,
    )
```

---

## Troubleshooting

### Issue: Loss Not Decreasing

**Possible Causes:**
1. Learning rate too high
2. Clipping too aggressive
3. Reward function issues

**Solutions:**
```python
# Try lower learning rate
gspo_config.learning_rate = 5e-6

# Relax clipping slightly
gspo_config.clip_range_left = 5e-4
gspo_config.clip_range_right = 6e-4

# Increase group size
gspo_config.num_generations = 6
```

### Issue: Higher Clipping Fraction Than Expected

**This is normal!** GSPO typically has ~15% clipping fraction vs GRPO's ~0.15%. The sequence-level clipping works differently and is expected to clip more often while still being more efficient.

### Issue: Memory Usage Different

GSPO and GRPO have similar memory profiles, but generation memory may vary:

```python
# If running out of memory, try:
gspo_config.gradient_checkpointing = True
gspo_config.per_device_train_batch_size = 1
gspo_config.gradient_accumulation_steps = 8
```

### Issue: Training Takes Longer

GSPO computation per step is similar to GRPO. If training is slower:

1. Check generation parameters are the same
2. Verify batch sizes match
3. Check for unnecessary logging

### Issue: Results Different Than GRPO

Some variance is expected. GSPO often produces:
- More stable training curves
- Better final performance
- Different intermediate checkpoints

If results are significantly worse:
1. Verify reward function works correctly with GSPO
2. Check all parameters were migrated correctly
3. Try GSPO-token variant for more GRPO-like behavior

---

## GSPO-Token: Middle Ground

If you need finer control (like GRPO) but want GSPO stability:

```python
from stateset_agents.training.gspo_trainer import train_with_gspo_token

trained_agent = await train_with_gspo_token(
    config=gspo_config,
    agent=agent,
    environment=environment,
    reward_model=reward_model,
)
```

GSPO-token provides:
- Sequence-level importance ratios (stable)
- Token-level advantage customization (flexible)
- Best of both worlds for multi-turn conversations

---

## Migration Checklist

- [ ] Update imports to use `GSPOConfig` and `train_with_gspo`
- [ ] Replace `cliprange` with `clip_range_left` and `clip_range_right`
- [ ] Set `beta` (KL penalty) - usually 0.0 for GSPO
- [ ] Keep `num_generations` the same (or increase for stability)
- [ ] Keep `learning_rate` the same (or slightly lower)
- [ ] Update output directory to avoid overwriting GRPO results
- [ ] Run validation to compare results
- [ ] Monitor training metrics for expected behavior

---

## Additional Resources

- [GSPO Guide](GSPO_GUIDE.md) - Detailed GSPO documentation
- [Performance Tuning Guide](PERFORMANCE_TUNING_GUIDE.md) - Optimization tips
- [Original GSPO Paper](https://arxiv.org/abs/2507.18071v2) - Academic reference
