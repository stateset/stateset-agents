
# Hyperparameter Optimization (HPO) Guide

**StateSet Agents** provides comprehensive, production-ready hyperparameter optimization to automatically find the best training configuration for your reinforcement learning agents.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Search Spaces](#search-spaces)
- [HPO Backends](#hpo-backends)
- [Configuration](#configuration)
- [Integration with GRPO](#integration-with-grpo)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)
- [Examples](#examples)

---

## Overview

### What is HPO?

Hyperparameter optimization automates the process of finding the best hyperparameters for your model. Instead of manually tuning learning rates, batch sizes, and other parameters, HPO systematically explores the parameter space to find optimal configurations.

### Key Features

- **Multiple Backends**: Optuna (TPE), Ray Tune, W&B Sweeps
- **Pre-defined Search Spaces**: Domain-specific and algorithm-specific
- **Smart Pruning**: Early stopping of unpromising trials
- **Distributed Support**: Run trials in parallel across GPUs/nodes
- **Rich Visualization**: Optimization history, parameter importance, parallel coordinates
- **Seamless Integration**: Works with existing GRPO trainers

### Installation

```bash
# Basic HPO with Optuna
pip install stateset-agents[training] optuna

# Optional: visualization
pip install plotly kaleido

# Optional: Ray Tune for distributed HPO
pip install ray[tune]

# Optional: W&B integration
pip install wandb
```

---

## Quick Start

### 1. Quick HPO (Simplest)

```python
from training.hpo import quick_hpo
from stateset_agents.core.agent import MultiTurnAgent, AgentConfig
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import CompositeReward
from training.config import TrainingConfig

# Setup components
agent = MultiTurnAgent(AgentConfig(...))
environment = ConversationEnvironment(scenarios)
reward_function = CompositeReward([...])
base_config = TrainingConfig(...)

# Run HPO with defaults
summary = await quick_hpo(
    agent=agent,
    environment=environment,
    reward_function=reward_function,
    base_config=base_config,
    n_trials=50,  # Number of configurations to try
    search_space_name="grpo"  # Pre-defined search space
)

print(f"Best params: {summary.best_params}")
print(f"Best metric: {summary.best_metric}")
```

### 2. Full Workflow (HPO + Training)

```python
from training.hpo import GRPOHPOTrainer, HPOConfig

# Configure HPO
hpo_config = HPOConfig(
    backend="optuna",
    search_space_name="grpo",
    n_trials=100,
    objective_metric="reward",
    direction="maximize"
)

# Create HPO trainer
hpo_trainer = GRPOHPOTrainer(
    agent=agent,
    environment=environment,
    reward_function=reward_function,
    base_config=base_config,
    hpo_config=hpo_config
)

# Step 1: Find best hyperparameters
summary = await hpo_trainer.optimize()

# Step 2: Train with best parameters
final_agent = await hpo_trainer.train_with_best_params(
    full_episodes=500  # Full training episodes
)

# Step 3: Analyze results
hpo_trainer.plot_results()  # Generate visualizations
```

---

## Search Spaces

### Pre-defined Search Spaces

StateSet Agents provides battle-tested search spaces for common scenarios:

#### Algorithm-Specific

```python
from training.hpo.search_spaces import (
    create_grpo_search_space,
    create_optimizer_search_space,
    create_model_architecture_search_space,
    create_generation_search_space,
)

# Core GRPO parameters
grpo_space = create_grpo_search_space(
    include_value_function=True,
    include_kl_penalty=True,
    include_ppo_clipping=False
)

# Optimizer hyperparameters
opt_space = create_optimizer_search_space()

# Model architecture (LoRA, value head)
arch_space = create_model_architecture_search_space()

# Generation parameters (temperature, top_p, etc.)
gen_space = create_generation_search_space()

# Full search space (all of the above)
full_space = create_full_search_space(
    include_optimizer=True,
    include_architecture=True,
    include_generation=True
)
```

#### Domain-Specific

```python
from training.hpo.search_spaces import (
    create_customer_service_search_space,
    create_technical_support_search_space,
    create_sales_assistant_search_space,
)

# Optimized for customer service
cs_space = create_customer_service_search_space()
# Includes: helpfulness_weight, safety_weight, engagement_weight, etc.

# Optimized for technical support
tech_space = create_technical_support_search_space()
# Includes: correctness_weight, helpfulness_weight, detail_level, etc.

# Optimized for sales
sales_space = create_sales_assistant_search_space()
# Includes: engagement_weight, persuasiveness_weight, task_completion_weight
```

#### Training Profiles

```python
from training.hpo.search_spaces import (
    create_conservative_search_space,
    create_aggressive_search_space,
)

# Conservative: narrow ranges, safe defaults
conservative = create_conservative_search_space()
# - learning_rate: 1e-6 to 1e-5
# - kl_penalty_coef: 0.05 to 0.15

# Aggressive: wide ranges, exploration-focused
aggressive = create_aggressive_search_space()
# - learning_rate: 5e-6 to 1e-3
# - kl_penalty_coef: 0.001 to 0.05
```

### Custom Search Spaces

```python
from training.hpo import SearchSpace, SearchDimension, SearchSpaceType

custom_space = SearchSpace([
    # Log-uniform (best for learning rates)
    SearchDimension(
        "learning_rate",
        SearchSpaceType.LOGUNIFORM,
        low=1e-6,
        high=1e-3,
        default=1e-5
    ),

    # Uniform (for continuous parameters)
    SearchDimension(
        "gamma",
        SearchSpaceType.UNIFORM,
        low=0.9,
        high=0.999,
        default=0.99
    ),

    # Categorical (for discrete choices)
    SearchDimension(
        "optimizer",
        SearchSpaceType.CATEGORICAL,
        choices=["adam", "adamw", "sgd"]
    ),

    # Choice (for numeric discrete values)
    SearchDimension(
        "batch_size",
        SearchSpaceType.CHOICE,
        choices=[16, 32, 64, 128]
    ),
])
```

### Available Parameters

Here are the key GRPO parameters you can optimize:

| Parameter | Type | Default Range | Description |
|-----------|------|---------------|-------------|
| `learning_rate` | loguniform | 1e-6 to 1e-3 | Policy learning rate |
| `value_lr_multiplier` | uniform | 0.5 to 2.0 | Value function LR multiplier |
| `gamma` | uniform | 0.9 to 0.999 | Discount factor |
| `gae_lambda` | uniform | 0.9 to 0.99 | GAE lambda |
| `kl_penalty_coef` | loguniform | 0.001 to 0.1 | KL divergence penalty |
| `max_grad_norm` | uniform | 0.5 to 2.0 | Gradient clipping |
| `group_size` | choice | [4, 8, 16, 32] | GRPO group size |
| `per_device_train_batch_size` | choice | [1, 2, 4, 8, 16] | Batch size per GPU |
| `gradient_accumulation_steps` | choice | [1, 2, 4, 8, 16] | Gradient accumulation |
| `temperature` | uniform | 0.7 to 1.5 | Generation temperature |
| `top_p` | uniform | 0.8 to 0.99 | Nucleus sampling |

---

## HPO Backends

### Optuna (Recommended)

Optuna uses Tree-structured Parzen Estimator (TPE) for efficient Bayesian optimization.

```python
hpo_config = HPOConfig(
    backend="optuna",
    search_space_name="grpo",
    n_trials=100,
    optuna_config={
        "sampler": "tpe",           # TPE, random, or cmaes
        "pruner": "median",         # median, hyperband, percentile, none
        "n_startup_trials": 10,     # Random trials before TPE
        "n_warmup_steps": 5,        # Warmup before pruning
        "storage": None,            # Optional: "sqlite:///optuna.db"
    }
)
```

**Samplers**:
- `tpe`: Tree-structured Parzen Estimator (Bayesian optimization) - **recommended**
- `random`: Random search (baseline)
- `cmaes`: CMA-ES (for continuous parameters)

**Pruners**:
- `median`: Prune if below median of all trials - **recommended**
- `hyperband`: Aggressive pruning with Hyperband
- `percentile`: Prune if below Nth percentile
- `none`: No pruning

### Ray Tune (Coming Soon)

Distributed optimization with Ray Tune:

```python
hpo_config = HPOConfig(
    backend="ray_tune",
    search_space_name="full",
    n_trials=200,
    n_parallel_trials=8,  # Run 8 trials in parallel
    ray_config={
        "scheduler": "asha",      # ASHA, pbt, hyperband
        "search_alg": "bayesopt", # bayesopt, ax, hyperopt
        "max_concurrent": 8,
    }
)
```

### W&B Sweeps (Coming Soon)

Integration with Weights & Biases:

```python
hpo_config = HPOConfig(
    backend="wandb",
    search_space_name="grpo",
    n_trials=100,
    wandb_config={
        "method": "bayes",        # bayes, grid, random
        "project": "my-project",
        "entity": "my-team",
    }
)
```

---

## Configuration

### HPOConfig Options

```python
from training.hpo import HPOConfig

config = HPOConfig(
    # Core settings
    backend="optuna",                    # "optuna", "ray_tune", "wandb"
    search_space=None,                   # Custom SearchSpace
    search_space_name="grpo",            # Or use pre-defined
    n_trials=100,                        # Number of trials
    timeout=None,                        # Optional timeout (seconds)
    objective_metric="reward",           # Metric to optimize
    direction="maximize",                # "maximize" or "minimize"
    output_dir="./hpo_results",          # Output directory
    study_name=None,                     # Optional study name

    # Backend-specific
    optuna_config={...},                 # Optuna settings
    ray_config={...},                    # Ray Tune settings
    wandb_config={...},                  # W&B settings

    # Training integration
    base_training_config=None,           # Base config to override
    checkpoint_freq=10,                  # Checkpoint every N trials
    keep_best_n=5,                       # Keep top 5 checkpoints
    early_stopping_patience=None,        # Stop if no improvement

    # Resources
    n_parallel_trials=1,                 # Parallel trials
    gpu_per_trial=1.0,                   # GPUs per trial
    cpu_per_trial=4,                     # CPUs per trial

    # Logging
    verbose=True,                        # Verbose output
    log_to_wandb=False,                  # Log to W&B
    save_plots=True,                     # Save visualization plots
)
```

### Pre-defined Profiles

```python
from training.hpo import get_hpo_config

# Conservative: safe, narrow ranges
conservative_config = get_hpo_config("conservative")

# Aggressive: wide exploration
aggressive_config = get_hpo_config("aggressive")

# Quick: fast, fewer trials
quick_config = get_hpo_config("quick")

# Distributed: parallel trials
distributed_config = get_hpo_config("distributed")
```

---

## Integration with GRPO

### Basic Integration

```python
from training.hpo import GRPOHPOTrainer, HPOConfig

# Your existing setup
agent = MultiTurnAgent(...)
environment = ConversationEnvironment(...)
reward_function = CompositeReward([...])
base_config = TrainingConfig(
    learning_rate=1e-5,  # Will be overridden by HPO
    num_episodes=500,
    # ... other settings
)

# Add HPO
hpo_config = HPOConfig(
    backend="optuna",
    search_space_name="grpo",
    n_trials=50
)

trainer = GRPOHPOTrainer(
    agent=agent,
    environment=environment,
    reward_function=reward_function,
    base_config=base_config,
    hpo_config=hpo_config
)

summary = await trainer.optimize()
```

### What Gets Optimized?

HPO searches for optimal values of:
- **GRPO algorithm parameters**: learning rate, gamma, GAE lambda, KL penalty
- **Training parameters**: batch size, gradient accumulation
- **Optimizer parameters**: Adam betas, weight decay
- **Model parameters**: LoRA rank, dropout
- **Generation parameters**: temperature, top_p, top_k
- **Domain-specific weights**: reward component weights

The `base_config` provides defaults for parameters not in the search space.

---

## Advanced Usage

### Custom Objective Function

```python
class MyHPOTrainer(GRPOHPOTrainer):
    async def _objective_function(self, params: Dict[str, Any]) -> float:
        """Custom objective with multiple metrics."""

        # Train with params
        config = self._create_training_config(params)
        trainer = MultiTurnGRPOTrainer(...)
        metrics = await trainer.train()

        # Custom objective combining multiple metrics
        reward = metrics.get("reward", 0.0)
        safety = metrics.get("safety_score", 0.0)

        # Weighted combination
        objective = 0.7 * reward + 0.3 * safety

        return objective
```

### Callbacks for Monitoring

```python
from training.hpo import HPOCallback

class MyCallback:
    def on_trial_start(self, trial_id: str, params: Dict[str, Any]):
        print(f"Starting trial {trial_id} with params: {params}")

    def on_trial_end(self, result: HPOResult):
        print(f"Trial {result.trial_id} completed: {result.best_metric:.4f}")

    def on_hpo_start(self, search_space: SearchSpace, n_trials: int):
        print(f"Starting HPO with {n_trials} trials")

    def on_hpo_end(self, summary: HPOSummary):
        print(f"HPO complete! Best: {summary.best_metric:.4f}")

trainer = GRPOHPOTrainer(
    ...,
    callbacks=[MyCallback()]
)
```

### Multi-Objective Optimization

```python
# Optimize for both reward and safety
hpo_config = HPOConfig(
    backend="optuna",
    search_space_name="customer_service",
    objective_metric="composite_score",  # Custom metric
    n_trials=100
)

async def compute_composite_score(params):
    # Train
    metrics = await train_with_params(params)

    # Combine metrics
    reward = metrics["reward"]
    safety = metrics["safety_score"]
    efficiency = metrics["avg_turns"]

    # Pareto-optimal combination
    score = (0.5 * reward +
             0.3 * safety -
             0.2 * (efficiency / 10))  # Penalize too many turns

    return score
```

### Warm Starting

```python
# Load previous results
hpo_config = HPOConfig(
    backend="optuna",
    search_space_name="grpo",
    n_trials=50,
    optuna_config={
        "storage": "sqlite:///optuna.db",
        "study_name": "my_study",
    }
)

trainer = GRPOHPOTrainer(...)

# Continue from previous run
summary = await trainer.optimize()
```

---

## Best Practices

### 1. Start Small

```python
# Quick exploration (10-20 trials)
quick_summary = await quick_hpo(
    ...,
    n_trials=20,
    search_space_name="conservative"
)

# Then refine (50-100 trials)
refined_config = HPOConfig(
    search_space_name="grpo",
    n_trials=100
)
```

### 2. Use Appropriate Search Spaces

- **Conservative**: When stability is critical (production systems)
- **Aggressive**: When exploring new domains
- **Domain-specific**: When you know your use case (customer service, tech support)
- **Full**: When you have compute budget and want comprehensive search

### 3. Choose the Right Number of Trials

| Budget | Trials | Use Case |
|--------|--------|----------|
| Low | 10-20 | Quick iteration, prototyping |
| Medium | 50-100 | Standard optimization |
| High | 200+ | Production systems, competition |

### 4. Leverage Pruning

```python
hpo_config = HPOConfig(
    optuna_config={
        "pruner": "median",  # Stop bad trials early
        "n_warmup_steps": 5,  # Give trials a chance
    }
)
```

### 5. Monitor Progress

```python
# Save intermediate results
hpo_config = HPOConfig(
    checkpoint_freq=10,  # Save every 10 trials
    save_plots=True,     # Generate plots
    log_to_wandb=True,   # Track in W&B
)

# Visualize
trainer.plot_results()
```

### 6. Distributed Optimization

For large-scale HPO:

```python
# Use Ray Tune for parallel trials
distributed_config = HPOConfig(
    backend="ray_tune",
    n_trials=200,
    n_parallel_trials=8,  # 8 GPUs
    gpu_per_trial=1.0
)
```

---

## Examples

### Example 1: Customer Service Agent

```python
from training.hpo import GRPOHPOTrainer, HPOConfig
from training.hpo.search_spaces import create_customer_service_search_space

# Setup
agent = MultiTurnAgent(...)
environment = ConversationEnvironment(customer_service_scenarios)
reward_function = CompositeReward([
    (HelpfulnessReward(), 0.4),
    (SafetyReward(), 0.3),
    (EngagementReward(), 0.3)
])

# HPO
hpo_config = HPOConfig(
    backend="optuna",
    search_space=create_customer_service_search_space(),
    n_trials=100,
    objective_metric="reward"
)

trainer = GRPOHPOTrainer(
    agent=agent,
    environment=environment,
    reward_function=reward_function,
    base_config=base_config,
    hpo_config=hpo_config
)

summary = await trainer.optimize()
final_agent = await trainer.train_with_best_params(full_episodes=500)
```

### Example 2: Multi-Domain Comparison

```python
domains = ["customer_service", "technical_support", "sales_assistant"]
results = {}

for domain in domains:
    search_space = get_search_space(domain)

    hpo_config = HPOConfig(
        search_space=search_space,
        n_trials=50
    )

    trainer = GRPOHPOTrainer(...)
    summary = await trainer.optimize()

    results[domain] = {
        "best_params": summary.best_params,
        "best_metric": summary.best_metric
    }

# Compare
for domain, result in results.items():
    print(f"{domain}: {result['best_metric']:.4f}")
```

---

## Troubleshooting

### Common Issues

**1. HPO is too slow**
- Reduce `n_trials`
- Use `search_space_name="conservative"` for smaller search space
- Enable pruning: `pruner="median"`
- Use fewer training episodes during HPO

**2. All trials fail**
- Check base_config is valid
- Verify search space ranges are reasonable
- Check logs for training errors

**3. No improvement over baseline**
- Widen search space: use `"aggressive"`
- Increase `n_trials`
- Check if objective_metric is correct

**4. Out of memory**
- Reduce `per_device_train_batch_size` in search space
- Enable `gradient_checkpointing`
- Use smaller model or LoRA

---

## Summary

StateSet Agents HPO provides:

✅ **Automated hyperparameter search** - No manual tuning
✅ **Multiple backends** - Optuna, Ray Tune, W&B
✅ **Pre-defined search spaces** - Domain-specific and algorithm-specific
✅ **Smart pruning** - Early stopping of bad trials
✅ **Rich visualization** - Optimization history and parameter importance
✅ **Seamless integration** - Works with existing GRPO trainers

**Start optimizing today:**

```python
from training.hpo import quick_hpo

summary = await quick_hpo(
    agent=agent,
    environment=environment,
    reward_function=reward_function,
    base_config=base_config,
    n_trials=50
)

print(f"Best params: {summary.best_params}")
```

---

**Next Steps:**
- See `examples/hpo_training_example.py` for complete examples
- Check `tests/unit/test_hpo.py` for usage patterns
- Read the API reference for detailed documentation
