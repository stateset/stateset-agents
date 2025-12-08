# stateset-rl-core

High-performance Rust implementations of reinforcement learning operations, with optional Python bindings via PyO3.

## Features

- **GAE (Generalized Advantage Estimation)** - Fast, parallel GAE computation
- **Advantage Computation** - Group-relative advantages for GRPO/PPO training
- **Reward Normalization** - Welford's algorithm for numerically stable online normalization
- **GSPO Support** - Sequence-level importance ratios and clipping for GSPO
- **PPO Surrogate** - Clipped surrogate objective computation
- **Parallel Processing** - Automatic parallelization via Rayon

## Installation

### As a Rust crate

```toml
[dependencies]
stateset-rl-core = "0.1"
```

### As a Python extension

```bash
cd rust_core
maturin develop --release
```

## Usage

### Rust

```rust
use stateset_rl_core::{compute_gae_internal, compute_advantages_for_group};

// Compute GAE
let rewards = vec![1.0, 0.0, 1.0, 0.0];
let values = vec![0.5, 0.5, 0.5, 0.5, 0.0]; // n+1 values for bootstrap
let advantages = compute_gae_internal(&rewards, &values, 0.99, 0.95);

// Compute group-relative advantages
let group_rewards = vec![1.0, 2.0, 3.0, 4.0];
let advantages = compute_advantages_for_group(&group_rewards, "mean", true);
```

### Python

```python
import numpy as np
import stateset_rl_core

# Compute GAE
rewards = np.array([1.0, 0.0, 1.0, 0.0])
values = np.array([0.5, 0.5, 0.5, 0.5, 0.0])
advantages = stateset_rl_core.compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95)

# Batch GAE (parallel)
all_rewards = [np.random.randn(100) for _ in range(32)]
all_values = [np.random.randn(101) for _ in range(32)]
all_advantages = stateset_rl_core.batch_compute_gae(all_rewards, all_values)

# Group-relative advantages for GRPO
rewards_2d = np.random.randn(16, 4)  # 16 groups, 4 samples each
advantages = stateset_rl_core.compute_group_advantages(rewards_2d, "mean", normalize=True)

# Reward normalization with running stats
rewards = np.array([1.0, 2.0, 3.0])
normalized, mean, var, count = stateset_rl_core.normalize_rewards(rewards)

# GSPO importance ratios
log_probs_new = np.array([-10.0, -12.0, -11.0])
log_probs_old = np.array([-10.5, -11.5, -11.0])
seq_lengths = np.array([50, 60, 55])
ratios = stateset_rl_core.compute_gspo_importance_ratios(log_probs_new, log_probs_old, seq_lengths)

# PPO surrogate objective
ratios = np.array([1.1, 0.9, 1.05])
advantages = np.array([1.0, -1.0, 0.5])
objectives = stateset_rl_core.compute_ppo_surrogate(ratios, advantages, clip_epsilon=0.2)
```

## API Reference

### GAE Functions

- `compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95)` - Single trajectory GAE
- `batch_compute_gae(all_rewards, all_values, gamma=0.99, gae_lambda=0.95)` - Parallel batch GAE

### Advantage Functions

- `compute_group_advantages(rewards_2d, baseline_type, normalize)` - GRPO-style group advantages
  - `baseline_type`: `"mean"`, `"median"`, or `"min"`

### Reward Functions

- `normalize_rewards(rewards, running_mean=0, running_var=1, count=0, epsilon=1e-8)` - Online normalization
- `clip_rewards(rewards, min_val, max_val)` - Reward clipping
- `compute_reward_statistics(rewards)` - Compute mean, std, min, max, median

### Policy Gradient Functions

- `compute_gspo_importance_ratios(log_probs_new, log_probs_old, sequence_lengths)` - GSPO ratios
- `apply_gspo_clipping(ratios, advantages, clip_left=3e-4, clip_right=4e-4)` - GSPO clipping
- `compute_ppo_surrogate(ratios, advantages, clip_epsilon=0.2)` - PPO clipped objective

## Performance

This crate is optimized for performance:

- **LTO enabled** - Link-time optimization for maximum speed
- **Single codegen unit** - Better optimization opportunities
- **Rayon parallelization** - Automatic multi-threading for batch operations
- **Zero-copy Python interop** - Minimal overhead when called from Python

Typical speedups over pure Python/NumPy: **10-100x** for batch operations.

## License

MIT
