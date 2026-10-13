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
from stateset_agents.training.trainer import SingleTurnGRPOTrainer
from stateset_agents.training.config import TrainingConfig

async def train_single_turn():
    # Create a basic agent
    config = AgentConfig(
        model_name="stub://quickstart",
        use_stub_model=True,
        max_new_tokens=50,
    )
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
    await trainer.save_checkpoint(checkpoint_name="single_turn_model")

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
- `async save_checkpoint(is_best=False, checkpoint_name=None)` - Save model and training state under `config.output_dir`
- `load_checkpoint(path, trusted=False)` - Restore model and training state
- `add_callback(callback)` - Add training callback

## Configuration

### TrainingConfig Options

```python
from stateset_agents.training.config import TrainingConfig

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

The shared training callback dispatcher supports synchronous and asynchronous
callbacks, including synchronous methods that return a Future or another
awaitable. It waits for returned awaitables before dispatching the next callback.
For legacy argument variants, it binds the callback signature before execution;
a `TypeError` inside a callback does not trigger another invocation. Callables
without an inspectable signature receive the canonical argument form once.
Existing best-effort error handling remains in place; cancellation propagates
and stops dispatch.
Callbacks that guard required state can set `fail_on_error = True` to propagate
their errors and stop dispatch instead of treating them as optional diagnostics.
The API progress callback uses this mode to reject invalid progress observations.

Native single-turn and multi-turn GRPO require a finite differentiable scalar
loss before backward and finite gradients before an ordinary optimizer step.
Clipping limits must be finite and nonnegative; a non-finite total gradient norm
also aborts the update even when individual gradient values are finite. Rejected
work is cleared before the error propagates. AMP overflow instead consumes the
accumulation window and reduces the scaler, without advancing the learning-rate
schedule, optimizer-step counter, or rollout synchronization. A subsequent valid
batch can continue. Empty gradient windows also consume no optimizer step.

On successful completion, a partial accumulation window averages its actual
batches: if only `K` of the configured `N` batches remain, the accumulated
gradients are multiplied by `N / K` after AMP unscaling and before clipping.
This preserves the mean-batch update instead of shrinking it by `K / N`.
Checkpoint resume follows the same rule even when no episodes remain; the
restored AMP scale is initialized without advancing its growth tracker. The
completed window is cleared, so resuming its saved result cannot flush it twice.

With `num_gradient_updates > 1`, token rollouts use full inner optimizer updates.
If a preceding sequence-path batch left accumulated gradients, both trainers
first flush that window separately using its actual mean, then reset the
accumulation counter. The token batch's old-policy log-probabilities are frozen
before that flush; inner losses are computed at the updated weights against
that same frozen policy. Switching back to sequence batches starts a new
accumulation window. Continual-learning EWC penalties apply to every inner loss.

Learning-rate schedules are sized in optimizer updates. For one update per
batch, the budget is `ceil(planned_batches / gradient_accumulation_steps)`.
With multiple inner updates, it is `planned_batches * num_gradient_updates`;
inner steps bypass accumulation, so dividing this budget by accumulation would
exhaust the schedule too early. Single-turn training plans up to
`num_episodes * max_steps_per_episode` batches; multi-turn training plans one
batch per episode. `TrainingConfig.get_total_steps()` uses the latter convention.
Warmup uses the same budget. If token metadata is missing, episodes end early,
or updates are skipped, fewer steps may commit and the schedule may not reach
its endpoint. The budget is fixed for the run, including checkpoint continuation
with the same configuration; skipped updates do not advance the schedule.

`global_step` advances when an optimizer step commits, not merely when an episode
runs. Stub or simulation runs with disconnected losses can therefore complete
with zero optimizer steps; use episode callbacks to track rollout progress.
Multi-turn `optimizer_step` reports whether at least one update committed, and
`optimizer_updates` counts committed updates for the batch, including inner
updates and a preceding accumulation flush. Inner-update metrics also report
`accumulation_flush_updates` (`0` or `1`) separately; `inner_updates` counts only
the attempted inner updates. These counts do not prove parameter changes or
improved model quality.
An optimizer step that committed before a later scheduler or synchronization
failure remains counted and is not rolled back.

Single-turn loop errors from loss computation, environment calls, optimizer work,
or required callbacks propagate to the caller. Cancellation and errors clear
pending gradients and reset the accumulation counter; they do not flush a partial
window or invoke remaining success-only finalization. Already committed optimizer
updates and environment/replay side effects are not rolled back. Older zero-argument
`reset()` and one-argument `step(response)` methods remain supported: their
signatures are checked before invocation, so an internal `TypeError` is never
used to retry the call. A callable without an inspectable signature receives the
canonical arguments once.

A tracker exposing `finish_run(summary)` is closed on every training exit, with
`errored` and `cancelled` flags. Closing runs off the event loop and remains owned
through repeated cancellation; an existing training error takes precedence over
ordinary tracking cleanup errors. A hung tracking backend can delay cancellation.
Trackers exposing only `log(metrics)` remain supported without automatic closing;
the packaged W&B logger uses `log_metrics(metrics, step=...)`.

### W&B Integration

```python
from stateset_agents.utils.wandb_integration import WandBLogger

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

Native GSPO and GSPO-token report `nonzero_advantage_fraction`,
`zero_advantage_group_fraction`, and `advantage_group_count` in step metrics and
training history. These are measured from the computed advantages: a batch can
have positive pooled `reward_std` yet zero advantages when every response to
each individual prompt earns the same reward. Singleton groups also have zero
group-relative advantages. The fractions count responses and groups respectively.

Pass `ZeroSignalGuard(max_zero_steps=5)` from `stateset_agents.training.callbacks`
in `train_with_gspo(..., callbacks=[...])` to stop after consecutive zero-advantage
steps. Native GSPO also warns after five such steps without a callback. Missing
or invalid signal metrics break the streak; legacy metrics require an explicitly
zero reward mean and standard deviation. An abort remains latched. Inspect reward
context, response diversity, and task difficulty before increasing the run length.
These diagnostics cover the policy advantage term before clipping; they do not
prove parameter updates, held-out improvement, or absence of KL and optimizer-state
effects. GSPO-token emits the metrics but its entrypoint does not accept callbacks.

Both native GSPO variants reject empty rollout groups, non-finite rollout log
probabilities, invalid reward vectors or float32 reward statistics, and non-finite
computed advantages. Before updating, they require a finite scalar loss and a
finite gradient norm. A failed backward or gradient check clears partial gradients
and raises instead of advancing the optimizer, scheduler, or successful-step
history. These checks do not roll back rollout costs, random-number state, or
arbitrary side effects in custom model/reward code; nor do they make an optimizer
failure after its update begins transactional.

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
