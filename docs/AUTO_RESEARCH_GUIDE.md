# Auto-Research Guide

Autonomous hyperparameter optimization for stateset-agents. Run experiments overnight — the loop proposes configurations, trains, evaluates, keeps improvements, and repeats.

## Installation

```bash
pip install 'stateset-agents[auto-research]'

# For LLM-driven proposals (Claude API):
pip install 'stateset-agents[auto-research-llm]'
```

## Quick Start

### CLI

```bash
# Test with stub model (no GPU needed)
stateset-agents auto-research --stub --max-experiments 5

# Real training with GSPO
stateset-agents auto-research --max-experiments 50 --time-budget 300

# Smart proposer (learns which params matter)
stateset-agents auto-research --proposer smart --improvement-patience 10

# From a config file
stateset-agents auto-research --config my_config.yaml
```

### Python API

```python
import asyncio
from stateset_agents import MultiTurnAgent, AgentConfig
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import CompositeReward, HelpfulnessReward, SafetyReward
from stateset_agents.training.auto_research import AutoResearchConfig, run_auto_research

async def main():
    agent = MultiTurnAgent(
        AgentConfig(model_name="stub://quickstart", use_stub_model=True)
    )
    await agent.initialize()

    environment = ConversationEnvironment(scenarios=[...], max_turns=8)
    eval_scenarios = [...]  # Held-out scenarios (never used for training)
    reward_fn = CompositeReward([HelpfulnessReward(0.6), SafetyReward(0.4)])

    config = AutoResearchConfig(
        time_budget=300,
        max_experiments=50,
        proposer="smart",
        search_space_name="auto_research",
        improvement_patience=10,
    )

    tracker = await run_auto_research(
        agent=agent,
        environment=environment,
        eval_scenarios=eval_scenarios,
        reward_fn=reward_fn,
        config=config,
        baseline_params={"learning_rate": 5e-6, "lora_r": 8},
    )

asyncio.run(main())
```

## Proposer Strategies

| Strategy | Description | Best for |
|----------|-------------|----------|
| `perturbation` | Small random changes to current best | Default, fast iteration |
| `smart` | Learns which params matter, focuses there | Long runs (50+ experiments) |
| `adaptive` | Starts broad, narrows down over time | Balancing exploration/exploitation |
| `random` | Pure random sampling | Broad exploration |
| `grid` | Systematic grid search | Small search spaces |
| `bayesian` | Optuna TPE (requires `optuna`) | When you have compute budget |
| `llm` | Claude/OpenAI proposes with reasoning | Creative exploration |

## Search Spaces

| Name | Dimensions | Description |
|------|-----------|-------------|
| `auto_research` | 14 | Comprehensive: LR, LoRA, GSPO, RL, generation |
| `multi_algorithm` | 7 + algo | Includes algorithm choice (GSPO/GRPO/DAPO/VAPO) |
| `quick` | 4 | Just LR, num_generations, temperature, lora_r |
| `reward` | 6 | Reward weight exploration |
| `model` | 6 | LoRA and generation architecture |

## Configuration

### From YAML

```yaml
# my_config.yaml
auto_research:
  time_budget: 300
  proposer: smart
  search_space_name: auto_research
  max_experiments: 100
  improvement_patience: 10
  algorithm: gspo
  eval_episodes: 10
  output_dir: ./results
```

```python
config = AutoResearchConfig.from_file("my_config.yaml")
```

### Key options

| Option | Default | Description |
|--------|---------|-------------|
| `time_budget` | 300 | Seconds per experiment (training + eval) |
| `max_experiments` | 0 | Max experiments (0 = unlimited) |
| `max_wall_clock` | 0 | Total time budget in seconds (0 = unlimited) |
| `improvement_patience` | 0 | Stop after N consecutive non-improvements (0 = disabled) |
| `proposer` | perturbation | Strategy name |
| `search_space_name` | grpo | Search space name |
| `trainer_algorithm` | gspo | Training algorithm (or "auto") |
| `calibrate_rewards` | false | Normalize reward scores to prevent drift |
| `save_checkpoints` | true | Save model weights for kept experiments |
| `log_to_wandb` | false | Log to Weights & Biases |

## Post-Run Analysis

### Load and inspect results

```python
from stateset_agents.training.auto_research import ExperimentTracker

tracker = ExperimentTracker.load("./results")
tracker.print_summary()  # Includes ASCII convergence chart + parameter importance

analysis = tracker.get_analysis()
# analysis["best_value"]              — best objective achieved
# analysis["parameter_importance"]    — which params matter (numeric + categorical)
# analysis["convergence_curve"]       — running best over time
# analysis["experiments"]             — list of all experiments (for Jupyter DataFrames)
```

### Compare multiple runs

```python
from stateset_agents.training.auto_research import compare_runs

print(compare_runs("./run_perturbation", "./run_smart", "./run_bayesian"))
```

Output:
```
RUN COMPARISON
Run                   Best    Exps  Kept   Rate     Time
─────────────────────────────────────────────────────────
run_perturbation    0.650000    50    12    24%   42.1m
run_smart           0.720000    50    18    36%   41.5m
run_bayesian        0.695000    50    15    30%   43.2m

Winner: run_smart (best eval_reward=0.720000)
```

### Import legacy results

```python
# Import from manual autoresearch results.tsv format
tracker = ExperimentTracker.from_legacy_tsv("results.tsv")
tracker.print_summary()
```

## How It Works

The loop follows the same pattern as [autoresearch](https://github.com/karpathy/autoresearch):

1. **Evaluate baseline** — run the untrained agent on held-out scenarios
2. **Propose** — the proposer suggests a new hyperparameter configuration
3. **Train** — train the agent with a time budget (`asyncio.wait_for`)
4. **Evaluate** — run on the same held-out scenarios
5. **Keep or revert** — if objective improved, save checkpoint; otherwise restore previous best
6. **Repeat** — loop until `max_experiments`, `max_wall_clock`, or `improvement_patience` triggers

The loop:
- **Resumes automatically** if restarted (reads `experiments.jsonl`)
- **Cleans up GPU memory** after crashes/timeouts
- **Skips training for stub agents** (useful for testing the loop itself)
- **Aborts bad experiments early** via `EarlyAbortCallback` (detects NaN loss, loss explosion, plateau)

## Examples

See `examples/auto_research_quickstart.py` for a complete, runnable example.
