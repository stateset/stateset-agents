# StateSet Agents: 5-Minute Quickstart

Get a working RL training pipeline in 5 minutes. No GPU required for the demo.

## Installation (30 seconds)

```bash
pip install stateset-agents
```

## Hello World (60 seconds)

```python
import asyncio
from stateset_agents import MultiTurnAgent, HelpfulnessReward
from stateset_agents.core.agent import AgentConfig

async def main():
    # Create agent (stub mode - no model download)
    agent = MultiTurnAgent(AgentConfig(
        model_name="stub://demo",
        use_stub_model=True,
    ))
    await agent.initialize()

    # Chat
    response = await agent.generate_response([
        {"role": "user", "content": "Hello!"}
    ])
    print(f"Agent: {response}")

asyncio.run(main())
```

## Train with Real Model (2 minutes)

```bash
pip install 'stateset-agents[training]'
```

```python
import asyncio
from stateset_agents import (
    MultiTurnAgent,
    ConversationEnvironment,
    CompositeReward,
    HelpfulnessReward,
    SafetyReward,
)
from stateset_agents.training import GSPOTrainer, GSPOConfig
from stateset_agents.core.agent import AgentConfig

async def train():
    # 1. Setup
    config = GSPOConfig(
        model_name="gpt2",           # Or: "Qwen/Qwen2.5-0.5B"
        num_generations=4,            # Responses per prompt
        num_outer_iterations=10,      # Training cycles
        use_lora=True,                # Memory efficient
    )

    # 2. Create reward
    reward = CompositeReward([
        HelpfulnessReward(weight=0.7),
        SafetyReward(weight=0.3),
    ])

    # 3. Create trainer
    trainer = GSPOTrainer(
        config=config,
        reward_fn=reward,
        prompts=["How do I reset my password?", "What's your return policy?"]
    )

    # 4. Train!
    await trainer.train()

asyncio.run(train())
```

## Algorithm Comparison

| Algorithm | Best For | Key Feature |
|-----------|----------|-------------|
| **GSPO** | General use | Stable, prevents collapse |
| **VAPO** | Math/reasoning | SOTA (60.4 AIME) |
| **DAPO** | Long CoT | Dynamic sampling |
| **GRPO** | Fast iteration | Simple baseline |

## Quick Config Recipes

### Customer Service Agent
```python
from stateset_agents.training import get_config_for_task
config = get_config_for_task("customer_service")
```

### Memory-Constrained (8GB VRAM)
```python
config = GSPOConfig(
    model_name="gpt2",
    use_lora=True,
    use_4bit=True,
    gradient_checkpointing=True,
    num_generations=2,
)
```

### Maximum Performance
```python
config = GSPOConfig(
    model_name="Qwen/Qwen2.5-7B",
    use_vllm=True,              # 5-20x faster generation
    bf16=True,
    num_generations=16,
)
```

## CLI Training

```bash
# Quick start
stateset-agents train --model gpt2 --algorithm gspo

# With config file
stateset-agents train --config my_config.yaml

# Monitor with W&B
stateset-agents train --model gpt2 --wandb-project my-project
```

## Next Steps

- **Examples**: `examples/` directory has 20+ complete examples
- **Guides**: `docs/GSPO_GUIDE.md` for algorithm deep-dives
- **API Docs**: `docs/API_REFERENCE.md` for full API
- **Benchmarks**: `docs/BENCHMARKS.md` for performance data

## Common Issues

**Out of memory?**
```python
config = GSPOConfig(use_4bit=True, gradient_checkpointing=True)
```

**Slow generation?**
```python
config = GSPOConfig(use_vllm=True)  # Requires: pip install vllm
```

**No GPU?**
```python
config = AgentConfig(use_stub_model=True)  # For development
```

## Get Help

- GitHub Issues: https://github.com/anthropics/stateset-agents/issues
- Documentation: https://stateset.io/docs
