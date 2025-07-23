# GRPO Agent Framework

A comprehensive framework for training multi-turn AI agents using Group Relative Policy Optimization (GRPO).

## 🎉 What's New in v0.3.0

Based on learnings from real-world implementations, we've significantly enhanced the framework:

- **🎭 Multi-Objective Rewards**: Complex reward architectures with empathy, action-orientation, and professionalism scoring
- **🧠 Neural Reward Models**: Self-improving reward functions that learn from trajectory data
- **🚀 Distributed Training**: Advanced multi-GPU training with proper rank handling and memory optimization
- **☁️ Cloud Deployment**: Automated RunPod integration with dynamic scaling and cost optimization
- **📊 Enhanced W&B Integration**: Comprehensive metrics, visualizations, and experiment tracking
- **📈 Production-Ready Features**: Fault tolerance, monitoring, and deployment automation

## Overview

GRPO Agent Framework provides a flexible and robust infrastructure for training conversational AI agents that can handle multi-turn interactions. It implements state-of-the-art reinforcement learning techniques with a focus on stability, performance, and ease of use.

### Key Features

- 🚀 **Multi-turn Conversation Support**: Native support for training agents on extended dialogues
- 🎯 **Advanced Reward Systems**: RULER LLM judges, multi-objective rewards, and neural reward models
- 🔧 **Production-Ready Training**: Distributed multi-GPU training with memory optimization
- 📊 **Comprehensive Monitoring**: Real-time metrics, W&B integration, and system diagnostics
- 🔌 **Extensible Architecture**: Easy to add custom agents, environments, and rewards
- ☁️ **Cloud Deployment**: Automated RunPod deployment with dynamic scaling
- 🧠 **Self-Improving Models**: Neural reward functions that learn from trajectory data
- 📈 **Battle-Tested**: Enhanced based on production implementations and the "Bitter Lesson"

## Installation

```bash
pip install grpo-agent-framework
```

For development:
```bash
git clone https://github.com/stateset/grpo-agent-framework
cd grpo-agent_framework
pip install -e ".[dev]"
```

## Quick Start

### Training a Simple Agent

```python
from grpo_agent_framework import Agent, Environment, train

# Define your agent
agent = Agent.from_pretrained("gpt2")

# Create environment
env = Environment.from_task("conversation")

# Train
trainer = train(
    agent=agent,
    environment=env,
    num_episodes=1000,
    profile="balanced"  # or "conservative", "aggressive"
)
```

### Enhanced Customer Service Agent (v0.3.0)

```python
from grpo_agent_framework import (
    MultiTurnAgent, 
    ConversationEnvironment,
    create_domain_reward,
    load_and_prepare_data
)
from grpo_agent_framework.training import DistributedGRPOTrainer, TrainingConfig, DistributedConfig
from grpo_agent_framework.rewards import RulerRewardFunction, MultiObjectiveRewardFunction
from grpo_agent_framework.utils import WandBIntegration, WandBConfig

# Load and prepare data with automatic splitting
train_data, eval_data = load_and_prepare_data(
    "customer_service_data.jsonl",
    max_examples=5000,
    validation_split=0.1
)

# Create sophisticated reward system
ruler_reward = RulerRewardFunction(
    model="openai/gpt-4",
    rubric_type="customer_service",
    weight=0.6
)

multi_objective_reward = MultiObjectiveRewardFunction(
    components=[
        EmpathyRewardComponent(weight=0.3),
        ProfessionalismRewardComponent(weight=0.3),
        ActionOrientedRewardComponent(weight=0.4)
    ],
    weight=0.4
)

# Configure distributed training
training_config = TrainingConfig(
    num_episodes=1000,
    num_generations=16,
    use_lora=True,
    bf16=True,
    run_post_eval=True
)

distributed_config = DistributedConfig(
    world_size=4,
    mixed_precision=True,
    gradient_accumulation_steps=2
)

# Setup W&B integration
wandb_config = WandBConfig(
    project="customer-service-grpo",
    log_trajectories=True,
    create_reward_plots=True
)

wandb_integration = WandBIntegration(wandb_config, training_config)

# Train with distributed GRPO
trainer = DistributedGRPOTrainer(
    agent=agent,
    environment=env,
    reward_function=ruler_reward,
    training_config=training_config,
    distributed_config=distributed_config
)

trained_agent = await trainer.train()

# Comprehensive evaluation with visualizations
eval_results = await trainer.run_post_training_evaluation(eval_data)
```

## Architecture

### Core Components

1. **Agents**: Base classes for implementing RL agents
   - `Agent`: Single-turn agent base class
   - `MultiTurnAgent`: Multi-turn conversation agent
   - `ToolAgent`: Agent with tool-use capabilities

2. **Environments**: Interaction environments for agents
   - `Environment`: Base environment class
   - `ConversationEnvironment`: Multi-turn conversation environment
   - `TaskEnvironment`: Task-oriented environment

3. **Rewards**: Flexible reward modeling system
   - `RewardFunction`: Base reward class
   - `CompositeReward`: Combine multiple rewards
   - **NEW**: Domain-specific rewards (CustomerService, TechnicalSupport, Sales)
   - **NEW**: Similarity-aware rewards for supervised fine-tuning

4. **Training**: GRPO training infrastructure
   - `GRPOTrainer`: Core training logic
   - `AutoTrainer`: Automatic hyperparameter optimization
   - `DiagnosticsMonitor`: Training health monitoring
   - **NEW**: Enhanced GRPO loss with KL penalties
   - **NEW**: Post-training evaluation pipeline

5. **Data Processing** (NEW in v0.2.0)
   - `DataLoader`: Load conversations from various formats
   - `DataProcessor`: Prepare data for GRPO training
   - Automatic train/eval splitting with stratification

## Advanced Features

### Automatic Configuration Optimization

```python
from grpo_agent_framework.training import ConfigOptimizer

optimizer = ConfigOptimizer()
best_config = optimizer.optimize(agent, environment, num_trials=20)
```

### Domain-Specific Training

```python
# Customer Service
from grpo_agent_framework import create_domain_reward

reward = create_domain_reward("customer_service")

# Technical Support
reward = create_domain_reward("technical_support")

# Sales Assistant
reward = create_domain_reward("sales", expected_responses=sales_data)
```

### Mixed Precision Training

```python
config = TrainingConfig(
    bf16=True,  # Use bfloat16 for stability
    gradient_checkpointing=True,  # Save memory
    use_lora=True,  # Parameter-efficient training
)
```

## Documentation

- [Usage Guide](USAGE_GUIDE.md) - Comprehensive usage instructions
- [CLI Reference](CLI_REFERENCE.md) - Command-line interface documentation
- [API Reference](docs/api/) - Complete API documentation
- [Examples](examples/) - Ready-to-run examples

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use GRPO Agent Framework in your research, please cite:

```bibtex
@software{grpo_agent_framework,
  title = {GRPO Agent Framework: Production-Ready Multi-Turn Agent Training},
  author = {GRPO Framework Team},
  year = {2024},
  version = {0.2.0},
  url = {https://github.com/yourusername/grpo-agent-framework}
}
```