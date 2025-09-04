# StateSet RL Agent Framework

A comprehensive framework for training multi-turn AI agents using reinforcement learning (RL).

## 🎉 What's New in v0.3.0

Major framework improvements for production-ready AI agent development:

### 🛡️ **Enhanced Error Handling & Resilience**
- **Comprehensive Exception Hierarchy**: Specialized exceptions for training, model, data, network, and resource errors
- **Retry Mechanisms**: Configurable async retry with exponential backoff and jitter
- **Circuit Breaker Pattern**: Automatic failure detection and recovery for external services
- **Rich Error Context**: Detailed error tracking with stack traces, categories, and recovery suggestions

### ⚡ **Performance Optimization**
- **Memory Management**: Real-time memory monitoring with automatic cleanup and optimization
- **Dynamic Batch Sizing**: Automatic batch size optimization based on resource availability
- **Model Compilation**: PyTorch 2.0 compilation support for faster inference
- **Mixed Precision Training**: Automated FP16/BF16 optimization with stability safeguards

### 🔍 **Type Safety & Validation**
- **Runtime Type Checking**: Comprehensive type validation for all framework components
- **Configuration Validation**: Type-safe configuration with detailed error reporting
- **Type-Safe Serialization**: Reliable serialization/deserialization with type preservation
- **Protocol Interfaces**: Clear contracts for extensible components

### 🚀 **Advanced Async Resource Management**
- **Connection Pooling**: High-performance async resource pools with health checking
- **Task Management**: Sophisticated async task scheduling with resource limits
- **Resource Lifecycle**: Automatic resource cleanup and scaling
- **Performance Metrics**: Real-time monitoring of resource utilization

### 📊 **Production Monitoring**
- **Real-time Metrics**: Comprehensive performance tracking and reporting
- **Health Checks**: Automated system health monitoring and alerting
- **Resource Optimization**: Dynamic optimization recommendations
- **Diagnostic Tools**: Advanced debugging and profiling capabilities

## Overview

GRPO Agent Framework provides a flexible and robust infrastructure for training conversational AI agents that can handle multi-turn interactions. It implements state-of-the-art reinforcement learning techniques with a focus on stability, performance, and ease of use.

### Key Features

- 🚀 **Multi-turn Conversation Support**: Native support for training agents on extended dialogues
- 🎯 **Advanced Reward Systems**: LLM judges, multi-objective rewards, and neural reward models
- 🔧 **Production-Ready Training**: Distributed multi-GPU training with memory optimization
- 🏆 **TRL GRPO Integration**: Fine-tune large models like `openai/gpt-oss-120b` using HuggingFace TRL's GRPO trainer with LoRA adapters
- 📊 **Comprehensive Monitoring**: Real-time metrics, W&B integration, and system diagnostics
- 🔌 **Extensible Architecture**: Easy to add custom agents, environments, and rewards
- ☁️ **Cloud Deployment**: Automated RunPod deployment with dynamic scaling
- 🧠 **Self-Improving Models**: Neural reward functions that learn from trajectory data
- 📈 **Battle-Tested**: Enhanced based on production implementations and the "Bitter Lesson"

## Installation

```bash
pip install stateset-agents
```

For development:
```bash
git clone https://github.com/stateset/stateset-agents
cd stateset-agents
pip install -e ".[dev]"
```

### CLI

Basic CLI is available via the `stateset-agents` entrypoint:

```bash
stateset-agents version
stateset-agents train --dry-run  # guidance and environment checks
stateset-agents serve            # starts the FastAPI service (requires [api] extras)
```
Install API extras to enable serving:
```bash
pip install "stateset-agents[api]"
```

## Quick Start

### Training a Simple Agent

```python
from stateset_agents import Agent, Environment, train

# Define your agent
agent = Agent.from_pretrained("openai/gpt-oss-120b")

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

### Enhanced Production Agent (v0.3.0)

```python
from stateset_agents import (
    # Core components
    MultiTurnAgent, ConversationEnvironment,
    HelpfulnessReward, SafetyReward, CompositeReward,
    
    # Enhanced features
    PerformanceOptimizer, OptimizationLevel,
    ErrorHandler, RetryConfig, NetworkException,
    TypeValidator, create_typed_config,
    AsyncTaskManager, managed_async_resources,
    
    # Type-safe configuration
    ModelConfig, DeviceType
)

# Create type-safe, production-ready configuration
model_config = create_typed_config(
    ModelConfig,
    model_name="gpt2",
    device=DeviceType.AUTO,
    torch_dtype="bfloat16",
    max_length=512,
    temperature=0.7,
    top_p=0.9
)

# Initialize enhanced error handling
error_handler = ErrorHandler()

# Setup performance optimization
optimizer = PerformanceOptimizer(OptimizationLevel.BALANCED)

# Create resilient agent with retries
@retry_async(RetryConfig(max_attempts=3, base_delay=1.0))
async def create_agent():
    agent = MultiTurnAgent(model_config)
    await agent.initialize()
    return agent

# Use managed async resources for automatic cleanup
async with managed_async_resources():
    try:
        # Create agent with error handling
        agent = await create_agent()
        
        # Create environment and rewards
        env = ConversationEnvironment(scenarios=scenarios)
        reward_fn = CompositeReward([
            HelpfulnessReward(weight=0.6),
            SafetyReward(weight=0.4)
        ])
        
        # Performance-optimized conversation
        with optimizer.memory_monitor.memory_context("conversation"):
            response = await agent.generate_response(conversation_turns)
            
            # Validate response quality
            reward_result = await reward_fn.compute_turn_reward(turn)
            
        # Get performance insights
        performance_report = optimizer.get_performance_report()
        print(f"Optimization recommendations: {performance_report['recommendations']}")
        
    except Exception as e:
        # Comprehensive error handling
        error_context = error_handler.handle_error(e, "agent", "conversation")
        print(f"Error handled: {error_context.error_id}")
        
        # Get error analytics
        error_summary = error_handler.get_error_summary()
        print(f"Error patterns: {error_summary}")
```

### TRL GRPO Training

Fine-tune large models like `openai/gpt-oss-120b` using HuggingFace TRL's GRPO trainer:

```python
from stateset_agents.training import train_customer_service_with_trl

# Quick training with TRL GRPO
agent = await train_customer_service_with_trl(
    model_name="openai/gpt-oss-120b",
    num_episodes=1000,
    use_lora=True,  # Efficient training with LoRA adapters
    lora_r=16,
    output_dir="./outputs/my_trl_agent"
)

# Or use the production script
# ./scripts/train_trl_grpo.sh
```

See the [TRL GRPO Training Guide](TRL_GRPO_TRAINING_GUIDE.md) for detailed instructions.

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

This project is licensed under the Business Source License 1.1 (BUSL-1.1).

- Additional Use Grant: non-production use (development, testing, staging, evaluation, research, personal use) is permitted prior to the Change Date.
- Change Date: 2029-09-03, after which the project will be available under the Apache License 2.0.
- See the full terms in the [LICENSE](LICENSE) file.

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
