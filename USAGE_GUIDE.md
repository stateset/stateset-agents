# GRPO Agent Framework - Usage Guide

This guide provides comprehensive instructions for using the GRPO Agent Framework to train multi-turn conversational AI agents.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Creating Agents](#creating-agents)
4. [Designing Environments](#designing-environments)
5. [Reward Functions](#reward-functions)
6. [Training Configuration](#training-configuration)
7. [Advanced Features](#advanced-features)
8. [Best Practices](#best-practices)
9. [Examples](#examples)

## Quick Start

### Installation

```bash
pip install grpo-agent-framework
```

### Basic Usage

```python
import asyncio
from grpo_agent_framework import (
    MultiTurnAgent, ConversationEnvironment, 
    HelpfulnessReward, train
)
from grpo_agent_framework.core.agent import AgentConfig

async def quick_example():
    # 1. Create an agent
    config = AgentConfig(
        model_name="openai/gpt-oss-120b",
        system_prompt="You are a helpful AI assistant.",
        temperature=0.8
    )
    agent = MultiTurnAgent(config)
    await agent.initialize()
    
    # 2. Create environment with scenarios
    scenarios = [{
        "id": "help_conversation",
        "user_responses": [
            "Hi! Can you help me?",
            "I need advice on learning Python.",
            "Thank you for the suggestions!"
        ]
    }]
    environment = ConversationEnvironment(scenarios=scenarios)
    
    # 3. Train the agent
    trained_agent = await train(
        agent=agent,
        environment=environment,
        num_episodes=100,
        profile="balanced"
    )
    
    # 4. Use the trained agent
    response = await trained_agent.generate_response([
        {"role": "user", "content": "Hello! How are you?"}
    ])
    print(f"Agent: {response}")

# Run the example
asyncio.run(quick_example())
```

## Core Concepts

### Agents
Agents are AI models that can engage in conversations. The framework provides:

- **Agent**: Base class for all agents
- **MultiTurnAgent**: Specialized for multi-turn conversations
- **ToolAgent**: Can use external tools and functions

### Environments
Environments define the interaction context and scenarios:

- **Environment**: Base environment class
- **ConversationEnvironment**: For open-ended conversations
- **TaskEnvironment**: For goal-oriented interactions

### Trajectories
Trajectories represent conversation episodes:

- **ConversationTurn**: Single message in a conversation
- **MultiTurnTrajectory**: Complete conversation episode
- **TrajectoryGroup**: Multiple trajectories for the same scenario

### Rewards
Reward functions evaluate agent performance:

- **RewardFunction**: Base class for all rewards
- **CompositeReward**: Combines multiple reward functions
- Built-in rewards: helpfulness, safety, correctness, etc.

## Creating Agents

### Basic Agent Configuration

```python
from grpo_agent_framework.core.agent import AgentConfig, MultiTurnAgent

config = AgentConfig(
    model_name="openai/gpt-oss-120b",
    system_prompt="You are a helpful assistant.",
    temperature=0.8,
    max_new_tokens=256,
    memory_window=10  # Remember last 10 turns
)

agent = MultiTurnAgent(config)
await agent.initialize()
```

### Custom Agent Classes

```python
class CustomAgent(MultiTurnAgent):
    def __init__(self, model_name):
        config = AgentConfig(
            model_name=model_name,
            system_prompt="Custom system prompt...",
            temperature=0.7
        )
        super().__init__(config)
        self.custom_state = {}
    
    async def process_turn(self, history, user_input, context=None):
        # Custom logic before generating response
        self.custom_state["turn_count"] = self.custom_state.get("turn_count", 0) + 1
        
        # Generate response using parent method
        turn = await super().process_turn(history, user_input, context)
        
        # Add custom metadata
        turn.metadata["custom_state"] = self.custom_state.copy()
        
        return turn
```

### Tool-Using Agents

```python
from grpo_agent_framework.core.agent import ToolAgent

# Define tools
def calculator_tool(expression):
    try:
        return eval(expression)  # Note: unsafe for production
    except:
        return "Error in calculation"

tools = [{
    "name": "calculator",
    "description": "Perform mathematical calculations",
    "function": calculator_tool
}]

# Create tool agent
tool_agent = ToolAgent(config, tools=tools)
await tool_agent.initialize()
```

## Designing Environments

### Conversation Environment

```python
from grpo_agent_framework.core.environment import ConversationEnvironment

scenarios = [
    {
        "id": "customer_support",
        "topic": "technical_help",
        "context": "User has a technical problem",
        "user_responses": [
            "My device isn't working properly.",
            "I've tried restarting it.",
            "That didn't help either.",
            "Okay, I'll try that solution."
        ]
    }
]

env = ConversationEnvironment(
    scenarios=scenarios,
    max_turns=12,
    persona="You are a user seeking technical support."
)
```

### Task Environment

```python
from grpo_agent_framework.core.environment import TaskEnvironment

tasks = [{
    "description": "Help user plan a vacation",
    "goal": "Create a complete vacation plan",
    "required_actions": [
        {"name": "destination", "keywords": ["where", "destination", "place"]},
        {"name": "budget", "keywords": ["cost", "budget", "price"]},
        {"name": "activities", "keywords": ["do", "activities", "attractions"]},
        {"name": "schedule", "keywords": ["when", "dates", "time"]}
    ]
}]

def success_criteria(turns, context):
    completed = len(context.get("completed_actions", []))
    required = len(context.get("required_actions", []))
    return completed >= required

env = TaskEnvironment(
    tasks=tasks,
    success_criteria=success_criteria,
    max_turns=15
)
```

### Custom Environment

```python
class CustomEnvironment(Environment):
    async def reset(self, scenario=None):
        # Initialize episode state
        state = EnvironmentState(
            episode_id=str(uuid.uuid4()),
            turn_count=0,
            status=EpisodeStatus.ONGOING,
            context={"custom_data": scenario}
        )
        return state
    
    async def step(self, state, action):
        # Process agent action
        new_state = state.copy()
        new_state.turn_count += 1
        
        # Generate environment response
        response = ConversationTurn(
            role="user",
            content="Custom response based on action"
        )
        
        # Calculate reward
        reward = self.calculate_custom_reward(action)
        
        # Check if done
        done = new_state.turn_count >= self.max_turns
        
        return new_state, response, reward, done
    
    async def get_initial_prompt(self, scenario=None):
        return "Custom initial prompt"
```

## Reward Functions

### Built-in Rewards

```python
from grpo_agent_framework.core.reward import (
    HelpfulnessReward, SafetyReward, CorrectnessReward,
    ConcisenessReward, EngagementReward, CompositeReward
)

# Individual rewards
helpfulness = HelpfulnessReward(weight=1.0)
safety = SafetyReward(weight=1.0)

# Composite reward
composite = CompositeReward([
    HelpfulnessReward(weight=0.4),
    SafetyReward(weight=0.3),
    EngagementReward(weight=0.3)
])
```

### Custom Reward Functions

```python
from grpo_agent_framework.core.reward import RewardFunction, RewardResult

class CustomReward(RewardFunction):
    async def compute_reward(self, turns, context=None):
        assistant_turns = [t for t in turns if t.role == "assistant"]
        
        # Custom scoring logic
        total_score = 0.0
        breakdown = {}
        
        for i, turn in enumerate(assistant_turns):
            score = self.evaluate_turn(turn.content)
            total_score += score
            breakdown[f"turn_{i}"] = score
        
        avg_score = total_score / max(1, len(assistant_turns))
        
        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={"num_turns": len(assistant_turns)}
        )
    
    def evaluate_turn(self, content):
        # Custom evaluation logic
        return len(content) / 100.0  # Simple example
```

### Reward Function Decorator

```python
from grpo_agent_framework.core.reward import reward_function, RewardType

@reward_function(weight=0.5, reward_type=RewardType.IMMEDIATE)
async def creativity_reward(turns, context=None):
    """Reward creative responses"""
    creative_words = ["creative", "innovative", "unique", "original"]
    
    assistant_content = " ".join(
        turn.content for turn in turns if turn.role == "assistant"
    ).lower()
    
    creativity_score = sum(
        0.1 for word in creative_words if word in assistant_content
    )
    
    return min(creativity_score, 1.0)
```

## Training Configuration

### Training Profiles

```python
from grpo_agent_framework.training.train import train

# Conservative: Maximum stability
await train(agent, environment, profile="conservative")

# Balanced: Good stability and performance
await train(agent, environment, profile="balanced")

# Aggressive: Maximum performance
await train(agent, environment, profile="aggressive")
```

### Custom Configuration

```python
config_overrides = {
    "learning_rate": 1e-5,
    "num_generations": 16,
    "max_grad_norm": 0.5,
    "early_stopping": True,
    "patience": 30
}

await train(
    agent=agent,
    environment=environment,
    num_episodes=500,
    profile="balanced",
    config_overrides=config_overrides
)
```

### Automatic Training

```python
from grpo_agent_framework.training.train import AutoTrainer

auto_trainer = AutoTrainer(
    auto_adjust=True,      # Automatically adjust hyperparameters
    early_stopping=True,   # Stop if no improvement
    patience=50           # Steps to wait before stopping
)

trained_agent = await auto_trainer.train(
    agent=agent,
    environment=environment,
    num_episodes=1000
)
```

## Advanced Features

### Multi-GPU Training

```python
from grpo_agent_framework.training.distributed import DistributedTrainer

trainer = DistributedTrainer(
    num_gpus=4,
    strategy="ddp"  # or "deepspeed"
)

await trainer.train(agent, environment, config)
```

### Real-time Monitoring

```python
from grpo_agent_framework.training.diagnostics import DiagnosticsMonitor

monitor = DiagnosticsMonitor()

# Monitor training health
await train(
    agent=agent,
    environment=environment,
    callbacks=[monitor]
)

# Get health report
health = monitor.get_health_status()
print(f"Training health: {health.status}")
```

### Checkpointing and Loading

```python
# Save checkpoint during training
await train(
    agent=agent,
    environment=environment,
    save_path="./checkpoints/my_agent"
)

# Load trained agent
from grpo_agent_framework.core.agent import load_agent_from_checkpoint

loaded_agent = await load_agent_from_checkpoint(
    "./checkpoints/my_agent",
    agent_class=MultiTurnAgent
)
```

## Best Practices

### 1. Start Simple
- Begin with basic conversation scenarios
- Use built-in reward functions
- Start with conservative training profiles

### 2. Task Analysis
- Evaluate baseline performance before training
- Ensure appropriate task difficulty
- Check reward diversity within trajectory groups

### 3. Iterative Development
- Start with small episode counts for testing
- Gradually increase complexity
- Monitor training health continuously

### 4. Reward Design
- Use composite rewards for balanced behavior
- Weight safety and helpfulness highly
- Test custom rewards thoroughly

### 5. Environment Design
- Create realistic user simulation
- Include edge cases and challenging scenarios
- Balance conversation length and complexity

## Examples

### Complete Examples

1. **Quick Start**: `grpo_agent_framework/examples/quick_start.py`
   - Basic agent training
   - Simple reward functions
   - Minimal configuration

2. **Customer Service Agent**: `grpo_agent_framework/examples/customer_service_agent.py`
   - Specialized agent class
   - Custom environment
   - Domain-specific rewards
   - Professional conversation handling

3. **Tutoring Agent**: `grpo_agent_framework/examples/tutoring_agent.py`
   - Educational conversation scenarios
   - Knowledge assessment rewards
   - Student progress tracking

### Running Examples

```bash
# Quick start example
python -m grpo_agent_framework.examples.quick_start

# Customer service example
python -m grpo_agent_framework.examples.customer_service_agent

# Create your own based on the templates
```

### Example Output

```
GRPO Agent Framework - Training Complete
======================================
Episodes: 500
Final Reward: 0.847 ± 0.123
Success Rate: 89.2%
Average Episode Length: 8.3 turns

Training Health: HEALTHY
- Reward diversity: 0.156 (good)
- Gradient stability: 0.834 (stable)
- No major issues detected

Agent saved to: ./checkpoints/my_trained_agent
```

## Troubleshooting

### Common Issues

1. **Low Reward Diversity**
   - Increase generation temperature
   - Use more diverse scenarios
   - Check reward function design

2. **Training Instability**
   - Use conservative profile
   - Reduce learning rate
   - Increase gradient clipping

3. **Poor Convergence**
   - Verify task is learnable
   - Check baseline performance
   - Increase episode count

4. **Memory Issues**
   - Reduce batch sizes
   - Use gradient accumulation
   - Enable model checkpointing

### Getting Help

- Check the documentation: https://grpo-framework.readthedocs.io
- Review examples in the `examples/` directory
- File issues: https://github.com/yourusername/grpo-agent-framework/issues

## Next Steps

1. **Experiment** with the provided examples
2. **Create** your own custom agents and environments
3. **Explore** advanced features like tool use and multi-modal interactions
4. **Contribute** to the framework development

Happy training! 🚀